#include "kv_backend.hpp"
#include "kv_tree.hpp"
#include "llama_backend.hpp"
#include "llama_wrap.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

namespace {

using inference_engine::BackendPos;
using inference_engine::BackendSeqId;
using inference_engine::KvRadixTree;
using inference_engine::Node;
using inference_engine::SeqRange;
using inference_engine::TokenId;
using inference_engine::llm::Context;
using inference_engine::llm::DecodeStatus;
using inference_engine::llm::LlamaBackend;
using inference_engine::llm::Model;

struct Config {
    std::string model_path;
    std::string arm;
    std::size_t branches{8};
    std::size_t prefix{2048};
    std::size_t suffix{200};
};

struct MemoryInfo {
    std::uint64_t private_bytes{0};
    std::uint64_t peak_working_set_bytes{0};
};

struct RunResult {
    std::uint32_t n_ctx;
    double prefill_ms;
    std::vector<double> fork_us;
    std::vector<double> step_us;
    std::vector<std::vector<TokenId>> generated;
    std::vector<std::vector<float>> final_logits;
    MemoryInfo memory;
};

[[nodiscard]] std::uint32_t pad256(std::size_t value) {
    const std::size_t remainder = value % 256U;
    const std::size_t padded = remainder == 0 ? value : value + 256U - remainder;
    if (padded > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context size exceeds uint32_t");
    }
    return static_cast<std::uint32_t>(padded);
}

[[nodiscard]] std::vector<TokenId> make_tokens(std::size_t count,
                                                int32_t vocab_size,
                                                std::size_t offset) {
    if (vocab_size < 2) {
        throw std::runtime_error("model vocabulary is too small");
    }
    std::vector<TokenId> tokens;
    tokens.reserve(count);
    const auto usable_vocab = static_cast<std::size_t>(vocab_size - 1);
    for (std::size_t index = 0; index < count; ++index) {
        tokens.push_back(static_cast<TokenId>(1 + ((offset + index) % usable_vocab)));
    }
    return tokens;
}

[[nodiscard]] double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::floor(fraction * static_cast<double>(values.size() - 1)));
    return values[index];
}

[[nodiscard]] MemoryInfo process_memory() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
            sizeof(counters)) != 0) {
        return {static_cast<std::uint64_t>(counters.PrivateUsage),
                static_cast<std::uint64_t>(counters.PeakWorkingSetSize)};
    }
#endif
    return {};
}

[[nodiscard]] std::optional<RunResult> run_once(const Model& model,
                                                const Config& config,
                                                std::uint32_t n_ctx) {
    try {
        const bool tree_arm = config.arm == "tree";
        Context context(model, n_ctx, 128,
                        static_cast<std::uint32_t>(config.branches), tree_arm);
        const auto prompt = make_tokens(config.prefix, model.info().n_vocab, 0);
        const auto start_prefill = std::chrono::steady_clock::now();
        std::optional<LlamaBackend> llama_backend;
        std::optional<KvRadixTree> tree;
        Node* prefix_node = nullptr;
        if (tree_arm) {
            llama_backend.emplace(context);
            tree.emplace(&llama_backend.value(),
                         static_cast<std::uint32_t>(config.branches));
            const SeqRange prefix_holder{
                0, 0, static_cast<BackendPos>(config.prefix)};
            if (context.prefill(0, prompt, 0) != DecodeStatus::Ok) {
                return std::nullopt;
            }
            prefix_node = tree->append(tree->root(), prompt, prefix_holder);
        } else {
            for (std::size_t branch = 0; branch < config.branches; ++branch) {
                if (context.prefill(static_cast<BackendSeqId>(branch), prompt, 0) !=
                    DecodeStatus::Ok) {
                    return std::nullopt;
                }
            }
        }
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start_prefill)
                .count();

        std::vector<double> fork_us;
        if (tree_arm) {
            const auto fork_start = std::chrono::steady_clock::now();
            if (tree->fork(prefix_node, 0) == nullptr) {
                return std::nullopt;
            }
            fork_us.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - fork_start)
                                  .count());
            for (std::size_t branch = 1; branch < config.branches; ++branch) {
                const auto branch_start = std::chrono::steady_clock::now();
                if (tree->fork(prefix_node,
                               static_cast<BackendSeqId>(branch)) == nullptr) {
                    return std::nullopt;
                }
                fork_us.push_back(std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - branch_start)
                                      .count());
            }
        }

        std::vector<BackendSeqId> sequence_ids(config.branches);
        for (std::size_t branch = 0; branch < config.branches; ++branch) {
            sequence_ids[branch] = static_cast<BackendSeqId>(branch);
        }
        std::vector<TokenId> next_tokens =
            make_tokens(config.branches, model.info().n_vocab, config.prefix);
        std::vector<std::vector<TokenId>> generated(config.branches);
        for (auto& branch_tokens : generated) {
            branch_tokens.reserve(config.suffix);
        }
        std::vector<double> step_us;
        step_us.reserve(config.suffix);
        std::vector<BackendPos> positions(config.branches);
        for (std::size_t step = 0; step < config.suffix; ++step) {
            std::fill(positions.begin(), positions.end(),
                      static_cast<BackendPos>(config.prefix + step));
            const auto step_start = std::chrono::steady_clock::now();
            const auto result = context.decode_lockstep(
                sequence_ids, next_tokens, positions);
            step_us.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - step_start)
                                  .count());
            if (result.status == DecodeStatus::NoKvSlot) {
                return std::nullopt;
            }
            if (result.status != DecodeStatus::Ok ||
                result.tokens.size() != config.branches) {
                return std::nullopt;
            }
            next_tokens = result.tokens;
            for (std::size_t branch = 0; branch < config.branches; ++branch) {
                generated[branch].push_back(next_tokens[branch]);
            }
        }

        std::vector<std::vector<float>> final_logits(config.branches);
        for (std::size_t branch = 0; branch < config.branches; ++branch) {
            final_logits[branch] = context.logits_ith(
                static_cast<std::int32_t>(branch));
        }

        if (tree_arm) {
            for (std::size_t branch = 0; branch < config.branches; ++branch) {
                const SeqRange holder{
                    static_cast<BackendSeqId>(branch), 0,
                    static_cast<BackendPos>(config.prefix + config.suffix)};
                static_cast<void>(tree->commit(prefix_node, generated[branch], holder));
            }
            if (!tree->validate()) {
                return std::nullopt;
            }
            for (std::size_t branch = 0; branch < config.branches; ++branch) {
                const auto match = tree->match_prefix(
                    std::span<const TokenId>(generated[branch]));
                static_cast<void>(match);
            }
        }

        return RunResult{context.n_ctx(), prefill_ms, std::move(fork_us),
                         std::move(step_us), std::move(generated),
                         std::move(final_logits), process_memory()};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

[[nodiscard]] std::optional<RunResult> find_min_context(
    const Model& model, const Config& config, std::uint32_t predicted) {
    std::vector<std::uint32_t> candidates;
    if (predicted >= 256) {
        candidates.push_back(predicted - 256);
    }
    candidates.push_back(predicted);
    std::optional<RunResult> result;
    for (const std::uint32_t candidate : candidates) {
        auto attempt = run_once(model, config, candidate);
        if (attempt.has_value()) {
            result = std::move(attempt);
        }
    }
    if (result.has_value()) {
        return result;
    }
    for (std::uint32_t candidate = predicted + 256;
         candidate <= predicted + 4096; candidate += 256) {
        auto attempt = run_once(model, config, candidate);
        if (attempt.has_value()) {
            return attempt;
        }
    }
    return result;
}

struct CorrectnessResult {
    double max_abs_logit_diff;
    double token_match_rate;
};

[[nodiscard]] std::optional<CorrectnessResult> check_correctness(
    const Model& model, const Config& config,
    const RunResult& tree_result) {
    if (config.arm != "tree" || config.branches != 8 || config.prefix != 2048 ||
        config.suffix != 64) {
        return std::nullopt;
    }
    double max_diff = 0.0;
    std::size_t matches = 0;
    std::size_t total = 0;
    const auto prompt = make_tokens(config.prefix, model.info().n_vocab, 0);
    for (std::size_t branch = 0; branch < config.branches; ++branch) {
        Context context(model, pad256(config.prefix + config.suffix + 256),
                        128, 1, false);
        if (context.prefill(0, prompt, 0) != DecodeStatus::Ok) {
            return std::nullopt;
        }
        TokenId next_token =
            make_tokens(config.branches, model.info().n_vocab, config.prefix)[branch];
        std::vector<TokenId> independent;
        independent.reserve(config.suffix);
        for (std::size_t step = 0; step < config.suffix; ++step) {
            const std::array<BackendSeqId, 1> seqs{0};
            const std::array<TokenId, 1> tokens{next_token};
            const std::array<BackendPos, 1> positions{
                static_cast<BackendPos>(config.prefix + step)};
            const auto result = context.decode_lockstep(seqs, tokens, positions);
            if (result.status != DecodeStatus::Ok) {
                return std::nullopt;
            }
            next_token = result.tokens.front();
            independent.push_back(next_token);
        }
        const auto logits = context.logits_ith(0);
        const auto& tree_logits = tree_result.final_logits[branch];
        for (std::size_t index = 0; index < logits.size(); ++index) {
            max_diff = std::max(max_diff,
                                std::abs(static_cast<double>(logits[index]) -
                                         static_cast<double>(tree_logits[index])));
        }
        for (std::size_t index = 0; index < independent.size(); ++index) {
            ++total;
            if (independent[index] == tree_result.generated[branch][index]) {
                ++matches;
            }
        }
    }
    return CorrectnessResult{max_diff,
                             total == 0 ? 0.0
                                        : static_cast<double>(matches) /
                                              static_cast<double>(total)};
}

[[nodiscard]] Config parse_args(int argc, char** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto require_value = [&](const std::string& name) {
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing value for " + name);
            }
            return std::string(argv[++index]);
        };
        if (argument == "--model") {
            config.model_path = require_value(argument);
        } else if (argument == "--arm") {
            config.arm = require_value(argument);
        } else if (argument == "--b") {
            config.branches = std::stoull(require_value(argument));
        } else if (argument == "--prefix") {
            config.prefix = std::stoull(require_value(argument));
        } else if (argument == "--suffix") {
            config.suffix = std::stoull(require_value(argument));
        } else if (argument == "--single") {
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (config.model_path.empty() ||
        (config.arm != "naive" && config.arm != "tree") ||
        config.branches == 0 || config.branches > 256 || config.prefix == 0) {
        throw std::invalid_argument(
            "usage: ie_tree_bench --model <path> --arm <naive|tree> "
            "--b <branches> --prefix <tokens> --suffix <tokens> [--single]");
    }
    return config;
}

[[nodiscard]] std::string csv_value(double value) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6) << value;
    return output.str();
}

[[nodiscard]] std::string csv_value(std::uint64_t value) {
    return std::to_string(value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_args(argc, argv);
        const Model model(config.model_path);
        const std::size_t n_ctx_pred = config.arm == "naive"
                                           ? config.branches *
                                                 pad256(config.prefix + config.suffix)
                                           : pad256(config.prefix +
                                                    config.branches * config.suffix);
        const auto result = find_min_context(model, config,
                                             static_cast<std::uint32_t>(n_ctx_pred));
        if (!result.has_value()) {
            throw std::runtime_error("no successful n_ctx found");
        }
        const auto correctness = check_correctness(model, config, result.value());
        const double kv_bytes = static_cast<double>(result->n_ctx) *
                                static_cast<double>(model.info().kappa_bytes);
        const double rho_pred = static_cast<double>(config.branches) *
                                static_cast<double>(config.prefix + config.suffix) /
                                static_cast<double>(config.prefix +
                                                     config.branches * config.suffix);
        const double steps_p50 = percentile(result->step_us, 0.50);
        const double steps_p90 = percentile(result->step_us, 0.90);
        const double steps_p99 = percentile(result->step_us, 0.99);
        const double decode_us =
            std::accumulate(result->step_us.begin(), result->step_us.end(), 0.0);
        const double tok_per_s =
            decode_us == 0.0 ? 0.0
                             : static_cast<double>(config.branches * config.suffix) /
                                   (decode_us / 1.0e6);
        std::cout << "model: " << std::filesystem::path(config.model_path).filename().string()
                  << "\narm: " << config.arm << "\n"
                  << "b/P/S: " << config.branches << "/" << config.prefix << "/"
                  << config.suffix << "\n"
                  << "n_ctx predicted/measured: " << n_ctx_pred << "/"
                  << result->n_ctx << "\n"
                  << "kappa bytes: " << model.info().kappa_bytes << "\n"
                  << "KV bytes: " << static_cast<std::uint64_t>(kv_bytes) << "\n"
                  << "step us p50/p90/p99: " << steps_p50 << "/" << steps_p90
                  << "/" << steps_p99 << "\n"
                  << "tokens/s: " << tok_per_s << "\n";

        std::cout << "CSV," << std::filesystem::path(config.model_path).filename().string()
                  << "," << config.arm << "," << config.branches << ","
                  << config.prefix << "," << config.suffix << "," << result->n_ctx
                  << "," << n_ctx_pred << "," << model.info().kappa_bytes << ","
                  << static_cast<std::uint64_t>(kv_bytes) << ",NA," << csv_value(rho_pred)
                  << "," << csv_value(result->prefill_ms) << ","
                  << csv_value(percentile(result->fork_us, 0.50)) << ","
                  << csv_value(percentile(result->fork_us, 0.99)) << ","
                  << csv_value(steps_p50) << "," << csv_value(steps_p90) << ","
                  << csv_value(steps_p99) << "," << csv_value(steps_p99 - steps_p50)
                  << "," << csv_value(tok_per_s) << ","
                  << csv_value(result->memory.private_bytes) << ","
                  << csv_value(result->memory.peak_working_set_bytes) << ",";
        if (correctness.has_value()) {
            std::cout << csv_value(correctness->max_abs_logit_diff) << ","
                      << csv_value(correctness->token_match_rate);
        } else {
            std::cout << "NA,NA";
        }
        std::cout << "\n";
    } catch (const std::exception& error) {
        std::cerr << "ie_tree_bench: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
