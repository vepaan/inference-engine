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
    int32_t threads{0};
    int32_t flash_attn{-1};
    std::uint32_t n_ctx{0};
    std::size_t warmup{0};
    bool verify{false};
    bool verify_batched{false};
    bool kv_f32{false};
    bool phase_markers{false};
    bool trace_attention{false};
    bool discovery_only{false};
    bool single{false};
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
    std::vector<std::vector<std::vector<float>>> verification_logits;
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
    const double position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    return values[lower] + (position - static_cast<double>(lower)) *
                               (values[upper] - values[lower]);
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
        Context context(model, n_ctx, static_cast<std::uint32_t>(std::max<std::size_t>(128, config.branches)),
                        static_cast<std::uint32_t>(config.branches), tree_arm,
                        config.threads, config.flash_attn, config.kv_f32, config.trace_attention);
        const auto prompt = make_tokens(config.prefix, model.info().n_vocab, 0);
        if (config.phase_markers) { std::cout << "PROFILE_PHASE,prefill" << std::endl; }
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
                std::cerr << "run_once n_ctx=" << n_ctx
                          << ": tree prefill failed\n";
                return std::nullopt;
            }
            prefix_node = tree->append(tree->root(), prompt, prefix_holder);
        } else {
            for (std::size_t branch = 0; branch < config.branches; ++branch) {
                if (context.prefill(static_cast<BackendSeqId>(branch), prompt, 0) !=
                    DecodeStatus::Ok) {
                    std::cerr << "run_once n_ctx=" << n_ctx
                              << ": naive prefill failed\n";
                    return std::nullopt;
                }
            }
        }
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start_prefill)
                .count();

        if (config.phase_markers) { std::cout << "PROFILE_PHASE,fork" << std::endl; }
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
        const auto initial_tokens = next_tokens;
        std::vector<std::vector<std::vector<float>>> verification_logits(config.branches);
        if (config.phase_markers) { std::cout << "PROFILE_PHASE,decode" << std::endl; }
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
                std::cerr << "run_once n_ctx=" << n_ctx
                          << ": lockstep decode reported NoKvSlot\n";
                return std::nullopt;
            }
            if (result.status != DecodeStatus::Ok ||
                result.tokens.size() != config.branches) {
                std::cerr << "run_once n_ctx=" << n_ctx
                          << ": lockstep decode returned Error\n";
                return std::nullopt;
            }
            next_tokens = result.tokens;
            for (std::size_t branch = 0; branch < config.branches; ++branch) {
                generated[branch].push_back(next_tokens[branch]);
                if (config.verify) {
                    verification_logits[branch].push_back(context.logits_ith(
                        static_cast<std::int32_t>(branch)));
                }
            }
        }

        if (config.phase_markers) { std::cout << "PROFILE_PHASE,complete" << std::endl; }
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
                // KV holds the initial suffix input followed by all generated
                // tokens except the final prediction (which has not been fed).
                auto consumed = generated[branch];
                consumed.pop_back();
                consumed.insert(consumed.begin(), initial_tokens[branch]);
                static_cast<void>(tree->commit(prefix_node, consumed, holder));
            }
            if (!tree->validate()) {
                std::cerr << "run_once n_ctx=" << n_ctx
                          << ": tree validation failed after commit\n";
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
                         std::move(final_logits), process_memory(),
                         std::move(verification_logits)};
    } catch (const std::exception& error) {
        std::cerr << "run_once n_ctx=" << n_ctx << ": " << error.what()
                  << "\n";
        return std::nullopt;
    }
}

[[nodiscard]] std::optional<RunResult> find_min_context(
    const Model& model, const Config& config, std::uint32_t predicted) {
    // Search on the backend's 256-cell allocation grid. Exploratory timings
    // are discarded; report the smallest successful aligned capacity.
    std::optional<RunResult> result;
    for (std::uint32_t candidate = predicted;
         candidate <= predicted + 4096; candidate += 256) {
        result = run_once(model, config, candidate);
        if (result.has_value()) {
            break;
        }
    }
    if (!result.has_value()) {
        return std::nullopt;
    }
    for (std::uint32_t candidate = result->n_ctx - 256;
         candidate >= 256; candidate -= 256) {
        auto smaller = run_once(model, config, candidate);
        if (!smaller.has_value()) {
            break;
        }
        // Non-unified contexts can round several requests to the same size.
        if (smaller->n_ctx < result->n_ctx) {
            result = std::move(smaller);
        }
    }
    return result;
}

// Numerical tolerance for correctness verification:
// - Deterministic greedy sampling: token match rate must be 1.0 (100% equivalent tokens).
// - Max absolute logit difference tolerance: 1e-3.
constexpr double kMaxLogitDiffTolerance = 1e-3;
constexpr double kMinTokenMatchRate = 1.0;

struct CorrectnessResult {
    double max_abs_logit_diff{0.0};
    double mean_abs_logit_diff{0.0};
    double token_match_rate{0.0};
    bool passed{false};
};

[[nodiscard]] std::optional<CorrectnessResult> check_correctness(
    const Model& model, const Config& config,
    const RunResult& test_result) {
    double max_diff = 0.0;
    double sum_diff = 0.0;
    std::size_t logit_count = 0;
    std::size_t matches = 0;
    std::size_t total = 0;
    const auto prompt = make_tokens(config.prefix, model.info().n_vocab, 0);
    const auto initial_tokens =
        make_tokens(config.branches, model.info().n_vocab, config.prefix);

    for (std::size_t branch = 0; branch < config.branches; ++branch) {
        // A separate context owns each reference branch, with the same context
        // dimensions and KV layout as the tested arm. No seq_cp is used here.
        Context context(model, test_result.n_ctx,
                        static_cast<std::uint32_t>(std::max<std::size_t>(128, config.branches)),
                        static_cast<std::uint32_t>(config.branches),
                        config.arm == "tree", config.threads, config.flash_attn, config.kv_f32, config.trace_attention);
        if (context.prefill(0, prompt, 0) != DecodeStatus::Ok) {
            std::cerr << "Verification error: independent prefill failed on branch "
                      << branch << "\n";
            return std::nullopt;
        }
        TokenId next_token = initial_tokens[branch];
        std::vector<TokenId> independent;
        independent.reserve(config.suffix);
        bool same_history = true;
        for (std::size_t step = 0; step < config.suffix; ++step) {
            const std::array<BackendSeqId, 1> seqs{0};
            const std::array<TokenId, 1> tokens{next_token};
            const std::array<BackendPos, 1> positions{
                static_cast<BackendPos>(config.prefix + step)};
            const auto result = context.decode_lockstep(seqs, tokens, positions);
            if (result.status != DecodeStatus::Ok) {
                std::cerr << "Verification error: independent decode failed on branch "
                          << branch << ", step " << step << "\n";
                return std::nullopt;
            }
            next_token = result.tokens.front();
            independent.push_back(next_token);
            // Logits are comparable only while input histories are identical.
            // Include the first different prediction, then stop logit comparisons
            // for this branch; still compare the complete free-running token streams.
            if (same_history) {
                const auto logits = context.logits_ith(0);
                const auto& test_logits = test_result.verification_logits[branch][step];
                double step_max = 0.0;
                for (std::size_t index = 0; index < logits.size(); ++index) {
                    const double diff = std::abs(static_cast<double>(logits[index]) -
                                                 static_cast<double>(test_logits[index]));
                    if (!std::isfinite(diff)) {
                        throw std::runtime_error("verification encountered non-finite logits");
                    }
                    step_max = std::max(step_max, diff);
                    max_diff = std::max(max_diff, diff);
                    sum_diff += diff;
                    ++logit_count;
                }
                if (next_token != test_result.generated[branch][step]) {
                    std::cerr << "First token divergence: branch=" << branch
                              << " step=" << step << " independent=" << next_token
                              << " shared=" << test_result.generated[branch][step]
                              << " comparable_logit_max_diff=" << step_max << "\n";
                    same_history = false;
                }
            }
        }
        for (std::size_t index = 0; index < independent.size(); ++index) {
            ++total;
            if (independent[index] == test_result.generated[branch][index]) {
                ++matches;
            }
        }
    }
    const double mean_diff =
        logit_count == 0 ? 0.0 : sum_diff / static_cast<double>(logit_count);
    const double match_rate =
        total == 0 ? 0.0 : static_cast<double>(matches) / static_cast<double>(total);
    const bool passed = (match_rate >= kMinTokenMatchRate) && (max_diff <= kMaxLogitDiffTolerance);
    return CorrectnessResult{max_diff, mean_diff, match_rate, passed};
}

[[nodiscard]] CorrectnessResult compare_batched_runs(
    const RunResult& tested, const RunResult& reference) {
    std::size_t matches = 0;
    std::size_t total = 0;
    std::size_t count = 0;
    double maximum = 0.0;
    double sum = 0.0;
    for (std::size_t branch = 0; branch < tested.generated.size(); ++branch) {
        bool same_history = true;
        for (std::size_t step = 0; step < tested.generated[branch].size(); ++step) {
            if (same_history) {
                const auto& a = tested.verification_logits[branch][step];
                const auto& b = reference.verification_logits[branch][step];
                for (std::size_t i = 0; i < a.size(); ++i) {
                    const double diff = std::abs(static_cast<double>(a[i]) - b[i]);
                    if (!std::isfinite(diff)) {
                        throw std::runtime_error("non-finite batched reference logits");
                    }
                    maximum = std::max(maximum, diff);
                    sum += diff;
                    ++count;
                }
            }
            const bool equal = tested.generated[branch][step] == reference.generated[branch][step];
            matches += equal ? 1U : 0U;
            ++total;
            same_history = same_history && equal;
        }
    }
    const double rate = static_cast<double>(matches) / static_cast<double>(total);
    return {maximum, sum / static_cast<double>(count), rate,
            rate == 1.0 && maximum <= kMaxLogitDiffTolerance};
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
        } else if (argument == "--kv-type") {
            const std::string value = require_value(argument);
            if (value != "f16" && value != "f32") {
                throw std::invalid_argument("kv-type must be f16 or f32");
            }
            config.kv_f32 = value == "f32";
        } else if (argument == "--attention-shapes") {
            config.trace_attention = true;
        } else if (argument == "--phase-markers") {
            config.phase_markers = true;
        } else if (argument == "--threads") {
            config.threads = std::stoi(require_value(argument));
        } else if (argument == "--flash-attn") {
            const std::string val = require_value(argument);
            if (val == "on" || val == "1" || val == "true") {
                config.flash_attn = 1;
            } else if (val == "off" || val == "0" || val == "false") {
                config.flash_attn = 0;
            } else if (val == "auto") {
                config.flash_attn = -1;
            } else {
                throw std::invalid_argument("flash-attn must be auto, on, or off");
            }
        } else if (argument == "--n-ctx") {
            config.n_ctx = static_cast<std::uint32_t>(std::stoul(require_value(argument)));
        } else if (argument == "--warmup") {
            config.warmup = std::stoull(require_value(argument));
        } else if (argument == "--verify-batched") {
            config.verify_batched = true;
        } else if (argument == "--verify") {
            config.verify = true;
        } else if (argument == "--discovery-only") {
            config.discovery_only = true;
        } else if (argument == "--single") {
            config.single = true;
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (config.model_path.empty() ||
        (config.arm != "naive" && config.arm != "tree") ||
        config.branches == 0 || config.branches > 256 || config.prefix == 0 ||
        config.suffix == 0 || config.threads < 0 ||
        config.prefix > 1000000 || config.suffix > 1000000) {
        throw std::invalid_argument(
            "usage: ie_tree_bench --model <path> --arm <naive|tree> "
            "--b <branches> --prefix <tokens> --suffix <tokens> [--threads <N>] "
            "[--kv-type <f16|f32>] [--flash-attn <auto|on|off>] [--n-ctx <N>] [--warmup <N>] [--verify] [--single]");
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
        const std::size_t kappa_bytes = model.info().kappa_bytes * (config.kv_f32 ? 2U : 1U);
        const std::size_t n_ctx_pred = config.arm == "naive"
                                           ? config.branches *
                                                 pad256(config.prefix + config.suffix)
                                           : pad256(config.prefix +
                                                    config.branches * config.suffix);

        std::uint32_t n_ctx_to_use = config.n_ctx;
        if (n_ctx_to_use == 0) {
            const auto min_attempt = find_min_context(
                model, config, static_cast<std::uint32_t>(n_ctx_pred));
            if (!min_attempt.has_value()) {
                throw std::runtime_error("no successful n_ctx found during discovery");
            }
            n_ctx_to_use = min_attempt->n_ctx;
        }

        if (config.discovery_only) {
            std::cout << "MIN_N_CTX," << n_ctx_to_use << "\n";
            return 0;
        }

        // Warm-up passes (discard timings to avoid contaminating benchmark measurements)
        for (std::size_t w = 0; w < config.warmup; ++w) {
            const auto warmup_result = run_once(model, config, n_ctx_to_use);
            if (!warmup_result.has_value()) {
                throw std::runtime_error("warmup trial failed");
            }
        }

        // Fresh measured trial
        Config measured_config = config;
        measured_config.verify = false;
        const auto result = run_once(model, measured_config, n_ctx_to_use);
        if (!result.has_value()) {
            throw std::runtime_error("measured benchmark trial failed");
        }

        // Correctness verification (only when requested, to avoid contaminating latency measurements)
        std::optional<CorrectnessResult> correctness;
        if (config.verify || config.verify_batched) {
            Config verification_config = config;
            verification_config.verify = true;
            const auto verification_run = run_once(model, verification_config, n_ctx_to_use);
            if (!verification_run.has_value()) {
                throw std::runtime_error("untimed verification workload failed");
            }
            Config paired_config = verification_config;
            paired_config.arm = config.arm == "tree" ? "naive" : "tree";
            const auto paired_capacity = paired_config.arm == "naive"
                ? config.branches * pad256(config.prefix + config.suffix)
                : static_cast<std::size_t>(pad256(config.prefix + config.branches * config.suffix));
            const auto paired_run = run_once(model, paired_config,
                static_cast<std::uint32_t>(paired_capacity));
            if (!paired_run.has_value()) {
                throw std::runtime_error("untimed batched reference workload failed");
            }
            const auto batched = compare_batched_runs(verification_run.value(), paired_run.value());
            std::cout << "CSV_BATCH_VERIFY,"
                      << std::filesystem::path(config.model_path).filename().string()
                      << "," << config.branches << "," << config.prefix << "," << config.suffix
                      << "," << config.threads << "," << config.flash_attn
                      << "," << csv_value(batched.token_match_rate)
                      << "," << csv_value(batched.max_abs_logit_diff)
                      << "," << csv_value(batched.mean_abs_logit_diff)
                      << "," << (batched.passed ? "PASSED" : "FAILED")
                      << "," << (config.kv_f32 ? "f32" : "f16") << "\n";
            if (config.verify) {
            correctness = check_correctness(model, config, verification_run.value());
            if (!correctness.has_value()) {
                std::cerr << "Verification failed to complete\n";
                return 2;
            }
            correctness->passed = correctness->passed && batched.passed;
            std::cout << "\n=== Correctness Verification ===\n"
                      << "Token match rate: " << std::fixed << std::setprecision(4)
                      << (correctness->token_match_rate * 100.0) << "%\n"
                      << "Max absolute logit diff: " << std::setprecision(6)
                      << correctness->max_abs_logit_diff << "\n"
                      << "Mean absolute logit diff: "
                      << correctness->mean_abs_logit_diff << "\n"
                      << "Status: " << (correctness->passed ? "PASSED" : "FAILED") << "\n\n";

            std::cout << "CSV_VERIFY,"
                      << std::filesystem::path(config.model_path).filename().string()
                      << "," << config.arm << "," << config.branches << ","
                      << config.prefix << "," << config.suffix << ","
                      << config.threads << "," << config.flash_attn << ","
                      << csv_value(correctness->token_match_rate) << ","
                      << csv_value(correctness->max_abs_logit_diff) << ","
                      << csv_value(correctness->mean_abs_logit_diff) << ","
                      << (correctness->passed ? "PASSED" : "FAILED")
                      << "," << (config.kv_f32 ? "f32" : "f16") << "\n";

            if (!correctness->passed) {
                std::cerr << "Verification failure: outputs do not match within tolerance (match_rate >= 1.0, max_diff <= 1e-3)\n";
                return 2;
            }
            } else if (!batched.passed) {
                std::cerr << "Batched naive/tree parity failed\n";
                return 2;
            }
        }

        const double kv_bytes = static_cast<double>(result->n_ctx) *
                                static_cast<double>(kappa_bytes);
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
                  << "threads: " << config.threads << "\n"
                  << "kv_type: " << (config.kv_f32 ? "f32" : "f16") << "\n"
                  << "flash_attn: " << config.flash_attn << "\n"
                  << "n_ctx predicted/measured: " << n_ctx_pred << "/"
                  << result->n_ctx << "\n"
                  << "kappa bytes: " << kappa_bytes << "\n"
                  << "KV bytes: " << static_cast<std::uint64_t>(kv_bytes) << "\n"
                  << "step us p50/p90/p99: " << steps_p50 << "/" << steps_p90
                  << "/" << steps_p99 << "\n"
                  << "tokens/s: " << tok_per_s << "\n";

        // Standard CSV output (compatible with existing 23-column format)
        std::cout << "CSV," << std::filesystem::path(config.model_path).filename().string()
                  << "," << config.arm << "," << config.branches << ","
                  << config.prefix << "," << config.suffix << "," << result->n_ctx
                  << "," << n_ctx_pred << "," << kappa_bytes << ","
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

        // Extended CSV line for Milestone B & C rich trial logging
        std::cout << "CSV_EXTENDED,"
                  << std::filesystem::path(config.model_path).filename().string()
                  << "," << config.arm << "," << config.branches << ","
                  << config.prefix << "," << config.suffix << ","
                  << config.threads << "," << config.flash_attn << ","
                  << result->n_ctx << "," << n_ctx_pred << ","
                  << kappa_bytes << ","
                  << static_cast<std::uint64_t>(kv_bytes) << ","
                  << csv_value(rho_pred) << ","
                  << csv_value(result->prefill_ms) << ","
                  << csv_value(percentile(result->fork_us, 0.50)) << ","
                  << csv_value(percentile(result->fork_us, 0.99)) << ","
                  << csv_value(steps_p50) << "," << csv_value(steps_p90) << ","
                  << csv_value(steps_p99) << ","
                  << csv_value(steps_p99 - steps_p50) << ","
                  << csv_value(tok_per_s) << ","
                  << csv_value(result->memory.private_bytes) << ","
                  << csv_value(result->memory.peak_working_set_bytes) << ",";
        if (correctness.has_value()) {
            std::cout << csv_value(correctness->max_abs_logit_diff) << ","
                      << csv_value(correctness->mean_abs_logit_diff) << ","
                      << csv_value(correctness->token_match_rate) << ","
                      << (correctness->passed ? "PASSED" : "FAILED");
        } else {
            std::cout << "NA,NA,NA,NA";
        }
        std::cout << "," << (config.kv_f32 ? "f32" : "f16") << "\n";
    } catch (const std::exception& error) {
        std::cerr << "ie_tree_bench: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
