#include "kv_tree.hpp"
#include "llama_backend.hpp"
#include "llama_wrap.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace inference_engine;
using namespace inference_engine::llm;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void lifecycle(const Model& model) {
    // A live anchor keeps the common prefix available while the second slot is
    // repeatedly released, evicted, and reused with different suffix inputs.
    constexpr BackendPos prefix_size = 128;
    Context context(model, 768, 128, 3, true, 1, 1);
    LlamaBackend backend(context);
    KvRadixTree tree(&backend, 3);
    std::vector<TokenId> prefix;
    for (TokenId token = 1; token <= prefix_size; ++token) { prefix.push_back(token); }
    require(context.prefill(0, prefix, 0) == DecodeStatus::Ok, "prefix prefill");
    Node* parent = tree.append(tree.root(), prefix, SeqRange{0, 0, prefix_size});
    static_cast<void>(tree.fork(parent, 0));
    const std::vector<TokenId> anchor_suffix{400, 401, 402, 403, 404, 405, 406, 407};
    require(context.prefill(0, anchor_suffix, prefix_size) == DecodeStatus::Ok, "anchor prefill");
    Node* anchor = tree.commit(parent, anchor_suffix, SeqRange{0, 0, prefix_size + 8});

    for (int cycle = 0; cycle < 12; ++cycle) {
        require(tree.fork(parent, 1) == parent, "prefix fork");
        require(context.seq_position_max(1) == prefix_size - 1,
                "internal prefix fork copied descendant suffix");
        std::vector<TokenId> suffix;
        for (int index = 0; index < 16; ++index) {
            suffix.push_back(static_cast<TokenId>(500 + cycle * 20 + index));
        }
        require(context.prefill(1, suffix, prefix_size) == DecodeStatus::Ok, "suffix prefill");
        Context reference(model, 768, 128, 3, true, 1, 1);
        require(reference.prefill(0, prefix, 0) == DecodeStatus::Ok, "reference prefix");
        require(reference.prefill(0, suffix, prefix_size) == DecodeStatus::Ok, "reference suffix");
        const std::array<TokenId, 1> input{static_cast<TokenId>(900 + cycle)};
        const std::array<BackendPos, 1> position{prefix_size + 16};
        const std::array<BackendSeqId, 1> tested_seq{1}, reference_seq{0};
        const auto actual = context.decode_lockstep(tested_seq, input, position);
        const auto expected = reference.decode_lockstep(reference_seq, input, position);
        require(actual.status == DecodeStatus::Ok && expected.status == DecodeStatus::Ok,
                "lifecycle decode");
        require(actual.tokens == expected.tokens, "reused slot token mismatch");
        const auto a = context.logits_ith(0), b = reference.logits_ith(0);
        double maximum = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double difference = std::abs(static_cast<double>(a[i]) - b[i]);
            require(std::isfinite(difference), "nonfinite lifecycle logits");
            maximum = std::max(maximum, difference);
        }
        require(maximum <= 1e-3, "reused slot logits mismatch");
        suffix.push_back(input.front());
        Node* leaf = tree.commit(parent, suffix, SeqRange{1, 0, prefix_size + 17});
        auto full = prefix;
        full.insert(full.end(), suffix.begin(), suffix.end());
        require(tree.match_prefix(full).matched_tokens == full.size(), "committed path unreachable");
        require(tree.validate(), "tree invalid before release");
        tree.release(leaf);
        require(tree.evict(1) == suffix.size(), "idle suffix not evicted");
        require(context.seq_position_max(1) == prefix_size - 1, "evicted suffix remains in KV");
        require(context.seq_position_max(0) == prefix_size + 7, "eviction changed live anchor KV");
        require(tree.match_prefix(full).matched_tokens == prefix.size(), "evicted path still reachable");
        require(tree.validate(), "tree invalid after eviction");
        std::cout << "LIFECYCLE," << cycle << "," << maximum << ",PASSED\n";
    }
    tree.release(anchor);
    require(tree.evict(1) == anchor_suffix.size(), "anchor eviction");
    require(tree.validate(), "final tree invalid");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) { throw std::invalid_argument("usage: ie_lifecycle_tests <GGUF>"); }
        const Model model(argv[1]);
        lifecycle(model);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
