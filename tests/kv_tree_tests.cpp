#include "kv_backend.hpp"
#include "kv_tree.hpp"
#include "minitest.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

using inference_engine::BackendPos;
using inference_engine::BackendSeqId;
using inference_engine::KvRadixTree;
using inference_engine::Node;
using inference_engine::SeqRange;
using inference_engine::TokenId;

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    [[nodiscard]] std::uint64_t next() {
        state_ += 0x9e3779b97f4a7c15ULL;
        std::uint64_t value = state_;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

private:
    std::uint64_t state_;
};

[[nodiscard]] std::size_t longest_common_prefix(
    std::span<const TokenId> query,
    const std::vector<std::vector<TokenId>>& sequences) {
    std::size_t result = 0;
    for (const auto& sequence : sequences) {
        const std::size_t count = std::min(query.size(), sequence.size());
        std::size_t common = 0;
        while (common < count && query[common] == sequence[common]) {
            ++common;
        }
        result = std::max(result, common);
    }
    return result;
}

IE_TEST_CASE(property_random_operations) {
    KvRadixTree tree;
    SplitMix64 random(0x123456789abcdef0ULL);
    std::vector<Node*> handles;
    std::size_t append_count = 0;

    for (std::size_t operation = 0; operation < 1000; ++operation) {
        const std::uint64_t choice = random.next() % 4U;
        if (choice == 0 || handles.empty()) {
            const TokenId token = static_cast<TokenId>(100000 + append_count++);
            const std::vector<TokenId> edge{token};
            const SeqRange holder{
                static_cast<BackendSeqId>(append_count), 0,
                static_cast<BackendPos>(edge.size())};
            Node* leaf = tree.append(tree.root(), edge, holder);
            static_cast<void>(tree.fork(leaf));
            handles.push_back(leaf);
        } else if (choice == 1) {
            const std::size_t index =
                static_cast<std::size_t>(random.next() % handles.size());
            Node* branch = handles[index];
            static_cast<void>(tree.fork(branch));
            handles.push_back(branch);
        } else if (choice == 2) {
            const std::size_t index =
                static_cast<std::size_t>(random.next() % handles.size());
            tree.release(handles[index]);
            handles.erase(handles.begin() + static_cast<std::ptrdiff_t>(index));
        } else {
            static_cast<void>(tree.evict(1));
        }
        IE_CHECK(tree.validate());
    }
}

IE_TEST_CASE(differential_prefix_matching) {
    KvRadixTree tree;
    const std::vector<std::vector<TokenId>> sequences{
        {1, 2, 3, 4}, {1, 2, 8}, {1, 9}, {20, 21, 22}};
    for (const auto& sequence : sequences) {
        static_cast<void>(tree.insert_sequence(sequence));
        IE_CHECK(tree.validate());
    }

    const std::vector<std::vector<TokenId>> queries{
        {1, 2, 3, 4}, {1, 2, 3}, {1, 2, 7}, {1, 9, 4},
        {20, 21},       {20, 25},  {99},       {1, 2, 8, 5},
    };
    for (const auto& query : queries) {
        const auto match = tree.match_prefix(query);
        IE_CHECK(match.matched_tokens ==
                  longest_common_prefix(query, sequences));
    }
}

IE_TEST_CASE(split_edge_cases) {
    KvRadixTree tree;
    static_cast<void>(tree.insert_sequence(std::vector<TokenId>{10, 11, 12, 13}));
    static_cast<void>(tree.insert_sequence(std::vector<TokenId>{10}));
    static_cast<void>(tree.insert_sequence(std::vector<TokenId>{10, 11, 20}));
    static_cast<void>(tree.insert_sequence(std::vector<TokenId>{10, 11, 21}));
    static_cast<void>(tree.insert_sequence(std::vector<TokenId>{10, 11, 21, 22}));
    IE_CHECK(tree.validate());

    const auto first = tree.match_prefix(std::vector<TokenId>{10});
    const auto second = tree.match_prefix(std::vector<TokenId>{10, 11});
    const auto repeated =
        tree.match_prefix(std::vector<TokenId>{10, 11, 21, 22});
    IE_CHECK(first.matched_tokens == 1);
    IE_CHECK(second.matched_tokens == 2);
    IE_CHECK(repeated.matched_tokens == 4);
}

IE_TEST_CASE(slot_exhaustion_is_reported) {
    inference_engine::MockBackend backend(1);
    KvRadixTree tree(&backend, 2);
    const std::vector<TokenId> prefix{7};
    const SeqRange source{0, 0, 1};
    Node* leaf = tree.commit(tree.root(), prefix, source);
    IE_CHECK(tree.fork(leaf, 1) == leaf);
    bool exhausted = false;
    try {
        static_cast<void>(tree.fork(leaf));
    } catch (const std::runtime_error&) {
        exhausted = true;
    }
    IE_CHECK(exhausted);
}

IE_TEST_CASE(commit_transfers_prefix_handle_to_leaf) {
    inference_engine::MockBackend backend(1);
    KvRadixTree tree(&backend, 2);
    const std::vector<TokenId> prefix{30, 31};
    const SeqRange source{0, 0, 2};
    Node* prefix_node = tree.append(tree.root(), prefix, source);
    static_cast<void>(tree.fork(prefix_node, 0));
    static_cast<void>(tree.fork(prefix_node, 1));

    const std::vector<TokenId> first_suffix{40};
    const std::vector<TokenId> second_suffix{41};
    static_cast<void>(tree.commit(prefix_node, first_suffix, {0, 0, 3}));
    static_cast<void>(tree.commit(prefix_node, second_suffix, {1, 0, 3}));
    IE_CHECK(tree.validate());
}

IE_TEST_CASE(commit_splits_shared_generated_prefix) {
    inference_engine::MockBackend backend(1);
    KvRadixTree tree(&backend, 2);
    const std::vector<TokenId> prefix{30, 31};
    Node* prefix_node = tree.append(tree.root(), prefix, SeqRange{0, 0, 2});
    static_cast<void>(tree.fork(prefix_node, 0));
    static_cast<void>(tree.fork(prefix_node, 1));

    static_cast<void>(tree.commit(prefix_node, std::vector<TokenId>{40, 41},
                                  SeqRange{0, 0, 4}));
    static_cast<void>(tree.commit(prefix_node, std::vector<TokenId>{40, 42},
                                  SeqRange{1, 0, 4}));
    IE_CHECK(tree.validate());
}

IE_TEST_CASE(commit_merges_identical_generated_edge) {
    inference_engine::MockBackend backend(1);
    KvRadixTree tree(&backend, 2);
    const std::vector<TokenId> prefix{30, 31};
    Node* prefix_node = tree.append(tree.root(), prefix, SeqRange{0, 0, 2});
    static_cast<void>(tree.fork(prefix_node, 0));
    static_cast<void>(tree.fork(prefix_node, 1));

    static_cast<void>(tree.commit(prefix_node, std::vector<TokenId>{40},
                                  SeqRange{0, 0, 3}));
    static_cast<void>(tree.commit(prefix_node, std::vector<TokenId>{40},
                                  SeqRange{1, 0, 3}));
    IE_CHECK(tree.validate());
}

}  // namespace

int main() { return minitest::run_all(); }
