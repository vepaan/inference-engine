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

IE_TEST_CASE(identical_prefix_insertion) {
    KvRadixTree tree;
    const std::vector<TokenId> prefix{1, 2, 3};
    Node* n1 = tree.insert_sequence(prefix);
    Node* n2 = tree.insert_sequence(prefix);
    IE_CHECK(n1 == n2);
    IE_CHECK(tree.validate());
    IE_CHECK(tree.node_count() == 2);
    const auto match = tree.match_prefix(prefix);
    IE_CHECK(match.matched_tokens == 3);
    IE_CHECK(match.node == n1);
}

IE_TEST_CASE(partial_prefix_overlap) {
    KvRadixTree tree;
    static_cast<void>(tree.insert_sequence(std::vector<TokenId>{10, 20, 30}));
    static_cast<void>(tree.insert_sequence(std::vector<TokenId>{10, 20, 40}));
    static_cast<void>(tree.insert_sequence(std::vector<TokenId>{10, 50}));
    static_cast<void>(tree.insert_sequence(std::vector<TokenId>{60}));
    IE_CHECK(tree.validate());

    const auto m1 = tree.match_prefix(std::vector<TokenId>{10, 20, 30, 99});
    IE_CHECK(m1.matched_tokens == 3);

    const auto m2 = tree.match_prefix(std::vector<TokenId>{10, 20, 40});
    IE_CHECK(m2.matched_tokens == 3);

    const auto m3 = tree.match_prefix(std::vector<TokenId>{10, 20});
    IE_CHECK(m3.matched_tokens == 2);

    const auto m4 = tree.match_prefix(std::vector<TokenId>{10, 99});
    IE_CHECK(m4.matched_tokens == 1);

    const auto m5 = tree.match_prefix(std::vector<TokenId>{60, 70});
    IE_CHECK(m5.matched_tokens == 1);

    const auto m6 = tree.match_prefix(std::vector<TokenId>{999});
    IE_CHECK(m6.matched_tokens == 0);
    IE_CHECK(m6.node == tree.root());
}

IE_TEST_CASE(split_existing_radix_edges) {
    KvRadixTree tree;
    Node* n1 = tree.append(tree.root(), std::vector<TokenId>{100, 101, 102, 103, 104});
    IE_CHECK(tree.validate());

    Node* mid1 = tree.split(n1, 2);
    IE_CHECK(tree.validate());
    IE_CHECK(mid1->tokens.size() == 2);
    IE_CHECK(mid1->tokens[0] == 100 && mid1->tokens[1] == 101);
    IE_CHECK(n1->tokens.size() == 3);
    IE_CHECK(n1->tokens[0] == 102 && n1->tokens[1] == 103 && n1->tokens[2] == 104);

    Node* mid2 = tree.split(mid1, 1);
    IE_CHECK(tree.validate());
    IE_CHECK(mid2->tokens.size() == 1);
    IE_CHECK(mid2->tokens[0] == 100);
    IE_CHECK(mid1->tokens.size() == 1);
    IE_CHECK(mid1->tokens[0] == 101);
}

IE_TEST_CASE(repeated_forks) {
    inference_engine::MockBackend backend(1);
    KvRadixTree tree(&backend, 8);
    const std::vector<TokenId> prefix{5, 6, 7};
    Node* prefix_node = tree.append(tree.root(), prefix, SeqRange{0, 0, 3});

    for (int i = 0; i < 5; ++i) {
        Node* forked = tree.fork(prefix_node, static_cast<BackendSeqId>(i + 1));
        IE_CHECK(forked == prefix_node);
        IE_CHECK(prefix_node->ref_count.load() == static_cast<std::uint32_t>(i + 1));
        IE_CHECK(tree.validate());
    }
}

IE_TEST_CASE(branch_release_order_lifo_fifo_interleaved) {
    inference_engine::MockBackend backend(1);
    // Test FIFO
    {
        KvRadixTree tree(&backend, 8);
        Node* p = tree.append(tree.root(), std::vector<TokenId>{1, 2}, SeqRange{0, 0, 2});
        static_cast<void>(tree.fork(p, 0));
        static_cast<void>(tree.fork(p, 1));
        static_cast<void>(tree.fork(p, 2));

        Node* c0 = tree.commit(p, std::vector<TokenId>{10}, SeqRange{0, 0, 3});
        Node* c1 = tree.commit(p, std::vector<TokenId>{11}, SeqRange{1, 0, 3});
        Node* c2 = tree.commit(p, std::vector<TokenId>{12}, SeqRange{2, 0, 3});
        IE_CHECK(tree.validate());

        tree.release(c0);
        IE_CHECK(tree.validate());
        tree.release(c1);
        IE_CHECK(tree.validate());
        tree.release(c2);
        IE_CHECK(tree.validate());
    }

    // Test LIFO
    {
        KvRadixTree tree(&backend, 8);
        Node* p = tree.append(tree.root(), std::vector<TokenId>{1, 2}, SeqRange{0, 0, 2});
        static_cast<void>(tree.fork(p, 0));
        static_cast<void>(tree.fork(p, 1));
        static_cast<void>(tree.fork(p, 2));

        Node* c0 = tree.commit(p, std::vector<TokenId>{10}, SeqRange{0, 0, 3});
        Node* c1 = tree.commit(p, std::vector<TokenId>{11}, SeqRange{1, 0, 3});
        Node* c2 = tree.commit(p, std::vector<TokenId>{12}, SeqRange{2, 0, 3});
        IE_CHECK(tree.validate());

        tree.release(c2);
        IE_CHECK(tree.validate());
        tree.release(c1);
        IE_CHECK(tree.validate());
        tree.release(c0);
        IE_CHECK(tree.validate());
    }

    // Test Interleaved
    {
        KvRadixTree tree(&backend, 8);
        Node* p = tree.append(tree.root(), std::vector<TokenId>{1, 2}, SeqRange{0, 0, 2});
        static_cast<void>(tree.fork(p, 0));
        static_cast<void>(tree.fork(p, 1));
        static_cast<void>(tree.fork(p, 2));
        static_cast<void>(tree.fork(p, 3));

        Node* c0 = tree.commit(p, std::vector<TokenId>{10}, SeqRange{0, 0, 3});
        Node* c1 = tree.commit(p, std::vector<TokenId>{11}, SeqRange{1, 0, 3});
        Node* c2 = tree.commit(p, std::vector<TokenId>{12}, SeqRange{2, 0, 3});
        Node* c3 = tree.commit(p, std::vector<TokenId>{13}, SeqRange{3, 0, 3});
        IE_CHECK(tree.validate());

        tree.release(c1);
        IE_CHECK(tree.validate());
        tree.release(c3);
        IE_CHECK(tree.validate());
        tree.release(c0);
        IE_CHECK(tree.validate());
        tree.release(c2);
        IE_CHECK(tree.validate());
    }
}

IE_TEST_CASE(reference_count_ownership_invariants) {
    inference_engine::MockBackend backend(1);
    KvRadixTree tree(&backend, 8);
    Node* p = tree.append(tree.root(), std::vector<TokenId>{1, 2}, SeqRange{0, 0, 2});
    static_cast<void>(tree.fork(p, 0));
    static_cast<void>(tree.fork(p, 1));

    Node* branch_a = tree.commit(p, std::vector<TokenId>{10}, SeqRange{0, 0, 3});
    Node* branch_b = tree.commit(p, std::vector<TokenId>{20}, SeqRange{1, 0, 3});

    IE_CHECK(p->ref_count.load() == 2);
    IE_CHECK(branch_a->ref_count.load() == 1);
    IE_CHECK(branch_b->ref_count.load() == 1);

    static_cast<void>(tree.fork(branch_a, 2));
    IE_CHECK(branch_a->ref_count.load() == 2);
    IE_CHECK(p->ref_count.load() == 3);

    tree.release(branch_a);
    IE_CHECK(branch_a->ref_count.load() == 1);
    IE_CHECK(p->ref_count.load() == 2);

    IE_CHECK(tree.validate());
}

IE_TEST_CASE(sequence_slot_exhaustion_and_reuse) {
    inference_engine::MockBackend backend(1);
    KvRadixTree tree(&backend, 3);
    Node* prefix = tree.append(tree.root(), std::vector<TokenId>{1, 2}, SeqRange{0, 0, 2});
    static_cast<void>(tree.fork(prefix, 1));
    static_cast<void>(tree.fork(prefix, 2));

    bool exhausted = false;
    try {
        static_cast<void>(tree.fork(prefix));
    } catch (const std::runtime_error&) {
        exhausted = true;
    }
    IE_CHECK(exhausted);

    Node* leaf1 = tree.commit(prefix, std::vector<TokenId>{10}, SeqRange{1, 0, 3});
    Node* leaf2 = tree.commit(prefix, std::vector<TokenId>{20}, SeqRange{2, 0, 3});
    IE_CHECK(leaf2 != nullptr);
    IE_CHECK(tree.validate());

    tree.release(leaf1);
    IE_CHECK(tree.validate());

    const std::size_t freed = tree.evict(1);
    IE_CHECK(freed > 0);

    bool acquired_after_evict = false;
    try {
        Node* forked = tree.fork(prefix, 1);
        IE_CHECK(forked != nullptr);
        Node* leaf3 = tree.commit(prefix, std::vector<TokenId>{30}, SeqRange{1, 0, 3});
        IE_CHECK(leaf3 != nullptr);
        IE_CHECK(tree.validate());
        acquired_after_evict = true;
    } catch (const std::exception&) {
        acquired_after_evict = false;
    }
    IE_CHECK(acquired_after_evict);
}

IE_TEST_CASE(eviction_of_idle_branches_lru) {
    inference_engine::MockBackend backend(10);
    KvRadixTree tree(&backend, 8);
    Node* p = tree.append(tree.root(), std::vector<TokenId>{1, 2}, SeqRange{0, 0, 2});
    static_cast<void>(tree.fork(p, 0));
    static_cast<void>(tree.fork(p, 1));
    static_cast<void>(tree.fork(p, 2));

    Node* leaf0 = tree.commit(p, std::vector<TokenId>{10}, SeqRange{0, 0, 3});
    Node* leaf1 = tree.commit(p, std::vector<TokenId>{20}, SeqRange{1, 0, 3});
    Node* leaf2 = tree.commit(p, std::vector<TokenId>{30}, SeqRange{2, 0, 3});
    IE_CHECK(leaf2 != nullptr);

    tree.release(leaf1);
    tree.release(leaf0);

    IE_CHECK(tree.validate());

    const std::size_t freed = tree.evict(1);
    IE_CHECK(freed == 1);
    IE_CHECK(tree.validate());

    const auto m1 = tree.match_prefix(std::vector<TokenId>{1, 2, 20});
    IE_CHECK(m1.matched_tokens == 2);

    const auto m0 = tree.match_prefix(std::vector<TokenId>{1, 2, 10});
    IE_CHECK(m0.matched_tokens == 3);

    const auto m2 = tree.match_prefix(std::vector<TokenId>{1, 2, 30});
    IE_CHECK(m2.matched_tokens == 3);
}

IE_TEST_CASE(tree_validation_after_repeated_mutations) {
    inference_engine::MockBackend backend(1);
    KvRadixTree tree(&backend, 16);
    Node* p = tree.append(tree.root(), std::vector<TokenId>{100, 200}, SeqRange{0, 0, 2});
    IE_CHECK(tree.validate());

    std::vector<Node*> leaves;
    for (int i = 0; i < 8; ++i) {
        static_cast<void>(tree.fork(p, static_cast<BackendSeqId>(i)));
        Node* leaf = tree.commit(p, std::vector<TokenId>{static_cast<TokenId>(300 + i)},
                                 SeqRange{static_cast<BackendSeqId>(i), 0, 3});
        leaves.push_back(leaf);
        IE_CHECK(tree.validate());
    }

    for (std::size_t i = 0; i < leaves.size(); ++i) {
        if (i % 2 == 0) {
            tree.release(leaves[i]);
            IE_CHECK(tree.validate());
        }
    }

    static_cast<void>(tree.evict(2));
    IE_CHECK(tree.validate());

    for (std::size_t i = 0; i < leaves.size(); ++i) {
        if (i % 2 != 0) {
            tree.release(leaves[i]);
            IE_CHECK(tree.validate());
        }
    }
    IE_CHECK(tree.validate());
}

IE_TEST_CASE(split_preserves_unrelated_siblings) {
    KvRadixTree tree;
    const std::vector<TokenId> first{9, 8, 7};
    const std::vector<TokenId> second{1, 2, 3};
    static_cast<void>(tree.insert_sequence(first));
    Node* split_target = tree.insert_sequence(second);
    static_cast<void>(tree.split(split_target, 1));
    IE_CHECK(tree.node_count() == 4);
    IE_CHECK(tree.match_prefix(first).matched_tokens == first.size());
    IE_CHECK(tree.match_prefix(second).matched_tokens == second.size());
    IE_CHECK(tree.validate());
    static_cast<void>(tree.insert_sequence(std::vector<TokenId>{1, 2, 4}));
    IE_CHECK(tree.match_prefix(first).matched_tokens == first.size());
    IE_CHECK(tree.match_prefix(second).matched_tokens == second.size());
    IE_CHECK(tree.validate());
}

IE_TEST_CASE(deterministic_randomized_fuzz_fixed_seed) {
    KvRadixTree tree;
    SplitMix64 rng(0xFEEDFACECAFE1234ULL);

    std::vector<std::vector<TokenId>> all_sequences;

    for (std::size_t step = 0; step < 1000; ++step) {
        const std::size_t len = 1 + static_cast<std::size_t>(rng.next() % 8);
        std::vector<TokenId> seq(len);
        for (std::size_t j = 0; j < len; ++j) {
            seq[j] = static_cast<TokenId>(rng.next() % 16);
        }
        all_sequences.push_back(seq);
        static_cast<void>(tree.insert_sequence(seq));
        IE_CHECK(tree.validate());

        if (step % 50 == 0) {
            const auto& query = all_sequences[rng.next() % all_sequences.size()];
            const auto match = tree.match_prefix(query);
            IE_CHECK(match.matched_tokens == query.size());
        }
    }
}

}  // namespace

int main() { return minitest::run_all(); }
