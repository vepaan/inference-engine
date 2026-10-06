#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "kv_backend.hpp"

namespace inference_engine {

using TokenId = std::int32_t;
struct Node {
    std::atomic<std::uint32_t> ref_count{0};
    std::atomic<Node*> first_child{nullptr};
    Node* next_sibling{nullptr};
    Node* parent{nullptr};
    std::vector<TokenId> tokens;
    std::optional<SeqRange> holder;

    Node(Node* parent_node, std::span<const TokenId> edge_tokens,
         std::optional<SeqRange> edge_holder);
};

static_assert(sizeof(SeqRange) == 12);
static_assert(offsetof(Node, ref_count) == 0);
static_assert(offsetof(Node, first_child) == 8);
static_assert(offsetof(Node, next_sibling) == 16);
static_assert(offsetof(Node, parent) == 24);
static_assert(offsetof(Node, tokens) == 32);
static_assert(offsetof(Node, holder) == 56);

class KvRadixTree {
public:
    struct Match {
        Node* node;
        std::uint32_t matched_tokens;
    };

    KvRadixTree();
    ~KvRadixTree();

    KvRadixTree(const KvRadixTree&) = delete;
    KvRadixTree& operator=(const KvRadixTree&) = delete;

    [[nodiscard]] Node* root() noexcept;
    [[nodiscard]] const Node* root() const noexcept;

    [[nodiscard]] Node* append(Node* parent,
                                std::span<const TokenId> edge_tokens,
                                std::optional<SeqRange> edge_holder = std::nullopt);

    // Retains the path to a logical branch. KV pages remain owned by the tree.
    [[nodiscard]] Node* fork(Node* at, std::uint32_t seq_id);

    [[nodiscard]] Match match_prefix(std::span<const TokenId> tokens) const;

    [[nodiscard]] std::size_t node_count() const;

private:
    Node* root_;
    mutable std::mutex structure_mutex_;

    static void destroy_subtree(Node* node) noexcept;
    static void retain_path(Node* node);
};

}  // namespace inference_engine