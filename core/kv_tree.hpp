#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
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
    std::uint64_t last_used_epoch{0};

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
static_assert(offsetof(Node, last_used_epoch) == 72);
static_assert(sizeof(Node) == 80);

class SlotAllocator {
public:
    explicit SlotAllocator(std::uint32_t capacity);

    [[nodiscard]] std::optional<BackendSeqId> acquire(
        std::optional<BackendSeqId> preferred = std::nullopt);
    void release(BackendSeqId seq);
    [[nodiscard]] bool reserve(BackendSeqId seq);

private:
    std::vector<BackendSeqId> free_stack_;
};

class KvRadixTree {
public:
    struct Match {
        Node* node;
        std::uint32_t matched_tokens;
    };

    explicit KvRadixTree(IKvBackend* backend = nullptr,
                         std::uint32_t max_seq = 256);
    ~KvRadixTree();

    KvRadixTree(const KvRadixTree&) = delete;
    KvRadixTree& operator=(const KvRadixTree&) = delete;

    [[nodiscard]] Node* root() noexcept;
    [[nodiscard]] const Node* root() const noexcept;

    [[nodiscard]] Node* append(Node* parent,
                                std::span<const TokenId> edge_tokens,
                                std::optional<SeqRange> edge_holder = std::nullopt);

    [[nodiscard]] Node* insert_sequence(
        std::span<const TokenId> tokens,
        std::optional<SeqRange> holder = std::nullopt);

    [[nodiscard]] Node* commit(Node* parent,
                               std::span<const TokenId> edge_tokens,
                               SeqRange holder);

    [[nodiscard]] Node* split(Node* node, std::uint32_t edge_offset);

    // Retains the path to a logical branch. KV pages remain owned by the tree.
    [[nodiscard]] Node* fork(Node* at,
                             BackendSeqId preferred_seq = -1);

    void release(Node* branch);
    [[nodiscard]] std::size_t evict(std::size_t cells_needed);
    [[nodiscard]] bool validate() const;

    [[nodiscard]] Match match_prefix(std::span<const TokenId> tokens) const;

    [[nodiscard]] std::size_t node_count() const;

private:
    Node* root_;
    IKvBackend* backend_;
    SlotAllocator slots_;
    mutable std::mutex structure_mutex_;
    std::uint64_t logical_clock_{0};

    static void destroy_subtree(Node* node) noexcept;
    static void retain_path(Node* node);
    static void release_path(Node* node);
    static std::size_t depth(const Node* node);
    static Node* find_leaf(Node* node);
    static bool has_children(const Node* node);
    static void validate_node(const Node* node, const Node* expected_parent,
                              bool is_root);
    void mark_used(Node* node);
    void detach(Node* node);
};

}  // namespace inference_engine