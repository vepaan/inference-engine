#include "kv_tree.hpp"

#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>

namespace inference_engine {

Node::Node(Node* parent_node, std::span<const TokenId> edge_tokens,
         std::optional<SeqRange> edge_holder)
    : parent(parent_node), tokens(edge_tokens.begin(), edge_tokens.end()),
    holder(edge_holder) {}

KvRadixTree::KvRadixTree() : root_(new Node(nullptr, {}, std::nullopt)) {}

KvRadixTree::~KvRadixTree() { destroy_subtree(root_); }

Node* KvRadixTree::root() noexcept { return root_; }

const Node* KvRadixTree::root() const noexcept { return root_; }

Node* KvRadixTree::append(Node* parent,
                          std::span<const TokenId> edge_tokens,
                          std::optional<SeqRange> edge_holder) {
    if (parent == nullptr) {
        throw std::invalid_argument("cannot append to a null radix node");
    }

    auto* child = new Node(parent, edge_tokens, edge_holder);
    std::lock_guard lock(structure_mutex_);
    Node* first_child = parent->first_child.load(std::memory_order_relaxed);
    child->next_sibling = first_child;
    parent->first_child.store(child, std::memory_order_release);
    return child;
}

Node* KvRadixTree::fork(Node* at, std::uint32_t seq_id) {
    static_cast<void>(seq_id);
    if (at == nullptr) {
        throw std::invalid_argument("cannot fork a null radix node");
    }

    std::lock_guard lock(structure_mutex_);
    retain_path(at);
    return at;
}

KvRadixTree::Match KvRadixTree::match_prefix(
    std::span<const TokenId> tokens) const {
    std::lock_guard lock(structure_mutex_);
    Node* current = root_;
    std::size_t matched = 0;

    while (matched < tokens.size()) {
        Node* child = current->first_child.load(std::memory_order_acquire);
        Node* matching_child = nullptr;
        while (child != nullptr) {
            const std::size_t edge_size = child->tokens.size();
            if (matched + edge_size <= tokens.size() &&
                std::equal(child->tokens.begin(), child->tokens.end(),
                           tokens.begin() + static_cast<std::ptrdiff_t>(matched))) {
                matching_child = child;
                break;
            }
            child = child->next_sibling;
        }

        if (matching_child == nullptr) {
            break;
        }
        matched += matching_child->tokens.size();
        current = matching_child;
    }

    if (matched > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("matched token count exceeds uint32_t");
    }
    return {current, static_cast<std::uint32_t>(matched)};
}

std::size_t KvRadixTree::node_count() const {
    std::lock_guard lock(structure_mutex_);
    std::size_t count = 0;
    std::vector<const Node*> pending{root_};
    while (!pending.empty()) {
        const Node* node = pending.back();
        pending.pop_back();
        ++count;
        Node* child = node->first_child.load(std::memory_order_acquire);
        while (child != nullptr) {
            pending.push_back(child);
            child = child->next_sibling;
        }
    }
    return count;
}

void KvRadixTree::destroy_subtree(Node* node) noexcept {
    if (node == nullptr) {
        return;
    }
    Node* child = node->first_child.load(std::memory_order_relaxed);
    while (child != nullptr) {
        Node* next = child->next_sibling;
        destroy_subtree(child);
        child = next;
    }
    delete node;
}

void KvRadixTree::retain_path(Node* node) {
    for (Node* current = node; current != nullptr; current = current->parent) {
        current->ref_count.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace inference_engine