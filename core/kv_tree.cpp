#include "kv_tree.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

namespace inference_engine {

Node::Node(Node* parent_node, std::span<const TokenId> edge_tokens,
         std::optional<SeqRange> edge_holder)
    : parent(parent_node), tokens(edge_tokens.begin(), edge_tokens.end()),
    holder(edge_holder) {}

SlotAllocator::SlotAllocator(std::uint32_t capacity) {
    free_stack_.reserve(capacity);
    for (std::uint32_t index = capacity; index > 0; --index) {
        free_stack_.push_back(static_cast<BackendSeqId>(index - 1));
    }
}

std::optional<BackendSeqId> SlotAllocator::acquire(
    std::optional<BackendSeqId> preferred) {
    if (preferred.has_value()) {
        const auto it = std::find(free_stack_.begin(), free_stack_.end(),
                                  preferred.value());
        if (it != free_stack_.end()) {
            free_stack_.erase(it);
            return preferred;
        }
    }
    if (free_stack_.empty()) {
        return std::nullopt;
    }
    const BackendSeqId result = free_stack_.back();
    free_stack_.pop_back();
    return result;
}

void SlotAllocator::release(BackendSeqId seq) {
    if (std::find(free_stack_.begin(), free_stack_.end(), seq) ==
        free_stack_.end()) {
        free_stack_.push_back(seq);
    }
}

bool SlotAllocator::reserve(BackendSeqId seq) {
    const auto it = std::find(free_stack_.begin(), free_stack_.end(), seq);
    if (it == free_stack_.end()) {
        return false;
    }
    free_stack_.erase(it);
    return true;
}

KvRadixTree::KvRadixTree(IKvBackend* backend, std::uint32_t max_seq)
    : root_(new Node(nullptr, {}, std::nullopt)),
      backend_(backend),
      slots_(max_seq) {
    if (max_seq == 0 || max_seq > 256) {
        throw std::invalid_argument("invalid sequence slot capacity");
    }
    static_cast<void>(slots_.reserve(0));
}

KvRadixTree::~KvRadixTree() { destroy_subtree(root_); }

Node* KvRadixTree::root() noexcept { return root_; }

const Node* KvRadixTree::root() const noexcept { return root_; }

Node* KvRadixTree::append(Node* parent,
                          std::span<const TokenId> edge_tokens,
                          std::optional<SeqRange> edge_holder) {
    if (parent == nullptr) {
        throw std::invalid_argument("cannot append to a null radix node");
    }

    std::lock_guard lock(structure_mutex_);
    Node* attach_parent = parent;
    std::span<const TokenId> remaining = edge_tokens;
    while (!remaining.empty()) {
        Node* matching_child = nullptr;
        for (Node* sibling = attach_parent->first_child.load(
                 std::memory_order_relaxed);
             sibling != nullptr; sibling = sibling->next_sibling) {
            if (!sibling->tokens.empty() &&
                sibling->tokens.front() == remaining.front()) {
                matching_child = sibling;
                break;
            }
        }

        if (matching_child == nullptr) {
            break;
        }

        const std::size_t common_limit =
            std::min(matching_child->tokens.size(), remaining.size());
        std::size_t common = 0;
        while (common < common_limit &&
               matching_child->tokens[common] == remaining[common]) {
            ++common;
        }
        if (common == matching_child->tokens.size()) {
            if (common == remaining.size()) {
                if (edge_holder.has_value()) {
                    matching_child->holder = edge_holder;
                    matching_child->ref_count.fetch_add(
                        1, std::memory_order_relaxed);
                }
                return matching_child;
            }
            attach_parent = matching_child;
            remaining = remaining.subspan(common);
            continue;
        }

        auto middle = std::make_unique<Node>(
            attach_parent,
            std::span<const TokenId>(matching_child->tokens.data(), common),
            std::nullopt);
        Node* previous = nullptr;
        Node* sibling = attach_parent->first_child.load(
            std::memory_order_relaxed);
        while (sibling != matching_child) {
            previous = sibling;
            sibling = sibling->next_sibling;
        }
        matching_child->tokens.erase(
            matching_child->tokens.begin(),
            matching_child->tokens.begin() + static_cast<std::ptrdiff_t>(common));
        middle->ref_count.store(
            matching_child->ref_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        matching_child->parent = middle.get();
        middle->first_child.store(matching_child, std::memory_order_relaxed);
        middle->next_sibling = matching_child->next_sibling;
        matching_child->next_sibling = nullptr;
        Node* middle_ptr = middle.release();
        if (previous == nullptr) {
            attach_parent->first_child.store(middle_ptr,
                                             std::memory_order_release);
        } else {
            previous->next_sibling = middle_ptr;
        }
        attach_parent = middle_ptr;
        remaining = remaining.subspan(common);
    }

    if (attach_parent->first_child.load(std::memory_order_relaxed) == nullptr &&
        attach_parent->holder.has_value()) {
        if (!edge_holder.has_value()) {
            edge_holder = attach_parent->holder;
        }
        attach_parent->holder.reset();
    }
    auto child = std::make_unique<Node>(attach_parent, remaining, edge_holder);
    Node* existing = attach_parent->first_child.load(std::memory_order_relaxed);
    child->next_sibling = existing;
    Node* result = child.release();
    attach_parent->first_child.store(result, std::memory_order_release);
    if (edge_holder.has_value()) {
        result->last_used_epoch = ++logical_clock_;
    }
    return result;
}

Node* KvRadixTree::insert_sequence(std::span<const TokenId> tokens,
                                   std::optional<SeqRange> holder) {
    if (tokens.empty()) {
        return root_;
    }
    Match match = match_prefix(tokens);
    Node* parent = match.node;
    if (parent != root_ && match.matched_tokens < tokens.size()) {
        const std::size_t parent_depth = depth(parent->parent);
        const std::size_t edge_offset = match.matched_tokens - parent_depth;
        if (edge_offset > 0 && edge_offset < parent->tokens.size()) {
            parent = split(parent, static_cast<std::uint32_t>(edge_offset));
        }
    }
    if (parent != root_ && match.matched_tokens == tokens.size() &&
        depth(parent) > tokens.size()) {
        const std::size_t edge_offset =
            tokens.size() - depth(parent->parent);
        return split(parent, static_cast<std::uint32_t>(edge_offset));
    }
    const std::size_t consumed = depth(parent);
    if (consumed == tokens.size()) {
        if (holder.has_value()) {
            parent->holder = holder;
            parent->ref_count.store(1, std::memory_order_relaxed);
        }
        return parent;
    }
    return append(parent, tokens.subspan(consumed), holder);
}

Node* KvRadixTree::commit(Node* parent,
                          std::span<const TokenId> edge_tokens,
                          SeqRange holder) {
    if (parent == nullptr) {
        throw std::invalid_argument("cannot commit to a null radix node");
    }
    if (parent->holder.has_value() &&
        parent->ref_count.load(std::memory_order_relaxed) == 0) {
        std::lock_guard lock(structure_mutex_);
        retain_path(parent);
    }
    Node* leaf = append(parent, edge_tokens, holder);
    std::lock_guard lock(structure_mutex_);
    leaf->ref_count.store(1, std::memory_order_relaxed);
    leaf->last_used_epoch = ++logical_clock_;
    return leaf;
}

Node* KvRadixTree::split(Node* node, std::uint32_t edge_offset) {
    if (node == nullptr || node == root_ || edge_offset == 0 ||
        edge_offset >= node->tokens.size()) {
        throw std::invalid_argument("invalid radix edge split");
    }
    std::lock_guard lock(structure_mutex_);
    auto middle = std::make_unique<Node>(node->parent,
                                         std::span<const TokenId>(
                                             node->tokens.data(), edge_offset),
                                         std::nullopt);
    node->tokens.erase(node->tokens.begin(),
                       node->tokens.begin() + edge_offset);
    middle->ref_count.store(
        node->ref_count.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    Node* parent = node->parent;
    Node* previous = nullptr;
    Node* child = parent->first_child.load(std::memory_order_relaxed);
    while (child != node) {
        if (child == nullptr) {
            throw std::logic_error("radix node is detached from its parent");
        }
        previous = child;
        child = child->next_sibling;
    }
    middle->first_child.store(node, std::memory_order_relaxed);
    node->next_sibling = nullptr;
    node->parent = middle.get();
    Node* middle_ptr = middle.release();
    if (previous == nullptr) {
        parent->first_child.store(middle_ptr, std::memory_order_release);
    } else {
        previous->next_sibling = middle_ptr;
    }
    return middle_ptr;
}

Node* KvRadixTree::fork(Node* at, BackendSeqId preferred_seq) {
    if (at == nullptr) {
        throw std::invalid_argument("cannot fork a null radix node");
    }

    std::lock_guard lock(structure_mutex_);
    Node* source = find_leaf(at);
    if (source == nullptr || !source->holder.has_value()) {
        throw std::invalid_argument("cannot fork a node without a leaf holder");
    }
    const bool same_source_slot =
        preferred_seq >= 0 && preferred_seq == source->holder->seq;
    if (backend_ != nullptr && !same_source_slot) {
        const auto destination = slots_.acquire(
            preferred_seq >= 0 ? std::optional<BackendSeqId>(preferred_seq)
                               : std::nullopt);
        if (!destination.has_value()) {
            throw std::runtime_error("no physical sequence slot is available");
        }
        backend_->share(source->holder->seq, destination.value(), -1, -1);
    }
    retain_path(at);
    return at;
}

void KvRadixTree::release(Node* branch) {
    if (branch == nullptr || branch == root_) {
        throw std::invalid_argument("cannot release a null or root branch");
    }
    std::lock_guard lock(structure_mutex_);
    release_path(branch);
    if (branch->ref_count.load(std::memory_order_relaxed) == 0) {
        branch->last_used_epoch = ++logical_clock_;
    }
}

std::size_t KvRadixTree::evict(std::size_t cells_needed) {
    std::lock_guard lock(structure_mutex_);
    std::vector<Node*> candidates;
    std::vector<Node*> pending{root_};
    while (!pending.empty()) {
        Node* node = pending.back();
        pending.pop_back();
        if (node != root_ && node->first_child.load(std::memory_order_relaxed) == nullptr &&
            node->holder.has_value() &&
            node->ref_count.load(std::memory_order_relaxed) == 0) {
            candidates.push_back(node);
        }
        for (Node* child = node->first_child.load(std::memory_order_relaxed);
             child != nullptr; child = child->next_sibling) {
            pending.push_back(child);
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Node* left, const Node* right) {
                  return left->last_used_epoch < right->last_used_epoch;
              });
    std::size_t freed = 0;
    for (Node* leaf : candidates) {
        if (freed >= cells_needed || !leaf->holder.has_value()) {
            break;
        }
        const std::size_t start = depth(leaf->parent);
        const std::size_t end = depth(leaf);
        if (backend_ != nullptr) {
            backend_->drop(leaf->holder->seq,
                           static_cast<BackendPos>(start), -1);
        }
        slots_.release(leaf->holder->seq);
        freed += end - start;
        detach(leaf);
        delete leaf;
    }
    return freed;
}

bool KvRadixTree::validate() const {
    std::lock_guard lock(structure_mutex_);
    try {
        validate_node(root_, nullptr, true);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

KvRadixTree::Match KvRadixTree::match_prefix(
    std::span<const TokenId> tokens) const {
    std::lock_guard lock(structure_mutex_);
    Node* current = root_;
    std::size_t matched = 0;

    while (matched < tokens.size()) {
        Node* child = current->first_child.load(std::memory_order_acquire);
        Node* matching_child = nullptr;
        bool partial_edge = false;
        while (child != nullptr) {
            const std::size_t edge_size = child->tokens.size();
            const std::size_t remaining = tokens.size() - matched;
            const std::size_t common = std::min(edge_size, remaining);
            std::size_t equal_tokens = 0;
            while (equal_tokens < common &&
                   child->tokens[equal_tokens] ==
                       tokens[matched + equal_tokens]) {
                ++equal_tokens;
            }
            if (equal_tokens > 0) {
                matching_child = child;
                matched += equal_tokens;
                partial_edge = equal_tokens < edge_size;
                break;
            }
            child = child->next_sibling;
        }

        if (matching_child == nullptr) {
            break;
        }
        current = matching_child;
        if (partial_edge) {
            break;
        }
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
    std::vector<Node*> pending{node};
    while (!pending.empty()) {
        Node* current = pending.back();
        pending.pop_back();
        Node* child = current->first_child.load(std::memory_order_relaxed);
        while (child != nullptr) {
            pending.push_back(child);
            child = child->next_sibling;
        }
        delete current;
    }
}

void KvRadixTree::retain_path(Node* node) {
    for (Node* current = node; current != nullptr && current->parent != nullptr;
         current = current->parent) {
        current->ref_count.fetch_add(1, std::memory_order_relaxed);
    }
}

void KvRadixTree::release_path(Node* node) {
    for (Node* current = node; current != nullptr && current->parent != nullptr;
         current = current->parent) {
        const auto count = current->ref_count.load(std::memory_order_relaxed);
        if (count == 0) {
            throw std::logic_error("branch reference count underflow");
        }
        current->ref_count.fetch_sub(1, std::memory_order_acq_rel);
    }
}

std::size_t KvRadixTree::depth(const Node* node) {
    std::size_t result = 0;
    for (const Node* current = node; current != nullptr && current->parent != nullptr;
         current = current->parent) {
        result += current->tokens.size();
    }
    return result;
}

Node* KvRadixTree::find_leaf(Node* node) {
    if (node == nullptr) {
        return nullptr;
    }
    Node* current = node;
    while (current->first_child.load(std::memory_order_relaxed) != nullptr) {
        current = current->first_child.load(std::memory_order_relaxed);
    }
    return current;
}

bool KvRadixTree::has_children(const Node* node) {
    return node != nullptr &&
           node->first_child.load(std::memory_order_relaxed) != nullptr;
}

void KvRadixTree::validate_node(const Node* node, const Node* expected_parent,
                                bool is_root) {
    if (node == nullptr || node->parent != expected_parent) {
        throw std::logic_error("invalid radix parent pointer");
    }
    const Node* child = node->first_child.load(std::memory_order_relaxed);
    std::uint32_t expected_refs = 0;
    if (child == nullptr) {
        if (!is_root && node->holder.has_value()) {
            expected_refs = node->ref_count.load(std::memory_order_relaxed);
        }
    } else {
        if (node->holder.has_value()) {
            throw std::logic_error("internal node owns a sequence holder");
        }
        for (const Node* current = child; current != nullptr;
             current = current->next_sibling) {
            validate_node(current, node, false);
            expected_refs += current->ref_count.load(std::memory_order_relaxed);
        }
    }
    if (!is_root && node->ref_count.load(std::memory_order_relaxed) != expected_refs) {
        throw std::logic_error("invalid radix reference count");
    }
}

void KvRadixTree::mark_used(Node* node) {
    if (node != nullptr) {
        node->last_used_epoch = ++logical_clock_;
    }
}

void KvRadixTree::detach(Node* node) {
    Node* parent = node->parent;
    Node* previous = nullptr;
    Node* current = parent->first_child.load(std::memory_order_relaxed);
    while (current != node) {
        if (current == nullptr) {
            throw std::logic_error("cannot detach an unknown child");
        }
        previous = current;
        current = current->next_sibling;
    }
    if (previous == nullptr) {
        parent->first_child.store(node->next_sibling, std::memory_order_release);
    } else {
        previous->next_sibling = node->next_sibling;
    }
}

}  // namespace inference_engine