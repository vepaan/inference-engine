#include "kv_tree.hpp"
#include "memory_mock.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using inference_engine::KvRadixTree;
using inference_engine::MockKvAllocator;
using inference_engine::TokenId;

constexpr std::size_t kBranchFactor = 8;
constexpr std::size_t kSharedPrefixTokens = 2048;
constexpr std::size_t kSuffixTokens = 200;
constexpr std::size_t kBytesPerToken = 32U * 1024U;

[[nodiscard]] std::vector<TokenId> make_tokens(std::size_t count,
                                                TokenId first_token) {
    std::vector<TokenId> tokens;
    tokens.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        tokens.push_back(first_token + static_cast<TokenId>(index));
    }
    return tokens;
}

[[nodiscard]] double to_megabytes(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

int main() {
    MockKvAllocator naive_allocator(kBytesPerToken);
    for (std::size_t branch = 0; branch < kBranchFactor; ++branch) {
        static_cast<void>(naive_allocator.allocate_tokens(kSharedPrefixTokens));
        static_cast<void>(naive_allocator.allocate_tokens(kSuffixTokens));
    }

    MockKvAllocator shared_allocator(kBytesPerToken);
    KvRadixTree tree;
    const std::vector<TokenId> prefix_tokens =
        make_tokens(kSharedPrefixTokens, 0);
    const std::vector<inference_engine::PageId> prefix_pages =
        shared_allocator.allocate_tokens(kSharedPrefixTokens);
    inference_engine::Node* prefix = tree.append(
        tree.root(), prefix_tokens, prefix_pages);

    for (std::size_t branch = 0; branch < kBranchFactor; ++branch) {
        const std::vector<TokenId> suffix_tokens =
            make_tokens(kSuffixTokens, static_cast<TokenId>(branch + 1));
        const std::vector<inference_engine::PageId> suffix_pages =
            shared_allocator.allocate_tokens(kSuffixTokens);
        inference_engine::Node* leaf =
            tree.append(prefix, suffix_tokens, suffix_pages);
        static_cast<void>(
            tree.fork(leaf, static_cast<std::uint32_t>(branch)));
    }

    const double naive_megabytes = to_megabytes(naive_allocator.allocated_bytes());
    const double shared_megabytes = to_megabytes(shared_allocator.allocated_bytes());
    const double reduction =
        (1.0 - (shared_megabytes / naive_megabytes)) * 100.0;

    std::cout << std::fixed << std::setprecision(2)
              << "Concurrent Radix-Tree KV Cache MVP\n"
              << "  branches: " << kBranchFactor << "\n"
              << "  prefix tokens: " << kSharedPrefixTokens << "\n"
              << "  suffix tokens per branch: " << kSuffixTokens << "\n"
              << "  bytes per token: " << kBytesPerToken << "\n"
              << "  naive KV allocation: " << naive_megabytes << " MiB\n"
              << "  prefix-paged KV allocation: " << shared_megabytes
              << " MiB\n"
              << "  peak allocation reduction: " << reduction << "%\n"
              << "  radix-tree nodes: " << tree.node_count() << "\n"
              << "  shared prefix ref_count: "
              << prefix->ref_count.load(std::memory_order_relaxed) << "\n";

    return reduction >= 60.0 ? 0 : 1;
}