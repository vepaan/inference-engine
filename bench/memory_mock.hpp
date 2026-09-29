#pragma once

#include "kv_tree.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace inference_engine {

class MockKvAllocator {
public:
    explicit MockKvAllocator(std::size_t bytes_per_token)
        : bytes_per_token_(bytes_per_token) {}

    [[nodiscard]] std::vector<PageId> allocate_tokens(std::size_t token_count) {
        if (token_count > std::numeric_limits<std::size_t>::max() / bytes_per_token_) {
            throw std::overflow_error("mock KV allocation size overflow");
        }
        const std::size_t bytes = token_count * bytes_per_token_;
        allocated_bytes_ += bytes;
        std::vector<PageId> pages;
        pages.reserve(token_count);
        for (std::size_t index = 0; index < token_count; ++index) {
            pages.push_back(next_page_id_++);
        }
        return pages;
    }

    [[nodiscard]] std::size_t allocated_bytes() const noexcept {
        return allocated_bytes_;
    }

private:
    std::size_t bytes_per_token_;
    std::size_t allocated_bytes_{0};
    PageId next_page_id_{0};
};

}  // namespace inference_engine