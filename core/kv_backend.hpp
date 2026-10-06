#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace inference_engine {

using BackendSeqId = std::int32_t;
using BackendPos = std::int32_t;

struct SeqRange {
    BackendSeqId seq;
    BackendPos p0;
    BackendPos p1;
};

class IKvBackend {
public:
    virtual ~IKvBackend() = default;

    virtual void share(BackendSeqId src_seq, BackendSeqId dst_seq,
                       BackendPos p0, BackendPos p1) = 0;
    virtual void drop(BackendSeqId seq, BackendPos p0, BackendPos p1) = 0;
    [[nodiscard]] virtual std::size_t cells_in_use() const noexcept = 0;
};

class MockBackend final : public IKvBackend {
public:
    explicit MockBackend(std::size_t bytes_per_token)
        : bytes_per_token_(bytes_per_token) {}

    [[nodiscard]] SeqRange allocate(BackendSeqId seq, std::size_t token_count) {
        if (token_count > (static_cast<std::size_t>(-1) / bytes_per_token_)) {
            throw std::overflow_error("mock KV allocation size overflow");
        }
        allocated_bytes_ += token_count * bytes_per_token_;
        cells_in_use_ += token_count;
        return {seq, 0, static_cast<BackendPos>(token_count)};
    }

    void share(BackendSeqId, BackendSeqId, BackendPos, BackendPos) override {}

    void drop(BackendSeqId, BackendPos p0, BackendPos p1) override {
        if (p0 >= 0 && p1 >= p0) {
            const auto count = static_cast<std::size_t>(p1 - p0);
            cells_in_use_ = count > cells_in_use_ ? 0 : cells_in_use_ - count;
        }
    }

    [[nodiscard]] std::size_t cells_in_use() const noexcept override {
        return cells_in_use_;
    }

    [[nodiscard]] std::size_t allocated_bytes() const noexcept {
        return allocated_bytes_;
    }

private:
    std::size_t bytes_per_token_;
    std::size_t allocated_bytes_{0};
    std::size_t cells_in_use_{0};
};

}  // namespace inference_engine