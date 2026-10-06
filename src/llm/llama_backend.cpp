#include "llama_backend.hpp"

namespace inference_engine::llm {

void LlamaBackend::share(BackendSeqId src_seq, BackendSeqId dst_seq,
                         BackendPos p0, BackendPos p1) {
    context_.seq_cp(src_seq, dst_seq, p0, p1);
    if (p0 >= 0 && p1 >= p0) {
        cells_in_use_ += static_cast<std::size_t>(p1 - p0);
    }
}

void LlamaBackend::drop(BackendSeqId seq, BackendPos p0, BackendPos p1) {
    if (context_.seq_rm(seq, p0, p1) && p0 >= 0 && p1 >= p0) {
        const auto count = static_cast<std::size_t>(p1 - p0);
        cells_in_use_ = count > cells_in_use_ ? 0 : cells_in_use_ - count;
    }
}

std::size_t LlamaBackend::cells_in_use() const noexcept {
    return cells_in_use_;
}

}  // namespace inference_engine::llm
