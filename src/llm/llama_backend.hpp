#pragma once

#include "kv_backend.hpp"
#include "llama_wrap.hpp"

namespace inference_engine::llm {

class LlamaBackend final : public IKvBackend {
public:
    explicit LlamaBackend(Context& context) : context_(context) {}

    void share(BackendSeqId src_seq, BackendSeqId dst_seq, BackendPos p0,
               BackendPos p1) override;
    void drop(BackendSeqId seq, BackendPos p0, BackendPos p1) override;
    [[nodiscard]] std::size_t cells_in_use() const noexcept override;

private:
    Context& context_;
    std::size_t cells_in_use_{0};
};

}  // namespace inference_engine::llm
