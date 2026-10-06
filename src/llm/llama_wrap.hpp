#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "llama.h"

namespace inference_engine::llm {

struct ModelInfo {
    int32_t n_layer;
    int32_t n_head;
    int32_t n_head_kv;
    int32_t n_embd;
    int32_t head_dim;
    int32_t n_vocab;
    std::uintmax_t file_bytes;
    std::size_t kappa_bytes;
};

class Model {
public:
    explicit Model(const std::string& path);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const ModelInfo& info() const noexcept;
    [[nodiscard]] llama_model* handle() const noexcept;

private:
    llama_model* model_;
    ModelInfo info_;
};

enum class DecodeStatus {
    Ok,
    NoKvSlot,
    Error,
};

struct DecodeResult {
    DecodeStatus status;
    std::vector<llama_token> tokens;
};

class Context {
public:
    Context(const Model& model, std::uint32_t n_ctx, std::uint32_t n_batch,
            std::uint32_t n_seq_max, bool kv_unified);
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    [[nodiscard]] DecodeStatus prefill(llama_seq_id seq,
                                       std::span<const llama_token> tokens,
                                       llama_pos pos0);
    [[nodiscard]] DecodeResult decode_lockstep(
        std::span<const llama_seq_id> seqs, std::span<const llama_token> tokens,
        std::span<const llama_pos> positions);
    [[nodiscard]] std::vector<float> logits_ith(std::int32_t index) const;
    void seq_cp(llama_seq_id src_seq, llama_seq_id dst_seq, llama_pos p0,
                llama_pos p1);
    [[nodiscard]] bool seq_rm(llama_seq_id seq, llama_pos p0, llama_pos p1);

    [[nodiscard]] std::uint32_t n_ctx() const noexcept;
    [[nodiscard]] std::uint32_t n_seq_max() const noexcept;

private:
    llama_context* context_;
    std::uint32_t n_batch_;
    std::uint32_t n_seq_max_;
    int32_t n_vocab_;

    [[nodiscard]] DecodeStatus decode(llama_batch batch) const;
};

}  // namespace inference_engine::llm