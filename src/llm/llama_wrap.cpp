#include "llama_wrap.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace inference_engine::llm {

namespace {

std::size_t checked_kappa(int32_t n_layer, int32_t n_head_kv,
                          int32_t head_dim) {
    if (n_layer <= 0 || n_head_kv <= 0 || head_dim <= 0) {
        throw std::runtime_error("model reports invalid KV dimensions");
    }
    const std::uintmax_t bytes =
        static_cast<std::uintmax_t>(2) * static_cast<std::uintmax_t>(n_layer) *
        static_cast<std::uintmax_t>(n_head_kv) *
        static_cast<std::uintmax_t>(head_dim) * static_cast<std::uintmax_t>(2);
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("KV bytes per token exceed size_t");
    }
    return static_cast<std::size_t>(bytes);
}

llama_batch make_batch(std::span<const llama_seq_id> seqs,
                       std::span<const llama_token> tokens,
                       std::span<const llama_pos> positions,
                       std::uint32_t n_seq_max, bool all_logits) {
    if (seqs.size() != tokens.size() || seqs.size() != positions.size() ||
        seqs.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("batch spans have incompatible sizes");
    }
    auto batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0,
                                  static_cast<int32_t>(n_seq_max));
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        batch.token[index] = tokens[index];
        batch.pos[index] = positions[index];
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = seqs[index];
        batch.logits[index] = all_logits || index + 1 == tokens.size() ? 1 : 0;
    }
    return batch;
}

}  // namespace

Model::Model(const std::string& path) : model_(nullptr), info_{} {
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_ = llama_model_load_from_file(path.c_str(), params);
    if (model_ == nullptr) {
        llama_backend_free();
        throw std::runtime_error("failed to load model: " + path);
    }

    const int32_t n_head = llama_model_n_head(model_);
    const int32_t n_embd = llama_model_n_embd(model_);
    if (n_head <= 0 || n_embd % n_head != 0) {
        llama_model_free(model_);
        llama_backend_free();
        throw std::runtime_error("model has incompatible embedding dimensions");
    }
    const auto file_size = std::filesystem::file_size(path);
    const auto* vocab = llama_model_get_vocab(model_);
    info_ = {
        llama_model_n_layer(model_),
        n_head,
        llama_model_n_head_kv(model_),
        n_embd,
        n_embd / n_head,
        llama_vocab_n_tokens(vocab),
        file_size,
        checked_kappa(llama_model_n_layer(model_), llama_model_n_head_kv(model_),
                      n_embd / n_head),
    };
}

Model::~Model() {
    if (model_ != nullptr) {
        llama_model_free(model_);
        llama_backend_free();
    }
}

const ModelInfo& Model::info() const noexcept { return info_; }

llama_model* Model::handle() const noexcept { return model_; }

Context::Context(const Model& model, std::uint32_t n_ctx,
                 std::uint32_t n_batch, std::uint32_t n_seq_max,
                 bool kv_unified)
    : context_(nullptr), n_batch_(n_batch), n_seq_max_(n_seq_max),
      n_vocab_(model.info().n_vocab) {
    if (n_ctx == 0 || n_batch == 0 || n_seq_max == 0 ||
        n_seq_max > 256) {
        throw std::invalid_argument("invalid llama context dimensions");
    }
    auto params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.n_seq_max = n_seq_max;
    params.n_outputs_max = n_batch;
    params.kv_unified = kv_unified;
    params.no_perf = false;
    context_ = llama_init_from_model(model.handle(), params);
    if (context_ == nullptr) {
        throw std::runtime_error("failed to create llama context");
    }
}

Context::~Context() {
    if (context_ != nullptr) {
        llama_free(context_);
    }
}

DecodeStatus Context::prefill(llama_seq_id seq,
                              std::span<const llama_token> tokens,
                              llama_pos pos0) {
    std::size_t offset = 0;
    while (offset < tokens.size()) {
        const std::size_t count =
            std::min<std::size_t>(n_batch_, tokens.size() - offset);
        std::vector<llama_seq_id> seqs(count, seq);
        std::vector<llama_pos> positions(count);
        for (std::size_t index = 0; index < count; ++index) {
            positions[index] = pos0 + static_cast<llama_pos>(offset + index);
        }
        auto batch = make_batch(
            std::span<const llama_seq_id>(seqs),
            tokens.subspan(offset, count), std::span<const llama_pos>(positions),
            n_seq_max_, false);
        const DecodeStatus status = decode(batch);
        llama_batch_free(batch);
        if (status != DecodeStatus::Ok) {
            return status;
        }
        offset += count;
    }
    return DecodeStatus::Ok;
}

DecodeResult Context::decode_lockstep(
    std::span<const llama_seq_id> seqs, std::span<const llama_token> tokens,
    std::span<const llama_pos> positions) {
    auto batch = make_batch(seqs, tokens, positions, n_seq_max_, true);
    const DecodeStatus status = decode(batch);
    if (status != DecodeStatus::Ok) {
        llama_batch_free(batch);
        return {status, {}};
    }
    std::vector<llama_token> result;
    result.reserve(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        float* logits = llama_get_logits_ith(
            context_, static_cast<int32_t>(index));
        if (logits == nullptr) {
            llama_batch_free(batch);
            return {DecodeStatus::Error, {}};
        }
        const auto* max_it = std::max_element(logits, logits + n_vocab_);
        result.push_back(static_cast<llama_token>(max_it - logits));
    }
    llama_batch_free(batch);
    return {DecodeStatus::Ok, std::move(result)};
}

std::vector<float> Context::logits_ith(std::int32_t index) const {
    float* logits = llama_get_logits_ith(context_, index);
    if (logits == nullptr) {
        throw std::runtime_error("llama returned null logits");
    }
    return std::vector<float>(logits, logits + n_vocab_);
}

void Context::seq_cp(llama_seq_id src_seq, llama_seq_id dst_seq, llama_pos p0,
                     llama_pos p1) {
    llama_memory_seq_cp(llama_get_memory(context_), src_seq, dst_seq, p0, p1);
}

bool Context::seq_rm(llama_seq_id seq, llama_pos p0, llama_pos p1) {
    return llama_memory_seq_rm(llama_get_memory(context_), seq, p0, p1);
}

std::uint32_t Context::n_ctx() const noexcept { return llama_n_ctx(context_); }

std::uint32_t Context::n_seq_max() const noexcept { return n_seq_max_; }

DecodeStatus Context::decode(llama_batch batch) const {
    const int32_t result = llama_decode(context_, batch);
    if (result == 0) {
        return DecodeStatus::Ok;
    }
    if (result == 1) {
        return DecodeStatus::NoKvSlot;
    }
    return DecodeStatus::Error;
}

}  // namespace inference_engine::llm