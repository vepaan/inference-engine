#include "llama_wrap.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: ie_smoke <model.gguf>\n";
        return 2;
    }

    try {
        inference_engine::llm::Model model(argv[1]);
        const auto& info = model.info();
        std::cout << "model bytes: " << info.file_bytes << "\n"
                  << "layers: " << info.n_layer << "\n"
                  << "heads: " << info.n_head << "\n"
                  << "KV heads: " << info.n_head_kv << "\n"
                  << "embedding: " << info.n_embd << "\n"
                  << "head dim: " << info.head_dim << "\n"
                  << "vocab: " << info.n_vocab << "\n"
                  << "kappa bytes: " << info.kappa_bytes << "\n";

        inference_engine::llm::Context context(model, 512, 32, 1, true);
        std::vector<llama_token> prompt;
        prompt.reserve(256);
        for (std::int32_t index = 0; index < 256; ++index) {
            prompt.push_back(static_cast<llama_token>(1 + index));
        }
        const auto prefill_status = context.prefill(0, prompt, 0);
        if (prefill_status != inference_engine::llm::DecodeStatus::Ok) {
            throw std::runtime_error("prefill failed");
        }

        llama_token next_token = 1;
        std::vector<llama_token> generated;
        generated.reserve(32);
        const auto start = std::chrono::steady_clock::now();
        for (std::int32_t index = 0; index < 32; ++index) {
            const llama_pos position = 256 + index;
            const std::array<llama_seq_id, 1> seqs{0};
            const std::array<llama_token, 1> tokens{next_token};
            const std::array<llama_pos, 1> positions{position};
            const auto output = context.decode_lockstep(seqs, tokens, positions);
            if (output.status != inference_engine::llm::DecodeStatus::Ok) {
                throw std::runtime_error("decode failed");
            }
            next_token = output.tokens.front();
            generated.push_back(next_token);
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start);
        std::cout << "generated tokens:";
        for (const llama_token token : generated) {
            std::cout << ' ' << token;
        }
        std::cout << "\nms/token: " << elapsed.count() / generated.size()
                  << "\n";
    } catch (const std::exception& error) {
        std::cerr << "ie_smoke: " << error.what() << "\n";
        return 1;
    }
    return 0;
}