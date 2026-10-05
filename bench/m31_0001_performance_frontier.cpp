#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/sha256.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using miinfer::Qwen35Model;
using miinfer::sha256_file;
using miinfer::prefill_v2::GenerateOptions;
using miinfer::prefill_v2::GenerateStats;
using miinfer::prefill_v2::PrefillV2Model;

namespace {

struct ContextConfig {
    const char* name;
    std::size_t tokens;
    std::uint32_t seed;
};

std::vector<std::uint32_t> prompt_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint32_t> result(count);
    std::uint32_t value = seed;
    for (auto& token : result) {
        value = value * 1664525u + 1013904223u;
        token = value % 151643u + 1;
    }
    return result;
}

GenerateOptions options(std::size_t decode_tokens) {
    GenerateOptions result;
    result.max_new_tokens = decode_tokens;
    result.reset_state_before = true;
    result.use_hip_graph = true;
    result.temperature = 0.0f;
    result.top_p = 1.0f;
    result.top_k = 1;
    result.stop_token_ids.clear();
    result.seed = 42;
    return result;
}

bool valid(const GenerateStats& stats, std::uint32_t vocab_size, std::size_t expected) {
    if (stats.generated_tokens.size() != expected) return false;
    for (const auto token : stats.generated_tokens) {
        if (token == 0 || token >= vocab_size) return false;
    }
    return true;
}

void print_row(
    const ContextConfig& context,
    std::size_t decode_tokens,
    double cold_ttft_ms,
    const GenerateStats& warm,
    std::size_t vram_bytes,
    std::size_t free_bytes,
    bool parity) {
    std::cout << std::fixed << std::setprecision(3)
              << "{\"context\":\"" << context.name
              << "\",\"prompt_tokens\":" << context.tokens
              << ",\"decode_tokens\":" << decode_tokens
              << ",\"cold_ttft_ms\":" << cold_ttft_ms
              << ",\"warm_ttft_ms\":" << warm.ttft_ms
              << ",\"prefill_tok_s\":" << warm.prefill_tok_per_sec
              << ",\"decode_tok_s\":" << warm.decode_tok_per_sec
              << ",\"decode_step_ms\":" << warm.avg_decode_latency_ms
              << ",\"vram_bytes\":" << vram_bytes
              << ",\"free_vram_bytes\":" << free_bytes
              << ",\"output_parity\":\"" << (parity ? "PASS" : "FAIL") << "\"}\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-m31-0001-performance-frontier MODEL.gguf\n";
        return 2;
    }

    try {
        const std::string model_path = argv[1];
        const auto model_data = Qwen35Model::load(model_path);
        const std::vector<ContextConfig> contexts = {
            {"8K", 8192, 77}, {"32K", 32768, 123},
            {"64K", 65536, 456}, {"128K", 131072, 789},
        };

        std::cout << "M31-0001 performance frontier\n"
                  << "model_sha256=" << sha256_file(model_path) << "\n"
                  << "semantics=synthetic prompt; warm means repeated prompt after one TG128 warmup; greedy\n";

        for (const auto& context : contexts) {
            PrefillV2Model model(model_data, static_cast<std::uint32_t>(context.tokens + 256), true);
            const auto prompt = prompt_tokens(context.tokens, context.seed);
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            MIINFER_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));

            const auto cold = model.generate(prompt, options(128));
            (void)model.generate(prompt, options(128));
            const auto warm128 = model.generate(prompt, options(128));
            const bool cold_valid = valid(cold, model.vocab_size(), 128);
            const bool warm128_valid = valid(warm128, model.vocab_size(), 128);
            print_row(context, 128, cold.ttft_ms, warm128, model.total_vram_bytes(), free_bytes,
                      cold_valid && warm128_valid && cold.generated_tokens == warm128.generated_tokens);

            for (const auto decode_tokens : {std::size_t{32}, std::size_t{64}}) {
                const auto warm = model.generate(prompt, options(decode_tokens));
                print_row(context, decode_tokens, cold.ttft_ms, warm, model.total_vram_bytes(), free_bytes,
                          valid(warm, model.vocab_size(), decode_tokens));
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "M31-0001 performance frontier: FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
