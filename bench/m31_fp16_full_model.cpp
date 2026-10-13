#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen3_tokenizer.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

namespace {

template <typename T>
T* allocate_device(std::size_t count) {
    T* ptr = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}

struct Result {
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    std::uint32_t top1 = 0;
    std::uint32_t second_token = 0;
    std::vector<float> logits;
};

Result run_mode(const miinfer::Qwen35Model& model,
                miinfer::prefill_v2::KvCacheQuantMode mode,
                const char* label, const std::vector<std::uint32_t>& tokens,
                std::uint32_t capacity) {
    using namespace miinfer::prefill_v2;
    PrefillV2Model runtime(model, capacity, true, mode);
    std::cout << label << " capacity_tokens=" << capacity
              << " static_vram_bytes=" << runtime.total_vram_bytes()
              << " weights_bytes=" << runtime.persistent_weight_bytes()
              << " state_bytes=" << runtime.persistent_state_bytes()
              << " workspace_bytes=" << runtime.workspace_bytes() << '\n';

    float* hidden = allocate_device<float>(tokens.size() * kHidden);
    float* logits = allocate_device<float>(model.config().vocab_size);
    runtime.reset_state();
    const auto start = std::chrono::steady_clock::now();
    runtime.prefill_sequence(std::span<const std::uint32_t>(tokens), hidden);
    runtime.compute_logits(hidden + (tokens.size() - 1) * kHidden, logits);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();

    Result result;
    result.prefill_seconds = std::chrono::duration<double>(stop - start).count();
    result.logits.resize(model.config().vocab_size);
    MIINFER_HIP_CHECK(hipMemcpy(result.logits.data(), logits,
        result.logits.size() * sizeof(float), hipMemcpyDeviceToHost));
    result.top1 = static_cast<std::uint32_t>(std::distance(
        result.logits.begin(), std::max_element(result.logits.begin(), result.logits.end())));
    const auto decode_start = std::chrono::steady_clock::now();
    result.second_token = runtime.decode_step(result.top1, static_cast<std::uint32_t>(tokens.size()));
    result.decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - decode_start).count();
    MIINFER_HIP_CHECK(hipFree(hidden));
    MIINFER_HIP_CHECK(hipFree(logits));
    std::cout << label << " prefill_seconds=" << std::fixed << std::setprecision(3)
              << result.prefill_seconds << " first_token=" << result.top1
              << " decode_seconds=" << result.decode_seconds
              << " second_token=" << result.second_token << '\n';
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    bool fp16_first = false;
    bool fp16_only = false;
    std::uint32_t prompt_tokens = 8192;
    std::string prompt_file;
    for (int i = 2; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--fp16-first") {
            fp16_first = true;
        } else if (arg == "--fp16-only") {
            fp16_only = true;
        } else if (arg == "--tokens" && i + 1 < argc) {
            char* end = nullptr;
            const auto parsed = std::strtoul(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || parsed == 0 || parsed > 65536) {
                std::cerr << "--tokens must be between 1 and 65536\n";
                return 2;
            }
            prompt_tokens = static_cast<std::uint32_t>(parsed);
        } else if (arg == "--prompt-file" && i + 1 < argc) {
            prompt_file = argv[++i];
        } else {
            std::cerr << "usage: miinfer-m31-fp16-full-model MODEL.gguf "
                         "[--fp16-first] [--fp16-only] [--tokens N] [--prompt-file PATH]\n";
            return 2;
        }
    }
    if (argc < 2 || (fp16_first && fp16_only)) {
        std::cerr << "usage: miinfer-m31-fp16-full-model MODEL.gguf "
                     "[--fp16-first] [--fp16-only] [--tokens N] [--prompt-file PATH]\n";
        return 2;
    }
    miinfer::DeviceInfo device{};
    std::string error;
    if (!miinfer::validate_gfx906_device(0, device, error)) {
        std::cerr << "device validation failed: " << error << '\n';
        return 1;
    }
    char bdf[32]{};
    MIINFER_HIP_CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), 0));
    std::cout << "device_bdf=" << bdf << " architecture=" << device.architecture << '\n';

    const std::uint32_t capacity = std::max<std::uint32_t>(65536, prompt_tokens + 4096);
    const auto model = miinfer::Qwen35Model::load(argv[1]);
    std::vector<std::uint32_t> tokens(prompt_tokens);
    if (prompt_file.empty()) {
        for (std::uint32_t i = 0; i < prompt_tokens; ++i) {
            tokens[i] = (i * 37U + 101U) % model.config().vocab_size;
        }
    } else {
        std::ifstream input(prompt_file, std::ios::binary);
        if (!input) {
            std::cerr << "failed to open prompt file: " << prompt_file << '\n';
            return 2;
        }
        const std::string text(std::istreambuf_iterator<char>(input), {});
        const auto encoded = miinfer::Qwen3Tokenizer::load(*model.file()).encode(text);
        if (encoded.size() < prompt_tokens) {
            std::cerr << "prompt file encodes to " << encoded.size()
                      << " tokens, fewer than requested " << prompt_tokens << '\n';
            return 2;
        }
        std::copy_n(encoded.begin(), prompt_tokens, tokens.begin());
        std::cout << "prompt_text_bytes=" << text.size()
                  << " prompt_tokens_available=" << encoded.size()
                  << " prompt_tokens_processed=" << prompt_tokens << '\n';
    }

    if (::setenv("MIINFER_V2_0043_GQA_ATTENTION", "1", 1) != 0) {
        std::cerr << "failed to enable existing opt-in GQA attention path\n";
        return 1;
    }
    if (fp16_only) {
        const auto fp16 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kFp16Fp16,
                                   "fp16_tiled_candidate", tokens, capacity);
        const bool finite = std::all_of(fp16.logits.begin(), fp16.logits.end(),
                                        [](float value) { return std::isfinite(value); });
        std::cout << "logits_finite=" << finite << '\n';
        return finite ? 0 : 1;
    }
    Result q8;
    Result fp16;
    if (fp16_first) {
        fp16 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kFp16Fp16,
                        "fp16_tiled_candidate", tokens, capacity);
        q8 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                      "q8_control", tokens, capacity);
    } else {
        q8 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                      "q8_control", tokens, capacity);
        fp16 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kFp16Fp16,
                        "fp16_tiled_candidate", tokens, capacity);
    }
    if (q8.logits.size() != fp16.logits.size()) return 1;
    double dot = 0.0, q8_norm = 0.0, fp16_norm = 0.0;
    double max_abs = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < q8.logits.size(); ++i) {
        finite = finite && std::isfinite(q8.logits[i]) && std::isfinite(fp16.logits[i]);
        max_abs = std::max(max_abs, std::abs(static_cast<double>(q8.logits[i]) - fp16.logits[i]));
        dot += static_cast<double>(q8.logits[i]) * fp16.logits[i];
        q8_norm += static_cast<double>(q8.logits[i]) * q8.logits[i];
        fp16_norm += static_cast<double>(fp16.logits[i]) * fp16.logits[i];
    }
    const double cosine = dot / (std::sqrt(q8_norm * fp16_norm) + 1e-30);
    std::cout << std::setprecision(9) << "logits_max_abs=" << max_abs
              << " logits_cosine=" << cosine
              << " first_token_equal=" << (q8.top1 == fp16.top1)
              << " second_token_equal=" << (q8.second_token == fp16.second_token) << '\n';
    return finite && std::isfinite(max_abs) && std::isfinite(cosine) ? 0 : 1;
}
