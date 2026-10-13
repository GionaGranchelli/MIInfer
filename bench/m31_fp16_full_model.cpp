#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
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
    std::cout << label << " static_vram_bytes=" << runtime.total_vram_bytes()
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
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--fp16-first")) {
        std::cerr << "usage: miinfer-m31-fp16-full-model MODEL.gguf [--fp16-first]\n";
        return 2;
    }
    const bool fp16_first = argc == 3;
    miinfer::DeviceInfo device{};
    std::string error;
    if (!miinfer::validate_gfx906_device(0, device, error)) {
        std::cerr << "device validation failed: " << error << '\n';
        return 1;
    }
    char bdf[32]{};
    MIINFER_HIP_CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), 0));
    std::cout << "device_bdf=" << bdf << " architecture=" << device.architecture << '\n';

    constexpr std::uint32_t kPromptTokens = 8192;
    constexpr std::uint32_t kCapacity = 65536;
    const auto model = miinfer::Qwen35Model::load(argv[1]);
    std::vector<std::uint32_t> tokens(kPromptTokens);
    for (std::uint32_t i = 0; i < kPromptTokens; ++i) {
        tokens[i] = (i * 37U + 101U) % model.config().vocab_size;
    }

    if (::setenv("MIINFER_V2_0043_GQA_ATTENTION", "1", 1) != 0) {
        std::cerr << "failed to enable existing opt-in GQA attention path\n";
        return 1;
    }
    Result q8;
    Result fp16;
    if (fp16_first) {
        fp16 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kFp16Fp16,
                        "fp16_tiled_candidate", tokens, kCapacity);
        q8 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                      "q8_control", tokens, kCapacity);
    } else {
        q8 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                      "q8_control", tokens, kCapacity);
        fp16 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kFp16Fp16,
                        "fp16_tiled_candidate", tokens, kCapacity);
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
