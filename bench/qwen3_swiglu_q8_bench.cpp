#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/q4_q8_gemv.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kElements = 12288;
constexpr int kWarmup = 30;
constexpr int kIterations = 1000;

struct Stats {
    double mean_us = 0.0;
    double median_us = 0.0;
};

Stats summarize(std::vector<float> samples) {
    if (samples.empty()) throw std::runtime_error("empty benchmark sample set");
    std::sort(samples.begin(), samples.end());
    double sum = 0.0;
    for (const float sample : samples) sum += sample;
    Stats result;
    result.mean_us = sum / static_cast<double>(samples.size());
    const auto middle = samples.size() / 2;
    result.median_us = samples.size() % 2 == 0
        ? (static_cast<double>(samples[middle - 1]) + samples[middle]) / 2.0
        : samples[middle];
    return result;
}

std::vector<float> measure(
    bool fused,
    const float* gate,
    const float* up,
    float* swiglu,
    __half* swiglu_half,
    miinfer::Q8ExactBlock* output) {
    const auto launch = [&] {
        if (fused) {
            miinfer::launch_silu_mul_q8_exact(gate, up, output, kElements);
        } else {
            miinfer::launch_qwen3_silu_mul(gate, up, swiglu, kElements);
            miinfer::launch_qwen3_f32_to_f16(swiglu, swiglu_half, kElements);
            miinfer::launch_q8_exact_quantize(swiglu_half, output, kElements);
        }
    };
    for (int index = 0; index < kWarmup; ++index) launch();
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(kIterations);
    for (int index = 0; index < kIterations; ++index) {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        MIINFER_HIP_CHECK(hipEventCreate(&start));
        MIINFER_HIP_CHECK(hipEventCreate(&stop));
        MIINFER_HIP_CHECK(hipEventRecord(start));
        launch();
        MIINFER_HIP_CHECK(hipEventRecord(stop));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(milliseconds * 1000.0F);
        MIINFER_HIP_CHECK(hipEventDestroy(stop));
        MIINFER_HIP_CHECK(hipEventDestroy(start));
    }
    return samples;
}

}  // namespace

int main() {
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) {
        std::cerr << "SwiGLU/Q8 benchmark unavailable: " << error << '\n';
        return 1;
    }

    std::vector<float> gate(kElements);
    std::vector<float> up(kElements);
    std::mt19937 generator(0xC9B6U);
    std::uniform_real_distribution<float> distribution(-12.0F, 12.0F);
    for (int index = 0; index < kElements; ++index) {
        gate[static_cast<std::size_t>(index)] = distribution(generator);
        up[static_cast<std::size_t>(index)] = distribution(generator);
    }

    float* device_gate = nullptr;
    float* device_up = nullptr;
    float* device_swiglu = nullptr;
    __half* device_swiglu_half = nullptr;
    miinfer::Q8ExactBlock* device_separate = nullptr;
    miinfer::Q8ExactBlock* device_fused = nullptr;
    const auto blocks = static_cast<std::size_t>(kElements / miinfer::kQ8_1BlockSize);
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_gate), gate.size() * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_up), up.size() * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_swiglu), gate.size() * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_swiglu_half),
                                gate.size() * sizeof(__half)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_separate),
                                blocks * sizeof(miinfer::Q8ExactBlock)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_fused),
                                blocks * sizeof(miinfer::Q8ExactBlock)));
    MIINFER_HIP_CHECK(hipMemcpy(device_gate, gate.data(), gate.size() * sizeof(float),
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(device_up, up.data(), up.size() * sizeof(float),
                                hipMemcpyHostToDevice));

    const auto separate_samples = measure(false, device_gate, device_up, device_swiglu,
                                          device_swiglu_half, device_separate);
    const auto fused_samples = measure(true, device_gate, device_up, device_swiglu,
                                       device_swiglu_half, device_fused);
    std::vector<miinfer::Q8ExactBlock> separate(blocks);
    std::vector<miinfer::Q8ExactBlock> fused(blocks);
    MIINFER_HIP_CHECK(hipMemcpy(separate.data(), device_separate,
                                separate.size() * sizeof(miinfer::Q8ExactBlock),
                                hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(fused.data(), device_fused,
                                fused.size() * sizeof(miinfer::Q8ExactBlock),
                                hipMemcpyDeviceToHost));
    const bool byte_identity = std::memcmp(separate.data(), fused.data(),
                                           separate.size() * sizeof(miinfer::Q8ExactBlock)) == 0;
    const auto separate_stats = summarize(separate_samples);
    const auto fused_stats = summarize(fused_samples);
    std::cout << std::fixed << std::setprecision(3)
              << "SwiGLU+Q8Exact microbenchmark elements=" << kElements
              << " warmup=" << kWarmup << " iterations=" << kIterations << '\n'
              << "separate median_us=" << separate_stats.median_us
              << " mean_us=" << separate_stats.mean_us << " dispatches=3\n"
              << "fused median_us=" << fused_stats.median_us
              << " mean_us=" << fused_stats.mean_us << " dispatches=1\n"
              << "byte_identity=" << (byte_identity ? "PASS" : "FAIL") << '\n';

    MIINFER_HIP_CHECK(hipFree(device_fused));
    MIINFER_HIP_CHECK(hipFree(device_separate));
    MIINFER_HIP_CHECK(hipFree(device_swiglu_half));
    MIINFER_HIP_CHECK(hipFree(device_swiglu));
    MIINFER_HIP_CHECK(hipFree(device_up));
    MIINFER_HIP_CHECK(hipFree(device_gate));
    return byte_identity ? 0 : 1;
}
