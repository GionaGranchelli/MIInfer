#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;

namespace {

struct TimingResult {
    double min_us = 1e9;
    double mean_us = 0.0;
};

template <typename F>
TimingResult measure(F&& fn, int warmup = 50, int iters = 500, hipStream_t stream = nullptr) {
    for (int i = 0; i < warmup; ++i) fn();
    MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start, stop;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));

    double sum = 0.0;
    double min_v = 1e9;
    for (int i = 0; i < iters; ++i) {
        MIINFER_HIP_CHECK(hipEventRecord(start, stream));
        fn();
        MIINFER_HIP_CHECK(hipEventRecord(stop, stream));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        double us = ms * 1000.0;
        min_v = std::min(min_v, us);
        sum += us;
    }

    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(stop));

    return {min_v, sum / iters};
}

} // namespace

int main() {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0010: GDN DeltaNet Step Kernel Bakeoff\n";
    std::cout << "  Hardware: 1 x AMD Instinct MI50 32GB (gfx906, Wave64)\n";
    std::cout << "===================================================================\n\n";

    constexpr std::uint32_t kKHeads = 16;
    constexpr std::uint32_t kVHeads = 16;
    constexpr std::uint32_t kState = 128;
    constexpr std::size_t kStateElements = kVHeads * kState * kState; // 16 * 128 * 128 = 262144 floats = 1 MiB

    hipStream_t stream = nullptr;
    MIINFER_HIP_CHECK(hipStreamCreate(&stream));

    // Allocate GPU buffers
    float *d_query = nullptr, *d_key = nullptr, *d_value = nullptr;
    float *d_beta = nullptr, *d_decay = nullptr;
    float *d_state_ref = nullptr, *d_state_test = nullptr;
    float *d_out_ref = nullptr, *d_out_test = nullptr;

    MIINFER_HIP_CHECK(hipMalloc(&d_query, kKHeads * kState * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_key, kKHeads * kState * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_value, kVHeads * kState * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_beta, kVHeads * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_decay, kVHeads * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_state_ref, kStateElements * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_state_test, kStateElements * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_out_ref, kVHeads * kState * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_out_test, kVHeads * kState * sizeof(float)));

    // Initialize random data
    std::vector<float> h_query(kKHeads * kState, 0.05f);
    std::vector<float> h_key(kKHeads * kState, 0.05f);
    std::vector<float> h_value(kVHeads * kState, 0.1f);
    std::vector<float> h_beta(kVHeads, 0.8f);
    std::vector<float> h_decay(kVHeads, 0.95f);
    std::vector<float> h_state(kStateElements, 0.01f);

    MIINFER_HIP_CHECK(hipMemcpy(d_query, h_query.data(), h_query.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_key, h_key.data(), h_key.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_value, h_value.data(), h_value.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_decay, h_decay.data(), h_decay.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_state_ref, h_state.data(), h_state.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_state_test, h_state.data(), h_state.size() * sizeof(float), hipMemcpyHostToDevice));

    // 1. Baseline: launch_qwen35_deltanet_state_update
    auto t_baseline = measure([&] {
        launch_qwen35_deltanet_state_update(d_query, d_key, d_value, d_beta, d_decay, d_state_ref, d_out_ref, kKHeads, kVHeads, kState, stream);
    }, 50, 500, stream);

    // 2. Transposed: launch_qwen35_deltanet_state_update_transposed
    auto t_transposed = measure([&] {
        launch_qwen35_deltanet_state_update_transposed(d_query, d_key, d_value, d_beta, d_decay, d_state_test, d_out_test, kKHeads, kVHeads, kState, stream);
    }, 50, 500, stream);

    // 3. Transposed No Decay Store: launch_qwen35_deltanet_state_update_transposed_no_decay_store
    auto t_no_decay = measure([&] {
        launch_qwen35_deltanet_state_update_transposed_no_decay_store(d_query, d_key, d_value, d_beta, d_decay, d_state_test, d_out_test, kKHeads, kVHeads, kState, stream);
    }, 50, 500, stream);

    // 4. Transposed LDS Inputs: launch_qwen35_deltanet_state_update_transposed_no_decay_store_lds_inputs
    auto t_lds = measure([&] {
        launch_qwen35_deltanet_state_update_transposed_no_decay_store_lds_inputs(d_query, d_key, d_value, d_beta, d_decay, d_state_test, d_out_test, kKHeads, kVHeads, kState, stream);
    }, 50, 500, stream);

    // 5. Transposed Row Waves: launch_qwen35_deltanet_state_update_transposed_row_waves
    auto t_row_waves = measure([&] {
        launch_qwen35_deltanet_state_update_transposed_row_waves(d_query, d_key, d_value, d_beta, d_decay, d_state_test, d_out_test, kKHeads, kVHeads, kState, stream);
    }, 50, 500, stream);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  1. Baseline (Uncoalesced Row-Major) : " << std::setw(8) << t_baseline.mean_us << " us (min=" << t_baseline.min_us << ")\n";
    std::cout << "  2. Transposed                       : " << std::setw(8) << t_transposed.mean_us << " us (min=" << t_transposed.min_us << ") -> SPEEDUP: " << (t_baseline.mean_us / t_transposed.mean_us) << "x\n";
    std::cout << "  3. Transposed No-Decay-Store        : " << std::setw(8) << t_no_decay.mean_us << " us (min=" << t_no_decay.min_us << ") -> SPEEDUP: " << (t_baseline.mean_us / t_no_decay.mean_us) << "x\n";
    std::cout << "  4. Transposed LDS Inputs            : " << std::setw(8) << t_lds.mean_us << " us (min=" << t_lds.min_us << ") -> SPEEDUP: " << (t_baseline.mean_us / t_lds.mean_us) << "x\n";
    std::cout << "  5. Transposed Row Waves (Wave64)    : " << std::setw(8) << t_row_waves.mean_us << " us (min=" << t_row_waves.min_us << ") -> SPEEDUP: " << (t_baseline.mean_us / t_row_waves.mean_us) << "x\n";

    double saving_per_layer_us = t_baseline.mean_us - t_row_waves.mean_us;
    double saving_48_layers_ms = saving_per_layer_us * 48.0 / 1000.0;
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Projected Savings across 48 Recurrent Layers:\n";
    std::cout << "  Savings Per Layer:  " << saving_per_layer_us << " us\n";
    std::cout << "  Full Model Savings: " << saving_48_layers_ms << " ms/token!\n";
    std::cout << "===================================================================\n";

    MIINFER_HIP_CHECK(hipFree(d_query));
    MIINFER_HIP_CHECK(hipFree(d_key));
    MIINFER_HIP_CHECK(hipFree(d_value));
    MIINFER_HIP_CHECK(hipFree(d_beta));
    MIINFER_HIP_CHECK(hipFree(d_decay));
    MIINFER_HIP_CHECK(hipFree(d_state_ref));
    MIINFER_HIP_CHECK(hipFree(d_state_test));
    MIINFER_HIP_CHECK(hipFree(d_out_ref));
    MIINFER_HIP_CHECK(hipFree(d_out_test));
    MIINFER_HIP_CHECK(hipStreamDestroy(stream));

    return 0;
}
