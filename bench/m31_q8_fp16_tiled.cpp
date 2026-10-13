#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

namespace {

template <typename T>
T* allocate_device(std::size_t count) {
    T* ptr = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}

double run_q8(const float* q, const std::int8_t* k, const __half* ks,
              const std::int8_t* v, const __half* vs, const float* gate,
              float* output, float* workspace) {
    hipEvent_t start{}, end{};
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&end));
    MIINFER_HIP_CHECK(hipEventRecord(start));
    miinfer::launch_qwen35_splitk_suffix_attention_quant(
        q, nullptr, nullptr, k, ks, v, vs, gate, output, workspace,
        512, 65024, 65536, 24, 4, 256, 1.0F / 16.0F, true, true, 32);
    MIINFER_HIP_CHECK(hipEventRecord(end));
    MIINFER_HIP_CHECK(hipEventSynchronize(end));
    float ms = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start, end));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(end));
    return ms;
}

double run_fp16(const float* q, const __half* k, const __half* v,
                const float* gate, float* output, float* workspace) {
    hipEvent_t start{}, end{};
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&end));
    MIINFER_HIP_CHECK(hipEventRecord(start));
    miinfer::launch_qwen35_kq_fragment_reuse_attention_batch_f16(
        q, k, v, gate, output, workspace, 512, 65024, 65536,
        24, 4, 256, 1.0F / 16.0F, 32);
    MIINFER_HIP_CHECK(hipEventRecord(end));
    MIINFER_HIP_CHECK(hipEventSynchronize(end));
    float ms = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start, end));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(end));
    return ms;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

}  // namespace

int main() {
    constexpr std::uint32_t kPrefix = 65024;
    constexpr std::uint32_t kQueries = 512;
    constexpr std::uint32_t kCapacity = 65536;
    constexpr std::uint32_t kHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kDim = 256;
    constexpr std::uint32_t kSplits = 32;
    constexpr std::size_t q_count = static_cast<std::size_t>(kQueries) * kHeads * kDim;
    constexpr std::size_t kv_count = static_cast<std::size_t>(kKvHeads) * kCapacity * kDim;
    constexpr std::size_t scale_count = static_cast<std::size_t>(kKvHeads) * kCapacity;
    constexpr std::size_t workspace_count = static_cast<std::size_t>(kSplits) * kQueries * kHeads * (kDim + 2);

    miinfer::DeviceInfo device{};
    std::string error;
    if (!miinfer::validate_gfx906_device(0, device, error)) {
        std::cerr << "device validation failed: " << error << '\n';
        return 1;
    }
    char bdf[32]{};
    MIINFER_HIP_CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), 0));
    std::cout << "device_bdf=" << bdf << " architecture=" << device.architecture << '\n';

    std::mt19937 rng(20261013);
    std::normal_distribution<float> normal(0.0F, 0.5F);
    std::uniform_int_distribution<int> quantized(-100, 100);
    std::vector<float> host_q(q_count), host_gate(q_count);
    std::vector<std::int8_t> host_k(kv_count), host_v(kv_count);
    std::vector<__half> host_ks(scale_count), host_vs(scale_count);
    std::vector<__half> host_k_f16(kv_count), host_v_f16(kv_count);
    for (std::size_t i = 0; i < q_count; ++i) {
        host_q[i] = normal(rng);
        host_gate[i] = normal(rng);
    }
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
        for (std::uint32_t p = 0; p < kPrefix + kQueries; ++p) {
            const std::size_t row = (static_cast<std::size_t>(h) * kCapacity + p) * kDim;
            const std::size_t scale_index = static_cast<std::size_t>(h) * kCapacity + p;
            host_ks[scale_index] = __float2half(0.01F);
            host_vs[scale_index] = __float2half(0.01F);
            const float key_scale = __half2float(host_ks[scale_index]);
            const float value_scale = __half2float(host_vs[scale_index]);
            for (std::uint32_t d = 0; d < kDim; ++d) {
                const int key = quantized(rng);
                const int value = quantized(rng);
                host_k[row + d] = static_cast<std::int8_t>(key);
                host_v[row + d] = static_cast<std::int8_t>(value);
                host_k_f16[row + d] = __float2half(static_cast<float>(key) * key_scale);
                host_v_f16[row + d] = __float2half(static_cast<float>(value) * value_scale);
            }
        }
    }

    float* d_q = allocate_device<float>(q_count);
    float* d_gate = allocate_device<float>(q_count);
    float* d_q8 = allocate_device<float>(q_count);
    float* d_fp16 = allocate_device<float>(q_count);
    std::int8_t* d_k_q8 = allocate_device<std::int8_t>(kv_count);
    std::int8_t* d_v_q8 = allocate_device<std::int8_t>(kv_count);
    __half* d_k_scale = allocate_device<__half>(scale_count);
    __half* d_v_scale = allocate_device<__half>(scale_count);
    __half* d_k_f16 = allocate_device<__half>(kv_count);
    __half* d_v_f16 = allocate_device<__half>(kv_count);
    float* d_workspace = allocate_device<float>(workspace_count);
    MIINFER_HIP_CHECK(hipMemcpy(d_q, host_q.data(), q_count * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate, host_gate.data(), q_count * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_k_q8, host_k.data(), kv_count * sizeof(std::int8_t), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_v_q8, host_v.data(), kv_count * sizeof(std::int8_t), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_k_scale, host_ks.data(), scale_count * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_v_scale, host_vs.data(), scale_count * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_k_f16, host_k_f16.data(), kv_count * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_v_f16, host_v_f16.data(), kv_count * sizeof(__half), hipMemcpyHostToDevice));

    run_q8(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale, d_gate, d_q8, d_workspace);
    run_fp16(d_q, d_k_f16, d_v_f16, d_gate, d_fp16, d_workspace);
    std::vector<float> q8_output(q_count), fp16_output(q_count);
    MIINFER_HIP_CHECK(hipMemcpy(q8_output.data(), d_q8, q_count * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(fp16_output.data(), d_fp16, q_count * sizeof(float), hipMemcpyDeviceToHost));
    double max_abs = 0.0, dot = 0.0, q8_norm = 0.0, fp16_norm = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < q_count; ++i) {
        finite = finite && std::isfinite(q8_output[i]) && std::isfinite(fp16_output[i]);
        max_abs = std::max(max_abs, std::abs(static_cast<double>(q8_output[i]) - fp16_output[i]));
        dot += static_cast<double>(q8_output[i]) * fp16_output[i];
        q8_norm += static_cast<double>(q8_output[i]) * q8_output[i];
        fp16_norm += static_cast<double>(fp16_output[i]) * fp16_output[i];
    }
    const double cosine = dot / (std::sqrt(q8_norm * fp16_norm) + 1e-30);
    std::cout << std::fixed << std::setprecision(7)
              << "correctness_max_abs=" << max_abs << " cosine=" << cosine << '\n';
    if (!finite || !std::isfinite(max_abs) || !std::isfinite(cosine)
            || max_abs > 0.01 || cosine < 0.9999) {
        std::cerr << "FP16 tiled output diverged from direct Q8 attention\n";
        return 1;
    }

    std::vector<double> q8_ms, fp16_ms;
    for (int i = 0; i < 5; ++i) {
        if (i % 2 == 0) {
            q8_ms.push_back(run_q8(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale, d_gate, d_q8, d_workspace));
            fp16_ms.push_back(run_fp16(d_q, d_k_f16, d_v_f16, d_gate, d_fp16, d_workspace));
        } else {
            fp16_ms.push_back(run_fp16(d_q, d_k_f16, d_v_f16, d_gate, d_fp16, d_workspace));
            q8_ms.push_back(run_q8(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale, d_gate, d_q8, d_workspace));
        }
    }
    const double q8_median = median(q8_ms);
    const double fp16_median = median(fp16_ms);
    std::cout << "q8_ms=";
    for (double ms : q8_ms) std::cout << ms << ',';
    std::cout << " fp16_tiled_ms=";
    for (double ms : fp16_ms) std::cout << ms << ',';
    std::cout << "\nmedian_q8_ms=" << q8_median << " median_fp16_tiled_ms=" << fp16_median
              << " speedup=" << q8_median / fp16_median << '\n';

    MIINFER_HIP_CHECK(hipFree(d_q));
    MIINFER_HIP_CHECK(hipFree(d_gate));
    MIINFER_HIP_CHECK(hipFree(d_q8));
    MIINFER_HIP_CHECK(hipFree(d_fp16));
    MIINFER_HIP_CHECK(hipFree(d_k_q8));
    MIINFER_HIP_CHECK(hipFree(d_v_q8));
    MIINFER_HIP_CHECK(hipFree(d_k_scale));
    MIINFER_HIP_CHECK(hipFree(d_v_scale));
    MIINFER_HIP_CHECK(hipFree(d_k_f16));
    MIINFER_HIP_CHECK(hipFree(d_v_f16));
    MIINFER_HIP_CHECK(hipFree(d_workspace));
    return 0;
}
