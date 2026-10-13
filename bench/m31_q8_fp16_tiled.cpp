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
              float* output, float* workspace, std::uint32_t queries,
              std::uint32_t prefix, std::uint32_t capacity) {
    hipEvent_t start{}, end{};
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&end));
    MIINFER_HIP_CHECK(hipEventRecord(start));
    miinfer::launch_qwen35_splitk_suffix_attention_quant(
        q, nullptr, nullptr, k, ks, v, vs, gate, output, workspace,
        queries, prefix, capacity, 24, 4, 256, 1.0F / 16.0F, true, true, 32);
    MIINFER_HIP_CHECK(hipEventRecord(end));
    MIINFER_HIP_CHECK(hipEventSynchronize(end));
    float ms = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start, end));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(end));
    return ms;
}

double run_fp16(const float* q, const __half* k, const __half* v,
                const float* gate, float* output, float* workspace,
                std::uint32_t queries, std::uint32_t prefix,
                std::uint32_t capacity) {
    hipEvent_t start{}, end{};
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&end));
    MIINFER_HIP_CHECK(hipEventRecord(start));
    miinfer::launch_qwen35_kq_fragment_reuse_attention_batch_f16(
        q, k, v, gate, output, workspace, queries, prefix, capacity,
        24, 4, 256, 1.0F / 16.0F, 32);
    MIINFER_HIP_CHECK(hipEventRecord(end));
    MIINFER_HIP_CHECK(hipEventSynchronize(end));
    float ms = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start, end));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(end));
    return ms;
}

double run_q8_tiled(const float* q, const std::int8_t* k, const __half* ks,
                    const std::int8_t* v, const __half* vs, const float* gate,
                    float* output, float* workspace, std::uint32_t queries,
                    std::uint32_t prefix, std::uint32_t capacity) {
    hipEvent_t start{}, end{};
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&end));
    MIINFER_HIP_CHECK(hipEventRecord(start));
    miinfer::launch_qwen35_kq_fragment_reuse_attention_batch_q8(
        q, k, ks, v, vs, gate, output, workspace,
        queries, prefix, capacity, 24, 4, 256, 1.0F / 16.0F, 32);
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

int main(int argc, char** argv) {
    constexpr std::uint32_t kCapacity = 65536;
    constexpr std::uint32_t kHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kDim = 256;
    constexpr std::uint32_t kSplits = 32;
    const std::uint32_t kQueries = argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 512;
    if (kQueries < 16 || kQueries > 2048 || kQueries % 16 != 0) {
        std::cerr << "queries must be a multiple of 16 in [16, 2048]\n";
        return 2;
    }
    const std::uint32_t kPrefix = kCapacity - kQueries;
    const std::size_t q_count = static_cast<std::size_t>(kQueries) * kHeads * kDim;
    constexpr std::size_t kv_count = static_cast<std::size_t>(kKvHeads) * kCapacity * kDim;
    constexpr std::size_t scale_count = static_cast<std::size_t>(kKvHeads) * kCapacity;
    const std::size_t workspace_count = static_cast<std::size_t>(kSplits) * kQueries * kHeads * (kDim + 2);

    miinfer::DeviceInfo device{};
    std::string error;
    if (!miinfer::validate_gfx906_device(0, device, error)) {
        std::cerr << "device validation failed: " << error << '\n';
        return 1;
    }
    char bdf[32]{};
    MIINFER_HIP_CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), 0));
    std::cout << "device_bdf=" << bdf << " architecture=" << device.architecture
              << " prefix=" << kPrefix << " queries=" << kQueries
              << " capacity=" << kCapacity << '\n';

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
        for (std::uint32_t p = 0; p < kCapacity; ++p) {
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
    float* d_q8_split = allocate_device<float>(q_count);
    float* d_q8_tiled = allocate_device<float>(q_count);
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

    run_q8(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale, d_gate, d_q8, d_workspace,
           kQueries, kPrefix, kCapacity);
    if (kQueries % 2 == 0) {
        const std::uint32_t half_queries = kQueries / 2;
        const std::size_t half_offset = static_cast<std::size_t>(half_queries) * kHeads * kDim;
        run_q8(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale, d_gate, d_q8_split, d_workspace,
               half_queries, kPrefix, kCapacity);
        run_q8(d_q + half_offset, d_k_q8, d_k_scale, d_v_q8, d_v_scale,
               d_gate + half_offset, d_q8_split + half_offset, d_workspace,
               half_queries, kPrefix + half_queries, kCapacity);
    }
    run_q8_tiled(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale, d_gate, d_q8_tiled,
                 d_workspace, kQueries, kPrefix, kCapacity);
    run_fp16(d_q, d_k_f16, d_v_f16, d_gate, d_fp16, d_workspace,
             kQueries, kPrefix, kCapacity);
    std::vector<float> q8_output(q_count), q8_split_output(q_count);
    std::vector<float> q8_tiled_output(q_count), fp16_output(q_count);
    MIINFER_HIP_CHECK(hipMemcpy(q8_output.data(), d_q8, q_count * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(q8_split_output.data(), d_q8_split, q_count * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(q8_tiled_output.data(), d_q8_tiled, q_count * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(fp16_output.data(), d_fp16, q_count * sizeof(float), hipMemcpyDeviceToHost));
    double max_abs = 0.0, dot = 0.0, q8_norm = 0.0, fp16_norm = 0.0;
    double split_max_abs = 0.0, split_dot = 0.0, split_norm = 0.0;
    double tiled_max_abs = 0.0, tiled_dot = 0.0, tiled_norm = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < q_count; ++i) {
        finite = finite && std::isfinite(q8_output[i]) && std::isfinite(q8_tiled_output[i])
            && std::isfinite(fp16_output[i]) && std::isfinite(q8_split_output[i]);
        max_abs = std::max(max_abs, std::abs(static_cast<double>(q8_output[i]) - fp16_output[i]));
        dot += static_cast<double>(q8_output[i]) * fp16_output[i];
        q8_norm += static_cast<double>(q8_output[i]) * q8_output[i];
        fp16_norm += static_cast<double>(fp16_output[i]) * fp16_output[i];
        tiled_max_abs = std::max(tiled_max_abs,
            std::abs(static_cast<double>(q8_output[i]) - q8_tiled_output[i]));
        tiled_dot += static_cast<double>(q8_output[i]) * q8_tiled_output[i];
        tiled_norm += static_cast<double>(q8_tiled_output[i]) * q8_tiled_output[i];
        split_max_abs = std::max(split_max_abs,
            std::abs(static_cast<double>(q8_output[i]) - q8_split_output[i]));
        split_dot += static_cast<double>(q8_output[i]) * q8_split_output[i];
        split_norm += static_cast<double>(q8_split_output[i]) * q8_split_output[i];
    }
    const double cosine = dot / (std::sqrt(q8_norm * fp16_norm) + 1e-30);
    const double tiled_cosine = tiled_dot / (std::sqrt(q8_norm * tiled_norm) + 1e-30);
    const double split_cosine = split_dot / (std::sqrt(q8_norm * split_norm) + 1e-30);
    std::cout << std::fixed << std::setprecision(7)
              << "fp16_correctness_max_abs=" << max_abs << " cosine=" << cosine << '\n'
              << "q8_split512_correctness_max_abs=" << split_max_abs
              << " cosine=" << split_cosine << '\n'
              << "q8_tiled_correctness_max_abs=" << tiled_max_abs
              << " cosine=" << tiled_cosine << '\n';
    if (!finite || !std::isfinite(max_abs) || !std::isfinite(cosine)
            || !std::isfinite(tiled_max_abs) || !std::isfinite(tiled_cosine)
            || !std::isfinite(split_max_abs) || !std::isfinite(split_cosine)
            || max_abs > 0.01 || cosine < 0.9999
            || split_max_abs > 0.01 || split_cosine < 0.9999
            || tiled_max_abs > 0.01 || tiled_cosine < 0.9999) {
        std::cerr << "tiled output diverged from direct Q8 attention\n";
        return 1;
    }

    std::vector<double> q8_ms, q8_tiled_ms;
    for (int i = 0; i < 5; ++i) {
        if (i % 2 == 0) {
            q8_ms.push_back(run_q8(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale, d_gate, d_q8,
                                   d_workspace, kQueries, kPrefix, kCapacity));
            q8_tiled_ms.push_back(run_q8_tiled(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale,
                                               d_gate, d_q8_tiled, d_workspace,
                                               kQueries, kPrefix, kCapacity));
        } else {
            q8_tiled_ms.push_back(run_q8_tiled(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale,
                                               d_gate, d_q8_tiled, d_workspace,
                                               kQueries, kPrefix, kCapacity));
            q8_ms.push_back(run_q8(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale, d_gate, d_q8,
                                   d_workspace, kQueries, kPrefix, kCapacity));
        }
    }
    const double q8_median = median(q8_ms);
    const double q8_tiled_median = median(q8_tiled_ms);
    std::cout << "q8_ms=";
    for (double ms : q8_ms) std::cout << ms << ',';
    std::cout << " q8_query_tiled_ms=";
    for (double ms : q8_tiled_ms) std::cout << ms << ',';
    std::cout << "\nmedian_q8_ms=" << q8_median << " median_q8_query_tiled_ms=" << q8_tiled_median
              << " speedup=" << q8_median / q8_tiled_median << '\n';

    MIINFER_HIP_CHECK(hipFree(d_q));
    MIINFER_HIP_CHECK(hipFree(d_gate));
    MIINFER_HIP_CHECK(hipFree(d_q8));
    MIINFER_HIP_CHECK(hipFree(d_q8_split));
    MIINFER_HIP_CHECK(hipFree(d_q8_tiled));
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
