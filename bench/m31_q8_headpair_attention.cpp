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

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double run(bool shared_headpair, const float* q, const std::int8_t* k, const __half* ks,
           const std::int8_t* v, const __half* vs, const float* gate, float* output,
           float* workspace, std::uint32_t query_count, std::uint32_t prefix,
           std::uint32_t capacity) {
    hipEvent_t start{}, end{};
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&end));
    MIINFER_HIP_CHECK(hipEventRecord(start));
    if (shared_headpair) {
        miinfer::launch_qwen35_splitk_suffix_attention_q8_headpair(
            q, k, ks, v, vs, gate, output, workspace, query_count, prefix,
            capacity, 24, 4, 256, 1.0F / 16.0F, 32);
    } else {
        miinfer::launch_qwen35_splitk_suffix_attention_quant(
            q, nullptr, nullptr, k, ks, v, vs, gate, output, workspace,
            query_count, prefix, capacity, 24, 4, 256, 1.0F / 16.0F,
            true, true, 32);
    }
    MIINFER_HIP_CHECK(hipEventRecord(end));
    MIINFER_HIP_CHECK(hipEventSynchronize(end));
    float ms = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start, end));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(end));
    return ms;
}

}  // namespace

int main() {
    constexpr std::uint32_t kPrefix = 8192;
    constexpr std::uint32_t kQueries = 512;
    constexpr std::uint32_t kCapacity = 65536;
    constexpr std::uint32_t kQueryHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kHeadDim = 256;
    constexpr std::uint32_t kSplits = 32;
    const std::size_t q_count = static_cast<std::size_t>(kQueries) * kQueryHeads * kHeadDim;
    const std::size_t kv_capacity = static_cast<std::size_t>(kKvHeads) * kCapacity * kHeadDim;
    const std::size_t scale_capacity = static_cast<std::size_t>(kKvHeads) * kCapacity;
    const std::size_t workspace_count = static_cast<std::size_t>(kSplits) * kQueries * kQueryHeads * (kHeadDim + 2);

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
    std::vector<std::int8_t> host_k(kv_capacity), host_v(kv_capacity);
    std::vector<__half> host_ks(scale_capacity), host_vs(scale_capacity);
    for (std::size_t i = 0; i < q_count; ++i) {
        host_q[i] = normal(rng);
        host_gate[i] = normal(rng);
    }
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
        for (std::uint32_t pos = 0; pos < kPrefix + kQueries; ++pos) {
            const std::size_t row = (static_cast<std::size_t>(h) * kCapacity + pos) * kHeadDim;
            const std::size_t scale_idx = static_cast<std::size_t>(h) * kCapacity + pos;
            host_ks[scale_idx] = __float2half(0.01F);
            host_vs[scale_idx] = __float2half(0.01F);
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                host_k[row + d] = static_cast<std::int8_t>(quantized(rng));
                host_v[row + d] = static_cast<std::int8_t>(quantized(rng));
            }
        }
    }

    float* d_q = allocate_device<float>(q_count);
    float* d_gate = allocate_device<float>(q_count);
    float* d_ref = allocate_device<float>(q_count);
    float* d_candidate = allocate_device<float>(q_count);
    std::int8_t* d_k = allocate_device<std::int8_t>(kv_capacity);
    std::int8_t* d_v = allocate_device<std::int8_t>(kv_capacity);
    __half* d_ks = allocate_device<__half>(scale_capacity);
    __half* d_vs = allocate_device<__half>(scale_capacity);
    float* d_workspace = allocate_device<float>(workspace_count);
    MIINFER_HIP_CHECK(hipMemcpy(d_q, host_q.data(), q_count * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate, host_gate.data(), q_count * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_k, host_k.data(), kv_capacity * sizeof(std::int8_t), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_v, host_v.data(), kv_capacity * sizeof(std::int8_t), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_ks, host_ks.data(), scale_capacity * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_vs, host_vs.data(), scale_capacity * sizeof(__half), hipMemcpyHostToDevice));

    run(false, d_q, d_k, d_ks, d_v, d_vs, d_gate, d_ref, d_workspace, kQueries, kPrefix, kCapacity);
    run(true, d_q, d_k, d_ks, d_v, d_vs, d_gate, d_candidate, d_workspace, kQueries, kPrefix, kCapacity);
    std::vector<float> ref(q_count), candidate(q_count);
    MIINFER_HIP_CHECK(hipMemcpy(ref.data(), d_ref, q_count * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(candidate.data(), d_candidate, q_count * sizeof(float), hipMemcpyDeviceToHost));
    double max_abs = 0.0;
    double sq_error = 0.0;
    double dot = 0.0;
    double ref_norm = 0.0;
    double candidate_norm = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
        const double a = ref[i];
        const double b = candidate[i];
        const double diff = std::abs(a - b);
        max_abs = std::max(max_abs, diff);
        sq_error += diff * diff;
        dot += a * b;
        ref_norm += a * a;
        candidate_norm += b * b;
    }
    const double rmse = std::sqrt(sq_error / q_count);
    const double cosine = dot / (std::sqrt(ref_norm * candidate_norm) + 1e-30);
    std::cout << std::fixed << std::setprecision(7)
              << "correctness_max_abs=" << max_abs << " rmse=" << rmse << " cosine=" << cosine << '\n';
    if (!std::isfinite(max_abs) || max_abs > 0.01 || cosine < 0.9999) {
        std::cerr << "Q8 Q=2 candidate diverged from the existing Q8/Q8 control\n";
        return 1;
    }

    std::vector<double> baseline_ms, candidate_ms;
    for (int i = 0; i < 5; ++i) {
        baseline_ms.push_back(run(false, d_q, d_k, d_ks, d_v, d_vs, d_gate, d_ref,
                                  d_workspace, kQueries, kPrefix, kCapacity));
        candidate_ms.push_back(run(true, d_q, d_k, d_ks, d_v, d_vs, d_gate, d_candidate,
                                   d_workspace, kQueries, kPrefix, kCapacity));
    }
    std::cout << "baseline_ms=";
    for (double ms : baseline_ms) std::cout << ms << ',';
    std::cout << " candidate_ms=";
    for (double ms : candidate_ms) std::cout << ms << ',';
    const double base_median = median(baseline_ms);
    const double candidate_median = median(candidate_ms);
    std::cout << "\nmedian_baseline_ms=" << base_median
              << " median_candidate_ms=" << candidate_median
              << " speedup=" << (base_median / candidate_median) << '\n';

    MIINFER_HIP_CHECK(hipFree(d_q));
    MIINFER_HIP_CHECK(hipFree(d_gate));
    MIINFER_HIP_CHECK(hipFree(d_ref));
    MIINFER_HIP_CHECK(hipFree(d_candidate));
    MIINFER_HIP_CHECK(hipFree(d_k));
    MIINFER_HIP_CHECK(hipFree(d_v));
    MIINFER_HIP_CHECK(hipFree(d_ks));
    MIINFER_HIP_CHECK(hipFree(d_vs));
    MIINFER_HIP_CHECK(hipFree(d_workspace));
    return 0;
}
