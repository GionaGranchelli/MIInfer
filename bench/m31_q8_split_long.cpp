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

double run(std::uint32_t splits, std::uint32_t prefix, std::uint32_t capacity,
           const float* q, const std::int8_t* k,
           const __half* ks, const std::int8_t* v, const __half* vs,
           const float* gate, float* output, float* workspace) {
    hipEvent_t start{}, end{};
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&end));
    MIINFER_HIP_CHECK(hipEventRecord(start));
    miinfer::launch_qwen35_splitk_suffix_attention_quant(
        q, nullptr, nullptr, k, ks, v, vs, gate, output, workspace,
        512, prefix, capacity, 24, 4, 256, 1.0F / 16.0F, true, true, splits);
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
    constexpr std::uint32_t kReferenceSplits = 32;
    constexpr std::uint32_t kCandidateSplits = 64;
    constexpr std::size_t q_count = static_cast<std::size_t>(kQueries) * kHeads * kDim;
    constexpr std::size_t kv_count = static_cast<std::size_t>(kKvHeads) * kCapacity * kDim;
    constexpr std::size_t scale_count = static_cast<std::size_t>(kKvHeads) * kCapacity;
    constexpr std::size_t workspace_count = static_cast<std::size_t>(kReferenceSplits) * kQueries * kHeads * (kDim + 2);

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
            for (std::uint32_t d = 0; d < kDim; ++d) {
                host_k[row + d] = static_cast<std::int8_t>(quantized(rng));
                host_v[row + d] = static_cast<std::int8_t>(quantized(rng));
            }
        }
    }

    float* d_q = allocate_device<float>(q_count);
    float* d_gate = allocate_device<float>(q_count);
    float* d_ref = allocate_device<float>(q_count);
    float* d_candidate = allocate_device<float>(q_count);
    std::int8_t* d_k = allocate_device<std::int8_t>(kv_count);
    std::int8_t* d_v = allocate_device<std::int8_t>(kv_count);
    __half* d_ks = allocate_device<__half>(scale_count);
    __half* d_vs = allocate_device<__half>(scale_count);
    float* d_workspace = allocate_device<float>(workspace_count);
    MIINFER_HIP_CHECK(hipMemcpy(d_q, host_q.data(), q_count * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate, host_gate.data(), q_count * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_k, host_k.data(), kv_count * sizeof(std::int8_t), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_v, host_v.data(), kv_count * sizeof(std::int8_t), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_ks, host_ks.data(), scale_count * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_vs, host_vs.data(), scale_count * sizeof(__half), hipMemcpyHostToDevice));

    run(kReferenceSplits, kPrefix, kCapacity, d_q, d_k, d_ks, d_v, d_vs, d_gate, d_ref, d_workspace);
    run(kCandidateSplits, kPrefix, kCapacity, d_q, d_k, d_ks, d_v, d_vs, d_gate, d_candidate, d_workspace);
    std::vector<float> ref(q_count), candidate(q_count);
    MIINFER_HIP_CHECK(hipMemcpy(ref.data(), d_ref, q_count * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(candidate.data(), d_candidate, q_count * sizeof(float), hipMemcpyDeviceToHost));
    double max_abs = 0.0, dot = 0.0, ref_norm = 0.0, candidate_norm = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < q_count; ++i) {
        finite = finite && std::isfinite(ref[i]) && std::isfinite(candidate[i]);
        max_abs = std::max(max_abs, std::abs(static_cast<double>(ref[i]) - candidate[i]));
        dot += static_cast<double>(ref[i]) * candidate[i];
        ref_norm += static_cast<double>(ref[i]) * ref[i];
        candidate_norm += static_cast<double>(candidate[i]) * candidate[i];
    }
    const double cosine = dot / (std::sqrt(ref_norm * candidate_norm) + 1e-30);
    std::cout << std::fixed << std::setprecision(7)
              << "correctness_max_abs=" << max_abs << " cosine=" << cosine << '\n';
    if (!finite || !std::isfinite(max_abs) || !std::isfinite(cosine)
            || max_abs > 0.01 || cosine < 0.9999) {
        std::cerr << "3-split Q8/Q8 output diverged from 32-split control\n";
        return 1;
    }

    std::vector<double> reference_ms, candidate_ms;
    for (int i = 0; i < 5; ++i) {
        if (i % 2 == 0) {
            reference_ms.push_back(run(kReferenceSplits, kPrefix, kCapacity, d_q, d_k, d_ks, d_v, d_vs, d_gate, d_ref, d_workspace));
            candidate_ms.push_back(run(kCandidateSplits, kPrefix, kCapacity, d_q, d_k, d_ks, d_v, d_vs, d_gate, d_candidate, d_workspace));
        } else {
            candidate_ms.push_back(run(kCandidateSplits, kPrefix, kCapacity, d_q, d_k, d_ks, d_v, d_vs, d_gate, d_candidate, d_workspace));
            reference_ms.push_back(run(kReferenceSplits, kPrefix, kCapacity, d_q, d_k, d_ks, d_v, d_vs, d_gate, d_ref, d_workspace));
        }
    }
    std::cout << "prefix=" << kPrefix << " query_count=" << kQueries << " split32_ms=";
    for (double ms : reference_ms) std::cout << ms << ',';
    std::cout << " split64_ms=";
    for (double ms : candidate_ms) std::cout << ms << ',';
    const double ref_median = median(reference_ms);
    const double candidate_median = median(candidate_ms);
    std::cout << "\nmedian_split32_ms=" << ref_median << " median_split3_ms=" << candidate_median
              << " speedup=" << (ref_median / candidate_median) << '\n';

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
