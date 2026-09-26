#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/prefill_v2/kv_cache.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

struct GpuBuffer {
    void* ptr = nullptr;
    explicit GpuBuffer(std::size_t bytes) {
        MIINFER_HIP_CHECK(hipMalloc(&ptr, bytes));
    }
    ~GpuBuffer() {
        if (ptr != nullptr) {
            hipFree(ptr);
        }
    }
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
};

void compute_metrics(
    const std::vector<float>& ref,
    const std::vector<float>& test,
    double& max_abs,
    double& mae,
    double& rmse,
    double& cosine_sim) {
    if (ref.size() != test.size() || ref.empty()) {
        max_abs = mae = rmse = 0.0;
        cosine_sim = 1.0;
        return;
    }
    max_abs = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_test = 0.0;

    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i];
        const double t = test[i];
        const double diff = std::fabs(r - t);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
        sum_sq += diff * diff;
        dot += r * t;
        norm_ref += r * r;
        norm_test += t * t;
    }
    mae = sum_abs / ref.size();
    rmse = std::sqrt(sum_sq / ref.size());
    cosine_sim = dot / (std::sqrt(norm_ref) * std::sqrt(norm_test) + 1e-12);
}

}  // namespace

int main() {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0019: Quantized KV Cache Attention Numerical Parity\n";
    std::cout << "  Testing FP16 vs Q8-K, Q8-V, and Q8-K+V Direct Attention Kernels\n";
    std::cout << "===================================================================\n\n";

    constexpr std::uint32_t kQueryHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kHeadDim = 256;
    constexpr float kScale = 1.0f / 16.0f; // 1 / sqrt(256)

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    const std::vector<std::pair<std::uint32_t, std::uint32_t>> test_cases = {
        {64, 16},
        {512, 64},
        {4096, 128},
        {16384, 512}
    };

    bool all_passed = true;

    for (const auto& [prefix_len, suffix_len] : test_cases) {
        const std::uint32_t total_tokens = prefix_len + suffix_len;
        const std::uint32_t cache_capacity = total_tokens + 128;

        std::cout << "-------------------------------------------------------------------\n";
        std::cout << " Test Case: Prefix = " << prefix_len << ", Suffix = " << suffix_len
                  << " (Total = " << total_tokens << ", Capacity = " << cache_capacity << ")\n";
        std::cout << "-------------------------------------------------------------------\n";

        // Generate synthetic Q, K, V, Gate
        const std::size_t q_count = static_cast<std::size_t>(suffix_len) * kQueryHeads * kHeadDim;
        const std::size_t gate_count = q_count;
        const std::size_t kv_count = static_cast<std::size_t>(kKvHeads) * cache_capacity * kHeadDim;
        const std::size_t scale_count = static_cast<std::size_t>(kKvHeads) * cache_capacity;

        std::vector<float> h_q(q_count);
        std::vector<float> h_gate(gate_count);
        std::vector<__half> h_key_f16(kv_count);
        std::vector<__half> h_val_f16(kv_count);
        std::vector<int8_t> h_key_q8(kv_count);
        std::vector<__half> h_key_scales(scale_count);
        std::vector<int8_t> h_val_q8(kv_count);
        std::vector<__half> h_val_scales(scale_count);

        for (auto& v : h_q) v = dist(rng);
        for (auto& v : h_gate) v = dist(rng);

        // Populate K, V with realistic values and quantize to Q8
        for (std::uint32_t h = 0; h < kKvHeads; ++h) {
            for (std::uint32_t pos = 0; pos < total_tokens; ++pos) {
                const std::size_t base = (static_cast<std::size_t>(h) * cache_capacity + pos) * kHeadDim;
                const std::size_t sidx = static_cast<std::size_t>(h) * cache_capacity + pos;

                // K vector
                float max_k = 0.0f;
                std::vector<float> k_vec(kHeadDim);
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    k_vec[d] = dist(rng);
                    max_k = std::max(max_k, std::fabs(k_vec[d]));
                }
                const float ks = max_k / 127.0f;
                const float inv_ks = ks > 0.0f ? 1.0f / ks : 0.0f;
                h_key_scales[sidx] = __float2half(ks);
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    h_key_f16[base + d] = __float2half(k_vec[d]);
                    h_key_q8[base + d] = static_cast<int8_t>(std::clamp(std::round(k_vec[d] * inv_ks), -127.0f, 127.0f));
                }

                // V vector
                float max_v = 0.0f;
                std::vector<float> v_vec(kHeadDim);
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    v_vec[d] = dist(rng);
                    max_v = std::max(max_v, std::fabs(v_vec[d]));
                }
                const float vs = max_v / 127.0f;
                const float inv_vs = vs > 0.0f ? 1.0f / vs : 0.0f;
                h_val_scales[sidx] = __float2half(vs);
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    h_val_f16[base + d] = __float2half(v_vec[d]);
                    h_val_q8[base + d] = static_cast<int8_t>(std::clamp(std::round(v_vec[d] * inv_vs), -127.0f, 127.0f));
                }
            }
        }

        // Allocate GPU buffers
        GpuBuffer d_q(q_count * sizeof(float));
        GpuBuffer d_gate(gate_count * sizeof(float));
        GpuBuffer d_out_ref(q_count * sizeof(float));
        GpuBuffer d_out_q8k(q_count * sizeof(float));
        GpuBuffer d_out_q8v(q_count * sizeof(float));
        GpuBuffer d_out_q8kv(q_count * sizeof(float));

        const std::size_t split_ws_bytes = static_cast<std::size_t>(64) * suffix_len * kQueryHeads * (2 + kHeadDim) * sizeof(float);
        GpuBuffer d_split_ws(split_ws_bytes);

        GpuBuffer d_key_f16(kv_count * sizeof(__half));
        GpuBuffer d_val_f16(kv_count * sizeof(__half));
        GpuBuffer d_key_q8(kv_count * sizeof(int8_t));
        GpuBuffer d_key_scales(scale_count * sizeof(__half));
        GpuBuffer d_val_q8(kv_count * sizeof(int8_t));
        GpuBuffer d_val_scales(scale_count * sizeof(__half));

        // Copy inputs to device
        MIINFER_HIP_CHECK(hipMemcpy(d_q.ptr, h_q.data(), q_count * sizeof(float), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_gate.ptr, h_gate.data(), gate_count * sizeof(float), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_key_f16.ptr, h_key_f16.data(), kv_count * sizeof(__half), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_val_f16.ptr, h_val_f16.data(), kv_count * sizeof(__half), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_key_q8.ptr, h_key_q8.data(), kv_count * sizeof(int8_t), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_key_scales.ptr, h_key_scales.data(), scale_count * sizeof(__half), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_val_q8.ptr, h_val_q8.data(), kv_count * sizeof(int8_t), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_val_scales.ptr, h_val_scales.data(), scale_count * sizeof(__half), hipMemcpyHostToDevice));

        // Run Candidate A (FP16 / FP16 Reference)
        miinfer::launch_qwen35_splitk_suffix_attention_quant(
            static_cast<const float*>(d_q.ptr),
            static_cast<const __half*>(d_key_f16.ptr),
            static_cast<const __half*>(d_val_f16.ptr),
            static_cast<const int8_t*>(d_key_q8.ptr),
            static_cast<const __half*>(d_key_scales.ptr),
            static_cast<const int8_t*>(d_val_q8.ptr),
            static_cast<const __half*>(d_val_scales.ptr),
            static_cast<const float*>(d_gate.ptr),
            static_cast<float*>(d_out_ref.ptr),
            static_cast<float*>(d_split_ws.ptr),
            suffix_len, prefix_len, cache_capacity,
            kQueryHeads, kKvHeads, kHeadDim, kScale,
            /*is_k_q8=*/false, /*is_v_q8=*/false, 32);

        // Run Candidate B (Q8 K + FP16 V)
        miinfer::launch_qwen35_splitk_suffix_attention_quant(
            static_cast<const float*>(d_q.ptr),
            static_cast<const __half*>(d_key_f16.ptr),
            static_cast<const __half*>(d_val_f16.ptr),
            static_cast<const int8_t*>(d_key_q8.ptr),
            static_cast<const __half*>(d_key_scales.ptr),
            static_cast<const int8_t*>(d_val_q8.ptr),
            static_cast<const __half*>(d_val_scales.ptr),
            static_cast<const float*>(d_gate.ptr),
            static_cast<float*>(d_out_q8k.ptr),
            static_cast<float*>(d_split_ws.ptr),
            suffix_len, prefix_len, cache_capacity,
            kQueryHeads, kKvHeads, kHeadDim, kScale,
            /*is_k_q8=*/true, /*is_v_q8=*/false, 32);

        // Run Candidate C (FP16 K + Q8 V)
        miinfer::launch_qwen35_splitk_suffix_attention_quant(
            static_cast<const float*>(d_q.ptr),
            static_cast<const __half*>(d_key_f16.ptr),
            static_cast<const __half*>(d_val_f16.ptr),
            static_cast<const int8_t*>(d_key_q8.ptr),
            static_cast<const __half*>(d_key_scales.ptr),
            static_cast<const int8_t*>(d_val_q8.ptr),
            static_cast<const __half*>(d_val_scales.ptr),
            static_cast<const float*>(d_gate.ptr),
            static_cast<float*>(d_out_q8v.ptr),
            static_cast<float*>(d_split_ws.ptr),
            suffix_len, prefix_len, cache_capacity,
            kQueryHeads, kKvHeads, kHeadDim, kScale,
            /*is_k_q8=*/false, /*is_v_q8=*/true, 32);

        // Run Candidate D (Q8 K + Q8 V)
        miinfer::launch_qwen35_splitk_suffix_attention_quant(
            static_cast<const float*>(d_q.ptr),
            static_cast<const __half*>(d_key_f16.ptr),
            static_cast<const __half*>(d_val_f16.ptr),
            static_cast<const int8_t*>(d_key_q8.ptr),
            static_cast<const __half*>(d_key_scales.ptr),
            static_cast<const int8_t*>(d_val_q8.ptr),
            static_cast<const __half*>(d_val_scales.ptr),
            static_cast<const float*>(d_gate.ptr),
            static_cast<float*>(d_out_q8kv.ptr),
            static_cast<float*>(d_split_ws.ptr),
            suffix_len, prefix_len, cache_capacity,
            kQueryHeads, kKvHeads, kHeadDim, kScale,
            /*is_k_q8=*/true, /*is_v_q8=*/true, 32);

        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> h_out_ref(q_count);
        std::vector<float> h_out_q8k(q_count);
        std::vector<float> h_out_q8v(q_count);
        std::vector<float> h_out_q8kv(q_count);

        MIINFER_HIP_CHECK(hipMemcpy(h_out_ref.data(), d_out_ref.ptr, q_count * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(h_out_q8k.data(), d_out_q8k.ptr, q_count * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(h_out_q8v.data(), d_out_q8v.ptr, q_count * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(h_out_q8kv.data(), d_out_q8kv.ptr, q_count * sizeof(float), hipMemcpyDeviceToHost));

        double max_abs_b = 0, mae_b = 0, rmse_b = 0, cos_b = 0;
        double max_abs_c = 0, mae_c = 0, rmse_c = 0, cos_c = 0;
        double max_abs_d = 0, mae_d = 0, rmse_d = 0, cos_d = 0;

        compute_metrics(h_out_ref, h_out_q8k, max_abs_b, mae_b, rmse_b, cos_b);
        compute_metrics(h_out_ref, h_out_q8v, max_abs_c, mae_c, rmse_c, cos_c);
        compute_metrics(h_out_ref, h_out_q8kv, max_abs_d, mae_d, rmse_d, cos_d);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Candidate B (Q8-K / FP16-V): MaxAbs=" << max_abs_b << ", MAE=" << mae_b
                  << ", RMSE=" << rmse_b << ", CosSim=" << std::setprecision(8) << cos_b << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Candidate C (FP16-K / Q8-V): MaxAbs=" << max_abs_c << ", MAE=" << mae_c
                  << ", RMSE=" << rmse_c << ", CosSim=" << std::setprecision(8) << cos_c << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Candidate D (Q8-K  / Q8-V) : MaxAbs=" << max_abs_d << ", MAE=" << mae_d
                  << ", RMSE=" << rmse_d << ", CosSim=" << std::setprecision(8) << cos_d << "\n";

        if (cos_b < 0.999 || cos_c < 0.999 || cos_d < 0.999 || mae_d > 0.05) {
            std::cerr << "  [FAIL] Accuracy threshold exceeded!\n";
            all_passed = false;
        } else {
            std::cout << "  [PASS] Accuracy metrics well within qualification thresholds.\n";
        }
    }

    if (all_passed) {
        std::cout << "\n>>> ALL QUANTIZED KV ATTENTION NUMERICAL TESTS PASSED <<<\n";
        return 0;
    } else {
        std::cerr << "\n>>> NUMERICAL TESTS FAILED <<<\n";
        return 1;
    }
}
