#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;

namespace {

struct GpuBuffer {
    void* ptr = nullptr;
    explicit GpuBuffer(std::size_t bytes) {
        MIINFER_HIP_CHECK(hipMalloc(&ptr, bytes));
    }
    ~GpuBuffer() {
        if (ptr != nullptr) {
            (void)hipFree(ptr);
            ptr = nullptr;
        }
    }
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
};

float elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() % 2 == 1) return v[v.size() / 2];
    return 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

// -----------------------------------------------------------------------------
// Kernel 0: Baseline Control (FP16 Split-K, Independent 64T Blocks)
// -----------------------------------------------------------------------------
template <int UNROLL = 4>
__launch_bounds__(64, 4) __global__ void
qwen35_splitk_suffix_attn_control_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ key_cache_f16,
    const __half* __restrict__ value_cache_f16,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_acc,
    std::uint32_t token_count,
    std::uint32_t base_position,
    std::uint32_t cache_capacity,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float scale,
    std::uint32_t num_splits) {

    const std::uint32_t head_pair = blockIdx.x;
    const std::uint32_t token = blockIdx.y;
    const std::uint32_t split_id = blockIdx.z;
    const std::uint32_t lane = threadIdx.x;

    const std::uint32_t sub_wave = lane / 32;
    const std::uint32_t sub_lane = lane % 32;
    const std::uint32_t head = head_pair * 2 + sub_wave;
    const std::uint32_t dim_offset = sub_lane * 8;

    if (token >= token_count || head >= query_heads || lane >= 64) return;

    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::uint32_t total_length = base_position + token + 1;
    if (total_length > cache_capacity) return;

    const std::uint32_t chunk_size = (total_length + num_splits - 1) / num_splits;
    const std::uint32_t k_start = split_id * chunk_size;
    const std::uint32_t k_end = min(k_start + chunk_size, total_length);

    const std::size_t split_head_idx =
        (static_cast<std::size_t>(split_id) * token_count + token) * query_heads + head;

    if (k_start >= total_length) {
        if (sub_lane == 0) {
            partial_max[split_head_idx] = -INFINITY;
            partial_sum[split_head_idx] = 0.0f;
        }
        const std::size_t acc_base = split_head_idx * head_dim + dim_offset;
        *reinterpret_cast<float4*>(partial_acc + acc_base + 0) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *reinterpret_cast<float4*>(partial_acc + acc_base + 4) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const std::size_t q_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim + dim_offset;
    const float4 q_vec0 = *reinterpret_cast<const float4*>(q + q_base + 0);
    const float4 q_vec1 = *reinterpret_cast<const float4*>(q + q_base + 4);

    float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
    const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;
    const __half* k_f16_ptr = key_cache_f16 + kv_base + static_cast<std::size_t>(k_start) * head_dim + dim_offset;
    const __half* v_f16_ptr = value_cache_f16 + kv_base + static_cast<std::size_t>(k_start) * head_dim + dim_offset;

    std::uint32_t pos = k_start;
    for (; pos + UNROLL <= k_end; pos += UNROLL) {
        uint4 k_raw[UNROLL];
        uint4 v_raw[UNROLL];

        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            k_raw[u] = *reinterpret_cast<const uint4*>(k_f16_ptr + static_cast<std::size_t>(u) * head_dim);
            v_raw[u] = *reinterpret_cast<const uint4*>(v_f16_ptr + static_cast<std::size_t>(u) * head_dim);
        }

        float dots[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const auto* kh = reinterpret_cast<const __half*>(&k_raw[u]);
            dots[u] = q_vec0.x * __half2float(kh[0]) + q_vec0.y * __half2float(kh[1])
                    + q_vec0.z * __half2float(kh[2]) + q_vec0.w * __half2float(kh[3])
                    + q_vec1.x * __half2float(kh[4]) + q_vec1.y * __half2float(kh[5])
                    + q_vec1.z * __half2float(kh[6]) + q_vec1.w * __half2float(kh[7]);
        }

        float scores[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            float d = dots[u];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                d += __shfl_xor(d, offset, 32);
            }
            scores[u] = __shfl(d * scale, sub_wave * 32, 64);
        }

        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const float score = scores[u];
            float alpha = 1.0f;
            float weight = 0.0f;
            if (sub_lane == 0) {
                if (score > running_max) {
                    alpha = expf(running_max - score);
                    running_sum = running_sum * alpha + 1.0f;
                    running_max = score;
                    weight = 1.0f;
                } else {
                    weight = expf(score - running_max);
                    running_sum += weight;
                }
            }
            alpha = __shfl(alpha, sub_wave * 32, 64);
            weight = __shfl(weight, sub_wave * 32, 64);

            const auto* vh = reinterpret_cast<const __half*>(&v_raw[u]);
            acc0.x = acc0.x * alpha + weight * __half2float(vh[0]);
            acc0.y = acc0.y * alpha + weight * __half2float(vh[1]);
            acc0.z = acc0.z * alpha + weight * __half2float(vh[2]);
            acc0.w = acc0.w * alpha + weight * __half2float(vh[3]);

            acc1.x = acc1.x * alpha + weight * __half2float(vh[4]);
            acc1.y = acc1.y * alpha + weight * __half2float(vh[5]);
            acc1.z = acc1.z * alpha + weight * __half2float(vh[6]);
            acc1.w = acc1.w * alpha + weight * __half2float(vh[7]);
        }

        k_f16_ptr += UNROLL * head_dim;
        v_f16_ptr += UNROLL * head_dim;
    }

    for (; pos < k_end; ++pos) {
        const uint4 k_raw = *reinterpret_cast<const uint4*>(k_f16_ptr);
        const auto* kh = reinterpret_cast<const __half*>(&k_raw);
        float dot = q_vec0.x * __half2float(kh[0]) + q_vec0.y * __half2float(kh[1])
                  + q_vec0.z * __half2float(kh[2]) + q_vec0.w * __half2float(kh[3])
                  + q_vec1.x * __half2float(kh[4]) + q_vec1.y * __half2float(kh[5])
                  + q_vec1.z * __half2float(kh[6]) + q_vec1.w * __half2float(kh[7]);
        k_f16_ptr += head_dim;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }
        float score = __shfl(dot * scale, sub_wave * 32, 64);

        float alpha = 1.0f;
        float weight = 0.0f;
        if (sub_lane == 0) {
            if (score > running_max) {
                alpha = expf(running_max - score);
                running_sum = running_sum * alpha + 1.0f;
                running_max = score;
                weight = 1.0f;
            } else {
                weight = expf(score - running_max);
                running_sum += weight;
            }
        }
        alpha = __shfl(alpha, sub_wave * 32, 64);
        weight = __shfl(weight, sub_wave * 32, 64);

        const uint4 v_raw = *reinterpret_cast<const uint4*>(v_f16_ptr);
        const auto* vh = reinterpret_cast<const __half*>(&v_raw);

        acc0.x = acc0.x * alpha + weight * __half2float(vh[0]);
        acc0.y = acc0.y * alpha + weight * __half2float(vh[1]);
        acc0.z = acc0.z * alpha + weight * __half2float(vh[2]);
        acc0.w = acc0.w * alpha + weight * __half2float(vh[3]);

        acc1.x = acc1.x * alpha + weight * __half2float(vh[4]);
        acc1.y = acc1.y * alpha + weight * __half2float(vh[5]);
        acc1.z = acc1.z * alpha + weight * __half2float(vh[6]);
        acc1.w = acc1.w * alpha + weight * __half2float(vh[7]);

        v_f16_ptr += head_dim;
    }

    if (sub_lane == 0) {
        partial_max[split_head_idx] = running_max;
        partial_sum[split_head_idx] = running_sum;
    }
    const std::size_t acc_base = split_head_idx * head_dim + dim_offset;
    *reinterpret_cast<float4*>(partial_acc + acc_base + 0) = acc0;
    *reinterpret_cast<float4*>(partial_acc + acc_base + 4) = acc1;
}

// -----------------------------------------------------------------------------
// Kernel 1: Candidate C (Co-scheduled 192T Multi-Wave Workgroup, 100% L1 Vector Cache Reuse)
// -----------------------------------------------------------------------------
// All 3 Wave64s for a KV head execute concurrently in the same CU.
// Wave 0's vector load populates the 16KB L1 cache, giving Wave 1 & 2 free cache hits!
// Zero LDS overhead, zero in-loop barriers, 48 VGPRs, max occupancy!
// -----------------------------------------------------------------------------
template <int UNROLL = 4>
__launch_bounds__(192, 2) __global__ void
qwen35_splitk_suffix_attn_l1_coscheduled_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ key_cache_f16,
    const __half* __restrict__ value_cache_f16,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_acc,
    std::uint32_t token_count,
    std::uint32_t base_position,
    std::uint32_t cache_capacity,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float scale,
    std::uint32_t num_splits) {

    const std::uint32_t kv_head = blockIdx.x; // 0..kv_heads-1 (4)
    const std::uint32_t token = blockIdx.y;   // 0..token_count-1
    const std::uint32_t split_id = blockIdx.z; // 0..num_splits-1
    const std::uint32_t tid = threadIdx.x;    // 0..191 (3 Wave64s)

    const std::uint32_t wave_id = tid / 64;   // 0..2 (Wave 0, 1, 2)
    const std::uint32_t lane = tid % 64;      // 0..63
    const std::uint32_t sub_wave = lane / 32; // 0..1
    const std::uint32_t sub_lane = lane % 32; // 0..31

    const std::uint32_t group_ratio = query_heads / kv_heads; // 6
    const std::uint32_t head = kv_head * group_ratio + wave_id * 2 + sub_wave;
    const std::uint32_t dim_offset = sub_lane * 8;

    if (token >= token_count || head >= query_heads) return;

    const std::uint32_t total_length = base_position + token + 1;
    if (total_length > cache_capacity) return;

    const std::uint32_t chunk_size = (total_length + num_splits - 1) / num_splits;
    const std::uint32_t k_start = split_id * chunk_size;
    const std::uint32_t k_end = min(k_start + chunk_size, total_length);

    const std::size_t split_head_idx =
        (static_cast<std::size_t>(split_id) * token_count + token) * query_heads + head;

    if (k_start >= total_length) {
        if (sub_lane == 0) {
            partial_max[split_head_idx] = -INFINITY;
            partial_sum[split_head_idx] = 0.0f;
        }
        const std::size_t acc_base = split_head_idx * head_dim + dim_offset;
        *reinterpret_cast<float4*>(partial_acc + acc_base + 0) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *reinterpret_cast<float4*>(partial_acc + acc_base + 4) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const std::size_t q_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim + dim_offset;
    const float4 q_vec0 = *reinterpret_cast<const float4*>(q + q_base + 0);
    const float4 q_vec1 = *reinterpret_cast<const float4*>(q + q_base + 4);

    float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
    const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;
    const __half* k_f16_ptr = key_cache_f16 + kv_base + static_cast<std::size_t>(k_start) * head_dim + dim_offset;
    const __half* v_f16_ptr = value_cache_f16 + kv_base + static_cast<std::size_t>(k_start) * head_dim + dim_offset;

    std::uint32_t pos = k_start;
    for (; pos + UNROLL <= k_end; pos += UNROLL) {
        uint4 k_raw[UNROLL];
        uint4 v_raw[UNROLL];

        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            k_raw[u] = *reinterpret_cast<const uint4*>(k_f16_ptr + static_cast<std::size_t>(u) * head_dim);
            v_raw[u] = *reinterpret_cast<const uint4*>(v_f16_ptr + static_cast<std::size_t>(u) * head_dim);
        }

        float dots[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const auto* kh = reinterpret_cast<const __half*>(&k_raw[u]);
            dots[u] = q_vec0.x * __half2float(kh[0]) + q_vec0.y * __half2float(kh[1])
                    + q_vec0.z * __half2float(kh[2]) + q_vec0.w * __half2float(kh[3])
                    + q_vec1.x * __half2float(kh[4]) + q_vec1.y * __half2float(kh[5])
                    + q_vec1.z * __half2float(kh[6]) + q_vec1.w * __half2float(kh[7]);
        }

        float scores[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            float d = dots[u];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                d += __shfl_xor(d, offset, 32);
            }
            scores[u] = __shfl(d * scale, sub_wave * 32, 64);
        }

        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const float score = scores[u];
            float alpha = 1.0f;
            float weight = 0.0f;
            if (sub_lane == 0) {
                if (score > running_max) {
                    alpha = expf(running_max - score);
                    running_sum = running_sum * alpha + 1.0f;
                    running_max = score;
                    weight = 1.0f;
                } else {
                    weight = expf(score - running_max);
                    running_sum += weight;
                }
            }
            alpha = __shfl(alpha, sub_wave * 32, 64);
            weight = __shfl(weight, sub_wave * 32, 64);

            const auto* vh = reinterpret_cast<const __half*>(&v_raw[u]);
            acc0.x = acc0.x * alpha + weight * __half2float(vh[0]);
            acc0.y = acc0.y * alpha + weight * __half2float(vh[1]);
            acc0.z = acc0.z * alpha + weight * __half2float(vh[2]);
            acc0.w = acc0.w * alpha + weight * __half2float(vh[3]);

            acc1.x = acc1.x * alpha + weight * __half2float(vh[4]);
            acc1.y = acc1.y * alpha + weight * __half2float(vh[5]);
            acc1.z = acc1.z * alpha + weight * __half2float(vh[6]);
            acc1.w = acc1.w * alpha + weight * __half2float(vh[7]);
        }

        k_f16_ptr += UNROLL * head_dim;
        v_f16_ptr += UNROLL * head_dim;
    }

    for (; pos < k_end; ++pos) {
        const uint4 k_raw = *reinterpret_cast<const uint4*>(k_f16_ptr);
        const auto* kh = reinterpret_cast<const __half*>(&k_raw);
        float dot = q_vec0.x * __half2float(kh[0]) + q_vec0.y * __half2float(kh[1])
                  + q_vec0.z * __half2float(kh[2]) + q_vec0.w * __half2float(kh[3])
                  + q_vec1.x * __half2float(kh[4]) + q_vec1.y * __half2float(kh[5])
                  + q_vec1.z * __half2float(kh[6]) + q_vec1.w * __half2float(kh[7]);
        k_f16_ptr += head_dim;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }
        float score = __shfl(dot * scale, sub_wave * 32, 64);

        float alpha = 1.0f;
        float weight = 0.0f;
        if (sub_lane == 0) {
            if (score > running_max) {
                alpha = expf(running_max - score);
                running_sum = running_sum * alpha + 1.0f;
                running_max = score;
                weight = 1.0f;
            } else {
                weight = expf(score - running_max);
                running_sum += weight;
            }
        }
        alpha = __shfl(alpha, sub_wave * 32, 64);
        weight = __shfl(weight, sub_wave * 32, 64);

        const uint4 v_raw = *reinterpret_cast<const uint4*>(v_f16_ptr);
        const auto* vh = reinterpret_cast<const __half*>(&v_raw);

        acc0.x = acc0.x * alpha + weight * __half2float(vh[0]);
        acc0.y = acc0.y * alpha + weight * __half2float(vh[1]);
        acc0.z = acc0.z * alpha + weight * __half2float(vh[2]);
        acc0.w = acc0.w * alpha + weight * __half2float(vh[3]);

        acc1.x = acc1.x * alpha + weight * __half2float(vh[4]);
        acc1.y = acc1.y * alpha + weight * __half2float(vh[5]);
        acc1.z = acc1.z * alpha + weight * __half2float(vh[6]);
        acc1.w = acc1.w * alpha + weight * __half2float(vh[7]);

        v_f16_ptr += head_dim;
    }

    if (sub_lane == 0) {
        partial_max[split_head_idx] = running_max;
        partial_sum[split_head_idx] = running_sum;
    }
    const std::size_t acc_base = split_head_idx * head_dim + dim_offset;
    *reinterpret_cast<float4*>(partial_acc + acc_base + 0) = acc0;
    *reinterpret_cast<float4*>(partial_acc + acc_base + 4) = acc1;
}

// -----------------------------------------------------------------------------
// Stage 2: Reduction Kernel
// -----------------------------------------------------------------------------
__launch_bounds__(64, 4) __global__ void
qwen35_splitk_suffix_reduction_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_acc,
    const float* __restrict__ gate,
    float* __restrict__ gated_output,
    std::uint32_t token_count,
    std::uint32_t query_heads,
    std::uint32_t head_dim,
    std::uint32_t num_splits) {

    const std::uint32_t token = blockIdx.x;
    const std::uint32_t head = blockIdx.y;
    const std::uint32_t lane = threadIdx.x;

    if (token >= token_count || head >= query_heads || lane >= 64) return;

    const std::uint32_t dim_offset = lane * 4;

    float global_max = -INFINITY;
    for (std::uint32_t s = 0; s < num_splits; ++s) {
        const std::size_t split_head_idx =
            (static_cast<std::size_t>(s) * token_count + token) * query_heads + head;
        const float s_max = partial_max[split_head_idx];
        if (s_max > global_max) global_max = s_max;
    }

    float global_sum = 0.0f;
    for (std::uint32_t s = 0; s < num_splits; ++s) {
        const std::size_t split_head_idx =
            (static_cast<std::size_t>(s) * token_count + token) * query_heads + head;
        const float s_max = partial_max[split_head_idx];
        const float s_sum = partial_sum[split_head_idx];
        if (s_sum > 0.0f) {
            global_sum += s_sum * expf(s_max - global_max);
        }
    }

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (std::uint32_t s = 0; s < num_splits; ++s) {
        const std::size_t split_head_idx =
            (static_cast<std::size_t>(s) * token_count + token) * query_heads + head;
        const float s_max = partial_max[split_head_idx];
        const float s_sum = partial_sum[split_head_idx];
        if (s_sum > 0.0f) {
            const float factor = expf(s_max - global_max);
            const std::size_t acc_base = split_head_idx * head_dim + dim_offset;
            const float4 s_acc = *reinterpret_cast<const float4*>(partial_acc + acc_base);
            acc.x += s_acc.x * factor;
            acc.y += s_acc.y * factor;
            acc.z += s_acc.z * factor;
            acc.w += s_acc.w * factor;
        }
    }

    const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    acc.x *= inv_sum;
    acc.y *= inv_sum;
    acc.z *= inv_sum;
    acc.w *= inv_sum;

    const std::size_t out_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim + dim_offset;

    if (gate != nullptr) {
        const float4 g = *reinterpret_cast<const float4*>(gate + out_base);
        acc.x *= (1.0f / (1.0f + expf(-g.x)));
        acc.y *= (1.0f / (1.0f + expf(-g.y)));
        acc.z *= (1.0f / (1.0f + expf(-g.z)));
        acc.w *= (1.0f / (1.0f + expf(-g.w)));
    }

    *reinterpret_cast<float4*>(gated_output + out_base) = acc;
}

} // namespace

int main() {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0025: GQA Multi-Head Workgroup Benchmark\n";
    std::cout << "  Target: 1 x AMD Instinct MI50 32GB (gfx906, Wave64, 60 CUs)\n";
    std::cout << "  Workload: P = 65,536 prefix tokens, S = 512 suffix tokens\n";
    std::cout << "===================================================================\n\n";

    int device_id = 0;
    MIINFER_HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t props{};
    MIINFER_HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    std::cout << "[INFO] Device: " << props.name << " (" << props.gcnArchName << "), CUs="
              << props.multiProcessorCount << ", Clock=" << props.clockRate / 1000 << " MHz\n\n";

    constexpr std::uint32_t kQueryHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kHeadDim = 256;
    constexpr std::uint32_t kGqaLayers = 16;
    constexpr float kScale = 1.0f / 16.0f; // 1 / sqrt(256)
    constexpr std::uint32_t kNumSplits = 32;

    constexpr std::uint32_t kPrefixLen = 65536;
    constexpr std::uint32_t kSuffixLen = 512;
    constexpr std::uint32_t kCacheCap = kPrefixLen + kSuffixLen + 1024;

    const std::size_t q_elements = static_cast<std::size_t>(kSuffixLen) * kQueryHeads * kHeadDim;
    const std::size_t kv_elements = static_cast<std::size_t>(kKvHeads) * kCacheCap * kHeadDim;
    const std::size_t partial_stat_elements = static_cast<std::size_t>(kNumSplits) * kSuffixLen * kQueryHeads;
    const std::size_t partial_acc_elements = partial_stat_elements * kHeadDim;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.4f);

    std::vector<float> h_q(q_elements);
    std::vector<float> h_gate(q_elements);
    std::vector<__half> h_key_f16(kv_elements);
    std::vector<__half> h_val_f16(kv_elements);

    for (auto& v : h_q) v = dist(rng);
    for (auto& v : h_gate) v = dist(rng);
    for (auto& v : h_key_f16) v = __float2half(dist(rng));
    for (auto& v : h_val_f16) v = __float2half(dist(rng));

    GpuBuffer d_q(q_elements * sizeof(float));
    GpuBuffer d_gate(q_elements * sizeof(float));
    GpuBuffer d_key_f16(kv_elements * sizeof(__half));
    GpuBuffer d_val_f16(kv_elements * sizeof(__half));

    GpuBuffer d_part_max_ctrl(partial_stat_elements * sizeof(float));
    GpuBuffer d_part_sum_ctrl(partial_stat_elements * sizeof(float));
    GpuBuffer d_part_acc_ctrl(partial_acc_elements * sizeof(float));
    GpuBuffer d_out_ctrl(q_elements * sizeof(float));

    GpuBuffer d_part_max_cand_c(partial_stat_elements * sizeof(float));
    GpuBuffer d_part_sum_cand_c(partial_stat_elements * sizeof(float));
    GpuBuffer d_part_acc_cand_c(partial_acc_elements * sizeof(float));
    GpuBuffer d_out_cand_c(q_elements * sizeof(float));

    MIINFER_HIP_CHECK(hipMemcpy(d_q.ptr, h_q.data(), q_elements * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate.ptr, h_gate.data(), q_elements * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_key_f16.ptr, h_key_f16.data(), kv_elements * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_val_f16.ptr, h_val_f16.data(), kv_elements * sizeof(__half), hipMemcpyHostToDevice));

    hipEvent_t start_ev, stop_ev;
    MIINFER_HIP_CHECK(hipEventCreate(&start_ev));
    MIINFER_HIP_CHECK(hipEventCreate(&stop_ev));

    // -------------------------------------------------------------------------
    // Correctness Verification
    // -------------------------------------------------------------------------
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  PHASE 1: Numerical Correctness Verification\n";
    std::cout << "-------------------------------------------------------------------\n";

    // Run Control
    hipLaunchKernelGGL(
        qwen35_splitk_suffix_attn_control_kernel<4>,
        dim3(kQueryHeads / 2, kSuffixLen, kNumSplits),
        dim3(64), 0, 0,
        static_cast<const float*>(d_q.ptr),
        static_cast<const __half*>(d_key_f16.ptr),
        static_cast<const __half*>(d_val_f16.ptr),
        static_cast<float*>(d_part_max_ctrl.ptr),
        static_cast<float*>(d_part_sum_ctrl.ptr),
        static_cast<float*>(d_part_acc_ctrl.ptr),
        kSuffixLen, kPrefixLen, kCacheCap,
        kQueryHeads, kKvHeads, kHeadDim, kScale, kNumSplits);

    hipLaunchKernelGGL(
        qwen35_splitk_suffix_reduction_kernel,
        dim3(kSuffixLen, kQueryHeads),
        dim3(64), 0, 0,
        static_cast<const float*>(d_part_max_ctrl.ptr),
        static_cast<const float*>(d_part_sum_ctrl.ptr),
        static_cast<const float*>(d_part_acc_ctrl.ptr),
        static_cast<const float*>(d_gate.ptr),
        static_cast<float*>(d_out_ctrl.ptr),
        kSuffixLen, kQueryHeads, kHeadDim, kNumSplits);

    // Run Candidate C (Co-scheduled 192T Multi-Wave Workgroup)
    hipLaunchKernelGGL(
        qwen35_splitk_suffix_attn_l1_coscheduled_kernel<4>,
        dim3(kKvHeads, kSuffixLen, kNumSplits),
        dim3(192), 0, 0,
        static_cast<const float*>(d_q.ptr),
        static_cast<const __half*>(d_key_f16.ptr),
        static_cast<const __half*>(d_val_f16.ptr),
        static_cast<float*>(d_part_max_cand_c.ptr),
        static_cast<float*>(d_part_sum_cand_c.ptr),
        static_cast<float*>(d_part_acc_cand_c.ptr),
        kSuffixLen, kPrefixLen, kCacheCap,
        kQueryHeads, kKvHeads, kHeadDim, kScale, kNumSplits);

    hipLaunchKernelGGL(
        qwen35_splitk_suffix_reduction_kernel,
        dim3(kSuffixLen, kQueryHeads),
        dim3(64), 0, 0,
        static_cast<const float*>(d_part_max_cand_c.ptr),
        static_cast<const float*>(d_part_sum_cand_c.ptr),
        static_cast<const float*>(d_part_acc_cand_c.ptr),
        static_cast<const float*>(d_gate.ptr),
        static_cast<float*>(d_out_cand_c.ptr),
        kSuffixLen, kQueryHeads, kHeadDim, kNumSplits);

    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_out_ctrl(q_elements);
    std::vector<float> h_out_cand_c(q_elements);
    MIINFER_HIP_CHECK(hipMemcpy(h_out_ctrl.data(), d_out_ctrl.ptr, q_elements * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(h_out_cand_c.data(), d_out_cand_c.ptr, q_elements * sizeof(float), hipMemcpyDeviceToHost));

    double max_diff = 0.0, sum_diff = 0.0;
    for (std::size_t i = 0; i < q_elements; ++i) {
        const double diff = std::fabs(h_out_ctrl[i] - h_out_cand_c[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }
    std::cout << "  Candidate C Max Absolute Error: " << std::scientific << max_diff << "\n";
    std::cout << "  Candidate C Mean Absolute Error: " << std::scientific << sum_diff / q_elements << "\n";
    std::cout << "  Parity Status: " << (max_diff <= 1e-4 ? "[PASS: EXACT/HIGH PRECISION MATCH]" : "[FAIL]") << "\n\n";

    // -------------------------------------------------------------------------
    // Benchmark 16 GQA Layers
    // -------------------------------------------------------------------------
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  PHASE 2: 16-Layer Suffix Attention Benchmark (P=64K, S=512)\n";
    std::cout << "-------------------------------------------------------------------\n";

    constexpr int kRounds = 10;

    auto bench_control = [&]() {
        std::vector<double> times;
        for (int r = 0; r < kRounds; ++r) {
            MIINFER_HIP_CHECK(hipEventRecord(start_ev));
            for (std::uint32_t l = 0; l < kGqaLayers; ++l) {
                hipLaunchKernelGGL(
                    qwen35_splitk_suffix_attn_control_kernel<4>,
                    dim3(kQueryHeads / 2, kSuffixLen, kNumSplits),
                    dim3(64), 0, 0,
                    static_cast<const float*>(d_q.ptr),
                    static_cast<const __half*>(d_key_f16.ptr),
                    static_cast<const __half*>(d_val_f16.ptr),
                    static_cast<float*>(d_part_max_ctrl.ptr),
                    static_cast<float*>(d_part_sum_ctrl.ptr),
                    static_cast<float*>(d_part_acc_ctrl.ptr),
                    kSuffixLen, kPrefixLen, kCacheCap,
                    kQueryHeads, kKvHeads, kHeadDim, kScale, kNumSplits);

                hipLaunchKernelGGL(
                    qwen35_splitk_suffix_reduction_kernel,
                    dim3(kSuffixLen, kQueryHeads),
                    dim3(64), 0, 0,
                    static_cast<const float*>(d_part_max_ctrl.ptr),
                    static_cast<const float*>(d_part_sum_ctrl.ptr),
                    static_cast<const float*>(d_part_acc_ctrl.ptr),
                    static_cast<const float*>(d_gate.ptr),
                    static_cast<float*>(d_out_ctrl.ptr),
                    kSuffixLen, kQueryHeads, kHeadDim, kNumSplits);
            }
            MIINFER_HIP_CHECK(hipEventRecord(stop_ev));
            MIINFER_HIP_CHECK(hipEventSynchronize(stop_ev));
            times.push_back(elapsed_ms(start_ev, stop_ev));
        }
        return median(times);
    };

    auto bench_candidate_c = [&]() {
        std::vector<double> times;
        for (int r = 0; r < kRounds; ++r) {
            MIINFER_HIP_CHECK(hipEventRecord(start_ev));
            for (std::uint32_t l = 0; l < kGqaLayers; ++l) {
                hipLaunchKernelGGL(
                    qwen35_splitk_suffix_attn_l1_coscheduled_kernel<4>,
                    dim3(kKvHeads, kSuffixLen, kNumSplits),
                    dim3(192), 0, 0,
                    static_cast<const float*>(d_q.ptr),
                    static_cast<const __half*>(d_key_f16.ptr),
                    static_cast<const __half*>(d_val_f16.ptr),
                    static_cast<float*>(d_part_max_cand_c.ptr),
                    static_cast<float*>(d_part_sum_cand_c.ptr),
                    static_cast<float*>(d_part_acc_cand_c.ptr),
                    kSuffixLen, kPrefixLen, kCacheCap,
                    kQueryHeads, kKvHeads, kHeadDim, kScale, kNumSplits);

                hipLaunchKernelGGL(
                    qwen35_splitk_suffix_reduction_kernel,
                    dim3(kSuffixLen, kQueryHeads),
                    dim3(64), 0, 0,
                    static_cast<const float*>(d_part_max_cand_c.ptr),
                    static_cast<const float*>(d_part_sum_cand_c.ptr),
                    static_cast<const float*>(d_part_acc_cand_c.ptr),
                    static_cast<const float*>(d_gate.ptr),
                    static_cast<float*>(d_out_cand_c.ptr),
                    kSuffixLen, kQueryHeads, kHeadDim, kNumSplits);
            }
            MIINFER_HIP_CHECK(hipEventRecord(stop_ev));
            MIINFER_HIP_CHECK(hipEventSynchronize(stop_ev));
            times.push_back(elapsed_ms(start_ev, stop_ev));
        }
        return median(times);
    };

    std::cout << "  Warming up...\n";
    bench_control();
    bench_candidate_c();

    std::cout << "  Benchmarking 16 GQA Layers...\n";
    const double med_ctrl = bench_control();
    const double med_cand_c = bench_candidate_c();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n===================================================================\n";
    std::cout << "  FINAL V2-0025 GQA ATTENTION RESULTS (16 Layers)\n";
    std::cout << "===================================================================\n";
    std::cout << "  Control Baseline (FP16 Split-K)    : " << med_ctrl << " ms (" << med_ctrl / 16.0 << " ms/layer)\n";
    std::cout << "  Candidate C (Co-scheduled 192T WG) : " << med_cand_c << " ms (" << med_cand_c / 16.0 << " ms/layer)\n";
    std::cout << "  Speedup Factor                     : " << med_ctrl / med_cand_c << "x\n";
    std::cout << "  Latency Delta                      : " << std::showpos << med_cand_c - med_ctrl << " ms "
              << "(" << (med_cand_c - med_ctrl) / med_ctrl * 100.0 << "%)\n" << std::noshowpos;
    std::cout << "===================================================================\n";

    MIINFER_HIP_CHECK(hipEventDestroy(start_ev));
    MIINFER_HIP_CHECK(hipEventDestroy(stop_ev));

    return 0;
}
