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

namespace {

struct GpuBuffer {
    void* ptr = nullptr;
    explicit GpuBuffer(std::size_t bytes) {
        MIINFER_HIP_CHECK(hipMalloc(&ptr, bytes));
    }
    ~GpuBuffer() {
        if (ptr != nullptr) hipFree(ptr);
    }
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
};

float elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

// -----------------------------------------------------------------------------
// Pipeline 0: Control Baseline (Unpipelined Unroll=4)
// -----------------------------------------------------------------------------
template <int UNROLL = 4>
__launch_bounds__(64, 4) __global__ void
splitk_suffix_attn_pipe0_control_kernel(
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
        uint4 k_raw_f16[UNROLL];
        uint4 v_raw_f16[UNROLL];

        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            k_raw_f16[u] = *reinterpret_cast<const uint4*>(k_f16_ptr + static_cast<std::size_t>(u) * head_dim);
            v_raw_f16[u] = *reinterpret_cast<const uint4*>(v_f16_ptr + static_cast<std::size_t>(u) * head_dim);
        }

        float dots[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const auto* kh = reinterpret_cast<const __half*>(&k_raw_f16[u]);
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

            const auto* vh = reinterpret_cast<const __half*>(&v_raw_f16[u]);
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

    // Tail loop
    for (; pos < k_end; ++pos) {
        const uint4 k_raw_f16 = *reinterpret_cast<const uint4*>(k_f16_ptr);
        const auto* kh = reinterpret_cast<const __half*>(&k_raw_f16);
        float dot = q_vec0.x * __half2float(kh[0]) + q_vec0.y * __half2float(kh[1])
                  + q_vec0.z * __half2float(kh[2]) + q_vec0.w * __half2float(kh[3])
                  + q_vec1.x * __half2float(kh[4]) + q_vec1.y * __half2float(kh[5])
                  + q_vec1.z * __half2float(kh[6]) + q_vec1.w * __half2float(kh[7]);
        k_f16_ptr += head_dim;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }
        const float score = __shfl(dot * scale, sub_wave * 32, 64);

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

        const uint4 v_raw_f16 = *reinterpret_cast<const uint4*>(v_f16_ptr);
        const auto* vh = reinterpret_cast<const __half*>(&v_raw_f16);
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
// Pipeline 1: 1-Token Double-Buffered Software Pipelining (U=1)
// -----------------------------------------------------------------------------
__launch_bounds__(64, 4) __global__ void
splitk_suffix_attn_pipe1_double_buf_u1_kernel(
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

    uint4 k_cur, v_cur;
    bool has_item = false;
    if (pos < k_end) {
        k_cur = *reinterpret_cast<const uint4*>(k_f16_ptr);
        v_cur = *reinterpret_cast<const uint4*>(v_f16_ptr);
        k_f16_ptr += head_dim;
        v_f16_ptr += head_dim;
        ++pos;
        has_item = true;
    }

    for (; pos < k_end; ++pos) {
        const uint4 k_next = *reinterpret_cast<const uint4*>(k_f16_ptr);
        const uint4 v_next = *reinterpret_cast<const uint4*>(v_f16_ptr);
        k_f16_ptr += head_dim;
        v_f16_ptr += head_dim;

        const auto* kh = reinterpret_cast<const __half*>(&k_cur);
        float dot = q_vec0.x * __half2float(kh[0]) + q_vec0.y * __half2float(kh[1])
                  + q_vec0.z * __half2float(kh[2]) + q_vec0.w * __half2float(kh[3])
                  + q_vec1.x * __half2float(kh[4]) + q_vec1.y * __half2float(kh[5])
                  + q_vec1.z * __half2float(kh[6]) + q_vec1.w * __half2float(kh[7]);

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }
        const float score = __shfl(dot * scale, sub_wave * 32, 64);

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

        const auto* vh = reinterpret_cast<const __half*>(&v_cur);
        acc0.x = acc0.x * alpha + weight * __half2float(vh[0]);
        acc0.y = acc0.y * alpha + weight * __half2float(vh[1]);
        acc0.z = acc0.z * alpha + weight * __half2float(vh[2]);
        acc0.w = acc0.w * alpha + weight * __half2float(vh[3]);

        acc1.x = acc1.x * alpha + weight * __half2float(vh[4]);
        acc1.y = acc1.y * alpha + weight * __half2float(vh[5]);
        acc1.z = acc1.z * alpha + weight * __half2float(vh[6]);
        acc1.w = acc1.w * alpha + weight * __half2float(vh[7]);

        k_cur = k_next;
        v_cur = v_next;
    }

    if (has_item) {
        const auto* kh = reinterpret_cast<const __half*>(&k_cur);
        float dot = q_vec0.x * __half2float(kh[0]) + q_vec0.y * __half2float(kh[1])
                  + q_vec0.z * __half2float(kh[2]) + q_vec0.w * __half2float(kh[3])
                  + q_vec1.x * __half2float(kh[4]) + q_vec1.y * __half2float(kh[5])
                  + q_vec1.z * __half2float(kh[6]) + q_vec1.w * __half2float(kh[7]);

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }
        const float score = __shfl(dot * scale, sub_wave * 32, 64);

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

        const auto* vh = reinterpret_cast<const __half*>(&v_cur);
        acc0.x = acc0.x * alpha + weight * __half2float(vh[0]);
        acc0.y = acc0.y * alpha + weight * __half2float(vh[1]);
        acc0.z = acc0.z * alpha + weight * __half2float(vh[2]);
        acc0.w = acc0.w * alpha + weight * __half2float(vh[3]);

        acc1.x = acc1.x * alpha + weight * __half2float(vh[4]);
        acc1.y = acc1.y * alpha + weight * __half2float(vh[5]);
        acc1.z = acc1.z * alpha + weight * __half2float(vh[6]);
        acc1.w = acc1.w * alpha + weight * __half2float(vh[7]);
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
// Pipeline 2: 2-Token Double-Buffered Software Pipelining (U=2)
// -----------------------------------------------------------------------------
__launch_bounds__(64) __global__ void
splitk_suffix_attn_pipe2_double_buf_u2_kernel(
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

    uint4 k_c0, k_c1, v_c0, v_c1;
    bool has_pair = false;
    if (pos + 2 <= k_end) {
        k_c0 = *reinterpret_cast<const uint4*>(k_f16_ptr + 0 * head_dim);
        v_c0 = *reinterpret_cast<const uint4*>(v_f16_ptr + 0 * head_dim);
        k_c1 = *reinterpret_cast<const uint4*>(k_f16_ptr + 1 * head_dim);
        v_c1 = *reinterpret_cast<const uint4*>(v_f16_ptr + 1 * head_dim);
        k_f16_ptr += 2 * head_dim;
        v_f16_ptr += 2 * head_dim;
        pos += 2;
        has_pair = true;
    }

    for (; pos + 2 <= k_end; pos += 2) {
        const uint4 k_n0 = *reinterpret_cast<const uint4*>(k_f16_ptr + 0 * head_dim);
        const uint4 v_n0 = *reinterpret_cast<const uint4*>(v_f16_ptr + 0 * head_dim);
        const uint4 k_n1 = *reinterpret_cast<const uint4*>(k_f16_ptr + 1 * head_dim);
        const uint4 v_n1 = *reinterpret_cast<const uint4*>(v_f16_ptr + 1 * head_dim);
        k_f16_ptr += 2 * head_dim;
        v_f16_ptr += 2 * head_dim;

        // Token 0
        {
            const auto* kh0 = reinterpret_cast<const __half*>(&k_c0);
            float dot0 = q_vec0.x * __half2float(kh0[0]) + q_vec0.y * __half2float(kh0[1])
                       + q_vec0.z * __half2float(kh0[2]) + q_vec0.w * __half2float(kh0[3])
                       + q_vec1.x * __half2float(kh0[4]) + q_vec1.y * __half2float(kh0[5])
                       + q_vec1.z * __half2float(kh0[6]) + q_vec1.w * __half2float(kh0[7]);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) dot0 += __shfl_xor(dot0, offset, 32);
            const float score0 = __shfl(dot0 * scale, sub_wave * 32, 64);
            float alpha0 = 1.0f, weight0 = 0.0f;
            if (sub_lane == 0) {
                if (score0 > running_max) {
                    alpha0 = expf(running_max - score0);
                    running_sum = running_sum * alpha0 + 1.0f;
                    running_max = score0;
                    weight0 = 1.0f;
                } else {
                    weight0 = expf(score0 - running_max);
                    running_sum += weight0;
                }
            }
            alpha0 = __shfl(alpha0, sub_wave * 32, 64);
            weight0 = __shfl(weight0, sub_wave * 32, 64);
            const auto* vh0 = reinterpret_cast<const __half*>(&v_c0);
            acc0.x = acc0.x * alpha0 + weight0 * __half2float(vh0[0]);
            acc0.y = acc0.y * alpha0 + weight0 * __half2float(vh0[1]);
            acc0.z = acc0.z * alpha0 + weight0 * __half2float(vh0[2]);
            acc0.w = acc0.w * alpha0 + weight0 * __half2float(vh0[3]);
            acc1.x = acc1.x * alpha0 + weight0 * __half2float(vh0[4]);
            acc1.y = acc1.y * alpha0 + weight0 * __half2float(vh0[5]);
            acc1.z = acc1.z * alpha0 + weight0 * __half2float(vh0[6]);
            acc1.w = acc1.w * alpha0 + weight0 * __half2float(vh0[7]);
        }

        // Token 1
        {
            const auto* kh1 = reinterpret_cast<const __half*>(&k_c1);
            float dot1 = q_vec0.x * __half2float(kh1[0]) + q_vec0.y * __half2float(kh1[1])
                       + q_vec0.z * __half2float(kh1[2]) + q_vec0.w * __half2float(kh1[3])
                       + q_vec1.x * __half2float(kh1[4]) + q_vec1.y * __half2float(kh1[5])
                       + q_vec1.z * __half2float(kh1[6]) + q_vec1.w * __half2float(kh1[7]);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) dot1 += __shfl_xor(dot1, offset, 32);
            const float score1 = __shfl(dot1 * scale, sub_wave * 32, 64);
            float alpha1 = 1.0f, weight1 = 0.0f;
            if (sub_lane == 0) {
                if (score1 > running_max) {
                    alpha1 = expf(running_max - score1);
                    running_sum = running_sum * alpha1 + 1.0f;
                    running_max = score1;
                    weight1 = 1.0f;
                } else {
                    weight1 = expf(score1 - running_max);
                    running_sum += weight1;
                }
            }
            alpha1 = __shfl(alpha1, sub_wave * 32, 64);
            weight1 = __shfl(weight1, sub_wave * 32, 64);
            const auto* vh1 = reinterpret_cast<const __half*>(&v_c1);
            acc0.x = acc0.x * alpha1 + weight1 * __half2float(vh1[0]);
            acc0.y = acc0.y * alpha1 + weight1 * __half2float(vh1[1]);
            acc0.z = acc0.z * alpha1 + weight1 * __half2float(vh1[2]);
            acc0.w = acc0.w * alpha1 + weight1 * __half2float(vh1[3]);
            acc1.x = acc1.x * alpha1 + weight1 * __half2float(vh1[4]);
            acc1.y = acc1.y * alpha1 + weight1 * __half2float(vh1[5]);
            acc1.z = acc1.z * alpha1 + weight1 * __half2float(vh1[6]);
            acc1.w = acc1.w * alpha1 + weight1 * __half2float(vh1[7]);
        }

        k_c0 = k_n0; v_c0 = v_n0;
        k_c1 = k_n1; v_c1 = v_n1;
    }

    if (has_pair) {
        // Epilogue for last pair
        {
            const auto* kh0 = reinterpret_cast<const __half*>(&k_c0);
            float dot0 = q_vec0.x * __half2float(kh0[0]) + q_vec0.y * __half2float(kh0[1])
                       + q_vec0.z * __half2float(kh0[2]) + q_vec0.w * __half2float(kh0[3])
                       + q_vec1.x * __half2float(kh0[4]) + q_vec1.y * __half2float(kh0[5])
                       + q_vec1.z * __half2float(kh0[6]) + q_vec1.w * __half2float(kh0[7]);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) dot0 += __shfl_xor(dot0, offset, 32);
            const float score0 = __shfl(dot0 * scale, sub_wave * 32, 64);
            float alpha0 = 1.0f, weight0 = 0.0f;
            if (sub_lane == 0) {
                if (score0 > running_max) {
                    alpha0 = expf(running_max - score0);
                    running_sum = running_sum * alpha0 + 1.0f;
                    running_max = score0;
                    weight0 = 1.0f;
                } else {
                    weight0 = expf(score0 - running_max);
                    running_sum += weight0;
                }
            }
            alpha0 = __shfl(alpha0, sub_wave * 32, 64);
            weight0 = __shfl(weight0, sub_wave * 32, 64);
            const auto* vh0 = reinterpret_cast<const __half*>(&v_c0);
            acc0.x = acc0.x * alpha0 + weight0 * __half2float(vh0[0]);
            acc0.y = acc0.y * alpha0 + weight0 * __half2float(vh0[1]);
            acc0.z = acc0.z * alpha0 + weight0 * __half2float(vh0[2]);
            acc0.w = acc0.w * alpha0 + weight0 * __half2float(vh0[3]);
            acc1.x = acc1.x * alpha0 + weight0 * __half2float(vh0[4]);
            acc1.y = acc1.y * alpha0 + weight0 * __half2float(vh0[5]);
            acc1.z = acc1.z * alpha0 + weight0 * __half2float(vh0[6]);
            acc1.w = acc1.w * alpha0 + weight0 * __half2float(vh0[7]);
        }

        {
            const auto* kh1 = reinterpret_cast<const __half*>(&k_c1);
            float dot1 = q_vec0.x * __half2float(kh1[0]) + q_vec0.y * __half2float(kh1[1])
                       + q_vec0.z * __half2float(kh1[2]) + q_vec0.w * __half2float(kh1[3])
                       + q_vec1.x * __half2float(kh1[4]) + q_vec1.y * __half2float(kh1[5])
                       + q_vec1.z * __half2float(kh1[6]) + q_vec1.w * __half2float(kh1[7]);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) dot1 += __shfl_xor(dot1, offset, 32);
            const float score1 = __shfl(dot1 * scale, sub_wave * 32, 64);
            float alpha1 = 1.0f, weight1 = 0.0f;
            if (sub_lane == 0) {
                if (score1 > running_max) {
                    alpha1 = expf(running_max - score1);
                    running_sum = running_sum * alpha1 + 1.0f;
                    running_max = score1;
                    weight1 = 1.0f;
                } else {
                    weight1 = expf(score1 - running_max);
                    running_sum += weight1;
                }
            }
            alpha1 = __shfl(alpha1, sub_wave * 32, 64);
            weight1 = __shfl(weight1, sub_wave * 32, 64);
            const auto* vh1 = reinterpret_cast<const __half*>(&v_c1);
            acc0.x = acc0.x * alpha1 + weight1 * __half2float(vh1[0]);
            acc0.y = acc0.y * alpha1 + weight1 * __half2float(vh1[1]);
            acc0.z = acc0.z * alpha1 + weight1 * __half2float(vh1[2]);
            acc0.w = acc0.w * alpha1 + weight1 * __half2float(vh1[3]);
            acc1.x = acc1.x * alpha1 + weight1 * __half2float(vh1[4]);
            acc1.y = acc1.y * alpha1 + weight1 * __half2float(vh1[5]);
            acc1.z = acc1.z * alpha1 + weight1 * __half2float(vh1[6]);
            acc1.w = acc1.w * alpha1 + weight1 * __half2float(vh1[7]);
        }
    }

    // Tail loop
    for (; pos < k_end; ++pos) {
        const uint4 k_raw = *reinterpret_cast<const uint4*>(k_f16_ptr);
        const auto* kh = reinterpret_cast<const __half*>(&k_raw);
        float dot = q_vec0.x * __half2float(kh[0]) + q_vec0.y * __half2float(kh[1])
                  + q_vec0.z * __half2float(kh[2]) + q_vec0.w * __half2float(kh[3])
                  + q_vec1.x * __half2float(kh[4]) + q_vec1.y * __half2float(kh[5])
                  + q_vec1.z * __half2float(kh[6]) + q_vec1.w * __half2float(kh[7]);
        k_f16_ptr += head_dim;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_xor(dot, offset, 32);
        const float score = __shfl(dot * scale, sub_wave * 32, 64);

        float alpha = 1.0f, weight = 0.0f;
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
// Stage 2 reduction kernel
// -----------------------------------------------------------------------------
__launch_bounds__(64) __global__ void
splitk_suffix_attn_stage2_bench_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_acc,
    const float* __restrict__ gate,
    float* __restrict__ gated_output,
    std::uint32_t token_count,
    std::uint32_t query_heads,
    std::uint32_t head_dim,
    std::uint32_t num_splits) {

    const std::uint32_t token_head_idx = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;

    if (token_head_idx >= token_count * query_heads || lane >= 64) return;

    const std::uint32_t token = token_head_idx / query_heads;
    const std::uint32_t head = token_head_idx % query_heads;

    __shared__ float s_global_max;
    __shared__ float s_global_sum;

    if (lane == 0) {
        float global_max = -INFINITY;
        for (std::uint32_t s = 0; s < num_splits; ++s) {
            const std::size_t split_idx =
                (static_cast<std::size_t>(s) * token_count + token) * query_heads + head;
            const float m = partial_max[split_idx];
            if (m > global_max) global_max = m;
        }
        s_global_max = global_max;

        float global_sum = 0.0f;
        for (std::uint32_t s = 0; s < num_splits; ++s) {
            const std::size_t split_idx =
                (static_cast<std::size_t>(s) * token_count + token) * query_heads + head;
            const float m = partial_max[split_idx];
            const float l = partial_sum[split_idx];
            if (l > 0.0f && std::isfinite(m)) {
                global_sum += l * expf(m - global_max);
            }
        }
        s_global_sum = global_sum;
    }
    __syncthreads();

    const float global_max = s_global_max;
    const float global_sum = s_global_sum;
    const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    float4 total_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const std::size_t dim_offset = lane * 4;

    for (std::uint32_t s = 0; s < num_splits; ++s) {
        const std::size_t split_idx =
            (static_cast<std::size_t>(s) * token_count + token) * query_heads + head;
        const float m = partial_max[split_idx];
        const float l = partial_sum[split_idx];
        if (l > 0.0f && std::isfinite(m)) {
            const float alpha = expf(m - global_max);
            const std::size_t acc_base = split_idx * head_dim + dim_offset;
            const float4 partial = *reinterpret_cast<const float4*>(partial_acc + acc_base);
            total_acc.x += partial.x * alpha;
            total_acc.y += partial.y * alpha;
            total_acc.z += partial.z * alpha;
            total_acc.w += partial.w * alpha;
        }
    }

    const std::size_t out_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim + dim_offset;
    if (gate != nullptr) {
        const float4 g_vec = *reinterpret_cast<const float4*>(gate + out_base);
        total_acc.x *= (1.0f / (1.0f + expf(-g_vec.x)));
        total_acc.y *= (1.0f / (1.0f + expf(-g_vec.y)));
        total_acc.z *= (1.0f / (1.0f + expf(-g_vec.z)));
        total_acc.w *= (1.0f / (1.0f + expf(-g_vec.w)));
    }

    float4 out = make_float4(
        total_acc.x * inv_sum, total_acc.y * inv_sum, total_acc.z * inv_sum, total_acc.w * inv_sum);
    *reinterpret_cast<float4*>(gated_output + out_base) = out;
}

void compute_diff(const std::vector<float>& ref, const std::vector<float>& test, double& max_diff, double& cos_sim) {
    max_diff = 0.0;
    double dot = 0.0, norm_ref = 0.0, norm_test = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        double d = std::abs(static_cast<double>(ref[i]) - static_cast<double>(test[i]));
        if (d > max_diff) max_diff = d;
        dot += static_cast<double>(ref[i]) * static_cast<double>(test[i]);
        norm_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        norm_test += static_cast<double>(test[i]) * static_cast<double>(test[i]);
    }
    cos_sim = dot / (std::sqrt(norm_ref) * std::sqrt(norm_test) + 1e-12);
}

}  // namespace

int main() {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0020: FP16 Suffix Attention Pipeline Microbenchmark\n";
    std::cout << "  Workload: P = 65,536 cached prefix, S = 512 new suffix tokens\n";
    std::cout << "  Testing Pipeline 0 (Control) vs Pipeline 1/2 (Double Buffering)\n";
    std::cout << "===================================================================\n" << std::flush;

    constexpr std::uint32_t kQueryHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kHeadDim = 256;
    constexpr float kScale = 1.0f / 16.0f;
    constexpr std::uint32_t kSplits = 32;

    constexpr std::uint32_t kPrefix = 65536;
    constexpr std::uint32_t kSuffix = 512;
    constexpr std::uint32_t kCap = kPrefix + kSuffix + 1024;

    const std::size_t q_cnt = static_cast<std::size_t>(kSuffix) * kQueryHeads * kHeadDim;
    const std::size_t kv_cnt = static_cast<std::size_t>(kKvHeads) * kCap * kHeadDim;
    const std::size_t ws_cnt = static_cast<std::size_t>(kSplits) * kSuffix * kQueryHeads * (2 + kHeadDim);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> h_q(q_cnt), h_gate(q_cnt);
    std::vector<__half> h_key(kv_cnt), h_val(kv_cnt);

    for (auto& x : h_q) x = dist(rng);
    for (auto& x : h_gate) x = dist(rng);
    for (auto& x : h_key) x = __float2half(dist(rng));
    for (auto& x : h_val) x = __float2half(dist(rng));

    GpuBuffer d_q(q_cnt * sizeof(float));
    GpuBuffer d_gate(q_cnt * sizeof(float));
    GpuBuffer d_key(kv_cnt * sizeof(__half));
    GpuBuffer d_val(kv_cnt * sizeof(__half));
    GpuBuffer d_ws(ws_cnt * sizeof(float));

    GpuBuffer d_out_p0(q_cnt * sizeof(float));
    GpuBuffer d_out_p1(q_cnt * sizeof(float));
    GpuBuffer d_out_p2(q_cnt * sizeof(float));

    MIINFER_HIP_CHECK(hipMemcpy(d_q.ptr, h_q.data(), q_cnt * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate.ptr, h_gate.data(), q_cnt * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_key.ptr, h_key.data(), kv_cnt * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_val.ptr, h_val.data(), kv_cnt * sizeof(__half), hipMemcpyHostToDevice));

    const std::size_t total_meta = static_cast<std::size_t>(kSplits) * kSuffix * kQueryHeads;
    float* d_p_max = static_cast<float*>(d_ws.ptr);
    float* d_p_sum = d_p_max + total_meta;
    float* d_p_acc = d_p_sum + total_meta;

    dim3 grid1(kQueryHeads / 2, kSuffix, kSplits);
    dim3 block1(64);
    dim3 grid2(kSuffix * kQueryHeads);
    dim3 block2(64);

    hipEvent_t ev0, ev1;
    MIINFER_HIP_CHECK(hipEventCreate(&ev0));
    MIINFER_HIP_CHECK(hipEventCreate(&ev1));

    auto run_benchmark = [&](const std::string& name, auto stage1_kernel, float* out_ptr) {
        std::cout << ">>> Running " << name << "...\n" << std::flush;
        // Warmup
        for (int w = 0; w < 2; ++w) {
            hipLaunchKernelGGL(stage1_kernel, grid1, block1, 0, 0,
                static_cast<const float*>(d_q.ptr),
                static_cast<const __half*>(d_key.ptr),
                static_cast<const __half*>(d_val.ptr),
                d_p_max, d_p_sum, d_p_acc,
                kSuffix, kPrefix, kCap,
                kQueryHeads, kKvHeads, kHeadDim, kScale, kSplits);
            hipLaunchKernelGGL(splitk_suffix_attn_stage2_bench_kernel, grid2, block2, 0, 0,
                d_p_max, d_p_sum, d_p_acc,
                static_cast<const float*>(d_gate.ptr),
                out_ptr,
                kSuffix, kQueryHeads, kHeadDim, kSplits);
        }
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        constexpr int kRounds = 5;
        std::vector<float> single_layer_ms;
        std::vector<float> full_16_layer_ms;

        for (int r = 0; r < kRounds; ++r) {
            // 1 layer timing
            MIINFER_HIP_CHECK(hipEventRecord(ev0));
            hipLaunchKernelGGL(stage1_kernel, grid1, block1, 0, 0,
                static_cast<const float*>(d_q.ptr),
                static_cast<const __half*>(d_key.ptr),
                static_cast<const __half*>(d_val.ptr),
                d_p_max, d_p_sum, d_p_acc,
                kSuffix, kPrefix, kCap,
                kQueryHeads, kKvHeads, kHeadDim, kScale, kSplits);
            hipLaunchKernelGGL(splitk_suffix_attn_stage2_bench_kernel, grid2, block2, 0, 0,
                d_p_max, d_p_sum, d_p_acc,
                static_cast<const float*>(d_gate.ptr),
                out_ptr,
                kSuffix, kQueryHeads, kHeadDim, kSplits);
            MIINFER_HIP_CHECK(hipEventRecord(ev1));
            MIINFER_HIP_CHECK(hipEventSynchronize(ev1));
            single_layer_ms.push_back(elapsed_ms(ev0, ev1));

            // 16 layer timing
            MIINFER_HIP_CHECK(hipEventRecord(ev0));
            for (int l = 0; l < 16; ++l) {
                hipLaunchKernelGGL(stage1_kernel, grid1, block1, 0, 0,
                    static_cast<const float*>(d_q.ptr),
                    static_cast<const __half*>(d_key.ptr),
                    static_cast<const __half*>(d_val.ptr),
                    d_p_max, d_p_sum, d_p_acc,
                    kSuffix, kPrefix, kCap,
                    kQueryHeads, kKvHeads, kHeadDim, kScale, kSplits);
                hipLaunchKernelGGL(splitk_suffix_attn_stage2_bench_kernel, grid2, block2, 0, 0,
                    d_p_max, d_p_sum, d_p_acc,
                    static_cast<const float*>(d_gate.ptr),
                    out_ptr,
                    kSuffix, kQueryHeads, kHeadDim, kSplits);
            }
            MIINFER_HIP_CHECK(hipEventRecord(ev1));
            MIINFER_HIP_CHECK(hipEventSynchronize(ev1));
            full_16_layer_ms.push_back(elapsed_ms(ev0, ev1));
        }

        std::sort(single_layer_ms.begin(), single_layer_ms.end());
        std::sort(full_16_layer_ms.begin(), full_16_layer_ms.end());

        const float l1_med = single_layer_ms[single_layer_ms.size() / 2];
        const float l16_med = full_16_layer_ms[full_16_layer_ms.size() / 2];
        return std::make_pair(l1_med, l16_med);
    };

    const auto [p0_l1, p0_l16] = run_benchmark("Pipeline 0 (Control: U=4 Unpipelined)", splitk_suffix_attn_pipe0_control_kernel<4>, static_cast<float*>(d_out_p0.ptr));
    const auto [p1_l1, p1_l16] = run_benchmark("Pipeline 1 (Double-Buffered U=1)", splitk_suffix_attn_pipe1_double_buf_u1_kernel, static_cast<float*>(d_out_p1.ptr));
    const auto [p2_l1, p2_l16] = run_benchmark("Pipeline 2 (Double-Buffered U=2)", splitk_suffix_attn_pipe2_double_buf_u2_kernel, static_cast<float*>(d_out_p2.ptr));

    std::vector<float> h_out_p0(q_cnt), h_out_p1(q_cnt), h_out_p2(q_cnt);
    MIINFER_HIP_CHECK(hipMemcpy(h_out_p0.data(), d_out_p0.ptr, q_cnt * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(h_out_p1.data(), d_out_p1.ptr, q_cnt * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(h_out_p2.data(), d_out_p2.ptr, q_cnt * sizeof(float), hipMemcpyDeviceToHost));

    double max1 = 0, cos1 = 0, max2 = 0, cos2 = 0;
    compute_diff(h_out_p0, h_out_p1, max1, cos1);
    compute_diff(h_out_p0, h_out_p2, max2, cos2);

    std::cout << "\n===================================================================\n";
    std::cout << "  V2-0020 PIPELINE CANDIDATE RESULTS (P=65536, S=512)\n";
    std::cout << "===================================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Pipeline 0 (Control: U=4 Unpipelined)  : 1-Layer = " << std::setw(6) << p0_l1
              << " ms | 16-Layer = " << std::setw(8) << p0_l16 << " ms | Baseline\n";

    std::cout << "  Pipeline 1 (Double-Buffered U=1)       : 1-Layer = " << std::setw(6) << p1_l1
              << " ms | 16-Layer = " << std::setw(8) << p1_l16 << " ms | Speedup = "
              << p0_l16 / p1_l16 << "x | MaxAbsDiff = " << std::setprecision(6) << max1
              << " | CosSim = " << std::setprecision(8) << cos1 << "\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Pipeline 2 (Double-Buffered U=2)       : 1-Layer = " << std::setw(6) << p2_l1
              << " ms | 16-Layer = " << std::setw(8) << p2_l16 << " ms | Speedup = "
              << p0_l16 / p2_l16 << "x | MaxAbsDiff = " << std::setprecision(6) << max2
              << " | CosSim = " << std::setprecision(8) << cos2 << "\n";
    std::cout << "===================================================================\n";

    MIINFER_HIP_CHECK(hipEventDestroy(ev0));
    MIINFER_HIP_CHECK(hipEventDestroy(ev1));

    return 0;
}
