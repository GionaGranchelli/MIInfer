#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "miinfer/hip_check.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr std::uint32_t kQueryHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr float kScale = 0.0625f; // 1.0 / sqrt(256)

struct AccuracyResult {
    double max_abs_error = 0.0;
    double cosine = 0.0;
    bool is_finite = true;
};

AccuracyResult compute_accuracy(const std::vector<float>& cand, const std::vector<float>& ref) {
    AccuracyResult res;
    double dot = 0.0;
    double norm_cand = 0.0;
    double norm_ref = 0.0;

    for (std::size_t i = 0; i < ref.size(); ++i) {
        float c = cand[i];
        float r = ref[i];
        if (!std::isfinite(c) || !std::isfinite(r)) {
            res.is_finite = false;
        }
        double diff = std::abs(static_cast<double>(c) - static_cast<double>(r));
        if (diff > res.max_abs_error) res.max_abs_error = diff;

        dot += static_cast<double>(c) * static_cast<double>(r);
        norm_cand += static_cast<double>(c) * static_cast<double>(c);
        norm_ref += static_cast<double>(r) * static_cast<double>(r);
    }

    if (norm_cand > 0.0 && norm_ref > 0.0) {
        res.cosine = dot / (std::sqrt(norm_cand) * std::sqrt(norm_ref));
    }
    return res;
}

// -----------------------------------------------------------------------------
// Reference Baseline: V2-0015 Tiled Online Attention
// -----------------------------------------------------------------------------
__global__ void qwen35_tiled_online_attention_batch_f16_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const float* __restrict__ gate,
    float* __restrict__ gated_output,
    std::uint32_t token_count,
    std::uint32_t base_position,
    std::uint32_t cache_capacity,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float scale) {

    const std::uint32_t head = blockIdx.x % query_heads;
    const std::uint32_t token = blockIdx.x / query_heads;
    const std::uint32_t lane = threadIdx.x;

    if (token >= token_count || head >= query_heads || lane >= 64) return;

    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::uint32_t total_length = base_position + token + 1;
    if (total_length > cache_capacity) return;

    const std::size_t q_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim + lane * 4;
    const float4 q_vec = *reinterpret_cast<const float4*>(q + q_base);

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
    const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;

    for (std::uint32_t pos = 0; pos < total_length; ++pos) {
        const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos) * head_dim + lane * 4;
        const uint2 k_raw = *reinterpret_cast<const uint2*>(key_cache + cache_base);
        const auto* k_half = reinterpret_cast<const __half*>(&k_raw);

        float dot = q_vec.x * __half2float(k_half[0]) + q_vec.y * __half2float(k_half[1])
                  + q_vec.z * __half2float(k_half[2]) + q_vec.w * __half2float(k_half[3]);

        #pragma unroll
        for (int offset = 32; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 64);
        }

        const float score = __shfl(dot * scale, 0, 64);
        float alpha = 1.0f;
        float weight = 0.0f;
        if (lane == 0) {
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
        alpha = __shfl(alpha, 0, 64);
        weight = __shfl(weight, 0, 64);

        const uint2 v_raw = *reinterpret_cast<const uint2*>(value_cache + cache_base);
        const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

        acc.x = acc.x * alpha + weight * __half2float(v_half[0]);
        acc.y = acc.y * alpha + weight * __half2float(v_half[1]);
        acc.z = acc.z * alpha + weight * __half2float(v_half[2]);
        acc.w = acc.w * alpha + weight * __half2float(v_half[3]);
    }

    const float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    float4 out = make_float4(
        acc.x * inv_sum, acc.y * inv_sum, acc.z * inv_sum, acc.w * inv_sum);

    if (gate != nullptr) {
        const float4 g_vec = *reinterpret_cast<const float4*>(gate + q_base);
        out.x *= (1.0f / (1.0f + expf(-g_vec.x)));
        out.y *= (1.0f / (1.0f + expf(-g_vec.y)));
        out.z *= (1.0f / (1.0f + expf(-g_vec.z)));
        out.w *= (1.0f / (1.0f + expf(-g_vec.w)));
    }

    *reinterpret_cast<float4*>(gated_output + q_base) = out;
}

// -----------------------------------------------------------------------------
// Stage 2: Shared-Memory Reduction Kernel (V2-0016 Qualified)
// -----------------------------------------------------------------------------
__global__ void qwen35_splitk_suffix_attn_stage2_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_acc,
    const float* __restrict__ gate,
    float* __restrict__ gated_output,
    std::uint32_t token_count,
    std::uint32_t query_heads,
    std::uint32_t head_dim,
    std::uint32_t num_splits) {

    const std::uint32_t pair = blockIdx.x;
    const std::uint32_t token = pair / query_heads;
    const std::uint32_t head = pair % query_heads;
    const std::uint32_t lane = threadIdx.x;

    if (token >= token_count || head >= query_heads || lane >= 64) return;

    __shared__ float s_global_max;
    __shared__ float s_global_sum;

    if (lane == 0) {
        float global_max = -INFINITY;
        for (std::uint32_t s = 0; s < num_splits; ++s) {
            const std::size_t split_idx =
                (static_cast<std::size_t>(s) * token_count + token) * query_heads + head;
            const float m = partial_max[split_idx];
            if (m > global_max) {
                global_max = m;
            }
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

// -----------------------------------------------------------------------------
// Candidate J (V2-0016): Half-Wave Dual-Head (2 Q-Heads / Wave, Split-K)
// -----------------------------------------------------------------------------
template <int UNROLL = 4>
__global__ void qwen35_splitk_suffix_attn_cand_j_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
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

    const std::uint32_t head_pair = blockIdx.x;        // [0, query_heads / 2)
    const std::uint32_t token = blockIdx.y;            // [0, token_count)
    const std::uint32_t split_id = blockIdx.z;         // [0, num_splits)
    const std::uint32_t lane = threadIdx.x;            // [0, 64)

    const std::uint32_t sub_wave = lane / 32;          // 0 or 1
    const std::uint32_t sub_lane = lane % 32;          // [0, 31]
    const std::uint32_t head = head_pair * 2 + sub_wave;
    const std::uint32_t dim_offset = sub_lane * 8;     // 8 elements

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

    std::uint32_t pos = k_start;

    // Main unrolled loop
    for (; pos + UNROLL <= k_end; pos += UNROLL) {
        uint4 k_raw[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos + u) * head_dim + dim_offset;
            k_raw[u] = *reinterpret_cast<const uint4*>(key_cache + cache_base);
        }

        float dots[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const auto* k_half = reinterpret_cast<const __half*>(&k_raw[u]);
            dots[u] = q_vec0.x * __half2float(k_half[0]) + q_vec0.y * __half2float(k_half[1])
                    + q_vec0.z * __half2float(k_half[2]) + q_vec0.w * __half2float(k_half[3])
                    + q_vec1.x * __half2float(k_half[4]) + q_vec1.y * __half2float(k_half[5])
                    + q_vec1.z * __half2float(k_half[6]) + q_vec1.w * __half2float(k_half[7]);
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

        uint4 v_raw[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos + u) * head_dim + dim_offset;
            v_raw[u] = *reinterpret_cast<const uint4*>(value_cache + cache_base);
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

            const auto* v_half = reinterpret_cast<const __half*>(&v_raw[u]);
            acc0.x = acc0.x * alpha + weight * __half2float(v_half[0]);
            acc0.y = acc0.y * alpha + weight * __half2float(v_half[1]);
            acc0.z = acc0.z * alpha + weight * __half2float(v_half[2]);
            acc0.w = acc0.w * alpha + weight * __half2float(v_half[3]);

            acc1.x = acc1.x * alpha + weight * __half2float(v_half[4]);
            acc1.y = acc1.y * alpha + weight * __half2float(v_half[5]);
            acc1.z = acc1.z * alpha + weight * __half2float(v_half[6]);
            acc1.w = acc1.w * alpha + weight * __half2float(v_half[7]);
        }
    }

    // Remainder loop
    for (; pos < k_end; ++pos) {
        const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos) * head_dim + dim_offset;
        const uint4 k_raw = *reinterpret_cast<const uint4*>(key_cache + cache_base);
        const auto* k_half = reinterpret_cast<const __half*>(&k_raw);

        float dot = q_vec0.x * __half2float(k_half[0]) + q_vec0.y * __half2float(k_half[1])
                  + q_vec0.z * __half2float(k_half[2]) + q_vec0.w * __half2float(k_half[3])
                  + q_vec1.x * __half2float(k_half[4]) + q_vec1.y * __half2float(k_half[5])
                  + q_vec1.z * __half2float(k_half[6]) + q_vec1.w * __half2float(k_half[7]);

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

        const uint4 v_raw = *reinterpret_cast<const uint4*>(value_cache + cache_base);
        const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

        acc0.x = acc0.x * alpha + weight * __half2float(v_half[0]);
        acc0.y = acc0.y * alpha + weight * __half2float(v_half[1]);
        acc0.z = acc0.z * alpha + weight * __half2float(v_half[2]);
        acc0.w = acc0.w * alpha + weight * __half2float(v_half[3]);

        acc1.x = acc1.x * alpha + weight * __half2float(v_half[4]);
        acc1.y = acc1.y * alpha + weight * __half2float(v_half[5]);
        acc1.z = acc1.z * alpha + weight * __half2float(v_half[6]);
        acc1.w = acc1.w * alpha + weight * __half2float(v_half[7]);
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
// Candidate 1: 3-Wave 6:1 GQA Workgroup in LDS (192 threads, 4 KiB LDS)
// 1 Workgroup has 3 Wave64s (192 threads).
// Wave 0 handles Q-Heads (6k+0, 6k+1)
// Wave 1 handles Q-Heads (6k+2, 6k+3)
// Wave 2 handles Q-Heads (6k+4, 6k+5)
// All 6 Query heads for KV-head k cooperate in ONE workgroup!
// Threads 0..127 load 4 tokens of K and V (4096 bytes) into LDS in 1 instruction.
// -----------------------------------------------------------------------------
template <int UNROLL = 4>
__global__ void qwen35_splitk_suffix_attn_cand1_3wave_gqa_lds_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
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

    const std::uint32_t kv_head = blockIdx.x;          // [0, kv_heads = 4)
    const std::uint32_t token = blockIdx.y;            // [0, token_count)
    const std::uint32_t split_id = blockIdx.z;         // [0, num_splits)

    const std::uint32_t tid = threadIdx.x;             // [0, 192)
    const std::uint32_t wave_id = tid / 64;            // 0, 1, 2
    const std::uint32_t lane = tid % 64;               // [0, 63]

    const std::uint32_t sub_wave = lane / 32;          // 0 or 1
    const std::uint32_t sub_lane = lane % 32;          // [0, 31]
    const std::uint32_t head = kv_head * 6 + wave_id * 2 + sub_wave;
    const std::uint32_t dim_offset = sub_lane * 8;     // 8 elements

    // 4 KiB Shared Memory: 4 tokens x 256 halfs = 1024 halfs (2048 bytes K, 2048 bytes V)
    __shared__ __half lds_k[UNROLL][256];
    __shared__ __half lds_v[UNROLL][256];

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

    std::uint32_t pos = k_start;

    // Main cooperative unrolled loop
    for (; pos + UNROLL <= k_end; pos += UNROLL) {
        // Step 1: 128 threads load 4 tokens of K and V into LDS cooperatively
        // Total elements = 4 x 256 = 1024 halfs = 128 uint4s (16-bytes each)
        if (tid < 128) {
            const std::uint32_t u_idx = tid / 32;       // 0..3 token in tile
            const std::uint32_t d_idx = (tid % 32) * 8; // 0..255 element in head
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos + u_idx) * head_dim + d_idx;
            const uint4 k_raw = *reinterpret_cast<const uint4*>(key_cache + cache_base);
            const uint4 v_raw = *reinterpret_cast<const uint4*>(value_cache + cache_base);

            *reinterpret_cast<uint4*>(&lds_k[u_idx][d_idx]) = k_raw;
            *reinterpret_cast<uint4*>(&lds_v[u_idx][d_idx]) = v_raw;
        }
        __syncthreads();

        // Step 2: Each of the 3 waves computes dot products and online softmax from LDS
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const uint4 k_raw = *reinterpret_cast<const uint4*>(&lds_k[u][dim_offset]);
            const auto* k_half = reinterpret_cast<const __half*>(&k_raw);

            float dot = q_vec0.x * __half2float(k_half[0]) + q_vec0.y * __half2float(k_half[1])
                      + q_vec0.z * __half2float(k_half[2]) + q_vec0.w * __half2float(k_half[3])
                      + q_vec1.x * __half2float(k_half[4]) + q_vec1.y * __half2float(k_half[5])
                      + q_vec1.z * __half2float(k_half[6]) + q_vec1.w * __half2float(k_half[7]);

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

            const uint4 v_raw = *reinterpret_cast<const uint4*>(&lds_v[u][dim_offset]);
            const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

            acc0.x = acc0.x * alpha + weight * __half2float(v_half[0]);
            acc0.y = acc0.y * alpha + weight * __half2float(v_half[1]);
            acc0.z = acc0.z * alpha + weight * __half2float(v_half[2]);
            acc0.w = acc0.w * alpha + weight * __half2float(v_half[3]);

            acc1.x = acc1.x * alpha + weight * __half2float(v_half[4]);
            acc1.y = acc1.y * alpha + weight * __half2float(v_half[5]);
            acc1.z = acc1.z * alpha + weight * __half2float(v_half[6]);
            acc1.w = acc1.w * alpha + weight * __half2float(v_half[7]);
        }
        __syncthreads();
    }

    // Remainder loop
    for (; pos < k_end; ++pos) {
        if (tid < 32) {
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos) * head_dim + tid * 8;
            *reinterpret_cast<uint4*>(&lds_k[0][tid * 8]) = *reinterpret_cast<const uint4*>(key_cache + cache_base);
            *reinterpret_cast<uint4*>(&lds_v[0][tid * 8]) = *reinterpret_cast<const uint4*>(value_cache + cache_base);
        }
        __syncthreads();

        const uint4 k_raw = *reinterpret_cast<const uint4*>(&lds_k[0][dim_offset]);
        const auto* k_half = reinterpret_cast<const __half*>(&k_raw);

        float dot = q_vec0.x * __half2float(k_half[0]) + q_vec0.y * __half2float(k_half[1])
                  + q_vec0.z * __half2float(k_half[2]) + q_vec0.w * __half2float(k_half[3])
                  + q_vec1.x * __half2float(k_half[4]) + q_vec1.y * __half2float(k_half[5])
                  + q_vec1.z * __half2float(k_half[6]) + q_vec1.w * __half2float(k_half[7]);

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

        const uint4 v_raw = *reinterpret_cast<const uint4*>(&lds_v[0][dim_offset]);
        const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

        acc0.x = acc0.x * alpha + weight * __half2float(v_half[0]);
        acc0.y = acc0.y * alpha + weight * __half2float(v_half[1]);
        acc0.z = acc0.z * alpha + weight * __half2float(v_half[2]);
        acc0.w = acc0.w * alpha + weight * __half2float(v_half[3]);

        acc1.x = acc1.x * alpha + weight * __half2float(v_half[4]);
        acc1.y = acc1.y * alpha + weight * __half2float(v_half[5]);
        acc1.z = acc1.z * alpha + weight * __half2float(v_half[6]);
        acc1.w = acc1.w * alpha + weight * __half2float(v_half[7]);
        __syncthreads();
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
// Candidate 2: 3-Wave 6:1 GQA Double-Buffered Ping-Pong LDS (192 threads, 8 KiB LDS)
// Overlaps global HBM memory fetching with math computation
// -----------------------------------------------------------------------------
template <int UNROLL = 4>
__global__ void qwen35_splitk_suffix_attn_cand2_3wave_double_buf_lds_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
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

    const std::uint32_t kv_head = blockIdx.x;          // [0, kv_heads = 4)
    const std::uint32_t token = blockIdx.y;            // [0, token_count)
    const std::uint32_t split_id = blockIdx.z;         // [0, num_splits)

    const std::uint32_t tid = threadIdx.x;             // [0, 192)
    const std::uint32_t wave_id = tid / 64;            // 0, 1, 2
    const std::uint32_t lane = tid % 64;               // [0, 63]

    const std::uint32_t sub_wave = lane / 32;          // 0 or 1
    const std::uint32_t sub_lane = lane % 32;          // [0, 31]
    const std::uint32_t head = kv_head * 6 + wave_id * 2 + sub_wave;
    const std::uint32_t dim_offset = sub_lane * 8;     // 8 elements

    // 8 KiB Shared Memory: 2 ping-pong buffers x 4 tokens x 256 halfs
    __shared__ __half lds_k[2][UNROLL][256];
    __shared__ __half lds_v[2][UNROLL][256];

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

    std::uint32_t pos = k_start;
    int buf_idx = 0;

    // Prefetch first buffer
    if (pos + UNROLL <= k_end) {
        if (tid < 128) {
            const std::uint32_t u_idx = tid / 32;
            const std::uint32_t d_idx = (tid % 32) * 8;
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos + u_idx) * head_dim + d_idx;
            *reinterpret_cast<uint4*>(&lds_k[0][u_idx][d_idx]) = *reinterpret_cast<const uint4*>(key_cache + cache_base);
            *reinterpret_cast<uint4*>(&lds_v[0][u_idx][d_idx]) = *reinterpret_cast<const uint4*>(value_cache + cache_base);
        }
        __syncthreads();
    }

    // Main pipelined loop
    for (; pos + UNROLL <= k_end; pos += UNROLL) {
        const int curr_buf = buf_idx;
        const int next_buf = 1 - buf_idx;

        // Start prefetching next buffer while computing current buffer
        uint4 next_k, next_v;
        const bool has_next = (pos + 2 * UNROLL <= k_end);
        if (has_next && tid < 128) {
            const std::uint32_t u_idx = tid / 32;
            const std::uint32_t d_idx = (tid % 32) * 8;
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos + UNROLL + u_idx) * head_dim + d_idx;
            next_k = *reinterpret_cast<const uint4*>(key_cache + cache_base);
            next_v = *reinterpret_cast<const uint4*>(value_cache + cache_base);
        }

        // Compute on curr_buf
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const uint4 k_raw = *reinterpret_cast<const uint4*>(&lds_k[curr_buf][u][dim_offset]);
            const auto* k_half = reinterpret_cast<const __half*>(&k_raw);

            float dot = q_vec0.x * __half2float(k_half[0]) + q_vec0.y * __half2float(k_half[1])
                      + q_vec0.z * __half2float(k_half[2]) + q_vec0.w * __half2float(k_half[3])
                      + q_vec1.x * __half2float(k_half[4]) + q_vec1.y * __half2float(k_half[5])
                      + q_vec1.z * __half2float(k_half[6]) + q_vec1.w * __half2float(k_half[7]);

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

            const uint4 v_raw = *reinterpret_cast<const uint4*>(&lds_v[curr_buf][u][dim_offset]);
            const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

            acc0.x = acc0.x * alpha + weight * __half2float(v_half[0]);
            acc0.y = acc0.y * alpha + weight * __half2float(v_half[1]);
            acc0.z = acc0.z * alpha + weight * __half2float(v_half[2]);
            acc0.w = acc0.w * alpha + weight * __half2float(v_half[3]);

            acc1.x = acc1.x * alpha + weight * __half2float(v_half[4]);
            acc1.y = acc1.y * alpha + weight * __half2float(v_half[5]);
            acc1.z = acc1.z * alpha + weight * __half2float(v_half[6]);
            acc1.w = acc1.w * alpha + weight * __half2float(v_half[7]);
        }

        // Store prefetched data into next_buf
        if (has_next && tid < 128) {
            const std::uint32_t u_idx = tid / 32;
            const std::uint32_t d_idx = (tid % 32) * 8;
            *reinterpret_cast<uint4*>(&lds_k[next_buf][u_idx][d_idx]) = next_k;
            *reinterpret_cast<uint4*>(&lds_v[next_buf][u_idx][d_idx]) = next_v;
        }

        buf_idx = next_buf;
        __syncthreads();
    }

    // Remainder loop
    for (; pos < k_end; ++pos) {
        if (tid < 32) {
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos) * head_dim + tid * 8;
            *reinterpret_cast<uint4*>(&lds_k[0][0][tid * 8]) = *reinterpret_cast<const uint4*>(key_cache + cache_base);
            *reinterpret_cast<uint4*>(&lds_v[0][0][tid * 8]) = *reinterpret_cast<const uint4*>(value_cache + cache_base);
        }
        __syncthreads();

        const uint4 k_raw = *reinterpret_cast<const uint4*>(&lds_k[0][0][dim_offset]);
        const auto* k_half = reinterpret_cast<const __half*>(&k_raw);

        float dot = q_vec0.x * __half2float(k_half[0]) + q_vec0.y * __half2float(k_half[1])
                  + q_vec0.z * __half2float(k_half[2]) + q_vec0.w * __half2float(k_half[3])
                  + q_vec1.x * __half2float(k_half[4]) + q_vec1.y * __half2float(k_half[5])
                  + q_vec1.z * __half2float(k_half[6]) + q_vec1.w * __half2float(k_half[7]);

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

        const uint4 v_raw = *reinterpret_cast<const uint4*>(&lds_v[0][0][dim_offset]);
        const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

        acc0.x = acc0.x * alpha + weight * __half2float(v_half[0]);
        acc0.y = acc0.y * alpha + weight * __half2float(v_half[1]);
        acc0.z = acc0.z * alpha + weight * __half2float(v_half[2]);
        acc0.w = acc0.w * alpha + weight * __half2float(v_half[3]);

        acc1.x = acc1.x * alpha + weight * __half2float(v_half[4]);
        acc1.y = acc1.y * alpha + weight * __half2float(v_half[5]);
        acc1.z = acc1.z * alpha + weight * __half2float(v_half[6]);
        acc1.w = acc1.w * alpha + weight * __half2float(v_half[7]);
        __syncthreads();
    }

    if (sub_lane == 0) {
        partial_max[split_head_idx] = running_max;
        partial_sum[split_head_idx] = running_sum;
    }
    const std::size_t acc_base = split_head_idx * head_dim + dim_offset;
    *reinterpret_cast<float4*>(partial_acc + acc_base + 0) = acc0;
    *reinterpret_cast<float4*>(partial_acc + acc_base + 4) = acc1;
}

template <typename T>
struct GpuBuffer {
    T* ptr = nullptr;
    std::size_t count = 0;

    void allocate(std::size_t n) {
        count = n;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
    }
    ~GpuBuffer() {
        if (ptr) (void)hipFree(ptr);
    }
};

} // namespace

int main() {
    constexpr std::size_t P = 65536;
    constexpr std::size_t S = 512;
    constexpr std::size_t capacity = 67000;

    std::cout << "===================================================================\n"
              << "  MIInfer V2-0017: Hierarchical GQA & Query-Tile KV Reuse Bakeoff\n"
              << "  Target: 1 x AMD Instinct MI50 (gfx906, Wave64)\n"
              << "  Workload: P=" << P << ", S=" << S << " (Total Context = " << (P + S) << ")\n"
              << "  Model Geometry: 24 Q-Heads, 4 KV-Heads (6:1 GQA Ratio)\n"
              << "===================================================================\n\n";

    GpuBuffer<float> d_q, d_gate, d_out_ref, d_out_cand;
    GpuBuffer<__half> d_k, d_v;

    d_q.allocate(S * kQueryHeads * kHeadDim);
    d_gate.allocate(S * kQueryHeads * kHeadDim);
    d_out_ref.allocate(S * kQueryHeads * kHeadDim);
    d_out_cand.allocate(S * kQueryHeads * kHeadDim);

    d_k.allocate(kKvHeads * capacity * kHeadDim);
    d_v.allocate(kKvHeads * capacity * kHeadDim);

    // Initialize inputs
    std::vector<float> h_q(d_q.count);
    std::vector<float> h_gate(d_gate.count);
    for (std::size_t i = 0; i < h_q.size(); ++i) {
        h_q[i] = static_cast<float>((i % 100) - 50) * 0.01f;
        h_gate[i] = static_cast<float>((i % 70) - 35) * 0.02f;
    }
    MIINFER_HIP_CHECK(hipMemcpy(d_q.ptr, h_q.data(), d_q.count * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate.ptr, h_gate.data(), d_gate.count * sizeof(float), hipMemcpyHostToDevice));

    std::vector<__half> h_kv(d_k.count);
    for (std::size_t i = 0; i < h_kv.size(); ++i) {
        h_kv[i] = __float2half(static_cast<float>((i % 80) - 40) * 0.005f);
    }
    MIINFER_HIP_CHECK(hipMemcpy(d_k.ptr, h_kv.data(), d_k.count * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_v.ptr, h_kv.data(), d_v.count * sizeof(__half), hipMemcpyHostToDevice));

    // Run Reference
    std::cout << "[INFO] Running Reference Kernel (V2-0015 baseline)...\n";
    hipEvent_t start, stop;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));

    dim3 ref_block(64);
    dim3 ref_grid(static_cast<unsigned int>(S * kQueryHeads));

    hipLaunchKernelGGL(
        qwen35_tiled_online_attention_batch_f16_kernel,
        ref_grid, ref_block, 0, nullptr,
        d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_ref.ptr,
        static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
        static_cast<std::uint32_t>(capacity),
        kQueryHeads, kKvHeads, kHeadDim, kScale);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    MIINFER_HIP_CHECK(hipEventRecord(start, nullptr));
    hipLaunchKernelGGL(
        qwen35_tiled_online_attention_batch_f16_kernel,
        ref_grid, ref_block, 0, nullptr,
        d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_ref.ptr,
        static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
        static_cast<std::uint32_t>(capacity),
        kQueryHeads, kKvHeads, kHeadDim, kScale);
    MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
    MIINFER_HIP_CHECK(hipEventSynchronize(stop));

    float ref_ms = 0.0f;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ref_ms, start, stop));
    std::cout << "  Reference 1-Layer GPU Time: " << ref_ms << " ms (" << (ref_ms * 16.0 / 1000.0) << " s for 16 layers)\n\n";

    std::vector<float> h_ref(d_out_ref.count);
    MIINFER_HIP_CHECK(hipMemcpy(h_ref.data(), d_out_ref.ptr, h_ref.size() * sizeof(float), hipMemcpyDeviceToHost));

    // Allocate Split-K Workspace for maximum splits (256)
    constexpr std::uint32_t kMaxSplits = 256;
    GpuBuffer<float> d_p_max, d_p_sum, d_p_acc;
    d_p_max.allocate(kMaxSplits * S * kQueryHeads);
    d_p_sum.allocate(kMaxSplits * S * kQueryHeads);
    d_p_acc.allocate(kMaxSplits * S * kQueryHeads * kHeadDim);

    struct TestConfig {
        std::string name;
        int cand_id;
        std::uint32_t splits;
    };

    std::vector<TestConfig> configs = {
        {"V2-0016 Baseline (Cand J, Splits=32)", 0, 32},
        {"V2-0016 Baseline (Cand J, Splits=64)", 0, 64},
        {"V2-0016 Baseline (Cand J, Splits=128)", 0, 128},
        {"Cand 1: 3-Wave 6:1 GQA in LDS (Splits=32)", 1, 32},
        {"Cand 1: 3-Wave 6:1 GQA in LDS (Splits=64)", 1, 64},
        {"Cand 1: 3-Wave 6:1 GQA in LDS (Splits=96)", 1, 96},
        {"Cand 1: 3-Wave 6:1 GQA in LDS (Splits=128)", 1, 128},
        {"Cand 1: 3-Wave 6:1 GQA in LDS (Splits=192)", 1, 192},
        {"Cand 1: 3-Wave 6:1 GQA in LDS (Splits=256)", 1, 256},
        {"Cand 2: 3-Wave Double-Buf LDS (Splits=32)", 2, 32},
        {"Cand 2: 3-Wave Double-Buf LDS (Splits=64)", 2, 64},
        {"Cand 2: 3-Wave Double-Buf LDS (Splits=96)", 2, 96},
        {"Cand 2: 3-Wave Double-Buf LDS (Splits=128)", 2, 128},
        {"Cand 2: 3-Wave Double-Buf LDS (Splits=192)", 2, 192},
        {"Cand 2: 3-Wave Double-Buf LDS (Splits=256)", 2, 256},
    };

    std::cout << "========================================================================================================\n"
              << "Configuration                               1-Layer (ms)  16-Layer (s)     Speedup   Max Abs Err    Cosine Sim    Status\n"
              << "--------------------------------------------------------------------------------------------------------\n";

    for (const auto& cfg : configs) {
        auto run_candidate = [&]() {
            if (cfg.cand_id == 0) {
                dim3 block(64);
                dim3 grid(kQueryHeads / 2, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_cand_j_kernel<4>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.cand_id == 1) {
                dim3 block(192); // 3 Wave64s
                dim3 grid(kKvHeads, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_cand1_3wave_gqa_lds_kernel<4>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.cand_id == 2) {
                dim3 block(192); // 3 Wave64s
                dim3 grid(kKvHeads, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_cand2_3wave_double_buf_lds_kernel<4>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            }

            dim3 block2(64);
            dim3 grid2(static_cast<unsigned int>(S * kQueryHeads));
            hipLaunchKernelGGL(
                qwen35_splitk_suffix_attn_stage2_kernel,
                grid2, block2, 0, nullptr,
                d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                d_gate.ptr, d_out_cand.ptr,
                static_cast<std::uint32_t>(S), kQueryHeads, kHeadDim, cfg.splits);
        };

        // Warmup
        run_candidate();
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        // Measure
        MIINFER_HIP_CHECK(hipEventRecord(start, nullptr));
        constexpr int kIters = 3;
        for (int i = 0; i < kIters; ++i) {
            run_candidate();
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));

        float cand_ms = 0.0f;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&cand_ms, start, stop));
        cand_ms /= kIters;

        std::vector<float> h_cand(d_out_cand.count);
        MIINFER_HIP_CHECK(hipMemcpy(h_cand.data(), d_out_cand.ptr, h_cand.size() * sizeof(float), hipMemcpyDeviceToHost));
        auto acc = compute_accuracy(h_cand, h_ref);

        double total_16_s = cand_ms * 16.0 / 1000.0;
        double speedup = ref_ms / cand_ms;
        bool pass = (acc.is_finite && acc.cosine > 0.9999);

        std::cout << std::left << std::setw(42) << cfg.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << cand_ms
                  << std::setw(14) << total_16_s
                  << std::setw(11) << speedup << "x"
                  << std::scientific << std::setprecision(3)
                  << std::setw(14) << acc.max_abs_error
                  << std::fixed << std::setprecision(6)
                  << std::setw(14) << acc.cosine
                  << std::setw(10) << (pass ? "PASS" : "FAIL") << "\n" << std::flush;
    }

    std::cout << "========================================================================================================\n\n";

    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return 0;
}
