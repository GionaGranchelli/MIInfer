#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t count = 0;

    explicit DeviceBuffer(std::size_t n) : count(n) {
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr) (void)hipFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct AccuracyMetrics {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double rms_error = 0.0;
    double rel_rms_error = 0.0;
    double cosine = 0.0;
    bool is_finite = true;
};

AccuracyMetrics compute_accuracy(const std::vector<float>& act, const std::vector<float>& ref) {
    AccuracyMetrics m;
    if (act.size() != ref.size() || act.empty()) return m;

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    double act_sq = 0.0;
    double dot = 0.0;

    for (std::size_t i = 0; i < act.size(); ++i) {
        float a = act[i];
        float r = ref[i];

        if (!std::isfinite(a) || !std::isfinite(r)) {
            m.is_finite = false;
        }

        double diff = std::abs(static_cast<double>(a) - static_cast<double>(r));
        if (diff > m.max_abs_error) m.max_abs_error = diff;
        sum_abs += diff;
        sum_sq += diff * diff;
        ref_sq += static_cast<double>(r) * r;
        act_sq += static_cast<double>(a) * a;
        dot += static_cast<double>(a) * r;
    }

    m.mean_abs_error = sum_abs / act.size();
    m.rms_error = std::sqrt(sum_sq / act.size());
    double ref_rms = std::sqrt(ref_sq / act.size());
    m.rel_rms_error = (ref_rms > 0.0) ? (m.rms_error / ref_rms) : 0.0;

    double denom = std::sqrt(act_sq) * std::sqrt(ref_sq);
    m.cosine = (denom > 0.0) ? (dot / denom) : 0.0;

    return m;
}

// -----------------------------------------------------------------------------
// gfx906 Helper Device Functions
// -----------------------------------------------------------------------------

__device__ __forceinline__ float d_wave_sum(float value) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        value += __shfl_xor(value, offset, 64);
    }
    return value;
}

__device__ __forceinline__ float d_wave_max(float value) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor(value, offset, 64));
    }
    return value;
}

// -----------------------------------------------------------------------------
// Stage 1: Candidate A — Split-K with 1 Wave64 per (token, q_head, split)
// Vectorized global FP16 loads (uint64_t = 4 halfs per lane)
// -----------------------------------------------------------------------------

__global__ void qwen35_splitk_suffix_attn_stage1_1w_kernel(
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

    const std::uint32_t head = blockIdx.x;      // [0, query_heads)
    const std::uint32_t token = blockIdx.y;     // [0, token_count)
    const std::uint32_t split_id = blockIdx.z;  // [0, num_splits)
    const std::uint32_t lane = threadIdx.x;     // [0, 64)

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
        if (lane == 0) {
            partial_max[split_head_idx] = -INFINITY;
            partial_sum[split_head_idx] = 0.0f;
        }
        const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
        *reinterpret_cast<float4*>(partial_acc + acc_base) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const std::size_t q_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim;
    const float4 q_vec = *reinterpret_cast<const float4*>(q + q_base + lane * 4);

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
    const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;

    for (std::uint32_t pos = k_start; pos < k_end; ++pos) {
        const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos) * head_dim;
        const auto k_raw = *reinterpret_cast<const uint64_t*>(key_cache + cache_base + lane * 4);
        const auto* k_half = reinterpret_cast<const __half*>(&k_raw);

        float dot = q_vec.x * __half2float(k_half[0]) + q_vec.y * __half2float(k_half[1])
                  + q_vec.z * __half2float(k_half[2]) + q_vec.w * __half2float(k_half[3]);
        float score = d_wave_sum(dot) * scale;
        score = __shfl(score, 0, 64);

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

        const auto v_raw = *reinterpret_cast<const uint64_t*>(value_cache + cache_base + lane * 4);
        const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

        acc.x = acc.x * alpha + weight * __half2float(v_half[0]);
        acc.y = acc.y * alpha + weight * __half2float(v_half[1]);
        acc.z = acc.z * alpha + weight * __half2float(v_half[2]);
        acc.w = acc.w * alpha + weight * __half2float(v_half[3]);
    }

    if (lane == 0) {
        partial_max[split_head_idx] = running_max;
        partial_sum[split_head_idx] = running_sum;
    }
    const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
    *reinterpret_cast<float4*>(partial_acc + acc_base) = acc;
}

// -----------------------------------------------------------------------------
// Stage 1: Candidate F — Fast Wave64 Split-K Direct Attention (Branchless Softmax)
// -----------------------------------------------------------------------------

__global__ void qwen35_splitk_suffix_attn_stage1_fast_kernel(
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

    const std::uint32_t head = blockIdx.x;      // [0, query_heads)
    const std::uint32_t token = blockIdx.y;     // [0, token_count)
    const std::uint32_t split_id = blockIdx.z;  // [0, num_splits)
    const std::uint32_t lane = threadIdx.x;     // [0, 64)

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
        if (lane == 0) {
            partial_max[split_head_idx] = -INFINITY;
            partial_sum[split_head_idx] = 0.0f;
        }
        const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
        *reinterpret_cast<float4*>(partial_acc + acc_base) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const std::size_t q_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim;
    const float4 q_vec = *reinterpret_cast<const float4*>(q + q_base + lane * 4);

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
    const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;

    for (std::uint32_t pos = k_start; pos < k_end; ++pos) {
        const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos) * head_dim;
        const auto k_raw = *reinterpret_cast<const uint64_t*>(key_cache + cache_base + lane * 4);
        const auto* k_half = reinterpret_cast<const __half*>(&k_raw);

        const float k0 = __half2float(k_half[0]);
        const float k1 = __half2float(k_half[1]);
        const float k2 = __half2float(k_half[2]);
        const float k3 = __half2float(k_half[3]);

        float dot = q_vec.x * k0 + q_vec.y * k1 + q_vec.z * k2 + q_vec.w * k3;
        float score = d_wave_sum(dot) * scale;
        score = __shfl(score, 0, 64);

        float new_max = fmaxf(running_max, score);
        float alpha = (new_max > -INFINITY) ? expf(running_max - new_max) : 1.0f;
        float weight = (new_max > -INFINITY) ? expf(score - new_max) : 0.0f;

        running_sum = running_sum * alpha + weight;
        running_max = new_max;

        const auto v_raw = *reinterpret_cast<const uint64_t*>(value_cache + cache_base + lane * 4);
        const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

        acc.x = acc.x * alpha + weight * __half2float(v_half[0]);
        acc.y = acc.y * alpha + weight * __half2float(v_half[1]);
        acc.z = acc.z * alpha + weight * __half2float(v_half[2]);
        acc.w = acc.w * alpha + weight * __half2float(v_half[3]);
    }

    if (lane == 0) {
        partial_max[split_head_idx] = running_max;
        partial_sum[split_head_idx] = running_sum;
    }
    const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
    *reinterpret_cast<float4*>(partial_acc + acc_base) = acc;
}

// -----------------------------------------------------------------------------
// Stage 1: Candidate B — 6:1 GQA LDS Reuse (384 threads = 6 waves per workgroup)
// 1 Workgroup handles 1 KV head (all 6 Query heads), Bk = 16 tokens
// -----------------------------------------------------------------------------

template <int BK>
__global__ void qwen35_splitk_suffix_attn_stage1_gqa6_lds_kernel(
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

    // 6 waves in workgroup: wave_id in [0, 5], lane in [0, 63]
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t wave_id = tid / 64;
    const std::uint32_t lane = tid % 64;

    const std::uint32_t kv_head = blockIdx.x;    // [0, kv_heads) = [0, 4)
    const std::uint32_t token = blockIdx.y;      // [0, token_count)
    const std::uint32_t split_id = blockIdx.z;   // [0, num_splits)

    if (token >= token_count || kv_head >= kv_heads || wave_id >= 6) return;

    const std::uint32_t q_head = kv_head * 6 + wave_id;
    const std::uint32_t total_length = base_position + token + 1;
    if (total_length > cache_capacity) return;

    const std::uint32_t chunk_size = (total_length + num_splits - 1) / num_splits;
    const std::uint32_t k_start = split_id * chunk_size;
    const std::uint32_t k_end = min(k_start + chunk_size, total_length);

    const std::size_t split_head_idx =
        (static_cast<std::size_t>(split_id) * token_count + token) * query_heads + q_head;

    if (k_start >= total_length) {
        if (lane == 0) {
            partial_max[split_head_idx] = -INFINITY;
            partial_sum[split_head_idx] = 0.0f;
        }
        const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
        *reinterpret_cast<float4*>(partial_acc + acc_base) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // Allocate LDS for BK tokens of K and V
    // BK tokens * 256 halfs = BK * 64 uint64_t
    __shared__ uint64_t s_k[BK * 64];
    __shared__ uint64_t s_v[BK * 64];

    const std::size_t q_base =
        (static_cast<std::size_t>(token) * query_heads + q_head) * head_dim;
    const float4 q_vec = *reinterpret_cast<const float4*>(q + q_base + lane * 4);

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
    const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;

    const std::uint32_t total_u64_per_tile = BK * 64;

    for (std::uint32_t tile_start = k_start; tile_start < k_end; tile_start += BK) {
        const std::uint32_t cur_bk = min(static_cast<std::uint32_t>(BK), k_end - tile_start);

        // Cooperative load of K and V into LDS by all 384 threads (6 waves)
        for (std::uint32_t idx = tid; idx < total_u64_per_tile; idx += 384) {
            const std::uint32_t t_idx = idx / 64;
            const std::uint32_t l_idx = idx % 64;
            const std::uint32_t global_pos = tile_start + t_idx;

            if (t_idx < cur_bk) {
                const std::size_t cache_base = kv_base + static_cast<std::size_t>(global_pos) * head_dim;
                s_k[idx] = *reinterpret_cast<const uint64_t*>(key_cache + cache_base + l_idx * 4);
                s_v[idx] = *reinterpret_cast<const uint64_t*>(value_cache + cache_base + l_idx * 4);
            } else {
                s_k[idx] = 0;
                s_v[idx] = 0;
            }
        }
        __syncthreads();

        // Process all cur_bk tokens in LDS
        for (std::uint32_t j = 0; j < cur_bk; ++j) {
            const auto k_raw = s_k[j * 64 + lane];
            const auto* k_half = reinterpret_cast<const __half*>(&k_raw);

            float dot = q_vec.x * __half2float(k_half[0]) + q_vec.y * __half2float(k_half[1])
                      + q_vec.z * __half2float(k_half[2]) + q_vec.w * __half2float(k_half[3]);
            float score = d_wave_sum(dot) * scale;
            score = __shfl(score, 0, 64);

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

            const auto v_raw = s_v[j * 64 + lane];
            const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

            acc.x = acc.x * alpha + weight * __half2float(v_half[0]);
            acc.y = acc.y * alpha + weight * __half2float(v_half[1]);
            acc.z = acc.z * alpha + weight * __half2float(v_half[2]);
            acc.w = acc.w * alpha + weight * __half2float(v_half[3]);
        }
        __syncthreads();
    }

    if (lane == 0) {
        partial_max[split_head_idx] = running_max;
        partial_sum[split_head_idx] = running_sum;
    }
    const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
    *reinterpret_cast<float4*>(partial_acc + acc_base) = acc;
}

// -----------------------------------------------------------------------------
// Stage 1: Candidate C — Query-Tiled BQ with Split-K
// Reuses 1 loaded K/V vector across BQ query tokens in registers
// -----------------------------------------------------------------------------

template <int BQ>
__global__ void qwen35_splitk_suffix_attn_stage1_bqtiled_kernel(
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

    const std::uint32_t head = blockIdx.x;              // [0, query_heads)
    const std::uint32_t token_block = blockIdx.y;       // [0, (token_count + BQ - 1) / BQ)
    const std::uint32_t split_id = blockIdx.z;          // [0, num_splits)
    const std::uint32_t lane = threadIdx.x;             // [0, 64)

    const std::uint32_t token_base = token_block * BQ;
    if (token_base >= token_count || head >= query_heads || lane >= 64) return;

    const std::uint32_t active_tokens = min(static_cast<std::uint32_t>(BQ), token_count - token_base);
    const std::uint32_t kv_head = head / (query_heads / kv_heads);

    const std::uint32_t max_total_length = base_position + token_base + active_tokens;
    if (max_total_length > cache_capacity) return;

    const std::uint32_t chunk_size = (max_total_length + num_splits - 1) / num_splits;
    const std::uint32_t k_start = split_id * chunk_size;
    const std::uint32_t k_end = min(k_start + chunk_size, max_total_length);

    float4 q_vecs[BQ];
    float4 acc[BQ];
    float running_max[BQ];
    float running_sum[BQ];

    #pragma unroll
    for (int t = 0; t < BQ; ++t) {
        acc[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        running_max[t] = -INFINITY;
        running_sum[t] = 0.0f;
        if (t < active_tokens) {
            const std::uint32_t cur_token = token_base + t;
            const std::size_t q_base =
                (static_cast<std::size_t>(cur_token) * query_heads + head) * head_dim;
            q_vecs[t] = *reinterpret_cast<const float4*>(q + q_base + lane * 4);
        }
    }

    if (k_start < max_total_length) {
        const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
        const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;

        for (std::uint32_t pos = k_start; pos < k_end; ++pos) {
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos) * head_dim;
            const auto k_raw = *reinterpret_cast<const uint64_t*>(key_cache + cache_base + lane * 4);
            const auto* k_half = reinterpret_cast<const __half*>(&k_raw);
            const auto v_raw = *reinterpret_cast<const uint64_t*>(value_cache + cache_base + lane * 4);
            const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

            const float k0 = __half2float(k_half[0]);
            const float k1 = __half2float(k_half[1]);
            const float k2 = __half2float(k_half[2]);
            const float k3 = __half2float(k_half[3]);

            const float v0 = __half2float(v_half[0]);
            const float v1 = __half2float(v_half[1]);
            const float v2 = __half2float(v_half[2]);
            const float v3 = __half2float(v_half[3]);

            #pragma unroll
            for (int t = 0; t < BQ; ++t) {
                if (t < active_tokens) {
                    const std::uint32_t cur_token_causal_len = base_position + token_base + t + 1;
                    if (pos < cur_token_causal_len) {
                        float dot = q_vecs[t].x * k0 + q_vecs[t].y * k1
                                  + q_vecs[t].z * k2 + q_vecs[t].w * k3;
                        float score = d_wave_sum(dot) * scale;
                        score = __shfl(score, 0, 64);

                        float alpha = 1.0f;
                        float weight = 0.0f;
                        if (lane == 0) {
                            if (score > running_max[t]) {
                                alpha = expf(running_max[t] - score);
                                running_sum[t] = running_sum[t] * alpha + 1.0f;
                                running_max[t] = score;
                                weight = 1.0f;
                            } else {
                                weight = expf(score - running_max[t]);
                                running_sum[t] += weight;
                            }
                        }
                        alpha = __shfl(alpha, 0, 64);
                        weight = __shfl(weight, 0, 64);

                        acc[t].x = acc[t].x * alpha + weight * v0;
                        acc[t].y = acc[t].y * alpha + weight * v1;
                        acc[t].z = acc[t].z * alpha + weight * v2;
                        acc[t].w = acc[t].w * alpha + weight * v3;
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < BQ; ++t) {
        if (t < active_tokens) {
            const std::uint32_t cur_token = token_base + t;
            const std::size_t split_head_idx =
                (static_cast<std::size_t>(split_id) * token_count + cur_token) * query_heads + head;

            if (lane == 0) {
                partial_max[split_head_idx] = running_max[t];
                partial_sum[split_head_idx] = running_sum[t];
            }
            const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
            *reinterpret_cast<float4*>(partial_acc + acc_base) = acc[t];
        }
    }
}

// -----------------------------------------------------------------------------
// Stage 1: Candidate D — Wave64 Blocked Tile-64 Split-K Suffix Attention
// Eliminates per-position wave reductions: 10 shuffles per 64 tokens!
// -----------------------------------------------------------------------------

__global__ void qwen35_splitk_suffix_attn_stage1_wave64_blocked_kernel(
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

    const std::uint32_t head = blockIdx.x;      // [0, query_heads)
    const std::uint32_t token = blockIdx.y;     // [0, token_count)
    const std::uint32_t split_id = blockIdx.z;  // [0, num_splits)
    const std::uint32_t lane = threadIdx.x;     // [0, 64)

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
        if (lane == 0) {
            partial_max[split_head_idx] = -INFINITY;
            partial_sum[split_head_idx] = 0.0f;
        }
        const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
        *reinterpret_cast<float4*>(partial_acc + acc_base) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    __shared__ float s_q[256];
    __shared__ float s_P[64];

    const std::size_t q_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim;
    const float4 q_in = *reinterpret_cast<const float4*>(q + q_base + lane * 4);
    s_q[lane * 4 + 0] = q_in.x;
    s_q[lane * 4 + 1] = q_in.y;
    s_q[lane * 4 + 2] = q_in.z;
    s_q[lane * 4 + 3] = q_in.w;
    __syncthreads();

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
    const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;

    for (std::uint32_t tile_start = k_start; tile_start < k_end; tile_start += 64) {
        const std::uint32_t pos = tile_start + lane;

        float dot = 0.0f;
        if (pos < k_end && pos < total_length) {
            const std::size_t k_pos_base = kv_base + static_cast<std::size_t>(pos) * head_dim;
            #pragma unroll 8
            for (int d8 = 0; d8 < 32; ++d8) {
                const auto k_raw = *reinterpret_cast<const uint4*>(key_cache + k_pos_base + d8 * 8);
                const auto* k_half = reinterpret_cast<const __half*>(&k_raw);
                dot += s_q[d8 * 8 + 0] * __half2float(k_half[0])
                     + s_q[d8 * 8 + 1] * __half2float(k_half[1])
                     + s_q[d8 * 8 + 2] * __half2float(k_half[2])
                     + s_q[d8 * 8 + 3] * __half2float(k_half[3])
                     + s_q[d8 * 8 + 4] * __half2float(k_half[4])
                     + s_q[d8 * 8 + 5] * __half2float(k_half[5])
                     + s_q[d8 * 8 + 6] * __half2float(k_half[6])
                     + s_q[d8 * 8 + 7] * __half2float(k_half[7]);
            }
        } else {
            dot = -INFINITY;
        }

        float score = (dot > -1e30f) ? (dot * scale) : -INFINITY;

        float tile_max = d_wave_max(score);
        float p_val = (tile_max > -INFINITY && score > -INFINITY) ? expf(score - tile_max) : 0.0f;
        float tile_sum = d_wave_sum(p_val);

        float new_max = fmaxf(running_max, tile_max);
        float alpha = (new_max > -INFINITY) ? expf(running_max - new_max) : 1.0f;
        float beta  = (new_max > -INFINITY && tile_max > -INFINITY) ? expf(tile_max - new_max) : 0.0f;

        running_sum = running_sum * alpha + tile_sum * beta;
        running_max = new_max;

        acc.x *= alpha;
        acc.y *= alpha;
        acc.z *= alpha;
        acc.w *= alpha;

        s_P[lane] = p_val * beta;
        __syncthreads();

        const std::uint32_t valid_count = min(64u, k_end - tile_start);
        #pragma unroll 4
        for (std::uint32_t j = 0; j < valid_count; ++j) {
            const float weight = s_P[j];
            if (weight > 0.0f) {
                const std::size_t v_base = kv_base + static_cast<std::size_t>(tile_start + j) * head_dim + lane * 4;
                const auto v_raw = *reinterpret_cast<const uint64_t*>(value_cache + v_base);
                const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

                acc.x += weight * __half2float(v_half[0]);
                acc.y += weight * __half2float(v_half[1]);
                acc.z += weight * __half2float(v_half[2]);
                acc.w += weight * __half2float(v_half[3]);
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        partial_max[split_head_idx] = running_max;
        partial_sum[split_head_idx] = running_sum;
    }
    const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
    *reinterpret_cast<float4*>(partial_acc + acc_base) = acc;
}

// -----------------------------------------------------------------------------
// Stage 1: Candidate E — Tile-32 Query-KV LDS Shared Split-K Attention
// 256 threads (4 waves), 32 Query tokens x BK KV tokens per LDS tile
// -----------------------------------------------------------------------------

template <int BK>
__global__ void qwen35_splitk_suffix_attn_stage1_tile32_kernel(
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

    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t wave_id = tid / 64;  // 0..3
    const std::uint32_t lane = tid % 64;     // 0..63

    const std::uint32_t head = blockIdx.x;        // [0, query_heads)
    const std::uint32_t q_block = blockIdx.y;     // [0, (token_count + 31) / 32)
    const std::uint32_t split_id = blockIdx.z;    // [0, num_splits)

    const std::uint32_t token_base = q_block * 32;
    if (token_base >= token_count || head >= query_heads) return;

    const std::uint32_t active_q = min(32u, token_count - token_base);
    const std::uint32_t kv_head = head / (query_heads / kv_heads);

    const std::uint32_t max_total_length = base_position + token_base + active_q;
    if (max_total_length > cache_capacity) return;

    const std::uint32_t chunk_size = (max_total_length + num_splits - 1) / num_splits;
    const std::uint32_t k_start = split_id * chunk_size;
    const std::uint32_t k_end = min(k_start + chunk_size, max_total_length);

    __shared__ uint64_t s_k[BK * 64];
    __shared__ uint64_t s_v[BK * 64];

    const std::uint32_t wave_token_base = token_base + wave_id * 8;
    const std::uint32_t active_wave_tokens =
        (wave_token_base < token_count) ? min(8u, token_count - wave_token_base) : 0u;

    float4 q_vecs[8];
    float4 acc[8];
    float running_max[8];
    float running_sum[8];

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        acc[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        running_max[t] = -INFINITY;
        running_sum[t] = 0.0f;
        if (static_cast<uint32_t>(t) < active_wave_tokens) {
            const std::uint32_t cur_token = wave_token_base + t;
            const std::size_t q_base =
                (static_cast<std::size_t>(cur_token) * query_heads + head) * head_dim;
            q_vecs[t] = *reinterpret_cast<const float4*>(q + q_base + lane * 4);
        }
    }

    if (k_start < max_total_length) {
        const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
        const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;
        constexpr std::uint32_t total_u64 = BK * 64;

        for (std::uint32_t tile_start = k_start; tile_start < k_end; tile_start += BK) {
            const std::uint32_t cur_bk = min(static_cast<std::uint32_t>(BK), k_end - tile_start);

            for (std::uint32_t idx = tid; idx < total_u64; idx += 256) {
                const std::uint32_t t_idx = idx / 64;
                const std::uint32_t l_idx = idx % 64;
                const std::uint32_t global_pos = tile_start + t_idx;

                if (t_idx < cur_bk) {
                    const std::size_t cache_base = kv_base + static_cast<std::size_t>(global_pos) * head_dim;
                    s_k[idx] = *reinterpret_cast<const uint64_t*>(key_cache + cache_base + l_idx * 4);
                    s_v[idx] = *reinterpret_cast<const uint64_t*>(value_cache + cache_base + l_idx * 4);
                } else {
                    s_k[idx] = 0;
                    s_v[idx] = 0;
                }
            }
            __syncthreads();

            for (std::uint32_t j = 0; j < cur_bk; ++j) {
                const uint32_t global_pos = tile_start + j;
                const auto k_raw = s_k[j * 64 + lane];
                const auto* k_half = reinterpret_cast<const __half*>(&k_raw);
                const auto v_raw = s_v[j * 64 + lane];
                const auto* v_half = reinterpret_cast<const __half*>(&v_raw);

                const float k0 = __half2float(k_half[0]);
                const float k1 = __half2float(k_half[1]);
                const float k2 = __half2float(k_half[2]);
                const float k3 = __half2float(k_half[3]);

                const float v0 = __half2float(v_half[0]);
                const float v1 = __half2float(v_half[1]);
                const float v2 = __half2float(v_half[2]);
                const float v3 = __half2float(v_half[3]);

                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (static_cast<uint32_t>(t) < active_wave_tokens) {
                        const std::uint32_t cur_causal_len = base_position + wave_token_base + t + 1;
                        if (global_pos < cur_causal_len) {
                            float dot = q_vecs[t].x * k0 + q_vecs[t].y * k1
                                      + q_vecs[t].z * k2 + q_vecs[t].w * k3;
                            float score = d_wave_sum(dot) * scale;
                            score = __shfl(score, 0, 64);

                            float alpha = 1.0f;
                            float weight = 0.0f;
                            if (lane == 0) {
                                if (score > running_max[t]) {
                                    alpha = expf(running_max[t] - score);
                                    running_sum[t] = running_sum[t] * alpha + 1.0f;
                                    running_max[t] = score;
                                    weight = 1.0f;
                                } else {
                                    weight = expf(score - running_max[t]);
                                    running_sum[t] += weight;
                                }
                            }
                            alpha = __shfl(alpha, 0, 64);
                            weight = __shfl(weight, 0, 64);

                            acc[t].x = acc[t].x * alpha + weight * v0;
                            acc[t].y = acc[t].y * alpha + weight * v1;
                            acc[t].z = acc[t].z * alpha + weight * v2;
                            acc[t].w = acc[t].w * alpha + weight * v3;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (static_cast<uint32_t>(t) < active_wave_tokens) {
            const std::uint32_t cur_token = wave_token_base + t;
            const std::size_t split_head_idx =
                (static_cast<std::size_t>(split_id) * token_count + cur_token) * query_heads + head;

            if (lane == 0) {
                partial_max[split_head_idx] = running_max[t];
                partial_sum[split_head_idx] = running_sum[t];
            }
            const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
            *reinterpret_cast<float4*>(partial_acc + acc_base) = acc[t];
        }
    }
}

// -----------------------------------------------------------------------------
// Stage 2: Fast Online-Softmax Split-K Merge & Gated Projection
// 1 Wave64 per (token, q_head)
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

    // 1. Find global maximum logit across splits: M = max_s(m_s)
    float local_m = -INFINITY;
    if (lane < num_splits) {
        const std::size_t split_head_idx =
            (static_cast<std::size_t>(lane) * token_count + token) * query_heads + head;
        local_m = partial_max[split_head_idx];
    }
    const float global_max = d_wave_max(local_m);

    // 2. Compute rescaled denominator: L = sum_s(exp(m_s - M) * l_s)
    float local_denom = 0.0f;
    if (lane < num_splits && local_m > -INFINITY) {
        const std::size_t split_head_idx =
            (static_cast<std::size_t>(lane) * token_count + token) * query_heads + head;
        const float sum_val = partial_sum[split_head_idx];
        local_denom = expf(local_m - global_max) * sum_val;
    }
    const float total_denom = d_wave_sum(local_denom);
    const float inv_denom = (total_denom > 0.0f) ? (1.0f / total_denom) : 0.0f;

    // 3. Accumulate weighted V outputs across splits for lane's 4 dimensions
    float4 total_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll 4
    for (std::uint32_t s = 0; s < num_splits; ++s) {
        const std::size_t split_head_idx =
            (static_cast<std::size_t>(s) * token_count + token) * query_heads + head;
        const float split_m = partial_max[split_head_idx];

        if (split_m > -INFINITY) {
            const float alpha = expf(split_m - global_max);
            const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
            const float4 p_acc = *reinterpret_cast<const float4*>(partial_acc + acc_base);

            total_acc.x += alpha * p_acc.x;
            total_acc.y += alpha * p_acc.y;
            total_acc.z += alpha * p_acc.z;
            total_acc.w += alpha * p_acc.w;
        }
    }

    // 4. Apply Sigmoid Gate & Store Final Output
    const std::size_t out_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim + lane * 4;
    const float4 gate_vec = *reinterpret_cast<const float4*>(gate + out_base);

    float4 out;
    out.x = (total_acc.x * inv_denom) * (1.0f / (1.0f + expf(-gate_vec.x)));
    out.y = (total_acc.y * inv_denom) * (1.0f / (1.0f + expf(-gate_vec.y)));
    out.z = (total_acc.z * inv_denom) * (1.0f / (1.0f + expf(-gate_vec.z)));
    out.w = (total_acc.w * inv_denom) * (1.0f / (1.0f + expf(-gate_vec.w)));

    *reinterpret_cast<float4*>(gated_output + out_base) = out;
}

// -----------------------------------------------------------------------------
// Stage 1: Candidate G — Wave64 2D Blocked Suffix FlashAttention (Bq=16, Bk=16)
// 1 Wave64 per workgroup processes (Bq=16 query tokens) x (Bk=16 KV tile).
// HBM KV loads are shared across all 16 query tokens via LDS.
// Dot-product across 4 threads per query has only 2 shuffles per KV position.
// P x V GEMM has ZERO shuffles and zero bank conflicts.
// -----------------------------------------------------------------------------

template <int BQ = 16, int BK = 16>
__global__ void qwen35_splitk_suffix_flash_attn_stage1_kernel(
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

    static_assert(BQ == 16, "BQ must be 16 for 4 threads per query");
    static_assert(BK == 16, "BK must be 16");

    const std::uint32_t head = blockIdx.x;      // [0, query_heads)
    const std::uint32_t q_blk = blockIdx.y;     // [0, (token_count + BQ - 1)/BQ)
    const std::uint32_t split_id = blockIdx.z;  // [0, num_splits)
    const std::uint32_t lane = threadIdx.x;     // [0, 64)

    if (head >= query_heads || lane >= 64) return;

    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::uint32_t q_local = lane / 4;    // [0, 16)
    const std::uint32_t t_sub = lane % 4;      // [0, 4)
    const std::uint32_t d_offset = t_sub * 64; // 64 floats/halfs = 128 bytes
    const std::uint32_t q_global = q_blk * BQ + q_local;

    // LDS storage for K and V tiles (16 KB total)
    __shared__ __half lds_k[BK][256];
    __shared__ __half lds_v[BK][256];

    // Total sequence length for split calculation
    const std::uint32_t max_total_len = base_position + token_count;
    if (max_total_len > cache_capacity) return;

    const std::uint32_t chunk_size = (max_total_len + num_splits - 1) / num_splits;
    const std::uint32_t k_start = split_id * chunk_size;
    const std::uint32_t k_end = min(k_start + chunk_size, max_total_len);

    const std::size_t split_head_idx =
        (static_cast<std::size_t>(split_id) * token_count + q_global) * query_heads + head;

    // Load Query into registers (64 floats = 16 float4)
    float4 q_reg[16];
    if (q_global < token_count) {
        const std::size_t q_base =
            (static_cast<std::size_t>(q_global) * query_heads + head) * head_dim + d_offset;
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            q_reg[u] = *reinterpret_cast<const float4*>(q + q_base + u * 4);
        }
    } else {
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            q_reg[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Online Softmax State
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float4 acc[16];
    #pragma unroll
    for (int u = 0; u < 16; ++u) {
        acc[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
    const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;

    const std::uint32_t q_pos = base_position + q_global;

    // Loop unconditionally over KV sequence in chunks of BK = 16
    for (std::uint32_t kb_start = k_start; kb_start < k_end; kb_start += BK) {
        // 1. Cooperative load into LDS: 16 tokens x 256 halfs = 4096 halfs = 512 uint4
        // 64 threads -> 8 uint4 per thread
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t chunk_idx = lane + j * 64; // [0, 512)
            const std::uint32_t k_token_in_tile = chunk_idx / 32; // [0, 16)
            const std::uint32_t d_off = (chunk_idx % 32) * 8; // [0, 256) in steps of 8 halfs
            const std::uint32_t k_global = kb_start + k_token_in_tile;

            if (k_global < k_end) {
                const std::size_t kv_offset =
                    kv_base + static_cast<std::size_t>(k_global) * head_dim + d_off;
                reinterpret_cast<uint4*>(lds_k)[chunk_idx] =
                    *reinterpret_cast<const uint4*>(key_cache + kv_offset);
                reinterpret_cast<uint4*>(lds_v)[chunk_idx] =
                    *reinterpret_cast<const uint4*>(value_cache + kv_offset);
            } else {
                reinterpret_cast<uint4*>(lds_k)[chunk_idx] = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(lds_v)[chunk_idx] = make_uint4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        // 2. Compute QK dot products for this 16-token tile
        float scores[BK];
        #pragma unroll
        for (int k_idx = 0; k_idx < BK; ++k_idx) {
            const std::uint32_t k_pos = kb_start + k_idx;
            float dot = 0.0f;
            const uint4* k_u4 = reinterpret_cast<const uint4*>(&lds_k[k_idx][d_offset]);
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                uint4 raw = k_u4[u];
                __half2 h0 = *reinterpret_cast<const __half2*>(&raw.x);
                __half2 h1 = *reinterpret_cast<const __half2*>(&raw.y);
                __half2 h2 = *reinterpret_cast<const __half2*>(&raw.z);
                __half2 h3 = *reinterpret_cast<const __half2*>(&raw.w);

                dot += q_reg[u * 2 + 0].x * __low2float(h0) +
                       q_reg[u * 2 + 0].y * __high2float(h0) +
                       q_reg[u * 2 + 0].z * __low2float(h1) +
                       q_reg[u * 2 + 0].w * __high2float(h1);

                dot += q_reg[u * 2 + 1].x * __low2float(h2) +
                       q_reg[u * 2 + 1].y * __high2float(h2) +
                       q_reg[u * 2 + 1].z * __low2float(h3) +
                       q_reg[u * 2 + 1].w * __high2float(h3);
            }

            // Unconditional 4-thread reduction for this query's group
            dot += __shfl_xor(dot, 1, 64);
            dot += __shfl_xor(dot, 2, 64);

            const bool valid = (k_pos <= q_pos) && (k_pos < k_end) && (q_global < token_count);
            scores[k_idx] = valid ? (dot * scale) : -INFINITY;
        }

        // 3. Find tile max
        float tile_max = -INFINITY;
        #pragma unroll
        for (int k_idx = 0; k_idx < BK; ++k_idx) {
            tile_max = fmaxf(tile_max, scores[k_idx]);
        }

        // 4. Online softmax update
        if (tile_max > -INFINITY) {
            const float new_max = fmaxf(row_max, tile_max);
            const float exp_scale = (row_max > -INFINITY) ? expf(row_max - new_max) : 0.0f;
            row_sum *= exp_scale;

            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                acc[u].x *= exp_scale;
                acc[u].y *= exp_scale;
                acc[u].z *= exp_scale;
                acc[u].w *= exp_scale;
            }

            #pragma unroll
            for (int k_idx = 0; k_idx < BK; ++k_idx) {
                if (scores[k_idx] > -INFINITY) {
                    const float p = expf(scores[k_idx] - new_max);
                    row_sum += p;

                    const uint4* v_u4 = reinterpret_cast<const uint4*>(&lds_v[k_idx][d_offset]);
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        uint4 raw = v_u4[u];
                        __half2 h0 = *reinterpret_cast<const __half2*>(&raw.x);
                        __half2 h1 = *reinterpret_cast<const __half2*>(&raw.y);
                        __half2 h2 = *reinterpret_cast<const __half2*>(&raw.z);
                        __half2 h3 = *reinterpret_cast<const __half2*>(&raw.w);

                        acc[u * 2 + 0].x += p * __low2float(h0);
                        acc[u * 2 + 0].y += p * __high2float(h0);
                        acc[u * 2 + 0].z += p * __low2float(h1);
                        acc[u * 2 + 0].w += p * __high2float(h1);

                        acc[u * 2 + 1].x += p * __low2float(h2);
                        acc[u * 2 + 1].y += p * __high2float(h2);
                        acc[u * 2 + 1].z += p * __low2float(h3);
                        acc[u * 2 + 1].w += p * __high2float(h3);
                    }
                }
            }
            row_max = new_max;
        }

        __syncthreads();
    }

    // Write stage 1 results to global memory
    if (q_global < token_count) {
        if (t_sub == 0) {
            partial_max[split_head_idx] = row_max;
            partial_sum[split_head_idx] = row_sum;
        }

        const std::size_t acc_base = split_head_idx * head_dim + d_offset;
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            *reinterpret_cast<float4*>(partial_acc + acc_base + u * 4) = acc[u];
        }
    }
}

// -----------------------------------------------------------------------------
// Stage 1: Candidate H — Wave64 Register Blocked (BQ=4, BK=8)
// 1 Wave64 per workgroup processes BQ=4 query tokens.
// Loads BK KV vectors into registers (vectorized uint64_t = 4 halfs per lane).
// Computes BQ x BK dot products in registers with batch wave reductions.
// ZERO LDS, ZERO barrier synchronizations, ZERO shuffles in Softmax/PV!
// -----------------------------------------------------------------------------

template <int BQ = 4, int BK = 8>
__global__ void qwen35_splitk_suffix_attn_stage1_regblocked_kernel(
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

    const std::uint32_t head = blockIdx.x;              // [0, query_heads)
    const std::uint32_t token_block = blockIdx.y;       // [0, (token_count + BQ - 1) / BQ)
    const std::uint32_t split_id = blockIdx.z;          // [0, num_splits)
    const std::uint32_t lane = threadIdx.x;             // [0, 64)

    const std::uint32_t token_base = token_block * BQ;
    if (token_base >= token_count || head >= query_heads || lane >= 64) return;

    const std::uint32_t active_tokens = min(static_cast<std::uint32_t>(BQ), token_count - token_base);
    const std::uint32_t kv_head = head / (query_heads / kv_heads);

    const std::uint32_t max_total_length = base_position + token_base + active_tokens;
    if (max_total_length > cache_capacity) return;

    const std::uint32_t chunk_size = (max_total_length + num_splits - 1) / num_splits;
    const std::uint32_t k_start = split_id * chunk_size;
    const std::uint32_t k_end = min(k_start + chunk_size, max_total_length);

    float4 q_vecs[BQ];
    float4 acc[BQ];
    float running_max[BQ];
    float running_sum[BQ];

    #pragma unroll
    for (int t = 0; t < BQ; ++t) {
        acc[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        running_max[t] = -INFINITY;
        running_sum[t] = 0.0f;
        if (t < active_tokens) {
            const std::uint32_t cur_token = token_base + t;
            const std::size_t q_base =
                (static_cast<std::size_t>(cur_token) * query_heads + head) * head_dim;
            q_vecs[t] = *reinterpret_cast<const float4*>(q + q_base + lane * 4);
        }
    }

    if (k_start < max_total_length) {
        const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
        const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;

        for (std::uint32_t kb_start = k_start; kb_start < k_end; kb_start += BK) {
            // 1. Vectorized load BK key vectors into registers
            uint64_t k_raw[BK];
            #pragma unroll
            for (int k_idx = 0; k_idx < BK; ++k_idx) {
                const std::uint32_t pos = kb_start + k_idx;
                if (pos < k_end) {
                    const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos) * head_dim;
                    k_raw[k_idx] = *reinterpret_cast<const uint64_t*>(key_cache + cache_base + lane * 4);
                } else {
                    k_raw[k_idx] = 0;
                }
            }

            // 2. Compute local dot products for (BQ x BK)
            float dots[BQ][BK];
            #pragma unroll
            for (int k_idx = 0; k_idx < BK; ++k_idx) {
                const auto* k_half = reinterpret_cast<const __half*>(&k_raw[k_idx]);
                const float k0 = __half2float(k_half[0]);
                const float k1 = __half2float(k_half[1]);
                const float k2 = __half2float(k_half[2]);
                const float k3 = __half2float(k_half[3]);

                #pragma unroll
                for (int t = 0; t < BQ; ++t) {
                    dots[t][k_idx] = q_vecs[t].x * k0 + q_vecs[t].y * k1
                                   + q_vecs[t].z * k2 + q_vecs[t].w * k3;
                }
            }

            // 3. Vectorized wave reductions for all (BQ x BK) scores
            float scores[BQ][BK];
            #pragma unroll
            for (int t = 0; t < BQ; ++t) {
                #pragma unroll
                for (int k_idx = 0; k_idx < BK; ++k_idx) {
                    const std::uint32_t pos = kb_start + k_idx;
                    const std::uint32_t cur_token_causal_len = base_position + token_base + t + 1;
                    float score = d_wave_sum(dots[t][k_idx]) * scale;
                    scores[t][k_idx] = (pos < cur_token_causal_len && pos < k_end && t < active_tokens) ? score : -INFINITY;
                }
            }

            // 4. Vectorized load BK value vectors into registers
            uint64_t v_raw[BK];
            #pragma unroll
            for (int k_idx = 0; k_idx < BK; ++k_idx) {
                const std::uint32_t pos = kb_start + k_idx;
                if (pos < k_end) {
                    const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos) * head_dim;
                    v_raw[k_idx] = *reinterpret_cast<const uint64_t*>(value_cache + cache_base + lane * 4);
                } else {
                    v_raw[k_idx] = 0;
                }
            }

            // 5. Online Softmax update & P x V accumulation (ZERO cross-lane shuffles!)
            #pragma unroll
            for (int k_idx = 0; k_idx < BK; ++k_idx) {
                const auto* v_half = reinterpret_cast<const __half*>(&v_raw[k_idx]);
                const float v0 = __half2float(v_half[0]);
                const float v1 = __half2float(v_half[1]);
                const float v2 = __half2float(v_half[2]);
                const float v3 = __half2float(v_half[3]);

                #pragma unroll
                for (int t = 0; t < BQ; ++t) {
                    const float s = scores[t][k_idx];
                    if (s > -INFINITY) {
                        if (s > running_max[t]) {
                            const float alpha = expf(running_max[t] - s);
                            running_sum[t] = running_sum[t] * alpha + 1.0f;
                            running_max[t] = s;
                            acc[t].x = acc[t].x * alpha + v0;
                            acc[t].y = acc[t].y * alpha + v1;
                            acc[t].z = acc[t].z * alpha + v2;
                            acc[t].w = acc[t].w * alpha + v3;
                        } else {
                            const float weight = expf(s - running_max[t]);
                            running_sum[t] += weight;
                            acc[t].x += weight * v0;
                            acc[t].y += weight * v1;
                            acc[t].z += weight * v2;
                            acc[t].w += weight * v3;
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < BQ; ++t) {
        if (t < active_tokens) {
            const std::uint32_t cur_token = token_base + t;
            const std::size_t split_head_idx =
                (static_cast<std::size_t>(split_id) * token_count + cur_token) * query_heads + head;

            if (lane == 0) {
                partial_max[split_head_idx] = running_max[t];
                partial_sum[split_head_idx] = running_sum[t];
            }
            const std::size_t acc_base = split_head_idx * head_dim + lane * 4;
            *reinterpret_cast<float4*>(partial_acc + acc_base) = acc[t];
        }
    }
}

// -----------------------------------------------------------------------------
// Stage 1: Candidate I — Half-Wave Dual-Head (32-lane, 16-byte uint4 vectorized)
// 1 Wave64 handles 2 query heads concurrently (lanes 0..31 = Head 0, lanes 32..63 = Head 1).
// Each lane processes 8 elements (16 bytes half, 32 bytes float).
// 100% 16-byte aligned memory transactions.
// Coalesced K/V cache loads across the two heads in the same wave!
// Extremely low VGPR footprint (26 VGPRs -> Maximum Occupancy).
// -----------------------------------------------------------------------------

__global__ void qwen35_splitk_suffix_attn_stage1_halfwave_kernel(
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

    for (std::uint32_t pos = k_start; pos < k_end; ++pos) {
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
        float score = dot * scale;
        score = __shfl(score, sub_wave * 32, 64);

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
// Stage 1: Candidate J — Half-Wave Dual-Head with 4x Loop Unrolling & Load Dual-Issue
// Unrolls pos loop by 4 tokens to maximize memory-level parallelism (MLP).
// Issues 4 x 16-byte global loads in flight before consuming data.
// Extremely high instruction density and latency hiding on MI50.
// -----------------------------------------------------------------------------

template <int UNROLL = 4>
__global__ void qwen35_splitk_suffix_attn_stage1_halfwave_unrolled_kernel(
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
        // 1. Issue UNROLL key loads in flight
        uint4 k_raw[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos + u) * head_dim + dim_offset;
            k_raw[u] = *reinterpret_cast<const uint4*>(key_cache + cache_base);
        }

        // 2. Compute dot products
        float dots[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const auto* k_half = reinterpret_cast<const __half*>(&k_raw[u]);
            dots[u] = q_vec0.x * __half2float(k_half[0]) + q_vec0.y * __half2float(k_half[1])
                    + q_vec0.z * __half2float(k_half[2]) + q_vec0.w * __half2float(k_half[3])
                    + q_vec1.x * __half2float(k_half[4]) + q_vec1.y * __half2float(k_half[5])
                    + q_vec1.z * __half2float(k_half[6]) + q_vec1.w * __half2float(k_half[7]);
        }

        // 3. Half-wave reductions
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

        // 4. Issue UNROLL value loads in flight
        uint4 v_raw[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos + u) * head_dim + dim_offset;
            v_raw[u] = *reinterpret_cast<const uint4*>(value_cache + cache_base);
        }

        // 5. Softmax updates and PV accumulations
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

    // Scalar cleanup tail
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
        float score = dot * scale;
        score = __shfl(score, sub_wave * 32, 64);

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
// Stage 1: Candidate K — Half-Wave Multi-Query (4 Query Tokens per Wave64)
// Subwave 0 (lanes 0..31) processes Tokens (T0, T1).
// Subwave 1 (lanes 32..63) processes Tokens (T2, T3).
// Both subwaves read the EXACT SAME K and V vectors from global memory (4:1 KV reuse).
// Cuts HBM traffic by 4x.
// 100% 16-byte uint4 vectorized loads.
// Low VGPR count (~40 VGPRs -> Maximum Occupancy).
// -----------------------------------------------------------------------------

template <int UNROLL = 2>
__global__ void qwen35_splitk_suffix_attn_stage1_multiquery_kernel(
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

    const std::uint32_t head = blockIdx.x;              // [0, query_heads)
    const std::uint32_t token_block = blockIdx.y;       // [0, (token_count + 3) / 4)
    const std::uint32_t split_id = blockIdx.z;          // [0, num_splits)
    const std::uint32_t lane = threadIdx.x;             // [0, 64)

    const std::uint32_t sub_wave = lane / 32;          // 0 or 1
    const std::uint32_t sub_lane = lane % 32;          // [0, 31]
    const std::uint32_t dim_offset = sub_lane * 8;     // 8 elements

    const std::uint32_t t0 = token_block * 4 + sub_wave * 2 + 0;
    const std::uint32_t t1 = token_block * 4 + sub_wave * 2 + 1;

    if (head >= query_heads || lane >= 64) return;

    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::uint32_t max_block_token = min(token_block * 4 + 4, token_count);
    const std::uint32_t max_total_length = base_position + max_block_token;
    if (max_total_length > cache_capacity) return;

    const std::uint32_t chunk_size = (max_total_length + num_splits - 1) / num_splits;
    const std::uint32_t k_start = split_id * chunk_size;
    const std::uint32_t k_end = min(k_start + chunk_size, max_total_length);

    // Load Q vectors for t0 and t1
    float4 q0_vec0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q0_vec1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1_vec0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1_vec1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (t0 < token_count) {
        const std::size_t q_base0 =
            (static_cast<std::size_t>(t0) * query_heads + head) * head_dim + dim_offset;
        q0_vec0 = *reinterpret_cast<const float4*>(q + q_base0 + 0);
        q0_vec1 = *reinterpret_cast<const float4*>(q + q_base0 + 4);
    }
    if (t1 < token_count) {
        const std::size_t q_base1 =
            (static_cast<std::size_t>(t1) * query_heads + head) * head_dim + dim_offset;
        q1_vec0 = *reinterpret_cast<const float4*>(q + q_base1 + 0);
        q1_vec1 = *reinterpret_cast<const float4*>(q + q_base1 + 4);
    }

    float4 acc0_0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc0_1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max0 = -INFINITY;
    float running_sum0 = 0.0f;

    float4 acc1_0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc1_1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max1 = -INFINITY;
    float running_sum1 = 0.0f;

    const std::size_t kv_head_stride = static_cast<std::size_t>(cache_capacity) * head_dim;
    const std::size_t kv_base = static_cast<std::size_t>(kv_head) * kv_head_stride;

    if (k_start < max_total_length) {
        const std::uint32_t causal_len0 = (t0 < token_count) ? (base_position + t0 + 1) : 0;
        const std::uint32_t causal_len1 = (t1 < token_count) ? (base_position + t1 + 1) : 0;

        std::uint32_t pos = k_start;
        for (; pos + UNROLL <= k_end; pos += UNROLL) {
            uint4 k_raw[UNROLL];
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos + u) * head_dim + dim_offset;
                k_raw[u] = *reinterpret_cast<const uint4*>(key_cache + cache_base);
            }

            float dots0[UNROLL];
            float dots1[UNROLL];
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const auto* k_half = reinterpret_cast<const __half*>(&k_raw[u]);
                const float k0 = __half2float(k_half[0]), k1 = __half2float(k_half[1]);
                const float k2 = __half2float(k_half[2]), k3 = __half2float(k_half[3]);
                const float k4 = __half2float(k_half[4]), k5 = __half2float(k_half[5]);
                const float k6 = __half2float(k_half[6]), k7 = __half2float(k_half[7]);

                dots0[u] = q0_vec0.x * k0 + q0_vec0.y * k1 + q0_vec0.z * k2 + q0_vec0.w * k3
                         + q0_vec1.x * k4 + q0_vec1.y * k5 + q0_vec1.z * k6 + q0_vec1.w * k7;

                dots1[u] = q1_vec0.x * k0 + q1_vec0.y * k1 + q1_vec0.z * k2 + q1_vec0.w * k3
                         + q1_vec1.x * k4 + q1_vec1.y * k5 + q1_vec1.z * k6 + q1_vec1.w * k7;
            }

            float scores0[UNROLL];
            float scores1[UNROLL];
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                float d0 = dots0[u];
                float d1 = dots1[u];
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    d0 += __shfl_xor(d0, offset, 32);
                    d1 += __shfl_xor(d1, offset, 32);
                }
                const std::uint32_t p = pos + u;
                scores0[u] = (p < causal_len0) ? (__shfl(d0 * scale, sub_wave * 32, 64)) : -INFINITY;
                scores1[u] = (p < causal_len1) ? (__shfl(d1 * scale, sub_wave * 32, 64)) : -INFINITY;
            }

            uint4 v_raw[UNROLL];
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos + u) * head_dim + dim_offset;
                v_raw[u] = *reinterpret_cast<const uint4*>(value_cache + cache_base);
            }

            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const auto* v_half = reinterpret_cast<const __half*>(&v_raw[u]);
                const float v0 = __half2float(v_half[0]), v1 = __half2float(v_half[1]);
                const float v2 = __half2float(v_half[2]), v3 = __half2float(v_half[3]);
                const float v4 = __half2float(v_half[4]), v5 = __half2float(v_half[5]);
                const float v6 = __half2float(v_half[6]), v7 = __half2float(v_half[7]);

                // T0 update
                const float s0 = scores0[u];
                if (s0 > -INFINITY) {
                    float a0 = 1.0f, w0 = 0.0f;
                    if (sub_lane == 0) {
                        if (s0 > running_max0) {
                            a0 = expf(running_max0 - s0);
                            running_sum0 = running_sum0 * a0 + 1.0f;
                            running_max0 = s0;
                            w0 = 1.0f;
                        } else {
                            w0 = expf(s0 - running_max0);
                            running_sum0 += w0;
                        }
                    }
                    a0 = __shfl(a0, sub_wave * 32, 64);
                    w0 = __shfl(w0, sub_wave * 32, 64);

                    acc0_0.x = acc0_0.x * a0 + w0 * v0;
                    acc0_0.y = acc0_0.y * a0 + w0 * v1;
                    acc0_0.z = acc0_0.z * a0 + w0 * v2;
                    acc0_0.w = acc0_0.w * a0 + w0 * v3;
                    acc0_1.x = acc0_1.x * a0 + w0 * v4;
                    acc0_1.y = acc0_1.y * a0 + w0 * v5;
                    acc0_1.z = acc0_1.z * a0 + w0 * v6;
                    acc0_1.w = acc0_1.w * a0 + w0 * v7;
                }

                // T1 update
                const float s1 = scores1[u];
                if (s1 > -INFINITY) {
                    float a1 = 1.0f, w1 = 0.0f;
                    if (sub_lane == 0) {
                        if (s1 > running_max1) {
                            a1 = expf(running_max1 - s1);
                            running_sum1 = running_sum1 * a1 + 1.0f;
                            running_max1 = s1;
                            w1 = 1.0f;
                        } else {
                            w1 = expf(s1 - running_max1);
                            running_sum1 += w1;
                        }
                    }
                    a1 = __shfl(a1, sub_wave * 32, 64);
                    w1 = __shfl(w1, sub_wave * 32, 64);

                    acc1_0.x = acc1_0.x * a1 + w1 * v0;
                    acc1_0.y = acc1_0.y * a1 + w1 * v1;
                    acc1_0.z = acc1_0.z * a1 + w1 * v2;
                    acc1_0.w = acc1_0.w * a1 + w1 * v3;
                    acc1_1.x = acc1_1.x * a1 + w1 * v4;
                    acc1_1.y = acc1_1.y * a1 + w1 * v5;
                    acc1_1.z = acc1_1.z * a1 + w1 * v6;
                    acc1_1.w = acc1_1.w * a1 + w1 * v7;
                }
            }
        }

        // Tail loop
        for (; pos < k_end; ++pos) {
            const std::size_t cache_base = kv_base + static_cast<std::size_t>(pos) * head_dim + dim_offset;
            const uint4 k_raw = *reinterpret_cast<const uint4*>(key_cache + cache_base);
            const auto* k_half = reinterpret_cast<const __half*>(&k_raw);
            const float k0 = __half2float(k_half[0]), k1 = __half2float(k_half[1]);
            const float k2 = __half2float(k_half[2]), k3 = __half2float(k_half[3]);
            const float k4 = __half2float(k_half[4]), k5 = __half2float(k_half[5]);
            const float k6 = __half2float(k_half[6]), k7 = __half2float(k_half[7]);

            float d0 = q0_vec0.x * k0 + q0_vec0.y * k1 + q0_vec0.z * k2 + q0_vec0.w * k3
                     + q0_vec1.x * k4 + q0_vec1.y * k5 + q0_vec1.z * k6 + q0_vec1.w * k7;
            float d1 = q1_vec0.x * k0 + q1_vec0.y * k1 + q1_vec0.z * k2 + q1_vec0.w * k3
                     + q1_vec1.x * k4 + q1_vec1.y * k5 + q1_vec1.z * k6 + q1_vec1.w * k7;

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                d0 += __shfl_xor(d0, offset, 32);
                d1 += __shfl_xor(d1, offset, 32);
            }
            const float s0 = (pos < causal_len0) ? (__shfl(d0 * scale, sub_wave * 32, 64)) : -INFINITY;
            const float s1 = (pos < causal_len1) ? (__shfl(d1 * scale, sub_wave * 32, 64)) : -INFINITY;

            const uint4 v_raw = *reinterpret_cast<const uint4*>(value_cache + cache_base);
            const auto* v_half = reinterpret_cast<const __half*>(&v_raw);
            const float v0 = __half2float(v_half[0]), v1 = __half2float(v_half[1]);
            const float v2 = __half2float(v_half[2]), v3 = __half2float(v_half[3]);
            const float v4 = __half2float(v_half[4]), v5 = __half2float(v_half[5]);
            const float v6 = __half2float(v_half[6]), v7 = __half2float(v_half[7]);

            if (s0 > -INFINITY) {
                float a0 = 1.0f, w0 = 0.0f;
                if (sub_lane == 0) {
                    if (s0 > running_max0) {
                        a0 = expf(running_max0 - s0);
                        running_sum0 = running_sum0 * a0 + 1.0f;
                        running_max0 = s0;
                        w0 = 1.0f;
                    } else {
                        w0 = expf(s0 - running_max0);
                        running_sum0 += w0;
                    }
                }
                a0 = __shfl(a0, sub_wave * 32, 64);
                w0 = __shfl(w0, sub_wave * 32, 64);

                acc0_0.x = acc0_0.x * a0 + w0 * v0;
                acc0_0.y = acc0_0.y * a0 + w0 * v1;
                acc0_0.z = acc0_0.z * a0 + w0 * v2;
                acc0_0.w = acc0_0.w * a0 + w0 * v3;
                acc0_1.x = acc0_1.x * a0 + w0 * v4;
                acc0_1.y = acc0_1.y * a0 + w0 * v5;
                acc0_1.z = acc0_1.z * a0 + w0 * v6;
                acc0_1.w = acc0_1.w * a0 + w0 * v7;
            }

            if (s1 > -INFINITY) {
                float a1 = 1.0f, w1 = 0.0f;
                if (sub_lane == 0) {
                    if (s1 > running_max1) {
                        a1 = expf(running_max1 - s1);
                        running_sum1 = running_sum1 * a1 + 1.0f;
                        running_max1 = s1;
                        w1 = 1.0f;
                    } else {
                        w1 = expf(s1 - running_max1);
                        running_sum1 += w1;
                    }
                }
                a1 = __shfl(a1, sub_wave * 32, 64);
                w1 = __shfl(w1, sub_wave * 32, 64);

                acc1_0.x = acc1_0.x * a1 + w1 * v0;
                acc1_0.y = acc1_0.y * a1 + w1 * v1;
                acc1_0.z = acc1_0.z * a1 + w1 * v2;
                acc1_0.w = acc1_0.w * a1 + w1 * v3;
                acc1_1.x = acc1_1.x * a1 + w1 * v4;
                acc1_1.y = acc1_1.y * a1 + w1 * v5;
                acc1_1.z = acc1_1.z * a1 + w1 * v6;
                acc1_1.w = acc1_1.w * a1 + w1 * v7;
            }
        }
    }

    if (t0 < token_count) {
        const std::size_t split_head_idx0 =
            (static_cast<std::size_t>(split_id) * token_count + t0) * query_heads + head;
        if (sub_lane == 0) {
            partial_max[split_head_idx0] = running_max0;
            partial_sum[split_head_idx0] = running_sum0;
        }
        const std::size_t acc_base0 = split_head_idx0 * head_dim + dim_offset;
        *reinterpret_cast<float4*>(partial_acc + acc_base0 + 0) = acc0_0;
        *reinterpret_cast<float4*>(partial_acc + acc_base0 + 4) = acc0_1;
    }

    if (t1 < token_count) {
        const std::size_t split_head_idx1 =
            (static_cast<std::size_t>(split_id) * token_count + t1) * query_heads + head;
        if (sub_lane == 0) {
            partial_max[split_head_idx1] = running_max1;
            partial_sum[split_head_idx1] = running_sum1;
        }
        const std::size_t acc_base1 = split_head_idx1 * head_dim + dim_offset;
        *reinterpret_cast<float4*>(partial_acc + acc_base1 + 0) = acc1_0;
        *reinterpret_cast<float4*>(partial_acc + acc_base1 + 4) = acc1_1;
    }
}

} // namespace

int main() {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0016: Specialized Wave64 Split-K Suffix Attention Bakeoff\n";
    std::cout << "  Target: 1 x AMD Instinct MI50 (gfx906, Wave64)\n";
    std::cout << "  Geometry: 24 Q Heads, 4 KV Heads (6:1 GQA), Head Dim 256\n";
    std::cout << "===================================================================\n\n";

    constexpr std::uint32_t kQueryHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kHeadDim = 256;
    constexpr float kScale = 1.0F / 16.0F; // 1.0 / sqrt(256)

    const std::size_t P = 65536;
    const std::size_t S = 512;
    const std::size_t capacity = P + S + 1024;

    std::cout << "[INFO] Allocating GPU memory for P=" << P << ", S=" << S
              << " (Total Context = " << P + S << ")...\n";

    DeviceBuffer<float> d_q(S * kQueryHeads * kHeadDim);
    DeviceBuffer<float> d_gate(S * kQueryHeads * kHeadDim);
    DeviceBuffer<__half> d_k(kKvHeads * capacity * kHeadDim);
    DeviceBuffer<__half> d_v(kKvHeads * capacity * kHeadDim);
    DeviceBuffer<float> d_out_ref(S * kQueryHeads * kHeadDim);
    DeviceBuffer<float> d_out_cand(S * kQueryHeads * kHeadDim);

    // Fill synthetic data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.05F, 0.05F);

    std::vector<float> h_q(S * kQueryHeads * kHeadDim);
    std::vector<float> h_gate(S * kQueryHeads * kHeadDim);
    for (auto& v : h_q) v = dist(rng);
    for (auto& v : h_gate) v = dist(rng);

    std::vector<__half> h_k(kKvHeads * capacity * kHeadDim);
    std::vector<__half> h_v(kKvHeads * capacity * kHeadDim);
    for (auto& v : h_k) v = __float2half(dist(rng));
    for (auto& v : h_v) v = __float2half(dist(rng));

    MIINFER_HIP_CHECK(hipMemcpy(d_q.ptr, h_q.data(), h_q.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate.ptr, h_gate.data(), h_gate.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_k.ptr, h_k.data(), h_k.size() * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_v.ptr, h_v.data(), h_v.size() * sizeof(__half), hipMemcpyHostToDevice));

    // Allocate Split-K Workspace (Up to 256 splits)
    constexpr std::uint32_t kMaxSplits = 256;
    DeviceBuffer<float> d_p_max(kMaxSplits * S * kQueryHeads);
    DeviceBuffer<float> d_p_sum(kMaxSplits * S * kQueryHeads);
    DeviceBuffer<float> d_p_acc(kMaxSplits * S * kQueryHeads * kHeadDim);

    std::cout << "[INFO] Split-K Workspace allocated: "
              << (d_p_max.count * 4 + d_p_sum.count * 4 + d_p_acc.count * 4) / (1024.0 * 1024.0)
              << " MiB\n\n";

    // -----------------------------------------------------------------
    // Run Control / Reference: launch_qwen35_tiled_online_attention_batch_f16
    // -----------------------------------------------------------------
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Benchmarking Reference Kernel (V2-0015 Baseline)...\n";
    std::cout << "-------------------------------------------------------------------\n";

    hipEvent_t start, stop;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));

    // Warmup
    miinfer::launch_qwen35_tiled_online_attention_batch_f16(
        d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_ref.ptr,
        S, P, capacity, kQueryHeads, kKvHeads, kHeadDim, kScale, nullptr);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    MIINFER_HIP_CHECK(hipEventRecord(start, nullptr));
    for (int i = 0; i < 3; ++i) {
        miinfer::launch_qwen35_tiled_online_attention_batch_f16(
            d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_ref.ptr,
            S, P, capacity, kQueryHeads, kKvHeads, kHeadDim, kScale, nullptr);
    }
    MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
    MIINFER_HIP_CHECK(hipEventSynchronize(stop));

    float ref_ms = 0.0f;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ref_ms, start, stop));
    ref_ms /= 3.0f;

    std::cout << "  Reference 1 Layer GPU Time: " << std::fixed << std::setprecision(2)
              << ref_ms << " ms  (16 Layers Total: " << ref_ms * 16.0 / 1000.0 << " s)\n\n";

    std::vector<float> h_ref(d_out_ref.count);
    MIINFER_HIP_CHECK(hipMemcpy(h_ref.data(), d_out_ref.ptr, h_ref.size() * sizeof(float), hipMemcpyDeviceToHost));

    // -----------------------------------------------------------------
    // Sweep Candidates
    // -----------------------------------------------------------------
    struct SweepConfig {
        std::string name;
        int candidate_type;
        std::uint32_t splits;
    };

    std::vector<SweepConfig> configs = {
        {"Cand A (1-Wave direct Split-K=32)", 0, 32},
        {"Cand J (HalfWave Unroll-4 Split-K=32)", 16, 32},
        {"Cand J (HalfWave Unroll-4 Split-K=64)", 16, 64},
        {"Cand J (HalfWave Unroll-4 Split-K=96)", 16, 96},
        {"Cand J (HalfWave Unroll-4 Split-K=128)", 16, 128},
        {"Cand J (HalfWave Unroll-4 Split-K=192)", 16, 192},
        {"Cand J (HalfWave Unroll-4 Split-K=256)", 16, 256},
        {"Cand J (HalfWave Unroll-4 Split-K=384)", 16, 384},
        {"Cand J (HalfWave Unroll-4 Split-K=512)", 16, 512},
    };

    std::cout << "========================================================================================================\n";
    std::cout << "  V2-0016 Suffix Attention Kernel Sweep (P=" << P << ", S=" << S << ")\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left << std::setw(44) << "Configuration"
              << std::right << std::setw(12) << "1-Layer (ms)"
              << std::setw(14) << "16-Layer (s)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "Max Abs Err"
              << std::setw(14) << "Cosine Sim"
              << std::setw(10) << "Status" << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    for (const auto& cfg : configs) {
        auto run_candidate = [&]() {
            if (cfg.candidate_type == 0) {
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    qwen35_splitk_suffix_attn_stage1_1w_kernel,
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 1) {
                dim3 block(384);
                dim3 grid(kKvHeads, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_gqa6_lds_kernel<16>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 2) {
                dim3 block(384);
                dim3 grid(kKvHeads, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_gqa6_lds_kernel<32>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 3) {
                constexpr int BQ = 4;
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + BQ - 1) / BQ), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_bqtiled_kernel<BQ>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 4) {
                constexpr int BQ = 8;
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + BQ - 1) / BQ), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_bqtiled_kernel<BQ>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 5) {
                constexpr int BQ = 16;
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + BQ - 1) / BQ), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_bqtiled_kernel<BQ>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 6) {
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    qwen35_splitk_suffix_attn_stage1_wave64_blocked_kernel,
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 7) {
                constexpr int BK = 16;
                dim3 block(256);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + 31) / 32), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_tile32_kernel<BK>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 8) {
                constexpr int BK = 32;
                dim3 block(256);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + 31) / 32), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_tile32_kernel<BK>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 9) {
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    qwen35_splitk_suffix_attn_stage1_fast_kernel,
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 10) {
                constexpr int BQ = 16;
                constexpr int BK = 16;
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + BQ - 1) / BQ), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_flash_attn_stage1_kernel<BQ, BK>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 11) {
                constexpr int BQ = 4;
                constexpr int BK = 8;
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + BQ - 1) / BQ), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_regblocked_kernel<BQ, BK>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 12) {
                constexpr int BQ = 4;
                constexpr int BK = 16;
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + BQ - 1) / BQ), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_regblocked_kernel<BQ, BK>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 13) {
                constexpr int BQ = 8;
                constexpr int BK = 8;
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + BQ - 1) / BQ), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_regblocked_kernel<BQ, BK>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 14) {
                dim3 block(64);
                dim3 grid(kQueryHeads / 2, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    qwen35_splitk_suffix_attn_stage1_halfwave_kernel,
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 15) {
                dim3 block(64);
                dim3 grid(kQueryHeads / 2, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_halfwave_unrolled_kernel<2>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 16) {
                dim3 block(64);
                dim3 grid(kQueryHeads / 2, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_halfwave_unrolled_kernel<4>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 17) {
                dim3 block(64);
                dim3 grid(kQueryHeads / 2, static_cast<unsigned int>(S), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_halfwave_unrolled_kernel<8>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 18) {
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + 3) / 4), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_multiquery_kernel<2>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            } else if (cfg.candidate_type == 19) {
                dim3 block(64);
                dim3 grid(kQueryHeads, static_cast<unsigned int>((S + 3) / 4), cfg.splits);
                hipLaunchKernelGGL(
                    (qwen35_splitk_suffix_attn_stage1_multiquery_kernel<4>),
                    grid, block, 0, nullptr,
                    d_q.ptr, d_k.ptr, d_v.ptr,
                    d_p_max.ptr, d_p_sum.ptr, d_p_acc.ptr,
                    static_cast<std::uint32_t>(S), static_cast<std::uint32_t>(P),
                    static_cast<std::uint32_t>(capacity),
                    kQueryHeads, kKvHeads, kHeadDim, kScale, cfg.splits);
            }

            // Stage 2
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
        constexpr int kIters = 2;
        for (int i = 0; i < kIters; ++i) {
            run_candidate();
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));

        float cand_ms = 0.0f;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&cand_ms, start, stop));
        cand_ms /= kIters;

        // Verify accuracy
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
