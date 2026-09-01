#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace miinfer {

struct Q6KDeviceBlock {
    std::uint8_t ql[128];
    std::uint8_t qh[64];
    std::int8_t scales[16];
    __half d;
};

static_assert(sizeof(Q6KDeviceBlock) == 210);

struct Q8KDeviceBlock {
    float d;
    std::int8_t qs[256];
    std::int16_t bsums[16];
};

static_assert(sizeof(Q8KDeviceBlock) == 292);

void launch_qwen3_q4_embedding(
    const std::byte* weights,
    std::uint32_t token,
    std::uint32_t vocabulary,
    std::uint32_t hidden_size,
    float* output,
    hipStream_t stream = nullptr);

void launch_qwen3_rms_norm(
    const float* input,
    const float* weights,
    float* output,
    std::uint32_t elements,
    float epsilon,
    hipStream_t stream = nullptr);

void launch_qwen3_rms_normalize(
    const float* input,
    float* output,
    std::uint32_t elements,
    float epsilon,
    hipStream_t stream = nullptr);

void launch_qwen3_elementwise_mul(
    const float* left,
    const float* right,
    float* output,
    std::uint32_t elements,
    hipStream_t stream = nullptr);

void launch_qwen3_head_rms_normalize(
    const float* input,
    float* output,
    std::uint32_t heads,
    std::uint32_t head_dim,
    float epsilon,
    hipStream_t stream = nullptr);

void launch_qwen3_head_mul(
    const float* input,
    const float* weights,
    float* output,
    std::uint32_t heads,
    std::uint32_t head_dim,
    hipStream_t stream = nullptr);

void launch_qwen3_rope(
    const float* input,
    float* output,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t position,
    float theta,
    hipStream_t stream = nullptr);

void launch_qwen3_single_token_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    float* scores,
    float* probabilities,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float scale,
    hipStream_t stream = nullptr);

void launch_qwen3_cached_attention(
    const float* q,
    const float* key_cache,
    const float* value_cache,
    std::uint32_t cache_length,
    std::uint32_t cache_capacity,
    float* output,
    float* scores,
    float* probabilities,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float scale,
    hipStream_t stream = nullptr);

// M5-C2 cooperative path: one 256-thread workgroup cooperates on each query
// head while reading the same persistent [head][position][dimension] cache
// layout as launch_qwen3_cached_attention().
void launch_qwen3_cached_attention_parallel(
    const float* q,
    const float* key_cache,
    const float* value_cache,
    std::uint32_t cache_length,
    std::uint32_t cache_capacity,
    float* output,
    float* scores,
    float* probabilities,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float scale,
    hipStream_t stream = nullptr);

void launch_qwen3_silu_mul(
    const float* gate,
    const float* up,
    float* output,
    std::uint32_t elements,
    hipStream_t stream = nullptr);

void launch_qwen3_add(
    const float* left,
    const float* right,
    float* output,
    std::uint32_t elements,
    hipStream_t stream = nullptr);

void launch_qwen3_f32_to_f16(
    const float* input,
    __half* output,
    std::uint32_t elements,
    hipStream_t stream = nullptr);

void launch_qwen3_f16_to_f32(
    const __half* input,
    float* output,
    std::uint32_t elements,
    hipStream_t stream = nullptr);

void launch_qwen3_q6_k_gemv(
    const Q6KDeviceBlock* weights,
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

void launch_qwen3_q8_k_quantize(
    const float* input,
    Q8KDeviceBlock* output,
    std::uint32_t elements,
    hipStream_t stream = nullptr);

void launch_qwen3_q6_k_q8_k_gemv(
    const Q6KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

}  // namespace miinfer
