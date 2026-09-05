#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "miinfer/q4_q8_gemv.hpp"

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

struct Q4KDeviceBlock {
    __half d;
    __half dmin;
    std::uint8_t scales[12];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q4KDeviceBlock) == 144);

struct Q5KDeviceBlock {
    __half d;
    __half dmin;
    std::uint8_t scales[12];
    std::uint8_t qh[32];
    std::uint8_t ql[128];
};

static_assert(sizeof(Q5KDeviceBlock) == 176);

void launch_qwen3_q4_embedding(
    const std::byte* weights,
    std::uint32_t token,
    std::uint32_t vocabulary,
    std::uint32_t hidden_size,
    float* output,
    hipStream_t stream = nullptr);

void launch_qwen35_q4_k_embedding(
    const Q4KDeviceBlock* weights,
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

// Qwen35 full-attention helpers. These preserve the model's interleaved
// Q+gate projection and sectioned 64-dimension RoPE contract.
void launch_qwen35_split_q_gate(
    const float* input,
    float* query,
    float* gate,
    std::uint32_t heads,
    std::uint32_t head_dim,
    hipStream_t stream = nullptr);

void launch_qwen35_rope_sections(
    const float* input,
    float* output,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t position,
    float theta,
    hipStream_t stream = nullptr);

void launch_qwen35_sigmoid_mul(
    const float* input,
    const float* gate,
    float* output,
    std::uint32_t elements,
    hipStream_t stream = nullptr);

// Qwen35 DeltaNet recurrent core. State is persistent in the [v_head][row][column]
// layout and is updated in place; one workgroup owns one value head.
void launch_qwen35_deltanet_state_update(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* state,
    float* output,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    hipStream_t stream = nullptr);

// Candidate that avoids materializing the decayed state between the two
// recurrence-dependent passes. Opt-in callers must validate its numerical
// behavior against the external state/output contract.
void launch_qwen35_deltanet_state_update_no_decay_store(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* state,
    float* output,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    hipStream_t stream = nullptr);

// Diagnostic state-update candidate using column-major logical state storage.
// The caller owns the transposed [v_head][column][row] representation.
void launch_qwen35_deltanet_state_update_transposed(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* state,
    float* output,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    hipStream_t stream = nullptr);

void launch_qwen35_deltanet_state_update_transposed_no_decay_store(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* state,
    float* output,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    hipStream_t stream = nullptr);

// Apply the four-tap recurrent convolution, SiLU, and Q/K/V split while
// updating a persistent circular history of raw QKV vectors.
void launch_qwen35_conv_silu_split(
    const float* current_qkv,
    const float* conv_weights,
    float* history,
    float* query,
    float* key,
    float* value,
    std::uint32_t position,
    std::uint32_t history_capacity,
    std::uint32_t channels,
    std::uint32_t conv_kernel,
    hipStream_t stream = nullptr);

void launch_qwen35_head_l2_normalize(
    const float* input,
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

// C12b diagnostic path: four Wave64 partitions cooperate over KV history
// while preserving the persistent [head][position][dimension] cache layout.
void launch_qwen3_cached_attention_history_parallel(
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

// Store one token's contiguous K/V vectors in the persistent
// [kv_head][position][head_dim] cache layout with one device launch.
void launch_qwen3_kv_cache_store(
    const float* keys,
    const float* values,
    float* key_cache,
    float* value_cache,
    std::uint32_t position,
    std::uint32_t cache_capacity,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
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

// Opt-in M6-B23 diagnostic candidate: packed gfx906 dot4 for the Q6_K x Q8_K
// recurrent QKV shape. The quantized representation remains unchanged.
void launch_qwen3_q6_k_q8_k_gemv_dot4(
    const Q6KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

// Opt-in M6-B14 diagnostic candidate: llama.cpp-style GCN MMVQ decomposition
// for the Q6_K x Q8_1 LM-head path. This is not production-selected.
void launch_qwen3_q6_k_q8_1_mmvq(
    const Q6KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

// Opt-in M6-B17 diagnostic candidate: gfx906 MMVQ decomposition for
// Q5_K x Q8_1 recurrent output projections.
void launch_qwen3_q5_k_q8_1_mmvq(
    const Q5KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

// Opt-in M6-B18 diagnostic candidate: gfx906 MMVQ decomposition for
// Q4_K x Q8_1 FFN Down projections.
void launch_qwen3_q4_k_q8_1_mmvq(
    const Q4KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

// gfx906 path: two independent output rows share one LDS-resident Q8_1
// activation tile while retaining the existing per-row reduction.
void launch_qwen3_q4_k_q8_1_mmvq_lds_input(
    const Q4KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

// Opt-in candidate: stage only Q4_K metadata in LDS while retaining global
// nibble loads and the existing two-row reduction.
void launch_qwen3_q4_k_q8_1_mmvq_lds_metadata(
    const Q4KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

// Opt-in candidate: decode each Q4_K block's metadata once into LDS before
// the existing two-row reduction.
void launch_qwen3_q4_k_q8_1_mmvq_lds_decoded_metadata(
    const Q4KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

// M6-A10 diagnostic primitive: canonical GGUF Q4_K weights with Q8_K input.
void launch_qwen3_q4_k_q8_k_gemv(
    const Q4KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

// Diagnostic gfx906 packed-dot candidate for Q4_K × Q8_K projections.
void launch_qwen3_q4_k_q8_k_gemv_dot4(
    const Q4KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

void launch_qwen3_q5_k_q8_k_gemv(
    const Q5KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

void launch_qwen35_f32_gemv(
    const float* weights,
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

void launch_qwen35_prepare_beta_decay(
    const float* beta_raw,
    const float* alpha_raw,
    const float* dt,
    const float* a,
    float* beta,
    float* decay,
    std::uint32_t elements,
    hipStream_t stream = nullptr);

// Reduce finite logits using first-index tie breaking, matching
// std::max_element, and write the selected vocabulary index to the device.
void launch_qwen3_argmax(
    const float* input,
    std::uint32_t* output,
    std::uint32_t elements,
    hipStream_t stream = nullptr);

}  // namespace miinfer
