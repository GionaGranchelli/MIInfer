#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include "miinfer/fp16_gemv.hpp"

#include <cstdint>
#include <vector>

namespace miinfer {

// Layouts match the pinned llama.cpp reference's ggml-common.h. These are
// canonical blocks, not the future gfx906-native packed layout.
constexpr int kQ4_0BlockSize = 32;
constexpr int kQ8_1BlockSize = 32;

struct Q4_0Block {
    __half d;
    std::uint8_t qs[kQ4_0BlockSize / 2];
};

struct Q8_1Block {
    __half d;
    __half s;
    std::int8_t qs[kQ8_1BlockSize];
};

// Exact zero-point metadata candidate.  The integer lane sum replaces the
// FP16-scaled Q8_1::s field, while preserving the canonical 36-byte block
// footprint and Q8 lane representation.
struct Q8ExactBlock {
    __half d;
    std::int16_t sum;
    std::int8_t qs[kQ8_1BlockSize];
};

static_assert(sizeof(Q4_0Block) == 18, "Q4_0 layout must be 18 bytes");
static_assert(sizeof(Q8_1Block) == 36, "Q8_1 layout must be 36 bytes");
static_assert(sizeof(Q8ExactBlock) == 36, "exact Q8 layout must be 36 bytes");

std::vector<Q4_0Block> quantize_q4_0(
    const std::vector<__half>& source,
    int rows,
    int columns);

std::vector<Q8_1Block> quantize_q8_1(const std::vector<__half>& source);

std::vector<Q8ExactBlock> quantize_q8_exact(const std::vector<__half>& source);

std::vector<float> q4_q8_cpu_reference(
    const std::vector<Q4_0Block>& weights,
    const std::vector<Q8_1Block>& input,
    int rows,
    int columns);

void launch_q8_1_quantize(
    const __half* input,
    Q8_1Block* output,
    int elements,
    hipStream_t stream = nullptr);

// Correctness-only diagnostic variant: quantize the supplied FP32 values
// directly, without the production FP32-to-FP16 input boundary.
void launch_q8_1_quantize_f32(
    const float* input,
    Q8_1Block* output,
    int elements,
    hipStream_t stream = nullptr);

// Exact metadata candidate: stores the integer Q8 lane sum instead of the
// lossy FP16-scaled Q8_1::s field.  The input conversion remains FP16 so the
// quantized lanes match the production projection boundary.
void launch_q8_exact_quantize(
    const __half* input,
    Q8ExactBlock* output,
    int elements,
    hipStream_t stream = nullptr);

void launch_q8_exact_quantize_f32(
    const float* input,
    Q8ExactBlock* output,
    int elements,
    hipStream_t stream = nullptr);

// Opt-in C10c candidate: reproduce the FFN RMSNorm -> norm-weight scale ->
// FP16 materialization -> exact Q8 stream in one gfx906 dispatch.  The
// optional materialized_output is diagnostic-only and receives the exact
// FP16 values that the separate production path would have stored.
void launch_qwen3_ffn_norm_to_q8_exact(
    const float* input,
    const float* weights,
    Q8ExactBlock* output,
    int elements,
    float epsilon,
    __half* materialized_output = nullptr,
    hipStream_t stream = nullptr);

// Candidate decode primitive: reproduce the production FP32 -> FP16
// SwiGLU boundary and write the resulting exact Q8 metadata directly,
// avoiding global materialization of the intermediate activation.
void launch_silu_mul_q8_exact(
    const float* gate,
    const float* up,
    Q8ExactBlock* output,
    int elements,
    hipStream_t stream = nullptr);

void launch_q4_q8_gemv(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

}  // namespace miinfer
