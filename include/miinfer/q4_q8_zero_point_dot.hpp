#pragma once

#include "miinfer/q4_q8_gemv.hpp"

namespace miinfer {

// gfx906-only Q4_0 x Q8_1 GEMV. Q4 nibbles remain unsigned in the dot
// operands; Q8_1::s supplies the algebraic Q4 zero-point correction.
void launch_q4_q8_gemv_zero_point_dot(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

// EXP-0009 controls: same EXP-0007 arithmetic with K/V-oriented geometry.
void launch_q4_q8_gemv_zero_point_dot_128(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

// EXP-0009 candidate: two Wave64 reductions with one cross-wave exchange.
void launch_q4_q8_gemv_zero_point_dot_wave64(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

// Production Qwen3 path: same geometry as the corresponding controls, but
// uses the exact integer sum of Q8 lanes for Q4 zero-point correction.
void launch_q4_q8_gemv_zero_point_dot_exact_sum(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

void launch_q4_q8_gemv_zero_point_dot_128_exact_sum(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

void launch_q4_q8_gemv_zero_point_dot_wave64_exact_sum(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

// Correctness-only diagnostic outputs.  These preserve the same arithmetic
// and geometry while exposing the GEMV result as FP32, so output-rounding
// effects can be separated from input quantization effects.
void launch_q4_q8_gemv_zero_point_dot_f32(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    float* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

void launch_q4_q8_gemv_zero_point_dot_128_f32(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    float* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

void launch_q4_q8_gemv_zero_point_dot_wave64_f32(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    float* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

// Correctness-only contract probes.  Both consume the canonical Q8_1 blocks
// used by production, but avoid the lossy FP16 `s` field for zero-point
// correction.  The direct variant is a scalar signed-Q4 oracle; neither is a
// production kernel.
void launch_q4_q8_gemv_zero_point_dot_exact_sum_f32(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    float* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

void launch_q4_q8_gemv_zero_point_dot_128_exact_sum_f32(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    float* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

void launch_q4_q8_gemv_zero_point_dot_wave64_exact_sum_f32(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    float* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

void launch_q4_q8_gemv_direct_signed_f32(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    float* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

}  // namespace miinfer
