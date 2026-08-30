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

static_assert(sizeof(Q4_0Block) == 18, "Q4_0 layout must be 18 bytes");
static_assert(sizeof(Q8_1Block) == 36, "Q8_1 layout must be 36 bytes");

std::vector<Q4_0Block> quantize_q4_0(
    const std::vector<__half>& source,
    int rows,
    int columns);

std::vector<Q8_1Block> quantize_q8_1(const std::vector<__half>& source);

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

void launch_q4_q8_gemv(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

}  // namespace miinfer
