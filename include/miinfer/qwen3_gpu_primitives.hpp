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

void launch_qwen3_q6_k_gemv(
    const Q6KDeviceBlock* weights,
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

}  // namespace miinfer
