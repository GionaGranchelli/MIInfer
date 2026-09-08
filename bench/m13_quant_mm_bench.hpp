#pragma once

#include "miinfer/kquant_wave_layout.hpp"

// M13 lab-only entry points. They intentionally stay outside the runtime API.
void launch_m13_q4k_wave_gemv_batched_mm(
    const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch,
    std::uint32_t input_stride_blocks, hipStream_t stream = nullptr);

void launch_m13_q6k_wave_gemv_batched_mm(
    const Q6KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch,
    std::uint32_t input_stride_blocks, hipStream_t stream = nullptr);
