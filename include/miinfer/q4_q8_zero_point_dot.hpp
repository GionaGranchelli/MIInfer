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

}  // namespace miinfer
