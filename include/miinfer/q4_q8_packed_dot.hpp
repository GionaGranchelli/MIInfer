#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include "miinfer/q4_q8_gemv.hpp"

namespace miinfer {

// gfx906-only Q4_0 x Q8_1 GEMV candidate. The canonical Q4_0/Q8_1 blocks are
// retained in global memory; only four-value groups are packed in registers.
void launch_q4_q8_gemv_packed_dot(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream = nullptr);

// Minimal ISA-lowering probe used to validate the compiler intrinsic before
// interpreting full-GEMV measurements.
void launch_q4_q8_dot4_probe(
    int lhs,
    int rhs,
    int* output,
    hipStream_t stream = nullptr);

}  // namespace miinfer
