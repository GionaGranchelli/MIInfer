#pragma once

#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace miinfer {

// M12 feasibility primitive: expand one canonical Q4_K matrix into row-major
// FP16 storage for a subsequent matrix-oriented prefill experiment.
void launch_m12_q4k_to_fp16(
    const Q4KDeviceBlock* source,
    __half* destination,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);

} // namespace miinfer
