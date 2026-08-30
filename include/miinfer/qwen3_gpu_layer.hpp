#pragma once

#include "miinfer/model_plan.hpp"
#include "miinfer/qwen3_layer.hpp"

#include <cstddef>
#include <cstdint>

namespace miinfer {

// Correctness-first MI50 execution of the same one-token, position-zero
// layer-0 fixture used by execute_qwen3_layer0_host().  Checkpoint vectors are
// copied back deliberately so the comparator can inspect every stage.
Qwen3LayerTrace execute_qwen3_layer0_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position = 0);

}  // namespace miinfer
