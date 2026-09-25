#pragma once

#include "miinfer/prefill_v2/attention_layer.hpp"
#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/prefill_v2/kv_cache.hpp"
#include "miinfer/prefill_v2/recurrent_layer.hpp"
#include "miinfer/prefill_v2/state.hpp"
#include "miinfer/prefill_v2/workspace.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace miinfer::prefill_v2 {

struct TopologyBlockProfileBreakdown {
    double gdn0_ms = 0.0;
    double gdn1_ms = 0.0;
    double gdn2_ms = 0.0;
    double gqa3_ms = 0.0;
    double total_block_ms = 0.0;
};

// Represents one canonical 4-layer repeating topology block in Qwen3.8-27B:
// 3 × Gated DeltaNet Recurrent Layers + 1 × Full GQA Attention Layer.
class PrefillV2TopologyBlock {
public:
    PrefillV2TopologyBlock(const miinfer::Qwen35Model& model, std::size_t block_index);
    ~PrefillV2TopologyBlock() = default;

    PrefillV2TopologyBlock(const PrefillV2TopologyBlock&) = delete;
    PrefillV2TopologyBlock& operator=(const PrefillV2TopologyBlock&) = delete;
    PrefillV2TopologyBlock(PrefillV2TopologyBlock&&) noexcept = default;
    PrefillV2TopologyBlock& operator=(PrefillV2TopologyBlock&&) noexcept = default;

    // Forward execution through the 4-layer block using ping-pong activations.
    // d_ping contains the initial input on entry; final block output is placed into d_final_output.
    // d_pong is used as intermediate scratch activation buffer ([max_tokens, kHidden] floats).
    void forward(
        float* d_ping,
        float* d_pong,
        float* d_final_output,
        const RecurrentLayerState& state0_in,
        RecurrentLayerState& state0_out,
        const RecurrentLayerState& state1_in,
        RecurrentLayerState& state1_out,
        const RecurrentLayerState& state2_in,
        RecurrentLayerState& state2_out,
        AttentionKvCacheView kv_cache3,
        PrefillV2Workspace& ws,
        std::uint32_t base_position,
        std::uint32_t token_count,
        hipStream_t stream = nullptr) const;

    // Specialized single-token decode execution through the 4 layers.
    void decode(
        float* d_ping,
        float* d_pong,
        float* d_final_output,
        RecurrentLayerState& state0,
        RecurrentLayerState& state1,
        RecurrentLayerState& state2,
        AttentionKvCacheView kv_cache3,
        PrefillV2Workspace& ws,
        std::uint32_t position,
        const DeviceDecodeState* decode_state = nullptr,
        hipStream_t stream = nullptr) const;

    // Profiled forward execution
    void forward_profiled(
        float* d_ping,
        float* d_pong,
        float* d_final_output,
        const RecurrentLayerState& state0_in,
        RecurrentLayerState& state0_out,
        const RecurrentLayerState& state1_in,
        RecurrentLayerState& state1_out,
        const RecurrentLayerState& state2_in,
        RecurrentLayerState& state2_out,
        AttentionKvCacheView kv_cache3,
        PrefillV2Workspace& ws,
        std::uint32_t base_position,
        std::uint32_t token_count,
        TopologyBlockProfileBreakdown& breakdown,
        hipStream_t stream = nullptr) const;

    [[nodiscard]] std::size_t block_index() const noexcept { return block_index_; }
    [[nodiscard]] std::size_t persistent_weight_bytes() const noexcept {
        return gdn0_.persistent_weight_bytes() + gdn1_.persistent_weight_bytes()
             + gdn2_.persistent_weight_bytes() + gqa3_.persistent_weight_bytes();
    }

    [[nodiscard]] const PrefillV2RecurrentLayer& gdn0() const noexcept { return gdn0_; }
    [[nodiscard]] const PrefillV2RecurrentLayer& gdn1() const noexcept { return gdn1_; }
    [[nodiscard]] const PrefillV2RecurrentLayer& gdn2() const noexcept { return gdn2_; }
    [[nodiscard]] const PrefillV2AttentionLayer& gqa3() const noexcept { return gqa3_; }

private:
    std::size_t block_index_ = 0;
    PrefillV2RecurrentLayer gdn0_;
    PrefillV2RecurrentLayer gdn1_;
    PrefillV2RecurrentLayer gdn2_;
    PrefillV2AttentionLayer gqa3_;
};

} // namespace miinfer::prefill_v2
