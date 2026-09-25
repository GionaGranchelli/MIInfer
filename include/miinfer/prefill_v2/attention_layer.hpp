#pragma once

#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/prefill_v2/kv_cache.hpp"
#include "miinfer/prefill_v2/workspace.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace miinfer::prefill_v2 {

struct AttentionLayerProfileBreakdown {
    double norm_ms = 0.0;
    double qkv_proj_ms = 0.0;
    double rope_kv_store_ms = 0.0;
    double causal_attn_ms = 0.0;
    double o_proj_ms = 0.0;
    double post_norm_ms = 0.0;
    double ffn_gate_up_ms = 0.0;
    double swiglu_down_res_ms = 0.0;
    double total_layer_ms = 0.0;
};

struct AttentionLayerDecodePhaseTimings {
    double norm_ms = 0.0;
    double qkv_proj_ms = 0.0;
    double qk_rope_kv_store_ms = 0.0;
    double splitk_attention_ms = 0.0;
    double o_proj_ms = 0.0;
    double residual_norm_ms = 0.0;
    double ffn_gate_up_swiglu_ms = 0.0;
    double ffn_down_residual_ms = 0.0;
    double total_layer_ms = 0.0;
};

// Clean-sheet Prefill V2 Full GQA Attention layer implementation.
// Designed exclusively for AMD Instinct MI50 (gfx906, Wave64).
class PrefillV2AttentionLayer {
public:
    PrefillV2AttentionLayer(const miinfer::Qwen35Model& model, std::size_t layer_index);
    ~PrefillV2AttentionLayer();
    PrefillV2AttentionLayer(const PrefillV2AttentionLayer&) = delete;
    PrefillV2AttentionLayer& operator=(const PrefillV2AttentionLayer&) = delete;
    PrefillV2AttentionLayer(PrefillV2AttentionLayer&& other) noexcept;
    PrefillV2AttentionLayer& operator=(PrefillV2AttentionLayer&& other) noexcept;

    // Standard hot-path forward execution (zero allocations)
    void forward(
        const float* d_input,
        float* d_output,
        AttentionKvCacheView kv_cache,
        const PrefillV2Workspace& ws,
        std::uint32_t base_position,
        std::uint32_t token_count,
        hipStream_t stream = nullptr) const;

    // Specialized single-token decode execution with Split-K attention
    void decode(
        const float* d_input,
        float* d_output,
        AttentionKvCacheView kv_cache,
        const PrefillV2Workspace& ws,
        std::uint32_t position,
        const DeviceDecodeState* decode_state = nullptr,
        hipStream_t stream = nullptr) const;

    // Profiled single-token decode variant for phase attribution
    void decode_profiled(
        const float* d_input,
        float* d_output,
        AttentionKvCacheView kv_cache,
        const PrefillV2Workspace& ws,
        std::uint32_t position,
        const DeviceDecodeState* decode_state,
        AttentionLayerDecodePhaseTimings& timings,
        hipStream_t stream = nullptr) const;

    // Profiled execution recording kernel phase events
    void forward_profiled(
        const float* d_input,
        float* d_output,
        AttentionKvCacheView kv_cache,
        const PrefillV2Workspace& ws,
        std::uint32_t base_position,
        std::uint32_t token_count,
        AttentionLayerProfileBreakdown& breakdown,
        hipStream_t stream = nullptr) const;

    [[nodiscard]] std::size_t layer_index() const noexcept { return layer_index_; }
    [[nodiscard]] std::size_t persistent_weight_bytes() const noexcept { return persistent_weight_bytes_; }
    [[nodiscard]] bool v_is_q6() const noexcept { return v_is_q6_; }
    [[nodiscard]] bool ffn_down_is_q6() const noexcept { return ffn_down_is_q6_; }

private:
    std::size_t layer_index_ = 0;
    std::size_t persistent_weight_bytes_ = 0;
    bool v_is_q6_ = false;
    bool ffn_down_is_q6_ = false;

    // Device Normalization & Scale Weights
    float* d_attn_norm_ = nullptr;        // [5120]
    float* d_q_norm_ = nullptr;           // [256]
    float* d_k_norm_ = nullptr;           // [256]
    float* d_post_attention_norm_ = nullptr; // [5120]

    // Device MMQ-packed weights (Mx compact format)
    std::uint8_t* d_q_mmq_ = nullptr;        // [5120, 12288] Q4_K
    std::uint8_t* d_k_mmq_ = nullptr;        // [5120, 1024] Q4_K
    std::uint8_t* d_v_mmq_ = nullptr;        // [5120, 1024] Q4_K or Q6_K
    std::uint8_t* d_o_mmq_ = nullptr;        // [6144, 5120] Q4_K
    std::uint8_t* d_ffn_gate_mmq_ = nullptr; // [5120, 17408] Q4_K
    std::uint8_t* d_ffn_up_mmq_ = nullptr;   // [5120, 17408] Q4_K
    std::uint8_t* d_ffn_down_mmq_ = nullptr; // [17408, 5120] Q4_K or Q6_K

    // Gfx906 resident Wave decode weights (coexisting with Mx compact prefill weights)
    Q4KWaveSwigluFusedTile* d_ffn_swiglu_fused_ = nullptr;
    Q4KWaveTile* d_o_wave_ = nullptr;
    Q4KWaveTile* d_q_wave_ = nullptr;
    Q4KWaveTile* d_k_wave_ = nullptr;
    void* d_v_wave_ = nullptr;
};

} // namespace miinfer::prefill_v2
