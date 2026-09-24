#pragma once

#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/prefill_v2/state.hpp"
#include "miinfer/prefill_v2/workspace.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace miinfer::prefill_v2 {

// Breakdown of fine-grained phase execution times for diagnostic / attribution purposes
struct RecurrentLayerPhaseTimings {
    double norm_beta_alpha_ms = 0.0;
    double qkv_gate_proj_ms = 0.0;
    double conv_l2_norm_ms = 0.0;
    double gdn_chunkwise_ms = 0.0;
    double ssm_post_out_ms = 0.0;
    double residual_norm_ms = 0.0;
    double ffn_gate_up_ms = 0.0;
    double swiglu_down_residual_ms = 0.0;
    double total_layer_ms = 0.0;
};

// Clean-sheet Prefill V2 Recurrent Layer.
// Specialization invariants:
// 1. Semantic invariance: logical recurrence semantics are independent of batch geometry.
// 2. C64 internal chunk: GDN evaluated as sequential C64 chunks across all prompt lengths.
// 3. Explicit state in/out: recurrent matrix state [48, 128, 128] and conv history [4, 10240].
// 4. Resident pre-repacked weights: MMQ tiles loaded and repacked once onto MI50 device memory.
// 5. Zero hot-path allocations.
class PrefillV2RecurrentLayer {
public:
    PrefillV2RecurrentLayer(const Qwen35Model& model, std::size_t layer_index);
    ~PrefillV2RecurrentLayer();
    PrefillV2RecurrentLayer(const PrefillV2RecurrentLayer&) = delete;
    PrefillV2RecurrentLayer& operator=(const PrefillV2RecurrentLayer&) = delete;
    PrefillV2RecurrentLayer(PrefillV2RecurrentLayer&& other) noexcept;
    PrefillV2RecurrentLayer& operator=(PrefillV2RecurrentLayer&& other) noexcept;

    [[nodiscard]] std::size_t layer_index() const noexcept { return layer_index_; }
    [[nodiscard]] std::size_t persistent_weight_bytes() const noexcept { return persistent_weight_bytes_; }
    [[nodiscard]] GgufTensorType qkv_type() const noexcept { return qkv_type_; }
    [[nodiscard]] GgufTensorType ffn_down_type() const noexcept { return ffn_down_type_; }

    // Executes one complete recurrent layer for `token_count` tokens (multiple of 64).
    void forward(
        const float* d_input,
        float* d_output,
        const RecurrentLayerState& incoming_state,
        RecurrentLayerState& outgoing_state,
        RecurrentLayerWorkspace& workspace,
        std::uint32_t token_count,
        hipStream_t stream = nullptr);

    // Profiled execution variant for fine-grained phase attribution
    void forward_profiled(
        const float* d_input,
        float* d_output,
        const RecurrentLayerState& incoming_state,
        RecurrentLayerState& outgoing_state,
        RecurrentLayerWorkspace& workspace,
        std::uint32_t token_count,
        RecurrentLayerPhaseTimings& timings,
        hipStream_t stream = nullptr);

private:
    std::size_t layer_index_ = 0;
    std::size_t persistent_weight_bytes_ = 0;

    // Device weights
    float* d_attn_norm_ = nullptr;
    void* d_qkv_mmq_ = nullptr;
    GgufTensorType qkv_type_ = GgufTensorType::q4_k;
    Q4KMmqTile* d_gate_mmq_ = nullptr;
    float* d_ssm_beta_ = nullptr;
    float* d_ssm_alpha_ = nullptr;
    float* d_ssm_dt_ = nullptr;
    float* d_ssm_a_ = nullptr;
    float* d_ssm_conv_ = nullptr;
    float* d_ssm_norm_ = nullptr;
    Q5KMmqTile* d_ssm_out_mmq_ = nullptr;
    float* d_post_norm_ = nullptr;
    Q4KMmqTile* d_ffn_gate_mmq_ = nullptr;
    Q4KMmqTile* d_ffn_up_mmq_ = nullptr;
    void* d_ffn_down_mmq_ = nullptr;
    GgufTensorType ffn_down_type_ = GgufTensorType::q6_k;
};

} // namespace miinfer::prefill_v2
