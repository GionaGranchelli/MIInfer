#pragma once

#include "miinfer/qwen3_model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace miinfer {

// LayerTrace deliberately mirrors the layer-0 checkpoints emitted by the
// pinned reference.  Keeping these vectors visible is a correctness feature;
// this is not the eventual hot-path runtime representation.
struct Qwen3LayerTrace {
    std::vector<float> embedding;
    std::vector<float> attn_rms;
    std::vector<float> attn_norm;
    std::vector<float> q_projection;
    std::vector<float> q_reshape;
    std::vector<float> q_rms;
    std::vector<float> q_normed;
    std::vector<float> q_rope;
    std::vector<float> v_projection;
    std::vector<float> v_reshape;
    std::vector<float> k_projection;
    std::vector<float> k_reshape;
    std::vector<float> k_rms;
    std::vector<float> k_normed;
    std::vector<float> k_rope;
    std::vector<float> k_view;
    std::vector<float> v_view;
    std::vector<float> q_view;
    std::vector<float> q_permuted;
    std::vector<float> attention_scores;
    std::vector<float> attention_probabilities;
    std::vector<float> attention_output;
    std::vector<float> ffn_input;
    std::vector<float> ffn_rms;
    std::vector<float> ffn_norm;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> swiglu;
    std::vector<float> ffn_output;
    std::vector<float> layer_output;
};

// Correctness-first single-token full-model trace.  Layer outputs are retained
// so depth-composition failures can be localized without re-running a full
// 36-layer forward pass by hand.  This is not the eventual hot-path runtime
// representation.
struct Qwen3ForwardTrace {
    std::vector<float> embedding;
    std::vector<std::vector<float>> layer_outputs;
    std::vector<float> final_norm;
    std::vector<float> logits;
};

// Layer-0 incremental-state contract used by M4-A4. Q/K RoPE uses Qwen3's
// NeoX first-half/second-half pairing. Keys are stored after RoPE in
// [kv_head][position][head_dim] order; values use the same layout.
// The cache owns only layer-0 state and is reset explicitly between sequences.
class Qwen3Layer0KvCache {
public:
    Qwen3Layer0KvCache(std::size_t kv_heads, std::size_t head_dim, std::size_t capacity);

    void reset() noexcept;
    void append(std::size_t position, std::span<const float> keys, std::span<const float> values);

    [[nodiscard]] std::size_t kv_heads() const noexcept { return kv_heads_; }
    [[nodiscard]] std::size_t head_dim() const noexcept { return head_dim_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t length() const noexcept { return length_; }
    [[nodiscard]] std::span<const float> keys() const noexcept { return keys_; }
    [[nodiscard]] std::span<const float> values() const noexcept { return values_; }

private:
    std::size_t kv_heads_;
    std::size_t head_dim_;
    std::size_t capacity_;
    std::size_t length_ = 0;
    std::vector<float> keys_;
    std::vector<float> values_;
};

// Executes only the deterministic one-token, position-zero layer-0 fixture.
// The implementation is intentionally host-side and correctness-first.  It
// uses the canonical model tensors and Q4_0 x Q8_1 reference arithmetic; it
// is not a performance path.
Qwen3LayerTrace execute_qwen3_layer0_host(
    const Qwen3Model& model,
    std::uint32_t token,
    std::size_t position = 0);

Qwen3LayerTrace execute_qwen3_layer0_host(
    const Qwen3Model& model,
    std::uint32_t token,
    std::size_t position,
    Qwen3Layer0KvCache& cache);

// Executes one explicit token at position zero through all supported Qwen3
// layers, final RMSNorm, and the Q6_K vocabulary projection.  The model is
// intentionally single-token/single-sequence here; incremental multi-position
// state remains covered by the layer-0 M4-A contract.
Qwen3ForwardTrace execute_qwen3_forward_host(
    const Qwen3Model& model,
    std::uint32_t token,
    std::size_t position = 0);

// Correctness-only teacher-forced replay.  The supplied hidden state is the
// independent reference output of the preceding layer (or the embedding for
// layer zero), so this diagnoses local layer arithmetic separately from
// free-running depth error.  The returned trace is intentionally complete;
// this is not a hot-path execution API.
Qwen3LayerTrace execute_qwen3_layer_host_teacher_forced(
    const Qwen3Model& model,
    std::size_t layer_index,
    std::span<const float> input,
    std::size_t position = 0);

}  // namespace miinfer
