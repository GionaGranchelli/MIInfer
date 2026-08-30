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

// Layer-0 incremental-state contract used by M4-A4.  Keys are stored after
// RoPE in [kv_head][position][head_dim] order; values use the same layout.
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

}  // namespace miinfer
