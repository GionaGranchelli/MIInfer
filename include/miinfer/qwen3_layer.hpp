#pragma once

#include "miinfer/qwen3_model.hpp"

#include <cstddef>
#include <cstdint>
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

// Executes only the deterministic one-token, position-zero layer-0 fixture.
// The implementation is intentionally host-side and correctness-first.  It
// uses the canonical model tensors and Q4_0 x Q8_1 reference arithmetic; it
// is not a performance path.
Qwen3LayerTrace execute_qwen3_layer0_host(
    const Qwen3Model& model,
    std::uint32_t token,
    std::size_t position = 0);

}  // namespace miinfer
