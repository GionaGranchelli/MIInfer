#pragma once

#include "miinfer/model_plan.hpp"
#include "miinfer/qwen3_layer.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace miinfer {

enum class Qwen3ProjectionPrecision {
    f16_input_q8_f16_output,
    f32_input_q8_f16_output,
    f16_input_q8_f32_output,
    f32_input_q8_f32_output,
};

struct Qwen3FfnProbeTrace {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> swiglu;
    std::vector<float> ffn_output;
    std::vector<float> layer_output;
};

// Layer-0 GPU KV-cache contract.  Device storage is laid out as
// [kv_head][position][head_dim], with only [0, length) positions valid.
// Keys are appended after RoPE; values are appended before any attention
// operation.  This class is intentionally single-sequence and layer-0-only.
class Qwen3Layer0GpuKvCache {
public:
    Qwen3Layer0GpuKvCache(std::size_t kv_heads, std::size_t head_dim, std::size_t capacity);
    Qwen3Layer0GpuKvCache(const Qwen3Layer0GpuKvCache&) = delete;
    Qwen3Layer0GpuKvCache& operator=(const Qwen3Layer0GpuKvCache&) = delete;
    ~Qwen3Layer0GpuKvCache();

    void reset();
    void append(std::size_t position, const float* keys, const float* values);

    [[nodiscard]] std::size_t kv_heads() const noexcept { return kv_heads_; }
    [[nodiscard]] std::size_t head_dim() const noexcept { return head_dim_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t length() const noexcept { return length_; }
    [[nodiscard]] std::vector<float> snapshot_keys() const;
    [[nodiscard]] std::vector<float> snapshot_values() const;

    [[nodiscard]] void* device_keys() const noexcept { return keys_; }
    [[nodiscard]] void* device_values() const noexcept { return values_; }

private:
    std::size_t kv_heads_;
    std::size_t head_dim_;
    std::size_t capacity_;
    std::size_t length_ = 0;
    void* keys_ = nullptr;
    void* values_ = nullptr;
};

// Correctness-first MI50 execution of the same one-token, position-zero
// layer-0 fixture used by execute_qwen3_layer0_host().  Checkpoint vectors are
// copied back deliberately so the comparator can inspect every stage.
Qwen3LayerTrace execute_qwen3_layer0_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position = 0);

Qwen3LayerTrace execute_qwen3_layer0_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3Layer0GpuKvCache& cache);

// Correctness-first single-token full-model GPU execution.  The returned
// trace intentionally copies layer outputs and logits back to the host;
// M4-B uses this for depth-composition validation, not serving performance.
Qwen3ForwardTrace execute_qwen3_forward_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position = 0);

// Correctness-only teacher-forced replay of one selected layer.  The input
// hidden state is copied to the device and the complete diagnostic trace is
// copied back, allowing local GPU-vs-reference comparison without conflating
// a layer defect with accumulated free-running depth error.
Qwen3LayerTrace execute_qwen3_layer_gpu_teacher_forced(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    std::span<const float> input,
    std::size_t position = 0);

// Correctness-only layer-6 FFN probes. Empty override spans select the real
// GPU operation; non-empty spans inject a host reference tensor at that stage.
// This isolates projection, nonlinear, down-projection, and residual errors
// without changing the production executor.
Qwen3FfnProbeTrace execute_qwen3_ffn_gpu_probe(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    std::span<const float> ffn_input,
    std::span<const float> ffn_norm,
    std::span<const float> gate_override = {},
    std::span<const float> up_override = {},
    std::span<const float> swiglu_override = {},
    std::span<const float> ffn_output_override = {},
    Qwen3ProjectionPrecision precision = Qwen3ProjectionPrecision::f16_input_q8_f16_output);

}  // namespace miinfer
