#pragma once

#include "miinfer/model_plan.hpp"
#include "miinfer/qwen3_layer.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace miinfer {

enum class Qwen3ProjectionPrecision {
    f16_input_q8_f16_output,
    f32_input_q8_f16_output,
    f16_input_q8_f32_output,
    f32_input_q8_f32_output,
};

enum class Qwen3Projection {
    q,
    k,
    v,
    o,
    gate,
    up,
    down,
};

struct Qwen3ProjectionProbeTrace {
    std::vector<float> output;
};

struct Qwen3FfnProbeTrace {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> swiglu;
    std::vector<float> ffn_output;
    std::vector<float> layer_output;
};

struct Qwen3DownProjectionContractTrace {
    std::vector<float> current_s_correction;
    std::vector<float> exact_sum_correction;
    std::vector<float> direct_signed_oracle;
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

// Minimal model-level MI50 decode state.  One layer-scoped cache is retained
// for each Qwen3 layer; allocation and layer ownership happen before decode.
class Qwen3GpuDecodeCache {
public:
    Qwen3GpuDecodeCache(std::size_t layers, std::size_t kv_heads,
                        std::size_t head_dim, std::size_t capacity);
    Qwen3GpuDecodeCache(const Qwen3GpuDecodeCache&) = delete;
    Qwen3GpuDecodeCache& operator=(const Qwen3GpuDecodeCache&) = delete;
    ~Qwen3GpuDecodeCache() = default;

    void reset();
    [[nodiscard]] std::size_t layers() const noexcept { return caches_.size(); }
    [[nodiscard]] std::size_t length() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    Qwen3Layer0GpuKvCache& layer(std::size_t layer_index);
    const Qwen3Layer0GpuKvCache& layer(std::size_t layer_index) const;

private:
    std::size_t capacity_;
    std::vector<std::unique_ptr<Qwen3Layer0GpuKvCache>> caches_;
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

// Executes one explicit token through all Qwen3 layers using persistent
// per-layer KV state.  This is the correctness-first MI50 decode boundary;
// it intentionally returns a diagnostic trace until M4-C is complete.
Qwen3ForwardTrace execute_qwen3_decode_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache);

// Correctness-only teacher-forced replay of one selected layer.  The input
// hidden state is copied to the device and the complete diagnostic trace is
// copied back, allowing local GPU-vs-reference comparison without conflating
// a layer defect with accumulated free-running depth error.
Qwen3LayerTrace execute_qwen3_layer_gpu_teacher_forced(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    std::span<const float> input,
    std::size_t position = 0);

// Correctness-only causal replay.  The supplied position-zero attention
// output replaces the computed attention output before O projection.  The
// normal production path is unchanged.
Qwen3LayerTrace execute_qwen3_layer_gpu_attention_override(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    std::span<const float> input,
    std::span<const float> attention_output,
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

// Correctness-only probe for one real Qwen3 projection. It always consumes
// Q8ExactBlock metadata; precision selects only the input/output FP16
// boundaries. The projection name is resolved once by this diagnostic API,
// not by the execution hot path.
Qwen3ProjectionProbeTrace execute_qwen3_projection_gpu_probe(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    Qwen3Projection projection,
    std::span<const float> input,
    Qwen3ProjectionPrecision precision);

// Correctness-only layer-6 down-projection contract probe.  The supplied
// activation is quantized using the same F16-input Q8_1 path as production;
// all returned outputs are F32.  The arithmetic contracts differ only in how
// the Q4 zero point is corrected.
Qwen3DownProjectionContractTrace execute_qwen3_down_projection_contract_probe(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    std::span<const float> swiglu,
    bool direct_f32_input = false);

}  // namespace miinfer
