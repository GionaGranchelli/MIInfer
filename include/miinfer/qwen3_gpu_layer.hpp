#pragma once

#include "miinfer/model_plan.hpp"
#include "miinfer/qwen3_layer.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <span>
#include <vector>

namespace miinfer {

enum class Qwen3ProfileCategory {
    embedding,
    normalization,
    quantization,
    qkv_projection,
    o_projection,
    ffn_projection,
    rope,
    attention,
    activation,
    residual,
    conversion,
    lm_head,
    argmax,
    kv_cache,
    copies,
    count,
};

// Fine-grained FFN attribution used by the measurement-only M5-C9a profile.
// These counters do not select kernels or alter the production execution path.
enum class Qwen3FfnProfileStage {
    normalization,
    gate_input_quantization,
    gate_projection,
    up_input_quantization,
    up_projection,
    swiglu,
    swiglu_down_input_quantization,
    down_input_quantization,
    down_projection,
    residual,
    count,
};

// Measurement-only attribution for normalization, conversion, and
// quantization boundaries.  These labels describe the existing producer /
// representation / consumer path; they do not select kernels or alter it.
enum class Qwen3BoundaryProfileStage {
    attn_rms_normalize,
    attn_norm_scale,
    q_input_f32_to_f16,
    q_input_q8,
    q_output_f16_to_f32,
    q_head_rms_normalize,
    q_head_scale,
    k_input_f32_to_f16,
    k_input_q8,
    k_output_f16_to_f32,
    k_head_rms_normalize,
    k_head_scale,
    v_input_f32_to_f16,
    v_input_q8,
    v_output_f16_to_f32,
    attention_f32_to_f16,
    attention_f16_to_f32,
    o_input_f32_to_f16,
    o_input_q8,
    o_output_f16_to_f32,
    ffn_rms_normalize,
    ffn_norm_scale,
    gate_input_f32_to_f16,
    gate_input_q8,
    gate_output_f16_to_f32,
    up_input_f32_to_f16,
    up_input_q8,
    up_output_f16_to_f32,
    down_input_f32_to_f16,
    down_input_q8,
    down_output_f16_to_f32,
    final_rms_norm,
    final_norm_to_q8k,
    count,
};

constexpr std::size_t qwen3_profile_category_count =
    static_cast<std::size_t>(Qwen3ProfileCategory::count);
constexpr std::size_t qwen3_ffn_profile_stage_count =
    static_cast<std::size_t>(Qwen3FfnProfileStage::count);
constexpr std::size_t qwen3_boundary_profile_stage_count =
    static_cast<std::size_t>(Qwen3BoundaryProfileStage::count);

struct Qwen3GpuProfileEvent {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    Qwen3ProfileCategory category = Qwen3ProfileCategory::count;
    Qwen3FfnProfileStage ffn_stage = Qwen3FfnProfileStage::count;
    Qwen3BoundaryProfileStage boundary_stage = Qwen3BoundaryProfileStage::count;
    std::size_t dispatches = 0;
    std::size_t bytes = 0;
    bool copy = false;
};

struct Qwen3GpuProfile {
    std::array<double, qwen3_profile_category_count> gpu_ms{};
    std::array<double, qwen3_profile_category_count> copy_ms{};
    std::array<std::size_t, qwen3_profile_category_count> dispatches{};
    std::array<std::size_t, qwen3_profile_category_count> copy_bytes{};
    std::array<std::size_t, qwen3_profile_category_count> synchronizations{};
    std::array<double, qwen3_ffn_profile_stage_count> ffn_gpu_ms{};
    std::array<std::size_t, qwen3_ffn_profile_stage_count> ffn_dispatches{};
    std::array<double, qwen3_boundary_profile_stage_count> boundary_gpu_ms{};
    std::array<std::size_t, qwen3_boundary_profile_stage_count> boundary_dispatches{};
    std::array<std::size_t, qwen3_boundary_profile_stage_count> boundary_bytes{};
    std::size_t temporary_allocations = 0;
    std::size_t finalization_synchronizations = 0;
    std::size_t gate_up_q8_reuse_checks = 0;
    std::size_t gate_up_q8_reuse_mismatches = 0;
    bool deferred_timing = false;
    std::vector<Qwen3GpuProfileEvent> pending_events;

    Qwen3GpuProfile() = default;
    Qwen3GpuProfile(const Qwen3GpuProfile&) = delete;
    Qwen3GpuProfile& operator=(const Qwen3GpuProfile&) = delete;
    Qwen3GpuProfile(Qwen3GpuProfile&&) noexcept = default;
    Qwen3GpuProfile& operator=(Qwen3GpuProfile&&) noexcept = default;
    ~Qwen3GpuProfile() noexcept;

    void reset() noexcept {
        for (const auto& event : pending_events) {
            if (event.start != nullptr) (void)hipEventDestroy(event.start);
            if (event.stop != nullptr) (void)hipEventDestroy(event.stop);
        }
        pending_events.clear();
        gpu_ms.fill(0.0);
        copy_ms.fill(0.0);
        dispatches.fill(0);
        copy_bytes.fill(0);
        synchronizations.fill(0);
        ffn_gpu_ms.fill(0.0);
        ffn_dispatches.fill(0);
        boundary_gpu_ms.fill(0.0);
        boundary_dispatches.fill(0);
        boundary_bytes.fill(0);
        temporary_allocations = 0;
        finalization_synchronizations = 0;
        gate_up_q8_reuse_checks = 0;
        gate_up_q8_reuse_mismatches = 0;
        deferred_timing = false;
    }

    void enable_deferred_timing() noexcept {
        deferred_timing = true;
    }

    void finalize();

    void record_copy(Qwen3ProfileCategory category, std::size_t bytes) noexcept {
        const auto index = static_cast<std::size_t>(category);
        copy_bytes[index] += bytes;
        synchronizations[index] += 1;
    }
};

[[nodiscard]] const char* qwen3_profile_category_name(Qwen3ProfileCategory category) noexcept;
[[nodiscard]] const char* qwen3_ffn_profile_stage_name(Qwen3FfnProfileStage stage) noexcept;
[[nodiscard]] const char* qwen3_boundary_profile_stage_name(
    Qwen3BoundaryProfileStage stage) noexcept;

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
    void append(std::size_t position, const float* keys, const float* values,
                Qwen3GpuProfile* profile = nullptr);

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

class Qwen3GpuDecodeWorkspace;

// Minimal model-level MI50 decode state.  One layer-scoped cache is retained
// for each Qwen3 layer.  The decode workspace is allocated lazily on the first
// full decode and then retained for the lifetime of this cache.
class Qwen3GpuDecodeCache {
public:
    Qwen3GpuDecodeCache(std::size_t layers, std::size_t kv_heads,
                        std::size_t head_dim, std::size_t capacity);
    Qwen3GpuDecodeCache(const Qwen3GpuDecodeCache&) = delete;
    Qwen3GpuDecodeCache& operator=(const Qwen3GpuDecodeCache&) = delete;
    ~Qwen3GpuDecodeCache();

    void reset();
    void prepare(const Qwen3GpuPlan& plan);
    [[nodiscard]] std::size_t layers() const noexcept { return caches_.size(); }
    [[nodiscard]] std::size_t length() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    Qwen3Layer0GpuKvCache& layer(std::size_t layer_index);
    const Qwen3Layer0GpuKvCache& layer(std::size_t layer_index) const;
    Qwen3GpuDecodeWorkspace& workspace();

private:
    std::size_t capacity_;
    std::vector<std::unique_ptr<Qwen3Layer0GpuKvCache>> caches_;
    std::unique_ptr<Qwen3GpuDecodeWorkspace> workspace_;
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

// Opt-in profiling variant. It preserves the decode computation and adds HIP
// event timing around individual operation launches and device copies.
Qwen3ForwardTrace execute_qwen3_decode_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    Qwen3GpuProfile* profile);

// Trace-free decode path for performance measurement and eventual serving.
// It preserves the production layer/cache computation but only copies the
// final vocabulary logits needed by host-side greedy selection.
void execute_qwen3_decode_gpu_fast(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    std::span<float> logits);

// Trace-free decode with optional deferred profiling. When profile has
// deferred timing enabled, operation events are recorded without synchronizing
// each operation and resolved once after this token completes.
void execute_qwen3_decode_gpu_fast(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    std::span<float> logits,
    Qwen3GpuProfile* profile);

// Trace-free greedy decode. The logits remain device-resident and only the
// selected token ID is copied to the host. Inputs are required to produce
// finite logits, matching the existing full-logit greedy contract.
std::uint32_t execute_qwen3_decode_gpu_greedy(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache);

std::uint32_t execute_qwen3_decode_gpu_greedy(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    Qwen3GpuProfile* profile);

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
