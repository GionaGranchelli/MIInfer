#pragma once

#include "miinfer/model_plan.hpp"
#include "miinfer/qwen3_layer.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace miinfer {

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

}  // namespace miinfer
