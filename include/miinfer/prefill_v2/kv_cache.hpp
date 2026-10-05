#pragma once

#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/device_kv_pool.hpp"
#include "miinfer/device_kv_shard.hpp"

#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <span>
#include <vector>

namespace miinfer::prefill_v2 {

// Quantization mode for persistent KV cache storage (V2-0019)
enum class KvCacheQuantMode : std::uint8_t {
    kFp16Fp16 = 0, // Reference: FP16 Key + FP16 Value
    kQ8Fp16   = 1, // Candidate B: Q8 Key + FP16 Value
    kFp16Q8   = 2, // Candidate C: FP16 Key + Q8 Value
    kQ8Q8     = 3  // Candidate D: Q8 Key + Q8 Value
};

// One already-resolved physical KV head range. Logical page identity is not
// present: attention receives only device addresses and layout metadata.
struct PhysicalKvShardView {
    __half* key_cache = nullptr;
    __half* value_cache = nullptr;
    int8_t* key_cache_q8 = nullptr;
    __half* key_scales = nullptr;
    int8_t* value_cache_q8 = nullptr;
    __half* value_scales = nullptr;
    std::size_t head_begin = 0;
    std::size_t head_count = 0;
};

// Kernel-facing resolved physical view. The legacy fields are the N=1 fast
// path and remain directly usable by existing launches. Multi-shard callers
// consume the compact shard descriptors resolved before entering the hot loop.
struct PhysicalKvView {
    // FP16 pointers (used in FP16 or mixed modes)
    __half* key_cache = nullptr;       // [kKvHeads, capacity, kHeadDim]
    __half* value_cache = nullptr;     // [kKvHeads, capacity, kHeadDim]

    // Q8 pointers and per-head scales (used in Q8 or mixed modes)
    int8_t* key_cache_q8 = nullptr;    // [kKvHeads, capacity, kHeadDim]
    __half* key_scales = nullptr;      // [kKvHeads, capacity]
    int8_t* value_cache_q8 = nullptr;  // [kKvHeads, capacity, kHeadDim]
    __half* value_scales = nullptr;    // [kKvHeads, capacity]

    std::size_t capacity = 0;
    std::size_t head_count_kv = kKvHeads;
    std::size_t head_dim = kHeadDim;
    KvCacheQuantMode quant_mode = KvCacheQuantMode::kFp16Fp16;
    std::array<PhysicalKvShardView, 2> shards{};
    std::uint32_t shard_count = 1;

    [[nodiscard]] bool is_k_q8() const noexcept {
        return quant_mode == KvCacheQuantMode::kQ8Fp16 || quant_mode == KvCacheQuantMode::kQ8Q8;
    }
    [[nodiscard]] bool is_v_q8() const noexcept {
        return quant_mode == KvCacheQuantMode::kFp16Q8 || quant_mode == KvCacheQuantMode::kQ8Q8;
    }
    [[nodiscard]] bool is_single_shard() const noexcept { return shard_count == 1; }
};

using AttentionKvCacheView = PhysicalKvView;

// Device storage for an attention layer's Key and Value cache (FP16 / Q8).
class AttentionLayerKvCacheStorage {
public:
    explicit AttentionLayerKvCacheStorage(
        std::size_t capacity = kDefaultCacheCapacity,
        KvCacheQuantMode quant_mode = KvCacheQuantMode::kFp16Fp16);
    ~AttentionLayerKvCacheStorage();
    AttentionLayerKvCacheStorage(const AttentionLayerKvCacheStorage&) = delete;
    AttentionLayerKvCacheStorage& operator=(const AttentionLayerKvCacheStorage&) = delete;
    AttentionLayerKvCacheStorage(AttentionLayerKvCacheStorage&& other) noexcept;
    AttentionLayerKvCacheStorage& operator=(AttentionLayerKvCacheStorage&& other) noexcept;

    void reset(hipStream_t stream = nullptr);
    void download_key(std::vector<float>& host_key, std::size_t tokens) const;
    void download_value(std::vector<float>& host_value, std::size_t tokens) const;

    // Download/upload raw compact device KV buffers for tokens 0..tokens-1
    void download_raw(void* host_dst, std::size_t tokens, hipStream_t stream = nullptr) const;
    void upload_raw(const void* host_src, std::size_t tokens, hipStream_t stream = nullptr);
    void download_raw_range(void* host_dst, std::size_t begin, std::size_t tokens,
                            hipStream_t stream = nullptr) const;
    void upload_raw_range(const void* host_src, std::size_t begin, std::size_t tokens,
                          hipStream_t stream = nullptr);
    [[nodiscard]] std::size_t raw_tokens_bytes(std::size_t tokens) const noexcept;

    [[nodiscard]] AttentionKvCacheView view() const noexcept {
        return physical_view_;
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] KvCacheQuantMode quant_mode() const noexcept { return quant_mode_; }

    [[nodiscard]] std::size_t key_bytes() const noexcept {
        if (quant_mode_ == KvCacheQuantMode::kQ8Fp16 || quant_mode_ == KvCacheQuantMode::kQ8Q8) {
            return capacity_ * kKvHeads * (kHeadDim * sizeof(int8_t) + sizeof(__half));
        }
        return capacity_ * kKvHeads * kHeadDim * sizeof(__half);
    }

    [[nodiscard]] std::size_t value_bytes() const noexcept {
        if (quant_mode_ == KvCacheQuantMode::kFp16Q8 || quant_mode_ == KvCacheQuantMode::kQ8Q8) {
            return capacity_ * kKvHeads * (kHeadDim * sizeof(int8_t) + sizeof(__half));
        }
        return capacity_ * kKvHeads * kHeadDim * sizeof(__half);
    }

    [[nodiscard]] std::size_t total_bytes() const noexcept {
        return key_bytes() + value_bytes();
    }

    [[nodiscard]] std::size_t single_cache_bytes() const noexcept {
        return capacity_ * kKvHeads * kHeadDim * sizeof(__half);
    }

private:
    std::size_t capacity_ = 0;
    KvCacheQuantMode quant_mode_ = KvCacheQuantMode::kFp16Fp16;

    std::unique_ptr<miinfer::DeviceKvPool> physical_pool_;
    std::unique_ptr<miinfer::ContextSpace> physical_context_;
    std::unique_ptr<miinfer::PlacementPlan> physical_plan_;
    miinfer::DeviceKvBlock physical_block_{};
    PhysicalKvView physical_view_{};

    // FP16 pointers
    __half* d_key_cache_ = nullptr;
    __half* d_value_cache_ = nullptr;

    // Q8 pointers and per-head scales
    int8_t* d_key_cache_q8_ = nullptr;
    __half* d_key_scales_ = nullptr;
    int8_t* d_value_cache_q8_ = nullptr;
    __half* d_value_scales_ = nullptr;
};

} // namespace miinfer::prefill_v2
