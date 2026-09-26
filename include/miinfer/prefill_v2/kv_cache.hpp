#pragma once

#include "miinfer/prefill_v2/constants.hpp"

#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>
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

// Non-owning view of a single attention layer's KV cache on GPU.
struct AttentionKvCacheView {
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

    [[nodiscard]] bool is_k_q8() const noexcept {
        return quant_mode == KvCacheQuantMode::kQ8Fp16 || quant_mode == KvCacheQuantMode::kQ8Q8;
    }
    [[nodiscard]] bool is_v_q8() const noexcept {
        return quant_mode == KvCacheQuantMode::kFp16Q8 || quant_mode == KvCacheQuantMode::kQ8Q8;
    }
};

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

    [[nodiscard]] AttentionKvCacheView view() const noexcept {
        return AttentionKvCacheView{
            .key_cache = d_key_cache_,
            .value_cache = d_value_cache_,
            .key_cache_q8 = d_key_cache_q8_,
            .key_scales = d_key_scales_,
            .value_cache_q8 = d_value_cache_q8_,
            .value_scales = d_value_scales_,
            .capacity = capacity_,
            .head_count_kv = kKvHeads,
            .head_dim = kHeadDim,
            .quant_mode = quant_mode_
        };
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
