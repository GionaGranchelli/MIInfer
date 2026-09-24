#pragma once

#include "miinfer/prefill_v2/constants.hpp"

#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace miinfer::prefill_v2 {

// Non-owning view of a single attention layer's KV cache on GPU.
struct AttentionKvCacheView {
    __half* key_cache = nullptr;    // [kKvHeads, capacity, kHeadDim]
    __half* value_cache = nullptr;  // [kKvHeads, capacity, kHeadDim]
    std::size_t capacity = 0;
    std::size_t head_count_kv = kKvHeads;
    std::size_t head_dim = kHeadDim;
};

// Device storage for an attention layer's FP16 Key and Value cache.
class AttentionLayerKvCacheStorage {
public:
    explicit AttentionLayerKvCacheStorage(std::size_t capacity = kDefaultCacheCapacity);
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
            d_key_cache_,
            d_value_cache_,
            capacity_,
            kKvHeads,
            kHeadDim
        };
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t single_cache_bytes() const noexcept {
        return capacity_ * kKvHeads * kHeadDim * sizeof(__half);
    }
    [[nodiscard]] std::size_t total_bytes() const noexcept {
        return 2 * single_cache_bytes();
    }

private:
    std::size_t capacity_ = 0;
    __half* d_key_cache_ = nullptr;
    __half* d_value_cache_ = nullptr;
};

} // namespace miinfer::prefill_v2
