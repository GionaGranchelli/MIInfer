#include "miinfer/prefill_v2/kv_cache.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <utility>

namespace miinfer::prefill_v2 {

AttentionLayerKvCacheStorage::AttentionLayerKvCacheStorage(std::size_t capacity)
    : capacity_(capacity) {
    if (capacity_ == 0) {
        throw std::runtime_error("AttentionLayerKvCacheStorage: capacity must be > 0");
    }

    const std::size_t bytes = single_cache_bytes();
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_key_cache_), bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_value_cache_), bytes));
    reset(nullptr);
}

AttentionLayerKvCacheStorage::~AttentionLayerKvCacheStorage() {
    if (d_key_cache_ != nullptr) {
        (void)hipFree(d_key_cache_);
        d_key_cache_ = nullptr;
    }
    if (d_value_cache_ != nullptr) {
        (void)hipFree(d_value_cache_);
        d_value_cache_ = nullptr;
    }
}

AttentionLayerKvCacheStorage::AttentionLayerKvCacheStorage(AttentionLayerKvCacheStorage&& other) noexcept
    : capacity_(other.capacity_),
      d_key_cache_(std::exchange(other.d_key_cache_, nullptr)),
      d_value_cache_(std::exchange(other.d_value_cache_, nullptr)) {}

AttentionLayerKvCacheStorage& AttentionLayerKvCacheStorage::operator=(AttentionLayerKvCacheStorage&& other) noexcept {
    if (this != &other) {
        if (d_key_cache_ != nullptr) (void)hipFree(d_key_cache_);
        if (d_value_cache_ != nullptr) (void)hipFree(d_value_cache_);
        capacity_ = other.capacity_;
        d_key_cache_ = std::exchange(other.d_key_cache_, nullptr);
        d_value_cache_ = std::exchange(other.d_value_cache_, nullptr);
    }
    return *this;
}

void AttentionLayerKvCacheStorage::reset(hipStream_t stream) {
    const std::size_t bytes = single_cache_bytes();
    if (stream != nullptr) {
        MIINFER_HIP_CHECK(hipMemsetAsync(d_key_cache_, 0, bytes, stream));
        MIINFER_HIP_CHECK(hipMemsetAsync(d_value_cache_, 0, bytes, stream));
    } else {
        MIINFER_HIP_CHECK(hipMemset(d_key_cache_, 0, bytes));
        MIINFER_HIP_CHECK(hipMemset(d_value_cache_, 0, bytes));
    }
}

void AttentionLayerKvCacheStorage::download_key(std::vector<float>& host_key, std::size_t tokens) const {
    if (tokens > capacity_) {
        throw std::runtime_error("AttentionLayerKvCacheStorage: tokens exceeds capacity");
    }
    const std::size_t count = kKvHeads * tokens * kHeadDim;
    std::vector<__half> half_buf(count);
    // In our layout: [head, capacity, dim]
    for (std::size_t h = 0; h < kKvHeads; ++h) {
        const __half* src = d_key_cache_ + h * capacity_ * kHeadDim;
        __half* dst = half_buf.data() + h * tokens * kHeadDim;
        MIINFER_HIP_CHECK(hipMemcpy(dst, src, tokens * kHeadDim * sizeof(__half), hipMemcpyDeviceToHost));
    }
    host_key.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        host_key[i] = static_cast<float>(half_buf[i]);
    }
}

void AttentionLayerKvCacheStorage::download_value(std::vector<float>& host_value, std::size_t tokens) const {
    if (tokens > capacity_) {
        throw std::runtime_error("AttentionLayerKvCacheStorage: tokens exceeds capacity");
    }
    const std::size_t count = kKvHeads * tokens * kHeadDim;
    std::vector<__half> half_buf(count);
    for (std::size_t h = 0; h < kKvHeads; ++h) {
        const __half* src = d_value_cache_ + h * capacity_ * kHeadDim;
        __half* dst = half_buf.data() + h * tokens * kHeadDim;
        MIINFER_HIP_CHECK(hipMemcpy(dst, src, tokens * kHeadDim * sizeof(__half), hipMemcpyDeviceToHost));
    }
    host_value.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        host_value[i] = static_cast<float>(half_buf[i]);
    }
}

} // namespace miinfer::prefill_v2
