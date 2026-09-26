#include "miinfer/prefill_v2/kv_cache.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <utility>

namespace miinfer::prefill_v2 {

AttentionLayerKvCacheStorage::AttentionLayerKvCacheStorage(std::size_t capacity, KvCacheQuantMode quant_mode)
    : capacity_(capacity), quant_mode_(quant_mode) {
    if (capacity_ == 0) {
        throw std::runtime_error("AttentionLayerKvCacheStorage: capacity must be > 0");
    }

    // Allocate Key
    if (quant_mode_ == KvCacheQuantMode::kQ8Fp16 || quant_mode_ == KvCacheQuantMode::kQ8Q8) {
        const std::size_t q8_bytes = capacity_ * kKvHeads * kHeadDim * sizeof(int8_t);
        const std::size_t scale_bytes = capacity_ * kKvHeads * sizeof(__half);
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_key_cache_q8_), q8_bytes));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_key_scales_), scale_bytes));
    } else {
        const std::size_t fp16_bytes = capacity_ * kKvHeads * kHeadDim * sizeof(__half);
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_key_cache_), fp16_bytes));
    }

    // Allocate Value
    if (quant_mode_ == KvCacheQuantMode::kFp16Q8 || quant_mode_ == KvCacheQuantMode::kQ8Q8) {
        const std::size_t q8_bytes = capacity_ * kKvHeads * kHeadDim * sizeof(int8_t);
        const std::size_t scale_bytes = capacity_ * kKvHeads * sizeof(__half);
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_value_cache_q8_), q8_bytes));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_value_scales_), scale_bytes));
    } else {
        const std::size_t fp16_bytes = capacity_ * kKvHeads * kHeadDim * sizeof(__half);
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_value_cache_), fp16_bytes));
    }

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
    if (d_key_cache_q8_ != nullptr) {
        (void)hipFree(d_key_cache_q8_);
        d_key_cache_q8_ = nullptr;
    }
    if (d_key_scales_ != nullptr) {
        (void)hipFree(d_key_scales_);
        d_key_scales_ = nullptr;
    }
    if (d_value_cache_q8_ != nullptr) {
        (void)hipFree(d_value_cache_q8_);
        d_value_cache_q8_ = nullptr;
    }
    if (d_value_scales_ != nullptr) {
        (void)hipFree(d_value_scales_);
        d_value_scales_ = nullptr;
    }
}

AttentionLayerKvCacheStorage::AttentionLayerKvCacheStorage(AttentionLayerKvCacheStorage&& other) noexcept
    : capacity_(other.capacity_),
      quant_mode_(other.quant_mode_),
      d_key_cache_(std::exchange(other.d_key_cache_, nullptr)),
      d_value_cache_(std::exchange(other.d_value_cache_, nullptr)),
      d_key_cache_q8_(std::exchange(other.d_key_cache_q8_, nullptr)),
      d_key_scales_(std::exchange(other.d_key_scales_, nullptr)),
      d_value_cache_q8_(std::exchange(other.d_value_cache_q8_, nullptr)),
      d_value_scales_(std::exchange(other.d_value_scales_, nullptr)) {}

AttentionLayerKvCacheStorage& AttentionLayerKvCacheStorage::operator=(AttentionLayerKvCacheStorage&& other) noexcept {
    if (this != &other) {
        if (d_key_cache_ != nullptr) (void)hipFree(d_key_cache_);
        if (d_value_cache_ != nullptr) (void)hipFree(d_value_cache_);
        if (d_key_cache_q8_ != nullptr) (void)hipFree(d_key_cache_q8_);
        if (d_key_scales_ != nullptr) (void)hipFree(d_key_scales_);
        if (d_value_cache_q8_ != nullptr) (void)hipFree(d_value_cache_q8_);
        if (d_value_scales_ != nullptr) (void)hipFree(d_value_scales_);

        capacity_ = other.capacity_;
        quant_mode_ = other.quant_mode_;
        d_key_cache_ = std::exchange(other.d_key_cache_, nullptr);
        d_value_cache_ = std::exchange(other.d_value_cache_, nullptr);
        d_key_cache_q8_ = std::exchange(other.d_key_cache_q8_, nullptr);
        d_key_scales_ = std::exchange(other.d_key_scales_, nullptr);
        d_value_cache_q8_ = std::exchange(other.d_value_cache_q8_, nullptr);
        d_value_scales_ = std::exchange(other.d_value_scales_, nullptr);
    }
    return *this;
}

void AttentionLayerKvCacheStorage::reset(hipStream_t stream) {
    if (d_key_cache_ != nullptr) {
        const std::size_t bytes = capacity_ * kKvHeads * kHeadDim * sizeof(__half);
        if (stream != nullptr) MIINFER_HIP_CHECK(hipMemsetAsync(d_key_cache_, 0, bytes, stream));
        else MIINFER_HIP_CHECK(hipMemset(d_key_cache_, 0, bytes));
    }
    if (d_value_cache_ != nullptr) {
        const std::size_t bytes = capacity_ * kKvHeads * kHeadDim * sizeof(__half);
        if (stream != nullptr) MIINFER_HIP_CHECK(hipMemsetAsync(d_value_cache_, 0, bytes, stream));
        else MIINFER_HIP_CHECK(hipMemset(d_value_cache_, 0, bytes));
    }
    if (d_key_cache_q8_ != nullptr) {
        const std::size_t bytes = capacity_ * kKvHeads * kHeadDim * sizeof(int8_t);
        if (stream != nullptr) MIINFER_HIP_CHECK(hipMemsetAsync(d_key_cache_q8_, 0, bytes, stream));
        else MIINFER_HIP_CHECK(hipMemset(d_key_cache_q8_, 0, bytes));
    }
    if (d_key_scales_ != nullptr) {
        const std::size_t bytes = capacity_ * kKvHeads * sizeof(__half);
        if (stream != nullptr) MIINFER_HIP_CHECK(hipMemsetAsync(d_key_scales_, 0, bytes, stream));
        else MIINFER_HIP_CHECK(hipMemset(d_key_scales_, 0, bytes));
    }
    if (d_value_cache_q8_ != nullptr) {
        const std::size_t bytes = capacity_ * kKvHeads * kHeadDim * sizeof(int8_t);
        if (stream != nullptr) MIINFER_HIP_CHECK(hipMemsetAsync(d_value_cache_q8_, 0, bytes, stream));
        else MIINFER_HIP_CHECK(hipMemset(d_value_cache_q8_, 0, bytes));
    }
    if (d_value_scales_ != nullptr) {
        const std::size_t bytes = capacity_ * kKvHeads * sizeof(__half);
        if (stream != nullptr) MIINFER_HIP_CHECK(hipMemsetAsync(d_value_scales_, 0, bytes, stream));
        else MIINFER_HIP_CHECK(hipMemset(d_value_scales_, 0, bytes));
    }
}

void AttentionLayerKvCacheStorage::download_key(std::vector<float>& host_key, std::size_t tokens) const {
    if (tokens > capacity_) {
        throw std::runtime_error("AttentionLayerKvCacheStorage: tokens exceeds capacity");
    }
    const std::size_t count = kKvHeads * tokens * kHeadDim;
    host_key.resize(count);

    if (d_key_cache_ != nullptr) {
        std::vector<__half> half_buf(count);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            const __half* src = d_key_cache_ + h * capacity_ * kHeadDim;
            __half* dst = half_buf.data() + h * tokens * kHeadDim;
            MIINFER_HIP_CHECK(hipMemcpy(dst, src, tokens * kHeadDim * sizeof(__half), hipMemcpyDeviceToHost));
        }
        for (std::size_t i = 0; i < count; ++i) {
            host_key[i] = static_cast<float>(half_buf[i]);
        }
    } else if (d_key_cache_q8_ != nullptr && d_key_scales_ != nullptr) {
        std::vector<int8_t> q8_buf(count);
        std::vector<__half> scale_buf(kKvHeads * tokens);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            const int8_t* src_q = d_key_cache_q8_ + h * capacity_ * kHeadDim;
            int8_t* dst_q = q8_buf.data() + h * tokens * kHeadDim;
            MIINFER_HIP_CHECK(hipMemcpy(dst_q, src_q, tokens * kHeadDim * sizeof(int8_t), hipMemcpyDeviceToHost));

            const __half* src_s = d_key_scales_ + h * capacity_;
            __half* dst_s = scale_buf.data() + h * tokens;
            MIINFER_HIP_CHECK(hipMemcpy(dst_s, src_s, tokens * sizeof(__half), hipMemcpyDeviceToHost));
        }
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            for (std::size_t t = 0; t < tokens; ++t) {
                const float scale = static_cast<float>(scale_buf[h * tokens + t]);
                for (std::size_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t idx = (h * tokens + t) * kHeadDim + d;
                    host_key[idx] = static_cast<float>(q8_buf[idx]) * scale;
                }
            }
        }
    }
}

void AttentionLayerKvCacheStorage::download_value(std::vector<float>& host_value, std::size_t tokens) const {
    if (tokens > capacity_) {
        throw std::runtime_error("AttentionLayerKvCacheStorage: tokens exceeds capacity");
    }
    const std::size_t count = kKvHeads * tokens * kHeadDim;
    host_value.resize(count);

    if (d_value_cache_ != nullptr) {
        std::vector<__half> half_buf(count);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            const __half* src = d_value_cache_ + h * capacity_ * kHeadDim;
            __half* dst = half_buf.data() + h * tokens * kHeadDim;
            MIINFER_HIP_CHECK(hipMemcpy(dst, src, tokens * kHeadDim * sizeof(__half), hipMemcpyDeviceToHost));
        }
        for (std::size_t i = 0; i < count; ++i) {
            host_value[i] = static_cast<float>(half_buf[i]);
        }
    } else if (d_value_cache_q8_ != nullptr && d_value_scales_ != nullptr) {
        std::vector<int8_t> q8_buf(count);
        std::vector<__half> scale_buf(kKvHeads * tokens);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            const int8_t* src_q = d_value_cache_q8_ + h * capacity_ * kHeadDim;
            int8_t* dst_q = q8_buf.data() + h * tokens * kHeadDim;
            MIINFER_HIP_CHECK(hipMemcpy(dst_q, src_q, tokens * kHeadDim * sizeof(int8_t), hipMemcpyDeviceToHost));

            const __half* src_s = d_value_scales_ + h * capacity_;
            __half* dst_s = scale_buf.data() + h * tokens;
            MIINFER_HIP_CHECK(hipMemcpy(dst_s, src_s, tokens * sizeof(__half), hipMemcpyDeviceToHost));
        }
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            for (std::size_t t = 0; t < tokens; ++t) {
                const float scale = static_cast<float>(scale_buf[h * tokens + t]);
                for (std::size_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t idx = (h * tokens + t) * kHeadDim + d;
                    host_value[idx] = static_cast<float>(q8_buf[idx]) * scale;
                }
            }
        }
    }
}

} // namespace miinfer::prefill_v2
