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

    int device = 0;
    MIINFER_HIP_CHECK(hipGetDevice(&device));
    physical_pool_ = std::make_unique<miinfer::DeviceKvPool>(device, total_bytes());
    physical_block_ = physical_pool_->allocate(total_bytes(), alignof(std::max_align_t));
    auto* base = static_cast<std::byte*>(physical_block_.data);
    std::size_t offset = 0;

    // Allocate Key inside one placement-owned physical block.
    if (quant_mode_ == KvCacheQuantMode::kQ8Fp16 || quant_mode_ == KvCacheQuantMode::kQ8Q8) {
        const std::size_t q8_bytes = capacity_ * kKvHeads * kHeadDim * sizeof(int8_t);
        const std::size_t scale_bytes = capacity_ * kKvHeads * sizeof(__half);
        d_key_cache_q8_ = reinterpret_cast<int8_t*>(base + offset);
        offset += q8_bytes;
        d_key_scales_ = reinterpret_cast<__half*>(base + offset);
        offset += scale_bytes;
    } else {
        const std::size_t fp16_bytes = capacity_ * kKvHeads * kHeadDim * sizeof(__half);
        d_key_cache_ = reinterpret_cast<__half*>(base + offset);
        offset += fp16_bytes;
    }

    // Allocate Value
    if (quant_mode_ == KvCacheQuantMode::kFp16Q8 || quant_mode_ == KvCacheQuantMode::kQ8Q8) {
        const std::size_t q8_bytes = capacity_ * kKvHeads * kHeadDim * sizeof(int8_t);
        const std::size_t scale_bytes = capacity_ * kKvHeads * sizeof(__half);
        d_value_cache_q8_ = reinterpret_cast<int8_t*>(base + offset);
        offset += q8_bytes;
        d_value_scales_ = reinterpret_cast<__half*>(base + offset);
        offset += scale_bytes;
    } else {
        const std::size_t fp16_bytes = capacity_ * kKvHeads * kHeadDim * sizeof(__half);
        d_value_cache_ = reinterpret_cast<__half*>(base + offset);
        offset += fp16_bytes;
    }

    physical_context_ = std::make_unique<miinfer::ContextSpace>(capacity_, capacity_);
    const auto logical = physical_context_->append(capacity_);
    physical_context_->remap(physical_context_->page_id_at(logical.begin),
                             {physical_block_.data, physical_block_.bytes});
    physical_plan_ = std::make_unique<miinfer::PlacementPlan>(*physical_context_);
    physical_plan_->place(miinfer::DeviceKvShard(
        device, physical_context_->page_id_at(logical.begin), logical,
        {physical_block_.data, 0, physical_block_.bytes}, {0, kKvHeads}));
    const auto resolved = physical_plan_->resolve(logical, {0, kKvHeads});
    const auto physical = resolved.front().physical_range();
    if (physical.data != physical_block_.data || physical.offset != 0
        || physical.bytes != physical_block_.bytes) {
        throw std::logic_error("AttentionLayerKvCacheStorage: physical view resolution mismatch");
    }
    auto* resolved_base = const_cast<std::byte*>(static_cast<const std::byte*>(physical.data))
        + physical.offset;
    const auto resolve_pointer = [&](auto* pointer) {
        if (pointer == nullptr) return pointer;
        const auto byte_offset = reinterpret_cast<const std::byte*>(pointer) - base;
        return reinterpret_cast<decltype(pointer)>(resolved_base + byte_offset);
    };

    physical_view_.key_cache = resolve_pointer(d_key_cache_);
    physical_view_.value_cache = resolve_pointer(d_value_cache_);
    physical_view_.key_cache_q8 = resolve_pointer(d_key_cache_q8_);
    physical_view_.key_scales = resolve_pointer(d_key_scales_);
    physical_view_.value_cache_q8 = resolve_pointer(d_value_cache_q8_);
    physical_view_.value_scales = resolve_pointer(d_value_scales_);
    physical_view_.capacity = capacity_;
    physical_view_.head_count_kv = kKvHeads;
    physical_view_.head_dim = kHeadDim;
    physical_view_.quant_mode = quant_mode_;
    physical_view_.shards[0] = {
        physical_view_.key_cache, physical_view_.value_cache,
        physical_view_.key_cache_q8, physical_view_.key_scales,
        physical_view_.value_cache_q8, physical_view_.value_scales, 0, kKvHeads};

    reset(nullptr);
}

AttentionLayerKvCacheStorage::~AttentionLayerKvCacheStorage() = default;

AttentionLayerKvCacheStorage::AttentionLayerKvCacheStorage(AttentionLayerKvCacheStorage&& other) noexcept
    : capacity_(other.capacity_),
      quant_mode_(other.quant_mode_),
      physical_pool_(std::move(other.physical_pool_)),
      physical_context_(std::move(other.physical_context_)),
      physical_plan_(std::move(other.physical_plan_)),
      physical_block_(other.physical_block_),
      physical_view_(other.physical_view_),
      d_key_cache_(std::exchange(other.d_key_cache_, nullptr)),
      d_value_cache_(std::exchange(other.d_value_cache_, nullptr)),
      d_key_cache_q8_(std::exchange(other.d_key_cache_q8_, nullptr)),
      d_key_scales_(std::exchange(other.d_key_scales_, nullptr)),
      d_value_cache_q8_(std::exchange(other.d_value_cache_q8_, nullptr)),
      d_value_scales_(std::exchange(other.d_value_scales_, nullptr)) {}

AttentionLayerKvCacheStorage& AttentionLayerKvCacheStorage::operator=(AttentionLayerKvCacheStorage&& other) noexcept {
    if (this != &other) {
        capacity_ = other.capacity_;
        quant_mode_ = other.quant_mode_;
        physical_pool_ = std::move(other.physical_pool_);
        physical_context_ = std::move(other.physical_context_);
        physical_plan_ = std::move(other.physical_plan_);
        physical_block_ = other.physical_block_;
        physical_view_ = other.physical_view_;
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

std::size_t AttentionLayerKvCacheStorage::raw_tokens_bytes(std::size_t tokens) const noexcept {
    if (quant_mode_ == KvCacheQuantMode::kQ8Fp16) {
        return tokens * kKvHeads * (kHeadDim * sizeof(int8_t) + sizeof(__half))
             + tokens * kKvHeads * kHeadDim * sizeof(__half);
    } else if (quant_mode_ == KvCacheQuantMode::kFp16Q8) {
        return tokens * kKvHeads * kHeadDim * sizeof(__half)
             + tokens * kKvHeads * (kHeadDim * sizeof(int8_t) + sizeof(__half));
    } else if (quant_mode_ == KvCacheQuantMode::kQ8Q8) {
        return 2 * tokens * kKvHeads * (kHeadDim * sizeof(int8_t) + sizeof(__half));
    }
    return 2 * tokens * kKvHeads * kHeadDim * sizeof(__half);
}

void AttentionLayerKvCacheStorage::download_raw(void* host_dst, std::size_t tokens, hipStream_t stream) const {
    if (tokens > capacity_) {
        throw std::runtime_error("AttentionLayerKvCacheStorage::download_raw: tokens exceeds capacity");
    }
    if (tokens == 0 || host_dst == nullptr) return;

    std::uint8_t* dst = static_cast<std::uint8_t*>(host_dst);
    std::size_t offset = 0;

    if (d_key_cache_ != nullptr) {
        const std::size_t slice_bytes = tokens * kHeadDim * sizeof(__half);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            const __half* src = d_key_cache_ + h * capacity_ * kHeadDim;
            if (stream != nullptr) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst + offset, src, slice_bytes, hipMemcpyDeviceToHost, stream));
            } else {
                MIINFER_HIP_CHECK(hipMemcpy(dst + offset, src, slice_bytes, hipMemcpyDeviceToHost));
            }
            offset += slice_bytes;
        }
    } else if (d_key_cache_q8_ != nullptr && d_key_scales_ != nullptr) {
        const std::size_t q_slice = tokens * kHeadDim * sizeof(int8_t);
        const std::size_t s_slice = tokens * sizeof(__half);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            const int8_t* src_q = d_key_cache_q8_ + h * capacity_ * kHeadDim;
            const __half* src_s = d_key_scales_ + h * capacity_;
            if (stream != nullptr) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst + offset, src_q, q_slice, hipMemcpyDeviceToHost, stream));
                offset += q_slice;
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst + offset, src_s, s_slice, hipMemcpyDeviceToHost, stream));
                offset += s_slice;
            } else {
                MIINFER_HIP_CHECK(hipMemcpy(dst + offset, src_q, q_slice, hipMemcpyDeviceToHost));
                offset += q_slice;
                MIINFER_HIP_CHECK(hipMemcpy(dst + offset, src_s, s_slice, hipMemcpyDeviceToHost));
                offset += s_slice;
            }
        }
    }

    if (d_value_cache_ != nullptr) {
        const std::size_t slice_bytes = tokens * kHeadDim * sizeof(__half);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            const __half* src = d_value_cache_ + h * capacity_ * kHeadDim;
            if (stream != nullptr) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst + offset, src, slice_bytes, hipMemcpyDeviceToHost, stream));
            } else {
                MIINFER_HIP_CHECK(hipMemcpy(dst + offset, src, slice_bytes, hipMemcpyDeviceToHost));
            }
            offset += slice_bytes;
        }
    } else if (d_value_cache_q8_ != nullptr && d_value_scales_ != nullptr) {
        const std::size_t q_slice = tokens * kHeadDim * sizeof(int8_t);
        const std::size_t s_slice = tokens * sizeof(__half);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            const int8_t* src_q = d_value_cache_q8_ + h * capacity_ * kHeadDim;
            const __half* src_s = d_value_scales_ + h * capacity_;
            if (stream != nullptr) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst + offset, src_q, q_slice, hipMemcpyDeviceToHost, stream));
                offset += q_slice;
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst + offset, src_s, s_slice, hipMemcpyDeviceToHost, stream));
                offset += s_slice;
            } else {
                MIINFER_HIP_CHECK(hipMemcpy(dst + offset, src_q, q_slice, hipMemcpyDeviceToHost));
                offset += q_slice;
                MIINFER_HIP_CHECK(hipMemcpy(dst + offset, src_s, s_slice, hipMemcpyDeviceToHost));
                offset += s_slice;
            }
        }
    }
}

void AttentionLayerKvCacheStorage::upload_raw(const void* host_src, std::size_t tokens, hipStream_t stream) {
    if (tokens > capacity_) {
        throw std::runtime_error("AttentionLayerKvCacheStorage::upload_raw: tokens exceeds capacity");
    }
    if (tokens == 0 || host_src == nullptr) return;

    const std::uint8_t* src = static_cast<const std::uint8_t*>(host_src);
    std::size_t offset = 0;

    if (d_key_cache_ != nullptr) {
        const std::size_t slice_bytes = tokens * kHeadDim * sizeof(__half);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            __half* dst = d_key_cache_ + h * capacity_ * kHeadDim;
            if (stream != nullptr) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst, src + offset, slice_bytes, hipMemcpyHostToDevice, stream));
            } else {
                MIINFER_HIP_CHECK(hipMemcpy(dst, src + offset, slice_bytes, hipMemcpyHostToDevice));
            }
            offset += slice_bytes;
        }
    } else if (d_key_cache_q8_ != nullptr && d_key_scales_ != nullptr) {
        const std::size_t q_slice = tokens * kHeadDim * sizeof(int8_t);
        const std::size_t s_slice = tokens * sizeof(__half);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            int8_t* dst_q = d_key_cache_q8_ + h * capacity_ * kHeadDim;
            __half* dst_s = d_key_scales_ + h * capacity_;
            if (stream != nullptr) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst_q, src + offset, q_slice, hipMemcpyHostToDevice, stream));
                offset += q_slice;
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst_s, src + offset, s_slice, hipMemcpyHostToDevice, stream));
                offset += s_slice;
            } else {
                MIINFER_HIP_CHECK(hipMemcpy(dst_q, src + offset, q_slice, hipMemcpyHostToDevice));
                offset += q_slice;
                MIINFER_HIP_CHECK(hipMemcpy(dst_s, src + offset, s_slice, hipMemcpyHostToDevice));
                offset += s_slice;
            }
        }
    }

    if (d_value_cache_ != nullptr) {
        const std::size_t slice_bytes = tokens * kHeadDim * sizeof(__half);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            __half* dst = d_value_cache_ + h * capacity_ * kHeadDim;
            if (stream != nullptr) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst, src + offset, slice_bytes, hipMemcpyHostToDevice, stream));
            } else {
                MIINFER_HIP_CHECK(hipMemcpy(dst, src + offset, slice_bytes, hipMemcpyHostToDevice));
            }
            offset += slice_bytes;
        }
    } else if (d_value_cache_q8_ != nullptr && d_value_scales_ != nullptr) {
        const std::size_t q_slice = tokens * kHeadDim * sizeof(int8_t);
        const std::size_t s_slice = tokens * sizeof(__half);
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            int8_t* dst_q = d_value_cache_q8_ + h * capacity_ * kHeadDim;
            __half* dst_s = d_value_scales_ + h * capacity_;
            if (stream != nullptr) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst_q, src + offset, q_slice, hipMemcpyHostToDevice, stream));
                offset += q_slice;
                MIINFER_HIP_CHECK(hipMemcpyAsync(dst_s, src + offset, s_slice, hipMemcpyHostToDevice, stream));
                offset += s_slice;
            } else {
                MIINFER_HIP_CHECK(hipMemcpy(dst_q, src + offset, q_slice, hipMemcpyHostToDevice));
                offset += q_slice;
                MIINFER_HIP_CHECK(hipMemcpy(dst_s, src + offset, s_slice, hipMemcpyHostToDevice));
                offset += s_slice;
            }
        }
    }
}

void AttentionLayerKvCacheStorage::download_raw_range(
    void* host_dst, std::size_t begin, std::size_t tokens, hipStream_t stream) const {
    if (begin > capacity_ || tokens > capacity_ - begin) {
        throw std::runtime_error("AttentionLayerKvCacheStorage::download_raw_range: range exceeds capacity");
    }
    if (tokens == 0 || host_dst == nullptr) return;

    auto copy = [stream](void* dst, const void* src, std::size_t bytes) {
        if (stream != nullptr) MIINFER_HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream));
        else MIINFER_HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
    };
    auto* dst = static_cast<std::uint8_t*>(host_dst);
    const bool key_q8 = quant_mode_ == KvCacheQuantMode::kQ8Fp16
        || quant_mode_ == KvCacheQuantMode::kQ8Q8;
    const bool value_q8 = quant_mode_ == KvCacheQuantMode::kFp16Q8
        || quant_mode_ == KvCacheQuantMode::kQ8Q8;
    const std::size_t fp16_values = tokens * kHeadDim * sizeof(__half);
    const std::size_t q_values = tokens * kHeadDim * sizeof(std::int8_t);
    const std::size_t scales = tokens * sizeof(__half);
    std::size_t offset = 0;

    if (!key_q8) {
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            copy(dst + offset,
                 d_key_cache_ + h * capacity_ * kHeadDim + begin * kHeadDim,
                 fp16_values);
            offset += fp16_values;
        }
    } else {
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            copy(dst + offset,
                 d_key_cache_q8_ + h * capacity_ * kHeadDim + begin * kHeadDim,
                 q_values);
            offset += q_values;
            copy(dst + offset, d_key_scales_ + h * capacity_ + begin, scales);
            offset += scales;
        }
    }
    if (!value_q8) {
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            copy(dst + offset,
                 d_value_cache_ + h * capacity_ * kHeadDim + begin * kHeadDim,
                 fp16_values);
            offset += fp16_values;
        }
    } else {
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            copy(dst + offset,
                 d_value_cache_q8_ + h * capacity_ * kHeadDim + begin * kHeadDim,
                 q_values);
            offset += q_values;
            copy(dst + offset, d_value_scales_ + h * capacity_ + begin, scales);
            offset += scales;
        }
    }
}

void AttentionLayerKvCacheStorage::upload_raw_range(
    const void* host_src, std::size_t begin, std::size_t tokens, hipStream_t stream) {
    if (begin > capacity_ || tokens > capacity_ - begin) {
        throw std::runtime_error("AttentionLayerKvCacheStorage::upload_raw_range: range exceeds capacity");
    }
    if (tokens == 0 || host_src == nullptr) return;

    auto copy = [stream](void* dst, const void* src, std::size_t bytes) {
        if (stream != nullptr) MIINFER_HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream));
        else MIINFER_HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
    };
    const auto* src = static_cast<const std::uint8_t*>(host_src);
    const bool key_q8 = quant_mode_ == KvCacheQuantMode::kQ8Fp16
        || quant_mode_ == KvCacheQuantMode::kQ8Q8;
    const bool value_q8 = quant_mode_ == KvCacheQuantMode::kFp16Q8
        || quant_mode_ == KvCacheQuantMode::kQ8Q8;
    const std::size_t fp16_values = tokens * kHeadDim * sizeof(__half);
    const std::size_t q_values = tokens * kHeadDim * sizeof(std::int8_t);
    const std::size_t scales = tokens * sizeof(__half);
    std::size_t offset = 0;

    if (!key_q8) {
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            copy(d_key_cache_ + h * capacity_ * kHeadDim + begin * kHeadDim,
                 src + offset, fp16_values);
            offset += fp16_values;
        }
    } else {
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            copy(d_key_cache_q8_ + h * capacity_ * kHeadDim + begin * kHeadDim,
                 src + offset, q_values);
            offset += q_values;
            copy(d_key_scales_ + h * capacity_ + begin, src + offset, scales);
            offset += scales;
        }
    }
    if (!value_q8) {
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            copy(d_value_cache_ + h * capacity_ * kHeadDim + begin * kHeadDim,
                 src + offset, fp16_values);
            offset += fp16_values;
        }
    } else {
        for (std::size_t h = 0; h < kKvHeads; ++h) {
            copy(d_value_cache_q8_ + h * capacity_ * kHeadDim + begin * kHeadDim,
                 src + offset, q_values);
            offset += q_values;
            copy(d_value_scales_ + h * capacity_ + begin, src + offset, scales);
            offset += scales;
        }
    }
}

} // namespace miinfer::prefill_v2
