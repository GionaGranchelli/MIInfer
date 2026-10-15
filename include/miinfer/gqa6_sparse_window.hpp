#pragma once

#include <cstdint>

#if defined(__HIPCC__)
#define MIINFER_HOST_DEVICE __host__ __device__
#else
#define MIINFER_HOST_DEVICE
#endif

namespace miinfer {

inline constexpr std::uint32_t kGqa6SparseWindow = 2048;
inline constexpr std::uint32_t kGqa6SparseAnchorStride = 512;

MIINFER_HOST_DEVICE constexpr std::uint32_t gqa6_sparse_key_count(std::uint32_t total_length) {
    if (total_length <= kGqa6SparseWindow) return total_length;
    const std::uint32_t window_start = total_length - kGqa6SparseWindow;
    const std::uint32_t anchors = (window_start + kGqa6SparseAnchorStride - 1)
        / kGqa6SparseAnchorStride;
    return kGqa6SparseWindow + anchors;
}

MIINFER_HOST_DEVICE constexpr std::uint32_t gqa6_sparse_key_at(
    std::uint32_t total_length, std::uint32_t ordinal) {
    if (total_length <= kGqa6SparseWindow) return ordinal;
    const std::uint32_t window_start = total_length - kGqa6SparseWindow;
    const std::uint32_t anchors = (window_start + kGqa6SparseAnchorStride - 1)
        / kGqa6SparseAnchorStride;
    return ordinal < anchors ? ordinal * kGqa6SparseAnchorStride
        : window_start + ordinal - anchors;
}

MIINFER_HOST_DEVICE constexpr std::uint32_t gqa6_sparse_split_start(
    std::uint32_t key_count, std::uint32_t split, std::uint32_t splits) {
    const std::uint32_t chunk = (key_count + splits - 1) / splits;
    const std::uint32_t start = split * chunk;
    return start < key_count ? start : key_count;
}

MIINFER_HOST_DEVICE constexpr std::uint32_t gqa6_sparse_split_end(
    std::uint32_t key_count, std::uint32_t split, std::uint32_t splits) {
    const std::uint32_t end = gqa6_sparse_split_start(key_count, split, splits)
        + (key_count + splits - 1) / splits;
    return end < key_count ? end : key_count;
}

}  // namespace miinfer

#undef MIINFER_HOST_DEVICE
