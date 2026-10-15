#pragma once

#include <cstdint>

#if defined(__HIPCC__)
#define MIINFER_HOST_DEVICE __host__ __device__
#else
#define MIINFER_HOST_DEVICE
#endif

namespace miinfer {

inline constexpr std::uint32_t kGqa6QuerySparseWindow = 2048;
inline constexpr std::uint32_t kGqa6QuerySparseBlock = 512;
inline constexpr std::uint32_t kGqa6QuerySparseSamples = 16;
inline constexpr std::uint32_t kGqa6QuerySparseTopBlocks = 16;

MIINFER_HOST_DEVICE constexpr std::uint32_t gqa6_query_sparse_local_start(
    std::uint32_t total_length) {
    if (total_length <= kGqa6QuerySparseWindow) return 0;
    return ((total_length - kGqa6QuerySparseWindow) / kGqa6QuerySparseBlock)
        * kGqa6QuerySparseBlock;
}

MIINFER_HOST_DEVICE constexpr std::uint32_t gqa6_query_sparse_key_count(
    std::uint32_t total_length, std::uint32_t selected_blocks) {
    const std::uint32_t local_start = gqa6_query_sparse_local_start(total_length);
    if (local_start == 0) return total_length;
    return selected_blocks * kGqa6QuerySparseBlock + total_length - local_start;
}

MIINFER_HOST_DEVICE constexpr std::uint32_t gqa6_query_sparse_key_at(
    std::uint32_t total_length, const std::uint32_t* selected_blocks,
    std::uint32_t selected_count, std::uint32_t ordinal) {
    const std::uint32_t local_start = gqa6_query_sparse_local_start(total_length);
    if (local_start == 0) return ordinal;
    const std::uint32_t selected_keys = selected_count * kGqa6QuerySparseBlock;
    if (ordinal < selected_keys) {
        return selected_blocks[ordinal / kGqa6QuerySparseBlock]
            * kGqa6QuerySparseBlock + ordinal % kGqa6QuerySparseBlock;
    }
    return local_start + ordinal - selected_keys;
}

}  // namespace miinfer

#undef MIINFER_HOST_DEVICE
