#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace miinfer {

// M12 lab primitive. Workspace is one 64-token chunk per value head and is
// reused across sequential chunk launches.
struct M12GdnChunkWorkspace {
    float* cumulative = nullptr;
    float* pairwise_decay = nullptr;
    float* system = nullptr;
    float* inverse = nullptr;
    float* new_values = nullptr;
    float* decayed_keys = nullptr;
    float* solved_values = nullptr;
    float* solved_keys = nullptr;
    float* corrected_values = nullptr;
    float* intra_attention = nullptr;
};

void launch_m12_gdn_chunk(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* state,
    float* output,
    M12GdnChunkWorkspace workspace,
    std::uint32_t token_count,
    std::uint32_t chunk_start,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    std::uint32_t chunk_size,
    hipStream_t stream = nullptr);

} // namespace miinfer
