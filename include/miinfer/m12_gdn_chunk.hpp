#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace miinfer {

// M12 lab primitive. Workspace is one 64-token chunk per value head and is
// reused across sequential chunk launches.
struct M12GdnChunkWorkspace {
    float* new_values = nullptr;
    float* decayed_keys = nullptr;
    float* solved_values = nullptr;
    float* solved_keys = nullptr;
    float* corrected_values = nullptr;
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

// mx-style wide scan: one block owns a value-head/value tile and keeps the
// recurrent state shard in registers across the complete token chunk. The
// public state remains MIInfer's [head][key][value] layout.
void launch_m12_gdn_direct(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* state,
    float* output,
    std::uint32_t token_count,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    hipStream_t stream = nullptr);

// Adapted from mxxm-t/mx-llama.cpp @ 2e9d29fe, gated_delta_net_chunk.cu.
// The fixed Qwen3.8 path keeps the transposed state shard in registers across
// the complete token sequence and uses one 64x2 Wave64 block per column tile.
void launch_mx_gdn_chunk(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* state,
    float* output,
    std::uint32_t token_count,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    hipStream_t stream = nullptr);

// Same scan with the pinned oracle's persistent [value][key] recurrent-state
// layout. The caller must use that layout for all state consumers.
void launch_mx_gdn_chunk_external_state(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* state,
    float* output,
    std::uint32_t token_count,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    hipStream_t stream = nullptr);

void launch_m12_gdn_postprocess(
    const float* recurrent_output,
    const float* gate,
    const float* ssm_norm,
    float* gated_output,
    std::uint32_t token_count,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    float epsilon,
    hipStream_t stream = nullptr);

} // namespace miinfer
