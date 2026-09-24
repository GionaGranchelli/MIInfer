#pragma once

#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/m12_gdn_chunk.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace miinfer::prefill_v2 {

// Non-owning layout of pre-allocated scratch memory for a Prefill V2 recurrent layer.
struct RecurrentLayerWorkspace {
    // Activations
    float* normalized = nullptr;          // [max_tokens, kHidden]
    float* qkv = nullptr;                 // [max_tokens, kChannels]
    float* gate = nullptr;                // [max_tokens, kInner]
    float* raw_beta = nullptr;            // [max_tokens, kVHeads]
    float* raw_alpha = nullptr;           // [max_tokens, kVHeads]
    float* beta = nullptr;                // [max_tokens, kVHeads]
    float* decay = nullptr;               // [max_tokens, kVHeads]

    // Conv split outputs
    float* query = nullptr;               // [max_tokens, kKHeads * kState]
    float* key = nullptr;                 // [max_tokens, kKHeads * kState]
    float* value = nullptr;               // [max_tokens, kVHeads * kState]

    // GDN chunkwise scratch and outputs
    M12GdnChunkWorkspace gdn_scratch{};  // 5 buffers each [kVHeads * kGdnChunkSize * kState] floats
    float* gdn_raw_output = nullptr;      // [max_tokens, kVHeads * kState]
    float* gated_output = nullptr;        // [max_tokens, kVHeads * kState]

    // SSM output & residual
    float* ssm_output = nullptr;          // [max_tokens, kHidden]
    float* residual = nullptr;            // [max_tokens, kHidden]
    float* post_normalized = nullptr;     // [max_tokens, kHidden]

    // MMQ Q8_1 quantization blocks (compact Mx contract)
    MxQ8_1MmqBlock* mmq_q8 = nullptr;   // [max_tokens * (max(kHidden, kInner, kFfnInner) / 128)] blocks

    // FFN Activations
    float* ffn_gate = nullptr;            // [max_tokens, kFfnInner]
    float* ffn_up = nullptr;              // [max_tokens, kFfnInner]
    float* ffn_activation = nullptr;      // [max_tokens, kFfnInner]
    float* ffn_down = nullptr;            // [max_tokens, kHidden]
};

// Manager allocating and owning the monolithic workspace buffer for Prefill V2 recurrent layers.
class RecurrentLayerWorkspaceManager {
public:
    explicit RecurrentLayerWorkspaceManager(std::size_t max_tokens = kMaxPrefillBatch);
    ~RecurrentLayerWorkspaceManager();
    RecurrentLayerWorkspaceManager(const RecurrentLayerWorkspaceManager&) = delete;
    RecurrentLayerWorkspaceManager& operator=(const RecurrentLayerWorkspaceManager&) = delete;
    RecurrentLayerWorkspaceManager(RecurrentLayerWorkspaceManager&& other) noexcept;
    RecurrentLayerWorkspaceManager& operator=(RecurrentLayerWorkspaceManager&& other) noexcept;

    [[nodiscard]] const RecurrentLayerWorkspace& workspace() const noexcept { return workspace_; }
    [[nodiscard]] std::size_t max_tokens() const noexcept { return max_tokens_; }
    [[nodiscard]] std::size_t total_workspace_bytes() const noexcept { return total_bytes_; }

private:
    std::size_t max_tokens_ = 0;
    std::size_t total_bytes_ = 0;
    void* d_buffer_ = nullptr;
    RecurrentLayerWorkspace workspace_{};
};

} // namespace miinfer::prefill_v2
