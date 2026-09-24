#pragma once

#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/m12_gdn_chunk.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace miinfer::prefill_v2 {

// Non-owning layout of pre-allocated scratch memory for Prefill V2 layers (Recurrent and Attention).
struct PrefillV2Workspace {
    // Shared Activations
    float* normalized = nullptr;          // [max_tokens, kHidden]
    float* residual = nullptr;            // [max_tokens, kHidden]
    float* post_normalized = nullptr;     // [max_tokens, kHidden]
    float* projected = nullptr;           // [max_tokens, kHidden] (SSM-out or Attn-O output)
    float* ssm_output = nullptr;          // Alias to projected

    // Recurrent Activations
    float* qkv = nullptr;                 // [max_tokens, kChannels]
    float* gate = nullptr;                // [max_tokens, kInner]
    float* raw_beta = nullptr;            // [max_tokens, kVHeads]
    float* raw_alpha = nullptr;           // [max_tokens, kVHeads]
    float* beta = nullptr;                // [max_tokens, kVHeads]
    float* decay = nullptr;               // [max_tokens, kVHeads]

    // Recurrent Conv split outputs
    float* query = nullptr;               // [max_tokens, kKHeads * kState]
    float* key = nullptr;                 // [max_tokens, kKHeads * kState]
    float* value = nullptr;               // [max_tokens, kVHeads * kState]

    // Recurrent GDN chunkwise scratch and outputs
    M12GdnChunkWorkspace gdn_scratch{};  // 5 buffers each [kVHeads * kGdnChunkSize * kState] floats
    float* gdn_raw_output = nullptr;      // [max_tokens, kVHeads * kState]
    float* gated_output = nullptr;        // [max_tokens, kVHeads * kState]

    // Attention Activations
    float* attn_qfull = nullptr;          // [max_tokens, kQFullDim (12288)]
    float* attn_q_rope = nullptr;         // [max_tokens, kQDim (6144)]
    float* attn_k = nullptr;              // [max_tokens, kKvDim (1024)]
    float* attn_v = nullptr;              // [max_tokens, kKvDim (1024)]
    float* attn_gated_output = nullptr;   // [max_tokens, kQDim (6144)]

    // MMQ Q8_1 quantization blocks (compact Mx contract)
    MxQ8_1MmqBlock* mmq_q8 = nullptr;   // [max_tokens * (max(kHidden, kInner, kFfnInner, kQFullDim) / 128)] blocks

    // FFN Activations
    float* ffn_gate = nullptr;            // [max_tokens, kFfnInner]
    float* ffn_up = nullptr;              // [max_tokens, kFfnInner]
    float* ffn_activation = nullptr;      // [max_tokens, kFfnInner]
    float* ffn_down = nullptr;            // [max_tokens, kHidden]
};

using RecurrentLayerWorkspace = PrefillV2Workspace;

// Manager allocating and owning the monolithic workspace buffer for Prefill V2 layers.
class PrefillV2WorkspaceManager {
public:
    explicit PrefillV2WorkspaceManager(std::size_t max_tokens = kMaxPrefillBatch);
    ~PrefillV2WorkspaceManager();
    PrefillV2WorkspaceManager(const PrefillV2WorkspaceManager&) = delete;
    PrefillV2WorkspaceManager& operator=(const PrefillV2WorkspaceManager&) = delete;
    PrefillV2WorkspaceManager(PrefillV2WorkspaceManager&& other) noexcept;
    PrefillV2WorkspaceManager& operator=(PrefillV2WorkspaceManager&& other) noexcept;

    [[nodiscard]] const PrefillV2Workspace& workspace() const noexcept { return workspace_; }
    [[nodiscard]] std::size_t max_tokens() const noexcept { return max_tokens_; }
    [[nodiscard]] std::size_t total_workspace_bytes() const noexcept { return total_bytes_; }

private:
    std::size_t max_tokens_ = 0;
    std::size_t total_bytes_ = 0;
    void* d_buffer_ = nullptr;
    PrefillV2Workspace workspace_{};
};

using RecurrentLayerWorkspaceManager = PrefillV2WorkspaceManager;

} // namespace miinfer::prefill_v2
