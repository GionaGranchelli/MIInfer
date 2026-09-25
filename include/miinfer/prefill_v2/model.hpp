#pragma once

#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/prefill_v2/kv_cache.hpp"
#include "miinfer/prefill_v2/state.hpp"
#include "miinfer/prefill_v2/topology_block.hpp"
#include "miinfer/prefill_v2/workspace.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <span>

namespace miinfer::prefill_v2 {

struct ModelProfileBreakdown {
    double embedding_ms = 0.0;
    std::vector<TopologyBlockProfileBreakdown> block_breakdowns;
    double final_norm_ms = 0.0;
    double total_model_ms = 0.0;
};

// Represents the complete 64-layer Prefill V2 model:
// - Token embedding table (Q4_K)
// - 16 Repeating Topology Blocks (48 GDN + 16 GQA = 64 layers)
// - Output RMS norm (F32)
// - 48 Persistent RecurrentLayerStateStorage instances
// - 16 Persistent AttentionLayerKvCacheStorage instances
// - Monolithic Shared Workspace
// - Ping-pong activation buffers
class PrefillV2Model {
public:
    explicit PrefillV2Model(const miinfer::Qwen35Model& model, std::uint32_t kv_capacity = 32768);
    ~PrefillV2Model();

    PrefillV2Model(const PrefillV2Model&) = delete;
    PrefillV2Model& operator=(const PrefillV2Model&) = delete;
    PrefillV2Model(PrefillV2Model&&) noexcept;
    PrefillV2Model& operator=(PrefillV2Model&&) noexcept;

    // Reset all 48 recurrent states and 16 KV caches
    void reset_state();

    // Forward tokens through the full 64-layer pipeline
    // d_tokens: device pointer to token IDs [token_count]
    // base_position: starting token index in the sequence (0 for start)
    // token_count: number of tokens to process
    // d_final_hidden_out: output buffer for normalized final hidden states [token_count, kHidden]
    void forward(
        const std::uint32_t* d_tokens,
        std::uint32_t base_position,
        std::uint32_t token_count,
        float* d_final_hidden_out,
        hipStream_t stream = nullptr);

    // Host token overload: copies host tokens to temporary device buffer and calls forward
    void forward(
        std::span<const std::uint32_t> tokens,
        std::uint32_t base_position,
        float* d_final_hidden_out,
        hipStream_t stream = nullptr);

    // Profiled forward
    void forward_profiled(
        const std::uint32_t* d_tokens,
        std::uint32_t base_position,
        std::uint32_t token_count,
        float* d_final_hidden_out,
        ModelProfileBreakdown& breakdown,
        hipStream_t stream = nullptr);

    // Accessors
    [[nodiscard]] std::size_t block_count() const noexcept { return blocks_.size(); }
    [[nodiscard]] const PrefillV2TopologyBlock& block(std::size_t idx) const { return *blocks_[idx]; }
    [[nodiscard]] const RecurrentLayerStateStorage& recurrent_storage(std::size_t gdn_idx) const { return recurrent_states_[gdn_idx]; }
    [[nodiscard]] RecurrentLayerStateStorage& recurrent_storage(std::size_t gdn_idx) { return recurrent_states_[gdn_idx]; }
    [[nodiscard]] const AttentionLayerKvCacheStorage& kv_storage(std::size_t gqa_idx) const { return kv_caches_[gqa_idx]; }
    [[nodiscard]] AttentionLayerKvCacheStorage& kv_storage(std::size_t gqa_idx) { return kv_caches_[gqa_idx]; }
    [[nodiscard]] PrefillV2Workspace& workspace() noexcept { return const_cast<PrefillV2Workspace&>(ws_mgr_->workspace()); }

    [[nodiscard]] std::size_t persistent_weight_bytes() const noexcept;
    [[nodiscard]] std::size_t persistent_state_bytes() const noexcept;
    [[nodiscard]] std::size_t total_vram_bytes() const noexcept;

    [[nodiscard]] const void* embedding_weights() const noexcept { return d_embedding_weights_; }
    [[nodiscard]] const float* final_norm_weights() const noexcept { return d_final_norm_weights_; }

private:
    std::uint32_t vocab_size_ = 0;
    float rms_epsilon_ = 1e-6f;
    std::uint32_t kv_capacity_ = 32768;

    // Weights on device
    void* d_embedding_weights_ = nullptr;     // Q4_K [vocab_size, hidden_size]
    std::size_t embedding_bytes_ = 0;
    float* d_final_norm_weights_ = nullptr;   // F32 [hidden_size]
    std::size_t final_norm_bytes_ = 0;

    // 16 Blocks
    std::vector<std::unique_ptr<PrefillV2TopologyBlock>> blocks_;

    // States
    std::vector<RecurrentLayerStateStorage> recurrent_states_; // 48 states
    std::vector<AttentionLayerKvCacheStorage> kv_caches_;       // 16 KV caches

    // Shared Workspace & Ping-Pong Activation Buffers
    std::unique_ptr<PrefillV2WorkspaceManager> ws_mgr_;

    float* d_ping_ = nullptr; // [kMaxPrefillTokens * kHidden] floats
    float* d_pong_ = nullptr; // [kMaxPrefillTokens * kHidden] floats
    std::uint32_t* d_temp_tokens_ = nullptr; // [kMaxPrefillTokens] uint32_t

    void allocate_resources();
    void free_resources();
};

} // namespace miinfer::prefill_v2
