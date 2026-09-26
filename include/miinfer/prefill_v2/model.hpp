#pragma once

#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/prefill_v2/kv_cache.hpp"
#include "miinfer/prefill_v2/reusable_context.hpp"
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
#include <string>

#include <functional>

namespace miinfer::prefill_v2 {

struct GenerateOptions {
    std::size_t max_new_tokens = 128;
    bool reset_state_before = true;
    bool use_hip_graph = true;
    bool enable_prefix_reuse = false;   // When true, attempts to reuse cached prefix if matching
    bool cache_prefix_after = false;    // When true, caches prefix state after prefill
    std::size_t cache_prefix_len = 0;   // If 0, caches entire prompt; otherwise first N tokens
    std::function<void(std::uint32_t)> on_token = nullptr;
};

struct GenerateStats {
    std::vector<std::uint32_t> prompt_tokens;
    std::vector<std::uint32_t> generated_tokens;
    double prefill_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    double prefill_tok_per_sec = 0.0;
    double decode_tok_per_sec = 0.0;
    double avg_decode_latency_ms = 0.0;
    bool used_hip_graph = false;

    // Reuse telemetry & observability
    bool reuse_hit = false;
    std::uint32_t prefix_tokens_reused = 0;
    std::uint32_t suffix_tokens_dispatched = 0;
    std::uint32_t gqa_kv_reused_tokens = 0;
    std::uint32_t gdn_checkpoint_position = 0;
    double restore_ms = 0.0;
    double suffix_prefill_ms = 0.0;
};

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
// - LM Head Output Projection (Q6_K)
// - 48 Persistent RecurrentLayerStateStorage instances
// - 16 Persistent AttentionLayerKvCacheStorage instances
// - Monolithic Shared Workspace (sized for kMaxPrefillBatch = 512)
// - Ping-pong activation buffers
class PrefillV2Model {
public:
    explicit PrefillV2Model(
        const miinfer::Qwen35Model& model,
        std::uint32_t kv_capacity = 32768,
        bool load_lm_head = true,
        KvCacheQuantMode kv_quant_mode = KvCacheQuantMode::kFp16Fp16);
    ~PrefillV2Model();

    PrefillV2Model(const PrefillV2Model&) = delete;
    PrefillV2Model& operator=(const PrefillV2Model&) = delete;
    PrefillV2Model(PrefillV2Model&&) noexcept;
    PrefillV2Model& operator=(PrefillV2Model&&) noexcept;

    [[nodiscard]] KvCacheQuantMode kv_quant_mode() const noexcept { return kv_quant_mode_; }

    // Reset all 48 recurrent states and 16 KV caches
    void reset_state();

    // Physical single-chunk forward execution through the 64-layer pipeline (token_count <= 512)
    // d_tokens: device pointer to token IDs [token_count]
    // base_position: starting token index in the sequence
    // token_count: number of tokens to process (must be <= kMaxPrefillBatch)
    // d_final_hidden_out: output buffer for normalized final hidden states [token_count, kHidden]
    void forward(
        const std::uint32_t* d_tokens,
        std::uint32_t base_position,
        std::uint32_t token_count,
        float* d_final_hidden_out,
        hipStream_t stream = nullptr,
        const DevicePrefillState* prefill_state = nullptr);

    // Host token overload: copies host tokens to temporary device buffer and calls forward
    void forward(
        std::span<const std::uint32_t> tokens,
        std::uint32_t base_position,
        float* d_final_hidden_out,
        hipStream_t stream = nullptr);

    // Profiled forward for a single chunk (token_count <= 512)
    void forward_profiled(
        const std::uint32_t* d_tokens,
        std::uint32_t base_position,
        std::uint32_t token_count,
        float* d_final_hidden_out,
        ModelProfileBreakdown& breakdown,
        hipStream_t stream = nullptr);

    // Native Full-Model Macro Scheduler (V2 Macro Tile = 512)
    // Sequentially executes arbitrarily long sequences in 512-token full-model macro tiles,
    // seamlessly advancing base_position and persisting all 48 recurrent states and 16 KV caches.
    void prefill_sequence(
        const std::uint32_t* d_tokens,
        std::uint32_t total_tokens,
        float* d_final_hidden_out,
        hipStream_t stream = nullptr);

    void prefill_sequence(
        std::span<const std::uint32_t> tokens,
        float* d_final_hidden_out,
        hipStream_t stream = nullptr);

    // Compute model logits for the final token from its normalized hidden state
    // d_final_hidden_last_token: pointer to normalized hidden state [kHidden]
    // d_logits_out: device pointer to output logits [vocab_size] (if nullptr, writes to internal d_logits_)
    void compute_logits(
        const float* d_final_hidden_last_token,
        float* d_logits_out = nullptr,
        hipStream_t stream = nullptr);

    // Single-Token Autoregressive Decode Step:
    // Takes input_token at sequence position `position`, executes the full 64-layer model,
    // updates persistent recurrent states and appends to persistent KV caches in place with ZERO copies,
    // computes next-token logits, and returns the newly generated token ID via device argmax.
    std::uint32_t decode_step(
        std::uint32_t input_token,
        std::uint32_t position,
        float* d_logits_out = nullptr,
        hipStream_t stream = nullptr);

    // End-to-End Generation Pipeline:
    // 1. Prefills the full prompt using native 512-token macro-tiling.
    // 2. Evaluates the initial token logits (TTFT).
    // 3. Hands off persistent recurrent states and KV caches directly to the autoregressive decode loop with ZERO device copies.
    // 4. Decodes up to max_new_tokens sequentially.
    GenerateStats generate(
        std::span<const std::uint32_t> prompt,
        const GenerateOptions& options = GenerateOptions(),
        hipStream_t stream = nullptr);

    // Accessors
    [[nodiscard]] std::size_t block_count() const noexcept { return blocks_.size(); }
    [[nodiscard]] const PrefillV2TopologyBlock& block(std::size_t idx) const { return *blocks_[idx]; }
    [[nodiscard]] const RecurrentLayerStateStorage& recurrent_storage(std::size_t gdn_idx) const { return recurrent_states_[gdn_idx]; }
    [[nodiscard]] RecurrentLayerStateStorage& recurrent_storage(std::size_t gdn_idx) { return recurrent_states_[gdn_idx]; }
    [[nodiscard]] const AttentionLayerKvCacheStorage& kv_storage(std::size_t gqa_idx) const { return kv_caches_[gqa_idx]; }
    [[nodiscard]] AttentionLayerKvCacheStorage& kv_storage(std::size_t gqa_idx) { return kv_caches_[gqa_idx]; }
    [[nodiscard]] PrefillV2Workspace& workspace() noexcept { return const_cast<PrefillV2Workspace&>(ws_mgr_->workspace()); }
    [[nodiscard]] std::uint32_t vocab_size() const noexcept { return vocab_size_; }
    [[nodiscard]] const std::string& model_name() const noexcept { return model_name_; }
    [[nodiscard]] const std::string& quantization() const noexcept { return quantization_; }

    // Reusable Context & Checkpoint Storage
    [[nodiscard]] const ReusableContext& reusable_context() const noexcept { return reusable_context_; }
    [[nodiscard]] ReusableContext& reusable_context() noexcept { return reusable_context_; }
    [[nodiscard]] std::size_t cached_state_bytes() const noexcept { return reusable_context_.memory_bytes(); }
    void restore_reusable_context(hipStream_t stream = nullptr);

    // Memory footprints
    [[nodiscard]] std::size_t persistent_weight_bytes() const noexcept;
    [[nodiscard]] std::size_t persistent_state_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_bytes() const noexcept;
    [[nodiscard]] std::size_t activation_bytes() const noexcept;
    [[nodiscard]] std::size_t total_vram_bytes() const noexcept;

    [[nodiscard]] const void* embedding_weights() const noexcept { return d_embedding_weights_; }
    [[nodiscard]] const float* final_norm_weights() const noexcept { return d_final_norm_weights_; }
    [[nodiscard]] const void* output_weights() const noexcept { return d_output_weights_; }
    [[nodiscard]] float* logits_buffer() noexcept { return d_logits_; }

    // Reusable Decode Graph Capture & Replay
    void capture_decode_graph(hipStream_t stream = nullptr);
    void cleanup_decode_graph();
    [[nodiscard]] bool is_decode_graph_captured() const noexcept { return decode_graph_exec_ != nullptr; }

    // Reusable Suffix Prefill Graph Capture & Replay (512-token macro tile)
    void capture_suffix_graph(hipStream_t stream = nullptr);
    void cleanup_suffix_graph();
    [[nodiscard]] bool is_suffix_graph_captured() const noexcept { return suffix_graph_exec_ != nullptr; }

private:
    std::string model_name_ = "Qwen3.8-27B";
    std::string quantization_ = "Q4_K_M";
    std::uint32_t vocab_size_ = 0;
    float rms_epsilon_ = 1e-6f;
    std::uint32_t kv_capacity_ = 32768;
    KvCacheQuantMode kv_quant_mode_ = KvCacheQuantMode::kFp16Fp16;
    bool has_lm_head_ = false;

    // Reusable Context
    ReusableContext reusable_context_;

    // Weights on device
    void* d_embedding_weights_ = nullptr;     // Q4_K [vocab_size, hidden_size]
    std::size_t embedding_bytes_ = 0;
    float* d_final_norm_weights_ = nullptr;   // F32 [hidden_size]
    std::size_t final_norm_bytes_ = 0;
    void* d_output_weights_ = nullptr;        // Q6_K [vocab_size, hidden_size]
    std::size_t output_weight_bytes_ = 0;
    Q6KWaveTile* d_output_weights_wave_ = nullptr; // Gfx906 resident Wave Q6_K

    // LM Head Scratch & Logits Buffer
    void* d_lm_head_q8_k_ = nullptr;          // [kHidden / 256] Q8KDeviceBlock
    float* d_logits_ = nullptr;               // [vocab_size] float

    // 16 Blocks
    std::vector<std::unique_ptr<PrefillV2TopologyBlock>> blocks_;

    // States
    std::vector<RecurrentLayerStateStorage> recurrent_states_; // 48 states
    std::vector<AttentionLayerKvCacheStorage> kv_caches_;       // 16 KV caches

    // Shared Workspace & Ping-Pong Activation Buffers
    std::unique_ptr<PrefillV2WorkspaceManager> ws_mgr_;

    float* d_ping_ = nullptr; // [kMaxPrefillBatch * kHidden] floats
    float* d_pong_ = nullptr; // [kMaxPrefillBatch * kHidden] floats
    std::uint32_t* d_temp_tokens_ = nullptr; // [kMaxPrefillBatch] uint32_t

    // Reusable Device Decode State & Graph Exec
    void* d_decode_state_ = nullptr; // DeviceDecodeState
    std::uint32_t* d_decode_tokens_ = nullptr; // [kDefaultCacheCapacity] uint32_t
    hipGraphExec_t decode_graph_exec_ = nullptr;

    // Reusable Device Prefill State & Suffix Graph Exec (512 tokens)
    void* d_prefill_state_ = nullptr; // DevicePrefillState
    hipGraphExec_t suffix_graph_exec_ = nullptr;

    void allocate_resources();
    void free_resources();
};

} // namespace miinfer::prefill_v2
