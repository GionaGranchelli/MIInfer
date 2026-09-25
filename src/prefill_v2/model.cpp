#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace miinfer::prefill_v2 {

PrefillV2Model::PrefillV2Model(const miinfer::Qwen35Model& model, std::uint32_t kv_capacity, bool load_lm_head)
    : vocab_size_(model.config().vocab_size),
      rms_epsilon_(model.config().rms_epsilon),
      kv_capacity_(kv_capacity),
      has_lm_head_(load_lm_head) {

    // 1. Load Token Embedding Weights (Q4_K)
    const auto embd_t = model.tensor("token_embd.weight");
    embedding_bytes_ = embd_t.bytes();
    MIINFER_HIP_CHECK(hipMalloc(&d_embedding_weights_, embedding_bytes_));
    MIINFER_HIP_CHECK(hipMemcpy(d_embedding_weights_, embd_t.data(), embedding_bytes_, hipMemcpyHostToDevice));

    // 2. Load Final RMS Norm Weights (F32)
    const auto norm_t = model.tensor("output_norm.weight");
    final_norm_bytes_ = norm_t.bytes();
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_final_norm_weights_), final_norm_bytes_));
    MIINFER_HIP_CHECK(hipMemcpy(d_final_norm_weights_, norm_t.data(), final_norm_bytes_, hipMemcpyHostToDevice));

    // 3. Load LM Head Weights (Q6_K)
    if (has_lm_head_) {
        const auto out_t = model.tensor("output.weight");
        output_weight_bytes_ = out_t.bytes();
        MIINFER_HIP_CHECK(hipMalloc(&d_output_weights_, output_weight_bytes_));
        MIINFER_HIP_CHECK(hipMemcpy(d_output_weights_, out_t.data(), output_weight_bytes_, hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMalloc(&d_lm_head_q8_k_, (kHidden / 256) * sizeof(Q8KDeviceBlock)));
    }

    // 4. Construct 16 Repeating Topology Blocks (64 layers total)
    blocks_.reserve(16);
    for (std::size_t b = 0; b < 16; ++b) {
        blocks_.emplace_back(std::make_unique<PrefillV2TopologyBlock>(model, b));
    }

    // 5. Allocate 48 Persistent Recurrent Layer States
    recurrent_states_.reserve(48);
    for (std::size_t i = 0; i < 48; ++i) {
        recurrent_states_.emplace_back();
    }

    // 6. Allocate 16 Persistent Attention KV Caches
    kv_caches_.reserve(16);
    for (std::size_t i = 0; i < 16; ++i) {
        kv_caches_.emplace_back(kv_capacity_);
    }

    // 7. Allocate Monolithic Shared Workspace and Ping-Pong Buffers
    allocate_resources();
}

PrefillV2Model::~PrefillV2Model() {
    free_resources();
}

PrefillV2Model::PrefillV2Model(PrefillV2Model&& other) noexcept
    : vocab_size_(other.vocab_size_),
      rms_epsilon_(other.rms_epsilon_),
      kv_capacity_(other.kv_capacity_),
      has_lm_head_(other.has_lm_head_),
      d_embedding_weights_(other.d_embedding_weights_),
      embedding_bytes_(other.embedding_bytes_),
      d_final_norm_weights_(other.d_final_norm_weights_),
      final_norm_bytes_(other.final_norm_bytes_),
      d_output_weights_(other.d_output_weights_),
      output_weight_bytes_(other.output_weight_bytes_),
      d_lm_head_q8_k_(other.d_lm_head_q8_k_),
      blocks_(std::move(other.blocks_)),
      recurrent_states_(std::move(other.recurrent_states_)),
      kv_caches_(std::move(other.kv_caches_)),
      ws_mgr_(std::move(other.ws_mgr_)),
      d_ping_(other.d_ping_),
      d_pong_(other.d_pong_),
      d_temp_tokens_(other.d_temp_tokens_) {
    other.d_embedding_weights_ = nullptr;
    other.d_final_norm_weights_ = nullptr;
    other.d_output_weights_ = nullptr;
    other.d_lm_head_q8_k_ = nullptr;
    other.d_ping_ = nullptr;
    other.d_pong_ = nullptr;
    other.d_temp_tokens_ = nullptr;
}

PrefillV2Model& PrefillV2Model::operator=(PrefillV2Model&& other) noexcept {
    if (this != &other) {
        free_resources();
        vocab_size_ = other.vocab_size_;
        rms_epsilon_ = other.rms_epsilon_;
        kv_capacity_ = other.kv_capacity_;
        has_lm_head_ = other.has_lm_head_;
        d_embedding_weights_ = other.d_embedding_weights_;
        embedding_bytes_ = other.embedding_bytes_;
        d_final_norm_weights_ = other.d_final_norm_weights_;
        final_norm_bytes_ = other.final_norm_bytes_;
        d_output_weights_ = other.d_output_weights_;
        output_weight_bytes_ = other.output_weight_bytes_;
        d_lm_head_q8_k_ = other.d_lm_head_q8_k_;
        blocks_ = std::move(other.blocks_);
        recurrent_states_ = std::move(other.recurrent_states_);
        kv_caches_ = std::move(other.kv_caches_);
        ws_mgr_ = std::move(other.ws_mgr_);
        d_ping_ = other.d_ping_;
        d_pong_ = other.d_pong_;
        d_temp_tokens_ = other.d_temp_tokens_;

        other.d_embedding_weights_ = nullptr;
        other.d_final_norm_weights_ = nullptr;
        other.d_output_weights_ = nullptr;
        other.d_lm_head_q8_k_ = nullptr;
        other.d_ping_ = nullptr;
        other.d_pong_ = nullptr;
        other.d_temp_tokens_ = nullptr;
    }
    return *this;
}

void PrefillV2Model::allocate_resources() {
    ws_mgr_ = std::make_unique<PrefillV2WorkspaceManager>(kMaxPrefillBatch);

    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ping_), kMaxPrefillBatch * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_pong_), kMaxPrefillBatch * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_temp_tokens_), kMaxPrefillBatch * sizeof(std::uint32_t)));
}

void PrefillV2Model::free_resources() {
    if (d_embedding_weights_ != nullptr) {
        (void)hipFree(d_embedding_weights_);
        d_embedding_weights_ = nullptr;
    }
    if (d_final_norm_weights_ != nullptr) {
        (void)hipFree(d_final_norm_weights_);
        d_final_norm_weights_ = nullptr;
    }
    if (d_output_weights_ != nullptr) {
        (void)hipFree(d_output_weights_);
        d_output_weights_ = nullptr;
    }
    if (d_lm_head_q8_k_ != nullptr) {
        (void)hipFree(d_lm_head_q8_k_);
        d_lm_head_q8_k_ = nullptr;
    }
    if (d_ping_ != nullptr) {
        (void)hipFree(d_ping_);
        d_ping_ = nullptr;
    }
    if (d_pong_ != nullptr) {
        (void)hipFree(d_pong_);
        d_pong_ = nullptr;
    }
    if (d_temp_tokens_ != nullptr) {
        (void)hipFree(d_temp_tokens_);
        d_temp_tokens_ = nullptr;
    }
}

void PrefillV2Model::reset_state() {
    for (auto& st : recurrent_states_) {
        st.reset();
    }
    for (auto& kv : kv_caches_) {
        kv.reset();
    }
}

void PrefillV2Model::forward(
    const std::uint32_t* d_tokens,
    std::uint32_t base_position,
    std::uint32_t token_count,
    float* d_final_hidden_out,
    hipStream_t stream) {

    if (token_count == 0 || token_count > kMaxPrefillBatch) {
        throw std::runtime_error("PrefillV2Model::forward: invalid token_count " + std::to_string(token_count)
                                 + " (max physical tile is " + std::to_string(kMaxPrefillBatch) + ")");
    }

    auto& ws = const_cast<PrefillV2Workspace&>(ws_mgr_->workspace());

    // 1. Embed tokens: d_tokens -> d_ping_ [token_count, kHidden]
    launch_qwen35_q4_k_embedding_batch(
        static_cast<const Q4KDeviceBlock*>(d_embedding_weights_),
        d_tokens,
        token_count,
        vocab_size_,
        kHidden,
        d_ping_,
        stream);

    // 2. Execute 16 Topology Blocks (Block 0..15 = 64 layers)
    for (std::size_t b = 0; b < 16; ++b) {
        auto st0 = recurrent_states_[b * 3 + 0].view();
        auto st1 = recurrent_states_[b * 3 + 1].view();
        auto st2 = recurrent_states_[b * 3 + 2].view();

        blocks_[b]->forward(
            d_ping_,
            d_pong_,
            d_ping_,
            st0,
            st0,
            st1,
            st1,
            st2,
            st2,
            kv_caches_[b].view(),
            ws,
            base_position,
            token_count,
            stream);
    }

    // 3. Final RMS Norm: d_ping_ -> d_final_hidden_out
    launch_qwen3_rms_norm_batch(
        d_ping_,
        d_final_norm_weights_,
        d_final_hidden_out,
        token_count,
        kHidden,
        rms_epsilon_,
        stream);
}

void PrefillV2Model::forward(
    std::span<const std::uint32_t> tokens,
    std::uint32_t base_position,
    float* d_final_hidden_out,
    hipStream_t stream) {

    const std::uint32_t token_count = static_cast<std::uint32_t>(tokens.size());
    if (token_count == 0 || token_count > kMaxPrefillBatch) {
        throw std::runtime_error("PrefillV2Model::forward: invalid token_count " + std::to_string(token_count));
    }

    MIINFER_HIP_CHECK(hipMemcpyAsync(
        d_temp_tokens_,
        tokens.data(),
        token_count * sizeof(std::uint32_t),
        hipMemcpyHostToDevice,
        stream));

    forward(d_temp_tokens_, base_position, token_count, d_final_hidden_out, stream);
}

void PrefillV2Model::forward_profiled(
    const std::uint32_t* d_tokens,
    std::uint32_t base_position,
    std::uint32_t token_count,
    float* d_final_hidden_out,
    ModelProfileBreakdown& breakdown,
    hipStream_t stream) {

    if (token_count == 0 || token_count > kMaxPrefillBatch) {
        throw std::runtime_error("PrefillV2Model::forward_profiled: invalid token_count " + std::to_string(token_count));
    }

    auto& ws = const_cast<PrefillV2Workspace&>(ws_mgr_->workspace());

    hipEvent_t ev_start, ev_emb, ev_blocks_end, ev_final;
    MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_emb));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_blocks_end));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_final));

    breakdown.block_breakdowns.resize(16);

    MIINFER_HIP_CHECK(hipEventRecord(ev_start, stream));

    // 1. Embed tokens
    launch_qwen35_q4_k_embedding_batch(
        static_cast<const Q4KDeviceBlock*>(d_embedding_weights_),
        d_tokens,
        token_count,
        vocab_size_,
        kHidden,
        d_ping_,
        stream);

    MIINFER_HIP_CHECK(hipEventRecord(ev_emb, stream));

    // 2. Execute 16 Topology Blocks
    for (std::size_t b = 0; b < 16; ++b) {
        auto st0 = recurrent_states_[b * 3 + 0].view();
        auto st1 = recurrent_states_[b * 3 + 1].view();
        auto st2 = recurrent_states_[b * 3 + 2].view();

        blocks_[b]->forward_profiled(
            d_ping_,
            d_pong_,
            d_ping_,
            st0,
            st0,
            st1,
            st1,
            st2,
            st2,
            kv_caches_[b].view(),
            ws,
            base_position,
            token_count,
            breakdown.block_breakdowns[b],
            stream);
    }

    MIINFER_HIP_CHECK(hipEventRecord(ev_blocks_end, stream));

    // 3. Final RMS Norm
    launch_qwen3_rms_norm_batch(
        d_ping_,
        d_final_norm_weights_,
        d_final_hidden_out,
        token_count,
        kHidden,
        rms_epsilon_,
        stream);

    MIINFER_HIP_CHECK(hipEventRecord(ev_final, stream));
    MIINFER_HIP_CHECK(hipEventSynchronize(ev_final));

    const auto elapsed = [](hipEvent_t a, hipEvent_t b) {
        float ms = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
        return static_cast<double>(ms);
    };

    breakdown.embedding_ms = elapsed(ev_start, ev_emb);
    breakdown.final_norm_ms = elapsed(ev_blocks_end, ev_final);
    breakdown.total_model_ms = elapsed(ev_start, ev_final);

    MIINFER_HIP_CHECK(hipEventDestroy(ev_start));
    MIINFER_HIP_CHECK(hipEventDestroy(ev_emb));
    MIINFER_HIP_CHECK(hipEventDestroy(ev_blocks_end));
    MIINFER_HIP_CHECK(hipEventDestroy(ev_final));
}

void PrefillV2Model::prefill_sequence(
    const std::uint32_t* d_tokens,
    std::uint32_t total_tokens,
    float* d_final_hidden_out,
    hipStream_t stream) {

    if (total_tokens == 0) {
        throw std::runtime_error("PrefillV2Model::prefill_sequence: zero tokens");
    }

    std::uint32_t pos = 0;
    while (pos < total_tokens) {
        std::uint32_t chunk = std::min<std::uint32_t>(kPrefillV2MacroTile, total_tokens - pos);
        forward(d_tokens + pos, pos, chunk, d_final_hidden_out + pos * kHidden, stream);
        pos += chunk;
    }
}

void PrefillV2Model::prefill_sequence(
    std::span<const std::uint32_t> tokens,
    float* d_final_hidden_out,
    hipStream_t stream) {

    const std::uint32_t total_tokens = static_cast<std::uint32_t>(tokens.size());
    if (total_tokens == 0) {
        throw std::runtime_error("PrefillV2Model::prefill_sequence: zero tokens");
    }

    std::uint32_t pos = 0;
    while (pos < total_tokens) {
        std::uint32_t chunk = std::min<std::uint32_t>(kPrefillV2MacroTile, total_tokens - pos);
        MIINFER_HIP_CHECK(hipMemcpyAsync(
            d_temp_tokens_,
            tokens.data() + pos,
            chunk * sizeof(std::uint32_t),
            hipMemcpyHostToDevice,
            stream));
        forward(d_temp_tokens_, pos, chunk, d_final_hidden_out + pos * kHidden, stream);
        pos += chunk;
    }
}

void PrefillV2Model::compute_logits(
    const float* d_final_hidden_last_token,
    float* d_logits_out,
    hipStream_t stream) {

    if (!has_lm_head_ || d_output_weights_ == nullptr || d_lm_head_q8_k_ == nullptr) {
        throw std::runtime_error("PrefillV2Model::compute_logits: LM head weights not loaded");
    }

    launch_qwen3_q8_k_quantize(
        d_final_hidden_last_token,
        static_cast<Q8KDeviceBlock*>(d_lm_head_q8_k_),
        kHidden,
        stream);

    launch_qwen3_q6_k_q8_k_gemv(
        static_cast<const Q6KDeviceBlock*>(d_output_weights_),
        static_cast<const Q8KDeviceBlock*>(d_lm_head_q8_k_),
        d_logits_out,
        vocab_size_,
        kHidden,
        stream);
}

std::size_t PrefillV2Model::persistent_weight_bytes() const noexcept {
    std::size_t total = embedding_bytes_ + final_norm_bytes_ + output_weight_bytes_;
    for (const auto& blk : blocks_) {
        total += blk->persistent_weight_bytes();
    }
    return total;
}

std::size_t PrefillV2Model::persistent_state_bytes() const noexcept {
    std::size_t total = recurrent_states_.size() * (RecurrentLayerState::kStateBytes + RecurrentLayerState::kConvHistoryBytes);
    for (const auto& kv : kv_caches_) {
        total += kv.total_bytes();
    }
    return total;
}

std::size_t PrefillV2Model::workspace_bytes() const noexcept {
    return (ws_mgr_ != nullptr) ? ws_mgr_->total_workspace_bytes() : 0;
}

std::size_t PrefillV2Model::activation_bytes() const noexcept {
    std::size_t total = 2 * kMaxPrefillBatch * kHidden * sizeof(float); // ping + pong
    total += kMaxPrefillBatch * sizeof(std::uint32_t);                  // temp tokens
    if (d_lm_head_q8_k_ != nullptr) {
        total += (kHidden / 256) * sizeof(Q8KDeviceBlock);
    }
    return total;
}

std::size_t PrefillV2Model::total_vram_bytes() const noexcept {
    return persistent_weight_bytes() + persistent_state_bytes() + workspace_bytes() + activation_bytes();
}

} // namespace miinfer::prefill_v2
