#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

namespace miinfer::prefill_v2 {

PrefillV2Model::PrefillV2Model(const miinfer::Qwen35Model& model, std::uint32_t kv_capacity, bool load_lm_head)
    : model_name_(model.model_name()),
      quantization_("Q4_K_M"),
      vocab_size_(model.config().vocab_size),
      rms_epsilon_(model.config().rms_epsilon),
      kv_capacity_(kv_capacity),
      has_lm_head_(load_lm_head),
      reusable_context_(model.model_name(), "Q4_K_M") {

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

    // 3. Load LM Head Weights (Q6_K Wave layout)
    if (has_lm_head_) {
        const auto out_t = model.tensor("output.weight");
        const auto wave_host = pack_q6k_wave_tensor(*out_t.source);
        output_weight_bytes_ = wave_host.size() * sizeof(Q6KWaveTile);
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output_weights_wave_), output_weight_bytes_));
        MIINFER_HIP_CHECK(hipMemcpy(d_output_weights_wave_, wave_host.data(), output_weight_bytes_, hipMemcpyHostToDevice));
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
      d_output_weights_wave_(other.d_output_weights_wave_),
      d_lm_head_q8_k_(other.d_lm_head_q8_k_),
      d_logits_(other.d_logits_),
      blocks_(std::move(other.blocks_)),
      recurrent_states_(std::move(other.recurrent_states_)),
      kv_caches_(std::move(other.kv_caches_)),
      ws_mgr_(std::move(other.ws_mgr_)),
      d_ping_(other.d_ping_),
      d_pong_(other.d_pong_),
      d_temp_tokens_(other.d_temp_tokens_),
      d_decode_state_(other.d_decode_state_),
      d_decode_tokens_(other.d_decode_tokens_),
      decode_graph_exec_(other.decode_graph_exec_) {
    other.d_embedding_weights_ = nullptr;
    other.d_final_norm_weights_ = nullptr;
    other.d_output_weights_ = nullptr;
    other.d_output_weights_wave_ = nullptr;
    other.d_lm_head_q8_k_ = nullptr;
    other.d_logits_ = nullptr;
    other.d_ping_ = nullptr;
    other.d_pong_ = nullptr;
    other.d_temp_tokens_ = nullptr;
    other.d_decode_state_ = nullptr;
    other.d_decode_tokens_ = nullptr;
    other.decode_graph_exec_ = nullptr;
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
        d_output_weights_wave_ = other.d_output_weights_wave_;
        d_lm_head_q8_k_ = other.d_lm_head_q8_k_;
        d_logits_ = other.d_logits_;
        blocks_ = std::move(other.blocks_);
        recurrent_states_ = std::move(other.recurrent_states_);
        kv_caches_ = std::move(other.kv_caches_);
        ws_mgr_ = std::move(other.ws_mgr_);
        d_ping_ = other.d_ping_;
        d_pong_ = other.d_pong_;
        d_temp_tokens_ = other.d_temp_tokens_;
        d_decode_state_ = other.d_decode_state_;
        d_decode_tokens_ = other.d_decode_tokens_;
        decode_graph_exec_ = other.decode_graph_exec_;

        other.d_embedding_weights_ = nullptr;
        other.d_final_norm_weights_ = nullptr;
        other.d_output_weights_ = nullptr;
        other.d_output_weights_wave_ = nullptr;
        other.d_lm_head_q8_k_ = nullptr;
        other.d_logits_ = nullptr;
        other.d_ping_ = nullptr;
        other.d_pong_ = nullptr;
        other.d_temp_tokens_ = nullptr;
        other.d_decode_state_ = nullptr;
        other.d_decode_tokens_ = nullptr;
        other.decode_graph_exec_ = nullptr;
    }
    return *this;
}

void PrefillV2Model::allocate_resources() {
    ws_mgr_ = std::make_unique<PrefillV2WorkspaceManager>(kMaxPrefillBatch);

    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ping_), kMaxPrefillBatch * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_pong_), kMaxPrefillBatch * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_temp_tokens_), kMaxPrefillBatch * sizeof(std::uint32_t)));
    if (has_lm_head_) {
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_logits_), vocab_size_ * sizeof(float)));
    }
    MIINFER_HIP_CHECK(hipMalloc(&d_decode_state_, sizeof(DeviceDecodeState)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_decode_tokens_), kv_capacity_ * sizeof(std::uint32_t)));
    decode_graph_exec_ = nullptr;
}

void PrefillV2Model::free_resources() {
    cleanup_decode_graph();
    if (d_decode_state_ != nullptr) {
        (void)hipFree(d_decode_state_);
        d_decode_state_ = nullptr;
    }
    if (d_decode_tokens_ != nullptr) {
        (void)hipFree(d_decode_tokens_);
        d_decode_tokens_ = nullptr;
    }
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
    if (d_output_weights_wave_ != nullptr) {
        (void)hipFree(d_output_weights_wave_);
        d_output_weights_wave_ = nullptr;
    }
    if (d_lm_head_q8_k_ != nullptr) {
        (void)hipFree(d_lm_head_q8_k_);
        d_lm_head_q8_k_ = nullptr;
    }
    if (d_logits_ != nullptr) {
        (void)hipFree(d_logits_);
        d_logits_ = nullptr;
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

    if (!has_lm_head_ || d_output_weights_wave_ == nullptr) {
        throw std::runtime_error("PrefillV2Model::compute_logits: LM head weights not loaded");
    }

    float* target_logits = d_logits_out != nullptr ? d_logits_out : d_logits_;
    if (target_logits == nullptr) {
        throw std::runtime_error("PrefillV2Model::compute_logits: null logits destination buffer");
    }

    auto& ws = const_cast<PrefillV2Workspace&>(ws_mgr_->workspace());
    launch_q8_1_quantize_f32(
        d_final_hidden_last_token,
        ws.q8_1,
        kHidden,
        stream);

    launch_q6k_wave_gemv(
        d_output_weights_wave_,
        ws.q8_1,
        target_logits,
        vocab_size_,
        kHidden,
        stream);
}

std::uint32_t PrefillV2Model::decode_step(
    std::uint32_t input_token,
    std::uint32_t position,
    float* d_logits_out,
    hipStream_t stream) {

    if (!has_lm_head_) {
        throw std::runtime_error("PrefillV2Model::decode_step: LM head weights not loaded");
    }
    float* target_logits = d_logits_out != nullptr ? d_logits_out : d_logits_;
    if (target_logits == nullptr) {
        throw std::runtime_error("PrefillV2Model::decode_step: null logits destination buffer");
    }

    auto& ws = const_cast<PrefillV2Workspace&>(ws_mgr_->workspace());

    // 1. Copy single input token to device
    MIINFER_HIP_CHECK(hipMemcpyAsync(
        d_temp_tokens_,
        &input_token,
        sizeof(std::uint32_t),
        hipMemcpyHostToDevice,
        stream));

    // 2. Single token embedding: d_temp_tokens_ -> d_ping_
    launch_qwen35_q4_k_embedding_batch(
        static_cast<const Q4KDeviceBlock*>(d_embedding_weights_),
        d_temp_tokens_,
        1,
        vocab_size_,
        kHidden,
        d_ping_,
        stream);

    // 3. Execute 16 Topology Blocks via specialized decode execution path
    for (std::size_t b = 0; b < 16; ++b) {
        auto st0 = recurrent_states_[b * 3 + 0].view();
        auto st1 = recurrent_states_[b * 3 + 1].view();
        auto st2 = recurrent_states_[b * 3 + 2].view();

        blocks_[b]->decode(
            d_ping_,
            d_pong_,
            d_ping_,
            st0,
            st1,
            st2,
            kv_caches_[b].view(),
            ws,
            position,
            /*decode_state=*/nullptr,
            stream);
    }

    // 4. Final RMS Norm: d_ping_ -> d_pong_
    launch_qwen3_rms_norm(
        d_ping_,
        d_final_norm_weights_,
        d_pong_,
        kHidden,
        kRmsNormEpsilon,
        stream);

    // 5. Compute logits: d_pong_ -> target_logits
    compute_logits(d_pong_, target_logits, stream);

    // 6. Argmax on device: target_logits -> d_temp_tokens_[0]
    launch_qwen3_argmax(target_logits, d_temp_tokens_, vocab_size_, stream);

    // 7. Read back single output token
    std::uint32_t next_token = 0;
    MIINFER_HIP_CHECK(hipMemcpyAsync(
        &next_token,
        d_temp_tokens_,
        sizeof(std::uint32_t),
        hipMemcpyDeviceToHost,
        stream));
    MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

    return next_token;
}

GenerateStats PrefillV2Model::generate(
    std::span<const std::uint32_t> prompt,
    const GenerateOptions& options,
    hipStream_t stream) {

    if (prompt.empty()) {
        throw std::runtime_error("PrefillV2Model::generate: empty prompt");
    }
    if (!has_lm_head_) {
        throw std::runtime_error("PrefillV2Model::generate: LM head weights not loaded");
    }

    const std::uint32_t prompt_len = static_cast<std::uint32_t>(prompt.size());
    bool is_reuse = false;
    std::uint32_t prefix_len = 0;
    std::uint32_t suffix_len = prompt_len;

    if (options.enable_prefix_reuse && reusable_context_.has_valid_prefix()) {
        auto match = reusable_context_.check_match(prompt, model_name_, quantization_);
        if (match == ReusableContext::MatchResult::ExactMatch) {
            is_reuse = true;
            prefix_len = reusable_context_.prefix_length();
            suffix_len = prompt_len - prefix_len;
        }
    }

    GenerateStats stats;
    stats.prompt_tokens.assign(prompt.begin(), prompt.end());
    stats.generated_tokens.reserve(options.max_new_tokens);

    const auto t_start = std::chrono::steady_clock::now();
    std::uint32_t last_chunk = 0;

    if (is_reuse) {
        stats.reuse_hit = true;
        stats.prefix_tokens_reused = prefix_len;
        stats.suffix_tokens_dispatched = suffix_len;
        stats.gqa_kv_reused_tokens = prefix_len;
        stats.gdn_checkpoint_position = prefix_len;

        const auto t_restore_start = std::chrono::steady_clock::now();
        reusable_context_.restore_gdn_states(recurrent_states_, stream);
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
        const auto t_restore_end = std::chrono::steady_clock::now();
        stats.restore_ms = std::chrono::duration<double, std::milli>(t_restore_end - t_restore_start).count();

        const auto t_suffix_start = std::chrono::steady_clock::now();
        // Prefill ONLY the suffix tokens: [prefix_len .. prompt_len - 1]
        std::uint32_t pos = prefix_len;
        while (pos < prompt_len) {
            std::uint32_t chunk = std::min<std::uint32_t>(kPrefillV2MacroTile, prompt_len - pos);
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                d_temp_tokens_,
                prompt.data() + pos,
                chunk * sizeof(std::uint32_t),
                hipMemcpyHostToDevice,
                stream));
            forward(d_temp_tokens_, pos, chunk, d_pong_, stream);
            pos += chunk;
            last_chunk = chunk;
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
        const auto t_suffix_end = std::chrono::steady_clock::now();
        stats.suffix_prefill_ms = std::chrono::duration<double, std::milli>(t_suffix_end - t_suffix_start).count();
    } else {
        if (options.reset_state_before) {
            reset_state();
        }
        stats.reuse_hit = false;
        stats.prefix_tokens_reused = 0;
        stats.suffix_tokens_dispatched = prompt_len;

        // Cold prefill sequence using native macro scheduler (Macro Tile = 512)
        std::uint32_t pos = 0;
        while (pos < prompt_len) {
            std::uint32_t chunk = std::min<std::uint32_t>(kPrefillV2MacroTile, prompt_len - pos);
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                d_temp_tokens_,
                prompt.data() + pos,
                chunk * sizeof(std::uint32_t),
                hipMemcpyHostToDevice,
                stream));
            forward(d_temp_tokens_, pos, chunk, d_pong_, stream);
            pos += chunk;
            last_chunk = chunk;
        }

        // If requested, capture the prefix checkpoint immediately after prefill
        if (options.cache_prefix_after) {
            std::size_t save_len = (options.cache_prefix_len > 0) ? std::min(options.cache_prefix_len, prompt.size()) : prompt.size();
            reusable_context_.save(prompt.subspan(0, save_len), recurrent_states_, stream);
        }
    }

    // 2. Compute logits for final prompt token (at offset (last_chunk - 1) * kHidden in d_pong_)
    compute_logits(d_pong_ + (last_chunk - 1) * kHidden, d_logits_, stream);

    // 3. First token via device argmax (TTFT)
    launch_qwen3_argmax(d_logits_, d_temp_tokens_, vocab_size_, stream);
    std::uint32_t first_token = 0;
    MIINFER_HIP_CHECK(hipMemcpyAsync(
        &first_token,
        d_temp_tokens_,
        sizeof(std::uint32_t),
        hipMemcpyDeviceToHost,
        stream));
    MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

    const auto t_ttft = std::chrono::steady_clock::now();
    stats.prefill_ms = std::chrono::duration<double, std::milli>(t_ttft - t_start).count();
    stats.ttft_ms = stats.prefill_ms;
    stats.prefill_tok_per_sec = stats.prefill_ms > 0.0 ? (prompt_len * 1000.0 / stats.prefill_ms) : 0.0;

    stats.generated_tokens.push_back(first_token);
    if (options.on_token) {
        options.on_token(first_token);
    }

    if (options.max_new_tokens <= 1) {
        stats.total_ms = stats.ttft_ms;
        return stats;
    }

    // 4. Autoregressive Decode Loop (Zero device copy handoff)
    const char* env_graph = std::getenv("MIINFER_HIP_GRAPH");
    bool enable_graph = options.use_hip_graph && (env_graph == nullptr || std::string_view(env_graph) != "0");

    const std::size_t num_decode = options.max_new_tokens - 1;

    if (enable_graph) {
        hipStream_t exec_stream = (stream != nullptr) ? stream : hipStreamPerThread;
        capture_decode_graph(exec_stream);
        stats.used_hip_graph = true;

        DeviceDecodeState state{};
        state.current_token = first_token;
        state.position = prompt_len;
        state.generated = 0;
        state.stop = 0;
        state.max_generated = static_cast<std::uint32_t>(num_decode);

        MIINFER_HIP_CHECK(hipMemcpyAsync(d_decode_state_, &state, sizeof(state), hipMemcpyHostToDevice, exec_stream));

        const auto t_decode_start = std::chrono::steady_clock::now();

        for (std::size_t i = 0; i < num_decode; ++i) {
            MIINFER_HIP_CHECK(hipGraphLaunch(decode_graph_exec_, exec_stream));
        }

        std::vector<std::uint32_t> host_decode_tokens(num_decode);
        MIINFER_HIP_CHECK(hipMemcpyAsync(
            host_decode_tokens.data(),
            d_decode_tokens_,
            num_decode * sizeof(std::uint32_t),
            hipMemcpyDeviceToHost,
            exec_stream));
        MIINFER_HIP_CHECK(hipStreamSynchronize(exec_stream));

        const auto t_decode_end = std::chrono::steady_clock::now();
        stats.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

        for (std::uint32_t tok : host_decode_tokens) {
            stats.generated_tokens.push_back(tok);
            if (options.on_token) {
                options.on_token(tok);
            }
        }

        const std::uint32_t final_position = prompt_len + static_cast<std::uint32_t>(num_decode);
        for (auto& st : recurrent_states_) {
            st.set_position(final_position);
        }

        const std::size_t decode_count = stats.generated_tokens.size() - 1;
        stats.decode_tok_per_sec = stats.decode_ms > 0.0 ? (decode_count * 1000.0 / stats.decode_ms) : 0.0;
        stats.avg_decode_latency_ms = decode_count > 0 ? (stats.decode_ms / decode_count) : 0.0;
        stats.total_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_start).count();

        return stats;
    }

    const auto t_decode_start = std::chrono::steady_clock::now();
    std::uint32_t current_token = first_token;

    for (std::size_t k = 1; k < options.max_new_tokens; ++k) {
        const std::uint32_t position = prompt_len - 1 + static_cast<std::uint32_t>(k);
        if (position >= kv_capacity_) {
            break;
        }

        const std::uint32_t next_token = decode_step(current_token, position, d_logits_, stream);
        stats.generated_tokens.push_back(next_token);
        if (options.on_token) {
            options.on_token(next_token);
        }
        current_token = next_token;
    }

    const auto t_decode_end = std::chrono::steady_clock::now();
    stats.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    const std::size_t decode_count = stats.generated_tokens.size() - 1;
    stats.decode_tok_per_sec = stats.decode_ms > 0.0 ? (decode_count * 1000.0 / stats.decode_ms) : 0.0;
    stats.avg_decode_latency_ms = decode_count > 0 ? (stats.decode_ms / decode_count) : 0.0;
    stats.total_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_start).count();

    return stats;
}

void PrefillV2Model::cleanup_decode_graph() {
    if (decode_graph_exec_ != nullptr) {
        (void)hipGraphExecDestroy(decode_graph_exec_);
        decode_graph_exec_ = nullptr;
    }
}

void PrefillV2Model::capture_decode_graph(hipStream_t stream) {
    if (decode_graph_exec_ != nullptr) {
        return;
    }
    if (!has_lm_head_ || d_output_weights_wave_ == nullptr || d_embedding_weights_ == nullptr ||
        d_final_norm_weights_ == nullptr || d_decode_state_ == nullptr || d_decode_tokens_ == nullptr) {
        throw std::runtime_error("PrefillV2Model::capture_decode_graph: required weights/buffers not allocated");
    }

    hipStream_t capture_stream = (stream != nullptr) ? stream : hipStreamPerThread;

    auto& ws = const_cast<PrefillV2Workspace&>(ws_mgr_->workspace());
    auto* decode_state = static_cast<DeviceDecodeState*>(d_decode_state_);

    hipGraph_t graph = nullptr;
    MIINFER_HIP_CHECK(hipStreamBeginCapture(capture_stream, hipStreamCaptureModeRelaxed));

    // 1. Single token embedding from device pointer (&decode_state->current_token)
    launch_qwen35_q4_k_embedding_device_token(
        static_cast<const Q4KDeviceBlock*>(d_embedding_weights_),
        &decode_state->current_token,
        vocab_size_,
        kHidden,
        d_ping_,
        capture_stream);

    // 2. 16 Topology Blocks (64 layers) via dynamic decode path
    for (std::size_t b = 0; b < 16; ++b) {
        auto st0 = recurrent_states_[b * 3 + 0].view();
        auto st1 = recurrent_states_[b * 3 + 1].view();
        auto st2 = recurrent_states_[b * 3 + 2].view();

        blocks_[b]->decode(
            d_ping_,
            d_pong_,
            d_ping_,
            st0,
            st1,
            st2,
            kv_caches_[b].view(),
            ws,
            0,
            decode_state,
            capture_stream);
    }

    // 3. Final RMS Norm: d_ping_ -> d_pong_
    launch_qwen3_rms_norm(
        d_ping_,
        d_final_norm_weights_,
        d_pong_,
        kHidden,
        rms_epsilon_,
        capture_stream);

    // 4. Compute logits: d_pong_ -> d_logits_
    compute_logits(d_pong_, d_logits_, capture_stream);

    // 5. Device argmax: d_logits_ -> &decode_state->current_token
    launch_qwen3_argmax(
        d_logits_,
        &decode_state->current_token,
        vocab_size_,
        capture_stream);

    // 6. Decode state advance: records token to d_decode_tokens_, updates position and generated count
    launch_qwen35_decode_state_advance(
        decode_state,
        d_decode_tokens_,
        kv_capacity_,
        capture_stream);

    MIINFER_HIP_CHECK(hipStreamEndCapture(capture_stream, &graph));
    MIINFER_HIP_CHECK(hipGraphInstantiate(&decode_graph_exec_, graph, nullptr, nullptr, 0));
    MIINFER_HIP_CHECK(hipGraphDestroy(graph));
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
    if (d_logits_ != nullptr) {
        total += vocab_size_ * sizeof(float);
    }
    return total;
}

std::size_t PrefillV2Model::total_vram_bytes() const noexcept {
    return persistent_weight_bytes() + persistent_state_bytes() + workspace_bytes() + activation_bytes() + cached_state_bytes();
}

} // namespace miinfer::prefill_v2

