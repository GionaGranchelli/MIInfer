#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <filesystem>

#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/prefill_v2/persistent_session.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

namespace miinfer::prefill_v2 {

namespace {

// ponytail: immutable host backing with an 8-entry, 3 GiB cap; move to shared
// GPU pages only if measured range upload cost justifies the complexity.
constexpr std::size_t kPersistentCheckpointStride = kPrefillV2MacroTile;
constexpr std::size_t kMaxPersistentCheckpoints = 8;
constexpr std::size_t kPersistentCheckpointBudgetBytes = 3ULL * 1024 * 1024 * 1024;
void prune_persistent_checkpoints(
    const std::filesystem::path& directory,
    const std::string& model_id,
    const std::string& quantization,
    const std::filesystem::path& preserve) {
    struct Entry {
        std::filesystem::path path;
        std::size_t bytes = 0;
        std::uint64_t created = 0;
    };
    std::vector<Entry> entries;
    std::size_t total_bytes = 0;
    std::error_code ec;
    for (const auto& item : std::filesystem::directory_iterator(directory, ec)) {
        if (ec || !item.is_regular_file(ec) || item.path().extension() != ".miinfer") continue;
        PersistentSessionHeader header{};
        if (!PersistentSession::inspect_file(item.path().string(), header)) continue;
        if (std::string_view(header.model_id) != model_id
            || std::string_view(header.quantization) != quantization) continue;
        const auto bytes = static_cast<std::size_t>(item.file_size(ec));
        if (ec) continue;
        entries.push_back({item.path(), bytes, header.created_timestamp});
        total_bytes += bytes;
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& left, const Entry& right) {
        if (left.created != right.created) return left.created < right.created;
        return left.path.string() < right.path.string();
    });
    while ((entries.size() > kMaxPersistentCheckpoints
            || total_bytes > kPersistentCheckpointBudgetBytes) && !entries.empty()) {
        const auto candidate = std::find_if(entries.begin(), entries.end(),
            [&](const Entry& entry) { return entry.path != preserve; });
        if (candidate == entries.end()) break;
        std::filesystem::remove(candidate->path, ec);
        if (!ec) {
            total_bytes -= candidate->bytes;
            entries.erase(candidate);
        } else {
            break;
        }
    }
}

std::pair<std::size_t, std::size_t> persistent_checkpoint_stats(
    const std::filesystem::path& directory,
    const std::string& model_id,
    const std::string& quantization) {
    std::size_t count = 0;
    std::size_t bytes = 0;
    std::error_code ec;
    for (const auto& item : std::filesystem::directory_iterator(directory, ec)) {
        if (ec || !item.is_regular_file(ec) || item.path().extension() != ".miinfer") continue;
        PersistentSessionHeader header{};
        if (!PersistentSession::inspect_file(item.path().string(), header)
            || std::string_view(header.model_id) != model_id
            || std::string_view(header.quantization) != quantization) continue;
        bytes += static_cast<std::size_t>(item.file_size(ec));
        if (!ec) ++count;
    }
    return {count, bytes};
}

std::uint32_t sample_token_from_logits(
    std::span<const float> raw_logits,
    std::span<const std::uint32_t> prompt_tokens,
    std::span<const std::uint32_t> generated_tokens,
    const GenerateOptions& options,
    std::mt19937& rng,
    std::vector<float>& logits_buf,
    std::vector<std::pair<float, std::uint32_t>>& candidates_buf) {

    const std::size_t vocab_size = raw_logits.size();
    if (vocab_size == 0) return 0;

    logits_buf.assign(raw_logits.begin(), raw_logits.end());

    // 1. Repetition / Presence / Frequency penalty
    const std::size_t window_limit = options.repeat_last_n > 0 ? options.repeat_last_n : 256;
    std::unordered_map<std::uint32_t, std::size_t> counts;

    // Track generated tokens first
    const std::size_t gen_take = std::min(window_limit, generated_tokens.size());
    if (gen_take > 0) {
        auto gen_window = generated_tokens.subspan(generated_tokens.size() - gen_take, gen_take);
        for (std::uint32_t tok : gen_window) {
            counts[tok]++;
        }
    }

    // If window limit not full, fill remaining from prompt tail
    if (gen_take < window_limit && !prompt_tokens.empty()) {
        const std::size_t prompt_remain = window_limit - gen_take;
        const std::size_t prompt_take = std::min(prompt_remain, prompt_tokens.size());
        auto prompt_window = prompt_tokens.subspan(prompt_tokens.size() - prompt_take, prompt_take);
        for (std::uint32_t tok : prompt_window) {
            counts[tok]++;
        }
    }

    for (const auto& [tok, count] : counts) {
        if (tok < vocab_size) {
            // Standard repetition penalty
            if (options.repetition_penalty != 1.0f && options.repetition_penalty > 0.0f) {
                if (logits_buf[tok] < 0.0f) {
                    logits_buf[tok] *= options.repetition_penalty;
                } else {
                    logits_buf[tok] /= options.repetition_penalty;
                }
            }
            // Frequency penalty
            if (options.frequency_penalty != 0.0f) {
                logits_buf[tok] -= options.frequency_penalty * static_cast<float>(count);
            }
            // Presence penalty
            if (options.presence_penalty != 0.0f) {
                logits_buf[tok] -= options.presence_penalty;
            }
        }
    }

    // 2. Greedy Argmax
    if (options.temperature <= 0.001f) {
        float max_val = logits_buf[0];
        std::uint32_t max_idx = 0;
        for (std::size_t i = 1; i < vocab_size; ++i) {
            if (logits_buf[i] > max_val) {
                max_val = logits_buf[i];
                max_idx = static_cast<std::uint32_t>(i);
            }
        }
        return max_idx;
    }

    // 3. Temperature scaling
    const float inv_temp = 1.0f / options.temperature;
    for (std::size_t i = 0; i < vocab_size; ++i) {
        logits_buf[i] *= inv_temp;
    }

    // 4. Top-K candidates
    candidates_buf.resize(vocab_size);
    for (std::size_t i = 0; i < vocab_size; ++i) {
        candidates_buf[i] = {logits_buf[i], static_cast<std::uint32_t>(i)};
    }

    const std::size_t k = (options.top_k > 0 && options.top_k < vocab_size) ? options.top_k : vocab_size;
    if (k < vocab_size) {
        std::partial_sort(
            candidates_buf.begin(),
            candidates_buf.begin() + k,
            candidates_buf.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        candidates_buf.resize(k);
    } else {
        std::sort(
            candidates_buf.begin(),
            candidates_buf.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // 5. Softmax on top-K candidates
    const float max_logit = candidates_buf[0].first;
    float sum_exp = 0.0f;
    for (auto& c : candidates_buf) {
        c.first = std::exp(c.first - max_logit);
        sum_exp += c.first;
    }
    const float inv_sum = 1.0f / sum_exp;
    for (auto& c : candidates_buf) {
        c.first *= inv_sum;
    }

    // 6. Top-P (Nucleus) Filtering
    if (options.top_p > 0.0f && options.top_p < 1.0f) {
        float cumsum = 0.0f;
        std::size_t cutoff = candidates_buf.size();
        for (std::size_t i = 0; i < candidates_buf.size(); ++i) {
            cumsum += candidates_buf[i].first;
            if (cumsum >= options.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates_buf.resize(cutoff);

        float new_sum = 0.0f;
        for (const auto& c : candidates_buf) {
            new_sum += c.first;
        }
        if (new_sum > 0.0f) {
            const float inv_new_sum = 1.0f / new_sum;
            for (auto& c : candidates_buf) {
                c.first *= inv_new_sum;
            }
        }
    }

    // 7. Multinomial Sampling
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float r = dist(rng);
    float cum = 0.0f;
    for (const auto& c : candidates_buf) {
        cum += c.first;
        if (r <= cum) {
            return c.second;
        }
    }

    return candidates_buf.back().second;
}

} // namespace

PrefillV2Model::PrefillV2Model(
    const miinfer::Qwen35Model& model,
    std::uint32_t kv_capacity,
    bool load_lm_head,
    KvCacheQuantMode kv_quant_mode)
    : model_name_(model.model_name()),
      quantization_("Q4_K_M"),
      vocab_size_(model.config().vocab_size),
      rms_epsilon_(model.config().rms_epsilon),
      kv_capacity_(kv_capacity),
      kv_quant_mode_(kv_quant_mode),
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
        kv_caches_.emplace_back(kv_capacity_, kv_quant_mode_);
    }

    // 7. Allocate Monolithic Shared Workspace and Ping-Pong Buffers
    allocate_resources();
}

PrefillV2Model::~PrefillV2Model() {
    clear_snapshots();
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
      decode_graph_exec_(other.decode_graph_exec_),
      d_prefill_state_(other.d_prefill_state_),
      suffix_graph_exec_(other.suffix_graph_exec_),
      snapshots_(std::move(other.snapshots_)),
      next_snapshot_id_(other.next_snapshot_id_),
      snapshot_use_clock_(other.snapshot_use_clock_),
      snapshot_bytes_(other.snapshot_bytes_),
      snapshot_state_loaded_(other.snapshot_state_loaded_),
      snapshot_cow_events_(other.snapshot_cow_events_),
      snapshot_cow_bytes_(other.snapshot_cow_bytes_) {
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
    other.d_prefill_state_ = nullptr;
    other.suffix_graph_exec_ = nullptr;
    other.next_snapshot_id_ = 1;
    other.snapshot_use_clock_ = 0;
    other.snapshot_bytes_ = 0;
    other.snapshot_state_loaded_ = false;
    other.snapshot_cow_events_ = 0;
    other.snapshot_cow_bytes_ = 0;
}

PrefillV2Model& PrefillV2Model::operator=(PrefillV2Model&& other) noexcept {
    if (this != &other) {
        clear_snapshots();
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
        d_prefill_state_ = other.d_prefill_state_;
        suffix_graph_exec_ = other.suffix_graph_exec_;
        snapshots_ = std::move(other.snapshots_);
        next_snapshot_id_ = other.next_snapshot_id_;
        snapshot_use_clock_ = other.snapshot_use_clock_;
        snapshot_bytes_ = other.snapshot_bytes_;
        snapshot_state_loaded_ = other.snapshot_state_loaded_;
        snapshot_cow_events_ = other.snapshot_cow_events_;
        snapshot_cow_bytes_ = other.snapshot_cow_bytes_;

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
        other.d_prefill_state_ = nullptr;
        other.suffix_graph_exec_ = nullptr;
        other.next_snapshot_id_ = 1;
        other.snapshot_use_clock_ = 0;
        other.snapshot_bytes_ = 0;
        other.snapshot_state_loaded_ = false;
        other.snapshot_cow_events_ = 0;
        other.snapshot_cow_bytes_ = 0;
    }
    return *this;
}

void PrefillV2Model::allocate_resources() {
    // Keep the qualified 32-way scratch path through 64K. At larger capacities,
    // three-way scratch is sufficient for the same kernel and removes the
    // construction-time arena peak that blocked the 128K model.
    const std::uint32_t splitk_splits = kv_capacity_ > 66000 ? 3 : 32;
    ws_mgr_ = std::make_unique<PrefillV2WorkspaceManager>(kMaxPrefillBatch, splitk_splits);

    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ping_), kMaxPrefillBatch * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_pong_), kMaxPrefillBatch * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_temp_tokens_), kMaxPrefillBatch * sizeof(std::uint32_t)));
    if (has_lm_head_) {
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_logits_), vocab_size_ * sizeof(float)));
        host_logits_.resize(vocab_size_);
        logits_scratch_.resize(vocab_size_);
        candidates_buf_.resize(vocab_size_);
    }
    MIINFER_HIP_CHECK(hipMalloc(&d_decode_state_, sizeof(DeviceDecodeState)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_decode_tokens_), kv_capacity_ * sizeof(std::uint32_t)));
    decode_graph_exec_ = nullptr;

    MIINFER_HIP_CHECK(hipMalloc(&d_prefill_state_, sizeof(DevicePrefillState)));
    suffix_graph_exec_ = nullptr;
}

void PrefillV2Model::free_resources() {
    cleanup_decode_graph();
    cleanup_suffix_graph();
    if (d_prefill_state_ != nullptr) {
        (void)hipFree(d_prefill_state_);
        d_prefill_state_ = nullptr;
    }
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
    reusable_context_.clear();
    clear_snapshots();
    snapshot_state_loaded_ = false;
    for (auto& st : recurrent_states_) {
        st.reset();
    }
    for (auto& kv : kv_caches_) {
        kv.reset();
    }
}

void PrefillV2Model::restore_reusable_context(hipStream_t stream) {
    if (reusable_context_.has_valid_prefix()) {
        if (snapshot_state_loaded_) {
            snapshot_state_loaded_ = false;
        } else {
            reusable_context_.restore_gdn_states(recurrent_states_, stream);
        }
    }
}

void PrefillV2Model::forward(
    const std::uint32_t* d_tokens,
    std::uint32_t base_position,
    std::uint32_t token_count,
    float* d_final_hidden_out,
    hipStream_t stream,
    const DevicePrefillState* prefill_state) {

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
            stream,
            prefill_state);
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

    if (options.seed) {
        rng_.seed(*options.seed);
    } else if (options.reset_state_before) {
        rng_.seed(42);
    }

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
        if (match == ReusableContext::MatchResult::ExactMatch
            && (reusable_context_.prefix_length() < prompt_len
                || reusable_context_.has_final_hidden())) {
            is_reuse = true;
            prefix_len = reusable_context_.prefix_length();
            suffix_len = prompt_len - prefix_len;
        }
    }

    if (!is_reuse && options.enable_prefix_reuse && !options.persistent_session_dir.empty()) {
        std::uint32_t disk_prefix = 0;
        if (restore_matching_session(options.persistent_session_dir, prompt, disk_prefix, stream)) {
            if (disk_prefix < prompt_len) {
                is_reuse = true;
                prefix_len = disk_prefix;
                suffix_len = prompt_len - prefix_len;
            } else {
                // Disk sessions do not yet carry the boundary hidden vector;
                // discard the loaded state and use the correct full path.
                reset_state();
            }
        }
    }

    GenerateStats stats;
    stats.prompt_tokens.assign(prompt.begin(), prompt.end());
    stats.generated_tokens.reserve(options.max_new_tokens);

    const auto t_start = std::chrono::steady_clock::now();
    std::uint32_t last_chunk = 0;
    const auto save_persistent_checkpoint = [&](std::uint32_t length) {
        if (options.persistent_session_dir.empty()
            || length < kPersistentCheckpointStride
            || length % kPersistentCheckpointStride != 0) return;
        const auto prefix = prompt.first(length);
        const auto hash = compute_token_sequence_hash(prefix);
        const std::filesystem::path directory(options.persistent_session_dir);
        const auto path = directory / PersistentSession::format_session_filename(
            hash, length);
        if (std::filesystem::exists(path)) return;
        try {
            save_session(path.string(), prefix, stream);
            prune_persistent_checkpoints(directory, model_name_, quantization_, path);
        } catch (...) {
            // Persistent checkpoint capture is an optimization; generation remains non-fatal.
        }
    };

    if (is_reuse) {
        stats.reuse_hit = true;
        stats.prefix_tokens_reused = prefix_len;
        stats.suffix_tokens_dispatched = suffix_len;
        stats.suffix_tokens_executed = suffix_len;
        stats.prefix_tokens_replayed = 0;
        stats.checkpoint_position = prefix_len;
        stats.gqa_kv_reused_tokens = prefix_len;
        stats.gdn_checkpoint_position = prefix_len;

        const auto t_restore_start = std::chrono::steady_clock::now();
        reusable_context_.restore_gdn_states(recurrent_states_, stream);
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
        const auto t_restore_end = std::chrono::steady_clock::now();
        stats.restore_ms = std::chrono::duration<double, std::milli>(t_restore_end - t_restore_start).count();

        const auto t_suffix_start = std::chrono::steady_clock::now();
        const char* env_graph = std::getenv("MIINFER_HIP_GRAPH");
        bool enable_graph = options.use_hip_graph && (env_graph == nullptr || std::string_view(env_graph) != "0");
        hipStream_t exec_stream = (stream != nullptr) ? stream : hipStreamPerThread;

        // Prefill ONLY the suffix tokens: [prefix_len .. prompt_len - 1]
        std::uint32_t pos = prefix_len;
        while (pos < prompt_len) {
            std::uint32_t chunk = std::min<std::uint32_t>(kPrefillV2MacroTile, prompt_len - pos);
            if (enable_graph && chunk == kPrefillV2MacroTile) {
                capture_suffix_graph(exec_stream);
                DevicePrefillState prefill_state_host{};
                prefill_state_host.base_position = pos;
                prefill_state_host.token_count = chunk;
                prefill_state_host.total_length = pos + chunk;
                prefill_state_host.kv_capacity = kv_capacity_;

                MIINFER_HIP_CHECK(hipMemcpyAsync(
                    d_temp_tokens_,
                    prompt.data() + pos,
                    chunk * sizeof(std::uint32_t),
                    hipMemcpyHostToDevice,
                    exec_stream));
                MIINFER_HIP_CHECK(hipMemcpyAsync(
                    d_prefill_state_,
                    &prefill_state_host,
                    sizeof(DevicePrefillState),
                    hipMemcpyHostToDevice,
                    exec_stream));
                MIINFER_HIP_CHECK(hipGraphLaunch(suffix_graph_exec_, exec_stream));
            } else {
                MIINFER_HIP_CHECK(hipMemcpyAsync(
                    d_temp_tokens_,
                    prompt.data() + pos,
                    chunk * sizeof(std::uint32_t),
                    hipMemcpyHostToDevice,
                    exec_stream));
                forward(d_temp_tokens_, pos, chunk, d_pong_, exec_stream);
            }
            pos += chunk;
            last_chunk = chunk;
            save_persistent_checkpoint(pos);
        }
        for (auto& st : recurrent_states_) {
            st.set_position(pos);
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(exec_stream));
        const auto t_suffix_end = std::chrono::steady_clock::now();
        stats.suffix_prefill_ms = std::chrono::duration<double, std::milli>(t_suffix_end - t_suffix_start).count();
    } else {
        if (options.reset_state_before) {
            reset_state();
        }
        stats.reuse_hit = false;
        stats.prefix_tokens_reused = 0;
        stats.suffix_tokens_dispatched = prompt_len;
        stats.suffix_tokens_executed = prompt_len;
        stats.prefix_tokens_replayed = prompt_len;
        stats.checkpoint_position = 0;

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
            save_persistent_checkpoint(pos);
        }
    }

    // 1b. If requested, capture/update the prefix checkpoint immediately after prefill
    if (options.cache_prefix_after) {
        std::size_t save_len = (options.cache_prefix_len > 0) ? std::min(options.cache_prefix_len, prompt.size()) : prompt.size();
        if (save_len == prompt.size()) {
            if (!(is_reuse && suffix_len == 0)) {
                reusable_context_.save(prompt, recurrent_states_, d_pong_ + (last_chunk - 1) * kHidden, stream);
            }
        } else {
            // A single forward pass leaves state at prompt.size(), not save_len.
            reusable_context_.clear();
        }
        if (!options.persistent_session_dir.empty() && save_len == prompt.size() && save_len >= 256) {
            const auto prefix_sub = prompt.subspan(0, save_len);
            const std::uint64_t token_hash = compute_token_sequence_hash(prefix_sub);
            const std::string filename = PersistentSession::format_session_filename(
                token_hash, static_cast<std::uint32_t>(save_len));
            const std::filesystem::path sdir(options.persistent_session_dir);
            const std::filesystem::path save_path = sdir / filename;
            if (!std::filesystem::exists(save_path)) {
                try {
                    save_session(save_path.string(), prefix_sub, stream);
                    prune_persistent_checkpoints(sdir, model_name_, quantization_, save_path);
                } catch (...) {
                    // non-fatal disk write error
                }
            }
        }
    }

    if (!options.persistent_session_dir.empty()) {
        const auto [count, bytes] = persistent_checkpoint_stats(
            options.persistent_session_dir, model_name_, quantization_);
        stats.persistent_checkpoint_count = count;
        stats.persistent_checkpoint_bytes = bytes;
    }

    // 2. Compute logits for the final prompt token. A zero-suffix exact hit
    // restores the cached boundary hidden vector without replaying the prefix.
    if (is_reuse && suffix_len == 0) {
        reusable_context_.restore_final_hidden(d_pong_, stream);
        compute_logits(d_pong_, d_logits_, stream);
    } else {
        compute_logits(d_pong_ + (last_chunk - 1) * kHidden, d_logits_, stream);
    }

    // 3. First token via host sampling (TTFT)
    MIINFER_HIP_CHECK(hipMemcpyAsync(
        host_logits_.data(),
        d_logits_,
        vocab_size_ * sizeof(float),
        hipMemcpyDeviceToHost,
        stream));
    MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

    std::uint32_t first_token = sample_token_from_logits(
        host_logits_,
        prompt,
        {},
        options,
        rng_,
        logits_scratch_,
        candidates_buf_);

    const auto t_ttft = std::chrono::steady_clock::now();
    stats.prefill_ms = std::chrono::duration<double, std::milli>(t_ttft - t_start).count();
    stats.ttft_ms = stats.prefill_ms;
    stats.prefill_tok_per_sec = stats.prefill_ms > 0.0 ? (prompt_len * 1000.0 / stats.prefill_ms) : 0.0;

    const auto is_stop = [&](std::uint32_t tok) {
        for (auto st : options.stop_token_ids) {
            if (tok == st) return true;
        }
        return false;
    };

    if (is_stop(first_token)) {
        stats.stop_reason = miinfer::GenerationStopReason::kStopToken;
        stats.total_ms = stats.ttft_ms;
        return stats;
    }

    stats.generated_tokens.push_back(first_token);
    if (options.on_token) {
        if (!options.on_token(first_token)) {
            stats.stop_reason = miinfer::GenerationStopReason::kCancelled;
            stats.total_ms = stats.ttft_ms;
            return stats;
        }
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
        std::size_t actual_generated = 0;

        for (std::size_t i = 0; i < num_decode; ++i) {
            MIINFER_HIP_CHECK(hipGraphLaunch(decode_graph_exec_, exec_stream));

            MIINFER_HIP_CHECK(hipMemcpyAsync(
                host_logits_.data(),
                d_logits_,
                vocab_size_ * sizeof(float),
                hipMemcpyDeviceToHost,
                exec_stream));
            MIINFER_HIP_CHECK(hipStreamSynchronize(exec_stream));

            const std::uint32_t next_token = sample_token_from_logits(
                host_logits_,
                prompt,
                stats.generated_tokens,
                options,
                rng_,
                logits_scratch_,
                candidates_buf_);

            actual_generated++;

            if (is_stop(next_token)) {
                stats.stop_reason = miinfer::GenerationStopReason::kStopToken;
                break;
            }

            stats.generated_tokens.push_back(next_token);
            if (options.on_token) {
                if (!options.on_token(next_token)) {
                    stats.stop_reason = miinfer::GenerationStopReason::kCancelled;
                    break;
                }
            }

            state.current_token = next_token;
            state.position++;
            state.generated++;
            MIINFER_HIP_CHECK(hipMemcpyAsync(d_decode_state_, &state, sizeof(state), hipMemcpyHostToDevice, exec_stream));
        }

        const auto t_decode_end = std::chrono::steady_clock::now();
        stats.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

        const std::uint32_t final_position = prompt_len + static_cast<std::uint32_t>(actual_generated);
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
    auto& ws = const_cast<PrefillV2Workspace&>(ws_mgr_->workspace());

    for (std::size_t k = 1; k < options.max_new_tokens; ++k) {
        const std::uint32_t position = prompt_len - 1 + static_cast<std::uint32_t>(k);
        if (position >= kv_capacity_) {
            stats.stop_reason = miinfer::GenerationStopReason::kOutputLimit;
            break;
        }

        // 1. Copy single input token to device
        MIINFER_HIP_CHECK(hipMemcpyAsync(
            d_temp_tokens_,
            &current_token,
            sizeof(std::uint32_t),
            hipMemcpyHostToDevice,
            stream));

        // 2. Single token embedding
        launch_qwen35_q4_k_embedding_batch(
            static_cast<const Q4KDeviceBlock*>(d_embedding_weights_),
            d_temp_tokens_,
            1,
            vocab_size_,
            kHidden,
            d_ping_,
            stream);

        // 3. Execute 16 Topology Blocks
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

        // 4. Final RMS Norm
        launch_qwen3_rms_norm(
            d_ping_,
            d_final_norm_weights_,
            d_pong_,
            kHidden,
            kRmsNormEpsilon,
            stream);

        // 5. Compute logits
        compute_logits(d_pong_, d_logits_, stream);

        // 6. Copy logits to host
        MIINFER_HIP_CHECK(hipMemcpyAsync(
            host_logits_.data(),
            d_logits_,
            vocab_size_ * sizeof(float),
            hipMemcpyDeviceToHost,
            stream));
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

        // 7. Sample
        const std::uint32_t next_token = sample_token_from_logits(
            host_logits_,
            prompt,
            stats.generated_tokens,
            options,
            rng_,
            logits_scratch_,
            candidates_buf_);

        if (is_stop(next_token)) {
            stats.stop_reason = miinfer::GenerationStopReason::kStopToken;
            break;
        }
        stats.generated_tokens.push_back(next_token);
        if (options.on_token) {
            if (!options.on_token(next_token)) {
                stats.stop_reason = miinfer::GenerationStopReason::kCancelled;
                break;
            }
        }
        for (auto& st : recurrent_states_) {
            st.set_position(position + 1);
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
        d_final_norm_weights_ == nullptr || d_decode_state_ == nullptr) {
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

    MIINFER_HIP_CHECK(hipStreamEndCapture(capture_stream, &graph));
    MIINFER_HIP_CHECK(hipGraphInstantiate(&decode_graph_exec_, graph, nullptr, nullptr, 0));
    MIINFER_HIP_CHECK(hipGraphDestroy(graph));
}

void PrefillV2Model::cleanup_suffix_graph() {
    if (suffix_graph_exec_ != nullptr) {
        (void)hipGraphExecDestroy(suffix_graph_exec_);
        suffix_graph_exec_ = nullptr;
    }
}

void PrefillV2Model::capture_suffix_graph(hipStream_t stream) {
    if (suffix_graph_exec_ != nullptr) {
        return;
    }
    if (d_embedding_weights_ == nullptr || d_final_norm_weights_ == nullptr ||
        d_temp_tokens_ == nullptr || d_prefill_state_ == nullptr || ws_mgr_ == nullptr) {
        throw std::runtime_error("PrefillV2Model::capture_suffix_graph: required weights/buffers not allocated");
    }

    hipStream_t capture_stream = (stream != nullptr) ? stream : hipStreamPerThread;
    auto* prefill_state = static_cast<DevicePrefillState*>(d_prefill_state_);

    hipGraph_t graph = nullptr;
    MIINFER_HIP_CHECK(hipStreamBeginCapture(capture_stream, hipStreamCaptureModeRelaxed));

    forward(d_temp_tokens_, 0, kPrefillV2MacroTile, d_pong_, capture_stream, prefill_state);

    MIINFER_HIP_CHECK(hipStreamEndCapture(capture_stream, &graph));
    MIINFER_HIP_CHECK(hipGraphInstantiate(&suffix_graph_exec_, graph, nullptr, nullptr, 0));
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
    return persistent_weight_bytes() + persistent_state_bytes() + workspace_bytes()
        + activation_bytes() + cached_state_bytes() + snapshot_bytes_;
}

void PrefillV2Model::save_session(
    const std::string& file_path,
    std::span<const std::uint32_t> prefix_tokens,
    hipStream_t stream) const {
    PersistentSession::save_to_file(file_path, *this, prefix_tokens, stream);
}

bool PrefillV2Model::load_session(
    const std::string& file_path,
    std::vector<std::uint32_t>& out_prefix_tokens,
    hipStream_t stream) {
    return PersistentSession::load_from_file(file_path, *this, out_prefix_tokens, stream);
}

bool PrefillV2Model::restore_matching_session(
    const std::string& session_dir,
    std::span<const std::uint32_t> full_prompt,
    std::uint32_t& out_prefix_length,
    hipStream_t stream) {
    std::string session_path;
    if (!PersistentSession::find_matching_session(
            session_dir, model_name_, quantization_, full_prompt, session_path, out_prefix_length)) {
        return false;
    }
    std::vector<std::uint32_t> loaded_tokens;
    return load_session(session_path, loaded_tokens, stream);
}

PrefillV2Model::SnapshotBacking::SnapshotBacking(
    std::shared_ptr<SnapshotBacking> parent_backing)
    : parent(std::move(parent_backing)) {
    if (parent != nullptr) ++parent->references;
}

PrefillV2Model::SnapshotBacking::~SnapshotBacking() {
    if (d_gdn != nullptr) (void)hipFree(d_gdn);
    if (d_kv_suffix != nullptr) (void)hipFree(d_kv_suffix);
    if (parent != nullptr && parent->references > 0) --parent->references;
}

const PrefillV2Model::SnapshotRecord* PrefillV2Model::find_longest_snapshot_prefix(
    std::span<const std::uint32_t> tokens) const noexcept {
    const SnapshotRecord* best = nullptr;
    for (const auto& [id, record] : snapshots_) {
        (void)id;
        if (record.tokens.size() >= tokens.size()
            || (best != nullptr && record.tokens.size() <= best->tokens.size())) continue;
        if (std::equal(record.tokens.begin(), record.tokens.end(), tokens.begin())) best = &record;
    }
    return best;
}

std::shared_ptr<PrefillV2Model::SnapshotBacking>
PrefillV2Model::capture_snapshot_backing(
    std::span<const std::uint32_t> tokens, hipStream_t stream) {
    const SnapshotRecord* parent_record = find_longest_snapshot_prefix(tokens);
    const std::size_t parent_length = parent_record == nullptr ? 0 : parent_record->tokens.size();
    auto backing = std::make_shared<SnapshotBacking>(
        parent_record == nullptr ? nullptr : parent_record->backing);
    backing->prefix_length = tokens.size();
    backing->suffix_begin = parent_length;
    backing->suffix_tokens = tokens.size() - parent_length;

    const std::size_t gdn_layer_bytes = RecurrentLayerState::kStateBytes
        + RecurrentLayerState::kConvHistoryBytes;
    const std::size_t gdn_bytes = recurrent_states_.size() * gdn_layer_bytes;
    MIINFER_HIP_CHECK(hipMalloc(&backing->d_gdn, gdn_bytes));
    std::size_t gdn_offset = 0;
    for (std::size_t i = 0; i < recurrent_states_.size(); ++i) {
        const auto view = recurrent_storage(i).view();
        if (stream != nullptr) {
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                static_cast<std::uint8_t*>(backing->d_gdn) + gdn_offset,
                view.d_state, RecurrentLayerState::kStateBytes,
                hipMemcpyDeviceToDevice, stream));
            gdn_offset += RecurrentLayerState::kStateBytes;
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                static_cast<std::uint8_t*>(backing->d_gdn) + gdn_offset,
                view.d_conv_history, RecurrentLayerState::kConvHistoryBytes,
                hipMemcpyDeviceToDevice, stream));
            gdn_offset += RecurrentLayerState::kConvHistoryBytes;
        } else {
            MIINFER_HIP_CHECK(hipMemcpy(
                static_cast<std::uint8_t*>(backing->d_gdn) + gdn_offset,
                view.d_state, RecurrentLayerState::kStateBytes,
                hipMemcpyDeviceToDevice));
            gdn_offset += RecurrentLayerState::kStateBytes;
            MIINFER_HIP_CHECK(hipMemcpy(
                static_cast<std::uint8_t*>(backing->d_gdn) + gdn_offset,
                view.d_conv_history, RecurrentLayerState::kConvHistoryBytes,
                hipMemcpyDeviceToDevice));
            gdn_offset += RecurrentLayerState::kConvHistoryBytes;
        }
    }

    const std::size_t per_layer = kv_caches_.front().raw_tokens_bytes(backing->suffix_tokens);
    const std::size_t kv_bytes = kv_caches_.size() * per_layer;
    if (kv_bytes > 0) MIINFER_HIP_CHECK(hipMalloc(&backing->d_kv_suffix, kv_bytes));
    std::vector<std::uint8_t> host_range(per_layer);
    if (stream != nullptr) MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
    for (std::size_t i = 0; i < kv_caches_.size(); ++i) {
        kv_caches_[i].download_raw_range(
            host_range.data(), parent_length, backing->suffix_tokens, nullptr);
        if (per_layer > 0) {
            MIINFER_HIP_CHECK(hipMemcpy(
                static_cast<std::uint8_t*>(backing->d_kv_suffix) + i * per_layer,
                host_range.data(), per_layer, hipMemcpyHostToDevice));
        }
    }
    backing->bytes = gdn_bytes + kv_bytes;
    return backing;
}

void PrefillV2Model::restore_snapshot_backing(
    const std::shared_ptr<SnapshotBacking>& backing, hipStream_t stream) {
    for (auto& kv : kv_caches_) kv.reset(stream);

    std::vector<const SnapshotBacking*> chain;
    for (auto current = backing; current != nullptr; current = current->parent) {
        chain.push_back(current.get());
    }
    std::reverse(chain.begin(), chain.end());
    for (const auto* node : chain) {
        const std::size_t per_layer = kv_caches_.front().raw_tokens_bytes(node->suffix_tokens);
        std::vector<std::uint8_t> host_range(per_layer);
        for (std::size_t i = 0; i < kv_caches_.size(); ++i) {
            if (per_layer > 0) {
                MIINFER_HIP_CHECK(hipMemcpy(
                    host_range.data(),
                    static_cast<const std::uint8_t*>(node->d_kv_suffix) + i * per_layer,
                    per_layer, hipMemcpyDeviceToHost));
                kv_caches_[i].upload_raw_range(
                    host_range.data(), node->suffix_begin, node->suffix_tokens, stream);
            }
        }
    }

    std::size_t gdn_offset = 0;
    for (std::size_t i = 0; i < recurrent_states_.size(); ++i) {
        auto view = recurrent_storage(i).view();
        if (stream != nullptr) {
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                view.d_state, static_cast<const std::uint8_t*>(backing->d_gdn) + gdn_offset,
                RecurrentLayerState::kStateBytes, hipMemcpyDeviceToDevice, stream));
            gdn_offset += RecurrentLayerState::kStateBytes;
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                view.d_conv_history, static_cast<const std::uint8_t*>(backing->d_gdn) + gdn_offset,
                RecurrentLayerState::kConvHistoryBytes, hipMemcpyDeviceToDevice, stream));
            gdn_offset += RecurrentLayerState::kConvHistoryBytes;
        } else {
            MIINFER_HIP_CHECK(hipMemcpy(
                view.d_state, static_cast<const std::uint8_t*>(backing->d_gdn) + gdn_offset,
                RecurrentLayerState::kStateBytes, hipMemcpyDeviceToDevice));
            gdn_offset += RecurrentLayerState::kStateBytes;
            MIINFER_HIP_CHECK(hipMemcpy(
                view.d_conv_history, static_cast<const std::uint8_t*>(backing->d_gdn) + gdn_offset,
                RecurrentLayerState::kConvHistoryBytes, hipMemcpyDeviceToDevice));
            gdn_offset += RecurrentLayerState::kConvHistoryBytes;
        }
        recurrent_storage(i).set_position(static_cast<std::uint32_t>(backing->prefix_length));
    }
    if (stream != nullptr) MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
}

void PrefillV2Model::refresh_snapshot_bytes() noexcept {
    std::unordered_set<const SnapshotBacking*> seen;
    std::size_t total = 0;
    for (const auto& [id, record] : snapshots_) {
        (void)id;
        for (auto current = record.backing; current != nullptr; current = current->parent) {
            if (seen.insert(current.get()).second) total += current->bytes;
        }
    }
    snapshot_bytes_ = total;
}

PrefillV2Model::SnapshotTelemetry PrefillV2Model::snapshot_telemetry() const noexcept {
    SnapshotTelemetry result;
    result.logical_snapshot_count = snapshots_.size();
    std::unordered_set<const SnapshotBacking*> seen;
    for (const auto& [id, record] : snapshots_) {
        (void)id;
        for (auto current = record.backing; current != nullptr; current = current->parent) {
            if (!seen.insert(current.get()).second) continue;
            result.reference_count += current->references;
            if (current->references > 1) result.physical_shared_bytes += current->bytes;
            else result.private_branch_bytes += current->bytes;
        }
    }
    result.cow_events = snapshot_cow_events_;
    result.cow_bytes_copied = snapshot_cow_bytes_;
    return result;
}

void PrefillV2Model::release_snapshot_record(SnapshotRecord& record) noexcept {
    if (record.backing != nullptr && record.backing->references > 0) --record.backing->references;
    record.backing.reset();
}

void PrefillV2Model::evict_snapshots_except(SnapshotId preserve) noexcept {
    refresh_snapshot_bytes();
    while ((snapshots_.size() > kMaxPersistentCheckpoints
            || snapshot_bytes_ > kPersistentCheckpointBudgetBytes)
           && !snapshots_.empty()) {
        auto candidate = snapshots_.end();
        for (auto it = snapshots_.begin(); it != snapshots_.end(); ++it) {
            if (it->first == preserve) continue;
            if (candidate == snapshots_.end()
                || it->second.last_used < candidate->second.last_used) candidate = it;
        }
        if (candidate == snapshots_.end()) break;
        release_snapshot(candidate->first);
    }
}

PrefillV2Model::SnapshotId PrefillV2Model::snapshot(
    std::span<const std::uint32_t> tokens, hipStream_t stream) {
    if (tokens.empty() || tokens.size() > kv_capacity_) return kInvalidSnapshotId;
    try {
        SnapshotId id = next_snapshot_id_++;
        if (id == kInvalidSnapshotId) id = next_snapshot_id_++;

        std::shared_ptr<SnapshotBacking> backing;
        for (const auto& [existing_id, record] : snapshots_) {
            (void)existing_id;
            if (record.tokens.size() == tokens.size()
                && std::equal(record.tokens.begin(), record.tokens.end(), tokens.begin())) {
                backing = record.backing;
                break;
            }
        }
        if (backing == nullptr) {
            backing = capture_snapshot_backing(tokens, stream);
            snapshot_bytes_ += backing->bytes;
            if (backing->parent != nullptr) {
                ++snapshot_cow_events_;
                snapshot_cow_bytes_ += backing->bytes;
            }
        }
        ++backing->references;
        snapshots_.emplace(id, SnapshotRecord{
            backing, std::vector<std::uint32_t>(tokens.begin(), tokens.end()),
            ++snapshot_use_clock_});
        refresh_snapshot_bytes();
        evict_snapshots_except(id);
        return snapshots_.find(id) == snapshots_.end() ? kInvalidSnapshotId : id;
    } catch (...) {
        return kInvalidSnapshotId;
    }
}

PrefillV2Model::SnapshotId PrefillV2Model::fork(SnapshotId source) {
    const auto found = snapshots_.find(source);
    if (found == snapshots_.end()) return kInvalidSnapshotId;
    SnapshotId id = next_snapshot_id_++;
    if (id == kInvalidSnapshotId) id = next_snapshot_id_++;
    ++found->second.backing->references;
    snapshots_.emplace(id, SnapshotRecord{
        found->second.backing, found->second.tokens, ++snapshot_use_clock_});
    found->second.last_used = ++snapshot_use_clock_;
    evict_snapshots_except(id);
    return snapshots_.find(id) == snapshots_.end() ? kInvalidSnapshotId : id;
}

bool PrefillV2Model::restore_snapshot(SnapshotId id, hipStream_t stream) {
    const auto found = snapshots_.find(id);
    if (found == snapshots_.end()) return false;
    restore_snapshot_backing(found->second.backing, stream);
    reusable_context_.save(found->second.tokens, recurrent_states_, nullptr, stream);
    if (stream != nullptr) MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
    found->second.last_used = ++snapshot_use_clock_;
    snapshot_state_loaded_ = true;
    return true;
}

bool PrefillV2Model::rollback_snapshot(SnapshotId id, hipStream_t stream) {
    return restore_snapshot(id, stream);
}

bool PrefillV2Model::release_snapshot(SnapshotId id) noexcept {
    const auto found = snapshots_.find(id);
    if (found == snapshots_.end()) return false;
    release_snapshot_record(found->second);
    snapshots_.erase(found);
    refresh_snapshot_bytes();
    return true;
}

void PrefillV2Model::clear_snapshots() noexcept {
    snapshots_.clear();
    snapshot_bytes_ = 0;
    snapshot_use_clock_ = 0;
    snapshot_state_loaded_ = false;
    snapshot_cow_events_ = 0;
    snapshot_cow_bytes_ = 0;
}

} // namespace miinfer::prefill_v2
