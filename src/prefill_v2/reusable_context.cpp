#include "miinfer/prefill_v2/reusable_context.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace miinfer::prefill_v2 {

namespace {

constexpr std::size_t kGdnLayerCount = 48;

} // namespace

GdnCheckpointStorage::GdnCheckpointStorage() {
    const std::size_t states_bytes = kGdnLayerCount * RecurrentLayerState::kStateBytes;
    const std::size_t history_bytes = kGdnLayerCount * RecurrentLayerState::kConvHistoryBytes;
    total_bytes_ = states_bytes + history_bytes;

    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_checkpoint_states_), states_bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_checkpoint_conv_history_), history_bytes));
    is_valid_ = false;
}

GdnCheckpointStorage::~GdnCheckpointStorage() {
    if (d_checkpoint_states_ != nullptr) {
        (void)hipFree(d_checkpoint_states_);
        d_checkpoint_states_ = nullptr;
    }
    if (d_checkpoint_conv_history_ != nullptr) {
        (void)hipFree(d_checkpoint_conv_history_);
        d_checkpoint_conv_history_ = nullptr;
    }
    total_bytes_ = 0;
    is_valid_ = false;
}

GdnCheckpointStorage::GdnCheckpointStorage(GdnCheckpointStorage&& other) noexcept
    : d_checkpoint_states_(std::exchange(other.d_checkpoint_states_, nullptr)),
      d_checkpoint_conv_history_(std::exchange(other.d_checkpoint_conv_history_, nullptr)),
      total_bytes_(other.total_bytes_),
      is_valid_(other.is_valid_) {
    other.total_bytes_ = 0;
    other.is_valid_ = false;
}

GdnCheckpointStorage& GdnCheckpointStorage::operator=(GdnCheckpointStorage&& other) noexcept {
    if (this != &other) {
        if (d_checkpoint_states_ != nullptr) (void)hipFree(d_checkpoint_states_);
        if (d_checkpoint_conv_history_ != nullptr) (void)hipFree(d_checkpoint_conv_history_);

        d_checkpoint_states_ = std::exchange(other.d_checkpoint_states_, nullptr);
        d_checkpoint_conv_history_ = std::exchange(other.d_checkpoint_conv_history_, nullptr);
        total_bytes_ = other.total_bytes_;
        is_valid_ = other.is_valid_;

        other.total_bytes_ = 0;
        other.is_valid_ = false;
    }
    return *this;
}

void GdnCheckpointStorage::capture(
    const std::vector<RecurrentLayerStateStorage>& active_states,
    hipStream_t stream) {
    if (active_states.size() != kGdnLayerCount) {
        throw std::runtime_error("GdnCheckpointStorage::capture: expected " + std::to_string(kGdnLayerCount) +
                                 " active GDN states, got " + std::to_string(active_states.size()));
    }
    if (d_checkpoint_states_ == nullptr || d_checkpoint_conv_history_ == nullptr) {
        throw std::runtime_error("GdnCheckpointStorage::capture: unallocated checkpoint storage");
    }

    for (std::size_t i = 0; i < kGdnLayerCount; ++i) {
        auto view = active_states[i].view();
        float* dst_state = d_checkpoint_states_ + i * RecurrentLayerState::kStateElements;
        float* dst_history = d_checkpoint_conv_history_ + i * RecurrentLayerState::kConvHistoryElements;

        MIINFER_HIP_CHECK(hipMemcpyAsync(
            dst_state,
            view.d_state,
            RecurrentLayerState::kStateBytes,
            hipMemcpyDeviceToDevice,
            stream));

        MIINFER_HIP_CHECK(hipMemcpyAsync(
            dst_history,
            view.d_conv_history,
            RecurrentLayerState::kConvHistoryBytes,
            hipMemcpyDeviceToDevice,
            stream));
    }
    is_valid_ = true;
}

void GdnCheckpointStorage::restore(
    std::vector<RecurrentLayerStateStorage>& active_states,
    hipStream_t stream) const {
    if (!is_valid_) {
        throw std::runtime_error("GdnCheckpointStorage::restore: attempted to restore from invalid checkpoint");
    }
    if (active_states.size() != kGdnLayerCount) {
        throw std::runtime_error("GdnCheckpointStorage::restore: expected " + std::to_string(kGdnLayerCount) +
                                 " active GDN states, got " + std::to_string(active_states.size()));
    }

    for (std::size_t i = 0; i < kGdnLayerCount; ++i) {
        auto view = active_states[i].view();
        const float* src_state = d_checkpoint_states_ + i * RecurrentLayerState::kStateElements;
        const float* src_history = d_checkpoint_conv_history_ + i * RecurrentLayerState::kConvHistoryElements;

        MIINFER_HIP_CHECK(hipMemcpyAsync(
            view.d_state,
            src_state,
            RecurrentLayerState::kStateBytes,
            hipMemcpyDeviceToDevice,
            stream));

        MIINFER_HIP_CHECK(hipMemcpyAsync(
            view.d_conv_history,
            src_history,
            RecurrentLayerState::kConvHistoryBytes,
            hipMemcpyDeviceToDevice,
            stream));
    }
}

ReusableContext::ReusableContext(std::string model_id, std::string quantization) {
    fingerprint_.model_id = std::move(model_id);
    fingerprint_.quantization = std::move(quantization);
    fingerprint_.prefix_length = 0;
    fingerprint_.token_hash = 0;
    fingerprint_.state_layout_version = kStateLayoutVersion;
}

void ReusableContext::save(
    std::span<const std::uint32_t> prefix_tokens,
    const std::vector<RecurrentLayerStateStorage>& active_states,
    hipStream_t stream) {
    if (prefix_tokens.empty()) {
        clear();
        return;
    }

    gdn_checkpoint_.capture(active_states, stream);

    cached_tokens_.assign(prefix_tokens.begin(), prefix_tokens.end());
    fingerprint_.prefix_length = static_cast<std::uint32_t>(prefix_tokens.size());
    fingerprint_.token_hash = compute_token_sequence_hash(prefix_tokens);
    fingerprint_.state_layout_version = kStateLayoutVersion;
}

ReusableContext::MatchResult ReusableContext::check_match(
    std::span<const std::uint32_t> full_prompt,
    const std::string& model_id,
    const std::string& quantization) const {

    if (!has_valid_prefix()) {
        return MatchResult::EmptyCache;
    }
    if (fingerprint_.state_layout_version != kStateLayoutVersion) {
        return MatchResult::VersionMismatch;
    }
    if (fingerprint_.model_id != model_id) {
        return MatchResult::ModelMismatch;
    }
    if (fingerprint_.quantization != quantization) {
        return MatchResult::QuantizationMismatch;
    }
    if (full_prompt.size() < fingerprint_.prefix_length) {
        return MatchResult::PromptShorterThanPrefix;
    }

    // Verify token sequence identity for the prefix
    const std::uint32_t P = fingerprint_.prefix_length;
    for (std::uint32_t i = 0; i < P; ++i) {
        if (full_prompt[i] != cached_tokens_[i]) {
            return MatchResult::PrefixMismatch;
        }
    }

    return MatchResult::ExactMatch;
}

void ReusableContext::restore_gdn_states(
    std::vector<RecurrentLayerStateStorage>& active_states,
    hipStream_t stream) const {
    if (!has_valid_prefix()) {
        throw std::runtime_error("ReusableContext::restore_gdn_states: no valid prefix to restore");
    }
    gdn_checkpoint_.restore(active_states, stream);

    // Explicitly restore positions for all 48 GDN layers to prefix_length
    const std::uint32_t P = fingerprint_.prefix_length;
    for (auto& st : active_states) {
        st.set_position(P);
    }
}

void ReusableContext::clear() {
    gdn_checkpoint_.invalidate();
    cached_tokens_.clear();
    fingerprint_.prefix_length = 0;
    fingerprint_.token_hash = 0;
}

} // namespace miinfer::prefill_v2
