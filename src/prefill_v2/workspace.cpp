#include "miinfer/prefill_v2/workspace.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace miinfer::prefill_v2 {

namespace {

inline std::size_t align128(std::size_t bytes) {
    return (bytes + 127) & ~static_cast<std::size_t>(127);
}

} // namespace

RecurrentLayerWorkspaceManager::RecurrentLayerWorkspaceManager(std::size_t max_tokens)
    : max_tokens_(max_tokens) {
    if (max_tokens_ == 0 || max_tokens_ > kMaxPrefillBatch || max_tokens_ % kGdnChunkSize != 0) {
        throw std::runtime_error("RecurrentLayerWorkspaceManager: invalid max_tokens " + std::to_string(max_tokens_));
    }

    // 1. Calculate required bytes for all buffers
    const std::size_t norm_bytes = align128(max_tokens_ * kHidden * sizeof(float));
    const std::size_t qkv_bytes = align128(max_tokens_ * kChannels * sizeof(float));
    const std::size_t gate_bytes = align128(max_tokens_ * kInner * sizeof(float));
    const std::size_t beta_decay_bytes = align128(max_tokens_ * kVHeads * sizeof(float));

    const std::size_t query_bytes = align128(max_tokens_ * kKHeads * kState * sizeof(float));
    const std::size_t key_bytes = align128(max_tokens_ * kKHeads * kState * sizeof(float));
    const std::size_t value_bytes = align128(max_tokens_ * kVHeads * kState * sizeof(float));

    // GDN chunkwise workspace: 5 buffers of [kVHeads * kGdnChunkSize * kState]
    const std::size_t gdn_single_bytes = align128(kVHeads * kGdnChunkSize * kState * sizeof(float));
    const std::size_t gdn_scratch_bytes = 5 * gdn_single_bytes;

    const std::size_t gdn_raw_bytes = align128(max_tokens_ * kVHeads * kState * sizeof(float));
    const std::size_t gated_bytes = align128(max_tokens_ * kVHeads * kState * sizeof(float));

    const std::size_t ssm_out_bytes = align128(max_tokens_ * kHidden * sizeof(float));
    const std::size_t residual_bytes = align128(max_tokens_ * kHidden * sizeof(float));
    const std::size_t post_norm_bytes = align128(max_tokens_ * kHidden * sizeof(float));

    // MMQ Q8_1 quantization blocks: max columns is kFfnInner = 17408 (136 blocks of 128)
    const std::size_t max_blocks_per_token = kFfnInner / 128;
    const std::size_t mmq_q8_bytes = align128(max_tokens_ * max_blocks_per_token * sizeof(M23Q8_1MmqBlock));

    const std::size_t ffn_gate_bytes = align128(max_tokens_ * kFfnInner * sizeof(float));
    const std::size_t ffn_up_bytes = align128(max_tokens_ * kFfnInner * sizeof(float));
    const std::size_t ffn_act_bytes = align128(max_tokens_ * kFfnInner * sizeof(float));
    const std::size_t ffn_down_bytes = align128(max_tokens_ * kHidden * sizeof(float));

    total_bytes_ = norm_bytes + qkv_bytes + gate_bytes + 4 * beta_decay_bytes
        + query_bytes + key_bytes + value_bytes + gdn_scratch_bytes
        + gdn_raw_bytes + gated_bytes + ssm_out_bytes + residual_bytes + post_norm_bytes
        + mmq_q8_bytes + ffn_gate_bytes + ffn_up_bytes + ffn_act_bytes + ffn_down_bytes;

    MIINFER_HIP_CHECK(hipMalloc(&d_buffer_, total_bytes_));

    // 2. Assign buffer pointers into the contiguous arena
    std::byte* ptr = static_cast<std::byte*>(d_buffer_);

    workspace_.normalized = reinterpret_cast<float*>(ptr); ptr += norm_bytes;
    workspace_.qkv = reinterpret_cast<float*>(ptr); ptr += qkv_bytes;
    workspace_.gate = reinterpret_cast<float*>(ptr); ptr += gate_bytes;
    workspace_.raw_beta = reinterpret_cast<float*>(ptr); ptr += beta_decay_bytes;
    workspace_.raw_alpha = reinterpret_cast<float*>(ptr); ptr += beta_decay_bytes;
    workspace_.beta = reinterpret_cast<float*>(ptr); ptr += beta_decay_bytes;
    workspace_.decay = reinterpret_cast<float*>(ptr); ptr += beta_decay_bytes;

    workspace_.query = reinterpret_cast<float*>(ptr); ptr += query_bytes;
    workspace_.key = reinterpret_cast<float*>(ptr); ptr += key_bytes;
    workspace_.value = reinterpret_cast<float*>(ptr); ptr += value_bytes;

    workspace_.gdn_scratch.new_values = reinterpret_cast<float*>(ptr); ptr += gdn_single_bytes;
    workspace_.gdn_scratch.decayed_keys = reinterpret_cast<float*>(ptr); ptr += gdn_single_bytes;
    workspace_.gdn_scratch.solved_values = reinterpret_cast<float*>(ptr); ptr += gdn_single_bytes;
    workspace_.gdn_scratch.solved_keys = reinterpret_cast<float*>(ptr); ptr += gdn_single_bytes;
    workspace_.gdn_scratch.corrected_values = reinterpret_cast<float*>(ptr); ptr += gdn_single_bytes;

    workspace_.gdn_raw_output = reinterpret_cast<float*>(ptr); ptr += gdn_raw_bytes;
    workspace_.gated_output = reinterpret_cast<float*>(ptr); ptr += gated_bytes;

    workspace_.ssm_output = reinterpret_cast<float*>(ptr); ptr += ssm_out_bytes;
    workspace_.residual = reinterpret_cast<float*>(ptr); ptr += residual_bytes;
    workspace_.post_normalized = reinterpret_cast<float*>(ptr); ptr += post_norm_bytes;

    workspace_.mmq_q8 = reinterpret_cast<M23Q8_1MmqBlock*>(ptr); ptr += mmq_q8_bytes;

    workspace_.ffn_gate = reinterpret_cast<float*>(ptr); ptr += ffn_gate_bytes;
    workspace_.ffn_up = reinterpret_cast<float*>(ptr); ptr += ffn_up_bytes;
    workspace_.ffn_activation = reinterpret_cast<float*>(ptr); ptr += ffn_act_bytes;
    workspace_.ffn_down = reinterpret_cast<float*>(ptr); ptr += ffn_down_bytes;
}

RecurrentLayerWorkspaceManager::~RecurrentLayerWorkspaceManager() {
    if (d_buffer_ != nullptr) {
        (void)hipFree(d_buffer_);
        d_buffer_ = nullptr;
    }
}

RecurrentLayerWorkspaceManager::RecurrentLayerWorkspaceManager(RecurrentLayerWorkspaceManager&& other) noexcept
    : max_tokens_(other.max_tokens_),
      total_bytes_(other.total_bytes_),
      d_buffer_(std::exchange(other.d_buffer_, nullptr)),
      workspace_(other.workspace_) {}

RecurrentLayerWorkspaceManager& RecurrentLayerWorkspaceManager::operator=(RecurrentLayerWorkspaceManager&& other) noexcept {
    if (this != &other) {
        if (d_buffer_ != nullptr) (void)hipFree(d_buffer_);
        max_tokens_ = other.max_tokens_;
        total_bytes_ = other.total_bytes_;
        d_buffer_ = std::exchange(other.d_buffer_, nullptr);
        workspace_ = other.workspace_;
    }
    return *this;
}

} // namespace miinfer::prefill_v2
