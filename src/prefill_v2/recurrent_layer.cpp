#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "miinfer/prefill_v2/recurrent_layer.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/m12_gdn_chunk.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

namespace miinfer::prefill_v2 {

namespace {

const GgufTensor* require_tensor(const GgufFile& file, std::string_view name) {
    for (const auto& tensor : file.tensors()) {
        if (tensor.name == name) return &tensor;
    }
    throw std::runtime_error("PrefillV2RecurrentLayer: missing required tensor: " + std::string(name));
}

template <typename T>
T* upload_device_buffer(const void* host_data, std::size_t bytes, std::size_t& total_bytes) {
    T* device_ptr = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_ptr), bytes));
    MIINFER_HIP_CHECK(hipMemcpy(device_ptr, host_data, bytes, hipMemcpyHostToDevice));
    total_bytes += bytes;
    return device_ptr;
}

} // namespace

PrefillV2RecurrentLayer::PrefillV2RecurrentLayer(const Qwen35Model& model, std::size_t layer_index)
    : layer_index_(layer_index) {
    const auto& file = *model.file();
    const std::string prefix = "blk." + std::to_string(layer_index_) + ".";

    // 1. Attention RMS Norm weight: F32[5120]
    const auto* attn_norm_tensor = require_tensor(file, prefix + "attn_norm.weight");
    if (attn_norm_tensor->type != GgufTensorType::f32 || attn_norm_tensor->byte_size != kHidden * sizeof(float)) {
        throw std::runtime_error("PrefillV2: invalid attn_norm weight tensor");
    }
    d_attn_norm_ = upload_device_buffer<float>(
        attn_norm_tensor->data, attn_norm_tensor->byte_size, persistent_weight_bytes_);

    // 2. QKV Projection: Q4_K or Q6_K [5120, 10240]
    const auto* qkv_tensor = require_tensor(file, prefix + "attn_qkv.weight");
    qkv_type_ = qkv_tensor->type;
    if (qkv_type_ == GgufTensorType::q4_k) {
        const auto packed = pack_mx_q4k_repacked_tensor(*qkv_tensor);
        d_qkv_mmq_ = upload_device_buffer<std::uint8_t>(
            packed.data(), packed.size(), persistent_weight_bytes_);
    } else if (qkv_type_ == GgufTensorType::q6_k) {
        const auto packed = pack_mx_q6k_repacked_tensor(*qkv_tensor);
        d_qkv_mmq_ = upload_device_buffer<std::uint8_t>(
            packed.data(), packed.size(), persistent_weight_bytes_);
    } else {
        throw std::runtime_error("PrefillV2: unsupported attn_qkv quantization type");
    }

    // 3. Attention Gate Projection: Q4_K [5120, 6144]
    const auto* gate_tensor = require_tensor(file, prefix + "attn_gate.weight");
    if (gate_tensor->type != GgufTensorType::q4_k) {
        throw std::runtime_error("PrefillV2: unsupported attn_gate quantization type");
    }
    const auto gate_packed = pack_mx_q4k_repacked_tensor(*gate_tensor);
    d_gate_mmq_ = upload_device_buffer<std::uint8_t>(
        gate_packed.data(), gate_packed.size(), persistent_weight_bytes_);

    // 4. SSM Beta and Alpha weights: F32[5120, 48]
    const auto* beta_tensor = require_tensor(file, prefix + "ssm_beta.weight");
    if (beta_tensor->type != GgufTensorType::f32) {
        throw std::runtime_error("PrefillV2: invalid ssm_beta weight type");
    }
    d_ssm_beta_ = upload_device_buffer<float>(
        beta_tensor->data, beta_tensor->byte_size, persistent_weight_bytes_);

    const auto* alpha_tensor = require_tensor(file, prefix + "ssm_alpha.weight");
    if (alpha_tensor->type != GgufTensorType::f32) {
        throw std::runtime_error("PrefillV2: invalid ssm_alpha weight type");
    }
    d_ssm_alpha_ = upload_device_buffer<float>(
        alpha_tensor->data, alpha_tensor->byte_size, persistent_weight_bytes_);

    // 5. SSM dt bias and A: F32[48]
    const auto* dt_tensor = require_tensor(file, prefix + "ssm_dt.bias");
    if (dt_tensor->type != GgufTensorType::f32) {
        throw std::runtime_error("PrefillV2: invalid ssm_dt bias type");
    }
    d_ssm_dt_ = upload_device_buffer<float>(
        dt_tensor->data, dt_tensor->byte_size, persistent_weight_bytes_);

    const auto* a_tensor = require_tensor(file, prefix + "ssm_a");
    if (a_tensor->type != GgufTensorType::f32) {
        throw std::runtime_error("PrefillV2: invalid ssm_a tensor type");
    }
    d_ssm_a_ = upload_device_buffer<float>(
        a_tensor->data, a_tensor->byte_size, persistent_weight_bytes_);

    // 6. SSM Conv1D weights: F32[4, 10240]
    const auto* conv_tensor = require_tensor(file, prefix + "ssm_conv1d.weight");
    if (conv_tensor->type != GgufTensorType::f32) {
        throw std::runtime_error("PrefillV2: invalid ssm_conv1d weight type");
    }
    d_ssm_conv_ = upload_device_buffer<float>(
        conv_tensor->data, conv_tensor->byte_size, persistent_weight_bytes_);

    // 7. SSM Norm weight: F32[128]
    const auto* ssm_norm_tensor = require_tensor(file, prefix + "ssm_norm.weight");
    if (ssm_norm_tensor->type != GgufTensorType::f32) {
        throw std::runtime_error("PrefillV2: invalid ssm_norm weight type");
    }
    d_ssm_norm_ = upload_device_buffer<float>(
        ssm_norm_tensor->data, ssm_norm_tensor->byte_size, persistent_weight_bytes_);

    // 8. SSM Out projection: Q5_K [6144, 5120]
    const auto* ssm_out_tensor = require_tensor(file, prefix + "ssm_out.weight");
    if (ssm_out_tensor->type != GgufTensorType::q5_k) {
        throw std::runtime_error("PrefillV2: unsupported ssm_out quantization type");
    }
    const auto ssm_out_packed = pack_mx_q5k_repacked_tensor(*ssm_out_tensor);
    d_ssm_out_mmq_ = upload_device_buffer<std::uint8_t>(
        ssm_out_packed.data(), ssm_out_packed.size(), persistent_weight_bytes_);

    // 9. Post Attention RMS Norm: F32[5120]
    const auto* post_norm_tensor = require_tensor(file, prefix + "post_attention_norm.weight");
    if (post_norm_tensor->type != GgufTensorType::f32) {
        throw std::runtime_error("PrefillV2: invalid post_attention_norm weight type");
    }
    d_post_norm_ = upload_device_buffer<float>(
        post_norm_tensor->data, post_norm_tensor->byte_size, persistent_weight_bytes_);

    // 10. FFN Gate & Up projections: Q4_K [5120, 17408]
    const auto* ffn_gate_tensor = require_tensor(file, prefix + "ffn_gate.weight");
    if (ffn_gate_tensor->type != GgufTensorType::q4_k) {
        throw std::runtime_error("PrefillV2: unsupported ffn_gate quantization type");
    }
    const auto ffn_gate_packed = pack_mx_q4k_repacked_tensor(*ffn_gate_tensor);
    d_ffn_gate_mmq_ = upload_device_buffer<std::uint8_t>(
        ffn_gate_packed.data(), ffn_gate_packed.size(), persistent_weight_bytes_);

    const auto* ffn_up_tensor = require_tensor(file, prefix + "ffn_up.weight");
    if (ffn_up_tensor->type != GgufTensorType::q4_k) {
        throw std::runtime_error("PrefillV2: unsupported ffn_up quantization type");
    }
    const auto ffn_up_packed = pack_mx_q4k_repacked_tensor(*ffn_up_tensor);
    d_ffn_up_mmq_ = upload_device_buffer<std::uint8_t>(
        ffn_up_packed.data(), ffn_up_packed.size(), persistent_weight_bytes_);

    // 11. FFN Down projection: Q4_K or Q6_K [17408, 5120]
    const auto* ffn_down_tensor = require_tensor(file, prefix + "ffn_down.weight");
    ffn_down_type_ = ffn_down_tensor->type;
    if (ffn_down_type_ == GgufTensorType::q4_k) {
        const auto packed = pack_mx_q4k_repacked_tensor(*ffn_down_tensor);
        d_ffn_down_mmq_ = upload_device_buffer<std::uint8_t>(
            packed.data(), packed.size(), persistent_weight_bytes_);
    } else if (ffn_down_type_ == GgufTensorType::q6_k) {
        const auto packed = pack_mx_q6k_repacked_tensor(*ffn_down_tensor);
        d_ffn_down_mmq_ = upload_device_buffer<std::uint8_t>(
            packed.data(), packed.size(), persistent_weight_bytes_);
    } else {
        throw std::runtime_error("PrefillV2: unsupported ffn_down quantization type");
    }
}

PrefillV2RecurrentLayer::~PrefillV2RecurrentLayer() {
    if (d_attn_norm_ != nullptr) (void)hipFree(d_attn_norm_);
    if (d_qkv_mmq_ != nullptr) (void)hipFree(d_qkv_mmq_);
    if (d_gate_mmq_ != nullptr) (void)hipFree(d_gate_mmq_);
    if (d_ssm_beta_ != nullptr) (void)hipFree(d_ssm_beta_);
    if (d_ssm_alpha_ != nullptr) (void)hipFree(d_ssm_alpha_);
    if (d_ssm_dt_ != nullptr) (void)hipFree(d_ssm_dt_);
    if (d_ssm_a_ != nullptr) (void)hipFree(d_ssm_a_);
    if (d_ssm_conv_ != nullptr) (void)hipFree(d_ssm_conv_);
    if (d_ssm_norm_ != nullptr) (void)hipFree(d_ssm_norm_);
    if (d_ssm_out_mmq_ != nullptr) (void)hipFree(d_ssm_out_mmq_);
    if (d_post_norm_ != nullptr) (void)hipFree(d_post_norm_);
    if (d_ffn_gate_mmq_ != nullptr) (void)hipFree(d_ffn_gate_mmq_);
    if (d_ffn_up_mmq_ != nullptr) (void)hipFree(d_ffn_up_mmq_);
    if (d_ffn_down_mmq_ != nullptr) (void)hipFree(d_ffn_down_mmq_);
}

PrefillV2RecurrentLayer::PrefillV2RecurrentLayer(PrefillV2RecurrentLayer&& other) noexcept
    : layer_index_(other.layer_index_),
      persistent_weight_bytes_(other.persistent_weight_bytes_),
      d_attn_norm_(std::exchange(other.d_attn_norm_, nullptr)),
      d_qkv_mmq_(std::exchange(other.d_qkv_mmq_, nullptr)),
      qkv_type_(other.qkv_type_),
      d_gate_mmq_(std::exchange(other.d_gate_mmq_, nullptr)),
      d_ssm_beta_(std::exchange(other.d_ssm_beta_, nullptr)),
      d_ssm_alpha_(std::exchange(other.d_ssm_alpha_, nullptr)),
      d_ssm_dt_(std::exchange(other.d_ssm_dt_, nullptr)),
      d_ssm_a_(std::exchange(other.d_ssm_a_, nullptr)),
      d_ssm_conv_(std::exchange(other.d_ssm_conv_, nullptr)),
      d_ssm_norm_(std::exchange(other.d_ssm_norm_, nullptr)),
      d_ssm_out_mmq_(std::exchange(other.d_ssm_out_mmq_, nullptr)),
      d_post_norm_(std::exchange(other.d_post_norm_, nullptr)),
      d_ffn_gate_mmq_(std::exchange(other.d_ffn_gate_mmq_, nullptr)),
      d_ffn_up_mmq_(std::exchange(other.d_ffn_up_mmq_, nullptr)),
      d_ffn_down_mmq_(std::exchange(other.d_ffn_down_mmq_, nullptr)),
      ffn_down_type_(other.ffn_down_type_) {}

PrefillV2RecurrentLayer& PrefillV2RecurrentLayer::operator=(PrefillV2RecurrentLayer&& other) noexcept {
    if (this != &other) {
        if (d_attn_norm_ != nullptr) (void)hipFree(d_attn_norm_);
        if (d_qkv_mmq_ != nullptr) (void)hipFree(d_qkv_mmq_);
        if (d_gate_mmq_ != nullptr) (void)hipFree(d_gate_mmq_);
        if (d_ssm_beta_ != nullptr) (void)hipFree(d_ssm_beta_);
        if (d_ssm_alpha_ != nullptr) (void)hipFree(d_ssm_alpha_);
        if (d_ssm_dt_ != nullptr) (void)hipFree(d_ssm_dt_);
        if (d_ssm_a_ != nullptr) (void)hipFree(d_ssm_a_);
        if (d_ssm_conv_ != nullptr) (void)hipFree(d_ssm_conv_);
        if (d_ssm_norm_ != nullptr) (void)hipFree(d_ssm_norm_);
        if (d_ssm_out_mmq_ != nullptr) (void)hipFree(d_ssm_out_mmq_);
        if (d_post_norm_ != nullptr) (void)hipFree(d_post_norm_);
        if (d_ffn_gate_mmq_ != nullptr) (void)hipFree(d_ffn_gate_mmq_);
        if (d_ffn_up_mmq_ != nullptr) (void)hipFree(d_ffn_up_mmq_);
        if (d_ffn_down_mmq_ != nullptr) (void)hipFree(d_ffn_down_mmq_);

        layer_index_ = other.layer_index_;
        persistent_weight_bytes_ = other.persistent_weight_bytes_;
        d_attn_norm_ = std::exchange(other.d_attn_norm_, nullptr);
        d_qkv_mmq_ = std::exchange(other.d_qkv_mmq_, nullptr);
        qkv_type_ = other.qkv_type_;
        d_gate_mmq_ = std::exchange(other.d_gate_mmq_, nullptr);
        d_ssm_beta_ = std::exchange(other.d_ssm_beta_, nullptr);
        d_ssm_alpha_ = std::exchange(other.d_ssm_alpha_, nullptr);
        d_ssm_dt_ = std::exchange(other.d_ssm_dt_, nullptr);
        d_ssm_a_ = std::exchange(other.d_ssm_a_, nullptr);
        d_ssm_conv_ = std::exchange(other.d_ssm_conv_, nullptr);
        d_ssm_norm_ = std::exchange(other.d_ssm_norm_, nullptr);
        d_ssm_out_mmq_ = std::exchange(other.d_ssm_out_mmq_, nullptr);
        d_post_norm_ = std::exchange(other.d_post_norm_, nullptr);
        d_ffn_gate_mmq_ = std::exchange(other.d_ffn_gate_mmq_, nullptr);
        d_ffn_up_mmq_ = std::exchange(other.d_ffn_up_mmq_, nullptr);
        d_ffn_down_mmq_ = std::exchange(other.d_ffn_down_mmq_, nullptr);
        ffn_down_type_ = other.ffn_down_type_;
    }
    return *this;
}

void PrefillV2RecurrentLayer::forward(
    const float* d_input,
    float* d_output,
    const RecurrentLayerState& incoming_state,
    RecurrentLayerState& outgoing_state,
    RecurrentLayerWorkspace& ws,
    std::uint32_t token_count,
    hipStream_t stream) const {
    if (d_input == nullptr || d_output == nullptr) {
        throw std::runtime_error("PrefillV2: null input/output pointer");
    }
    if (token_count == 0 || token_count % kGdnChunkSize != 0 || token_count > kMaxPrefillBatch) {
        throw std::runtime_error("PrefillV2: token_count must be non-zero multiple of 64 and <= "
                                 + std::to_string(kMaxPrefillBatch) + ", got " + std::to_string(token_count));
    }
    if (incoming_state.d_state == nullptr || incoming_state.d_conv_history == nullptr
        || outgoing_state.d_state == nullptr || outgoing_state.d_conv_history == nullptr) {
        throw std::runtime_error("PrefillV2: null state pointers");
    }

    // 0. Explicit state preservation: if outgoing buffers are distinct from incoming, copy them.
    if (outgoing_state.d_state != incoming_state.d_state) {
        MIINFER_HIP_CHECK(hipMemcpyAsync(outgoing_state.d_state, incoming_state.d_state,
                                        RecurrentLayerState::kStateBytes, hipMemcpyDeviceToDevice, stream));
    }
    if (outgoing_state.d_conv_history != incoming_state.d_conv_history) {
        MIINFER_HIP_CHECK(hipMemcpyAsync(outgoing_state.d_conv_history, incoming_state.d_conv_history,
                                        RecurrentLayerState::kConvHistoryBytes, hipMemcpyDeviceToDevice, stream));
    }

    const std::uint32_t base_position = incoming_state.position;

    // 1. Input RMS Normalization
    launch_qwen3_rms_norm_batch(
        d_input, d_attn_norm_, ws.normalized, token_count, kHidden, kRmsNormEpsilon, stream);

    // 2. Dual beta/alpha projection and parameter preparation
    launch_qwen35_f32_dual_gemm_batch(
        d_ssm_beta_, d_ssm_alpha_, ws.normalized, ws.raw_beta, ws.raw_alpha,
        token_count, kVHeads, kHidden, stream);
    launch_qwen35_prepare_beta_decay(
        ws.raw_beta, ws.raw_alpha, d_ssm_dt_, d_ssm_a_, ws.beta, ws.decay,
        token_count * kVHeads, stream);

    // 3. QKV & Gate projections via compact Mx MMQ
    if (qkv_type_ == GgufTensorType::q4_k) {
        launch_mx_q8_1_mmq_quantize(
            ws.normalized, ws.mmq_q8, token_count, kHidden, /*affine=*/true, stream);
        launch_mx_q4k_repacked_mmq(
            static_cast<const std::uint8_t*>(d_qkv_mmq_), ws.mmq_q8,
            ws.qkv, kChannels, kHidden, token_count, stream);
    } else {
        launch_mx_q8_1_mmq_quantize(
            ws.normalized, ws.mmq_q8, token_count, kHidden, /*affine=*/false, stream);
        launch_mx_q6k_repacked_mmq(
            static_cast<const std::uint8_t*>(d_qkv_mmq_), ws.mmq_q8,
            ws.qkv, kChannels, kHidden, token_count, stream);
        // Gate is Q4_K (affine), re-quantize normalized with affine=true
        launch_mx_q8_1_mmq_quantize(
            ws.normalized, ws.mmq_q8, token_count, kHidden, /*affine=*/true, stream);
    }

    launch_mx_q4k_repacked_mmq(
        d_gate_mmq_, ws.mmq_q8, ws.gate, kInner, kHidden, token_count, stream);

    // 4. Convolution + SiLU + split into Q, K, V (updating conv history buffer in place)
    launch_qwen35_conv_silu_split_batch(
        ws.qkv, d_ssm_conv_, outgoing_state.d_conv_history,
        ws.query, ws.key, ws.value,
        base_position, token_count, kConvKernel, kChannels, kConvKernel, stream);

    // 5. Dual Head L2 Normalization
    launch_qwen35_dual_head_l2_normalize_batch(
        ws.query, ws.key, ws.query, ws.key,
        token_count, kKHeads, kState, stream);

    // 6. Chunkwise GDN: sequential execution of internal C=64 chunks
    for (std::uint32_t chunk_start = 0; chunk_start < token_count; chunk_start += kGdnChunkSize) {
        launch_m12_gdn_chunk(
            ws.query, ws.key, ws.value, ws.beta, ws.decay,
            outgoing_state.d_state, ws.gdn_raw_output, ws.gdn_scratch,
            token_count, chunk_start, kKHeads, kVHeads, kState,
            kGdnChunkSize, stream);
    }

    // 7. SSM Postprocessing (per-head RMS norm + SSM norm scale + SiLU gate)
    launch_m12_gdn_postprocess(
        ws.gdn_raw_output, ws.gate, d_ssm_norm_, ws.gated_output,
        token_count, kVHeads, kState, kRmsNormEpsilon, stream);

    // 8. SSM Out projection (Q5_K affine)
    launch_mx_q8_1_mmq_quantize(
        ws.gated_output, ws.mmq_q8, token_count, kInner, /*affine=*/true, stream);
    launch_mx_q5k_repacked_mmq(
        d_ssm_out_mmq_, ws.mmq_q8, ws.ssm_output,
        kHidden, kInner, token_count, stream);

    // 9. Residual + Post-attention RMS Norm
    launch_qwen3_fused_add_rms_norm_batch(
        d_input, ws.ssm_output, d_post_norm_,
        ws.residual, ws.post_normalized,
        token_count, kHidden, kRmsNormEpsilon, stream);

    // 10. FFN Gate & Up projections (Q4_K affine)
    launch_mx_q8_1_mmq_quantize(
        ws.post_normalized, ws.mmq_q8, token_count, kHidden, /*affine=*/true, stream);
    launch_mx_q4k_repacked_mmq(
        d_ffn_gate_mmq_, ws.mmq_q8, ws.ffn_gate,
        kFfnInner, kHidden, token_count, stream);
    launch_mx_q4k_repacked_mmq(
        d_ffn_up_mmq_, ws.mmq_q8, ws.ffn_up,
        kFfnInner, kHidden, token_count, stream);

    // 11. SwiGLU activation
    launch_qwen3_silu_mul(
        ws.ffn_gate, ws.ffn_up, ws.ffn_activation,
        token_count * kFfnInner, stream);

    // 12. FFN Down projection
    if (ffn_down_type_ == GgufTensorType::q4_k) {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, token_count, kFfnInner, /*affine=*/true, stream);
        launch_mx_q4k_repacked_mmq(
            static_cast<const std::uint8_t*>(d_ffn_down_mmq_), ws.mmq_q8,
            ws.ffn_down, kHidden, kFfnInner, token_count, stream);
    } else {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, token_count, kFfnInner, /*affine=*/false, stream);
        launch_mx_q6k_repacked_mmq(
            static_cast<const std::uint8_t*>(d_ffn_down_mmq_), ws.mmq_q8,
            ws.ffn_down, kHidden, kFfnInner, token_count, stream);
    }

    // 13. Final residual addition
    launch_qwen3_add(
        ws.residual, ws.ffn_down, d_output,
        token_count * kHidden, stream);

    outgoing_state.position = base_position + token_count;
}

void PrefillV2RecurrentLayer::forward_profiled(
    const float* d_input,
    float* d_output,
    const RecurrentLayerState& incoming_state,
    RecurrentLayerState& outgoing_state,
    RecurrentLayerWorkspace& ws,
    std::uint32_t token_count,
    RecurrentLayerPhaseTimings& timings,
    hipStream_t stream) const {
    hipEvent_t ev_start, ev_norm, ev_qkv, ev_conv, ev_gdn, ev_ssm, ev_res, ev_ffn_up, ev_end;
    MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_norm));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_qkv));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_conv));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_gdn));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_ssm));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_res));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_ffn_up));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_end));

    MIINFER_HIP_CHECK(hipEventRecord(ev_start, stream));

    if (outgoing_state.d_state != incoming_state.d_state) {
        MIINFER_HIP_CHECK(hipMemcpyAsync(outgoing_state.d_state, incoming_state.d_state,
                                        RecurrentLayerState::kStateBytes, hipMemcpyDeviceToDevice, stream));
    }
    if (outgoing_state.d_conv_history != incoming_state.d_conv_history) {
        MIINFER_HIP_CHECK(hipMemcpyAsync(outgoing_state.d_conv_history, incoming_state.d_conv_history,
                                        RecurrentLayerState::kConvHistoryBytes, hipMemcpyDeviceToDevice, stream));
    }

    const std::uint32_t base_position = incoming_state.position;

    // 1 & 2. Norm + Beta/Alpha
    launch_qwen3_rms_norm_batch(
        d_input, d_attn_norm_, ws.normalized, token_count, kHidden, kRmsNormEpsilon, stream);
    launch_qwen35_f32_dual_gemm_batch(
        d_ssm_beta_, d_ssm_alpha_, ws.normalized, ws.raw_beta, ws.raw_alpha,
        token_count, kVHeads, kHidden, stream);
    launch_qwen35_prepare_beta_decay(
        ws.raw_beta, ws.raw_alpha, d_ssm_dt_, d_ssm_a_, ws.beta, ws.decay,
        token_count * kVHeads, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_norm, stream));

    // 3. QKV & Gate
    if (qkv_type_ == GgufTensorType::q4_k) {
        launch_mx_q8_1_mmq_quantize(
            ws.normalized, ws.mmq_q8, token_count, kHidden, /*affine=*/true, stream);
        launch_mx_q4k_repacked_mmq(
            static_cast<const std::uint8_t*>(d_qkv_mmq_), ws.mmq_q8,
            ws.qkv, kChannels, kHidden, token_count, stream);
    } else {
        launch_mx_q8_1_mmq_quantize(
            ws.normalized, ws.mmq_q8, token_count, kHidden, /*affine=*/false, stream);
        launch_mx_q6k_repacked_mmq(
            static_cast<const std::uint8_t*>(d_qkv_mmq_), ws.mmq_q8,
            ws.qkv, kChannels, kHidden, token_count, stream);
        launch_mx_q8_1_mmq_quantize(
            ws.normalized, ws.mmq_q8, token_count, kHidden, /*affine=*/true, stream);
    }
    launch_mx_q4k_repacked_mmq(
        d_gate_mmq_, ws.mmq_q8, ws.gate, kInner, kHidden, token_count, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_qkv, stream));

    // 4 & 5. Conv + L2 Norm
    launch_qwen35_conv_silu_split_batch(
        ws.qkv, d_ssm_conv_, outgoing_state.d_conv_history,
        ws.query, ws.key, ws.value,
        base_position, token_count, kConvKernel, kChannels, kConvKernel, stream);
    launch_qwen35_dual_head_l2_normalize_batch(
        ws.query, ws.key, ws.query, ws.key,
        token_count, kKHeads, kState, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_conv, stream));

    // 6. GDN C64
    for (std::uint32_t chunk_start = 0; chunk_start < token_count; chunk_start += kGdnChunkSize) {
        launch_m12_gdn_chunk(
            ws.query, ws.key, ws.value, ws.beta, ws.decay,
            outgoing_state.d_state, ws.gdn_raw_output, ws.gdn_scratch,
            token_count, chunk_start, kKHeads, kVHeads, kState,
            kGdnChunkSize, stream);
    }
    MIINFER_HIP_CHECK(hipEventRecord(ev_gdn, stream));

    // 7 & 8. SSM Post + SSM Out
    launch_m12_gdn_postprocess(
        ws.gdn_raw_output, ws.gate, d_ssm_norm_, ws.gated_output,
        token_count, kVHeads, kState, kRmsNormEpsilon, stream);
    launch_mx_q8_1_mmq_quantize(
        ws.gated_output, ws.mmq_q8, token_count, kInner, /*affine=*/true, stream);
    launch_mx_q5k_repacked_mmq(
        d_ssm_out_mmq_, ws.mmq_q8, ws.ssm_output,
        kHidden, kInner, token_count, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_ssm, stream));

    // 9. Residual + Post Norm
    launch_qwen3_fused_add_rms_norm_batch(
        d_input, ws.ssm_output, d_post_norm_,
        ws.residual, ws.post_normalized,
        token_count, kHidden, kRmsNormEpsilon, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_res, stream));

    // 10. FFN Gate & Up
    launch_mx_q8_1_mmq_quantize(
        ws.post_normalized, ws.mmq_q8, token_count, kHidden, /*affine=*/true, stream);
    launch_mx_q4k_repacked_mmq(
        d_ffn_gate_mmq_, ws.mmq_q8, ws.ffn_gate,
        kFfnInner, kHidden, token_count, stream);
    launch_mx_q4k_repacked_mmq(
        d_ffn_up_mmq_, ws.mmq_q8, ws.ffn_up,
        kFfnInner, kHidden, token_count, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_ffn_up, stream));

    // 11, 12, 13. SwiGLU + Down + Final Residual Add
    launch_qwen3_silu_mul(
        ws.ffn_gate, ws.ffn_up, ws.ffn_activation,
        token_count * kFfnInner, stream);
    if (ffn_down_type_ == GgufTensorType::q4_k) {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, token_count, kFfnInner, /*affine=*/true, stream);
        launch_mx_q4k_repacked_mmq(
            static_cast<const std::uint8_t*>(d_ffn_down_mmq_), ws.mmq_q8,
            ws.ffn_down, kHidden, kFfnInner, token_count, stream);
    } else {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, token_count, kFfnInner, /*affine=*/false, stream);
        launch_mx_q6k_repacked_mmq(
            static_cast<const std::uint8_t*>(d_ffn_down_mmq_), ws.mmq_q8,
            ws.ffn_down, kHidden, kFfnInner, token_count, stream);
    }
    launch_qwen3_add(
        ws.residual, ws.ffn_down, d_output,
        token_count * kHidden, stream);

    MIINFER_HIP_CHECK(hipEventRecord(ev_end, stream));
    MIINFER_HIP_CHECK(hipEventSynchronize(ev_end));

    float ms = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_norm)); timings.norm_beta_alpha_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_norm, ev_qkv)); timings.qkv_gate_proj_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_qkv, ev_conv)); timings.conv_l2_norm_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_conv, ev_gdn)); timings.gdn_chunkwise_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_gdn, ev_ssm)); timings.ssm_post_out_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_ssm, ev_res)); timings.residual_norm_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_res, ev_ffn_up)); timings.ffn_gate_up_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_ffn_up, ev_end)); timings.swiglu_down_residual_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_end)); timings.total_layer_ms = ms;

    (void)hipEventDestroy(ev_start);
    (void)hipEventDestroy(ev_norm);
    (void)hipEventDestroy(ev_qkv);
    (void)hipEventDestroy(ev_conv);
    (void)hipEventDestroy(ev_gdn);
    (void)hipEventDestroy(ev_ssm);
    (void)hipEventDestroy(ev_res);
    (void)hipEventDestroy(ev_ffn_up);
    (void)hipEventDestroy(ev_end);

    outgoing_state.position = base_position + token_count;
}

} // namespace miinfer::prefill_v2
