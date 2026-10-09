#include "miinfer/prefill_v2/attention_layer.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace miinfer::prefill_v2 {

namespace {

bool v2_0043_gqa_attention_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("MIINFER_V2_0043_GQA_ATTENTION");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool v2_0043_real_compare_enabled() {
    const char* value = std::getenv("MIINFER_V2_0043_COMPARE_REAL");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

void compare_iteration26_real_operands(
    const float* q, const float* candidate, const float* control,
    std::size_t elements, hipStream_t stream) {
    std::vector<float> host_q(elements), host_candidate(elements), host_control(elements);
    MIINFER_HIP_CHECK(hipMemcpyAsync(host_q.data(), q, elements * sizeof(float),
                                     hipMemcpyDeviceToHost, stream));
    MIINFER_HIP_CHECK(hipMemcpyAsync(host_candidate.data(), candidate, elements * sizeof(float),
                                     hipMemcpyDeviceToHost, stream));
    MIINFER_HIP_CHECK(hipMemcpyAsync(host_control.data(), control, elements * sizeof(float),
                                     hipMemcpyDeviceToHost, stream));
    MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

    float q_max = 0.0F, q_square_sum = 0.0F, q_round_max = 0.0F, q_expansion_max = 0.0F;
    float max_abs = 0.0F, max_rel = 0.0F, square_error_sum = 0.0F;
    std::size_t exact = 0;
    for (std::size_t i = 0; i < elements; ++i) {
        const float qv = host_q[i];
        q_max = std::max(q_max, std::fabs(qv));
        q_square_sum += qv * qv;
        q_round_max = std::max(q_round_max,
            std::fabs(qv - __half2float(__float2half(qv))));
        const __half q_hi = __float2half(qv);
        const __half q_lo = __float2half(qv - __half2float(q_hi));
        q_expansion_max = std::max(q_expansion_max, std::fabs(
            qv - (__half2float(q_hi) + __half2float(q_lo))));
        const float error = std::fabs(host_candidate[i] - host_control[i]);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::fabs(host_control[i]), 1.0e-6F));
        square_error_sum += error * error;
        exact += error == 0.0F;
    }
    const float inv_n = 1.0F / static_cast<float>(elements);
    std::cerr << "miinfer_iteration26_real_compare elements=" << elements
              << " q_max_abs=" << q_max
              << " q_rms=" << std::sqrt(q_square_sum * inv_n)
              << " q_fp16_round_max_abs=" << q_round_max
              << " q_two_half_expansion_max_abs=" << q_expansion_max
              << " attn_max_abs=" << max_abs
              << " attn_max_rel=" << max_rel
              << " attn_rms_error=" << std::sqrt(square_error_sum * inv_n)
              << " attn_exact_values=" << exact << '\n';
}

void upload_to_device(const void* host_ptr, void* device_ptr, std::size_t bytes) {
    MIINFER_HIP_CHECK(hipMemcpy(device_ptr, host_ptr, bytes, hipMemcpyHostToDevice));
}

void upload_tensor_f32(const miinfer::Qwen35TensorView& tensor, float*& device_ptr, std::size_t& byte_counter) {
    if (tensor.type() != miinfer::GgufTensorType::f32) {
        throw std::runtime_error("upload_tensor_f32: expected f32 tensor, got " +
                                 std::string(miinfer::gguf_tensor_type_name(tensor.type())));
    }
    const std::size_t bytes = tensor.bytes();
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_ptr), bytes));
    upload_to_device(tensor.data(), device_ptr, bytes);
    byte_counter += bytes;
}

} // namespace

PrefillV2AttentionLayer::PrefillV2AttentionLayer(
    const miinfer::Qwen35Model& model, std::size_t layer_index, bool mmq_gateup_only)
    : layer_index_(layer_index) {

    const auto name = [this](const char* suffix) {
        return "blk." + std::to_string(layer_index_) + "." + suffix;
    };

    // 1. Load Normalization and Scales (F32)
    upload_tensor_f32(model.tensor(name("attn_norm.weight")), d_attn_norm_, persistent_weight_bytes_);
    upload_tensor_f32(model.tensor(name("attn_q_norm.weight")), d_q_norm_, persistent_weight_bytes_);
    upload_tensor_f32(model.tensor(name("attn_k_norm.weight")), d_k_norm_, persistent_weight_bytes_);
    upload_tensor_f32(model.tensor(name("post_attention_norm.weight")), d_post_attention_norm_, persistent_weight_bytes_);

    // Helper for uploading packed weights
    const auto upload_packed = [this](const auto& packed, auto*& dev_ptr) {
        using Elem = typename std::decay_t<decltype(packed)>::value_type;
        const std::size_t bytes = packed.size() * sizeof(Elem);
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_ptr), bytes));
        upload_to_device(packed.data(), dev_ptr, bytes);
        persistent_weight_bytes_ += bytes;
    };

    // 2. Q Projection (Q4_K, shape [5120, 12288])
    const auto q_t = model.tensor(name("attn_q.weight"));
    if (q_t.type() != miinfer::GgufTensorType::q4_k) {
        throw std::runtime_error("PrefillV2AttentionLayer: attn_q must be Q4_K");
    }
    upload_packed(pack_mx_q4k_repacked_tensor(*q_t.source), d_q_mmq_);

    // 3. K Projection (Q4_K, shape [5120, 1024])
    const auto k_t = model.tensor(name("attn_k.weight"));
    if (k_t.type() != miinfer::GgufTensorType::q4_k) {
        throw std::runtime_error("PrefillV2AttentionLayer: attn_k must be Q4_K");
    }
    upload_packed(pack_mx_q4k_repacked_tensor(*k_t.source), d_k_mmq_);

    // 4. V Projection (Q4_K or Q6_K, shape [5120, 1024])
    const auto v_t = model.tensor(name("attn_v.weight"));
    if (v_t.type() == miinfer::GgufTensorType::q4_k) {
        v_is_q6_ = false;
        upload_packed(pack_mx_q4k_repacked_tensor(*v_t.source), d_v_mmq_);
    } else if (v_t.type() == miinfer::GgufTensorType::q6_k) {
        v_is_q6_ = true;
        upload_packed(pack_mx_q6k_repacked_tensor(*v_t.source), d_v_mmq_);
    } else {
        throw std::runtime_error("PrefillV2AttentionLayer: attn_v must be Q4_K or Q6_K");
    }

    // 5. O Projection (Q4_K, shape [6144, 5120])
    const auto o_t = model.tensor(name("attn_output.weight"));
    if (o_t.type() != miinfer::GgufTensorType::q4_k) {
        throw std::runtime_error("PrefillV2AttentionLayer: attn_output must be Q4_K");
    }
    upload_packed(pack_mx_q4k_repacked_tensor(*o_t.source), d_o_mmq_);

    // 6. FFN Gate & Up (Q4_K, shape [5120, 17408])
    const auto ffn_gate_t = model.tensor(name("ffn_gate.weight"));
    const auto ffn_up_t = model.tensor(name("ffn_up.weight"));
    if (ffn_gate_t.type() != miinfer::GgufTensorType::q4_k || ffn_up_t.type() != miinfer::GgufTensorType::q4_k) {
        throw std::runtime_error("PrefillV2AttentionLayer: ffn_gate and ffn_up must be Q4_K");
    }
    upload_packed(pack_mx_q4k_repacked_tensor(*ffn_gate_t.source), d_ffn_gate_mmq_);
    upload_packed(pack_mx_q4k_repacked_tensor(*ffn_up_t.source), d_ffn_up_mmq_);

    // 7. FFN Down (Q4_K or Q6_K, shape [17408, 5120])
    const auto ffn_down_t = model.tensor(name("ffn_down.weight"));
    if (ffn_down_t.type() == miinfer::GgufTensorType::q4_k) {
        ffn_down_is_q6_ = false;
        upload_packed(pack_mx_q4k_repacked_tensor(*ffn_down_t.source), d_ffn_down_mmq_);
    } else if (ffn_down_t.type() == miinfer::GgufTensorType::q6_k) {
        ffn_down_is_q6_ = true;
        upload_packed(pack_mx_q6k_repacked_tensor(*ffn_down_t.source), d_ffn_down_mmq_);
    } else {
        throw std::runtime_error("PrefillV2AttentionLayer: ffn_down must be Q4_K or Q6_K");
    }

    // 8. Gfx906 resident Wave decode weights
    // (a) Paired FFN SwiGLU fused layout
    if (!mmq_gateup_only) {
        const auto fused_swiglu_host = pack_q4k_wave_swiglu_fused(*ffn_gate_t.source, *ffn_up_t.source);
        upload_packed(fused_swiglu_host, d_ffn_swiglu_fused_);
    }
}

PrefillV2AttentionLayer::~PrefillV2AttentionLayer() {
    if (d_attn_norm_) (void)hipFree(d_attn_norm_);
    if (d_q_norm_) (void)hipFree(d_q_norm_);
    if (d_k_norm_) (void)hipFree(d_k_norm_);
    if (d_post_attention_norm_) (void)hipFree(d_post_attention_norm_);

    if (d_q_mmq_) (void)hipFree(d_q_mmq_);
    if (d_k_mmq_) (void)hipFree(d_k_mmq_);
    if (d_v_mmq_) (void)hipFree(d_v_mmq_);
    if (d_o_mmq_) (void)hipFree(d_o_mmq_);
    if (d_ffn_gate_mmq_) (void)hipFree(d_ffn_gate_mmq_);
    if (d_ffn_up_mmq_) (void)hipFree(d_ffn_up_mmq_);
    if (d_ffn_down_mmq_) (void)hipFree(d_ffn_down_mmq_);

    if (d_ffn_swiglu_fused_) (void)hipFree(d_ffn_swiglu_fused_);
}

PrefillV2AttentionLayer::PrefillV2AttentionLayer(PrefillV2AttentionLayer&& other) noexcept
    : layer_index_(other.layer_index_),
      persistent_weight_bytes_(other.persistent_weight_bytes_),
      v_is_q6_(other.v_is_q6_),
      ffn_down_is_q6_(other.ffn_down_is_q6_),
      d_attn_norm_(std::exchange(other.d_attn_norm_, nullptr)),
      d_q_norm_(std::exchange(other.d_q_norm_, nullptr)),
      d_k_norm_(std::exchange(other.d_k_norm_, nullptr)),
      d_post_attention_norm_(std::exchange(other.d_post_attention_norm_, nullptr)),
      d_q_mmq_(std::exchange(other.d_q_mmq_, nullptr)),
      d_k_mmq_(std::exchange(other.d_k_mmq_, nullptr)),
      d_v_mmq_(std::exchange(other.d_v_mmq_, nullptr)),
      d_o_mmq_(std::exchange(other.d_o_mmq_, nullptr)),
      d_ffn_gate_mmq_(std::exchange(other.d_ffn_gate_mmq_, nullptr)),
      d_ffn_up_mmq_(std::exchange(other.d_ffn_up_mmq_, nullptr)),
      d_ffn_down_mmq_(std::exchange(other.d_ffn_down_mmq_, nullptr)),
      d_ffn_swiglu_fused_(std::exchange(other.d_ffn_swiglu_fused_, nullptr)) {}

PrefillV2AttentionLayer& PrefillV2AttentionLayer::operator=(PrefillV2AttentionLayer&& other) noexcept {
    if (this != &other) {
        if (d_attn_norm_) (void)hipFree(d_attn_norm_);
        if (d_q_norm_) (void)hipFree(d_q_norm_);
        if (d_k_norm_) (void)hipFree(d_k_norm_);
        if (d_post_attention_norm_) (void)hipFree(d_post_attention_norm_);

        if (d_q_mmq_) (void)hipFree(d_q_mmq_);
        if (d_k_mmq_) (void)hipFree(d_k_mmq_);
        if (d_v_mmq_) (void)hipFree(d_v_mmq_);
        if (d_o_mmq_) (void)hipFree(d_o_mmq_);
        if (d_ffn_gate_mmq_) (void)hipFree(d_ffn_gate_mmq_);
        if (d_ffn_up_mmq_) (void)hipFree(d_ffn_up_mmq_);
        if (d_ffn_down_mmq_) (void)hipFree(d_ffn_down_mmq_);

        if (d_ffn_swiglu_fused_) (void)hipFree(d_ffn_swiglu_fused_);

        layer_index_ = other.layer_index_;
        persistent_weight_bytes_ = other.persistent_weight_bytes_;
        v_is_q6_ = other.v_is_q6_;
        ffn_down_is_q6_ = other.ffn_down_is_q6_;

        d_attn_norm_ = std::exchange(other.d_attn_norm_, nullptr);
        d_q_norm_ = std::exchange(other.d_q_norm_, nullptr);
        d_k_norm_ = std::exchange(other.d_k_norm_, nullptr);
        d_post_attention_norm_ = std::exchange(other.d_post_attention_norm_, nullptr);

        d_q_mmq_ = std::exchange(other.d_q_mmq_, nullptr);
        d_k_mmq_ = std::exchange(other.d_k_mmq_, nullptr);
        d_v_mmq_ = std::exchange(other.d_v_mmq_, nullptr);
        d_o_mmq_ = std::exchange(other.d_o_mmq_, nullptr);
        d_ffn_gate_mmq_ = std::exchange(other.d_ffn_gate_mmq_, nullptr);
        d_ffn_up_mmq_ = std::exchange(other.d_ffn_up_mmq_, nullptr);
        d_ffn_down_mmq_ = std::exchange(other.d_ffn_down_mmq_, nullptr);

        d_ffn_swiglu_fused_ = std::exchange(other.d_ffn_swiglu_fused_, nullptr);
    }
    return *this;
}

void PrefillV2AttentionLayer::forward(
    const float* d_input,
    float* d_output,
    PhysicalKvView kv_cache,
    const PrefillV2Workspace& ws,
    std::uint32_t base_position,
    std::uint32_t token_count,
    hipStream_t stream,
    const DevicePrefillState* prefill_state) const {

    if (token_count == 0 || token_count > kMaxPrefillBatch) {
        throw std::runtime_error("PrefillV2AttentionLayer::forward: invalid token_count " + std::to_string(token_count));
    }

    // 1. Input Normalization
    launch_qwen3_rms_norm_batch(
        d_input, d_attn_norm_, ws.normalized, token_count, kHidden, kRmsNormEpsilon, stream);

    // 2. QKV Projections (Mx compact MMQ)
    launch_mx_q8_1_mmq_quantize(
        ws.normalized, ws.mmq_q8, token_count, kHidden, true, stream);

    // Q + Gate Projection [5120 -> 12288]
    launch_mx_q4k_repacked_mmq(
        d_q_mmq_, ws.mmq_q8, ws.attn_qfull, 12288, kHidden, token_count, stream);

    // K Projection [5120 -> 1024]
    launch_mx_q4k_repacked_mmq(
        d_k_mmq_, ws.mmq_q8, ws.attn_k, 1024, kHidden, token_count, stream);

    // V Projection [5120 -> 1024]
    if (v_is_q6_) {
        launch_mx_q8_1_mmq_quantize(
            ws.normalized, ws.mmq_q8, token_count, kHidden, false, stream);
        launch_mx_q6k_repacked_mmq(
            d_v_mmq_, ws.mmq_q8, ws.attn_v, 1024, kHidden, token_count, stream);
    } else {
        launch_mx_q4k_repacked_mmq(
            d_v_mmq_, ws.mmq_q8, ws.attn_v, 1024, kHidden, token_count, stream);
    }

    // 3. Q Split, RMSNorm, and RoPE
    launch_qwen35_decoupled_q_split_norm_rope_batch(
        ws.attn_qfull, d_q_norm_, ws.attn_q_rope, ws.gate,
        token_count, base_position, 24, 256, kRopeTheta, kRmsNormEpsilon, stream, prefill_state);

    // 4. K RMSNorm, RoPE, and Store K + V into KV Cache (FP16 or Q8)
    launch_qwen35_decoupled_k_norm_rope_kv_store_batch_quant(
        ws.attn_k, ws.attn_v, d_k_norm_,
        kv_cache.key_cache, kv_cache.value_cache,
        kv_cache.key_cache_q8, kv_cache.key_scales,
        kv_cache.value_cache_q8, kv_cache.value_scales,
        token_count, base_position, static_cast<std::uint32_t>(kv_cache.capacity),
        4, 256, kRopeTheta, kRmsNormEpsilon,
        kv_cache.is_k_q8(), kv_cache.is_v_q8(),
        stream, prefill_state);

    // 5. Tiled Online Causal Attention with Sigmoid Gating.
    if (v2_0043_gqa_attention_enabled() && base_position > 0 && prefill_state == nullptr
        && !kv_cache.is_k_q8() && !kv_cache.is_v_q8()
        && token_count >= 16 && token_count % 16 == 0
        && ws.splitk_attn_workspace != nullptr) {
        const auto launch_iteration26 = [&](float* attention_output) {
            launch_qwen35_kq_fragment_reuse_attention_batch_f16(
                ws.attn_q_rope, kv_cache.key_cache, kv_cache.value_cache,
                ws.gate, attention_output, ws.splitk_attn_workspace,
                token_count, base_position, static_cast<std::uint32_t>(kv_cache.capacity),
                24, 4, 256, 1.0F / std::sqrt(256.0F), 3, stream);
        };
        launch_iteration26(ws.attn_gated_output);
        if (v2_0043_real_compare_enabled() && layer_index_ == 3) {
            std::cerr << "miinfer_v2_0043_candidate_call base_position=" << base_position
                      << " token_count=" << token_count << '\n';
        }
        if (v2_0043_real_compare_enabled() && layer_index_ == 3
            && base_position == 512 && token_count == 512) {
            launch_iteration26(ws.attn_qfull);
            compare_iteration26_real_operands(
                ws.attn_q_rope, ws.attn_gated_output, ws.attn_qfull,
                static_cast<std::size_t>(token_count) * 24 * 256, stream);
        }
    } else if (((token_count <= 512 && base_position > 0) || prefill_state != nullptr)
        && ws.splitk_attn_workspace != nullptr) {
        launch_qwen35_splitk_suffix_attention_quant(
            ws.attn_q_rope,
            kv_cache.key_cache, kv_cache.value_cache,
            kv_cache.key_cache_q8, kv_cache.key_scales,
            kv_cache.value_cache_q8, kv_cache.value_scales,
            ws.gate, ws.attn_gated_output,
            ws.splitk_attn_workspace, token_count, base_position,
            static_cast<std::uint32_t>(kv_cache.capacity),
            24, 4, 256, 1.0F / std::sqrt(256.0F),
            kv_cache.is_k_q8(), kv_cache.is_v_q8(),
            ws.splitk_splits, stream, prefill_state);
    } else {
        launch_qwen35_tiled_online_attention_batch_f16(
            ws.attn_q_rope, kv_cache.key_cache, kv_cache.value_cache, ws.gate, ws.attn_gated_output,
            token_count, base_position, static_cast<std::uint32_t>(kv_cache.capacity),
            24, 4, 256, 1.0F / std::sqrt(256.0F), stream);
    }

    // 6. Attention Output Projection (O: [6144 -> 5120])
    launch_mx_q8_1_mmq_quantize(
        ws.attn_gated_output, ws.mmq_q8, token_count, kInner, true, stream);
    launch_mx_q4k_repacked_mmq(
        d_o_mmq_, ws.mmq_q8, ws.projected, kHidden, kInner, token_count, stream);

    // 7. Residual Connection + Post-Attention RMSNorm
    launch_qwen3_fused_add_rms_norm_batch(
        d_input, ws.projected, d_post_attention_norm_, ws.residual, ws.post_normalized,
        token_count, kHidden, kRmsNormEpsilon, stream);

    // 8. FFN Gate and Up Projections (Q4_K MMQ)
    launch_mx_q8_1_mmq_quantize(
        ws.post_normalized, ws.mmq_q8, token_count, kHidden, true, stream);
    launch_mx_q4k_repacked_mmq(
        d_ffn_gate_mmq_, ws.mmq_q8, ws.ffn_gate, kFfnInner, kHidden, token_count, stream);
    launch_mx_q4k_repacked_mmq(
        d_ffn_up_mmq_, ws.mmq_q8, ws.ffn_up, kFfnInner, kHidden, token_count, stream);

    // 9. SwiGLU Activation
    launch_qwen3_silu_mul(
        ws.ffn_gate, ws.ffn_up, ws.ffn_activation, token_count * kFfnInner, stream);

    // 10. FFN Down Projection & Final Residual
    if (ffn_down_is_q6_) {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, token_count, kFfnInner, false, stream);
        launch_mx_q6k_repacked_mmq(
            d_ffn_down_mmq_, ws.mmq_q8, ws.ffn_down, kHidden, kFfnInner, token_count, stream);
    } else {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, token_count, kFfnInner, true, stream);
        launch_mx_q4k_repacked_mmq(
            d_ffn_down_mmq_, ws.mmq_q8, ws.ffn_down, kHidden, kFfnInner, token_count, stream);
    }

    launch_qwen3_add(
        ws.residual, ws.ffn_down, d_output, token_count * kHidden, stream);
}

void PrefillV2AttentionLayer::forward_profiled(
    const float* d_input,
    float* d_output,
    PhysicalKvView kv_cache,
    const PrefillV2Workspace& ws,
    std::uint32_t base_position,
    std::uint32_t token_count,
    AttentionLayerProfileBreakdown& breakdown,
    hipStream_t stream) const {

    hipEvent_t ev_start, ev_norm, ev_qkv, ev_rope, ev_attn, ev_o, ev_post, ev_ffn_gu, ev_ffn_down;
    MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_norm));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_qkv));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_rope));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_attn));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_o));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_post));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_ffn_gu));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_ffn_down));

    MIINFER_HIP_CHECK(hipEventRecord(ev_start, stream));

    // 1. Input Normalization
    launch_qwen3_rms_norm_batch(
        d_input, d_attn_norm_, ws.normalized, token_count, kHidden, kRmsNormEpsilon, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_norm, stream));

    // 2. QKV Projections
    launch_mx_q8_1_mmq_quantize(
        ws.normalized, ws.mmq_q8, token_count, kHidden, true, stream);
    launch_mx_q4k_repacked_mmq(
        d_q_mmq_, ws.mmq_q8, ws.attn_qfull, 12288, kHidden, token_count, stream);
    launch_mx_q4k_repacked_mmq(
        d_k_mmq_, ws.mmq_q8, ws.attn_k, 1024, kHidden, token_count, stream);
    if (v_is_q6_) {
        launch_mx_q8_1_mmq_quantize(
            ws.normalized, ws.mmq_q8, token_count, kHidden, false, stream);
        launch_mx_q6k_repacked_mmq(
            d_v_mmq_, ws.mmq_q8, ws.attn_v, 1024, kHidden, token_count, stream);
    } else {
        launch_mx_q4k_repacked_mmq(
            d_v_mmq_, ws.mmq_q8, ws.attn_v, 1024, kHidden, token_count, stream);
    }
    MIINFER_HIP_CHECK(hipEventRecord(ev_qkv, stream));

    // 3 & 4. RoPE & KV Store
    launch_qwen35_decoupled_q_split_norm_rope_batch(
        ws.attn_qfull, d_q_norm_, ws.attn_q_rope, ws.gate,
        token_count, base_position, 24, 256, kRopeTheta, kRmsNormEpsilon, stream);
    launch_qwen35_decoupled_k_norm_rope_kv_store_batch_quant(
        ws.attn_k, ws.attn_v, d_k_norm_,
        kv_cache.key_cache, kv_cache.value_cache,
        kv_cache.key_cache_q8, kv_cache.key_scales,
        kv_cache.value_cache_q8, kv_cache.value_scales,
        token_count, base_position, static_cast<std::uint32_t>(kv_cache.capacity),
        4, 256, kRopeTheta, kRmsNormEpsilon,
        kv_cache.is_k_q8(), kv_cache.is_v_q8(),
        stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_rope, stream));

    // 5. Causal Attention (Split-K specialized for suffix prefill)
    if (v2_0043_gqa_attention_enabled() && base_position > 0
        && !kv_cache.is_k_q8() && !kv_cache.is_v_q8()
        && token_count >= 16 && token_count % 16 == 0
        && ws.splitk_attn_workspace != nullptr) {
        launch_qwen35_kq_fragment_reuse_attention_batch_f16(
            ws.attn_q_rope, kv_cache.key_cache, kv_cache.value_cache,
            ws.gate, ws.attn_gated_output, ws.splitk_attn_workspace,
            token_count, base_position, static_cast<std::uint32_t>(kv_cache.capacity),
            24, 4, 256, 1.0F / std::sqrt(256.0F), 3, stream);
    } else if (token_count <= 512 && base_position > 0 && ws.splitk_attn_workspace != nullptr) {
        launch_qwen35_splitk_suffix_attention_quant(
            ws.attn_q_rope,
            kv_cache.key_cache, kv_cache.value_cache,
            kv_cache.key_cache_q8, kv_cache.key_scales,
            kv_cache.value_cache_q8, kv_cache.value_scales,
            ws.gate, ws.attn_gated_output,
            ws.splitk_attn_workspace, token_count, base_position,
            static_cast<std::uint32_t>(kv_cache.capacity),
            24, 4, 256, 1.0F / std::sqrt(256.0F),
            kv_cache.is_k_q8(), kv_cache.is_v_q8(),
            ws.splitk_splits, stream);
    } else {
        launch_qwen35_tiled_online_attention_batch_f16(
            ws.attn_q_rope, kv_cache.key_cache, kv_cache.value_cache, ws.gate, ws.attn_gated_output,
            token_count, base_position, static_cast<std::uint32_t>(kv_cache.capacity),
            24, 4, 256, 1.0F / std::sqrt(256.0F), stream);
    }
    MIINFER_HIP_CHECK(hipEventRecord(ev_attn, stream));

    // 6. O Projection
    launch_mx_q8_1_mmq_quantize(
        ws.attn_gated_output, ws.mmq_q8, token_count, kInner, true, stream);
    launch_mx_q4k_repacked_mmq(
        d_o_mmq_, ws.mmq_q8, ws.projected, kHidden, kInner, token_count, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_o, stream));

    // 7. Residual + Post Norm
    launch_qwen3_fused_add_rms_norm_batch(
        d_input, ws.projected, d_post_attention_norm_, ws.residual, ws.post_normalized,
        token_count, kHidden, kRmsNormEpsilon, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_post, stream));

    // 8. FFN Gate/Up
    launch_mx_q8_1_mmq_quantize(
        ws.post_normalized, ws.mmq_q8, token_count, kHidden, true, stream);
    launch_mx_q4k_repacked_mmq(
        d_ffn_gate_mmq_, ws.mmq_q8, ws.ffn_gate, kFfnInner, kHidden, token_count, stream);
    launch_mx_q4k_repacked_mmq(
        d_ffn_up_mmq_, ws.mmq_q8, ws.ffn_up, kFfnInner, kHidden, token_count, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_ffn_gu, stream));

    // 9 & 10. SwiGLU, Down & Residual Add
    launch_qwen3_silu_mul(
        ws.ffn_gate, ws.ffn_up, ws.ffn_activation, token_count * kFfnInner, stream);
    if (ffn_down_is_q6_) {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, token_count, kFfnInner, false, stream);
        launch_mx_q6k_repacked_mmq(
            d_ffn_down_mmq_, ws.mmq_q8, ws.ffn_down, kHidden, kFfnInner, token_count, stream);
    } else {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, token_count, kFfnInner, true, stream);
        launch_mx_q4k_repacked_mmq(
            d_ffn_down_mmq_, ws.mmq_q8, ws.ffn_down, kHidden, kFfnInner, token_count, stream);
    }
    launch_qwen3_add(
        ws.residual, ws.ffn_down, d_output, token_count * kHidden, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_ffn_down, stream));

    MIINFER_HIP_CHECK(hipEventSynchronize(ev_ffn_down));

    const auto elapsed = [](hipEvent_t a, hipEvent_t b) {
        float ms = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
        return static_cast<double>(ms);
    };

    breakdown.norm_ms = elapsed(ev_start, ev_norm);
    breakdown.qkv_proj_ms = elapsed(ev_norm, ev_qkv);
    breakdown.rope_kv_store_ms = elapsed(ev_qkv, ev_rope);
    breakdown.causal_attn_ms = elapsed(ev_rope, ev_attn);
    breakdown.o_proj_ms = elapsed(ev_attn, ev_o);
    breakdown.post_norm_ms = elapsed(ev_o, ev_post);
    breakdown.ffn_gate_up_ms = elapsed(ev_post, ev_ffn_gu);
    breakdown.swiglu_down_res_ms = elapsed(ev_ffn_gu, ev_ffn_down);
    breakdown.total_layer_ms = elapsed(ev_start, ev_ffn_down);

    (void)hipEventDestroy(ev_start);
    (void)hipEventDestroy(ev_norm);
    (void)hipEventDestroy(ev_qkv);
    (void)hipEventDestroy(ev_rope);
    (void)hipEventDestroy(ev_attn);
    (void)hipEventDestroy(ev_o);
    (void)hipEventDestroy(ev_post);
    (void)hipEventDestroy(ev_ffn_gu);
    (void)hipEventDestroy(ev_ffn_down);
}

void PrefillV2AttentionLayer::decode(
    const float* d_input,
    float* d_output,
    PhysicalKvView kv_cache,
    const PrefillV2Workspace& ws,
    std::uint32_t position,
    const DeviceDecodeState* decode_state,
    hipStream_t stream) const {

    // 1. Input Normalization (single token)
    launch_qwen3_rms_norm(
        d_input, d_attn_norm_, ws.normalized, kHidden, kRmsNormEpsilon, stream);

    // 2. QKV Projections via compact Mx MMQ (N=1)
    launch_mx_q8_1_mmq_quantize(
        ws.normalized, ws.mmq_q8, 1, kHidden, true, stream);

    // Q + Gate Projection [5120 -> 12288] (Q4_K)
    launch_mx_q4k_repacked_mmq(
        d_q_mmq_, ws.mmq_q8, ws.attn_qfull, 12288, kHidden, 1, stream);

    // K Projection [5120 -> 1024] (Q4_K)
    launch_mx_q4k_repacked_mmq(
        d_k_mmq_, ws.mmq_q8, ws.attn_k, 1024, kHidden, 1, stream);

    // V Projection [5120 -> 1024] (Q4_K or Q6_K)
    if (v_is_q6_) {
        launch_mx_q8_1_mmq_quantize(
            ws.normalized, ws.mmq_q8, 1, kHidden, false, stream);
        launch_mx_q6k_repacked_mmq(
            d_v_mmq_, ws.mmq_q8, ws.attn_v, 1024, kHidden, 1, stream);
    } else {
        launch_mx_q4k_repacked_mmq(
            d_v_mmq_, ws.mmq_q8, ws.attn_v, 1024, kHidden, 1, stream);
    }

    // 3. Q Split, RMSNorm, and RoPE
    if (decode_state != nullptr) {
        launch_qwen35_fused_q_split_norm_rope_dynamic(
            ws.attn_qfull, d_q_norm_, ws.attn_q_rope, ws.gate,
            24, 256, decode_state, kRopeTheta, kRmsNormEpsilon, stream);
    } else {
        launch_qwen35_decoupled_q_split_norm_rope_batch(
            ws.attn_qfull, d_q_norm_, ws.attn_q_rope, ws.gate,
            1, position, 24, 256, kRopeTheta, kRmsNormEpsilon, stream);
    }

    // 4. K RMSNorm, RoPE, and Store K + V into KV Cache (FP16 or Q8)
    launch_qwen35_decoupled_k_norm_rope_kv_store_batch_quant(
        ws.attn_k, ws.attn_v, d_k_norm_,
        kv_cache.key_cache, kv_cache.value_cache,
        kv_cache.key_cache_q8, kv_cache.key_scales,
        kv_cache.value_cache_q8, kv_cache.value_scales,
        1, position, static_cast<std::uint32_t>(kv_cache.capacity),
        4, 256, kRopeTheta, kRmsNormEpsilon,
        kv_cache.is_k_q8(), kv_cache.is_v_q8(),
        stream,
        /*prefill_state=*/nullptr,
        decode_state);

    // 5. High-Occupancy Split-K Decode Attention with In-Register Sigmoid Gating
    if (decode_state != nullptr) {
        launch_qwen35_tiled_online_attention_quant_dynamic(
            ws.attn_q_rope,
            kv_cache.key_cache,
            kv_cache.value_cache,
            kv_cache.key_cache_q8,
            kv_cache.key_scales,
            kv_cache.value_cache_q8,
            kv_cache.value_scales,
            decode_state,
            static_cast<std::uint32_t>(kv_cache.capacity),
            /*output=*/nullptr,
            ws.gate,
            ws.attn_gated_output,
            24,
            4,
            256,
            1.0F / std::sqrt(256.0F),
            kv_cache.is_k_q8(),
            kv_cache.is_v_q8(),
            stream);
    } else {
        launch_qwen35_splitk_suffix_attention_quant(
            ws.attn_q_rope,
            kv_cache.key_cache,
            kv_cache.value_cache,
            kv_cache.key_cache_q8,
            kv_cache.key_scales,
            kv_cache.value_cache_q8,
            kv_cache.value_scales,
            ws.gate,
            ws.attn_gated_output,
            ws.splitk_attn_workspace,
            1,
            // The kernel adds token + 1 to base_position for the inclusive KV length.
            position,
            static_cast<std::uint32_t>(kv_cache.capacity),
            24,
            4,
            256,
            1.0F / std::sqrt(256.0F),
            kv_cache.is_k_q8(),
            kv_cache.is_v_q8(),
            ws.splitk_splits,
            stream);
    }

    // 6. Attention Output Projection via compact Mx MMQ (O: [6144 -> 5120])
    launch_mx_q8_1_mmq_quantize(
        ws.attn_gated_output, ws.mmq_q8, 1, kInner, true, stream);
    launch_mx_q4k_repacked_mmq(
        d_o_mmq_, ws.mmq_q8, ws.projected, kHidden, kInner, 1, stream);

    // 7. Residual Connection + Post-Attention RMSNorm (Fused with Q8_1 quantization for FFN Paired Wave)
    launch_qwen3_fused_add_rms_norm(
        d_input, ws.projected, d_post_attention_norm_, ws.residual, ws.post_normalized,
        kHidden, kRmsNormEpsilon, stream, ws.q8_1);

    // 8. FFN Gate & Up Projections + SwiGLU Activation (Fused into single resident kernel pass)
    if (d_ffn_swiglu_fused_) {
        launch_q4k_wave_fused_gate_up_swiglu_paired(
            d_ffn_swiglu_fused_, ws.q8_1, ws.ffn_activation,
            kFfnInner, kHidden, stream);
    } else {
        launch_mx_q8_1_mmq_quantize(
            ws.post_normalized, ws.mmq_q8, 1, kHidden, /*affine=*/true, stream);
        launch_mx_q4k_repacked_mmq(
            d_ffn_gate_mmq_, ws.mmq_q8, ws.ffn_gate, kFfnInner, kHidden, 1, stream);
        launch_mx_q4k_repacked_mmq(
            d_ffn_up_mmq_, ws.mmq_q8, ws.ffn_up, kFfnInner, kHidden, 1, stream);
        launch_qwen3_silu_mul(ws.ffn_gate, ws.ffn_up, ws.ffn_activation, kFfnInner, stream);
    }

    // 9. FFN Down Projection & Final Residual
    if (ffn_down_is_q6_) {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, 1, kFfnInner, /*affine=*/false, stream);
        launch_mx_q6k_repacked_mmq(
            d_ffn_down_mmq_, ws.mmq_q8, ws.ffn_down, kHidden, kFfnInner, 1, stream);
    } else {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, 1, kFfnInner, /*affine=*/true, stream);
        launch_mx_q4k_repacked_mmq(
            d_ffn_down_mmq_, ws.mmq_q8, ws.ffn_down, kHidden, kFfnInner, 1, stream);
    }

    launch_qwen3_add(
        ws.residual, ws.ffn_down, d_output, kHidden, stream);
}

void PrefillV2AttentionLayer::decode_profiled(
    const float* d_input,
    float* d_output,
    PhysicalKvView kv_cache,
    const PrefillV2Workspace& ws,
    std::uint32_t position,
    const DeviceDecodeState* decode_state,
    AttentionLayerDecodePhaseTimings& timings,
    hipStream_t stream) const {

    hipEvent_t ev_start, ev_norm, ev_qkv, ev_qk_rope, ev_splitk, ev_o, ev_res, ev_ffn_up, ev_end;
    MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_norm));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_qkv));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_qk_rope));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_splitk));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_o));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_res));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_ffn_up));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_end));

    MIINFER_HIP_CHECK(hipEventRecord(ev_start, stream));

    // 1. Input RMS Normalization
    launch_qwen3_rms_norm(
        d_input, d_attn_norm_, ws.normalized, kHidden, kRmsNormEpsilon, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_norm, stream));

    // 2. Q, K, V Projections via compact Mx MMQ (N=1)
    launch_mx_q8_1_mmq_quantize(
        ws.normalized, ws.mmq_q8, 1, kHidden, true, stream);
    launch_mx_q4k_repacked_mmq(
        d_q_mmq_, ws.mmq_q8, ws.attn_qfull, 12288, kHidden, 1, stream);
    launch_mx_q4k_repacked_mmq(
        d_k_mmq_, ws.mmq_q8, ws.attn_k, 1024, kHidden, 1, stream);
    if (v_is_q6_) {
        launch_mx_q8_1_mmq_quantize(
            ws.normalized, ws.mmq_q8, 1, kHidden, false, stream);
        launch_mx_q6k_repacked_mmq(
            d_v_mmq_, ws.mmq_q8, ws.attn_v, 1024, kHidden, 1, stream);
    } else {
        launch_mx_q4k_repacked_mmq(
            d_v_mmq_, ws.mmq_q8, ws.attn_v, 1024, kHidden, 1, stream);
    }
    MIINFER_HIP_CHECK(hipEventRecord(ev_qkv, stream));

    // 3. Q RMSNorm, RoPE, and Split (Q [6144] + Gate [6144])
    if (decode_state != nullptr) {
        launch_qwen35_fused_q_split_norm_rope_dynamic(
            ws.attn_qfull, d_q_norm_, ws.attn_q_rope, ws.gate,
            24, 256, decode_state, kRopeTheta, kRmsNormEpsilon, stream);
    } else {
        launch_qwen35_decoupled_q_split_norm_rope_batch(
            ws.attn_qfull, d_q_norm_, ws.attn_q_rope, ws.gate,
            1, position, 24, 256, kRopeTheta, kRmsNormEpsilon, stream);
    }

    // 4. K RMSNorm, RoPE, and Store K + V into KV Cache (FP16 or Q8)
    launch_qwen35_decoupled_k_norm_rope_kv_store_batch_quant(
        ws.attn_k, ws.attn_v, d_k_norm_,
        kv_cache.key_cache, kv_cache.value_cache,
        kv_cache.key_cache_q8, kv_cache.key_scales,
        kv_cache.value_cache_q8, kv_cache.value_scales,
        1, position, static_cast<std::uint32_t>(kv_cache.capacity),
        4, 256, kRopeTheta, kRmsNormEpsilon,
        kv_cache.is_k_q8(), kv_cache.is_v_q8(),
        stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_qk_rope, stream));

    // 5. High-Occupancy Split-K Decode Attention with In-Register Sigmoid Gating
    if (decode_state != nullptr) {
        launch_qwen35_tiled_online_attention_quant_dynamic(
            ws.attn_q_rope,
            kv_cache.key_cache,
            kv_cache.value_cache,
            kv_cache.key_cache_q8,
            kv_cache.key_scales,
            kv_cache.value_cache_q8,
            kv_cache.value_scales,
            decode_state,
            static_cast<std::uint32_t>(kv_cache.capacity),
            /*output=*/nullptr,
            ws.gate,
            ws.attn_gated_output,
            24,
            4,
            256,
            1.0F / std::sqrt(256.0F),
            kv_cache.is_k_q8(),
            kv_cache.is_v_q8(),
            stream);
    } else {
        launch_qwen35_splitk_suffix_attention_quant(
            ws.attn_q_rope,
            kv_cache.key_cache,
            kv_cache.value_cache,
            kv_cache.key_cache_q8,
            kv_cache.key_scales,
            kv_cache.value_cache_q8,
            kv_cache.value_scales,
            ws.gate,
            ws.attn_gated_output,
            ws.splitk_attn_workspace,
            1,
            position,
            static_cast<std::uint32_t>(kv_cache.capacity),
            24,
            4,
            256,
            1.0F / std::sqrt(256.0F),
            kv_cache.is_k_q8(),
            kv_cache.is_v_q8(),
            ws.splitk_splits,
            stream);
    }
    MIINFER_HIP_CHECK(hipEventRecord(ev_splitk, stream));

    // 6. O Projection via compact Mx MMQ (N=1)
    launch_mx_q8_1_mmq_quantize(
        ws.attn_gated_output, ws.mmq_q8, 1, kInner, true, stream);
    launch_mx_q4k_repacked_mmq(
        d_o_mmq_, ws.mmq_q8, ws.projected, kHidden, kInner, 1, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_o, stream));

    // 7. Residual + Post-Attention RMSNorm (Fused with Q8_1)
    launch_qwen3_fused_add_rms_norm(
        d_input, ws.projected, d_post_attention_norm_, ws.residual, ws.post_normalized,
        kHidden, kRmsNormEpsilon, stream, ws.q8_1);
    MIINFER_HIP_CHECK(hipEventRecord(ev_res, stream));

    // 8. FFN Gate & Up Projections + SwiGLU
    if (d_ffn_swiglu_fused_) {
        launch_q4k_wave_fused_gate_up_swiglu_paired(
            d_ffn_swiglu_fused_, ws.q8_1, ws.ffn_activation,
            kFfnInner, kHidden, stream);
    } else {
        launch_mx_q8_1_mmq_quantize(
            ws.post_normalized, ws.mmq_q8, 1, kHidden, /*affine=*/true, stream);
        launch_mx_q4k_repacked_mmq(
            d_ffn_gate_mmq_, ws.mmq_q8, ws.ffn_gate, kFfnInner, kHidden, 1, stream);
        launch_mx_q4k_repacked_mmq(
            d_ffn_up_mmq_, ws.mmq_q8, ws.ffn_up, kFfnInner, kHidden, 1, stream);
        launch_qwen3_silu_mul(ws.ffn_gate, ws.ffn_up, ws.ffn_activation, kFfnInner, stream);
    }
    MIINFER_HIP_CHECK(hipEventRecord(ev_ffn_up, stream));

    // 9. FFN Down Projection & Final Residual
    if (ffn_down_is_q6_) {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, 1, kFfnInner, /*affine=*/false, stream);
        launch_mx_q6k_repacked_mmq(
            d_ffn_down_mmq_, ws.mmq_q8, ws.ffn_down, kHidden, kFfnInner, 1, stream);
    } else {
        launch_mx_q8_1_mmq_quantize(
            ws.ffn_activation, ws.mmq_q8, 1, kFfnInner, /*affine=*/true, stream);
        launch_mx_q4k_repacked_mmq(
            d_ffn_down_mmq_, ws.mmq_q8, ws.ffn_down, kHidden, kFfnInner, 1, stream);
    }
    launch_qwen3_add(
        ws.residual, ws.ffn_down, d_output, kHidden, stream);

    MIINFER_HIP_CHECK(hipEventRecord(ev_end, stream));
    MIINFER_HIP_CHECK(hipEventSynchronize(ev_end));

    float ms = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_norm)); timings.norm_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_norm, ev_qkv)); timings.qkv_proj_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_qkv, ev_qk_rope)); timings.qk_rope_kv_store_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_qk_rope, ev_splitk)); timings.splitk_attention_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_splitk, ev_o)); timings.o_proj_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_o, ev_res)); timings.residual_norm_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_res, ev_ffn_up)); timings.ffn_gate_up_swiglu_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_ffn_up, ev_end)); timings.ffn_down_residual_ms = ms;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_end)); timings.total_layer_ms = ms;

    (void)hipEventDestroy(ev_start);
    (void)hipEventDestroy(ev_norm);
    (void)hipEventDestroy(ev_qkv);
    (void)hipEventDestroy(ev_qk_rope);
    (void)hipEventDestroy(ev_splitk);
    (void)hipEventDestroy(ev_o);
    (void)hipEventDestroy(ev_res);
    (void)hipEventDestroy(ev_ffn_up);
    (void)hipEventDestroy(ev_end);
}

} // namespace miinfer::prefill_v2
