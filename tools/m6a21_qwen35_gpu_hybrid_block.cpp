#define MIINFER_M6A3_HELPERS_ONLY
#include "m6a3_qwen35_layer.cpp"

#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kCacheCapacity = 128;

class DeviceBytes {
public:
    explicit DeviceBytes(std::size_t bytes) : bytes_(bytes) {
        MIINFER_HIP_CHECK(hipMalloc(&data_, bytes));
    }
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
    ~DeviceBytes() { if (data_ != nullptr) (void)hipFree(data_); }
    void* get() const noexcept { return data_; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

using Buffer = std::unique_ptr<DeviceBytes>;

Buffer allocate(std::size_t bytes) { return std::make_unique<DeviceBytes>(bytes); }

void upload(const void* source, void* destination, std::size_t bytes) {
    MIINFER_HIP_CHECK(hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice));
}

void upload_tensor(const miinfer::GgufTensor& source, const Buffer& destination) {
    upload(source.data, destination->get(), source.byte_size);
}

struct DetailedError {
    float max_abs = 0.0F;
    float mean_abs = 0.0F;
    float rms = 0.0F;
    float reference_rms = 0.0F;
    float relative_rms = 0.0F;
};

DetailedError detailed_compare(std::span<const float> actual,
                               std::span<const float> expected) {
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::runtime_error("comparison size mismatch");
    }
    double abs_sum = 0.0;
    double error_sum = 0.0;
    double reference_sum = 0.0;
    DetailedError result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error("non-finite comparison value");
        }
        const double delta = static_cast<double>(actual[i]) - expected[i];
        result.max_abs = std::max(result.max_abs, static_cast<float>(std::fabs(delta)));
        abs_sum += std::fabs(delta);
        error_sum += delta * delta;
        reference_sum += static_cast<double>(expected[i]) * expected[i];
    }
    result.mean_abs = static_cast<float>(abs_sum / actual.size());
    result.rms = static_cast<float>(std::sqrt(error_sum / actual.size()));
    result.reference_rms = static_cast<float>(std::sqrt(reference_sum / actual.size()));
    result.relative_rms = result.reference_rms == 0.0F
        ? result.rms : result.rms / result.reference_rms;
    return result;
}

DetailedError detailed_device_error(const float* device, std::size_t elements,
                                    const std::filesystem::path& expected_path) {
    std::vector<float> actual(elements);
    MIINFER_HIP_CHECK(hipMemcpy(actual.data(), device, elements * sizeof(float),
                                hipMemcpyDeviceToHost));
    return detailed_compare(actual, read_f32(expected_path, elements));
}

std::uint64_t fingerprint(const void* device, std::size_t bytes) {
    std::vector<std::byte> host(bytes);
    MIINFER_HIP_CHECK(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost));
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto byte : host) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void require_type(const miinfer::GgufTensor& tensor_value,
                 std::initializer_list<miinfer::GgufTensorType> types) {
    if (std::find(types.begin(), types.end(), tensor_value.type) == types.end()) {
        throw std::runtime_error("unsupported tensor type for " + tensor_value.name);
    }
}

void project(const miinfer::GgufTensor& weight, const Buffer& device_weight,
             const float* input, miinfer::Q8KDeviceBlock* q8, float* output,
             std::uint32_t rows, std::uint32_t columns) {
    miinfer::launch_qwen3_q8_k_quantize(input, q8, columns);
    switch (weight.type) {
    case miinfer::GgufTensorType::q4_k:
        miinfer::launch_qwen3_q4_k_q8_k_gemv(
            static_cast<const miinfer::Q4KDeviceBlock*>(device_weight->get()), q8,
            output, rows, columns);
        return;
    case miinfer::GgufTensorType::q5_k:
        miinfer::launch_qwen3_q5_k_q8_k_gemv(
            static_cast<const miinfer::Q5KDeviceBlock*>(device_weight->get()), q8,
            output, rows, columns);
        return;
    case miinfer::GgufTensorType::q6_k:
        miinfer::launch_qwen3_q6_k_q8_k_gemv(
            static_cast<const miinfer::Q6KDeviceBlock*>(device_weight->get()), q8,
            output, rows, columns);
        return;
    default:
        throw std::runtime_error("unsupported quantized projection: " + weight.name);
    }
}

struct RecurrentLayer {
    std::size_t index;
    const miinfer::Qwen35Model& model;

    const miinfer::GgufTensor& attn_norm;
    const miinfer::GgufTensor& qkv_weight;
    const miinfer::GgufTensor& gate_weight;
    const miinfer::GgufTensor& beta_weight;
    const miinfer::GgufTensor& alpha_weight;
    const miinfer::GgufTensor& dt_weight;
    const miinfer::GgufTensor& a_weight;
    const miinfer::GgufTensor& conv_weight;
    const miinfer::GgufTensor& ssm_norm_weight;
    const miinfer::GgufTensor& ssm_out_weight;
    const miinfer::GgufTensor& post_norm_weight;
    const miinfer::GgufTensor& ffn_gate_weight;
    const miinfer::GgufTensor& ffn_up_weight;
    const miinfer::GgufTensor& ffn_down_weight;

    Buffer d_attn_norm, d_qkv, d_gate, d_beta, d_alpha, d_conv, d_ssm_norm;
    Buffer d_ssm_out, d_post_norm, d_ffn_gate, d_ffn_up, d_ffn_down, d_dt, d_a;
    Buffer normalized, qkv, gate, beta_raw, alpha_raw, beta, decay, history;
    Buffer query, key, value, query_norm, key_norm, state, recurrent_output;
    Buffer head_norm, gated, projected, residual, post_normalized;
    Buffer ffn_gate, ffn_up, ffn_activation, layer_output, q8;

    RecurrentLayer(const miinfer::Qwen35Model& model_value, std::size_t layer,
                   const std::filesystem::path& fixture)
        : index(layer), model(model_value),
          attn_norm(tensor(*model.file(), name("attn_norm.weight"))),
          qkv_weight(tensor(*model.file(), name("attn_qkv.weight"))),
          gate_weight(tensor(*model.file(), name("attn_gate.weight"))),
          beta_weight(tensor(*model.file(), name("ssm_beta.weight"))),
          alpha_weight(tensor(*model.file(), name("ssm_alpha.weight"))),
          dt_weight(tensor(*model.file(), name("ssm_dt.bias"))),
          a_weight(tensor(*model.file(), name("ssm_a"))),
          conv_weight(tensor(*model.file(), name("ssm_conv1d.weight"))),
          ssm_norm_weight(tensor(*model.file(), name("ssm_norm.weight"))),
          ssm_out_weight(tensor(*model.file(), name("ssm_out.weight"))),
          post_norm_weight(tensor(*model.file(), name("post_attention_norm.weight"))),
          ffn_gate_weight(tensor(*model.file(), name("ffn_gate.weight"))),
          ffn_up_weight(tensor(*model.file(), name("ffn_up.weight"))),
          ffn_down_weight(tensor(*model.file(), name("ffn_down.weight"))) {
        require_type(qkv_weight, {miinfer::GgufTensorType::q4_k,
                                   miinfer::GgufTensorType::q6_k});
        require_type(gate_weight, {miinfer::GgufTensorType::q4_k});
        require_type(ssm_out_weight, {miinfer::GgufTensorType::q5_k});
        require_type(ffn_gate_weight, {miinfer::GgufTensorType::q4_k});
        require_type(ffn_up_weight, {miinfer::GgufTensorType::q4_k});
        require_type(ffn_down_weight, {miinfer::GgufTensorType::q4_k,
                                       miinfer::GgufTensorType::q6_k});

        d_attn_norm = allocate(attn_norm.byte_size);
        d_qkv = allocate(qkv_weight.byte_size);
        d_gate = allocate(gate_weight.byte_size);
        d_beta = allocate(beta_weight.byte_size);
        d_alpha = allocate(alpha_weight.byte_size);
        d_conv = allocate(conv_weight.byte_size);
        d_ssm_norm = allocate(ssm_norm_weight.byte_size);
        d_ssm_out = allocate(ssm_out_weight.byte_size);
        d_post_norm = allocate(post_norm_weight.byte_size);
        d_ffn_gate = allocate(ffn_gate_weight.byte_size);
        d_ffn_up = allocate(ffn_up_weight.byte_size);
        d_ffn_down = allocate(ffn_down_weight.byte_size);
        d_dt = allocate(dt_weight.byte_size);
        d_a = allocate(a_weight.byte_size);
        for (const auto& pair : std::initializer_list<std::pair<const miinfer::GgufTensor*, const Buffer*>>{
                 {&attn_norm, &d_attn_norm}, {&qkv_weight, &d_qkv},
                 {&gate_weight, &d_gate}, {&beta_weight, &d_beta},
                 {&alpha_weight, &d_alpha}, {&conv_weight, &d_conv},
                 {&ssm_norm_weight, &d_ssm_norm}, {&ssm_out_weight, &d_ssm_out},
                 {&post_norm_weight, &d_post_norm}, {&ffn_gate_weight, &d_ffn_gate},
                 {&ffn_up_weight, &d_ffn_up}, {&ffn_down_weight, &d_ffn_down},
                 {&dt_weight, &d_dt}, {&a_weight, &d_a}}) {
            upload_tensor(*pair.first, *pair.second);
        }

        normalized = allocate(kHidden * sizeof(float));
        qkv = allocate(kChannels * sizeof(float));
        gate = allocate(kInner * sizeof(float));
        beta_raw = allocate(kVHeads * sizeof(float));
        alpha_raw = allocate(kVHeads * sizeof(float));
        beta = allocate(kVHeads * sizeof(float));
        decay = allocate(kVHeads * sizeof(float));
        history = allocate(4 * kChannels * sizeof(float));
        query = allocate(kKHeads * kState * sizeof(float));
        key = allocate(kKHeads * kState * sizeof(float));
        value = allocate(kVHeads * kState * sizeof(float));
        query_norm = allocate(kKHeads * kState * sizeof(float));
        key_norm = allocate(kKHeads * kState * sizeof(float));
        state = allocate(kVHeads * kState * kState * sizeof(float));
        recurrent_output = allocate(kVHeads * kState * sizeof(float));
        head_norm = allocate(kVHeads * kState * sizeof(float));
        gated = allocate(kVHeads * kState * sizeof(float));
        projected = allocate(kHidden * sizeof(float));
        residual = allocate(kHidden * sizeof(float));
        post_normalized = allocate(kHidden * sizeof(float));
        ffn_gate = allocate(kFfnInner * sizeof(float));
        ffn_up = allocate(kFfnInner * sizeof(float));
        ffn_activation = allocate(kFfnInner * sizeof(float));
        layer_output = allocate(kHidden * sizeof(float));
        q8 = allocate((kFfnInner / 256) * sizeof(miinfer::Q8KDeviceBlock));

        MIINFER_HIP_CHECK(hipMemset(history->get(), 0, 4 * kChannels * sizeof(float)));
        const auto initial = read_f32(
            checkpoint(fixture, 0, "state_predelta-" + std::to_string(index)),
            kVHeads * kState * kState);
        upload(initial.data(), state->get(), initial.size() * sizeof(float));
    }

    std::string name(const char* suffix) const {
        return "blk." + std::to_string(index) + "." + suffix;
    }

    void run(const float* input, std::uint32_t position, float* output) {
        miinfer::launch_qwen3_rms_norm(
            input, static_cast<const float*>(d_attn_norm->get()),
            static_cast<float*>(normalized->get()), kHidden, model.config().rms_epsilon);
        project(qkv_weight, d_qkv, static_cast<const float*>(normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                static_cast<float*>(qkv->get()), kChannels, kHidden);
        project(gate_weight, d_gate, static_cast<const float*>(normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                static_cast<float*>(gate->get()), kInner, kHidden);
        miinfer::launch_qwen35_f32_gemv(
            static_cast<const float*>(d_beta->get()), static_cast<const float*>(normalized->get()),
            static_cast<float*>(beta_raw->get()), kVHeads, kHidden);
        miinfer::launch_qwen35_f32_gemv(
            static_cast<const float*>(d_alpha->get()), static_cast<const float*>(normalized->get()),
            static_cast<float*>(alpha_raw->get()), kVHeads, kHidden);
        miinfer::launch_qwen35_prepare_beta_decay(
            static_cast<const float*>(beta_raw->get()), static_cast<const float*>(alpha_raw->get()),
            static_cast<const float*>(d_dt->get()), static_cast<const float*>(d_a->get()),
            static_cast<float*>(beta->get()), static_cast<float*>(decay->get()), kVHeads);
        miinfer::launch_qwen35_conv_silu_split(
            static_cast<const float*>(qkv->get()), static_cast<const float*>(d_conv->get()),
            static_cast<float*>(history->get()), static_cast<float*>(query->get()),
            static_cast<float*>(key->get()), static_cast<float*>(value->get()),
            position, 4, kChannels, 4);
        miinfer::launch_qwen35_head_l2_normalize(
            static_cast<const float*>(query->get()), static_cast<float*>(query_norm->get()),
            kKHeads, kState);
        miinfer::launch_qwen35_head_l2_normalize(
            static_cast<const float*>(key->get()), static_cast<float*>(key_norm->get()),
            kKHeads, kState);
        miinfer::launch_qwen35_deltanet_state_update(
            static_cast<const float*>(query_norm->get()), static_cast<const float*>(key_norm->get()),
            static_cast<const float*>(value->get()), static_cast<const float*>(beta->get()),
            static_cast<const float*>(decay->get()), static_cast<float*>(state->get()),
            static_cast<float*>(recurrent_output->get()), kKHeads, kVHeads, kState);
        miinfer::launch_qwen3_head_rms_normalize(
            static_cast<const float*>(recurrent_output->get()), static_cast<float*>(head_norm->get()),
            kVHeads, kState, model.config().rms_epsilon);
        miinfer::launch_qwen3_head_mul(
            static_cast<const float*>(head_norm->get()), static_cast<const float*>(d_ssm_norm->get()),
            static_cast<float*>(gated->get()), kVHeads, kState);
        miinfer::launch_qwen3_silu_mul(
            static_cast<const float*>(gate->get()), static_cast<const float*>(gated->get()),
            static_cast<float*>(gated->get()), kVHeads * kState);
        project(ssm_out_weight, d_ssm_out, static_cast<const float*>(gated->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                static_cast<float*>(projected->get()), kHidden, kInner);
        miinfer::launch_qwen3_add(
            input, static_cast<const float*>(projected->get()),
            static_cast<float*>(residual->get()), kHidden);
        miinfer::launch_qwen3_rms_norm(
            static_cast<const float*>(residual->get()), static_cast<const float*>(d_post_norm->get()),
            static_cast<float*>(post_normalized->get()), kHidden, model.config().rms_epsilon);
        project(ffn_gate_weight, d_ffn_gate, static_cast<const float*>(post_normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden);
        project(ffn_up_weight, d_ffn_up, static_cast<const float*>(post_normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                static_cast<float*>(ffn_up->get()), kFfnInner, kHidden);
        miinfer::launch_qwen3_silu_mul(
            static_cast<const float*>(ffn_gate->get()), static_cast<const float*>(ffn_up->get()),
            static_cast<float*>(ffn_activation->get()), kFfnInner);
        project(ffn_down_weight, d_ffn_down, static_cast<const float*>(ffn_activation->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                static_cast<float*>(projected->get()), kHidden, kFfnInner);
        miinfer::launch_qwen3_add(
            static_cast<const float*>(residual->get()), static_cast<const float*>(projected->get()),
            static_cast<float*>(layer_output->get()), kHidden);
        (void)position;
        MIINFER_HIP_CHECK(hipMemcpy(output, layer_output->get(), kHidden * sizeof(float),
                                    hipMemcpyDeviceToDevice));
    }
};

struct FullAttentionLayer {
    const miinfer::Qwen35Model& model;
    const miinfer::GgufTensor& attn_norm;
    const miinfer::GgufTensor& q_weight;
    const miinfer::GgufTensor& k_weight;
    const miinfer::GgufTensor& v_weight;
    const miinfer::GgufTensor& o_weight;
    const miinfer::GgufTensor& q_norm;
    const miinfer::GgufTensor& k_norm;
    const miinfer::GgufTensor& post_norm;
    const miinfer::GgufTensor& ffn_gate_weight;
    const miinfer::GgufTensor& ffn_up_weight;
    const miinfer::GgufTensor& ffn_down_weight;
    Buffer d_attn_norm, d_q, d_k, d_v, d_o, d_q_norm, d_k_norm, d_post;
    Buffer d_ffn_gate, d_ffn_up, d_ffn_down;
    Buffer normalized, qfull, query, gate, key, key_norm, value, query_norm;
    Buffer query_rope, key_rope, key_cache, value_cache, attention, scores, probabilities;
    Buffer gated_attention, projected, residual, post_normalized, ffn_gate, ffn_up;
    Buffer ffn_activation, layer_output, q8;

    FullAttentionLayer(const miinfer::Qwen35Model& model_value, std::size_t layer)
        : model(model_value),
          attn_norm(tensor(*model.file(), prefix(layer, "attn_norm.weight"))),
          q_weight(tensor(*model.file(), prefix(layer, "attn_q.weight"))),
          k_weight(tensor(*model.file(), prefix(layer, "attn_k.weight"))),
          v_weight(tensor(*model.file(), prefix(layer, "attn_v.weight"))),
          o_weight(tensor(*model.file(), prefix(layer, "attn_output.weight"))),
          q_norm(tensor(*model.file(), prefix(layer, "attn_q_norm.weight"))),
          k_norm(tensor(*model.file(), prefix(layer, "attn_k_norm.weight"))),
          post_norm(tensor(*model.file(), prefix(layer, "post_attention_norm.weight"))),
          ffn_gate_weight(tensor(*model.file(), prefix(layer, "ffn_gate.weight"))),
          ffn_up_weight(tensor(*model.file(), prefix(layer, "ffn_up.weight"))),
          ffn_down_weight(tensor(*model.file(), prefix(layer, "ffn_down.weight"))) {
        for (const auto* weight : {&q_weight, &k_weight, &v_weight, &o_weight,
                                   &ffn_gate_weight, &ffn_up_weight, &ffn_down_weight}) {
            require_type(*weight, {miinfer::GgufTensorType::q4_k,
                                   miinfer::GgufTensorType::q6_k});
        }
        d_attn_norm = copy_weight(attn_norm); d_q = copy_weight(q_weight);
        d_k = copy_weight(k_weight); d_v = copy_weight(v_weight); d_o = copy_weight(o_weight);
        d_q_norm = copy_weight(q_norm); d_k_norm = copy_weight(k_norm);
        d_post = copy_weight(post_norm); d_ffn_gate = copy_weight(ffn_gate_weight);
        d_ffn_up = copy_weight(ffn_up_weight); d_ffn_down = copy_weight(ffn_down_weight);
        normalized = allocate(kHidden * sizeof(float)); qfull = allocate(12288 * sizeof(float));
        query = allocate(6144 * sizeof(float)); gate = allocate(6144 * sizeof(float));
        key = allocate(1024 * sizeof(float)); key_norm = allocate(1024 * sizeof(float));
        value = allocate(1024 * sizeof(float)); query_norm = allocate(6144 * sizeof(float));
        query_rope = allocate(6144 * sizeof(float)); key_rope = allocate(1024 * sizeof(float));
        key_cache = allocate(4 * kCacheCapacity * 256 * sizeof(float));
        value_cache = allocate(4 * kCacheCapacity * 256 * sizeof(float)); attention = allocate(6144 * sizeof(float));
        scores = allocate(24 * kCacheCapacity * sizeof(float)); probabilities = allocate(24 * kCacheCapacity * sizeof(float));
        gated_attention = allocate(6144 * sizeof(float)); projected = allocate(kHidden * sizeof(float));
        residual = allocate(kHidden * sizeof(float)); post_normalized = allocate(kHidden * sizeof(float));
        ffn_gate = allocate(kFfnInner * sizeof(float)); ffn_up = allocate(kFfnInner * sizeof(float));
        ffn_activation = allocate(kFfnInner * sizeof(float)); layer_output = allocate(kHidden * sizeof(float));
        q8 = allocate((kFfnInner / 256) * sizeof(miinfer::Q8KDeviceBlock));
        MIINFER_HIP_CHECK(hipMemset(key_cache->get(), 0, 4 * kCacheCapacity * 256 * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(value_cache->get(), 0, 4 * kCacheCapacity * 256 * sizeof(float)));
    }

    static std::string prefix(std::size_t layer, const char* suffix) {
        return "blk." + std::to_string(layer) + "." + suffix;
    }
    Buffer copy_weight(const miinfer::GgufTensor& weight) {
        auto result = allocate(weight.byte_size);
        upload_tensor(weight, result);
        return result;
    }

    void run(const float* input, std::uint32_t position, float* output) {
        miinfer::launch_qwen3_rms_norm(input, static_cast<const float*>(d_attn_norm->get()),
            static_cast<float*>(normalized->get()), kHidden, model.config().rms_epsilon);
        project(q_weight, d_q, static_cast<const float*>(normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(qfull->get()), 12288, kHidden);
        miinfer::launch_qwen35_split_q_gate(static_cast<const float*>(qfull->get()),
            static_cast<float*>(query->get()), static_cast<float*>(gate->get()), 24, 256);
        for (std::uint32_t h = 0; h < 24; ++h) {
            miinfer::launch_qwen3_rms_normalize(static_cast<const float*>(query->get()) + h * 256,
                static_cast<float*>(query_norm->get()) + h * 256, 256, model.config().rms_epsilon);
        }
        miinfer::launch_qwen3_head_mul(static_cast<const float*>(query_norm->get()),
            static_cast<const float*>(d_q_norm->get()), static_cast<float*>(query_norm->get()), 24, 256);
        project(k_weight, d_k, static_cast<const float*>(normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(key->get()), 1024, kHidden);
        for (std::uint32_t h = 0; h < 4; ++h) {
            miinfer::launch_qwen3_rms_normalize(static_cast<const float*>(key->get()) + h * 256,
                static_cast<float*>(key_norm->get()) + h * 256, 256, model.config().rms_epsilon);
        }
        miinfer::launch_qwen3_head_mul(static_cast<const float*>(key_norm->get()),
            static_cast<const float*>(d_k_norm->get()), static_cast<float*>(key_norm->get()), 4, 256);
        project(v_weight, d_v, static_cast<const float*>(normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(value->get()), 1024, kHidden);
        miinfer::launch_qwen35_rope_sections(static_cast<const float*>(query_norm->get()),
            static_cast<float*>(query_rope->get()), 24, 256, position, model.config().rope_theta);
        miinfer::launch_qwen35_rope_sections(static_cast<const float*>(key_norm->get()),
            static_cast<float*>(key_rope->get()), 4, 256, position, model.config().rope_theta);
        miinfer::launch_qwen3_kv_cache_store(static_cast<const float*>(key_rope->get()),
            static_cast<const float*>(value->get()), static_cast<float*>(key_cache->get()),
            static_cast<float*>(value_cache->get()), position, kCacheCapacity, 4, 256);
        miinfer::launch_qwen3_cached_attention_parallel(static_cast<const float*>(query_rope->get()),
            static_cast<const float*>(key_cache->get()), static_cast<const float*>(value_cache->get()),
            position + 1, kCacheCapacity, static_cast<float*>(attention->get()), static_cast<float*>(scores->get()),
            static_cast<float*>(probabilities->get()), 24, 4, 256, 1.0F / std::sqrt(256.0F));
        miinfer::launch_qwen35_sigmoid_mul(static_cast<const float*>(attention->get()),
            static_cast<const float*>(gate->get()), static_cast<float*>(gated_attention->get()), 6144);
        project(o_weight, d_o, static_cast<const float*>(gated_attention->get()),
            static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(projected->get()), kHidden, 6144);
        miinfer::launch_qwen3_add(input, static_cast<const float*>(projected->get()),
            static_cast<float*>(residual->get()), kHidden);
        miinfer::launch_qwen3_rms_norm(static_cast<const float*>(residual->get()),
            static_cast<const float*>(d_post->get()), static_cast<float*>(post_normalized->get()), kHidden,
            model.config().rms_epsilon);
        project(ffn_gate_weight, d_ffn_gate, static_cast<const float*>(post_normalized->get()),
            static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden);
        project(ffn_up_weight, d_ffn_up, static_cast<const float*>(post_normalized->get()),
            static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(ffn_up->get()), kFfnInner, kHidden);
        miinfer::launch_qwen3_silu_mul(static_cast<const float*>(ffn_gate->get()),
            static_cast<const float*>(ffn_up->get()), static_cast<float*>(ffn_activation->get()), kFfnInner);
        project(ffn_down_weight, d_ffn_down, static_cast<const float*>(ffn_activation->get()),
            static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(projected->get()), kHidden, kFfnInner);
        miinfer::launch_qwen3_add(static_cast<const float*>(residual->get()),
            static_cast<const float*>(projected->get()), static_cast<float*>(layer_output->get()), kHidden);
        MIINFER_HIP_CHECK(hipMemcpy(output, layer_output->get(), kHidden * sizeof(float), hipMemcpyDeviceToDevice));
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: miinfer-m6a21-qwen35-gpu-hybrid-block MODEL.gguf FIXTURE_DIR [--deep]\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        const bool deep = argc == 4 && std::string(argv[3]) == "--deep";
        if (argc == 4 && !deep) throw std::runtime_error("unknown option");
        RecurrentLayer recurrent0(model, 0, fixture);
        RecurrentLayer recurrent1(model, 1, fixture);
        RecurrentLayer recurrent2(model, 2, fixture);
        FullAttentionLayer attention(model, 3);
        Buffer input = allocate(kHidden * sizeof(float));
        Buffer state1 = allocate(kHidden * sizeof(float));
        Buffer state2 = allocate(kHidden * sizeof(float));
        Buffer state3 = allocate(kHidden * sizeof(float));
        Buffer output = allocate(kHidden * sizeof(float));
        float maximum = 0.0F;
        const auto generated = deep ? read_tokens(fixture / "generated_tokens.txt")
                                    : std::vector<std::uint32_t>{};
        if (deep && generated.size() < 64) throw std::runtime_error("fixture has fewer than 64 tokens");
        const auto is_checkpoint_position = [](std::size_t position) {
            return position == 0 || position == 1 || position == 2 || position == 4
                || position == 8 || position == 16 || position == 32 || position == 64;
        };
        if (deep) {
            std::cout << "position l0_max l0_rms l0_rel l1_max l1_rms l1_rel "
                         "l2_max l2_rms l2_rel l3_max l3_rms l3_rel state_correct\n";
        }

        const std::size_t last_position = deep ? 64 : 1;
        for (std::size_t position = 0; position <= last_position; ++position) {
            const auto host_input = deep && position > 1
                ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
            upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
            bool state_correct = true;
            if (deep && is_checkpoint_position(position)) {
                for (std::size_t layer = 0; layer < 3; ++layer) {
                    const auto& state_buffer = layer == 0 ? recurrent0.state
                        : layer == 1 ? recurrent1.state : recurrent2.state;
                    const auto state_error = detailed_device_error(
                        static_cast<const float*>(state_buffer->get()),
                        kVHeads * kState * kState,
                        checkpoint(fixture, position, "state_predelta-" + std::to_string(layer)));
                    state_correct = state_correct && state_error.max_abs <= 5.0e-2F;
                }
            }
            recurrent0.run(static_cast<const float*>(input->get()), position,
                           static_cast<float*>(state1->get()));
            recurrent1.run(static_cast<const float*>(state1->get()), position,
                           static_cast<float*>(state2->get()));
            recurrent2.run(static_cast<const float*>(state2->get()), position,
                           static_cast<float*>(state3->get()));
            attention.run(static_cast<const float*>(state3->get()), position,
                          static_cast<float*>(output->get()));
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            if (!deep || is_checkpoint_position(position)) {
                std::array<DetailedError, 4> errors{};
                for (std::size_t layer = 0; layer < 4; ++layer) {
                    const auto* layer_output = static_cast<const float*>(layer == 0 ? state1->get() :
                        layer == 1 ? state2->get() : layer == 2 ? state3->get() : output->get());
                    errors[layer] = detailed_device_error(
                        layer_output, kHidden,
                        checkpoint(fixture, position, "l_out-" + std::to_string(layer)));
                    require_match("hybrid layer output", Metrics{errors[layer].max_abs,
                                                                  errors[layer].rms, 0}, 2.0F);
                    maximum = std::max(maximum, errors[layer].max_abs);
                }
                if (deep && !state_correct) {
                    throw std::runtime_error("hybrid recurrent state mismatch");
                }
                if (deep) {
                    std::cout << position;
                    for (const auto& error : errors) {
                        std::cout << ' ' << error.max_abs << ' ' << error.rms
                                  << ' ' << error.relative_rms;
                    }
                    std::cout << ' ' << (state_correct ? "PASS" : "FAIL") << '\n';
                    std::cout << "  fingerprints state0="
                              << fingerprint(recurrent0.state->get(), kVHeads * kState * kState * sizeof(float))
                              << " state1="
                              << fingerprint(recurrent1.state->get(), kVHeads * kState * kState * sizeof(float))
                              << " state2="
                              << fingerprint(recurrent2.state->get(), kVHeads * kState * kState * sizeof(float))
                              << " K=" << fingerprint(attention.key_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float))
                              << " V=" << fingerprint(attention.value_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float)) << '\n';
                } else {
                    for (std::size_t layer = 0; layer < 4; ++layer) {
                        std::cout << "position=" << position << " layer=" << layer
                                  << " max_abs=" << errors[layer].max_abs
                                  << " rmse=" << errors[layer].rms << '\n';
                    }
                }
            }
        }
        std::cout << "max_error=" << maximum << '\n'
                  << (deep ? "M6-A22" : "M6-A21")
                  << " qwen35 GPU hybrid block PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A21 failed: " << error.what() << '\n';
        return 1;
    }
}
