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
#include <ctime>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
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

std::size_t g_device_allocations = 0;
std::size_t g_device_bytes = 0;
std::size_t g_peak_device_bytes = 0;

Buffer allocate(std::size_t bytes) {
    auto result = std::make_unique<DeviceBytes>(bytes);
    ++g_device_allocations;
    g_device_bytes += bytes;
    g_peak_device_bytes = std::max(g_peak_device_bytes, g_device_bytes);
    return result;
}

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

struct LocatedError {
    DetailedError metrics;
    std::size_t index = 0;
    float actual = 0.0F;
    float expected = 0.0F;
};

LocatedError located_device_error(const float* device, std::size_t elements,
                                  const std::filesystem::path& expected_path) {
    std::vector<float> actual(elements);
    MIINFER_HIP_CHECK(hipMemcpy(actual.data(), device, elements * sizeof(float),
                                hipMemcpyDeviceToHost));
    const auto expected = read_f32(expected_path, elements);
    LocatedError result;
    result.metrics = detailed_compare(actual, expected);
    for (std::size_t i = 0; i < elements; ++i) {
        if (std::fabs(actual[i] - expected[i]) >
            std::fabs(result.actual - result.expected)) {
            result.index = i;
            result.actual = actual[i];
            result.expected = expected[i];
        }
    }
    return result;
}

struct RecurrentTrace {
    const std::filesystem::path& fixture;
    std::size_t layer;
    std::uint32_t position;

    void report_at(std::uint32_t checkpoint_position, const char* label,
                   const float* device, std::size_t elements,
                   const std::string& reference_name) const {
        const auto error = located_device_error(
            device, elements, checkpoint(fixture, checkpoint_position, reference_name));
        std::cout << "trace label=" << label
                  << " max_abs=" << error.metrics.max_abs
                  << " mean_abs=" << error.metrics.mean_abs
                  << " rms=" << error.metrics.rms
                  << " relative_rms=" << error.metrics.relative_rms
                  << " max_index=" << error.index
                  << " reference=" << error.expected
                  << " gpu=" << error.actual << '\n';
    }

    void report(const char* label, const float* device, std::size_t elements,
                const std::string& reference_name) const {
        report_at(position, label, device, elements, reference_name);
    }
};

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
    RecurrentTrace* trace = nullptr;

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

    void poison() {
        MIINFER_HIP_CHECK(hipMemset(state->get(), 0xA5,
                                    kVHeads * kState * kState * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(history->get(), 0xFF,
                                    4 * kChannels * sizeof(float)));
    }

    void reset(const std::filesystem::path& fixture) {
        MIINFER_HIP_CHECK(hipMemset(history->get(), 0, 4 * kChannels * sizeof(float)));
        const auto initial = read_f32(
            checkpoint(fixture, 0, "state_predelta-" + std::to_string(index)),
            kVHeads * kState * kState);
        upload(initial.data(), state->get(), initial.size() * sizeof(float));
    }

    void trace_tensor(std::uint32_t position, const char* label, const float* device,
                      std::size_t elements, const std::string& reference_name) const {
        if (trace != nullptr && trace->layer == index && trace->position == position) {
            trace->report(label, device, elements, reference_name);
        }
    }

    void run(const float* input, std::uint32_t position, float* output) {
        miinfer::launch_qwen3_rms_norm(
            input, static_cast<const float*>(d_attn_norm->get()),
            static_cast<float*>(normalized->get()), kHidden, model.config().rms_epsilon);
        trace_tensor(position, "attn_norm", static_cast<const float*>(normalized->get()),
                     kHidden, "attn_norm-" + std::to_string(index));
        project(qkv_weight, d_qkv, static_cast<const float*>(normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                static_cast<float*>(qkv->get()), kChannels, kHidden);
        trace_tensor(position, "qkv", static_cast<const float*>(qkv->get()), kChannels,
                     "linear_attn_qkv_mixed-" + std::to_string(index));
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
        if (trace != nullptr && trace->layer == index && position + 1 == trace->position) {
            trace->report_at(position + 1, "state_after", static_cast<const float*>(state->get()),
                             kVHeads * kState * kState,
                             "state_predelta-" + std::to_string(index));
        }
        trace_tensor(position, "recurrent_output",
                     static_cast<const float*>(recurrent_output->get()), kVHeads * kState,
                     "attn_output-" + std::to_string(index));
        miinfer::launch_qwen3_head_rms_normalize(
            static_cast<const float*>(recurrent_output->get()), static_cast<float*>(head_norm->get()),
            kVHeads, kState, model.config().rms_epsilon);
        miinfer::launch_qwen3_head_mul(
            static_cast<const float*>(head_norm->get()), static_cast<const float*>(d_ssm_norm->get()),
            static_cast<float*>(gated->get()), kVHeads, kState);
        miinfer::launch_qwen3_silu_mul(
            static_cast<const float*>(gate->get()), static_cast<const float*>(gated->get()),
            static_cast<float*>(gated->get()), kVHeads * kState);
        trace_tensor(position, "gated", static_cast<const float*>(gated->get()), kVHeads * kState,
                     "final_output-" + std::to_string(index));
        project(ssm_out_weight, d_ssm_out, static_cast<const float*>(gated->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                static_cast<float*>(projected->get()), kHidden, kInner);
        miinfer::launch_qwen3_add(
            input, static_cast<const float*>(projected->get()),
            static_cast<float*>(residual->get()), kHidden);
        trace_tensor(position, "attention_residual", static_cast<const float*>(residual->get()),
                     kHidden, "attn_residual-" + std::to_string(index));
        miinfer::launch_qwen3_rms_norm(
            static_cast<const float*>(residual->get()), static_cast<const float*>(d_post_norm->get()),
            static_cast<float*>(post_normalized->get()), kHidden, model.config().rms_epsilon);
        trace_tensor(position, "post_attention_norm",
                     static_cast<const float*>(post_normalized->get()), kHidden,
                     "attn_post_norm-" + std::to_string(index));
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
        trace_tensor(position, "ffn_out", static_cast<const float*>(projected->get()), kHidden,
                     "ffn_out-" + std::to_string(index));
        miinfer::launch_qwen3_add(
            static_cast<const float*>(residual->get()), static_cast<const float*>(projected->get()),
            static_cast<float*>(layer_output->get()), kHidden);
        trace_tensor(position, "layer_output", static_cast<const float*>(layer_output->get()),
                     kHidden, "l_out-" + std::to_string(index));
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

    void poison() {
        MIINFER_HIP_CHECK(hipMemset(key_cache->get(), 0xA5,
                                    4 * kCacheCapacity * 256 * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(value_cache->get(), 0xA5,
                                    4 * kCacheCapacity * 256 * sizeof(float)));
    }

    void reset() {
        MIINFER_HIP_CHECK(hipMemset(key_cache->get(), 0,
                                    4 * kCacheCapacity * 256 * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(value_cache->get(), 0,
                                    4 * kCacheCapacity * 256 * sizeof(float)));
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

void run_hybrid_block(RecurrentLayer& recurrent0, RecurrentLayer& recurrent1,
                      RecurrentLayer& recurrent2, FullAttentionLayer& attention,
                      const float* input, std::uint32_t position,
                      float* state1, float* state2, float* state3, float* output) {
    recurrent0.run(input, position, state1);
    recurrent1.run(state1, position, state2);
    recurrent2.run(state2, position, state3);
    attention.run(state3, position, output);
}

struct GpuLayerRef {
    RecurrentLayer* recurrent = nullptr;
    FullAttentionLayer* attention = nullptr;

    void run(const float* input, std::uint32_t position, float* output) const {
        if (recurrent != nullptr) {
            recurrent->run(input, position, output);
        } else if (attention != nullptr) {
            attention->run(input, position, output);
        } else {
            throw std::runtime_error("empty qwen35 GPU layer reference");
        }
    }
};

void run_prefix(std::span<const GpuLayerRef> layers, std::span<float* const> outputs,
                const float* input, std::uint32_t position) {
    const float* current = input;
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        layers[layer].run(current, position, outputs[layer]);
        current = outputs[layer];
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: miinfer-m6a21-qwen35-gpu-hybrid-block MODEL.gguf FIXTURE_DIR "
                     "[--deep|--block4-7|--prefix8|--prefix16|--prefix32|--prefix32-locate]\n";
        return 2;
    }
    const std::string mode = argc == 4 ? argv[3] : "";
    const bool prefix8 = mode == "--prefix8";
    const bool prefix16 = mode == "--prefix16";
    const bool locate32 = mode == "--prefix32-locate";
    const bool prefix32 = mode == "--prefix32" || locate32;
    const bool deep = mode == "--deep" || mode == "--block4-7" || prefix8 || prefix16 || prefix32;
    const bool second_block = mode == "--block4-7" || prefix8 || prefix16 || prefix32;
    if (argc == 4 && !deep) {
        std::cerr << "unknown option: " << mode << '\n';
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        RecurrentLayer recurrent0(model, 0, fixture);
        RecurrentLayer recurrent1(model, 1, fixture);
        RecurrentLayer recurrent2(model, 2, fixture);
        FullAttentionLayer attention3(model, 3);
        std::unique_ptr<RecurrentLayer> recurrent4;
        std::unique_ptr<RecurrentLayer> recurrent5;
        std::unique_ptr<RecurrentLayer> recurrent6;
        std::unique_ptr<FullAttentionLayer> attention7;
        std::unique_ptr<RecurrentLayer> recurrent8;
        std::unique_ptr<RecurrentLayer> recurrent9;
        std::unique_ptr<RecurrentLayer> recurrent10;
        std::unique_ptr<FullAttentionLayer> attention11;
        std::unique_ptr<RecurrentLayer> recurrent12;
        std::unique_ptr<RecurrentLayer> recurrent13;
        std::unique_ptr<RecurrentLayer> recurrent14;
        std::unique_ptr<FullAttentionLayer> attention15;
        std::unique_ptr<RecurrentLayer> recurrent16;
        std::unique_ptr<RecurrentLayer> recurrent17;
        std::unique_ptr<RecurrentLayer> recurrent18;
        std::unique_ptr<FullAttentionLayer> attention19;
        std::unique_ptr<RecurrentLayer> recurrent20;
        std::unique_ptr<RecurrentLayer> recurrent21;
        std::unique_ptr<RecurrentLayer> recurrent22;
        std::unique_ptr<FullAttentionLayer> attention23;
        std::unique_ptr<RecurrentLayer> recurrent24;
        std::unique_ptr<RecurrentLayer> recurrent25;
        std::unique_ptr<RecurrentLayer> recurrent26;
        std::unique_ptr<FullAttentionLayer> attention27;
        std::unique_ptr<RecurrentLayer> recurrent28;
        std::unique_ptr<RecurrentLayer> recurrent29;
        std::unique_ptr<RecurrentLayer> recurrent30;
        std::unique_ptr<FullAttentionLayer> attention31;
        if (second_block) {
            recurrent4 = std::make_unique<RecurrentLayer>(model, 4, fixture);
            recurrent5 = std::make_unique<RecurrentLayer>(model, 5, fixture);
            recurrent6 = std::make_unique<RecurrentLayer>(model, 6, fixture);
            attention7 = std::make_unique<FullAttentionLayer>(model, 7);
        }
        if (prefix16 || prefix32) {
            recurrent8 = std::make_unique<RecurrentLayer>(model, 8, fixture);
            recurrent9 = std::make_unique<RecurrentLayer>(model, 9, fixture);
            recurrent10 = std::make_unique<RecurrentLayer>(model, 10, fixture);
            attention11 = std::make_unique<FullAttentionLayer>(model, 11);
            recurrent12 = std::make_unique<RecurrentLayer>(model, 12, fixture);
            recurrent13 = std::make_unique<RecurrentLayer>(model, 13, fixture);
            recurrent14 = std::make_unique<RecurrentLayer>(model, 14, fixture);
            attention15 = std::make_unique<FullAttentionLayer>(model, 15);
        }
        if (prefix32) {
            recurrent16 = std::make_unique<RecurrentLayer>(model, 16, fixture);
            recurrent17 = std::make_unique<RecurrentLayer>(model, 17, fixture);
            recurrent18 = std::make_unique<RecurrentLayer>(model, 18, fixture);
            attention19 = std::make_unique<FullAttentionLayer>(model, 19);
            recurrent20 = std::make_unique<RecurrentLayer>(model, 20, fixture);
            recurrent21 = std::make_unique<RecurrentLayer>(model, 21, fixture);
            recurrent22 = std::make_unique<RecurrentLayer>(model, 22, fixture);
            attention23 = std::make_unique<FullAttentionLayer>(model, 23);
            recurrent24 = std::make_unique<RecurrentLayer>(model, 24, fixture);
            recurrent25 = std::make_unique<RecurrentLayer>(model, 25, fixture);
            recurrent26 = std::make_unique<RecurrentLayer>(model, 26, fixture);
            attention27 = std::make_unique<FullAttentionLayer>(model, 27);
            recurrent28 = std::make_unique<RecurrentLayer>(model, 28, fixture);
            recurrent29 = std::make_unique<RecurrentLayer>(model, 29, fixture);
            recurrent30 = std::make_unique<RecurrentLayer>(model, 30, fixture);
            attention31 = std::make_unique<FullAttentionLayer>(model, 31);
        }

        if (locate32) {
            const std::array<std::size_t, 13> positions{{
                1, 2, 4, 8, 16, 32, 48, 56, 60, 61, 62, 63, 64}};
            const auto is_position = [&positions](std::size_t position) {
                return std::find(positions.begin(), positions.end(), position) != positions.end();
            };
            const std::array<GpuLayerRef, 32> layers{{
                {&recurrent0, nullptr}, {&recurrent1, nullptr}, {&recurrent2, nullptr},
                {nullptr, &attention3}, {recurrent4.get(), nullptr},
                {recurrent5.get(), nullptr}, {recurrent6.get(), nullptr},
                {nullptr, attention7.get()}, {recurrent8.get(), nullptr},
                {recurrent9.get(), nullptr}, {recurrent10.get(), nullptr},
                {nullptr, attention11.get()}, {recurrent12.get(), nullptr},
                {recurrent13.get(), nullptr}, {recurrent14.get(), nullptr},
                {nullptr, attention15.get()}, {recurrent16.get(), nullptr},
                {recurrent17.get(), nullptr}, {recurrent18.get(), nullptr},
                {nullptr, attention19.get()}, {recurrent20.get(), nullptr},
                {recurrent21.get(), nullptr}, {recurrent22.get(), nullptr},
                {nullptr, attention23.get()}, {recurrent24.get(), nullptr},
                {recurrent25.get(), nullptr}, {recurrent26.get(), nullptr},
                {nullptr, attention27.get()}, {recurrent28.get(), nullptr},
                {recurrent29.get(), nullptr}, {recurrent30.get(), nullptr},
                {nullptr, attention31.get()}}};
            std::array<Buffer, 32> outputs{};
            std::array<float*, 32> output_pointers{};
            for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                outputs[layer] = allocate(kHidden * sizeof(float));
                output_pointers[layer] = static_cast<float*>(outputs[layer]->get());
            }
            Buffer input = allocate(kHidden * sizeof(float));
            const auto generated = read_tokens(fixture / "generated_tokens.txt");
            if (generated.size() < 64) throw std::runtime_error("fixture has fewer than 64 tokens");
            RecurrentTrace l30_trace{fixture, 30, 64};
            recurrent30->trace = &l30_trace;
            const auto allocations_before_decode = g_device_allocations;
            const std::size_t state_bytes = kVHeads * kState * kState * sizeof(float);
            std::cout << "position l30_state_max l30_state_mean l30_state_rms l30_state_rel l30_index "
                         "l30_reference l30_gpu\n";
            for (std::size_t position = 0; position <= 64; ++position) {
                const auto host_input = position > 1
                    ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                    : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                if (is_position(position)) {
                    const auto state = located_device_error(
                        static_cast<const float*>(recurrent30->state->get()),
                        kVHeads * kState * kState,
                        checkpoint(fixture, position, "state_predelta-30"));
                    std::cout << position << ' ' << state.metrics.max_abs << ' '
                              << state.metrics.mean_abs << ' '
                              << state.metrics.rms << ' ' << state.metrics.relative_rms << ' '
                              << state.index << ' ' << state.expected << ' ' << state.actual << '\n';
                    if (position == 64) {
                        for (std::size_t layer = 28; layer < 31; ++layer) {
                            const auto adjacent = located_device_error(
                                static_cast<const float*>(layers[layer].recurrent->state->get()),
                                kVHeads * kState * kState,
                                checkpoint(fixture, position,
                                           "state_predelta-" + std::to_string(layer)));
                            std::cout << "entry_state_layer=" << layer
                                      << " max_abs=" << adjacent.metrics.max_abs
                                      << " mean_abs=" << adjacent.metrics.mean_abs
                                      << " rms=" << adjacent.metrics.rms
                                      << " relative_rms=" << adjacent.metrics.relative_rms
                                      << " max_index=" << adjacent.index
                                      << " reference=" << adjacent.expected
                                      << " gpu=" << adjacent.actual << '\n';
                        }
                    }
                }
                run_prefix(std::span<const GpuLayerRef>(layers),
                           std::span<float* const>(output_pointers),
                           static_cast<const float*>(input->get()), position);
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                if (position != 64) continue;
                std::cout << "p64_layers\n";
                for (std::size_t layer = 28; layer < 32; ++layer) {
                    const auto error = located_device_error(
                        static_cast<const float*>(outputs[layer]->get()), kHidden,
                        checkpoint(fixture, position, "l_out-" + std::to_string(layer)));
                    std::cout << "layer=" << layer << " max_abs=" << error.metrics.max_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " reference=" << error.expected
                              << " gpu=" << error.actual << '\n';
                }
                std::cout << "p64_fingerprints"
                          << " state28=" << fingerprint(recurrent28->state->get(), state_bytes)
                          << " state29=" << fingerprint(recurrent29->state->get(), state_bytes)
                          << " state30=" << fingerprint(recurrent30->state->get(), state_bytes)
                          << " K27=" << fingerprint(attention27->key_cache->get(),
                              4 * (position + 1) * 256 * sizeof(float))
                          << " V27=" << fingerprint(attention27->value_cache->get(),
                              4 * (position + 1) * 256 * sizeof(float)) << '\n';
            }
            std::cout << "allocations_during_decode="
                      << (g_device_allocations - allocations_before_decode)
                      << " device_bytes_after_setup=" << g_device_bytes
                      << " peak_device_bytes=" << g_peak_device_bytes << '\n'
                      << "M6-A26.1 qwen35 L30 state localization COMPLETE\n";
            return 0;
        }

        if (prefix8 || prefix16 || prefix32) {
            const std::size_t layer_count = prefix32 ? 32 : prefix16 ? 16 : 8;
            const std::array<GpuLayerRef, 32> layers{{
                {&recurrent0, nullptr}, {&recurrent1, nullptr}, {&recurrent2, nullptr},
                {nullptr, &attention3}, {recurrent4.get(), nullptr},
                {recurrent5.get(), nullptr}, {recurrent6.get(), nullptr},
                {nullptr, attention7.get()}, {recurrent8.get(), nullptr},
                {recurrent9.get(), nullptr}, {recurrent10.get(), nullptr},
                {nullptr, attention11.get()}, {recurrent12.get(), nullptr},
                {recurrent13.get(), nullptr}, {recurrent14.get(), nullptr},
                {nullptr, attention15.get()}, {recurrent16.get(), nullptr},
                {recurrent17.get(), nullptr}, {recurrent18.get(), nullptr},
                {nullptr, attention19.get()}, {recurrent20.get(), nullptr},
                {recurrent21.get(), nullptr}, {recurrent22.get(), nullptr},
                {nullptr, attention23.get()}, {recurrent24.get(), nullptr},
                {recurrent25.get(), nullptr}, {recurrent26.get(), nullptr},
                {nullptr, attention27.get()}, {recurrent28.get(), nullptr},
                {recurrent29.get(), nullptr}, {recurrent30.get(), nullptr},
                {nullptr, attention31.get()}}};
            std::array<Buffer, 32> outputs{};
            std::array<float*, 32> output_pointers{};
            for (std::size_t layer = 0; layer < layer_count; ++layer) {
                outputs[layer] = allocate(kHidden * sizeof(float));
                output_pointers[layer] = static_cast<float*>(outputs[layer]->get());
            }
            Buffer input = allocate(kHidden * sizeof(float));
            const auto generated = read_tokens(fixture / "generated_tokens.txt");
            if (generated.size() < 64) throw std::runtime_error("fixture has fewer than 64 tokens");
            const auto is_checkpoint_position = [](std::size_t position) {
                return position == 0 || position == 1 || position == 2 || position == 4
                    || position == 8 || position == 16 || position == 32 || position == 64;
            };
            const std::size_t state_bytes = kVHeads * kState * kState * sizeof(float);
            const auto active_fingerprint = [](const void* device, std::size_t bytes) {
                return bytes == 0 ? 1469598103934665603ULL : fingerprint(device, bytes);
            };
            const auto record_caches = [&](std::size_t position, bool before,
                                           const auto& record) {
                const std::size_t bytes = 4 * (position + (before ? 0 : 1))
                    * 256 * sizeof(float);
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    if (layers[layer].attention == nullptr) continue;
                    record(active_fingerprint(layers[layer].attention->key_cache->get(), bytes));
                    record(active_fingerprint(layers[layer].attention->value_cache->get(), bytes));
                }
            };
            std::vector<std::uint64_t> replay_fingerprints;
            const auto record = [&replay_fingerprints](std::uint64_t value) {
                replay_fingerprints.push_back(value);
            };
            const auto allocations_before_decode = g_device_allocations;
            double prefix_cpu_ms = 0.0;
            std::cout << "position";
            for (std::size_t layer = 0; layer < layer_count; ++layer) {
                std::cout << " l" << layer << "_max l" << layer << "_rms l" << layer << "_rel";
            }
            std::cout << " state_correct\n";
            float maximum = 0.0F;
            for (std::size_t position = 0; position <= 64; ++position) {
                const auto host_input = position > 1
                    ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                    : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                bool state_correct = true;
                if (is_checkpoint_position(position)) {
                    for (std::size_t layer = 0; layer < layer_count; ++layer) {
                        if (layers[layer].recurrent == nullptr) continue;
                        const auto error = detailed_device_error(
                            static_cast<const float*>(layers[layer].recurrent->state->get()),
                            kVHeads * kState * kState,
                            checkpoint(fixture, position,
                                       "state_predelta-" + std::to_string(layer)));
                        if (error.max_abs > 5.0e-2F) {
                            std::cerr << "state mismatch position=" << position
                                      << " layer=" << layer
                                      << " max_abs=" << error.max_abs
                                      << " rms=" << error.rms
                                      << " relative_rms=" << error.relative_rms << '\n';
                        }
                        state_correct = state_correct && error.max_abs <= 5.0e-2F;
                    }
                    for (std::size_t layer = 0; layer < layer_count; ++layer) {
                        if (layers[layer].recurrent != nullptr) {
                            record(fingerprint(layers[layer].recurrent->state->get(), state_bytes));
                        }
                    }
                    record_caches(position, true, record);
                }
                const auto run_start = std::clock();
                run_prefix(std::span<const GpuLayerRef>(layers).first(layer_count),
                           std::span<float* const>(output_pointers).first(layer_count),
                           static_cast<const float*>(input->get()), position);
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                prefix_cpu_ms += 1000.0 * static_cast<double>(std::clock() - run_start)
                    / static_cast<double>(CLOCKS_PER_SEC);
                if (!is_checkpoint_position(position)) continue;
                std::array<DetailedError, 32> errors{};
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    errors[layer] = detailed_device_error(
                        static_cast<const float*>(outputs[layer]->get()), kHidden,
                        checkpoint(fixture, position, "l_out-" + std::to_string(layer)));
                    require_match("prefix output",
                                  Metrics{errors[layer].max_abs, errors[layer].rms, 0}, 2.0F);
                    maximum = std::max(maximum, errors[layer].max_abs);
                }
                if (!state_correct) throw std::runtime_error("prefix recurrent state mismatch");
                std::cout << position;
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    const auto& error = errors[layer];
                    std::cout << ' ' << error.max_abs << ' ' << error.rms
                              << ' ' << error.relative_rms;
                }
                std::cout << " PASS\n  fingerprints hidden3="
                          << fingerprint(outputs[3]->get(), kHidden * sizeof(float))
                          << " hidden7=" << fingerprint(outputs[7]->get(), kHidden * sizeof(float));
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    if (layers[layer].recurrent != nullptr) {
                        std::cout << " state" << layer << '='
                                  << fingerprint(layers[layer].recurrent->state->get(), state_bytes);
                    }
                }
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    if (layers[layer].attention == nullptr) continue;
                    std::cout << " K" << layer << '=' << fingerprint(
                        layers[layer].attention->key_cache->get(),
                        4 * (position + 1) * 256 * sizeof(float))
                              << " V" << layer << '=' << fingerprint(
                        layers[layer].attention->value_cache->get(),
                        4 * (position + 1) * 256 * sizeof(float));
                }
                std::cout << '\n';
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    record(fingerprint(outputs[layer]->get(), kHidden * sizeof(float)));
                }
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    if (layers[layer].recurrent != nullptr) {
                        record(fingerprint(layers[layer].recurrent->state->get(), state_bytes));
                    }
                }
                record_caches(position, false, record);
            }
            for (std::size_t layer = 0; layer < layer_count; ++layer) {
                if (layers[layer].recurrent != nullptr) layers[layer].recurrent->poison();
                if (layers[layer].attention != nullptr) layers[layer].attention->poison();
            }
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            for (std::size_t layer = 0; layer < layer_count; ++layer) {
                if (layers[layer].recurrent != nullptr) layers[layer].recurrent->reset(fixture);
                if (layers[layer].attention != nullptr) layers[layer].attention->reset();
            }
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            std::size_t replay_index = 0;
            const auto expect_replay = [&](std::uint64_t actual, const char* label) {
                if (replay_index >= replay_fingerprints.size()
                    || replay_fingerprints[replay_index] != actual) {
                    throw std::runtime_error(std::string("poisoned reset replay mismatch: ") + label);
                }
                ++replay_index;
            };
            for (std::size_t position = 0; position <= 64; ++position) {
                const auto host_input = position > 1
                    ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                    : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                if (is_checkpoint_position(position)) {
                    for (std::size_t layer = 0; layer < layer_count; ++layer) {
                        if (layers[layer].recurrent != nullptr) {
                            expect_replay(fingerprint(layers[layer].recurrent->state->get(), state_bytes),
                                          "recurrent entry");
                        }
                    }
                    record_caches(position, true, [&](std::uint64_t value) {
                        expect_replay(value, "attention cache entry");
                    });
                }
                run_prefix(std::span<const GpuLayerRef>(layers).first(layer_count),
                           std::span<float* const>(output_pointers).first(layer_count),
                           static_cast<const float*>(input->get()), position);
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                if (!is_checkpoint_position(position)) continue;
                for (std::size_t layer = 0; layer < layer_count; ++layer) expect_replay(
                    fingerprint(outputs[layer]->get(), kHidden * sizeof(float)), "layer output");
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    if (layers[layer].recurrent != nullptr) {
                        expect_replay(fingerprint(layers[layer].recurrent->state->get(), state_bytes),
                                      "recurrent exit");
                    }
                }
                record_caches(position, false, [&](std::uint64_t value) {
                    expect_replay(value, "attention cache exit");
                });
            }
            if (replay_index != replay_fingerprints.size()) {
                throw std::runtime_error("poisoned reset replay fingerprint count mismatch");
            }
            std::cout << "max_error=" << maximum
                      << " prefix_cpu_ms=" << prefix_cpu_ms
                      << " prefix_cpu_ms_per_position=" << prefix_cpu_ms / 65.0
                      << " allocations_during_decode="
                      << (g_device_allocations - allocations_before_decode)
                      << " device_bytes_after_setup=" << g_device_bytes
                      << " peak_device_bytes=" << g_peak_device_bytes
                      << " dispatches=not-instrumented copies=not-instrumented\n"
                      << "poisoned_reset_replay=PASS\n"
                      << (prefix32 ? "M6-A26 qwen35 thirty-two-layer GPU prefix PASS\n"
                                    : prefix16 ? "M6-A25 qwen35 sixteen-layer GPU prefix PASS\n"
                                                : "M6-A24 qwen35 eight-layer GPU prefix PASS\n");
            return 0;
        }

        Buffer input = allocate(kHidden * sizeof(float));
        Buffer state1 = allocate(kHidden * sizeof(float));
        Buffer state2 = allocate(kHidden * sizeof(float));
        Buffer state3 = allocate(kHidden * sizeof(float));
        Buffer output3 = allocate(kHidden * sizeof(float));
        Buffer state5 = second_block ? allocate(kHidden * sizeof(float)) : nullptr;
        Buffer state6 = second_block ? allocate(kHidden * sizeof(float)) : nullptr;
        Buffer state7 = second_block ? allocate(kHidden * sizeof(float)) : nullptr;
        Buffer output7 = second_block ? allocate(kHidden * sizeof(float)) : nullptr;
        float maximum = 0.0F;
        const auto generated = deep ? read_tokens(fixture / "generated_tokens.txt")
                                    : std::vector<std::uint32_t>{};
        if (deep && generated.size() < 64) throw std::runtime_error("fixture has fewer than 64 tokens");
        const auto is_checkpoint_position = [](std::size_t position) {
            return position == 0 || position == 1 || position == 2 || position == 4
                || position == 8 || position == 16 || position == 32 || position == 64;
        };
        if (second_block) {
            std::cout << "position l4_max l4_rms l4_rel l5_max l5_rms l5_rel "
                         "l6_max l6_rms l6_rel l7_max l7_rms l7_rel state_correct\n";
        } else if (deep) {
            std::cout << "position l0_max l0_rms l0_rel l1_max l1_rms l1_rel "
                         "l2_max l2_rms l2_rel l3_max l3_rms l3_rel state_correct\n";
        }

        const std::size_t last_position = deep ? 64 : 1;
        const std::size_t state_bytes = kVHeads * kState * kState * sizeof(float);
        for (std::size_t position = 0; position <= last_position; ++position) {
            const auto host_input = deep && position > 1
                ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
            upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
            bool state_correct = true;
            if (deep && is_checkpoint_position(position)) {
                const std::array<const RecurrentLayer*, 3> first_layers{
                    &recurrent0, &recurrent1, &recurrent2};
                for (std::size_t layer = 0; layer < first_layers.size(); ++layer) {
                    const auto state_error = detailed_device_error(
                        static_cast<const float*>(first_layers[layer]->state->get()),
                        kVHeads * kState * kState,
                        checkpoint(fixture, position, "state_predelta-" + std::to_string(layer)));
                    state_correct = state_correct && state_error.max_abs <= 5.0e-2F;
                }
                if (second_block) {
                    const std::array<const RecurrentLayer*, 3> second_layers{
                        recurrent4.get(), recurrent5.get(), recurrent6.get()};
                    for (std::size_t offset = 0; offset < second_layers.size(); ++offset) {
                        const auto state_error = detailed_device_error(
                            static_cast<const float*>(second_layers[offset]->state->get()),
                            kVHeads * kState * kState,
                            checkpoint(fixture, position,
                                       "state_predelta-" + std::to_string(4 + offset)));
                        state_correct = state_correct && state_error.max_abs <= 5.0e-2F;
                    }
                }
            }

            run_hybrid_block(recurrent0, recurrent1, recurrent2, attention3,
                             static_cast<const float*>(input->get()), position,
                             static_cast<float*>(state1->get()), static_cast<float*>(state2->get()),
                             static_cast<float*>(state3->get()), static_cast<float*>(output3->get()));
            if (second_block) {
                run_hybrid_block(*recurrent4, *recurrent5, *recurrent6, *attention7,
                                 static_cast<const float*>(output3->get()), position,
                                 static_cast<float*>(state5->get()), static_cast<float*>(state6->get()),
                                 static_cast<float*>(state7->get()), static_cast<float*>(output7->get()));
            }
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            if (!deep || is_checkpoint_position(position)) {
                std::array<DetailedError, 8> errors{};
                const std::array<const float*, 8> outputs{
                    static_cast<const float*>(state1->get()), static_cast<const float*>(state2->get()),
                    static_cast<const float*>(state3->get()), static_cast<const float*>(output3->get()),
                    second_block ? static_cast<const float*>(state5->get()) : nullptr,
                    second_block ? static_cast<const float*>(state6->get()) : nullptr,
                    second_block ? static_cast<const float*>(state7->get()) : nullptr,
                    second_block ? static_cast<const float*>(output7->get()) : nullptr};
                const std::size_t first_layer = second_block ? 4 : 0;
                for (std::size_t offset = 0; offset < 4; ++offset) {
                    const std::size_t layer = first_layer + offset;
                    errors[layer] = detailed_device_error(
                        outputs[layer], kHidden,
                        checkpoint(fixture, position, "l_out-" + std::to_string(layer)));
                    require_match("hybrid layer output", Metrics{errors[layer].max_abs,
                                                                  errors[layer].rms, 0}, 2.0F);
                    maximum = std::max(maximum, errors[layer].max_abs);
                }
                if (second_block) {
                    for (std::size_t layer = 0; layer < 4; ++layer) {
                        const auto error = detailed_device_error(
                            outputs[layer], kHidden,
                            checkpoint(fixture, position, "l_out-" + std::to_string(layer)));
                        require_match("first hybrid layer output", Metrics{error.max_abs, error.rms, 0}, 2.0F);
                        maximum = std::max(maximum, error.max_abs);
                    }
                }
                if (deep && !state_correct) throw std::runtime_error("hybrid recurrent state mismatch");
                if (second_block) {
                    std::cout << position;
                    for (std::size_t layer = 4; layer < 8; ++layer) {
                        std::cout << ' ' << errors[layer].max_abs << ' ' << errors[layer].rms
                                  << ' ' << errors[layer].relative_rms;
                    }
                    std::cout << ' ' << (state_correct ? "PASS" : "FAIL") << '\n';
                    std::cout << "  first_block_output="
                              << fingerprint(output3->get(), kHidden * sizeof(float)) << '\n';
                    std::cout << "  fingerprints state0=" << fingerprint(recurrent0.state->get(), state_bytes)
                              << " state1=" << fingerprint(recurrent1.state->get(), state_bytes)
                              << " state2=" << fingerprint(recurrent2.state->get(), state_bytes)
                              << " state4=" << fingerprint(recurrent4->state->get(), state_bytes)
                              << " state5=" << fingerprint(recurrent5->state->get(), state_bytes)
                              << " state6=" << fingerprint(recurrent6->state->get(), state_bytes)
                              << " K3=" << fingerprint(attention3.key_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float))
                              << " V3=" << fingerprint(attention3.value_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float))
                              << " K7=" << fingerprint(attention7->key_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float))
                              << " V7=" << fingerprint(attention7->value_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float)) << '\n';
                } else if (deep) {
                    std::cout << position;
                    for (std::size_t layer = 0; layer < 4; ++layer) {
                        std::cout << ' ' << errors[layer].max_abs << ' ' << errors[layer].rms
                                  << ' ' << errors[layer].relative_rms;
                    }
                    std::cout << ' ' << (state_correct ? "PASS" : "FAIL") << '\n';
                    std::cout << "  fingerprints state0="
                              << fingerprint(recurrent0.state->get(), state_bytes)
                              << " state1=" << fingerprint(recurrent1.state->get(), state_bytes)
                              << " state2=" << fingerprint(recurrent2.state->get(), state_bytes)
                              << " K=" << fingerprint(attention3.key_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float))
                              << " V=" << fingerprint(attention3.value_cache->get(),
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
                  << (prefix32 ? "M6-A26" : prefix8 ? "M6-A24" : second_block ? "M6-A23" : deep ? "M6-A22" : "M6-A21")
                  << " qwen35 GPU hybrid block PASS\n";
    } catch (const std::exception& error) {
        std::cerr << (prefix32 ? "M6-A26" : prefix8 ? "M6-A24" : second_block ? "M6-A23" : deep ? "M6-A22" : "M6-A21")
                  << " failed: " << error.what() << '\n';
        return 1;
    }
}
