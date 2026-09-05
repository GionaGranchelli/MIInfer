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
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <time.h>
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

float cosine_similarity(std::span<const float> actual, std::span<const float> expected) {
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        dot += static_cast<double>(actual[i]) * expected[i];
        actual_norm += static_cast<double>(actual[i]) * actual[i];
        expected_norm += static_cast<double>(expected[i]) * expected[i];
    }
    return actual_norm == 0.0 || expected_norm == 0.0
        ? 0.0F : static_cast<float>(dot / std::sqrt(actual_norm * expected_norm));
}

std::size_t first_argmax(std::span<const float> values) {
    std::size_t result = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[result]) result = i;
    }
    return result;
}

std::vector<std::size_t> top_indices(std::span<const float> values, std::size_t count) {
    count = std::min(count, values.size());
    std::vector<std::size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    const auto better = [&values](std::size_t a, std::size_t b) {
        return values[a] > values[b] || (values[a] == values[b] && a < b);
    };
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), better);
    indices.resize(count);
    return indices;
}

std::size_t rank_of(std::span<const float> values, std::size_t index) {
    std::size_t rank = 1;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] > values[index] ||
            (values[i] == values[index] && i < index)) ++rank;
    }
    return rank;
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

struct UpdateProvenance {
    std::size_t head = 0;
    std::size_t row = 0;
    std::size_t column = 0;
    float previous = 0.0F;
    float decay = 0.0F;
    float beta = 0.0F;
    float value = 0.0F;
    float key = 0.0F;
    float query = 0.0F;
    float key_dot = 0.0F;
    float decayed = 0.0F;
    float delta = 0.0F;
    float candidate = 0.0F;
    float query_dot = 0.0F;
};

struct RecurrentOperands {
    std::vector<float> previous;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> beta;
    std::vector<float> decay;
};

struct KeyPathCapture {
    std::vector<float> input;
    std::vector<float> normalized;
    std::vector<float> qkv;
    std::vector<float> key;
    std::vector<float> key_norm;
};

struct LayerPathCapture {
    std::vector<float> input;
    std::vector<float> normalized;
    std::vector<float> qkv;
    std::vector<float> recurrent_output;
    std::vector<float> gated;
    std::vector<float> attention_residual;
    std::vector<float> post_normalized;
    std::vector<float> ffn_output;
    std::vector<float> layer_output;
};

struct OutputProjectionPathCapture {
    std::vector<float> gated;
    std::vector<std::byte> q8_input;
    std::vector<float> projected;
    std::vector<float> input;
    std::vector<float> residual;
};

struct GatePathCapture {
    std::vector<float> recurrent_output;
    std::vector<float> head_norm;
    std::vector<float> head_scaled;
    std::vector<float> normalized;
    std::vector<float> gate;
    std::vector<float> gated;
};

LocatedError located_host_error(std::span<const float> actual,
                                 std::span<const float> expected) {
    LocatedError result;
    result.metrics = detailed_compare(actual, expected);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) >
            std::fabs(result.actual - result.expected)) {
            result.index = i;
            result.actual = actual[i];
            result.expected = expected[i];
        }
    }
    return result;
}

std::vector<float> download(const void* device, std::size_t elements) {
    std::vector<float> host(elements);
    MIINFER_HIP_CHECK(hipMemcpy(host.data(), device, elements * sizeof(float),
                                hipMemcpyDeviceToHost));
    return host;
}

std::vector<std::byte> download_bytes(const void* device, std::size_t bytes) {
    std::vector<std::byte> host(bytes);
    MIINFER_HIP_CHECK(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost));
    return host;
}

std::uint64_t host_fingerprint(std::span<const std::byte> bytes) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto byte : bytes) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct ByteMismatch {
    std::size_t count = 0;
    std::size_t first = 0;
};

ByteMismatch compare_bytes(std::span<const std::byte> actual,
                           std::span<const std::byte> expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("byte comparison size mismatch");
    }
    ByteMismatch result;
    result.first = actual.size();
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            if (result.count == 0) result.first = i;
            ++result.count;
        }
    }
    return result;
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
             std::uint32_t rows, std::uint32_t columns,
             bool input_prequantized = false) {
    if (!input_prequantized) {
        miinfer::launch_qwen3_q8_k_quantize(input, q8, columns);
    }
    switch (weight.type) {
    case miinfer::GgufTensorType::q4_k: {
        const char* dot4 = std::getenv("MIINFER_Q4K_DOT4");
        if (dot4 == nullptr || std::strcmp(dot4, "0") != 0) {
            miinfer::launch_qwen3_q4_k_q8_k_gemv_dot4(
                static_cast<const miinfer::Q4KDeviceBlock*>(device_weight->get()), q8,
                output, rows, columns);
        } else {
            miinfer::launch_qwen3_q4_k_q8_k_gemv(
                static_cast<const miinfer::Q4KDeviceBlock*>(device_weight->get()), q8,
                output, rows, columns);
        }
        return;
    }
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

void project_q5_q8_1(const Buffer& device_weight, const float* input,
                     miinfer::Q8_1Block* q8, float* output,
                     std::uint32_t rows, std::uint32_t columns) {
    miinfer::launch_q8_1_quantize_f32(input, q8, columns);
    miinfer::launch_qwen3_q5_k_q8_1_mmvq(
        static_cast<const miinfer::Q5KDeviceBlock*>(device_weight->get()), q8,
        output, rows, columns);
}

void project_q4_q8_1_prequantized(const Buffer& device_weight,
                                  const miinfer::Q8_1Block* q8, float* output,
                                  std::uint32_t rows, std::uint32_t columns) {
    miinfer::launch_qwen3_q4_k_q8_1_mmvq(
        static_cast<const miinfer::Q4KDeviceBlock*>(device_weight->get()), q8,
        output, rows, columns);
}

void project_q4_q8_1(const Buffer& device_weight, const float* input,
                     miinfer::Q8_1Block* q8, float* output,
                     std::uint32_t rows, std::uint32_t columns) {
    miinfer::launch_q8_1_quantize_f32(input, q8, columns);
    project_q4_q8_1_prequantized(device_weight, q8, output, rows, columns);
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
    Buffer ffn_gate, ffn_up, ffn_activation, layer_output, q8, q8_1;
    RecurrentTrace* trace = nullptr;
    UpdateProvenance* provenance = nullptr;
    std::uint32_t provenance_position = 0;
    std::size_t provenance_index = 0;
    RecurrentOperands* operand_capture = nullptr;
    std::uint32_t operand_capture_position = 0;
    KeyPathCapture* key_path_capture = nullptr;
    std::uint32_t key_path_capture_position = 0;
    LayerPathCapture* layer_path_capture = nullptr;
    std::uint32_t layer_path_capture_position = 0;
    OutputProjectionPathCapture* output_projection_path_capture = nullptr;
    std::uint32_t output_projection_path_capture_position = 0;
    GatePathCapture* gate_path_capture = nullptr;
    std::uint32_t gate_path_capture_position = 0;
    struct StageProfile {
        std::array<hipEvent_t, 14> start{};
        std::array<hipEvent_t, 14> end{};
    };
    bool no_decay_store = false;
    StageProfile* stage_profile = nullptr;
    std::uint32_t stage_profile_position = 0;
    bool reuse_projection_q8 = false;
    bool q5_q8_1_mmvq = false;
    bool q4_q8_1_mmvq = false;
    bool q4_q8_1_gate_up = false;

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
        const char* no_decay_store_env = std::getenv("MIINFER_DELTA_NO_DECAY_STORE");
        no_decay_store = no_decay_store_env == nullptr
            || std::strcmp(no_decay_store_env, "0") != 0;
        const char* reuse_projection_q8_env = std::getenv("MIINFER_REUSE_PROJECTION_Q8");
        reuse_projection_q8 = reuse_projection_q8_env == nullptr
            || std::strcmp(reuse_projection_q8_env, "0") != 0;
        const char* q5_q8_1_mmvq_env = std::getenv("MIINFER_Q5K_Q8_1_MMVQ");
        q5_q8_1_mmvq = q5_q8_1_mmvq_env == nullptr
            || std::strcmp(q5_q8_1_mmvq_env, "0") != 0;
        const char* q4_q8_1_mmvq_env = std::getenv("MIINFER_Q4K_Q8_1_MMVQ");
        q4_q8_1_mmvq = q4_q8_1_mmvq_env == nullptr
            || std::strcmp(q4_q8_1_mmvq_env, "0") != 0;
        const char* q4_q8_1_gate_up_env = std::getenv("MIINFER_Q4K_Q8_1_MMVQ_FFN_GATE_UP");
        q4_q8_1_gate_up = q4_q8_1_gate_up_env == nullptr
            || std::strcmp(q4_q8_1_gate_up_env, "0") != 0;
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
        q8_1 = allocate((kFfnInner / miinfer::kQ8_1BlockSize) * sizeof(miinfer::Q8_1Block));

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

    UpdateProvenance sample_update(std::size_t flat_index) const {
        constexpr std::size_t state_size = kState;
        UpdateProvenance result;
        result.head = flat_index / (state_size * state_size);
        result.row = (flat_index / state_size) % state_size;
        result.column = flat_index % state_size;
        const std::size_t state_base =
            (result.head * state_size + result.row) * state_size;
        const std::size_t key_base = (result.head % kKHeads) * state_size;
        std::array<float, state_size> previous{};
        std::array<float, state_size> key_values{};
        std::array<float, state_size> query_values{};
        MIINFER_HIP_CHECK(hipMemcpy(previous.data(),
                                    static_cast<const float*>(state->get()) + state_base,
                                    sizeof(previous), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(key_values.data(),
                                    static_cast<const float*>(key_norm->get()) + key_base,
                                    sizeof(key_values), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(query_values.data(),
                                    static_cast<const float*>(query_norm->get()) + key_base,
                                    sizeof(query_values), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(&result.decay,
                                    static_cast<const float*>(decay->get()) + result.head,
                                    sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(&result.beta,
                                    static_cast<const float*>(beta->get()) + result.head,
                                    sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(&result.value,
                                    static_cast<const float*>(value->get())
                                        + result.head * state_size + result.row,
                                    sizeof(float), hipMemcpyDeviceToHost));
        result.previous = previous[result.column];
        result.key = key_values[result.column];
        result.query = query_values[result.column];
        for (std::size_t column = 0; column < state_size; ++column) {
            const float decayed = previous[column] * result.decay;
            result.key_dot += decayed * key_values[column];
        }
        result.decayed = result.previous * result.decay;
        result.delta = (result.value - result.key_dot) * result.beta;
        result.candidate = result.decayed + result.delta * result.key;
        for (std::size_t column = 0; column < state_size; ++column) {
            result.query_dot +=
                (previous[column] * result.decay + result.delta * key_values[column])
                * query_values[column];
        }
        return result;
    }

    void capture_operands(RecurrentOperands& captured) const {
        captured.previous = download(state->get(), kVHeads * kState * kState);
        captured.query = download(query_norm->get(), kKHeads * kState);
        captured.key = download(key_norm->get(), kKHeads * kState);
        captured.value = download(value->get(), kVHeads * kState);
        captured.beta = download(beta->get(), kVHeads);
        captured.decay = download(decay->get(), kVHeads);
    }

    float state_value(std::size_t flat_index) const {
        float result = 0.0F;
        MIINFER_HIP_CHECK(hipMemcpy(
            &result, static_cast<const float*>(state->get()) + flat_index,
            sizeof(float), hipMemcpyDeviceToHost));
        return result;
    }

    void stage_start(std::size_t stage, std::uint32_t position) const {
        if (stage_profile != nullptr && stage_profile_position == position) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->start[stage], nullptr));
        }
    }

    void stage_end(std::size_t stage, std::uint32_t position) const {
        if (stage_profile != nullptr && stage_profile_position == position) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->end[stage], nullptr));
        }
    }

    void run(const float* input, std::uint32_t position, float* output) {
        stage_start(0, position);
        miinfer::launch_qwen3_rms_norm(
            input, static_cast<const float*>(d_attn_norm->get()),
            static_cast<float*>(normalized->get()), kHidden, model.config().rms_epsilon);
        if (gate_path_capture != nullptr && gate_path_capture_position == position) {
            gate_path_capture->normalized = download(normalized->get(), kHidden);
        }
        trace_tensor(position, "attn_norm", static_cast<const float*>(normalized->get()),
                     kHidden, "attn_norm-" + std::to_string(index));
        stage_end(0, position);
        stage_start(1, position);
        project(qkv_weight, d_qkv, static_cast<const float*>(normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                static_cast<float*>(qkv->get()), kChannels, kHidden);
        trace_tensor(position, "qkv", static_cast<const float*>(qkv->get()), kChannels,
                     "linear_attn_qkv_mixed-" + std::to_string(index));
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->input = download(input, kHidden);
            layer_path_capture->normalized = download(normalized->get(), kHidden);
            layer_path_capture->qkv = download(qkv->get(), kChannels);
        }
        stage_end(1, position);
        stage_start(2, position);
        project(gate_weight, d_gate, static_cast<const float*>(normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                static_cast<float*>(gate->get()), kInner, kHidden,
                reuse_projection_q8);
        if (gate_path_capture != nullptr && gate_path_capture_position == position) {
            gate_path_capture->gate = download(gate->get(), kInner);
        }
        stage_end(2, position);
        stage_start(3, position);
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
        stage_end(3, position);
        stage_start(4, position);
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
        stage_end(4, position);
        if (provenance != nullptr && provenance_position == position) {
            *provenance = sample_update(provenance_index);
        }
        if (operand_capture != nullptr && operand_capture_position == position) {
            capture_operands(*operand_capture);
        }
        if (key_path_capture != nullptr && key_path_capture_position == position) {
            key_path_capture->input = download(input, kHidden);
            key_path_capture->normalized = download(normalized->get(), kHidden);
            key_path_capture->qkv = download(qkv->get(), kChannels);
            key_path_capture->key = download(key->get(), kKHeads * kState);
            key_path_capture->key_norm = download(key_norm->get(), kKHeads * kState);
        }
        stage_start(5, position);
        if (no_decay_store) {
            miinfer::launch_qwen35_deltanet_state_update_no_decay_store(
                static_cast<const float*>(query_norm->get()), static_cast<const float*>(key_norm->get()),
                static_cast<const float*>(value->get()), static_cast<const float*>(beta->get()),
                static_cast<const float*>(decay->get()), static_cast<float*>(state->get()),
                static_cast<float*>(recurrent_output->get()), kKHeads, kVHeads, kState);
        } else {
            miinfer::launch_qwen35_deltanet_state_update(
                static_cast<const float*>(query_norm->get()), static_cast<const float*>(key_norm->get()),
                static_cast<const float*>(value->get()), static_cast<const float*>(beta->get()),
                static_cast<const float*>(decay->get()), static_cast<float*>(state->get()),
                static_cast<float*>(recurrent_output->get()), kKHeads, kVHeads, kState);
        }
        if (trace != nullptr && trace->layer == index && position + 1 == trace->position) {
            trace->report_at(position + 1, "state_after", static_cast<const float*>(state->get()),
                             kVHeads * kState * kState,
                             "state_predelta-" + std::to_string(index));
        }
        stage_end(5, position);
        trace_tensor(position, "recurrent_output",
                     static_cast<const float*>(recurrent_output->get()), kVHeads * kState,
                     "attn_output-" + std::to_string(index));
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->recurrent_output = download(recurrent_output->get(), kVHeads * kState);
        }
        if (gate_path_capture != nullptr && gate_path_capture_position == position) {
            gate_path_capture->recurrent_output = download(recurrent_output->get(), kVHeads * kState);
        }
        stage_start(6, position);
        miinfer::launch_qwen3_head_rms_normalize(
            static_cast<const float*>(recurrent_output->get()), static_cast<float*>(head_norm->get()),
            kVHeads, kState, model.config().rms_epsilon);
        if (gate_path_capture != nullptr && gate_path_capture_position == position) {
            gate_path_capture->head_norm = download(head_norm->get(), kVHeads * kState);
        }
        miinfer::launch_qwen3_head_mul(
            static_cast<const float*>(head_norm->get()), static_cast<const float*>(d_ssm_norm->get()),
            static_cast<float*>(gated->get()), kVHeads, kState);
        if (gate_path_capture != nullptr && gate_path_capture_position == position) {
            gate_path_capture->head_scaled = download(gated->get(), kVHeads * kState);
        }
        miinfer::launch_qwen3_silu_mul(
            static_cast<const float*>(gate->get()), static_cast<const float*>(gated->get()),
            static_cast<float*>(gated->get()), kVHeads * kState);
        stage_end(6, position);
        trace_tensor(position, "gated", static_cast<const float*>(gated->get()), kVHeads * kState,
                     "final_output-" + std::to_string(index));
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->gated = download(gated->get(), kVHeads * kState);
        }
        if (output_projection_path_capture != nullptr
            && output_projection_path_capture_position == position) {
            output_projection_path_capture->gated = download(gated->get(), kVHeads * kState);
        }
        if (gate_path_capture != nullptr && gate_path_capture_position == position) {
            gate_path_capture->gated = download(gated->get(), kVHeads * kState);
        }
        stage_start(7, position);
        if (q5_q8_1_mmvq) {
            project_q5_q8_1(d_ssm_out, static_cast<const float*>(gated->get()),
                            static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                            static_cast<float*>(projected->get()), kHidden, kInner);
        } else {
            project(ssm_out_weight, d_ssm_out, static_cast<const float*>(gated->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(projected->get()), kHidden, kInner);
        }
        if (output_projection_path_capture != nullptr
            && output_projection_path_capture_position == position) {
            output_projection_path_capture->q8_input = download_bytes(
                q5_q8_1_mmvq ? q8_1->get() : q8->get(),
                q5_q8_1_mmvq
                    ? (kInner / miinfer::kQ8_1BlockSize) * sizeof(miinfer::Q8_1Block)
                    : (kInner / 256) * sizeof(miinfer::Q8KDeviceBlock));
            output_projection_path_capture->projected = download(projected->get(), kHidden);
        }
        stage_end(7, position);
        stage_start(8, position);
        miinfer::launch_qwen3_add(
            input, static_cast<const float*>(projected->get()),
            static_cast<float*>(residual->get()), kHidden);
        if (output_projection_path_capture != nullptr
            && output_projection_path_capture_position == position) {
            output_projection_path_capture->input = download(input, kHidden);
            output_projection_path_capture->residual = download(residual->get(), kHidden);
        }
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->attention_residual = download(residual->get(), kHidden);
        }
        trace_tensor(position, "attention_residual", static_cast<const float*>(residual->get()),
                     kHidden, "attn_residual-" + std::to_string(index));
        stage_end(8, position);
        stage_start(9, position);
        miinfer::launch_qwen3_rms_norm(
            static_cast<const float*>(residual->get()), static_cast<const float*>(d_post_norm->get()),
            static_cast<float*>(post_normalized->get()), kHidden, model.config().rms_epsilon);
        trace_tensor(position, "post_attention_norm",
                     static_cast<const float*>(post_normalized->get()), kHidden,
                     "attn_post_norm-" + std::to_string(index));
        stage_end(9, position);
        stage_start(10, position);
        if (q4_q8_1_gate_up
            && ffn_gate_weight.type == miinfer::GgufTensorType::q4_k
            && ffn_up_weight.type == miinfer::GgufTensorType::q4_k) {
            miinfer::launch_q8_1_quantize_f32(
                static_cast<const float*>(post_normalized->get()),
                static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            project_q4_q8_1_prequantized(
                d_ffn_gate, static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden);
            project_q4_q8_1_prequantized(
                d_ffn_up, static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(ffn_up->get()), kFfnInner, kHidden);
        } else {
            project(ffn_gate_weight, d_ffn_gate, static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden);
            project(ffn_up_weight, d_ffn_up, static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(ffn_up->get()), kFfnInner, kHidden,
                    reuse_projection_q8);
        }
        stage_end(10, position);
        stage_start(11, position);
        miinfer::launch_qwen3_silu_mul(
            static_cast<const float*>(ffn_gate->get()), static_cast<const float*>(ffn_up->get()),
            static_cast<float*>(ffn_activation->get()), kFfnInner);
        stage_end(11, position);
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->post_normalized = download(post_normalized->get(), kHidden);
        }
        stage_start(12, position);
        if (q4_q8_1_mmvq && ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
            project_q4_q8_1(d_ffn_down, static_cast<const float*>(ffn_activation->get()),
                            static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                            static_cast<float*>(projected->get()), kHidden, kFfnInner);
        } else {
            project(ffn_down_weight, d_ffn_down, static_cast<const float*>(ffn_activation->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(projected->get()), kHidden, kFfnInner);
        }
        trace_tensor(position, "ffn_out", static_cast<const float*>(projected->get()), kHidden,
                     "ffn_out-" + std::to_string(index));
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->ffn_output = download(projected->get(), kHidden);
        }
        stage_end(12, position);
        stage_start(13, position);
        miinfer::launch_qwen3_add(
            static_cast<const float*>(residual->get()), static_cast<const float*>(projected->get()),
            static_cast<float*>(layer_output->get()), kHidden);
        trace_tensor(position, "layer_output", static_cast<const float*>(layer_output->get()),
                     kHidden, "l_out-" + std::to_string(index));
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->layer_output = download(layer_output->get(), kHidden);
        }
        MIINFER_HIP_CHECK(hipMemcpy(output, layer_output->get(), kHidden * sizeof(float),
                                    hipMemcpyDeviceToDevice));
        stage_end(13, position);
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
    bool reuse_projection_q8 = false;
    Buffer q8_1;
    bool q4_q8_1_mmvq = false;
    bool q4_q8_1_gate_up = false;

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
        const char* reuse_projection_q8_env = std::getenv("MIINFER_REUSE_PROJECTION_Q8");
        reuse_projection_q8 = reuse_projection_q8_env == nullptr
            || std::strcmp(reuse_projection_q8_env, "0") != 0;
        const char* q4_q8_1_mmvq_env = std::getenv("MIINFER_Q4K_Q8_1_MMVQ");
        q4_q8_1_mmvq = q4_q8_1_mmvq_env == nullptr
            || std::strcmp(q4_q8_1_mmvq_env, "0") != 0;
        const char* q4_q8_1_gate_up_env = std::getenv("MIINFER_Q4K_Q8_1_MMVQ_FFN_GATE_UP");
        q4_q8_1_gate_up = q4_q8_1_gate_up_env == nullptr
            || std::strcmp(q4_q8_1_gate_up_env, "0") != 0;
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
        q8_1 = allocate((kFfnInner / miinfer::kQ8_1BlockSize) * sizeof(miinfer::Q8_1Block));
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
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(key->get()),
                1024, kHidden, reuse_projection_q8);
        for (std::uint32_t h = 0; h < 4; ++h) {
            miinfer::launch_qwen3_rms_normalize(static_cast<const float*>(key->get()) + h * 256,
                static_cast<float*>(key_norm->get()) + h * 256, 256, model.config().rms_epsilon);
        }
        miinfer::launch_qwen3_head_mul(static_cast<const float*>(key_norm->get()),
            static_cast<const float*>(d_k_norm->get()), static_cast<float*>(key_norm->get()), 4, 256);
        project(v_weight, d_v, static_cast<const float*>(normalized->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(value->get()),
                1024, kHidden, reuse_projection_q8);
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
        if (q4_q8_1_gate_up
            && ffn_gate_weight.type == miinfer::GgufTensorType::q4_k
            && ffn_up_weight.type == miinfer::GgufTensorType::q4_k) {
            miinfer::launch_q8_1_quantize_f32(
                static_cast<const float*>(post_normalized->get()),
                static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            project_q4_q8_1_prequantized(
                d_ffn_gate, static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden);
            project_q4_q8_1_prequantized(
                d_ffn_up, static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(ffn_up->get()), kFfnInner, kHidden);
        } else {
            project(ffn_gate_weight, d_ffn_gate, static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden);
            project(ffn_up_weight, d_ffn_up, static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(ffn_up->get()), kFfnInner, kHidden,
                    reuse_projection_q8);
        }
        miinfer::launch_qwen3_silu_mul(static_cast<const float*>(ffn_gate->get()),
            static_cast<const float*>(ffn_up->get()), static_cast<float*>(ffn_activation->get()), kFfnInner);
        if (q4_q8_1_mmvq && ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
            project_q4_q8_1(d_ffn_down, static_cast<const float*>(ffn_activation->get()),
                            static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                            static_cast<float*>(projected->get()), kHidden, kFfnInner);
        } else {
            project(ffn_down_weight, d_ffn_down, static_cast<const float*>(ffn_activation->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(projected->get()), kHidden, kFfnInner);
        }
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

RecurrentOperands read_external_operands(const std::filesystem::path& fixture) {
    RecurrentOperands result;
    const auto full_query = read_f32(checkpoint(fixture, 19, "q_in-30"), kVHeads * kState);
    const auto full_key = read_f32(checkpoint(fixture, 19, "k_in-30"), kVHeads * kState);
    result.query.assign(full_query.begin(), full_query.begin() + kKHeads * kState);
    result.key.assign(full_key.begin(), full_key.begin() + kKHeads * kState);
    result.value = read_f32(checkpoint(fixture, 19, "v_in-30"), kVHeads * kState);
    result.beta = read_f32(checkpoint(fixture, 19, "b_in-30"), kVHeads);
    const auto external_gate = read_f32(checkpoint(fixture, 19, "g_in-30"), kVHeads);
    result.decay.resize(kVHeads);
    for (std::size_t i = 0; i < kVHeads; ++i) result.decay[i] = std::exp(external_gate[i]);
    result.previous = read_f32(checkpoint(fixture, 19, "state_predelta-30"),
                               kVHeads * kState * kState);
    return result;
}

std::vector<float> replay_state(const RecurrentOperands& operands) {
    Buffer d_query = allocate(operands.query.size() * sizeof(float));
    Buffer d_key = allocate(operands.key.size() * sizeof(float));
    Buffer d_value = allocate(operands.value.size() * sizeof(float));
    Buffer d_beta = allocate(operands.beta.size() * sizeof(float));
    Buffer d_decay = allocate(operands.decay.size() * sizeof(float));
    Buffer d_state = allocate(operands.previous.size() * sizeof(float));
    Buffer d_output = allocate(kVHeads * kState * sizeof(float));
    upload(operands.query.data(), d_query->get(), operands.query.size() * sizeof(float));
    upload(operands.key.data(), d_key->get(), operands.key.size() * sizeof(float));
    upload(operands.value.data(), d_value->get(), operands.value.size() * sizeof(float));
    upload(operands.beta.data(), d_beta->get(), operands.beta.size() * sizeof(float));
    upload(operands.decay.data(), d_decay->get(), operands.decay.size() * sizeof(float));
    upload(operands.previous.data(), d_state->get(), operands.previous.size() * sizeof(float));
    miinfer::launch_qwen35_deltanet_state_update(
        static_cast<const float*>(d_query->get()), static_cast<const float*>(d_key->get()),
        static_cast<const float*>(d_value->get()), static_cast<const float*>(d_beta->get()),
        static_cast<const float*>(d_decay->get()), static_cast<float*>(d_state->get()),
        static_cast<float*>(d_output->get()), kKHeads, kVHeads, kState);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    return download(d_state->get(), operands.previous.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4 && argc != 5) {
        std::cerr << "usage: miinfer-m6a21-qwen35-gpu-hybrid-block MODEL.gguf FIXTURE_DIR "
                     "[EXTERNAL_OPERAND_FIXTURE] "
                     "[--deep|--block4-7|--prefix8|--prefix16|--prefix32|--prefix32-locate|"
                     "--prefix32-provenance|--prefix32-operand-attribution|"
                     "--prefix32-k-path-attribution|--prefix32-l29-path-attribution|"
                     "--prefix32-l29-gate-attribution|--prefix32-external-contract|"
                     "--prefix64-external-contract|--prefix64-l54-attribution|"
                     "--prefix64-l53-attribution|--prefix64-l53-gated-contract|"
                     "--prefix64-observable-contract|--prefix64-l0-l2-p2-trace|"
                     "--prefix64-l0-p2-output-projection|--generate16|--generate64|"
                     "--generate128|--bench64|--bench128|--profile64]\n";
        return 2;
    }
    const std::string mode = argc >= 4 ? argv[argc - 1] : "";
    const bool prefix8 = mode == "--prefix8";
    const bool prefix16 = mode == "--prefix16";
    const bool locate32 = mode == "--prefix32-locate";
    const bool provenance32 = mode == "--prefix32-provenance";
    const bool operand_attribution32 = mode == "--prefix32-operand-attribution";
    const bool k_path_attribution32 = mode == "--prefix32-k-path-attribution";
    const bool l29_path_attribution32 = mode == "--prefix32-l29-path-attribution";
    const bool l29_gate_attribution32 = mode == "--prefix32-l29-gate-attribution";
    const bool external_contract32 = mode == "--prefix32-external-contract";
    const bool external_contract64 = mode == "--prefix64-external-contract";
    const bool trace64 = mode == "--prefix64-l54-attribution";
    const bool trace53 = mode == "--prefix64-l53-attribution";
    const bool gate53_contract = mode == "--prefix64-l53-gated-contract";
    const bool observable64 = mode == "--prefix64-observable-contract";
    const bool trace012 = mode == "--prefix64-l0-l2-p2-trace";
    const bool trace_l0_output = mode == "--prefix64-l0-p2-output-projection";
    const bool generation = mode == "--generate16" || mode == "--generate64"
        || mode == "--generate128" || mode == "--bench64" || mode == "--bench128";
    const bool benchmark = mode == "--bench64" || mode == "--bench128";
    const bool profile64 = mode == "--profile64";
    const char* lm_mmvq_env = std::getenv("MIINFER_LM_Q8_1_MMVQ");
    const bool lm_mmvq = lm_mmvq_env == nullptr || std::strcmp(lm_mmvq_env, "0") != 0;
    const std::size_t generation_tokens = mode == "--generate16" ? 16
        : mode == "--generate64" || mode == "--bench64" ? 64 : 128;
    const bool prefix32 = mode == "--prefix32" || locate32 || provenance32
        || operand_attribution32 || k_path_attribution32 || l29_path_attribution32
        || l29_gate_attribution32 || external_contract32;
    const bool prefix64 = external_contract64 || trace64 || trace53 || gate53_contract
        || observable64 || trace012 || trace_l0_output || generation || profile64;
    const bool deep = mode == "--deep" || mode == "--block4-7" || prefix8 || prefix16 || prefix32
        || prefix64;
    const bool second_block = mode == "--block4-7" || prefix8 || prefix16 || prefix32 || prefix64;
    if (argc >= 4 && !deep) {
        std::cerr << "unknown option: " << mode << '\n';
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        const auto operand_fixture = argc == 5
            ? std::filesystem::path(argv[3]) : fixture;
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
        if (prefix16 || prefix32 || prefix64) {
            recurrent8 = std::make_unique<RecurrentLayer>(model, 8, fixture);
            recurrent9 = std::make_unique<RecurrentLayer>(model, 9, fixture);
            recurrent10 = std::make_unique<RecurrentLayer>(model, 10, fixture);
            attention11 = std::make_unique<FullAttentionLayer>(model, 11);
            recurrent12 = std::make_unique<RecurrentLayer>(model, 12, fixture);
            recurrent13 = std::make_unique<RecurrentLayer>(model, 13, fixture);
            recurrent14 = std::make_unique<RecurrentLayer>(model, 14, fixture);
            attention15 = std::make_unique<FullAttentionLayer>(model, 15);
        }
        if (prefix32 || prefix64) {
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
        std::array<std::unique_ptr<RecurrentLayer>, 24> recurrent32_plus;
        std::array<std::unique_ptr<FullAttentionLayer>, 8> attention32_plus;
        if (prefix64) {
            for (std::size_t layer = 0; layer < recurrent32_plus.size(); ++layer) {
                const std::size_t model_layer = 32 + layer + layer / 3;
                recurrent32_plus[layer] = std::make_unique<RecurrentLayer>(model, model_layer, fixture);
            }
            for (std::size_t layer = 0; layer < attention32_plus.size(); ++layer) {
                attention32_plus[layer] = std::make_unique<FullAttentionLayer>(model, 35 + layer * 4);
            }
        }

        if (locate32 || provenance32) {
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
            recurrent30->trace = locate32 ? &l30_trace : nullptr;
            const auto allocations_before_decode = g_device_allocations;
            const std::size_t state_bytes = kVHeads * kState * kState * sizeof(float);
            if (provenance32) {
                constexpr std::size_t tracked_index = 86909;
                const std::array<std::size_t, 7> transitions{{3, 7, 31, 59, 60, 62, 63}};
                std::cout << "tracked_state_index=" << tracked_index
                          << " head=5 row=38 column=125\n"
                          << "position max_index max_abs fixed_reference fixed_gpu fixed_abs\n";
                for (std::size_t position = 0; position <= 64; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    UpdateProvenance provenance;
                    const bool report_transition =
                        std::find(transitions.begin(), transitions.end(), position) != transitions.end();
                    const float* current = static_cast<const float*>(input->get());
                    for (std::size_t layer = 0; layer < 30; ++layer) {
                        layers[layer].run(current, position, output_pointers[layer]);
                        current = output_pointers[layer];
                    }
                    recurrent30->provenance = &provenance;
                    recurrent30->provenance_position = static_cast<std::uint32_t>(position);
                    recurrent30->provenance_index = tracked_index;
                    layers[30].run(current, position, output_pointers[30]);
                    layers[31].run(output_pointers[30], position, output_pointers[31]);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                    if (position == 64) continue;
                    const std::size_t next_position = position + 1;
                    const auto state = located_device_error(
                        static_cast<const float*>(recurrent30->state->get()),
                        kVHeads * kState * kState,
                        checkpoint(fixture, next_position, "state_predelta-30"));
                    const auto expected = read_f32(
                        checkpoint(fixture, next_position, "state_predelta-30"),
                        kVHeads * kState * kState);
                    const auto fixed_gpu = recurrent30->state_value(tracked_index);
                    const auto fixed_reference = expected[tracked_index];
                    std::cout << next_position << ' ' << state.index << ' '
                              << state.metrics.max_abs << ' ' << fixed_reference << ' '
                              << fixed_gpu << ' ' << std::fabs(fixed_gpu - fixed_reference) << '\n';
                    if (report_transition) {
                        const auto stored = recurrent30->state_value(tracked_index);
                        const auto expected_next = expected[tracked_index];
                        std::cout << "transition=" << position << "->" << next_position
                                  << " previous=" << provenance.previous
                                  << " decay=" << provenance.decay
                                  << " beta=" << provenance.beta
                                  << " value=" << provenance.value
                                  << " key=" << provenance.key
                                  << " query=" << provenance.query
                                  << " key_dot=" << provenance.key_dot
                                  << " decayed=" << provenance.decayed
                                  << " delta=" << provenance.delta
                                  << " candidate=" << provenance.candidate
                                  << " stored=" << stored
                                  << " reference_next=" << expected_next
                                  << " abs_error=" << std::fabs(stored - expected_next)
                                  << " query_dot=" << provenance.query_dot << '\n';
                    }
                }
                std::cout << "allocations_during_decode="
                          << (g_device_allocations - allocations_before_decode)
                          << " device_bytes_after_setup=" << g_device_bytes
                          << " peak_device_bytes=" << g_peak_device_bytes << '\n'
                          << "M6-A26.2 qwen35 L30 recurrent-update provenance COMPLETE\n";
                return 0;
            }
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

        if (prefix8 || prefix16 || prefix32 || prefix64) {
            const std::size_t layer_count = prefix64 ? 64 : prefix32 ? 32 : prefix16 ? 16 : 8;
            std::array<GpuLayerRef, 64> layers{{
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
            if (prefix64) {
                for (std::size_t layer = 32; layer < 64; ++layer) {
                    const std::size_t tail = layer - 32;
                    if (tail % 4 == 3) {
                        layers[layer] = {nullptr, attention32_plus[tail / 4].get()};
                    } else {
                        layers[layer] = {recurrent32_plus[tail - tail / 4].get(), nullptr};
                    }
                }
            }
            std::array<Buffer, 64> outputs{};
            std::array<float*, 64> output_pointers{};
            for (std::size_t layer = 0; layer < layer_count; ++layer) {
                outputs[layer] = allocate(kHidden * sizeof(float));
                output_pointers[layer] = static_cast<float*>(outputs[layer]->get());
            }
            Buffer input = allocate(kHidden * sizeof(float));
            const auto generated = read_tokens(fixture / "generated_tokens.txt");
            if (generated.size() < 64) throw std::runtime_error("fixture has fewer than 64 tokens");
            Buffer d_final_norm_weight;
            Buffer d_output_weight;
            Buffer final_norm;
            Buffer final_q8;
            Buffer final_q8_1;
            Buffer logits;
            Buffer d_embedding;
            Buffer argmax_token;
            if (observable64 || generation || profile64) {
                const auto& final_norm_weight = tensor(*model.file(), "output_norm.weight");
                const auto& output_weight = tensor(*model.file(), "output.weight");
                require_type(final_norm_weight, {miinfer::GgufTensorType::f32});
                require_type(output_weight, {miinfer::GgufTensorType::q6_k});
                d_final_norm_weight = allocate(final_norm_weight.byte_size);
                d_output_weight = allocate(output_weight.byte_size);
                upload_tensor(final_norm_weight, d_final_norm_weight);
                upload_tensor(output_weight, d_output_weight);
                final_norm = allocate(kHidden * sizeof(float));
                final_q8 = allocate((kHidden / 256) * sizeof(miinfer::Q8KDeviceBlock));
                if (lm_mmvq) {
                    final_q8_1 = allocate((kHidden / 32) * sizeof(miinfer::Q8_1Block));
                }
                logits = allocate(model.config().vocab_size * sizeof(float));
                if (generation || profile64) {
                    const auto& embedding_weight = tensor(*model.file(), "token_embd.weight");
                    require_type(embedding_weight, {miinfer::GgufTensorType::q4_k});
                    d_embedding = allocate(embedding_weight.byte_size);
                    upload_tensor(embedding_weight, d_embedding);
                    argmax_token = allocate(sizeof(std::uint32_t));
                }
            }
            RecurrentTrace l54_trace{fixture, 54, 1};
            if (trace64) recurrent32_plus[17]->trace = &l54_trace;
            LayerPathCapture l54_path;
            if (trace64) {
                recurrent32_plus[17]->layer_path_capture = &l54_path;
                recurrent32_plus[17]->layer_path_capture_position = 1;
            }
            RecurrentTrace l53_trace{fixture, 53, 1};
            if (trace53) recurrent32_plus[16]->trace = &l53_trace;
            LayerPathCapture l53_path;
            if (trace53) {
                recurrent32_plus[16]->layer_path_capture = &l53_path;
                recurrent32_plus[16]->layer_path_capture_position = 1;
            }
            GatePathCapture l53_gate_path;
            if (gate53_contract) {
                recurrent32_plus[16]->gate_path_capture = &l53_gate_path;
                recurrent32_plus[16]->gate_path_capture_position = 1;
            }
            if (profile64) {
                const auto reset_all = [&] {
                    for (const auto& layer : layers) {
                        if (layer.recurrent != nullptr) layer.recurrent->reset(fixture);
                        if (layer.attention != nullptr) layer.attention->reset();
                    }
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                };
                std::array<hipEvent_t, 64> layer_start{};
                std::array<hipEvent_t, 64> layer_end{};
                for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                    MIINFER_HIP_CHECK(hipEventCreate(&layer_start[layer]));
                    MIINFER_HIP_CHECK(hipEventCreate(&layer_end[layer]));
                }
                hipEvent_t token_start = nullptr;
                hipEvent_t token_end = nullptr;
                hipEvent_t final_start = nullptr;
                hipEvent_t final_norm_end = nullptr;
                hipEvent_t final_q8_end = nullptr;
                hipEvent_t final_lm_end = nullptr;
                RecurrentLayer::StageProfile recurrent_profile;
                for (std::size_t stage = 0; stage < recurrent_profile.start.size(); ++stage) {
                    MIINFER_HIP_CHECK(hipEventCreate(&recurrent_profile.start[stage]));
                    MIINFER_HIP_CHECK(hipEventCreate(&recurrent_profile.end[stage]));
                }
                MIINFER_HIP_CHECK(hipEventCreate(&token_start));
                MIINFER_HIP_CHECK(hipEventCreate(&token_end));
                MIINFER_HIP_CHECK(hipEventCreate(&final_start));
                MIINFER_HIP_CHECK(hipEventCreate(&final_norm_end));
                MIINFER_HIP_CHECK(hipEventCreate(&final_q8_end));
                MIINFER_HIP_CHECK(hipEventCreate(&final_lm_end));
                recurrent0.stage_profile = &recurrent_profile;
                recurrent0.stage_profile_position = 63;
                reset_all();
                const auto prompt = read_tokens(fixture / "prompt_tokens.txt");
                if (prompt.size() != 1 || generated.size() < 63) {
                    throw std::runtime_error("profile requires prompt and 63 generated tokens");
                }
                auto token = prompt.front();
                for (std::size_t position = 0; position < 63; ++position) {
                    miinfer::launch_qwen35_q4_k_embedding(
                        static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding->get()),
                        token, model.config().vocab_size, kHidden,
                        static_cast<float*>(input->get()));
                    run_prefix(std::span<const GpuLayerRef>(layers),
                               std::span<float* const>(output_pointers),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                    token = generated[position];
                }
                miinfer::launch_qwen35_q4_k_embedding(
                    static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding->get()),
                    token, model.config().vocab_size, kHidden,
                    static_cast<float*>(input->get()));
                MIINFER_HIP_CHECK(hipEventRecord(token_start, nullptr));
                for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                    MIINFER_HIP_CHECK(hipEventRecord(layer_start[layer], nullptr));
                    layers[layer].run(layer == 0
                                          ? static_cast<const float*>(input->get())
                                          : output_pointers[layer - 1],
                                      63, output_pointers[layer]);
                    MIINFER_HIP_CHECK(hipEventRecord(layer_end[layer], nullptr));
                }
                MIINFER_HIP_CHECK(hipEventRecord(final_start, nullptr));
                miinfer::launch_qwen3_rms_norm(
                    output_pointers[63], static_cast<const float*>(d_final_norm_weight->get()),
                    static_cast<float*>(final_norm->get()), kHidden, model.config().rms_epsilon);
                MIINFER_HIP_CHECK(hipEventRecord(final_norm_end, nullptr));
                if (lm_mmvq) {
                    miinfer::launch_q8_1_quantize_f32(
                        static_cast<const float*>(final_norm->get()),
                        static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden);
                    MIINFER_HIP_CHECK(hipEventRecord(final_q8_end, nullptr));
                    miinfer::launch_qwen3_q6_k_q8_1_mmvq(
                        static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                        static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                        static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                } else {
                    miinfer::launch_qwen3_q8_k_quantize(
                        static_cast<const float*>(final_norm->get()),
                        static_cast<miinfer::Q8KDeviceBlock*>(final_q8->get()), kHidden);
                    MIINFER_HIP_CHECK(hipEventRecord(final_q8_end, nullptr));
                    miinfer::launch_qwen3_q6_k_q8_k_gemv(
                        static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                        static_cast<const miinfer::Q8KDeviceBlock*>(final_q8->get()),
                        static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                }
                MIINFER_HIP_CHECK(hipEventRecord(final_lm_end, nullptr));
                miinfer::launch_qwen3_argmax(
                    static_cast<const float*>(logits->get()),
                    static_cast<std::uint32_t*>(argmax_token->get()), model.config().vocab_size);
                MIINFER_HIP_CHECK(hipEventRecord(token_end, nullptr));
                MIINFER_HIP_CHECK(hipEventSynchronize(token_end));
                float total_ms = 0.0F;
                float final_norm_ms = 0.0F;
                float final_q8_ms = 0.0F;
                float final_lm_ms = 0.0F;
                float final_argmax_ms = 0.0F;
                MIINFER_HIP_CHECK(hipEventElapsedTime(&total_ms, token_start, token_end));
                MIINFER_HIP_CHECK(hipEventElapsedTime(&final_norm_ms, final_start, final_norm_end));
                MIINFER_HIP_CHECK(hipEventElapsedTime(&final_q8_ms, final_norm_end, final_q8_end));
                MIINFER_HIP_CHECK(hipEventElapsedTime(&final_lm_ms, final_q8_end, final_lm_end));
                MIINFER_HIP_CHECK(hipEventElapsedTime(&final_argmax_ms, final_lm_end, token_end));
                std::cout << "profile_position=63 total_gpu_ms=" << total_ms
                          << " final_norm_ms=" << final_norm_ms
                          << " final_q8_ms=" << final_q8_ms
                          << " final_lm_ms=" << final_lm_ms
                          << " final_argmax_ms=" << final_argmax_ms << '\n';
                float layer_sum = 0.0F;
                for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                    float elapsed = 0.0F;
                    MIINFER_HIP_CHECK(hipEventElapsedTime(
                        &elapsed, layer_start[layer], layer_end[layer]));
                    layer_sum += elapsed;
                    std::cout << "layer=" << layer
                              << " kind=" << (layers[layer].recurrent != nullptr ? "recurrent" : "attention")
                              << " gpu_ms=" << elapsed << '\n';
                }
                static constexpr std::array<const char*, 14> stage_names{
                    "attn_norm", "qkv_projection", "gate_projection", "beta_alpha",
                    "conv_and_head_norm", "state_update", "recurrent_gate",
                    "ssm_output_projection", "attention_residual", "ffn_norm",
                    "ffn_gate_up", "ffn_activation", "ffn_down", "ffn_residual"};
                std::cout << "recurrent_layer0_stages\n";
                for (std::size_t stage = 0; stage < stage_names.size(); ++stage) {
                    float elapsed = 0.0F;
                    MIINFER_HIP_CHECK(hipEventElapsedTime(
                        &elapsed, recurrent_profile.start[stage], recurrent_profile.end[stage]));
                    std::cout << "stage=" << stage_names[stage]
                              << " gpu_ms=" << elapsed << '\n';
                }
                std::cout << "layer_sum_gpu_ms=" << layer_sum
                          << " dispatches=unknown_in_native_harness"
                          << " allocations_during_profile=0\n"
                          << "M6-B2 qwen35 native P64 profile PASS\n";
                for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                    MIINFER_HIP_CHECK(hipEventDestroy(layer_start[layer]));
                    MIINFER_HIP_CHECK(hipEventDestroy(layer_end[layer]));
                }
                MIINFER_HIP_CHECK(hipEventDestroy(token_start));
                MIINFER_HIP_CHECK(hipEventDestroy(token_end));
                MIINFER_HIP_CHECK(hipEventDestroy(final_start));
                MIINFER_HIP_CHECK(hipEventDestroy(final_norm_end));
                MIINFER_HIP_CHECK(hipEventDestroy(final_q8_end));
                MIINFER_HIP_CHECK(hipEventDestroy(final_lm_end));
                for (std::size_t stage = 0; stage < recurrent_profile.start.size(); ++stage) {
                    MIINFER_HIP_CHECK(hipEventDestroy(recurrent_profile.start[stage]));
                    MIINFER_HIP_CHECK(hipEventDestroy(recurrent_profile.end[stage]));
                }
                return 0;
            }
            if (generation) {
                const auto prompt = read_tokens(fixture / "prompt_tokens.txt");
                if (prompt.size() != 1 || prompt.front() >= model.config().vocab_size) {
                    throw std::runtime_error("generation requires one valid prompt token");
                }
                const std::size_t state_bytes = kVHeads * kState * kState * sizeof(float);
                const auto reset_all = [&] {
                    for (const auto& layer : layers) {
                        if (layer.recurrent != nullptr) layer.recurrent->reset(fixture);
                        if (layer.attention != nullptr) layer.attention->reset();
                    }
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                };
                struct GenerationResult {
                    std::vector<std::uint32_t> tokens;
                    std::uint64_t state_hash = 0;
                    double decode_ms = 0.0;
                };
                const auto monotonic_ms = [] {
                    timespec timestamp{};
                    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
                        throw std::runtime_error("clock_gettime failed");
                    }
                    return static_cast<double>(timestamp.tv_sec) * 1000.0
                        + static_cast<double>(timestamp.tv_nsec) / 1000000.0;
                };
                const auto run_generation = [&] {
                    std::vector<std::uint32_t> tokens;
                    tokens.reserve(generation_tokens);
                    auto token = prompt.front();
                    const double start = monotonic_ms();
                    for (std::size_t position = 0; position < generation_tokens; ++position) {
                        miinfer::launch_qwen35_q4_k_embedding(
                            static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding->get()),
                            token, model.config().vocab_size, kHidden,
                            static_cast<float*>(input->get()));
                        run_prefix(std::span<const GpuLayerRef>(layers),
                                   std::span<float* const>(output_pointers),
                                   static_cast<const float*>(input->get()), position);
                        miinfer::launch_qwen3_rms_norm(
                            output_pointers[63],
                            static_cast<const float*>(d_final_norm_weight->get()),
                            static_cast<float*>(final_norm->get()), kHidden,
                            model.config().rms_epsilon);
                        if (lm_mmvq) {
                            miinfer::launch_q8_1_quantize_f32(
                                static_cast<const float*>(final_norm->get()),
                                static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden);
                            miinfer::launch_qwen3_q6_k_q8_1_mmvq(
                                static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                                static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                                static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                        } else {
                            miinfer::launch_qwen3_q8_k_quantize(
                                static_cast<const float*>(final_norm->get()),
                                static_cast<miinfer::Q8KDeviceBlock*>(final_q8->get()), kHidden);
                            miinfer::launch_qwen3_q6_k_q8_k_gemv(
                                static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                                static_cast<const miinfer::Q8KDeviceBlock*>(final_q8->get()),
                                static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                        }
                        miinfer::launch_qwen3_argmax(
                            static_cast<const float*>(logits->get()),
                            static_cast<std::uint32_t*>(argmax_token->get()),
                            model.config().vocab_size);
                        MIINFER_HIP_CHECK(hipDeviceSynchronize());
                        std::uint32_t next = 0;
                        MIINFER_HIP_CHECK(hipMemcpy(&next, argmax_token->get(),
                                                    sizeof(next), hipMemcpyDeviceToHost));
                        tokens.push_back(next);
                        token = next;
                    }
                    const double end = monotonic_ms();
                    std::uint64_t state_hash = 0;
                    for (const auto& layer : layers) {
                        if (layer.recurrent != nullptr) {
                            state_hash ^= fingerprint(layer.recurrent->state->get(), state_bytes);
                        } else {
                            state_hash ^= fingerprint(layer.attention->key_cache->get(),
                                4 * kCacheCapacity * 256 * sizeof(float));
                            state_hash ^= fingerprint(layer.attention->value_cache->get(),
                                4 * kCacheCapacity * 256 * sizeof(float));
                        }
                    }
                    return GenerationResult{
                        std::move(tokens), state_hash,
                        end - start};
                };
                const auto allocations_before_generation = g_device_allocations;
                if (benchmark) {
                    reset_all();
                    const auto warmup = run_generation();
                    std::array<GenerationResult, 5> samples{};
                    for (auto& sample : samples) {
                        reset_all();
                        sample = run_generation();
                        if (sample.tokens != warmup.tokens || sample.state_hash != warmup.state_hash) {
                            throw std::runtime_error("benchmark generation replay mismatch");
                        }
                    }
                    std::array<double, 5> times{};
                    for (std::size_t i = 0; i < samples.size(); ++i) {
                        times[i] = samples[i].decode_ms;
                    }
                    std::sort(times.begin(), times.end());
                    const double median_ms = times[times.size() / 2];
                    std::cout << "benchmark_tokens=" << generation_tokens
                              << " warmup_ms=" << warmup.decode_ms
                              << " samples_ms=";
                    for (std::size_t i = 0; i < times.size(); ++i) {
                        if (i != 0) std::cout << ',';
                        std::cout << times[i];
                    }
                    std::cout << " median_ms=" << median_ms
                              << " median_tok_s=" << (1000.0 * generation_tokens / median_ms)
                              << " replay=PASS"
                              << " allocations_during_decode="
                              << (g_device_allocations - allocations_before_generation)
                              << " device_bytes_after_setup=" << g_device_bytes
                              << " peak_device_bytes=" << g_peak_device_bytes << '\n'
                              << "M6-B2 native qwen35 generation benchmark PASS\n";
                    return 0;
                }
                reset_all();
                const auto first = run_generation();
                reset_all();
                const auto second = run_generation();
                if (first.tokens != second.tokens || first.state_hash != second.state_hash) {
                    throw std::runtime_error("native generation replay mismatch");
                }
                const double first_tps = 1000.0 * first.tokens.size() / first.decode_ms;
                const double second_tps = 1000.0 * second.tokens.size() / second.decode_ms;
                std::cout << "generated_tokens=" << first.tokens.size()
                          << " first_token=" << first.tokens.front()
                          << " last_token=" << first.tokens.back()
                          << " replay=PASS"
                          << " first_decode_ms=" << first.decode_ms
                          << " first_tok_s=" << first_tps
                          << " second_decode_ms=" << second.decode_ms
                          << " second_tok_s=" << second_tps
                          << " state_fingerprint=" << first.state_hash
                          << " allocations_during_decode="
                          << (g_device_allocations - allocations_before_generation)
                          << " device_bytes_after_setup=" << g_device_bytes
                          << " peak_device_bytes=" << g_peak_device_bytes << '\n'
                          << "M6-A28 qwen35 native autoregressive GPU generation PASS\n";
                return 0;
            }
            if (trace012) {
                static RecurrentTrace l0_trace{fixture, 0, 2};
                static RecurrentTrace l1_trace{fixture, 1, 2};
                static RecurrentTrace l2_trace{fixture, 2, 2};
                recurrent0.trace = &l0_trace;
                recurrent1.trace = &l1_trace;
                recurrent2.trace = &l2_trace;
                for (std::size_t position = 0; position <= 2; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers).first(layer_count),
                               std::span<float* const>(output_pointers).first(layer_count),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }
                std::cout << "M6-A27.6 qwen35 P2 L0-L2 precision-boundary trace COMPLETE\n";
                return 0;
            }
            if (trace_l0_output) {
                OutputProjectionPathCapture production;
                recurrent0.output_projection_path_capture = &production;
                recurrent0.output_projection_path_capture_position = 2;
                for (std::size_t position = 0; position <= 2; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers).first(layer_count),
                               std::span<float* const>(output_pointers).first(layer_count),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }
                const auto external_input = read_f32(
                    checkpoint(fixture, 2, "model_input_embed"), kHidden);
                const auto external_gated = read_f32(
                    checkpoint(fixture, 2, "final_output-0"), kInner);
                const auto external_residual = read_f32(
                    checkpoint(fixture, 2, "attn_residual-0"), kHidden);
                std::vector<float> external_projected(kHidden);
                for (std::size_t i = 0; i < kHidden; ++i) {
                    external_projected[i] = external_residual[i] - external_input[i];
                }
                const auto report = [](const char* label, std::span<const float> actual,
                                       std::span<const float> expected) {
                    const auto error = located_host_error(actual, expected);
                    std::cout << "stage=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                report("production_gated", production.gated, external_gated);
                report("production_projected", production.projected, external_projected);
                report("production_residual", production.residual, external_residual);
                std::cout << "production_q8k_bytes=" << production.q8_input.size()
                          << " production_q8k_fingerprint="
                          << host_fingerprint(production.q8_input) << '\n';

                Buffer external_gated_device = allocate(external_gated.size() * sizeof(float));
                Buffer external_projected_device = allocate(kHidden * sizeof(float));
                Buffer external_q8_device = allocate(
                    (kInner / 256) * sizeof(miinfer::Q8KDeviceBlock));
                Buffer external_residual_device = allocate(kHidden * sizeof(float));
                upload(external_gated.data(), external_gated_device->get(),
                       external_gated.size() * sizeof(float));
                project(recurrent0.ssm_out_weight, recurrent0.d_ssm_out,
                        static_cast<const float*>(external_gated_device->get()),
                        static_cast<miinfer::Q8KDeviceBlock*>(external_q8_device->get()),
                        static_cast<float*>(external_projected_device->get()), kHidden, kInner);
                upload(external_input.data(), external_residual_device->get(),
                       external_input.size() * sizeof(float));
                miinfer::launch_qwen3_add(
                    static_cast<const float*>(external_residual_device->get()),
                    static_cast<const float*>(external_projected_device->get()),
                    static_cast<float*>(external_residual_device->get()), kHidden);
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                const auto replay_projected = download(external_projected_device->get(), kHidden);
                const auto replay_residual = download(external_residual_device->get(), kHidden);
                report("external_gated_replay_projected", replay_projected, external_projected);
                report("external_gated_replay_residual", replay_residual, external_residual);
                const auto replay_q8 = download_bytes(
                    external_q8_device->get(), (kInner / 256) * sizeof(miinfer::Q8KDeviceBlock));
                std::cout << "external_gated_replay_q8k_bytes=" << replay_q8.size()
                          << " external_gated_replay_q8k_fingerprint="
                          << host_fingerprint(replay_q8) << '\n';
                const auto reference_q8 = quantize_q8(external_gated);
                const auto reference_q8_bytes = std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(reference_q8.data()),
                    reference_q8.size() * sizeof(Q8K));
                const auto q8_mismatch = compare_bytes(replay_q8, reference_q8_bytes);
                std::cout << "llama_cpu_q8k_reference_bytes=" << reference_q8_bytes.size()
                          << " llama_cpu_q8k_reference_fingerprint="
                          << host_fingerprint(reference_q8_bytes)
                          << " q8k_mismatch_bytes=" << q8_mismatch.count
                          << " q8k_first_mismatch=" << q8_mismatch.first << '\n';
                const std::size_t row = located_host_error(replay_projected, external_projected).index;
                const std::size_t blocks = kInner / 256;
                std::vector<Q8K> host_q8(blocks);
                std::memcpy(host_q8.data(), replay_q8.data(), replay_q8.size());
                const auto* host_weights = reinterpret_cast<const Q5K*>(recurrent0.ssm_out_weight.data)
                                           + row * blocks;
                std::vector<float> host_contributions(blocks);
                for (std::size_t block = 0; block < blocks; ++block) {
                    std::array<std::int8_t, 256> q5{};
                    std::uint8_t high_bit = 1;
                    for (std::size_t group_pair = 0; group_pair < 4; ++group_pair) {
                        const std::size_t q_offset = group_pair * 64;
                        for (std::size_t index = 0; index < 32; ++index) {
                            q5[q_offset + index] = static_cast<std::int8_t>(
                                (host_weights[block].qs[group_pair * 32 + index] & 0x0fU)
                                + ((host_weights[block].qh[index] & high_bit) != 0 ? 16 : 0));
                            q5[q_offset + 32 + index] = static_cast<std::int8_t>(
                                (host_weights[block].qs[group_pair * 32 + index] >> 4U)
                                + ((host_weights[block].qh[index] & (high_bit << 1U)) != 0 ? 16 : 0));
                        }
                        high_bit = static_cast<std::uint8_t>(high_bit << 2U);
                    }
                    std::array<std::uint8_t, 8> scales{};
                    std::array<std::uint8_t, 8> minimums{};
                    for (std::size_t group = 0; group < 8; ++group) {
                        scale_min(host_weights[block].scales, group, scales[group], minimums[group]);
                    }
                    std::array<std::int32_t, 8> partials{};
                    for (std::size_t index = 0; index < 256; ++index) {
                        partials[index % 8] += static_cast<std::int32_t>(scales[index / 32])
                            * static_cast<std::int32_t>(q5[index])
                            * static_cast<std::int32_t>(host_q8[block].qs[index]);
                    }
                    int sumi = 0;
                    for (std::size_t group = 0; group < 16; ++group) {
                        sumi += host_q8[block].bsums[group] * minimums[group / 2];
                    }
                    const float d = miinfer::fp16_bits_to_float(host_weights[block].d)
                                    * host_q8[block].d;
                    const float dmin = miinfer::fp16_bits_to_float(host_weights[block].dmin)
                                       * host_q8[block].d;
                    host_contributions[block] = -dmin * static_cast<float>(sumi);
                    for (const auto partial : partials) {
                        host_contributions[block] += d * static_cast<float>(partial);
                    }
                }
                Buffer block_contributions_device = allocate(blocks * sizeof(float));
                const auto* device_weights = static_cast<const miinfer::Q5KDeviceBlock*>(
                    recurrent0.d_ssm_out->get()) + row * blocks;
                const auto* device_q8 = static_cast<const miinfer::Q8KDeviceBlock*>(
                    external_q8_device->get());
                for (std::size_t block = 0; block < blocks; ++block) {
                    miinfer::launch_qwen3_q5_k_q8_k_gemv(
                        device_weights + block, device_q8 + block,
                        static_cast<float*>(block_contributions_device->get()) + block,
                        1, 256);
                }
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                const auto gpu_contributions = download(block_contributions_device->get(), blocks);
                std::cout << "q5k_row=" << row << " q5k_blocks=" << blocks << '\n';
                for (std::size_t block = 0; block < blocks; ++block) {
                    std::cout << "q5k_block=" << block
                              << " host=" << host_contributions[block]
                              << " gpu=" << gpu_contributions[block]
                              << " abs_error=" << std::fabs(
                                  gpu_contributions[block] - host_contributions[block]) << '\n';
                }
                const auto host_total = std::accumulate(
                    host_contributions.begin(), host_contributions.end(), 0.0F);
                const auto gpu_total = std::accumulate(
                    gpu_contributions.begin(), gpu_contributions.end(), 0.0F);
                std::cout << "q5k_host_block_sum=" << host_total
                          << " q5k_gpu_block_sum=" << gpu_total
                          << " q5k_block_sum_abs_error=" << std::fabs(gpu_total - host_total)
                          << " external_row=" << external_projected[row]
                          << " production_row=" << replay_projected[row] << '\n';
                std::cout << "M6-A27.9 qwen35 L0 P2 Q5_K block contract COMPLETE\n";
                return 0;
            }
            if (l29_path_attribution32) {
                if (argc != 5) {
                    throw std::runtime_error("L29 path attribution requires an external fixture");
                }
                LayerPathCapture production;
                recurrent29->layer_path_capture = &production;
                recurrent29->layer_path_capture_position = 19;
                for (std::size_t position = 0; position <= 19; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers),
                               std::span<float* const>(output_pointers),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }
                const auto read = [&](const char* name, std::size_t elements) {
                    return read_f32(checkpoint(operand_fixture, 19, name), elements);
                };
                const auto report = [](const char* label, std::span<const float> actual,
                                       std::span<const float> expected) {
                    const auto error = located_host_error(actual, expected);
                    std::cout << "stage=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                std::cout << "M6-A26.6 L29 P19 output provenance\n";
                report("layer_input", production.input, read("l_out-28", kHidden));
                report("attn_norm", production.normalized, read("attn_norm-29", kHidden));
                report("qkv_projection", production.qkv, read("linear_attn_qkv_mixed-29", kChannels));
                report("recurrent_output", production.recurrent_output,
                       read("attn_output-29", kVHeads * kState));
                report("gated", production.gated, read("final_output-29", kVHeads * kState));
                report("attention_residual", production.attention_residual,
                       read("attn_residual-29", kHidden));
                report("post_normalized", production.post_normalized,
                       read("attn_post_norm-29", kHidden));
                report("ffn_output", production.ffn_output, read("ffn_out-29", kHidden));
                report("layer_output", production.layer_output, read("l_out-29", kHidden));
                std::cout << "M6-A26.6 qwen35 L29 output provenance COMPLETE\n";
                return 0;
            }
            if (l29_gate_attribution32) {
                if (argc != 5) {
                    throw std::runtime_error("gate attribution requires an external fixture");
                }
                GatePathCapture production;
                recurrent29->gate_path_capture = &production;
                recurrent29->gate_path_capture_position = 19;
                for (std::size_t position = 0; position <= 19; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers),
                               std::span<float* const>(output_pointers),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }

                const auto external_recurrent = read_f32(
                    checkpoint(operand_fixture, 19, "attn_output-29"), kVHeads * kState);
                const auto external_norm = read_f32(
                    checkpoint(operand_fixture, 19, "attn_norm-29"), kHidden);
                const auto external_gate = read_f32(
                    checkpoint(operand_fixture, 19, "z-29"), kInner);
                const auto external_gated = read_f32(
                    checkpoint(operand_fixture, 19, "final_output-29"), kVHeads * kState);
                const auto* ssm_norm = reinterpret_cast<const float*>(recurrent29->ssm_norm_weight.data);
                std::vector<float> external_head_norm(kVHeads * kState);
                std::vector<float> external_head_scaled(kVHeads * kState);
                std::vector<float> external_gate_silu(kVHeads * kState);
                for (std::size_t head = 0; head < kVHeads; ++head) {
                    const std::size_t base = head * kState;
                    std::array<float, kState> partials{};
                    for (std::size_t i = 0; i < kState; ++i) {
                        const float value = external_recurrent[base + i];
                        partials[i] = value * value;
                    }
                    for (std::size_t stride = kState / 2; stride > 0; stride /= 2) {
                        for (std::size_t i = 0; i < stride; ++i) {
                            partials[i] += partials[i + stride];
                        }
                    }
                    const float inverse_rms = 1.0F / std::sqrt(
                        partials[0] / static_cast<float>(kState) + model.config().rms_epsilon);
                    for (std::size_t i = 0; i < kState; ++i) {
                        const std::size_t index = base + i;
                        external_head_norm[index] = external_recurrent[index] * inverse_rms;
                        external_head_scaled[index] = external_head_norm[index] * ssm_norm[i];
                        external_gate_silu[index] = external_head_scaled[index] == 0.0F
                            ? 0.0F : external_gated[index] / external_head_scaled[index];
                    }
                }
                const auto silu = [](float value) {
                    return value / (1.0F + std::exp(-value));
                };
                std::vector<float> production_gate_silu(production.gate.size());
                std::transform(production.gate.begin(), production.gate.end(),
                               production_gate_silu.begin(), silu);
                const auto replay_gated = [&](std::span<const float> recurrent,
                                              std::span<const float> gate) {
                    Buffer d_recurrent = allocate(recurrent.size() * sizeof(float));
                    Buffer d_gate = allocate(gate.size() * sizeof(float));
                    Buffer d_head_norm = allocate(recurrent.size() * sizeof(float));
                    Buffer d_gated = allocate(recurrent.size() * sizeof(float));
                    upload(recurrent.data(), d_recurrent->get(), recurrent.size() * sizeof(float));
                    upload(gate.data(), d_gate->get(), gate.size() * sizeof(float));
                    miinfer::launch_qwen3_head_rms_normalize(
                        static_cast<const float*>(d_recurrent->get()),
                        static_cast<float*>(d_head_norm->get()), kVHeads, kState,
                        model.config().rms_epsilon);
                    miinfer::launch_qwen3_head_mul(
                        static_cast<const float*>(d_head_norm->get()),
                        static_cast<const float*>(recurrent29->d_ssm_norm->get()),
                        static_cast<float*>(d_gated->get()), kVHeads, kState);
                    miinfer::launch_qwen3_silu_mul(
                        static_cast<const float*>(d_gate->get()),
                        static_cast<const float*>(d_gated->get()),
                        static_cast<float*>(d_gated->get()), kInner);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                    return download(d_gated->get(), kVHeads * kState);
                };
                const auto report_variant = [&](const char* label,
                                                std::span<const float> recurrent,
                                                std::span<const float> gate) {
                    const auto actual = replay_gated(recurrent, gate);
                    const auto error = located_host_error(actual, external_gated);
                    std::cout << "substitution=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                const auto replay_gate_projection = [&](std::span<const float> normalized_input) {
                    Buffer d_input = allocate(normalized_input.size() * sizeof(float));
                    Buffer d_output = allocate(kInner * sizeof(float));
                    upload(normalized_input.data(), d_input->get(),
                           normalized_input.size() * sizeof(float));
                    project(recurrent29->gate_weight, recurrent29->d_gate,
                            static_cast<const float*>(d_input->get()),
                            static_cast<miinfer::Q8KDeviceBlock*>(recurrent29->q8->get()),
                            static_cast<float*>(d_output->get()), kInner, kHidden);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                    return download(d_output->get(), kInner);
                };
                const auto report_gate_input_variant = [&](const char* label,
                                                           std::span<const float> normalized_input) {
                    const auto actual = replay_gate_projection(normalized_input);
                    const auto error = located_host_error(actual, external_gate);
                    std::cout << "gate_input=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                const auto report = [](const char* label, std::span<const float> actual,
                                       std::span<const float> expected) {
                    const auto error = located_host_error(actual, expected);
                    std::cout << "stage=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                std::cout << "M6-A26.8 L29 gate-input provenance\n";
                report_gate_input_variant("production_norm", production.normalized);
                report_gate_input_variant("external_norm", external_norm);
                std::cout << "M6-A26.7 L29 P19 gated-output provenance\n";
                report("recurrent_output", production.recurrent_output, external_recurrent);
                report("head_norm", production.head_norm, external_head_norm);
                report("head_scaled", production.head_scaled, external_head_scaled);
                report("gate_projection", production.gate, external_gate);
                report("gate_silu", production_gate_silu, external_gate_silu);
                report("gated", production.gated, external_gated);
                report_variant("production", production.recurrent_output, production.gate);
                report_variant("external_recurrent", external_recurrent, production.gate);
                report_variant("external_gate", production.recurrent_output, external_gate);
                report_variant("external_both", external_recurrent, external_gate);
                std::cout << "M6-A26.8 qwen35 L29 gate-input provenance COMPLETE\n";
                return 0;
            }
            if (k_path_attribution32) {
                if (argc != 5) {
                    throw std::runtime_error("K-path attribution requires an external fixture");
                }
                KeyPathCapture production;
                recurrent30->key_path_capture = &production;
                recurrent30->key_path_capture_position = 19;
                for (std::size_t position = 0; position <= 19; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers),
                               std::span<float* const>(output_pointers),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }

                const auto external_input = read_f32(
                    checkpoint(operand_fixture, 19, "l_out-29"), kHidden);
                const auto external_norm = read_f32(
                    checkpoint(operand_fixture, 19, "attn_norm-30"), kHidden);
                const auto external_qkv = read_f32(
                    checkpoint(operand_fixture, 19, "linear_attn_qkv_mixed-30"), kChannels);
                const auto external_conv = read_f32(
                    checkpoint(operand_fixture, 19, "conv_output_raw-30"), kChannels);
                const auto external_key_full = read_f32(
                    checkpoint(operand_fixture, 19, "k_in-30"), kVHeads * kState);
                std::vector<float> external_key(kKHeads * kState);
                std::vector<float> external_key_norm(kKHeads * kState);
                for (std::size_t i = 0; i < external_key.size(); ++i) {
                    const float raw = external_conv[kKHeads * kState + i];
                    external_key[i] = raw / (1.0F + std::exp(-raw));
                    external_key_norm[i] = external_key_full[i];
                }
                const auto report = [](const char* label, std::span<const float> actual,
                                       std::span<const float> expected) {
                    const auto error = located_host_error(actual, expected);
                    std::cout << "stage=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                std::cout << "M6-A26.5 L30 P19 K-path provenance\n";
                report("layer_input", production.input, external_input);
                report("attn_norm", production.normalized, external_norm);
                report("qkv_projection", production.qkv, external_qkv);
                report("key_after_conv_silu", production.key, external_key);
                report("key_after_l2_norm", production.key_norm, external_key_norm);
                std::cout << "M6-A26.5 qwen35 L30 K-path provenance COMPLETE\n";
                return 0;
            }
            if (operand_attribution32) {
                if (argc != 5) {
                    throw std::runtime_error(
                        "operand attribution requires an external operand fixture");
                }
                RecurrentOperands production;
                recurrent30->operand_capture = &production;
                recurrent30->operand_capture_position = 19;
                for (std::size_t position = 0; position <= 19; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers),
                               std::span<float* const>(output_pointers),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }

                const auto external = read_external_operands(operand_fixture);
                const auto report_operand = [](const char* label,
                                               std::span<const float> actual,
                                               std::span<const float> expected) {
                    const auto error = located_host_error(actual, expected);
                    std::cout << "operand=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                std::cout << "M6-A26.4 L30 P19->P20 production operand attribution\n"
                          << "q/k compare first " << kKHeads
                          << " heads of the external " << kVHeads << "-head fixture\n";
                report_operand("previous_state", production.previous, external.previous);
                report_operand("q_in", production.query, external.query);
                report_operand("k_in", production.key, external.key);
                report_operand("v_in", production.value, external.value);
                report_operand("beta", production.beta, external.beta);
                report_operand("g_in_as_decay", production.decay, external.decay);

                const auto expected = read_f32(
                    checkpoint(operand_fixture, 20, "state_predelta-30"),
                    kVHeads * kState * kState);
                const auto report_variant = [&](const char* label,
                                                const std::vector<float>* previous,
                                                const std::vector<float>* query,
                                                const std::vector<float>* key,
                                                const std::vector<float>* value,
                                                const std::vector<float>* beta,
                                                const std::vector<float>* decay) {
                    RecurrentOperands operands = production;
                    if (previous != nullptr) operands.previous = *previous;
                    if (query != nullptr) operands.query = *query;
                    if (key != nullptr) operands.key = *key;
                    if (value != nullptr) operands.value = *value;
                    if (beta != nullptr) operands.beta = *beta;
                    if (decay != nullptr) operands.decay = *decay;
                    const auto state = replay_state(operands);
                    const auto error = located_host_error(state, expected);
                    std::cout << "substitution=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                report_variant("production", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                report_variant("previous_state", &external.previous, nullptr, nullptr, nullptr, nullptr, nullptr);
                report_variant("q_in", nullptr, &external.query, nullptr, nullptr, nullptr, nullptr);
                report_variant("k_in", nullptr, nullptr, &external.key, nullptr, nullptr, nullptr);
                report_variant("v_in", nullptr, nullptr, nullptr, &external.value, nullptr, nullptr);
                report_variant("beta", nullptr, nullptr, nullptr, nullptr, &external.beta, nullptr);
                report_variant("g_in_as_decay", nullptr, nullptr, nullptr, nullptr, nullptr, &external.decay);
                std::cout << "M6-A26.4 qwen35 L30 production operand attribution COMPLETE\n";
                return 0;
            }
            const auto is_checkpoint_position = [](std::size_t position) {
                return position == 0 || position == 1 || position == 2 || position == 4
                    || position == 8 || position == 16 || position == 32 || position == 64;
            };
            const auto is_observable_position = [&is_checkpoint_position](std::size_t position) {
                return is_checkpoint_position(position) || position == 12;
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
                if (observable64) {
                    miinfer::launch_qwen3_rms_norm(
                        static_cast<const float*>(outputs[63]->get()),
                        static_cast<const float*>(d_final_norm_weight->get()),
                        static_cast<float*>(final_norm->get()), kHidden,
                        model.config().rms_epsilon);
                    if (lm_mmvq) {
                        miinfer::launch_q8_1_quantize_f32(
                            static_cast<const float*>(final_norm->get()),
                            static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden);
                        miinfer::launch_qwen3_q6_k_q8_1_mmvq(
                            static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                            static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                            static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                    } else {
                        miinfer::launch_qwen3_q8_k_quantize(
                            static_cast<const float*>(final_norm->get()),
                            static_cast<miinfer::Q8KDeviceBlock*>(final_q8->get()), kHidden);
                        miinfer::launch_qwen3_q6_k_q8_k_gemv(
                            static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                            static_cast<const miinfer::Q8KDeviceBlock*>(final_q8->get()),
                            static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                    }
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                    const auto final_hidden_host = download(outputs[63]->get(), kHidden);
                    const auto final_norm_host = download(final_norm->get(), kHidden);
                    const auto logits_host = download(logits->get(), model.config().vocab_size);
                    const auto gpu_token = first_argmax(logits_host);
                    if (position < generated.size()) {
                        std::cout << "teacher_forced position=" << position
                                  << " reference_argmax=" << generated[position]
                                  << " gpu_argmax=" << gpu_token
                                  << " match=" << (gpu_token == generated[position] ? "PASS" : "FAIL")
                                  << '\n';
                    }
                    if (is_observable_position(position)) {
                        const auto expected_hidden = read_f32(
                            checkpoint(fixture, position, "l_out-63"), kHidden);
                        const auto expected_norm = read_f32(
                            checkpoint(fixture, position, "result_norm"), kHidden);
                        const auto expected_logits = read_f32(
                            fixture / "logits" / ("logits-" + std::to_string(position) + ".f32"),
                            model.config().vocab_size);
                        const auto hidden_error = detailed_compare(final_hidden_host, expected_hidden);
                        const auto norm_error = detailed_compare(final_norm_host, expected_norm);
                        const auto logits_error = detailed_compare(logits_host, expected_logits);
                        std::cout << "observable position=" << position
                                  << " final_hidden_max_abs=" << hidden_error.max_abs
                                  << " final_hidden_rms=" << hidden_error.rms
                                  << " final_hidden_relative_rms=" << hidden_error.relative_rms
                                  << " final_hidden_cosine="
                                  << cosine_similarity(final_hidden_host, expected_hidden)
                                  << " final_norm_max_abs=" << norm_error.max_abs
                                  << " final_norm_rms=" << norm_error.rms
                                  << " final_norm_relative_rms=" << norm_error.relative_rms
                                  << " final_norm_cosine="
                                  << cosine_similarity(final_norm_host, expected_norm)
                                  << " logits_max_abs=" << logits_error.max_abs
                                  << " logits_rms=" << logits_error.rms
                                  << " logits_relative_rms=" << logits_error.relative_rms
                                  << " logits_cosine="
                                  << cosine_similarity(logits_host, expected_logits);
                        const auto reference_top = top_indices(expected_logits, 5);
                        const auto gpu_top = top_indices(logits_host, 5);
                        std::vector<std::size_t> top_union = reference_top;
                        for (const auto index : gpu_top) {
                            if (std::find(top_union.begin(), top_union.end(), index) == top_union.end()) {
                                top_union.push_back(index);
                            }
                        }
                        float epsilon_top = 0.0F;
                        for (const auto index : top_union) {
                            epsilon_top = std::max(epsilon_top,
                                                   std::fabs(logits_host[index] - expected_logits[index]));
                        }
                        std::size_t overlap = 0;
                        for (const auto index : reference_top) {
                            if (std::find(gpu_top.begin(), gpu_top.end(), index) != gpu_top.end()) ++overlap;
                        }
                        const auto reference_token = first_argmax(expected_logits);
                        const auto gpu_winner = first_argmax(logits_host);
                        const auto gpu_top_two = top_indices(logits_host, 2);
                        const auto reference_top_two = top_indices(expected_logits, 2);
                        std::cout << " reference_argmax=" << reference_token
                                  << " gpu_argmax=" << gpu_winner
                                  << " top5_overlap=" << overlap
                                  << " reference_winner_rank_on_gpu="
                                  << rank_of(logits_host, reference_token)
                                  << " gpu_winner_rank_on_reference="
                                  << rank_of(expected_logits, gpu_winner)
                                  << " reference_margin="
                                  << expected_logits[reference_top_two[0]] - expected_logits[reference_top_two[1]]
                                  << " epsilon_top=" << epsilon_top
                                  << " margin_robust="
                                  << (expected_logits[reference_top_two[0]] - expected_logits[reference_top_two[1]]
                                              > 2.0F * epsilon_top
                                          ? "YES"
                                          : "NO")
                                  << " gpu_margin="
                                  << logits_host[gpu_top_two[0]] - logits_host[gpu_top_two[1]]
                                  << " gpu_minus_reference_at_reference_winner="
                                  << logits_host[reference_token] - expected_logits[reference_token]
                                  << '\n';
                        std::cout << "reference_top5=";
                        for (const auto index : reference_top) std::cout << index << ',';
                        std::cout << " gpu_top5=";
                        for (const auto index : gpu_top) std::cout << index << ',';
                        std::cout << '\n';
                    }
                }
                prefix_cpu_ms += 1000.0 * static_cast<double>(std::clock() - run_start)
                    / static_cast<double>(CLOCKS_PER_SEC);
                if (!is_checkpoint_position(position)) continue;
                std::array<DetailedError, 64> errors{};
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    errors[layer] = detailed_device_error(
                        static_cast<const float*>(outputs[layer]->get()), kHidden,
                        checkpoint(fixture, position, "l_out-" + std::to_string(layer)));
                    if (external_contract64 && errors[layer].max_abs > 2.0F) {
                        std::cerr << "prefix64 external output mismatch position=" << position
                                  << " layer=" << layer
                                  << " max_abs=" << errors[layer].max_abs
                                  << " rms=" << errors[layer].rms
                                  << " relative_rms=" << errors[layer].relative_rms << '\n';
                    }
                    if (!observable64 &&
                        !((trace64 || trace53 || gate53_contract) && position == 1)) {
                        require_match("prefix output",
                                      Metrics{errors[layer].max_abs, errors[layer].rms, 0}, 2.0F);
                    }
                    maximum = std::max(maximum, errors[layer].max_abs);
                }
                if (!state_correct && !external_contract64 && !trace64 && !trace53 &&
                    !gate53_contract && !observable64) {
                    throw std::runtime_error("prefix recurrent state mismatch");
                }
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
                if (trace64 && position == 1) {
                    const auto read = [&](const char* name, std::size_t elements) {
                        return read_f32(checkpoint(fixture, 1, name), elements);
                    };
                    const auto report = [](const char* label, std::span<const float> actual,
                                           std::span<const float> expected) {
                        const auto error = located_host_error(actual, expected);
                        std::cout << "path label=" << label
                                  << " max_abs=" << error.metrics.max_abs
                                  << " mean_abs=" << error.metrics.mean_abs
                                  << " rms=" << error.metrics.rms
                                  << " relative_rms=" << error.metrics.relative_rms
                                  << " max_index=" << error.index
                                  << " reference=" << error.expected
                                  << " gpu=" << error.actual << '\n';
                    };
                    std::cout << "M6-A27.1 L54 P1 input/path attribution\n";
                    report("layer_input", l54_path.input, read("l_out-53", kHidden));
                    report("attn_norm", l54_path.normalized, read("attn_norm-54", kHidden));
                    report("qkv_projection", l54_path.qkv,
                           read("linear_attn_qkv_mixed-54", kChannels));
                    report("recurrent_output", l54_path.recurrent_output,
                           read("attn_output-54", kVHeads * kState));
                    report("gated", l54_path.gated, read("final_output-54", kVHeads * kState));
                    report("attention_residual", l54_path.attention_residual,
                           read("attn_residual-54", kHidden));
                    report("post_normalized", l54_path.post_normalized,
                           read("attn_post_norm-54", kHidden));
                    report("ffn_output", l54_path.ffn_output, read("ffn_out-54", kHidden));
                    report("layer_output", l54_path.layer_output, read("l_out-54", kHidden));
                    std::cout << "M6-A27.1 qwen35 L54 P1 attribution COMPLETE\n";
                    return 0;
                }
                if (trace53 && position == 1) {
                    const auto read = [&](const char* name, std::size_t elements) {
                        return read_f32(checkpoint(fixture, 1, name), elements);
                    };
                    const auto report = [](const char* label, std::span<const float> actual,
                                           std::span<const float> expected) {
                        const auto error = located_host_error(actual, expected);
                        std::cout << "path label=" << label
                                  << " max_abs=" << error.metrics.max_abs
                                  << " mean_abs=" << error.metrics.mean_abs
                                  << " rms=" << error.metrics.rms
                                  << " relative_rms=" << error.metrics.relative_rms
                                  << " max_index=" << error.index
                                  << " reference=" << error.expected
                                  << " gpu=" << error.actual << '\n';
                    };
                    std::cout << "M6-A27.2 L53 P1 output provenance\n";
                    report("layer_input", l53_path.input, read("l_out-52", kHidden));
                    report("attn_norm", l53_path.normalized, read("attn_norm-53", kHidden));
                    report("qkv_projection", l53_path.qkv,
                           read("linear_attn_qkv_mixed-53", kChannels));
                    report("recurrent_output", l53_path.recurrent_output,
                           read("attn_output-53", kVHeads * kState));
                    report("gated", l53_path.gated, read("final_output-53", kVHeads * kState));
                    report("attention_residual", l53_path.attention_residual,
                           read("attn_residual-53", kHidden));
                    report("post_normalized", l53_path.post_normalized,
                           read("attn_post_norm-53", kHidden));
                    report("ffn_output", l53_path.ffn_output, read("ffn_out-53", kHidden));
                    report("layer_output", l53_path.layer_output, read("l_out-53", kHidden));
                    std::cout << "M6-A27.2 qwen35 L53 P1 attribution COMPLETE\n";
                    return 0;
                }
                if (gate53_contract && position == 1) {
                    const auto external_recurrent = read_f32(
                        checkpoint(fixture, 1, "attn_output-53"), kVHeads * kState);
                    const auto external_gate = read_f32(
                        checkpoint(fixture, 1, "z-53"), kInner);
                    const auto external_gated = read_f32(
                        checkpoint(fixture, 1, "final_output-53"), kVHeads * kState);
                    const auto replay_gated = [&](std::span<const float> recurrent,
                                                  std::span<const float> gate) {
                        Buffer d_recurrent = allocate(recurrent.size() * sizeof(float));
                        Buffer d_gate = allocate(gate.size() * sizeof(float));
                        Buffer d_head_norm = allocate(recurrent.size() * sizeof(float));
                        Buffer d_gated = allocate(recurrent.size() * sizeof(float));
                        upload(recurrent.data(), d_recurrent->get(), recurrent.size() * sizeof(float));
                        upload(gate.data(), d_gate->get(), gate.size() * sizeof(float));
                        miinfer::launch_qwen3_head_rms_normalize(
                            static_cast<const float*>(d_recurrent->get()),
                            static_cast<float*>(d_head_norm->get()), kVHeads, kState,
                            model.config().rms_epsilon);
                        miinfer::launch_qwen3_head_mul(
                            static_cast<const float*>(d_head_norm->get()),
                            static_cast<const float*>(recurrent32_plus[16]->d_ssm_norm->get()),
                            static_cast<float*>(d_gated->get()), kVHeads, kState);
                        miinfer::launch_qwen3_silu_mul(
                            static_cast<const float*>(d_gate->get()),
                            static_cast<const float*>(d_gated->get()),
                            static_cast<float*>(d_gated->get()), kVHeads * kState);
                        MIINFER_HIP_CHECK(hipDeviceSynchronize());
                        return download(d_gated->get(), kVHeads * kState);
                    };
                    const auto report = [&](const char* label, std::span<const float> actual,
                                            std::span<const float> expected) {
                        const auto error = located_host_error(actual, expected);
                        std::cout << "gated_contract=" << label
                                  << " max_abs=" << error.metrics.max_abs
                                  << " mean_abs=" << error.metrics.mean_abs
                                  << " rms=" << error.metrics.rms
                                  << " relative_rms=" << error.metrics.relative_rms
                                  << " max_index=" << error.index
                                  << " external=" << error.expected
                                  << " gpu=" << error.actual << '\n';
                    };
                    report("production_recurrent", l53_gate_path.recurrent_output,
                           external_recurrent);
                    report("production_gate", l53_gate_path.gate, external_gate);
                    report("external_operands", replay_gated(external_recurrent, external_gate),
                           external_gated);
                    report("external_recurrent", replay_gated(external_recurrent,
                                                               l53_gate_path.gate),
                           external_gated);
                    report("external_gate", replay_gated(l53_gate_path.recurrent_output,
                                                           external_gate),
                           external_gated);
                    report("production", replay_gated(l53_gate_path.recurrent_output,
                                                       l53_gate_path.gate),
                           external_gated);
                    std::cout << "M6-A27.3 qwen35 L53 P1 gated contract COMPLETE\n";
                    return 0;
                }
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
                      << (external_contract64
                              ? "M6-A27 qwen35 sixty-four-layer external-contract PASS\n"
                              : external_contract32
                              ? "M6-A26 qwen35 thirty-two-layer external-contract PASS\n"
                              : prefix32 ? "M6-A26 qwen35 thirty-two-layer GPU prefix PASS\n"
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
                if (deep && !state_correct && !external_contract32 && !external_contract64) {
                    throw std::runtime_error("hybrid recurrent state mismatch");
                }
                if (second_block) {
                    std::cout << position;
                    for (std::size_t layer = 4; layer < 8; ++layer) {
                        std::cout << ' ' << errors[layer].max_abs << ' ' << errors[layer].rms
                                  << ' ' << errors[layer].relative_rms;
                    }
                    std::cout << ' ' << (state_correct ? "PASS" : "DIAGNOSTIC_RETEST") << '\n';
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
                    std::cout << ' ' << (state_correct ? "PASS" : "DIAGNOSTIC_RETEST") << '\n';
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
