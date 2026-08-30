#include "miinfer/qwen3_gpu_layer.hpp"

#include "miinfer/hip_check.hpp"
#include "miinfer/q4_q8_gemv.hpp"
#include "miinfer/q4_q8_zero_point_dot.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace miinfer {

namespace {

class DeviceBytes {
public:
    explicit DeviceBytes(std::size_t bytes) {
        if (bytes == 0 || hipMalloc(&data_, bytes) != hipSuccess) {
            throw std::runtime_error("GPU layer buffer allocation failed");
        }
        bytes_ = bytes;
    }
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
    ~DeviceBytes() { if (data_ != nullptr) (void)hipFree(data_); }

    [[nodiscard]] void* data() const noexcept { return data_; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

template <typename T>
class DeviceBuffer final : public DeviceBytes {
public:
    explicit DeviceBuffer(std::size_t elements) : DeviceBytes(elements * sizeof(T)) {}
    [[nodiscard]] T* data() const noexcept { return static_cast<T*>(DeviceBytes::data()); }
};

void copy_to_host(const float* device, std::vector<float>& host) {
    MIINFER_HIP_CHECK(hipMemcpy(host.data(), device, host.size() * sizeof(float), hipMemcpyDeviceToHost));
}

void upload_f32(const Qwen3TensorView& tensor, float* device, std::size_t elements) {
    if (tensor.type() != GgufTensorType::f32 || tensor.bytes() != elements * sizeof(float)) {
        throw std::runtime_error("unexpected F32 layer tensor: " + tensor.name());
    }
    MIINFER_HIP_CHECK(hipMemcpy(device, tensor.data(), tensor.bytes(), hipMemcpyHostToDevice));
}

void launch_projection(
    const Qwen3GpuPlan& plan,
    const Qwen3TensorView& weight,
    const float* input,
    std::uint32_t input_elements,
    float* output,
    int rows,
    int columns,
    __half* input_half,
    Q8_1Block* input_q8,
    __half* output_half,
    const char* projection) {
    launch_qwen3_f32_to_f16(input, input_half, input_elements);
    launch_q8_1_quantize(input_half, input_q8, static_cast<int>(input_elements));
    const auto* device_weight = static_cast<const Q4_0Block*>(plan.device_tensor_data(weight.name()));
    switch (plan.kernel_for(projection)) {
    case Q4GemvKernel::zero_point_128:
        launch_q4_q8_gemv_zero_point_dot_128(device_weight, input_q8, output_half, rows, columns);
        break;
    case Q4GemvKernel::zero_point_128_wave64:
        launch_q4_q8_gemv_zero_point_dot_wave64(device_weight, input_q8, output_half, rows, columns);
        break;
    case Q4GemvKernel::zero_point_256:
        launch_q4_q8_gemv_zero_point_dot(device_weight, input_q8, output_half, rows, columns);
        break;
    }
    launch_qwen3_f16_to_f32(output_half, output, static_cast<std::uint32_t>(rows));
}

std::vector<float> capture(const float* device, std::size_t elements) {
    std::vector<float> host(elements);
    copy_to_host(device, host);
    return host;
}

void capture_qwen3_head_norm(
    const float* input,
    float* rms,
    float* weighted,
    const float* weights,
    std::uint32_t heads,
    std::uint32_t head_dim,
    float epsilon) {
    launch_qwen3_head_rms_normalize(input, rms, heads, head_dim, epsilon);
    launch_qwen3_head_mul(rms, weights, weighted, heads, head_dim);
}

}  // namespace

Qwen3LayerTrace execute_qwen3_layer0_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position) {
    if (position != 0) {
        throw std::invalid_argument("GPU layer-0 fixture currently supports position zero only");
    }
    const auto& model = plan.model();
    const auto& config = model.config();
    if (token >= config.vocab_size || model.layers().empty()) {
        throw std::invalid_argument("invalid layer-0 token/model");
    }
    constexpr std::size_t kMaxVector = 12288;
    constexpr std::size_t kMaxQ8Bytes = (kMaxVector / 32) * sizeof(Q8_1Block);
    const auto hidden = static_cast<std::size_t>(config.hidden_size);
    const auto intermediate = static_cast<std::size_t>(config.intermediate_size);
    const auto heads = static_cast<std::size_t>(config.attention_heads);
    const auto kv_heads = static_cast<std::size_t>(config.kv_heads);
    const auto head_dim = static_cast<std::size_t>(config.head_dim);
    const auto& layer = model.layers().front();

    DeviceBuffer<float> embedding(kMaxVector), attn_rms(kMaxVector), attn_norm(kMaxVector);
    DeviceBuffer<float> q(kMaxVector), q_rms(kMaxVector), q_normed(kMaxVector), q_rope(kMaxVector);
    DeviceBuffer<float> k(kMaxVector), k_rms(kMaxVector), k_normed(kMaxVector), k_rope(kMaxVector);
    DeviceBuffer<float> v(kMaxVector), attention(kMaxVector), attention_projected(kMaxVector);
    DeviceBuffer<float> ffn_input(kMaxVector), ffn_rms(kMaxVector), ffn_norm(kMaxVector);
    DeviceBuffer<float> gate(kMaxVector), up(kMaxVector), swiglu(kMaxVector);
    DeviceBuffer<float> ffn_output(kMaxVector), layer_output(kMaxVector);
    DeviceBuffer<float> norm_weights(kMaxVector), head_weights(head_dim);
    DeviceBuffer<float> scores(heads), probabilities(heads);
    DeviceBytes input_half(kMaxVector * sizeof(__half));
    DeviceBytes output_half(kMaxVector * sizeof(__half));
    DeviceBytes input_q8(kMaxQ8Bytes);
    auto* half_input = static_cast<__half*>(input_half.data());
    auto* half_output = static_cast<__half*>(output_half.data());
    auto* q8_input = static_cast<Q8_1Block*>(input_q8.data());

    Qwen3LayerTrace trace;
    launch_qwen3_q4_embedding(
        static_cast<const std::byte*>(plan.device_tensor_data(model.token_embeddings().name())),
        token, config.vocab_size, config.hidden_size, embedding.data());
    trace.embedding = capture(embedding.data(), hidden);

    upload_f32(layer.attention_norm, norm_weights.data(), hidden);
    launch_qwen3_rms_normalize(embedding.data(), attn_rms.data(), config.hidden_size, config.rms_epsilon);
    trace.attn_rms = capture(attn_rms.data(), hidden);
    launch_qwen3_elementwise_mul(attn_rms.data(), norm_weights.data(), attn_norm.data(), config.hidden_size);
    trace.attn_norm = capture(attn_norm.data(), hidden);

    launch_projection(plan, layer.q, attn_norm.data(), config.hidden_size, q.data(),
                      config.hidden_size, config.hidden_size, half_input, q8_input, half_output, "q");
    trace.q_projection = capture(q.data(), hidden);
    trace.q_reshape = trace.q_projection;
    upload_f32(layer.q_norm, head_weights.data(), head_dim);
    capture_qwen3_head_norm(q.data(), q_rms.data(), q_normed.data(), head_weights.data(),
                            config.attention_heads, config.head_dim, config.rms_epsilon);
    trace.q_rms = capture(q_rms.data(), hidden);
    trace.q_normed = capture(q_normed.data(), hidden);
    launch_qwen3_rope(q_normed.data(), q_rope.data(), config.attention_heads, config.head_dim,
                      0, config.rope_theta);
    trace.q_rope = capture(q_rope.data(), hidden);

    launch_projection(plan, layer.v, attn_norm.data(), config.hidden_size, v.data(),
                      config.kv_heads * config.head_dim, config.hidden_size,
                      half_input, q8_input, half_output, "v");
    trace.v_projection = capture(v.data(), kv_heads * head_dim);
    trace.v_reshape = trace.v_projection;

    launch_projection(plan, layer.k, attn_norm.data(), config.hidden_size, k.data(),
                      config.kv_heads * config.head_dim, config.hidden_size,
                      half_input, q8_input, half_output, "k");
    trace.k_projection = capture(k.data(), kv_heads * head_dim);
    trace.k_reshape = trace.k_projection;
    upload_f32(layer.k_norm, head_weights.data(), head_dim);
    capture_qwen3_head_norm(k.data(), k_rms.data(), k_normed.data(), head_weights.data(),
                            config.kv_heads, config.head_dim, config.rms_epsilon);
    trace.k_rms = capture(k_rms.data(), kv_heads * head_dim);
    trace.k_normed = capture(k_normed.data(), kv_heads * head_dim);
    launch_qwen3_rope(k_normed.data(), k_rope.data(), config.kv_heads, config.head_dim,
                      0, config.rope_theta);
    trace.k_rope = capture(k_rope.data(), kv_heads * head_dim);
    trace.k_view = trace.k_rope;
    trace.v_view = trace.v_reshape;
    trace.q_view = trace.q_rope;
    trace.q_permuted = trace.q_rope;

    launch_qwen3_single_token_attention(
        q_rope.data(), k_rope.data(), v.data(), attention.data(), scores.data(), probabilities.data(),
        config.attention_heads, config.kv_heads, config.head_dim,
        1.0F / std::sqrt(static_cast<float>(config.head_dim)));
    trace.attention_output = capture(attention.data(), hidden);
    launch_projection(plan, layer.output, attention.data(), config.hidden_size, attention_projected.data(),
                      config.hidden_size, config.hidden_size, half_input, q8_input, half_output, "o");
    launch_qwen3_add(attention_projected.data(), embedding.data(), ffn_input.data(), config.hidden_size);
    trace.ffn_input = capture(ffn_input.data(), hidden);

    upload_f32(layer.ffn_norm, norm_weights.data(), hidden);
    launch_qwen3_rms_normalize(ffn_input.data(), ffn_rms.data(), config.hidden_size, config.rms_epsilon);
    trace.ffn_rms = capture(ffn_rms.data(), hidden);
    launch_qwen3_elementwise_mul(ffn_rms.data(), norm_weights.data(), ffn_norm.data(), config.hidden_size);
    trace.ffn_norm = capture(ffn_norm.data(), hidden);
    launch_projection(plan, layer.gate, ffn_norm.data(), config.hidden_size, gate.data(),
                      config.intermediate_size, config.hidden_size, half_input, q8_input, half_output, "gate");
    trace.gate = capture(gate.data(), intermediate);
    launch_projection(plan, layer.up, ffn_norm.data(), config.hidden_size, up.data(),
                      config.intermediate_size, config.hidden_size, half_input, q8_input, half_output, "up");
    trace.up = capture(up.data(), intermediate);
    const char* mutation = std::getenv("MIINFER_GPU_LAYER_MUTATE");
    if (mutation != nullptr && std::strcmp(mutation, "swap-gate-up") == 0) {
        // Test-only discriminator.  It is opt-in through an environment
        // variable and never participates in normal execution.
        launch_qwen3_silu_mul(up.data(), gate.data(), swiglu.data(), config.intermediate_size);
    } else {
        launch_qwen3_silu_mul(gate.data(), up.data(), swiglu.data(), config.intermediate_size);
    }
    trace.swiglu = capture(swiglu.data(), intermediate);
    launch_projection(plan, layer.down, swiglu.data(), config.intermediate_size, ffn_output.data(),
                      config.hidden_size, config.intermediate_size, half_input, q8_input, half_output, "down");
    trace.ffn_output = capture(ffn_output.data(), hidden);
    launch_qwen3_add(ffn_output.data(), ffn_input.data(), layer_output.data(), config.hidden_size);
    trace.layer_output = capture(layer_output.data(), hidden);

    // These are not present in the 28-file reference fixture, but retaining
    // them in the trace is useful for GPU↔host triangulation of attention.
    trace.attention_scores = capture(scores.data(), heads);
    trace.attention_probabilities = capture(probabilities.data(), heads);
    return trace;
}

}  // namespace miinfer
