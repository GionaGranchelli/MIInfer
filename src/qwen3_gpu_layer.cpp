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
#include <memory>
#include <stdexcept>
#include <span>
#include <string_view>
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

bool use_exact_q8_metadata(const char* projection) {
    // The exact metadata contract is now the production Q4/Q8 contract.  An
    // explicit list remains useful for controlled A/B replay; an empty list
    // selects only the mandatory historical Down correction.
    const char* configured = std::getenv("MIINFER_EXACT_Q8_PROJECTIONS");
    if (configured == nullptr) return true;
    if (std::strcmp(projection, "down") == 0) return true;
    const std::string_view list(configured);
    const std::string_view name(projection);
    std::size_t begin = 0;
    while (begin < list.size()) {
        const auto end = list.find(',', begin);
        const auto token = list.substr(begin, end == std::string_view::npos
                                                 ? list.size() - begin : end - begin);
        if (token == name) return true;
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return false;
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
    Q8ExactBlock* input_q8_exact,
    __half* output_half,
    const char* projection) {
    const bool exact_metadata = use_exact_q8_metadata(projection);
    launch_qwen3_f32_to_f16(input, input_half, input_elements);
    if (exact_metadata) {
        launch_q8_exact_quantize(input_half, input_q8_exact, static_cast<int>(input_elements));
    } else {
        launch_q8_1_quantize(input_half, input_q8, static_cast<int>(input_elements));
    }
    const auto* device_weight = static_cast<const Q4_0Block*>(plan.device_tensor_data(weight.name()));
    switch (plan.kernel_for(projection)) {
    case Q4GemvKernel::zero_point_128:
        if (exact_metadata) {
            launch_q4_q8_gemv_zero_point_dot_128_exact_metadata(
                device_weight, input_q8_exact, output_half, rows, columns);
        } else {
            launch_q4_q8_gemv_zero_point_dot_128(device_weight, input_q8, output_half,
                                                 rows, columns);
        }
        break;
    case Q4GemvKernel::zero_point_128_wave64:
        if (exact_metadata) {
            launch_q4_q8_gemv_zero_point_dot_wave64_exact_metadata(
                device_weight, input_q8_exact, output_half, rows, columns);
        } else {
            launch_q4_q8_gemv_zero_point_dot_wave64(device_weight, input_q8, output_half,
                                                     rows, columns);
        }
        break;
    case Q4GemvKernel::zero_point_256:
        if (exact_metadata) {
            launch_q4_q8_gemv_zero_point_dot_exact_metadata(
                device_weight, input_q8_exact, output_half, rows, columns);
        } else {
            launch_q4_q8_gemv_zero_point_dot(device_weight, input_q8, output_half,
                                              rows, columns);
        }
        break;
    }
    launch_qwen3_f16_to_f32(output_half, output, static_cast<std::uint32_t>(rows));
}

void launch_projection_probe(
    const Qwen3GpuPlan& plan,
    const Qwen3TensorView& weight,
    const float* input,
    std::uint32_t input_elements,
    float* output,
    int rows,
    int columns,
    __half* input_half,
    Q8ExactBlock* input_q8_exact,
    __half* output_half,
    Qwen3ProjectionPrecision precision,
    const char* projection) {
    const bool f16_input = precision == Qwen3ProjectionPrecision::f16_input_q8_f16_output
        || precision == Qwen3ProjectionPrecision::f16_input_q8_f32_output;
    const bool f16_output = precision == Qwen3ProjectionPrecision::f16_input_q8_f16_output
        || precision == Qwen3ProjectionPrecision::f32_input_q8_f16_output;
    if (f16_input) {
        launch_qwen3_f32_to_f16(input, input_half, input_elements);
        launch_q8_exact_quantize(input_half, input_q8_exact, static_cast<int>(input_elements));
    } else {
        launch_q8_exact_quantize_f32(input, input_q8_exact, static_cast<int>(input_elements));
    }
    const auto* device_weight = static_cast<const Q4_0Block*>(plan.device_tensor_data(weight.name()));
    const auto kernel = plan.kernel_for(projection);
    if (f16_output) {
        switch (kernel) {
        case Q4GemvKernel::zero_point_128:
            launch_q4_q8_gemv_zero_point_dot_128_exact_metadata(
                device_weight, input_q8_exact, output_half, rows, columns);
            break;
        case Q4GemvKernel::zero_point_128_wave64:
            launch_q4_q8_gemv_zero_point_dot_wave64_exact_metadata(
                device_weight, input_q8_exact, output_half, rows, columns);
            break;
        case Q4GemvKernel::zero_point_256:
            launch_q4_q8_gemv_zero_point_dot_exact_metadata(
                device_weight, input_q8_exact, output_half, rows, columns);
            break;
        }
        launch_qwen3_f16_to_f32(output_half, output, static_cast<std::uint32_t>(rows));
    } else {
        switch (kernel) {
        case Q4GemvKernel::zero_point_128:
            launch_q4_q8_gemv_zero_point_dot_128_exact_metadata_f32(
                device_weight, input_q8_exact, output, rows, columns);
            break;
        case Q4GemvKernel::zero_point_128_wave64:
            launch_q4_q8_gemv_zero_point_dot_wave64_exact_metadata_f32(
                device_weight, input_q8_exact, output, rows, columns);
            break;
        case Q4GemvKernel::zero_point_256:
            launch_q4_q8_gemv_zero_point_dot_exact_metadata_f32(
                device_weight, input_q8_exact, output, rows, columns);
            break;
        }
    }
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

Qwen3Layer0GpuKvCache::Qwen3Layer0GpuKvCache(
    std::size_t kv_heads, std::size_t head_dim, std::size_t capacity)
    : kv_heads_(kv_heads), head_dim_(head_dim), capacity_(capacity) {
    if (kv_heads == 0 || head_dim == 0 || capacity == 0) {
        throw std::invalid_argument("invalid GPU KV-cache dimensions");
    }
    const auto bytes = kv_heads * head_dim * capacity * sizeof(float);
    MIINFER_HIP_CHECK(hipMalloc(&keys_, bytes));
    if (hipMalloc(&values_, bytes) != hipSuccess) {
        (void)hipFree(keys_);
        keys_ = nullptr;
        throw std::runtime_error("GPU value-cache allocation failed");
    }
    reset();
}

Qwen3Layer0GpuKvCache::~Qwen3Layer0GpuKvCache() {
    if (keys_ != nullptr) (void)hipFree(keys_);
    if (values_ != nullptr) (void)hipFree(values_);
}

void Qwen3Layer0GpuKvCache::reset() {
    const auto bytes = kv_heads_ * head_dim_ * capacity_ * sizeof(float);
    MIINFER_HIP_CHECK(hipMemset(keys_, 0, bytes));
    MIINFER_HIP_CHECK(hipMemset(values_, 0, bytes));
    length_ = 0;
}

void Qwen3Layer0GpuKvCache::append(
    std::size_t position, const float* keys, const float* values) {
    if (keys == nullptr || values == nullptr || position != length_
        || position >= capacity_) {
        throw std::invalid_argument("invalid GPU KV-cache append");
    }
    const auto bytes = head_dim_ * sizeof(float);
    for (std::size_t head = 0; head < kv_heads_; ++head) {
        const auto offset = (head * capacity_ + position) * head_dim_;
        MIINFER_HIP_CHECK(hipMemcpy(
            static_cast<float*>(keys_) + offset, keys + head * head_dim_, bytes,
            hipMemcpyDeviceToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(
            static_cast<float*>(values_) + offset, values + head * head_dim_, bytes,
            hipMemcpyDeviceToDevice));
    }
    ++length_;
}

std::vector<float> Qwen3Layer0GpuKvCache::snapshot_keys() const {
    const auto elements = kv_heads_ * capacity_ * head_dim_;
    std::vector<float> result(elements);
    MIINFER_HIP_CHECK(hipMemcpy(result.data(), keys_, elements * sizeof(float), hipMemcpyDeviceToHost));
    return result;
}

std::vector<float> Qwen3Layer0GpuKvCache::snapshot_values() const {
    const auto elements = kv_heads_ * capacity_ * head_dim_;
    std::vector<float> result(elements);
    MIINFER_HIP_CHECK(hipMemcpy(result.data(), values_, elements * sizeof(float), hipMemcpyDeviceToHost));
    return result;
}

Qwen3LayerTrace qwen3_layer_gpu_impl(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    const float* input_device,
    std::size_t position,
    Qwen3Layer0GpuKvCache& cache,
    float* output_device) {
    const auto& model = plan.model();
    const auto& config = model.config();
    if (input_device == nullptr || model.layers().empty() || layer_index >= model.layers().size()) {
        throw std::invalid_argument("invalid Qwen3 GPU layer input/index");
    }
    constexpr std::size_t kMaxVector = 12288;
    constexpr std::size_t kMaxQ8Bytes = (kMaxVector / 32) * sizeof(Q8_1Block);
    const auto hidden = static_cast<std::size_t>(config.hidden_size);
    const auto intermediate = static_cast<std::size_t>(config.intermediate_size);
    const auto heads = static_cast<std::size_t>(config.attention_heads);
    const auto kv_heads = static_cast<std::size_t>(config.kv_heads);
    const auto head_dim = static_cast<std::size_t>(config.head_dim);
    const auto& layer = model.layers()[layer_index];
    if (cache.kv_heads() != kv_heads || cache.head_dim() != head_dim
        || position != cache.length() || position >= cache.capacity()) {
        throw std::invalid_argument("GPU KV-cache does not match layer geometry or sequence position");
    }

    DeviceBuffer<float> embedding(kMaxVector), attn_rms(kMaxVector), attn_norm(kMaxVector);
    DeviceBuffer<float> q(kMaxVector), q_rms(kMaxVector), q_normed(kMaxVector), q_rope(kMaxVector);
    DeviceBuffer<float> k(kMaxVector), k_rms(kMaxVector), k_normed(kMaxVector), k_rope(kMaxVector);
    DeviceBuffer<float> v(kMaxVector), attention(kMaxVector), attention_projected(kMaxVector);
    DeviceBuffer<float> ffn_input(kMaxVector), ffn_rms(kMaxVector), ffn_norm(kMaxVector);
    DeviceBuffer<float> gate(kMaxVector), up(kMaxVector), swiglu(kMaxVector);
    DeviceBuffer<float> ffn_output(kMaxVector), layer_output(kMaxVector);
    DeviceBuffer<float> norm_weights(kMaxVector), head_weights(head_dim);
    DeviceBuffer<float> scores(heads * cache.capacity());
    DeviceBuffer<float> probabilities(heads * cache.capacity());
    DeviceBytes input_half(kMaxVector * sizeof(__half));
    DeviceBytes output_half(kMaxVector * sizeof(__half));
    DeviceBytes input_q8(kMaxQ8Bytes);
    DeviceBytes input_q8_exact(kMaxQ8Bytes);
    auto* half_input = static_cast<__half*>(input_half.data());
    auto* half_output = static_cast<__half*>(output_half.data());
    auto* q8_input = static_cast<Q8_1Block*>(input_q8.data());
    auto* q8_exact_input = static_cast<Q8ExactBlock*>(input_q8_exact.data());

    Qwen3LayerTrace trace;
    MIINFER_HIP_CHECK(hipMemcpy(embedding.data(), input_device, hidden * sizeof(float),
                                hipMemcpyDeviceToDevice));
    trace.embedding = capture(embedding.data(), hidden);

    upload_f32(layer.attention_norm, norm_weights.data(), hidden);
    launch_qwen3_rms_normalize(embedding.data(), attn_rms.data(), config.hidden_size, config.rms_epsilon);
    trace.attn_rms = capture(attn_rms.data(), hidden);
    launch_qwen3_elementwise_mul(attn_rms.data(), norm_weights.data(), attn_norm.data(), config.hidden_size);
    trace.attn_norm = capture(attn_norm.data(), hidden);

    launch_projection(plan, layer.q, attn_norm.data(), config.hidden_size, q.data(),
                      config.hidden_size, config.hidden_size, half_input, q8_input, q8_exact_input,
                      half_output, "q");
    trace.q_projection = capture(q.data(), hidden);
    trace.q_reshape = trace.q_projection;
    upload_f32(layer.q_norm, head_weights.data(), head_dim);
    capture_qwen3_head_norm(q.data(), q_rms.data(), q_normed.data(), head_weights.data(),
                            config.attention_heads, config.head_dim, config.rms_epsilon);
    trace.q_rms = capture(q_rms.data(), hidden);
    trace.q_normed = capture(q_normed.data(), hidden);
    const char* kv_mutation = std::getenv("MIINFER_GPU_KV_MUTATE");
    const bool wrong_rope_position = kv_mutation != nullptr
        && std::strcmp(kv_mutation, "wrong-rope-position") == 0 && position > 0;
    const auto rope_position = wrong_rope_position ? position - 1 : position;
    launch_qwen3_rope(q_normed.data(), q_rope.data(), config.attention_heads, config.head_dim,
                      static_cast<std::uint32_t>(rope_position), config.rope_theta);
    trace.q_rope = capture(q_rope.data(), hidden);

    launch_projection(plan, layer.v, attn_norm.data(), config.hidden_size, v.data(),
                      config.kv_heads * config.head_dim, config.hidden_size,
                      half_input, q8_input, q8_exact_input, half_output, "v");
    trace.v_projection = capture(v.data(), kv_heads * head_dim);
    trace.v_reshape = trace.v_projection;

    launch_projection(plan, layer.k, attn_norm.data(), config.hidden_size, k.data(),
                      config.kv_heads * config.head_dim, config.hidden_size,
                      half_input, q8_input, q8_exact_input, half_output, "k");
    trace.k_projection = capture(k.data(), kv_heads * head_dim);
    trace.k_reshape = trace.k_projection;
    upload_f32(layer.k_norm, head_weights.data(), head_dim);
    capture_qwen3_head_norm(k.data(), k_rms.data(), k_normed.data(), head_weights.data(),
                            config.kv_heads, config.head_dim, config.rms_epsilon);
    trace.k_rms = capture(k_rms.data(), kv_heads * head_dim);
    trace.k_normed = capture(k_normed.data(), kv_heads * head_dim);
    launch_qwen3_rope(k_normed.data(), k_rope.data(), config.kv_heads, config.head_dim,
                      static_cast<std::uint32_t>(rope_position), config.rope_theta);
    trace.k_rope = capture(k_rope.data(), kv_heads * head_dim);
    trace.k_view = trace.k_rope;
    trace.v_view = trace.v_reshape;
    trace.q_view = trace.q_rope;
    trace.q_permuted = trace.q_rope;

    cache.append(position, k_rope.data(), v.data());
    const bool ignore_earlier_cache = kv_mutation != nullptr
        && std::strcmp(kv_mutation, "ignore-earlier-cache") == 0 && position > 0;
    const auto attention_length = ignore_earlier_cache ? std::size_t{1} : cache.length();
    if (ignore_earlier_cache) {
        MIINFER_HIP_CHECK(hipMemset(scores.data(), 0, heads * cache.length() * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(probabilities.data(), 0,
                                    heads * cache.length() * sizeof(float)));
    }
    launch_qwen3_cached_attention(
        q_rope.data(), static_cast<const float*>(cache.device_keys()),
        static_cast<const float*>(cache.device_values()),
        static_cast<std::uint32_t>(attention_length), static_cast<std::uint32_t>(cache.capacity()),
        attention.data(), scores.data(), probabilities.data(), config.attention_heads,
        config.kv_heads, config.head_dim, 1.0F / std::sqrt(static_cast<float>(config.head_dim)));
    trace.attention_output = capture(attention.data(), hidden);
    launch_projection(plan, layer.output, attention.data(), config.hidden_size, attention_projected.data(),
                      config.hidden_size, config.hidden_size, half_input, q8_input, q8_exact_input,
                      half_output, "o");
    launch_qwen3_add(attention_projected.data(), embedding.data(), ffn_input.data(), config.hidden_size);
    trace.ffn_input = capture(ffn_input.data(), hidden);

    upload_f32(layer.ffn_norm, norm_weights.data(), hidden);
    launch_qwen3_rms_normalize(ffn_input.data(), ffn_rms.data(), config.hidden_size, config.rms_epsilon);
    trace.ffn_rms = capture(ffn_rms.data(), hidden);
    launch_qwen3_elementwise_mul(ffn_rms.data(), norm_weights.data(), ffn_norm.data(), config.hidden_size);
    trace.ffn_norm = capture(ffn_norm.data(), hidden);
    launch_projection(plan, layer.gate, ffn_norm.data(), config.hidden_size, gate.data(),
                      config.intermediate_size, config.hidden_size, half_input, q8_input, q8_exact_input,
                      half_output, "gate");
    trace.gate = capture(gate.data(), intermediate);
    launch_projection(plan, layer.up, ffn_norm.data(), config.hidden_size, up.data(),
                      config.intermediate_size, config.hidden_size, half_input, q8_input, q8_exact_input,
                      half_output, "up");
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
                      config.hidden_size, config.intermediate_size, half_input, q8_input, q8_exact_input,
                      half_output, "down");
    trace.ffn_output = capture(ffn_output.data(), hidden);
    launch_qwen3_add(ffn_output.data(), ffn_input.data(), layer_output.data(), config.hidden_size);
    trace.layer_output = capture(layer_output.data(), hidden);

    if (output_device != nullptr) {
        MIINFER_HIP_CHECK(hipMemcpy(output_device, layer_output.data(), hidden * sizeof(float),
                                    hipMemcpyDeviceToDevice));
    }

    // These are not present in the 28-file reference fixture, but retaining
    // them in the trace is useful for GPU↔host triangulation of attention.
    trace.attention_scores = capture(scores.data(), heads * cache.length());
    trace.attention_probabilities = capture(probabilities.data(), heads * cache.length());
    return trace;
}

Qwen3LayerTrace execute_qwen3_layer0_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position) {
    if (position != 0) {
        throw std::invalid_argument("standalone GPU layer-0 execution only supports position zero");
    }
    const auto& config = plan.model().config();
    Qwen3Layer0GpuKvCache cache(config.kv_heads, config.head_dim, 1);
    return execute_qwen3_layer0_gpu(plan, token, position, cache);
}

Qwen3LayerTrace execute_qwen3_layer0_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3Layer0GpuKvCache& cache) {
    const auto& model = plan.model();
    const auto& config = model.config();
    if (token >= config.vocab_size) throw std::invalid_argument("token ID is outside the vocabulary");
    DeviceBuffer<float> input(config.hidden_size);
    launch_qwen3_q4_embedding(
        static_cast<const std::byte*>(plan.device_tensor_data(model.token_embeddings().name())),
        token, config.vocab_size, config.hidden_size, input.data());
    return qwen3_layer_gpu_impl(plan, 0, input.data(), position, cache, nullptr);
}

Qwen3ForwardTrace execute_qwen3_forward_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position) {
    const auto& model = plan.model();
    const auto& config = model.config();
    if (position != 0) {
        throw std::invalid_argument("full GPU forward currently supports position zero only");
    }
    if (token >= config.vocab_size || model.layers().size() != config.layer_count) {
        throw std::invalid_argument("invalid Qwen3 full-forward token/model");
    }

    DeviceBuffer<float> input(config.hidden_size), output(config.hidden_size);
    launch_qwen3_q4_embedding(
        static_cast<const std::byte*>(plan.device_tensor_data(model.token_embeddings().name())),
        token, config.vocab_size, config.hidden_size, input.data());

    Qwen3ForwardTrace forward;
    forward.embedding = capture(input.data(), config.hidden_size);
    std::vector<std::unique_ptr<Qwen3Layer0GpuKvCache>> caches;
    caches.reserve(model.layers().size());
    for (std::size_t layer = 0; layer < model.layers().size(); ++layer) {
        caches.push_back(std::make_unique<Qwen3Layer0GpuKvCache>(
            config.kv_heads, config.head_dim, 1));
    }

    forward.layer_outputs.reserve(model.layers().size());
    auto* current = input.data();
    auto* next = output.data();
    for (std::size_t layer = 0; layer < model.layers().size(); ++layer) {
        const auto trace = qwen3_layer_gpu_impl(
            plan, layer, current, position, *caches[layer], next);
        forward.layer_outputs.push_back(trace.layer_output);
        std::swap(current, next);
    }

    DeviceBuffer<float> final_norm(config.hidden_size);
    DeviceBuffer<float> norm_weights(config.hidden_size);
    upload_f32(model.final_norm(), norm_weights.data(), config.hidden_size);
    launch_qwen3_rms_norm(current, norm_weights.data(), final_norm.data(),
                          config.hidden_size, config.rms_epsilon);
    forward.final_norm = capture(final_norm.data(), config.hidden_size);

    DeviceBuffer<float> logits(config.vocab_size);
    DeviceBuffer<Q8KDeviceBlock> quantized_final_norm(config.hidden_size / 256);
    const auto* output_weight = static_cast<const Q6KDeviceBlock*>(
        plan.device_tensor_data(model.output().name()));
    // The pinned reference CPU path quantizes the final activation to Q8_K
    // before its Q6_K output projection.  Keep that contract explicit for
    // full-forward correctness; the older F32-input probe remains available
    // for isolated primitive tests.
    launch_qwen3_q8_k_quantize(final_norm.data(), quantized_final_norm.data(), config.hidden_size);
    launch_qwen3_q6_k_q8_k_gemv(output_weight, quantized_final_norm.data(), logits.data(),
                                config.vocab_size, config.hidden_size);
    forward.logits = capture(logits.data(), config.vocab_size);
    return forward;
}

Qwen3LayerTrace execute_qwen3_layer_gpu_teacher_forced(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    std::span<const float> input,
    std::size_t position) {
    const auto& config = plan.model().config();
    if (position != 0) {
        throw std::invalid_argument("teacher-forced GPU replay currently supports position zero only");
    }
    if (input.size() != config.hidden_size || layer_index >= plan.model().layers().size()) {
        throw std::invalid_argument("invalid teacher-forced GPU layer input/index");
    }
    DeviceBuffer<float> input_device(config.hidden_size);
    MIINFER_HIP_CHECK(hipMemcpy(input_device.data(), input.data(),
                                input.size() * sizeof(float), hipMemcpyHostToDevice));
    Qwen3Layer0GpuKvCache cache(config.kv_heads, config.head_dim, 1);
    return qwen3_layer_gpu_impl(plan, layer_index, input_device.data(), position, cache, nullptr);
}

Qwen3FfnProbeTrace execute_qwen3_ffn_gpu_probe(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    std::span<const float> ffn_input,
    std::span<const float> ffn_norm,
    std::span<const float> gate_override,
    std::span<const float> up_override,
    std::span<const float> swiglu_override,
    std::span<const float> ffn_output_override,
    Qwen3ProjectionPrecision precision) {
    const auto& model = plan.model();
    const auto& config = model.config();
    const std::size_t hidden = config.hidden_size;
    const std::size_t intermediate = config.intermediate_size;
    if (layer_index >= model.layers().size() || ffn_input.size() != hidden
        || ffn_norm.size() != hidden
        || (!gate_override.empty() && gate_override.size() != intermediate)
        || (!up_override.empty() && up_override.size() != intermediate)
        || (!swiglu_override.empty() && swiglu_override.size() != intermediate)
        || (!ffn_output_override.empty() && ffn_output_override.size() != hidden)) {
        throw std::invalid_argument("invalid Qwen3 FFN probe dimensions");
    }

    DeviceBuffer<float> input_device(hidden), norm_device(hidden);
    DeviceBuffer<float> gate(intermediate), up(intermediate), swiglu(intermediate);
    DeviceBuffer<float> ffn_output(hidden), layer_output(hidden);
    DeviceBytes input_half(12288U * sizeof(__half));
    DeviceBytes output_half(12288U * sizeof(__half));
    DeviceBytes input_q8_exact((12288U / 32U) * sizeof(Q8ExactBlock));
    MIINFER_HIP_CHECK(hipMemcpy(input_device.data(), ffn_input.data(), hidden * sizeof(float),
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(norm_device.data(), ffn_norm.data(), hidden * sizeof(float),
                                hipMemcpyHostToDevice));
    auto* half_input = static_cast<__half*>(input_half.data());
    auto* half_output = static_cast<__half*>(output_half.data());
    auto* q8_exact_input = static_cast<Q8ExactBlock*>(input_q8_exact.data());
    const auto& layer = model.layers()[layer_index];
    if (gate_override.empty()) {
        launch_projection_probe(plan, layer.gate, norm_device.data(), config.hidden_size,
                                gate.data(), config.intermediate_size, config.hidden_size,
                                half_input, q8_exact_input, half_output, precision, "gate");
    } else {
        MIINFER_HIP_CHECK(hipMemcpy(gate.data(), gate_override.data(), intermediate * sizeof(float),
                                    hipMemcpyHostToDevice));
    }
    if (up_override.empty()) {
        launch_projection_probe(plan, layer.up, norm_device.data(), config.hidden_size,
                                up.data(), config.intermediate_size, config.hidden_size,
                                half_input, q8_exact_input, half_output, precision, "up");
    } else {
        MIINFER_HIP_CHECK(hipMemcpy(up.data(), up_override.data(), intermediate * sizeof(float),
                                    hipMemcpyHostToDevice));
    }
    if (swiglu_override.empty()) {
        launch_qwen3_silu_mul(gate.data(), up.data(), swiglu.data(), config.intermediate_size);
    } else {
        MIINFER_HIP_CHECK(hipMemcpy(swiglu.data(), swiglu_override.data(), intermediate * sizeof(float),
                                    hipMemcpyHostToDevice));
    }
    if (ffn_output_override.empty()) {
        launch_projection_probe(plan, layer.down, swiglu.data(), config.intermediate_size,
                                ffn_output.data(), config.hidden_size, config.intermediate_size,
                                half_input, q8_exact_input, half_output, precision, "down");
    } else {
        MIINFER_HIP_CHECK(hipMemcpy(ffn_output.data(), ffn_output_override.data(), hidden * sizeof(float),
                                    hipMemcpyHostToDevice));
    }
    launch_qwen3_add(ffn_output.data(), input_device.data(), layer_output.data(), config.hidden_size);
    Qwen3FfnProbeTrace trace;
    trace.gate = capture(gate.data(), intermediate);
    trace.up = capture(up.data(), intermediate);
    trace.swiglu = capture(swiglu.data(), intermediate);
    trace.ffn_output = capture(ffn_output.data(), hidden);
    trace.layer_output = capture(layer_output.data(), hidden);
    return trace;
}

Qwen3ProjectionProbeTrace execute_qwen3_projection_gpu_probe(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    Qwen3Projection projection,
    std::span<const float> input,
    Qwen3ProjectionPrecision precision) {
    const auto& model = plan.model();
    const auto& config = model.config();
    if (layer_index >= model.layers().size()) {
        throw std::invalid_argument("invalid Qwen3 projection probe layer");
    }

    const Qwen3TensorView* weight = nullptr;
    const char* name = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    const auto& layer = model.layers()[layer_index];
    switch (projection) {
    case Qwen3Projection::q:
        weight = &layer.q;
        name = "q";
        rows = config.hidden_size;
        columns = config.hidden_size;
        break;
    case Qwen3Projection::k:
        weight = &layer.k;
        name = "k";
        rows = config.kv_heads * config.head_dim;
        columns = config.hidden_size;
        break;
    case Qwen3Projection::v:
        weight = &layer.v;
        name = "v";
        rows = config.kv_heads * config.head_dim;
        columns = config.hidden_size;
        break;
    case Qwen3Projection::o:
        weight = &layer.output;
        name = "o";
        rows = config.hidden_size;
        columns = config.hidden_size;
        break;
    case Qwen3Projection::gate:
        weight = &layer.gate;
        name = "gate";
        rows = config.intermediate_size;
        columns = config.hidden_size;
        break;
    case Qwen3Projection::up:
        weight = &layer.up;
        name = "up";
        rows = config.intermediate_size;
        columns = config.hidden_size;
        break;
    case Qwen3Projection::down:
        weight = &layer.down;
        name = "down";
        rows = config.hidden_size;
        columns = config.intermediate_size;
        break;
    }
    if (weight == nullptr || input.size() != columns) {
        throw std::invalid_argument("invalid Qwen3 projection probe dimensions");
    }

    constexpr std::size_t kMaxVector = 12288;
    DeviceBuffer<float> input_device(input.size());
    DeviceBuffer<float> output(rows);
    DeviceBytes input_half(kMaxVector * sizeof(__half));
    DeviceBytes input_q8_exact((kMaxVector / kQ8_1BlockSize) * sizeof(Q8ExactBlock));
    DeviceBytes output_half(kMaxVector * sizeof(__half));
    MIINFER_HIP_CHECK(hipMemcpy(input_device.data(), input.data(),
                                input.size() * sizeof(float), hipMemcpyHostToDevice));
    launch_projection_probe(plan, *weight, input_device.data(), columns, output.data(),
                            static_cast<int>(rows), static_cast<int>(columns),
                            static_cast<__half*>(input_half.data()),
                            static_cast<Q8ExactBlock*>(input_q8_exact.data()),
                            static_cast<__half*>(output_half.data()), precision, name);

    Qwen3ProjectionProbeTrace trace;
    trace.output = capture(output.data(), rows);
    return trace;
}

Qwen3DownProjectionContractTrace execute_qwen3_down_projection_contract_probe(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    std::span<const float> swiglu,
    bool direct_f32_input) {
    const auto& model = plan.model();
    const auto& config = model.config();
    if (layer_index >= model.layers().size()
        || swiglu.size() != config.intermediate_size) {
        throw std::invalid_argument("invalid Qwen3 down-projection probe dimensions");
    }

    const auto& layer = model.layers()[layer_index];
    DeviceBuffer<float> input(config.intermediate_size);
    DeviceBuffer<float> current(config.hidden_size);
    DeviceBuffer<float> exact_sum(config.hidden_size);
    DeviceBuffer<float> direct_signed(config.hidden_size);
    DeviceBytes input_half(config.intermediate_size * sizeof(__half));
    DeviceBytes input_q8((config.intermediate_size / kQ8_1BlockSize) * sizeof(Q8_1Block));
    MIINFER_HIP_CHECK(hipMemcpy(input.data(), swiglu.data(),
                                swiglu.size() * sizeof(float), hipMemcpyHostToDevice));
    if (direct_f32_input) {
        launch_q8_1_quantize_f32(input.data(), static_cast<Q8_1Block*>(input_q8.data()),
                                 config.intermediate_size);
    } else {
        launch_qwen3_f32_to_f16(input.data(), static_cast<__half*>(input_half.data()),
                                config.intermediate_size);
        launch_q8_1_quantize(static_cast<const __half*>(input_half.data()),
                             static_cast<Q8_1Block*>(input_q8.data()), config.intermediate_size);
    }
    const auto* device_weight = static_cast<const Q4_0Block*>(
        plan.device_tensor_data(layer.down.name()));
    const auto* device_input = static_cast<const Q8_1Block*>(input_q8.data());
    launch_q4_q8_gemv_zero_point_dot_f32(
        device_weight, device_input, current.data(), config.hidden_size,
        config.intermediate_size);
    launch_q4_q8_gemv_zero_point_dot_exact_sum_f32(
        device_weight, device_input, exact_sum.data(), config.hidden_size,
        config.intermediate_size);
    launch_q4_q8_gemv_direct_signed_f32(
        device_weight, device_input, direct_signed.data(), config.hidden_size,
        config.intermediate_size);

    Qwen3DownProjectionContractTrace trace;
    trace.current_s_correction = capture(current.data(), config.hidden_size);
    trace.exact_sum_correction = capture(exact_sum.data(), config.hidden_size);
    trace.direct_signed_oracle = capture(direct_signed.data(), config.hidden_size);
    return trace;
}

}  // namespace miinfer
