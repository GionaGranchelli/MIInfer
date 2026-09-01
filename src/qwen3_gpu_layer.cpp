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
#include <utility>
#include <vector>

namespace miinfer {

const char* qwen3_profile_category_name(Qwen3ProfileCategory category) noexcept {
    switch (category) {
    case Qwen3ProfileCategory::embedding: return "embedding";
    case Qwen3ProfileCategory::normalization: return "normalization";
    case Qwen3ProfileCategory::quantization: return "quantization";
    case Qwen3ProfileCategory::qkv_projection: return "qkv_projection";
    case Qwen3ProfileCategory::o_projection: return "o_projection";
    case Qwen3ProfileCategory::ffn_projection: return "ffn_projection";
    case Qwen3ProfileCategory::rope: return "rope";
    case Qwen3ProfileCategory::attention: return "attention";
    case Qwen3ProfileCategory::activation: return "activation";
    case Qwen3ProfileCategory::residual: return "residual";
    case Qwen3ProfileCategory::conversion: return "conversion";
    case Qwen3ProfileCategory::lm_head: return "lm_head";
    case Qwen3ProfileCategory::kv_cache: return "kv_cache";
    case Qwen3ProfileCategory::copies: return "copies";
    case Qwen3ProfileCategory::count: break;
    }
    return "unknown";
}

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

class ProfileScope {
public:
    ProfileScope(Qwen3GpuProfile* profile, Qwen3ProfileCategory category,
                 std::size_t dispatches, bool copy)
        : profile_(profile), category_(category), dispatches_(dispatches), copy_(copy) {
        if (profile_ == nullptr) return;
        MIINFER_HIP_CHECK(hipEventCreate(&start_));
        MIINFER_HIP_CHECK(hipEventCreate(&stop_));
        MIINFER_HIP_CHECK(hipEventRecord(start_));
    }

    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

    ~ProfileScope() {
        if (start_ != nullptr) (void)hipEventDestroy(start_);
        if (stop_ != nullptr) (void)hipEventDestroy(stop_);
    }

    void finish() {
        if (profile_ == nullptr || finished_) return;
        MIINFER_HIP_CHECK(hipEventRecord(stop_));
        if (profile_->deferred_timing) {
            profile_->pending_events.push_back({
                start_, stop_, category_, dispatches_, 0, copy_});
            start_ = nullptr;
            stop_ = nullptr;
            finished_ = true;
            return;
        }
        MIINFER_HIP_CHECK(hipEventSynchronize(stop_));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start_, stop_));
        const auto index = static_cast<std::size_t>(category_);
        if (copy_) {
            profile_->copy_ms[index] += milliseconds;
        } else {
            profile_->gpu_ms[index] += milliseconds;
            profile_->dispatches[index] += dispatches_;
        }
        finished_ = true;
    }

private:
    Qwen3GpuProfile* profile_ = nullptr;
    Qwen3ProfileCategory category_ = Qwen3ProfileCategory::count;
    std::size_t dispatches_ = 0;
    bool copy_ = false;
    bool finished_ = false;
    hipEvent_t start_ = nullptr;
    hipEvent_t stop_ = nullptr;
};

template <typename Function>
void profile_gpu_call(Qwen3GpuProfile* profile, Qwen3ProfileCategory category,
                      std::size_t dispatches, Function&& function) {
    if (profile == nullptr) {
        function();
        return;
    }
    ProfileScope scope(profile, category, dispatches, false);
    function();
    scope.finish();
}

template <typename Function>
void profile_copy_call(Qwen3GpuProfile* profile, Qwen3ProfileCategory category,
                       std::size_t bytes, Function&& function) {
    if (profile == nullptr) {
        function();
        return;
    }
    ProfileScope scope(profile, category, 0, true);
    function();
    profile->record_copy(category, bytes);
    scope.finish();
}

template <typename Function>
void profile_copy_call(Qwen3GpuProfile* profile, std::size_t bytes, Function&& function) {
    profile_copy_call(profile, Qwen3ProfileCategory::copies, bytes,
                      std::forward<Function>(function));
}

template <typename T>
class DeviceBuffer final : public DeviceBytes {
public:
    explicit DeviceBuffer(std::size_t elements) : DeviceBytes(elements * sizeof(T)) {}
    [[nodiscard]] T* data() const noexcept { return static_cast<T*>(DeviceBytes::data()); }
};

void copy_to_host(const float* device, std::vector<float>& host) {
    MIINFER_HIP_CHECK(hipMemcpy(host.data(), device, host.size() * sizeof(float), hipMemcpyDeviceToHost));
}

const float* device_f32_tensor(const Qwen3GpuPlan& plan, const Qwen3TensorView& tensor,
                               std::size_t elements) {
    if (tensor.type() != GgufTensorType::f32 || tensor.bytes() != elements * sizeof(float)) {
        throw std::runtime_error("unexpected F32 layer tensor: " + tensor.name());
    }
    return static_cast<const float*>(plan.device_tensor_data(tensor.name()));
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

bool projection_selected(const char* variable, const char* projection) {
    // M4-B8 diagnostic controls. The default remains the accepted
    // F16-input/F16-output production contract; comma-separated projection
    // names opt individual operations into a correctness-only F32 boundary.
    const char* configured = std::getenv(variable);
    if (configured == nullptr || *configured == '\0') return false;
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

bool use_parallel_attention() {
    const char* configured = std::getenv("MIINFER_ATTENTION_KERNEL");
    // M5-C2 KEEP: the cooperative kernel is the production default. The
    // serial kernel remains available as an explicit A/B control.
    if (configured == nullptr || std::strcmp(configured, "parallel") == 0) return true;
    if (std::strcmp(configured, "serial") == 0) return false;
    throw std::invalid_argument(
        "MIINFER_ATTENTION_KERNEL must be 'parallel' or 'serial'");
}

Qwen3ProfileCategory projection_profile_category(const char* projection) {
    if (std::strcmp(projection, "q") == 0 || std::strcmp(projection, "k") == 0
        || std::strcmp(projection, "v") == 0) {
        return Qwen3ProfileCategory::qkv_projection;
    }
    if (std::strcmp(projection, "o") == 0) return Qwen3ProfileCategory::o_projection;
    return Qwen3ProfileCategory::ffn_projection;
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
    const char* projection,
    Qwen3GpuProfile* profile = nullptr) {
    const auto projection_category = projection_profile_category(projection);
    const bool exact_metadata = use_exact_q8_metadata(projection);
    const bool f32_input = projection_selected("MIINFER_F32_INPUT_PROJECTIONS", projection);
    const bool f32_output = projection_selected("MIINFER_F32_OUTPUT_PROJECTIONS", projection);
    if (f32_output && !exact_metadata) {
        throw std::invalid_argument("F32 projection output requires Q8Exact metadata");
    }
    if (!exact_metadata && f32_input) {
        profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
            launch_q8_1_quantize_f32(input, input_q8, static_cast<int>(input_elements));
        });
    } else if (f32_input) {
        profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
            launch_q8_exact_quantize_f32(input, input_q8_exact, static_cast<int>(input_elements));
        });
    } else {
        profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
            launch_qwen3_f32_to_f16(input, input_half, input_elements);
        });
        if (exact_metadata) {
            profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
                launch_q8_exact_quantize(input_half, input_q8_exact, static_cast<int>(input_elements));
            });
        } else {
            profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
                launch_q8_1_quantize(input_half, input_q8, static_cast<int>(input_elements));
            });
        }
    }
    const auto* device_weight = static_cast<const Q4_0Block*>(plan.device_tensor_data(weight.name()));
    profile_gpu_call(profile, projection_category, 1, [&] {
        switch (plan.kernel_for(projection)) {
        case Q4GemvKernel::zero_point_128:
            if (f32_output) {
                launch_q4_q8_gemv_zero_point_dot_128_exact_metadata_f32(
                    device_weight, input_q8_exact, output, rows, columns);
            } else if (exact_metadata) {
                launch_q4_q8_gemv_zero_point_dot_128_exact_metadata(
                    device_weight, input_q8_exact, output_half, rows, columns);
            } else {
                launch_q4_q8_gemv_zero_point_dot_128(device_weight, input_q8, output_half,
                                                     rows, columns);
            }
            break;
        case Q4GemvKernel::zero_point_128_wave64:
            if (f32_output) {
                launch_q4_q8_gemv_zero_point_dot_wave64_exact_metadata_f32(
                    device_weight, input_q8_exact, output, rows, columns);
            } else if (exact_metadata) {
                launch_q4_q8_gemv_zero_point_dot_wave64_exact_metadata(
                    device_weight, input_q8_exact, output_half, rows, columns);
            } else {
                launch_q4_q8_gemv_zero_point_dot_wave64(device_weight, input_q8, output_half,
                                                         rows, columns);
            }
            break;
        case Q4GemvKernel::zero_point_256:
            if (f32_output) {
                launch_q4_q8_gemv_zero_point_dot_exact_metadata_f32(
                    device_weight, input_q8_exact, output, rows, columns);
            } else if (exact_metadata) {
                launch_q4_q8_gemv_zero_point_dot_exact_metadata(
                    device_weight, input_q8_exact, output_half, rows, columns);
            } else {
                launch_q4_q8_gemv_zero_point_dot(device_weight, input_q8, output_half,
                                                  rows, columns);
            }
            break;
        }
    });
    if (!f32_output) {
        profile_gpu_call(profile, Qwen3ProfileCategory::conversion, 1, [&] {
            launch_qwen3_f16_to_f32(output_half, output, static_cast<std::uint32_t>(rows));
        });
    }
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

std::vector<float> capture(const float* device, std::size_t elements,
                           Qwen3GpuProfile* profile = nullptr) {
    std::vector<float> host(elements);
    profile_copy_call(profile, host.size() * sizeof(float), [&] { copy_to_host(device, host); });
    return host;
}

void capture_qwen3_head_norm(
    const float* input,
    float* rms,
    float* weighted,
    const float* weights,
    std::uint32_t heads,
    std::uint32_t head_dim,
    float epsilon,
    Qwen3GpuProfile* profile) {
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_head_rms_normalize(input, rms, heads, head_dim, epsilon);
    });
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_head_mul(rms, weights, weighted, heads, head_dim);
    });
}

}  // namespace

// One decode workspace is shared by all layers because layer execution is
// deliberately sequential. Keeping these buffers alive across tokens removes
// allocator churn from steady-state decode without changing any kernel launch
// or precision boundary.
class Qwen3GpuDecodeWorkspace {
public:
    Qwen3GpuDecodeWorkspace(std::size_t hidden, std::size_t vocab,
                            std::size_t heads, std::size_t capacity)
        : input(hidden), output(hidden),
          embedding(kMaxVector), attn_rms(kMaxVector), attn_norm(kMaxVector),
          q(kMaxVector), q_rms(kMaxVector), q_normed(kMaxVector), q_rope(kMaxVector),
          k(kMaxVector), k_rms(kMaxVector), k_normed(kMaxVector), k_rope(kMaxVector),
          v(kMaxVector), attention(kMaxVector), attention_projected(kMaxVector),
          ffn_input(kMaxVector), ffn_rms(kMaxVector), ffn_norm(kMaxVector),
          gate(kMaxVector), up(kMaxVector), swiglu(kMaxVector),
          ffn_output(kMaxVector), layer_output(kMaxVector),
          scores(heads * capacity), probabilities(heads * capacity),
          input_half(kMaxVector * sizeof(__half)),
          output_half(kMaxVector * sizeof(__half)),
          input_q8(kMaxQ8Bytes), input_q8_exact(kMaxQ8Bytes),
          final_norm(hidden), logits(vocab),
          quantized_final_norm(hidden / 256) {}

    static constexpr std::size_t kMaxVector = 12288;
    static constexpr std::size_t kMaxQ8Bytes =
        (kMaxVector / 32) * sizeof(Q8_1Block);

    DeviceBuffer<float> input;
    DeviceBuffer<float> output;
    DeviceBuffer<float> embedding;
    DeviceBuffer<float> attn_rms;
    DeviceBuffer<float> attn_norm;
    DeviceBuffer<float> q;
    DeviceBuffer<float> q_rms;
    DeviceBuffer<float> q_normed;
    DeviceBuffer<float> q_rope;
    DeviceBuffer<float> k;
    DeviceBuffer<float> k_rms;
    DeviceBuffer<float> k_normed;
    DeviceBuffer<float> k_rope;
    DeviceBuffer<float> v;
    DeviceBuffer<float> attention;
    DeviceBuffer<float> attention_projected;
    DeviceBuffer<float> ffn_input;
    DeviceBuffer<float> ffn_rms;
    DeviceBuffer<float> ffn_norm;
    DeviceBuffer<float> gate;
    DeviceBuffer<float> up;
    DeviceBuffer<float> swiglu;
    DeviceBuffer<float> ffn_output;
    DeviceBuffer<float> layer_output;
    DeviceBuffer<float> scores;
    DeviceBuffer<float> probabilities;
    DeviceBytes input_half;
    DeviceBytes output_half;
    DeviceBytes input_q8;
    DeviceBytes input_q8_exact;
    DeviceBuffer<float> final_norm;
    DeviceBuffer<float> logits;
    DeviceBuffer<Q8KDeviceBlock> quantized_final_norm;
};

Qwen3GpuProfile::~Qwen3GpuProfile() noexcept {
    for (const auto& event : pending_events) {
        if (event.start != nullptr) (void)hipEventDestroy(event.start);
        if (event.stop != nullptr) (void)hipEventDestroy(event.stop);
    }
}

void Qwen3GpuProfile::finalize() {
    if (pending_events.empty()) return;
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    finalization_synchronizations += 1;
    for (const auto& event : pending_events) {
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, event.start, event.stop));
        const auto index = static_cast<std::size_t>(event.category);
        if (event.copy) {
            copy_ms[index] += milliseconds;
        } else {
            gpu_ms[index] += milliseconds;
            dispatches[index] += event.dispatches;
        }
        (void)hipEventDestroy(event.start);
        (void)hipEventDestroy(event.stop);
    }
    pending_events.clear();
}

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
    std::size_t position, const float* keys, const float* values,
    Qwen3GpuProfile* profile) {
    if (keys == nullptr || values == nullptr || position != length_
        || position >= capacity_) {
        throw std::invalid_argument("invalid GPU KV-cache append");
    }
    const auto bytes = head_dim_ * sizeof(float);
    for (std::size_t head = 0; head < kv_heads_; ++head) {
        const auto offset = (head * capacity_ + position) * head_dim_;
        profile_copy_call(profile, Qwen3ProfileCategory::kv_cache, bytes, [&] {
            MIINFER_HIP_CHECK(hipMemcpy(
                static_cast<float*>(keys_) + offset, keys + head * head_dim_, bytes,
                hipMemcpyDeviceToDevice));
        });
        profile_copy_call(profile, Qwen3ProfileCategory::kv_cache, bytes, [&] {
            MIINFER_HIP_CHECK(hipMemcpy(
                static_cast<float*>(values_) + offset, values + head * head_dim_, bytes,
                hipMemcpyDeviceToDevice));
        });
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

Qwen3GpuDecodeCache::Qwen3GpuDecodeCache(
    std::size_t layers, std::size_t kv_heads, std::size_t head_dim, std::size_t capacity)
    : capacity_(capacity) {
    if (layers == 0 || kv_heads == 0 || head_dim == 0 || capacity == 0) {
        throw std::invalid_argument("invalid GPU Qwen3 decode-cache dimensions");
    }
    caches_.reserve(layers);
    for (std::size_t layer = 0; layer < layers; ++layer) {
        caches_.push_back(std::make_unique<Qwen3Layer0GpuKvCache>(
            kv_heads, head_dim, capacity));
    }
}

Qwen3GpuDecodeCache::~Qwen3GpuDecodeCache() = default;

void Qwen3GpuDecodeCache::prepare(const Qwen3GpuPlan& plan) {
    const auto& config = plan.model().config();
    if (config.layer_count != caches_.size() || config.kv_heads != layer(0).kv_heads()
        || config.head_dim != layer(0).head_dim() || config.hidden_size == 0
        || config.vocab_size == 0 || config.hidden_size % 256 != 0) {
        throw std::invalid_argument("GPU decode workspace does not match model geometry");
    }
    if (workspace_ != nullptr) return;
    workspace_ = std::make_unique<Qwen3GpuDecodeWorkspace>(
        static_cast<std::size_t>(config.hidden_size),
        static_cast<std::size_t>(config.vocab_size),
        static_cast<std::size_t>(config.attention_heads),
        capacity_);
}

void Qwen3GpuDecodeCache::reset() {
    for (auto& cache : caches_) cache->reset();
}

std::size_t Qwen3GpuDecodeCache::length() const noexcept {
    return caches_.empty() ? 0 : caches_.front()->length();
}

Qwen3Layer0GpuKvCache& Qwen3GpuDecodeCache::layer(std::size_t layer_index) {
    if (layer_index >= caches_.size()) {
        throw std::out_of_range("GPU Qwen3 decode-cache layer index");
    }
    return *caches_[layer_index];
}

const Qwen3Layer0GpuKvCache& Qwen3GpuDecodeCache::layer(std::size_t layer_index) const {
    if (layer_index >= caches_.size()) {
        throw std::out_of_range("GPU Qwen3 decode-cache layer index");
    }
    return *caches_[layer_index];
}

Qwen3GpuDecodeWorkspace& Qwen3GpuDecodeCache::workspace() {
    if (workspace_ == nullptr) {
        throw std::logic_error("GPU Qwen3 decode workspace is not prepared");
    }
    return *workspace_;
}

Qwen3LayerTrace qwen3_layer_gpu_impl(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    const float* input_device,
    std::size_t position,
    Qwen3Layer0GpuKvCache& cache,
    float* output_device,
    std::span<const float> attention_output_override = {},
    Qwen3GpuProfile* profile = nullptr,
    bool capture_trace = true,
    Qwen3GpuDecodeWorkspace* workspace = nullptr) {
    const auto& model = plan.model();
    const auto& config = model.config();
    if (input_device == nullptr || model.layers().empty() || layer_index >= model.layers().size()) {
        throw std::invalid_argument("invalid Qwen3 GPU layer input/index");
    }
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

    std::unique_ptr<Qwen3GpuDecodeWorkspace> owned_workspace;
    if (workspace == nullptr) {
        owned_workspace = std::make_unique<Qwen3GpuDecodeWorkspace>(
            hidden, static_cast<std::size_t>(config.vocab_size), heads,
            cache.capacity());
        workspace = owned_workspace.get();
    }
    auto& buffers = *workspace;
    auto& embedding = buffers.embedding;
    auto& attn_norm = buffers.attn_norm;
    auto& q_normed = buffers.q_normed;
    auto& q_rope = buffers.q_rope;
    auto& k = buffers.k;
    auto& k_rms = buffers.k_rms;
    auto& k_normed = buffers.k_normed;
    auto& k_rope = buffers.k_rope;
    auto& v = buffers.v;
    auto& attention = buffers.attention;
    auto& attention_projected = buffers.attention_projected;
    auto& ffn_input = buffers.ffn_input;
    auto& ffn_rms = buffers.ffn_rms;
    auto& ffn_norm = buffers.ffn_norm;
    auto& gate = buffers.gate;
    auto& up = buffers.up;
    auto& swiglu = buffers.swiglu;
    auto& ffn_output = buffers.ffn_output;
    auto& layer_output = buffers.layer_output;
    auto& scores = buffers.scores;
    auto& probabilities = buffers.probabilities;
    auto* half_input = static_cast<__half*>(buffers.input_half.data());
    auto* half_output = static_cast<__half*>(buffers.output_half.data());
    auto* q8_input = static_cast<Q8_1Block*>(buffers.input_q8.data());
    auto* q8_exact_input = static_cast<Q8ExactBlock*>(buffers.input_q8_exact.data());

    const auto capture_optional = [profile, capture_trace](const float* device,
                                                            std::size_t elements) {
        return capture_trace ? capture(device, elements, profile) : std::vector<float>{};
    };

    Qwen3LayerTrace trace;
    profile_copy_call(profile, hidden * sizeof(float), [&] {
        MIINFER_HIP_CHECK(hipMemcpy(buffers.embedding.data(), input_device, hidden * sizeof(float),
                                    hipMemcpyDeviceToDevice));
    });
    trace.embedding = capture_optional(buffers.embedding.data(), hidden);

    const auto* attention_norm_weights = device_f32_tensor(plan, layer.attention_norm, hidden);
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_rms_normalize(buffers.embedding.data(), buffers.attn_rms.data(), config.hidden_size,
                                   config.rms_epsilon);
    });
    trace.attn_rms = capture_optional(buffers.attn_rms.data(), hidden);
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_elementwise_mul(buffers.attn_rms.data(), attention_norm_weights, buffers.attn_norm.data(),
                                     config.hidden_size);
    });
    trace.attn_norm = capture_optional(buffers.attn_norm.data(), hidden);

    launch_projection(plan, layer.q, buffers.attn_norm.data(), config.hidden_size, buffers.q.data(),
                      config.hidden_size, config.hidden_size, half_input, q8_input, q8_exact_input,
                      half_output, "q", profile);
    trace.q_projection = capture_optional(buffers.q.data(), hidden);
    trace.q_reshape = trace.q_projection;
    const auto* q_norm_weights = device_f32_tensor(plan, layer.q_norm, head_dim);
    capture_qwen3_head_norm(buffers.q.data(), buffers.q_rms.data(), buffers.q_normed.data(), q_norm_weights,
                            config.attention_heads, config.head_dim, config.rms_epsilon, profile);
    trace.q_rms = capture_optional(buffers.q_rms.data(), hidden);
    trace.q_normed = capture_optional(buffers.q_normed.data(), hidden);
    const char* kv_mutation = std::getenv("MIINFER_GPU_KV_MUTATE");
    const bool wrong_rope_position = kv_mutation != nullptr
        && std::strcmp(kv_mutation, "wrong-rope-position") == 0 && position > 0;
    const auto rope_position = wrong_rope_position ? position - 1 : position;
    profile_gpu_call(profile, Qwen3ProfileCategory::rope, 1, [&] {
        launch_qwen3_rope(q_normed.data(), q_rope.data(), config.attention_heads, config.head_dim,
                          static_cast<std::uint32_t>(rope_position), config.rope_theta);
    });
    trace.q_rope = capture_optional(q_rope.data(), hidden);

    launch_projection(plan, layer.v, attn_norm.data(), config.hidden_size, v.data(),
                      config.kv_heads * config.head_dim, config.hidden_size,
                      half_input, q8_input, q8_exact_input, half_output, "v", profile);
    trace.v_projection = capture_optional(v.data(), kv_heads * head_dim);
    trace.v_reshape = trace.v_projection;

    launch_projection(plan, layer.k, attn_norm.data(), config.hidden_size, k.data(),
                      config.kv_heads * config.head_dim, config.hidden_size,
                      half_input, q8_input, q8_exact_input, half_output, "k", profile);
    trace.k_projection = capture_optional(k.data(), kv_heads * head_dim);
    trace.k_reshape = trace.k_projection;
    const auto* k_norm_weights = device_f32_tensor(plan, layer.k_norm, head_dim);
    capture_qwen3_head_norm(k.data(), k_rms.data(), k_normed.data(), k_norm_weights,
                            config.kv_heads, config.head_dim, config.rms_epsilon, profile);
    trace.k_rms = capture_optional(k_rms.data(), kv_heads * head_dim);
    trace.k_normed = capture_optional(k_normed.data(), kv_heads * head_dim);
    profile_gpu_call(profile, Qwen3ProfileCategory::rope, 1, [&] {
        launch_qwen3_rope(k_normed.data(), k_rope.data(), config.kv_heads, config.head_dim,
                          static_cast<std::uint32_t>(rope_position), config.rope_theta);
    });
    trace.k_rope = capture_optional(k_rope.data(), kv_heads * head_dim);
    trace.k_view = trace.k_rope;
    trace.v_view = trace.v_reshape;
    trace.q_view = trace.q_rope;
    trace.q_permuted = trace.q_rope;

    cache.append(position, k_rope.data(), v.data(), profile);
    const bool ignore_earlier_cache = kv_mutation != nullptr
        && std::strcmp(kv_mutation, "ignore-earlier-cache") == 0 && position > 0;
    const auto attention_length = ignore_earlier_cache ? std::size_t{1} : cache.length();
    if (ignore_earlier_cache) {
        MIINFER_HIP_CHECK(hipMemset(scores.data(), 0, heads * cache.length() * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(probabilities.data(), 0,
                                    heads * cache.length() * sizeof(float)));
    }
    profile_gpu_call(profile, Qwen3ProfileCategory::attention, 1, [&] {
        if (use_parallel_attention()) {
            launch_qwen3_cached_attention_parallel(
                q_rope.data(), static_cast<const float*>(cache.device_keys()),
                static_cast<const float*>(cache.device_values()),
                static_cast<std::uint32_t>(attention_length),
                static_cast<std::uint32_t>(cache.capacity()), attention.data(), scores.data(),
                probabilities.data(), config.attention_heads, config.kv_heads, config.head_dim,
                1.0F / std::sqrt(static_cast<float>(config.head_dim)));
        } else {
            launch_qwen3_cached_attention(
                q_rope.data(), static_cast<const float*>(cache.device_keys()),
                static_cast<const float*>(cache.device_values()),
                static_cast<std::uint32_t>(attention_length),
                static_cast<std::uint32_t>(cache.capacity()), attention.data(), scores.data(),
                probabilities.data(), config.attention_heads, config.kv_heads, config.head_dim,
                1.0F / std::sqrt(static_cast<float>(config.head_dim)));
        }
    });
    // Match the pinned reference's kqv_out representation: attention is
    // accumulated in F32, materialized as FP16, then presented to the O
    // projection through the existing F32 interface.
    profile_gpu_call(profile, Qwen3ProfileCategory::conversion, 1, [&] {
        launch_qwen3_f32_to_f16(attention.data(), half_output, hidden);
    });
    profile_gpu_call(profile, Qwen3ProfileCategory::conversion, 1, [&] {
        launch_qwen3_f16_to_f32(half_output, attention.data(), hidden);
    });
    if (!attention_output_override.empty()) {
        if (attention_output_override.size() != hidden) {
            throw std::invalid_argument("attention-output override size mismatch");
        }
        profile_copy_call(profile, hidden * sizeof(float), [&] {
            MIINFER_HIP_CHECK(hipMemcpy(attention.data(), attention_output_override.data(),
                                        hidden * sizeof(float), hipMemcpyHostToDevice));
        });
    }
    trace.attention_output = capture_optional(attention.data(), hidden);
    launch_projection(plan, layer.output, attention.data(), config.hidden_size, attention_projected.data(),
                      config.hidden_size, config.hidden_size, half_input, q8_input, q8_exact_input,
                      half_output, "o", profile);
    profile_gpu_call(profile, Qwen3ProfileCategory::residual, 1, [&] {
        launch_qwen3_add(attention_projected.data(), embedding.data(), ffn_input.data(), config.hidden_size);
    });
    trace.ffn_input = capture_optional(ffn_input.data(), hidden);

    const auto* ffn_norm_weights = device_f32_tensor(plan, layer.ffn_norm, hidden);
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_rms_normalize(ffn_input.data(), ffn_rms.data(), config.hidden_size,
                                   config.rms_epsilon);
    });
    trace.ffn_rms = capture_optional(ffn_rms.data(), hidden);
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_elementwise_mul(ffn_rms.data(), ffn_norm_weights, ffn_norm.data(),
                                     config.hidden_size);
    });
    trace.ffn_norm = capture_optional(ffn_norm.data(), hidden);
    launch_projection(plan, layer.gate, ffn_norm.data(), config.hidden_size, gate.data(),
                      config.intermediate_size, config.hidden_size, half_input, q8_input, q8_exact_input,
                      half_output, "gate", profile);
    trace.gate = capture_optional(gate.data(), intermediate);
    launch_projection(plan, layer.up, ffn_norm.data(), config.hidden_size, up.data(),
                      config.intermediate_size, config.hidden_size, half_input, q8_input, q8_exact_input,
                      half_output, "up", profile);
    trace.up = capture_optional(up.data(), intermediate);
    const char* mutation = std::getenv("MIINFER_GPU_LAYER_MUTATE");
    if (mutation != nullptr && std::strcmp(mutation, "swap-gate-up") == 0) {
        // Test-only discriminator.  It is opt-in through an environment
        // variable and never participates in normal execution.
        profile_gpu_call(profile, Qwen3ProfileCategory::activation, 1, [&] {
            launch_qwen3_silu_mul(up.data(), gate.data(), swiglu.data(), config.intermediate_size);
        });
    } else {
        profile_gpu_call(profile, Qwen3ProfileCategory::activation, 1, [&] {
            launch_qwen3_silu_mul(gate.data(), up.data(), swiglu.data(), config.intermediate_size);
        });
    }
    trace.swiglu = capture_optional(swiglu.data(), intermediate);
    launch_projection(plan, layer.down, swiglu.data(), config.intermediate_size, ffn_output.data(),
                      config.hidden_size, config.intermediate_size, half_input, q8_input, q8_exact_input,
                      half_output, "down", profile);
    trace.ffn_output = capture_optional(ffn_output.data(), hidden);
    profile_gpu_call(profile, Qwen3ProfileCategory::residual, 1, [&] {
        launch_qwen3_add(ffn_output.data(), ffn_input.data(), layer_output.data(), config.hidden_size);
    });
    trace.layer_output = capture_optional(layer_output.data(), hidden);

    if (output_device != nullptr) {
        profile_copy_call(profile, hidden * sizeof(float), [&] {
            MIINFER_HIP_CHECK(hipMemcpy(output_device, layer_output.data(), hidden * sizeof(float),
                                        hipMemcpyDeviceToDevice));
        });
    }

    // These are not present in the 28-file reference fixture, but retaining
    // them in the trace is useful for GPU↔host triangulation of attention.
    trace.attention_scores = capture_optional(scores.data(), heads * cache.length());
    trace.attention_probabilities = capture_optional(probabilities.data(), heads * cache.length());
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
    Qwen3GpuDecodeCache cache(config.layer_count, config.kv_heads, config.head_dim, 1);
    return execute_qwen3_decode_gpu(plan, token, position, cache);
}

Qwen3ForwardTrace execute_qwen3_decode_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache) {
    return execute_qwen3_decode_gpu(plan, token, position, cache, nullptr);
}

Qwen3ForwardTrace execute_qwen3_decode_gpu(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    Qwen3GpuProfile* profile) {
    const auto& model = plan.model();
    const auto& config = model.config();
    if (token >= config.vocab_size || model.layers().size() != config.layer_count
        || cache.layers() != config.layer_count || cache.capacity() == 0
        || position != cache.length() || position >= cache.capacity()) {
        throw std::invalid_argument("invalid Qwen3 GPU decode state");
    }
    cache.prepare(plan);
    auto& workspace = cache.workspace();

    profile_gpu_call(profile, Qwen3ProfileCategory::embedding, 1, [&] {
        launch_qwen3_q4_embedding(
            static_cast<const std::byte*>(plan.device_tensor_data(model.token_embeddings().name())),
            token, config.vocab_size, config.hidden_size, workspace.input.data());
    });

    Qwen3ForwardTrace forward;
    forward.embedding = capture(workspace.input.data(), config.hidden_size, profile);
    forward.layer_outputs.reserve(model.layers().size());
    auto* current = workspace.input.data();
    auto* next = workspace.output.data();
    for (std::size_t layer = 0; layer < model.layers().size(); ++layer) {
        const auto trace = qwen3_layer_gpu_impl(
            plan, layer, current, position, cache.layer(layer), next, {}, profile, true,
            &workspace);
        forward.layer_outputs.push_back(trace.layer_output);
        std::swap(current, next);
    }

    const auto* final_norm_weights = device_f32_tensor(plan, model.final_norm(), config.hidden_size);
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_rms_norm(current, final_norm_weights, workspace.final_norm.data(),
                              config.hidden_size, config.rms_epsilon);
    });
    forward.final_norm = capture(workspace.final_norm.data(), config.hidden_size, profile);

    const auto* output_weight = static_cast<const Q6KDeviceBlock*>(
        plan.device_tensor_data(model.output().name()));
    profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
        launch_qwen3_q8_k_quantize(workspace.final_norm.data(), workspace.quantized_final_norm.data(),
                                   config.hidden_size);
    });
    profile_gpu_call(profile, Qwen3ProfileCategory::lm_head, 1, [&] {
        launch_qwen3_q6_k_q8_k_gemv(output_weight, workspace.quantized_final_norm.data(),
                                    workspace.logits.data(),
                                    config.vocab_size, config.hidden_size);
    });
    forward.logits = capture(workspace.logits.data(), config.vocab_size, profile);
    if (profile != nullptr) profile->finalize();
    return forward;
}

void execute_qwen3_decode_gpu_fast(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    std::span<float> logits_host) {
    execute_qwen3_decode_gpu_fast(plan, token, position, cache, logits_host, nullptr);
}

void execute_qwen3_decode_gpu_fast(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    std::span<float> logits_host,
    Qwen3GpuProfile* profile) {
    const auto& model = plan.model();
    const auto& config = model.config();
    if (token >= config.vocab_size || logits_host.size() != config.vocab_size
        || model.layers().size() != config.layer_count
        || cache.layers() != config.layer_count || cache.capacity() == 0
        || position != cache.length() || position >= cache.capacity()) {
        throw std::invalid_argument("invalid Qwen3 GPU fast decode state");
    }
    cache.prepare(plan);
    auto& workspace = cache.workspace();

    profile_gpu_call(profile, Qwen3ProfileCategory::embedding, 1, [&] {
        launch_qwen3_q4_embedding(
            static_cast<const std::byte*>(plan.device_tensor_data(model.token_embeddings().name())),
            token, config.vocab_size, config.hidden_size, workspace.input.data());
    });

    auto* current = workspace.input.data();
    auto* next = workspace.output.data();
    for (std::size_t layer = 0; layer < model.layers().size(); ++layer) {
        (void)qwen3_layer_gpu_impl(
            plan, layer, current, position, cache.layer(layer), next, {}, profile, false,
            &workspace);
        std::swap(current, next);
    }

    const auto* final_norm_weights = device_f32_tensor(plan, model.final_norm(), config.hidden_size);
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_rms_norm(current, final_norm_weights, workspace.final_norm.data(),
                              config.hidden_size, config.rms_epsilon);
    });

    const auto* output_weight = static_cast<const Q6KDeviceBlock*>(
        plan.device_tensor_data(model.output().name()));
    profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
        launch_qwen3_q8_k_quantize(workspace.final_norm.data(), workspace.quantized_final_norm.data(),
                                   config.hidden_size);
    });
    profile_gpu_call(profile, Qwen3ProfileCategory::lm_head, 1, [&] {
        launch_qwen3_q6_k_q8_k_gemv(output_weight, workspace.quantized_final_norm.data(),
                                    workspace.logits.data(),
                                    config.vocab_size, config.hidden_size);
    });
    profile_copy_call(profile, logits_host.size() * sizeof(float), [&] {
        MIINFER_HIP_CHECK(hipMemcpy(logits_host.data(), workspace.logits.data(),
                                    config.vocab_size * sizeof(float), hipMemcpyDeviceToHost));
    });
    if (profile != nullptr) profile->finalize();
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

Qwen3LayerTrace execute_qwen3_layer_gpu_attention_override(
    const Qwen3GpuPlan& plan,
    std::size_t layer_index,
    std::span<const float> input,
    std::span<const float> attention_output,
    std::size_t position) {
    const auto& config = plan.model().config();
    if (position != 0 || input.size() != config.hidden_size
        || attention_output.size() != config.hidden_size
        || layer_index >= plan.model().layers().size()) {
        throw std::invalid_argument("invalid GPU attention-output override");
    }
    DeviceBuffer<float> input_device(config.hidden_size);
    MIINFER_HIP_CHECK(hipMemcpy(input_device.data(), input.data(),
                                input.size() * sizeof(float), hipMemcpyHostToDevice));
    Qwen3Layer0GpuKvCache cache(config.kv_heads, config.head_dim, 1);
    return qwen3_layer_gpu_impl(plan, layer_index, input_device.data(), position, cache, nullptr,
                                attention_output);
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
