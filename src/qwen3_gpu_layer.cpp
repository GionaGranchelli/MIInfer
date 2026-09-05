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
#include <sstream>
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
    case Qwen3ProfileCategory::argmax: return "argmax";
    case Qwen3ProfileCategory::kv_cache: return "kv_cache";
    case Qwen3ProfileCategory::copies: return "copies";
    case Qwen3ProfileCategory::count: break;
    }
    return "unknown";
}

const char* qwen3_ffn_profile_stage_name(Qwen3FfnProfileStage stage) noexcept {
    switch (stage) {
    case Qwen3FfnProfileStage::normalization: return "ffn_normalization";
    case Qwen3FfnProfileStage::gate_input_quantization: return "gate_input_quantization";
    case Qwen3FfnProfileStage::gate_projection: return "gate_projection";
    case Qwen3FfnProfileStage::up_input_quantization: return "up_input_quantization";
    case Qwen3FfnProfileStage::up_projection: return "up_projection";
    case Qwen3FfnProfileStage::swiglu: return "swiglu";
    case Qwen3FfnProfileStage::swiglu_down_input_quantization:
        return "swiglu_down_input_quantization";
    case Qwen3FfnProfileStage::down_input_quantization: return "down_input_quantization";
    case Qwen3FfnProfileStage::down_projection: return "down_projection";
    case Qwen3FfnProfileStage::residual: return "ffn_residual";
    case Qwen3FfnProfileStage::count: break;
    }
    return "unknown";
}

const char* qwen3_boundary_profile_stage_name(Qwen3BoundaryProfileStage stage) noexcept {
    switch (stage) {
    case Qwen3BoundaryProfileStage::attn_rms_normalize: return "attn_rms_normalize";
    case Qwen3BoundaryProfileStage::attn_norm_scale: return "attn_norm_scale";
    case Qwen3BoundaryProfileStage::q_input_f32_to_f16: return "q_input_f32_to_f16";
    case Qwen3BoundaryProfileStage::q_input_q8: return "q_input_q8";
    case Qwen3BoundaryProfileStage::q_output_f16_to_f32: return "q_output_f16_to_f32";
    case Qwen3BoundaryProfileStage::q_head_rms_normalize: return "q_head_rms_normalize";
    case Qwen3BoundaryProfileStage::q_head_scale: return "q_head_scale";
    case Qwen3BoundaryProfileStage::k_input_f32_to_f16: return "k_input_f32_to_f16";
    case Qwen3BoundaryProfileStage::k_input_q8: return "k_input_q8";
    case Qwen3BoundaryProfileStage::k_output_f16_to_f32: return "k_output_f16_to_f32";
    case Qwen3BoundaryProfileStage::k_head_rms_normalize: return "k_head_rms_normalize";
    case Qwen3BoundaryProfileStage::k_head_scale: return "k_head_scale";
    case Qwen3BoundaryProfileStage::v_input_f32_to_f16: return "v_input_f32_to_f16";
    case Qwen3BoundaryProfileStage::v_input_q8: return "v_input_q8";
    case Qwen3BoundaryProfileStage::v_output_f16_to_f32: return "v_output_f16_to_f32";
    case Qwen3BoundaryProfileStage::attention_f32_to_f16: return "attention_f32_to_f16";
    case Qwen3BoundaryProfileStage::attention_f16_to_f32: return "attention_f16_to_f32";
    case Qwen3BoundaryProfileStage::o_input_f32_to_f16: return "o_input_f32_to_f16";
    case Qwen3BoundaryProfileStage::o_input_q8: return "o_input_q8";
    case Qwen3BoundaryProfileStage::o_output_f16_to_f32: return "o_output_f16_to_f32";
    case Qwen3BoundaryProfileStage::ffn_rms_normalize: return "ffn_rms_normalize";
    case Qwen3BoundaryProfileStage::ffn_norm_scale: return "ffn_norm_scale";
    case Qwen3BoundaryProfileStage::ffn_norm_to_shared_q8:
        return "ffn_norm_to_shared_q8";
    case Qwen3BoundaryProfileStage::gate_input_f32_to_f16: return "gate_input_f32_to_f16";
    case Qwen3BoundaryProfileStage::gate_input_q8: return "gate_input_q8";
    case Qwen3BoundaryProfileStage::gate_output_f16_to_f32: return "gate_output_f16_to_f32";
    case Qwen3BoundaryProfileStage::up_input_f32_to_f16: return "up_input_f32_to_f16";
    case Qwen3BoundaryProfileStage::up_input_q8: return "up_input_q8";
    case Qwen3BoundaryProfileStage::up_output_f16_to_f32: return "up_output_f16_to_f32";
    case Qwen3BoundaryProfileStage::down_input_f32_to_f16: return "down_input_f32_to_f16";
    case Qwen3BoundaryProfileStage::down_input_q8: return "down_input_q8";
    case Qwen3BoundaryProfileStage::down_output_f16_to_f32: return "down_output_f16_to_f32";
    case Qwen3BoundaryProfileStage::final_rms_norm: return "final_rms_norm";
    case Qwen3BoundaryProfileStage::final_norm_to_q8k: return "final_norm_to_q8k";
    case Qwen3BoundaryProfileStage::count: break;
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
                 std::size_t dispatches, bool copy,
                 Qwen3FfnProfileStage ffn_stage = Qwen3FfnProfileStage::count,
                 Qwen3BoundaryProfileStage boundary_stage = Qwen3BoundaryProfileStage::count,
                 std::size_t bytes = 0)
        : profile_(profile), category_(category), ffn_stage_(ffn_stage),
          boundary_stage_(boundary_stage), dispatches_(dispatches), bytes_(bytes), copy_(copy) {
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
                start_, stop_, category_, ffn_stage_, boundary_stage_, dispatches_, bytes_, copy_});
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
        if (!copy_ && ffn_stage_ != Qwen3FfnProfileStage::count) {
            const auto ffn_index = static_cast<std::size_t>(ffn_stage_);
            profile_->ffn_gpu_ms[ffn_index] += milliseconds;
            profile_->ffn_dispatches[ffn_index] += dispatches_;
        }
        if (!copy_ && boundary_stage_ != Qwen3BoundaryProfileStage::count) {
            const auto boundary_index = static_cast<std::size_t>(boundary_stage_);
            profile_->boundary_gpu_ms[boundary_index] += milliseconds;
            profile_->boundary_dispatches[boundary_index] += dispatches_;
            profile_->boundary_bytes[boundary_index] += bytes_;
        }
        finished_ = true;
    }

private:
    Qwen3GpuProfile* profile_ = nullptr;
    Qwen3ProfileCategory category_ = Qwen3ProfileCategory::count;
    Qwen3FfnProfileStage ffn_stage_ = Qwen3FfnProfileStage::count;
    Qwen3BoundaryProfileStage boundary_stage_ = Qwen3BoundaryProfileStage::count;
    std::size_t dispatches_ = 0;
    std::size_t bytes_ = 0;
    bool copy_ = false;
    bool finished_ = false;
    hipEvent_t start_ = nullptr;
    hipEvent_t stop_ = nullptr;
};

template <typename Function>
void profile_gpu_call(Qwen3GpuProfile* profile, Qwen3ProfileCategory category,
                      std::size_t dispatches, Function&& function,
                      Qwen3FfnProfileStage ffn_stage = Qwen3FfnProfileStage::count,
                      Qwen3BoundaryProfileStage boundary_stage = Qwen3BoundaryProfileStage::count,
                      std::size_t bytes = 0) {
    if (profile == nullptr) {
        function();
        return;
    }
    ProfileScope scope(profile, category, dispatches, false, ffn_stage, boundary_stage, bytes);
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

enum class AttentionKernel {
    serial,
    parallel,
    history_parallel,
};

AttentionKernel selected_attention_kernel() {
    const char* configured = std::getenv("MIINFER_ATTENTION_KERNEL");
    // M5-C2 KEEP: the cooperative kernel is the production default. The
    // serial kernel remains available as an explicit A/B control. C12b's
    // history-partitioned candidate is diagnostic-only until it passes the
    // stateful generation contract.
    if (configured == nullptr || std::strcmp(configured, "parallel") == 0) {
        return AttentionKernel::parallel;
    }
    if (std::strcmp(configured, "serial") == 0) return AttentionKernel::serial;
    if (std::strcmp(configured, "history") == 0) return AttentionKernel::history_parallel;
    throw std::invalid_argument(
        "MIINFER_ATTENTION_KERNEL must be 'parallel', 'serial', or 'history'");
}

bool use_direct_layer_output_handoff() {
    const char* configured = std::getenv("MIINFER_LAYER_OUTPUT_HANDOFF");
    // The direct handoff is the production fast-path default.  The copy
    // variant remains available for isolated performance A/B measurements.
    if (configured == nullptr || std::strcmp(configured, "direct") == 0) return true;
    if (std::strcmp(configured, "copy") == 0) return false;
    throw std::invalid_argument(
        "MIINFER_LAYER_OUTPUT_HANDOFF must be 'direct' or 'copy'");
}

bool use_coalesced_kv_cache_store() {
    const char* configured = std::getenv("MIINFER_KV_CACHE_WRITE");
    // The one-launch device store is the production default.  The per-head
    // memcpy path remains available as an isolated structural control.
    if (configured == nullptr || std::strcmp(configured, "store") == 0) return true;
    if (std::strcmp(configured, "copy") == 0) return false;
    throw std::invalid_argument(
        "MIINFER_KV_CACHE_WRITE must be 'store' or 'copy'");
}

bool use_fused_swiglu_q8() {
    const char* configured = std::getenv("MIINFER_SWIGLU_Q8_FUSION");
    if (configured == nullptr || std::strcmp(configured, "separate") == 0) return false;
    if (std::strcmp(configured, "fused") == 0) return true;
    throw std::invalid_argument(
        "MIINFER_SWIGLU_Q8_FUSION must be 'separate' or 'fused'");
}

bool use_shared_gate_up_q8() {
    const char* configured = std::getenv("MIINFER_FFN_Q8_REUSE");
    // C9c KEEP: one exact Q8 activation is the production default. The
    // separate path remains an explicit regression/A-B control.
    if (configured == nullptr || std::strcmp(configured, "shared") == 0) return true;
    if (std::strcmp(configured, "separate") == 0) return false;
    throw std::invalid_argument(
        "MIINFER_FFN_Q8_REUSE must be 'separate' or 'shared'");
}

bool use_fused_ffn_norm_q8() {
    const char* configured = std::getenv("MIINFER_FFN_NORM_Q8_FUSION");
    if (configured == nullptr || std::strcmp(configured, "separate") == 0) return false;
    if (std::strcmp(configured, "fused") == 0) return true;
    throw std::invalid_argument(
        "MIINFER_FFN_NORM_Q8_FUSION must be 'separate' or 'fused'");
}

bool verify_fused_ffn_norm_q8() {
    const char* configured = std::getenv("MIINFER_VERIFY_FFN_NORM_Q8_FUSION");
    if (configured == nullptr || std::strcmp(configured, "0") == 0
        || std::strcmp(configured, "false") == 0) return false;
    if (std::strcmp(configured, "1") == 0 || std::strcmp(configured, "true") == 0) return true;
    throw std::invalid_argument(
        "MIINFER_VERIFY_FFN_NORM_Q8_FUSION must be '0', '1', 'false', or 'true'");
}

bool verify_shared_gate_up_q8() {
    const char* configured = std::getenv("MIINFER_VERIFY_FFN_Q8_REUSE");
    if (configured == nullptr || std::strcmp(configured, "0") == 0
        || std::strcmp(configured, "false") == 0) return false;
    if (std::strcmp(configured, "1") == 0 || std::strcmp(configured, "true") == 0) return true;
    throw std::invalid_argument(
        "MIINFER_VERIFY_FFN_Q8_REUSE must be '0', '1', 'false', or 'true'");
}

Qwen3ProfileCategory projection_profile_category(const char* projection) {
    if (std::strcmp(projection, "q") == 0 || std::strcmp(projection, "k") == 0
        || std::strcmp(projection, "v") == 0) {
        return Qwen3ProfileCategory::qkv_projection;
    }
    if (std::strcmp(projection, "o") == 0) return Qwen3ProfileCategory::o_projection;
    return Qwen3ProfileCategory::ffn_projection;
}

Qwen3FfnProfileStage projection_ffn_quantization_stage(const char* projection) {
    if (std::strcmp(projection, "gate") == 0) {
        return Qwen3FfnProfileStage::gate_input_quantization;
    }
    if (std::strcmp(projection, "up") == 0) {
        return Qwen3FfnProfileStage::up_input_quantization;
    }
    if (std::strcmp(projection, "down") == 0) {
        return Qwen3FfnProfileStage::down_input_quantization;
    }
    return Qwen3FfnProfileStage::count;
}

Qwen3FfnProfileStage projection_ffn_stage(const char* projection) {
    if (std::strcmp(projection, "gate") == 0) return Qwen3FfnProfileStage::gate_projection;
    if (std::strcmp(projection, "up") == 0) return Qwen3FfnProfileStage::up_projection;
    if (std::strcmp(projection, "down") == 0) return Qwen3FfnProfileStage::down_projection;
    return Qwen3FfnProfileStage::count;
}

Qwen3BoundaryProfileStage projection_input_f16_stage(const char* projection) {
    if (std::strcmp(projection, "q") == 0) return Qwen3BoundaryProfileStage::q_input_f32_to_f16;
    if (std::strcmp(projection, "k") == 0) return Qwen3BoundaryProfileStage::k_input_f32_to_f16;
    if (std::strcmp(projection, "v") == 0) return Qwen3BoundaryProfileStage::v_input_f32_to_f16;
    if (std::strcmp(projection, "o") == 0) return Qwen3BoundaryProfileStage::o_input_f32_to_f16;
    if (std::strcmp(projection, "gate") == 0) return Qwen3BoundaryProfileStage::gate_input_f32_to_f16;
    if (std::strcmp(projection, "up") == 0) return Qwen3BoundaryProfileStage::up_input_f32_to_f16;
    if (std::strcmp(projection, "down") == 0) return Qwen3BoundaryProfileStage::down_input_f32_to_f16;
    return Qwen3BoundaryProfileStage::count;
}

Qwen3BoundaryProfileStage projection_input_q8_stage(const char* projection) {
    if (std::strcmp(projection, "q") == 0) return Qwen3BoundaryProfileStage::q_input_q8;
    if (std::strcmp(projection, "k") == 0) return Qwen3BoundaryProfileStage::k_input_q8;
    if (std::strcmp(projection, "v") == 0) return Qwen3BoundaryProfileStage::v_input_q8;
    if (std::strcmp(projection, "o") == 0) return Qwen3BoundaryProfileStage::o_input_q8;
    if (std::strcmp(projection, "gate") == 0) return Qwen3BoundaryProfileStage::gate_input_q8;
    if (std::strcmp(projection, "up") == 0) return Qwen3BoundaryProfileStage::up_input_q8;
    if (std::strcmp(projection, "down") == 0) return Qwen3BoundaryProfileStage::down_input_q8;
    return Qwen3BoundaryProfileStage::count;
}

Qwen3BoundaryProfileStage projection_output_f32_stage(const char* projection) {
    if (std::strcmp(projection, "q") == 0) return Qwen3BoundaryProfileStage::q_output_f16_to_f32;
    if (std::strcmp(projection, "k") == 0) return Qwen3BoundaryProfileStage::k_output_f16_to_f32;
    if (std::strcmp(projection, "v") == 0) return Qwen3BoundaryProfileStage::v_output_f16_to_f32;
    if (std::strcmp(projection, "o") == 0) return Qwen3BoundaryProfileStage::o_output_f16_to_f32;
    if (std::strcmp(projection, "gate") == 0) return Qwen3BoundaryProfileStage::gate_output_f16_to_f32;
    if (std::strcmp(projection, "up") == 0) return Qwen3BoundaryProfileStage::up_output_f16_to_f32;
    if (std::strcmp(projection, "down") == 0) return Qwen3BoundaryProfileStage::down_output_f16_to_f32;
    return Qwen3BoundaryProfileStage::count;
}

void quantize_projection_input(
    const float* input,
    std::uint32_t input_elements,
    __half* input_half,
    Q8_1Block* input_q8,
    Q8ExactBlock* input_q8_exact,
    const char* projection,
    Qwen3GpuProfile* profile = nullptr) {
    const auto ffn_quantization_stage = projection_ffn_quantization_stage(projection);
    const auto input_f16_stage = projection_input_f16_stage(projection);
    const auto input_q8_stage = projection_input_q8_stage(projection);
    const auto q8_bytes = static_cast<std::size_t>(input_elements) / kQ8_1BlockSize
        * sizeof(Q8ExactBlock);
    const bool exact_metadata = use_exact_q8_metadata(projection);
    const bool f32_input = projection_selected("MIINFER_F32_INPUT_PROJECTIONS", projection);
    if (!exact_metadata && f32_input) {
        profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
            launch_q8_1_quantize_f32(input, input_q8, static_cast<int>(input_elements));
        }, ffn_quantization_stage, input_q8_stage, q8_bytes);
    } else if (f32_input) {
        profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
            launch_q8_exact_quantize_f32(input, input_q8_exact, static_cast<int>(input_elements));
        }, ffn_quantization_stage, input_q8_stage, q8_bytes);
    } else {
        profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
            launch_qwen3_f32_to_f16(input, input_half, input_elements);
        }, ffn_quantization_stage, input_f16_stage,
           static_cast<std::size_t>(input_elements) * sizeof(__half));
        if (exact_metadata) {
            profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
                launch_q8_exact_quantize(input_half, input_q8_exact, static_cast<int>(input_elements));
            }, ffn_quantization_stage, input_q8_stage, q8_bytes);
        } else {
            profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
                launch_q8_1_quantize(input_half, input_q8, static_cast<int>(input_elements));
            }, ffn_quantization_stage, input_q8_stage, q8_bytes);
        }
    }
}

void verify_q8_reuse_buffers(
    const void* first,
    const void* second,
    std::size_t bytes,
    Qwen3GpuProfile* profile) {
    std::vector<std::byte> first_host(bytes);
    std::vector<std::byte> second_host(bytes);
    MIINFER_HIP_CHECK(hipMemcpy(first_host.data(), first, bytes, hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(second_host.data(), second, bytes, hipMemcpyDeviceToHost));
    if (profile != nullptr) ++profile->gate_up_q8_reuse_checks;
    if (std::memcmp(first_host.data(), second_host.data(), bytes) != 0) {
        if (profile != nullptr) ++profile->gate_up_q8_reuse_mismatches;
        throw std::runtime_error("Gate and Up Q8 activation buffers differ");
    }
}

void verify_fused_ffn_norm_buffers(
    const void* fused_f16,
    const void* separate_f16,
    const void* fused_q8,
    const void* separate_q8,
    std::size_t f16_bytes,
    std::size_t q8_bytes,
    Qwen3GpuProfile* profile) {
    std::vector<std::byte> fused_f16_host(f16_bytes);
    std::vector<std::byte> separate_f16_host(f16_bytes);
    std::vector<std::byte> fused_q8_host(q8_bytes);
    std::vector<std::byte> separate_q8_host(q8_bytes);
    MIINFER_HIP_CHECK(hipMemcpy(fused_f16_host.data(), fused_f16, f16_bytes,
                                hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(separate_f16_host.data(), separate_f16, f16_bytes,
                                hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(fused_q8_host.data(), fused_q8, q8_bytes,
                                hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(separate_q8_host.data(), separate_q8, q8_bytes,
                                hipMemcpyDeviceToHost));
    if (profile != nullptr) {
        ++profile->ffn_norm_q8_fusion_f16_checks;
        ++profile->ffn_norm_q8_fusion_q8_checks;
    }
    if (std::memcmp(fused_f16_host.data(), separate_f16_host.data(), f16_bytes) != 0) {
        if (profile != nullptr) ++profile->ffn_norm_q8_fusion_f16_mismatches;
        std::size_t first_byte = 0;
        while (first_byte < f16_bytes
               && fused_f16_host[first_byte] == separate_f16_host[first_byte]) {
            ++first_byte;
        }
        const std::size_t element = first_byte / sizeof(__half);
        std::uint16_t fused_bits = 0;
        std::uint16_t separate_bits = 0;
        if (element < f16_bytes / sizeof(__half)) {
            std::memcpy(&fused_bits, fused_f16_host.data() + element * sizeof(__half),
                        sizeof(fused_bits));
            std::memcpy(&separate_bits, separate_f16_host.data() + element * sizeof(__half),
                        sizeof(separate_bits));
        }
        std::ostringstream message;
        message << "fused FFN norm FP16 materialization differs at element " << element
                << " (fused=0x" << std::hex << fused_bits
                << ", separate=0x" << separate_bits << ')';
        throw std::runtime_error(message.str());
    }
    if (std::memcmp(fused_q8_host.data(), separate_q8_host.data(), q8_bytes) != 0) {
        if (profile != nullptr) ++profile->ffn_norm_q8_fusion_q8_mismatches;
        std::size_t first_byte = 0;
        while (first_byte < q8_bytes
               && fused_q8_host[first_byte] == separate_q8_host[first_byte]) {
            ++first_byte;
        }
        std::ostringstream message;
        message << "fused FFN norm Q8 activation differs at byte " << first_byte;
        throw std::runtime_error(message.str());
    }
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
    Qwen3GpuProfile* profile = nullptr,
    bool input_prequantized = false) {
    const auto projection_category = projection_profile_category(projection);
    const auto ffn_projection_stage = projection_ffn_stage(projection);
    const auto output_f32_stage = projection_output_f32_stage(projection);
    const bool exact_metadata = use_exact_q8_metadata(projection);
    const bool f32_output = projection_selected("MIINFER_F32_OUTPUT_PROJECTIONS", projection);
    if (f32_output && !exact_metadata) {
        throw std::invalid_argument("F32 projection output requires Q8Exact metadata");
    }
    if (input_prequantized) {
        // The caller has already produced the exact Q8 block stream. This is
        // used only by the opt-in SwiGLU fusion candidate.
    } else {
        quantize_projection_input(input, input_elements, input_half, input_q8, input_q8_exact,
                                  projection, profile);
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
    }, ffn_projection_stage);
    if (!f32_output) {
        profile_gpu_call(profile, Qwen3ProfileCategory::conversion, 1, [&] {
            launch_qwen3_f16_to_f32(output_half, output, static_cast<std::uint32_t>(rows));
        }, Qwen3FfnProfileStage::count, output_f32_stage,
           static_cast<std::size_t>(rows) * sizeof(float));
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
    Qwen3GpuProfile* profile,
    Qwen3BoundaryProfileStage normalize_stage,
    Qwen3BoundaryProfileStage scale_stage) {
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_head_rms_normalize(input, rms, heads, head_dim, epsilon);
    }, Qwen3FfnProfileStage::count, normalize_stage,
       static_cast<std::size_t>(heads) * head_dim * sizeof(float));
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_head_mul(rms, weights, weighted, heads, head_dim);
    }, Qwen3FfnProfileStage::count, scale_stage,
       static_cast<std::size_t>(heads) * head_dim * sizeof(float));
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
          gate_up_q8_verify(kMaxQ8Bytes), ffn_norm_f16_verify(kMaxVector * sizeof(__half)),
          ffn_norm_q8_verify(kMaxQ8Bytes),
          final_norm(hidden), logits(vocab),
          quantized_final_norm(hidden / 256), argmax_token(1) {}

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
    // Only used by the opt-in C9c verifier; the production shared path uses
    // input_q8 or input_q8_exact directly and never copies these bytes.
    DeviceBytes gate_up_q8_verify;
    DeviceBytes ffn_norm_f16_verify;
    DeviceBytes ffn_norm_q8_verify;
    DeviceBuffer<float> final_norm;
    DeviceBuffer<float> logits;
    DeviceBuffer<Q8KDeviceBlock> quantized_final_norm;
    DeviceBuffer<std::uint32_t> argmax_token;
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
        if (!event.copy && event.ffn_stage != Qwen3FfnProfileStage::count) {
            const auto ffn_index = static_cast<std::size_t>(event.ffn_stage);
            ffn_gpu_ms[ffn_index] += milliseconds;
            ffn_dispatches[ffn_index] += event.dispatches;
        }
        if (!event.copy && event.boundary_stage != Qwen3BoundaryProfileStage::count) {
            const auto boundary_index = static_cast<std::size_t>(event.boundary_stage);
            boundary_gpu_ms[boundary_index] += milliseconds;
            boundary_dispatches[boundary_index] += event.dispatches;
            boundary_bytes[boundary_index] += event.bytes;
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
    if (use_coalesced_kv_cache_store()) {
        profile_gpu_call(profile, Qwen3ProfileCategory::kv_cache, 1, [&] {
            launch_qwen3_kv_cache_store(
                keys, values, static_cast<float*>(keys_), static_cast<float*>(values_),
                static_cast<std::uint32_t>(position), static_cast<std::uint32_t>(capacity_),
                static_cast<std::uint32_t>(kv_heads_), static_cast<std::uint32_t>(head_dim_));
        });
        ++length_;
        return;
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
    if (output_device != nullptr && output_device == input_device) {
        throw std::invalid_argument("Qwen3 GPU layer input and output buffers must differ");
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
    const bool direct_output = output_device != nullptr && !capture_trace
        && use_direct_layer_output_handoff();
    float* layer_output_target = direct_output ? output_device : layer_output.data();

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
    }, Qwen3FfnProfileStage::count, Qwen3BoundaryProfileStage::attn_rms_normalize,
       hidden * sizeof(float));
    trace.attn_rms = capture_optional(buffers.attn_rms.data(), hidden);
    profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
        launch_qwen3_elementwise_mul(buffers.attn_rms.data(), attention_norm_weights, buffers.attn_norm.data(),
                                     config.hidden_size);
    }, Qwen3FfnProfileStage::count, Qwen3BoundaryProfileStage::attn_norm_scale,
       hidden * sizeof(float));
    trace.attn_norm = capture_optional(buffers.attn_norm.data(), hidden);

    launch_projection(plan, layer.q, buffers.attn_norm.data(), config.hidden_size, buffers.q.data(),
                      config.hidden_size, config.hidden_size, half_input, q8_input, q8_exact_input,
                      half_output, "q", profile);
    trace.q_projection = capture_optional(buffers.q.data(), hidden);
    trace.q_reshape = trace.q_projection;
    const auto* q_norm_weights = device_f32_tensor(plan, layer.q_norm, head_dim);
    capture_qwen3_head_norm(buffers.q.data(), buffers.q_rms.data(), buffers.q_normed.data(), q_norm_weights,
                            config.attention_heads, config.head_dim, config.rms_epsilon, profile,
                            Qwen3BoundaryProfileStage::q_head_rms_normalize,
                            Qwen3BoundaryProfileStage::q_head_scale);
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
                            config.kv_heads, config.head_dim, config.rms_epsilon, profile,
                            Qwen3BoundaryProfileStage::k_head_rms_normalize,
                            Qwen3BoundaryProfileStage::k_head_scale);
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
        const auto attention_kernel = selected_attention_kernel();
        if (attention_kernel == AttentionKernel::history_parallel) {
            launch_qwen3_cached_attention_history_parallel(
                q_rope.data(), static_cast<const float*>(cache.device_keys()),
                static_cast<const float*>(cache.device_values()),
                static_cast<std::uint32_t>(attention_length),
                static_cast<std::uint32_t>(cache.capacity()), attention.data(), scores.data(),
                probabilities.data(), config.attention_heads, config.kv_heads, config.head_dim,
                1.0F / std::sqrt(static_cast<float>(config.head_dim)));
        } else if (attention_kernel == AttentionKernel::parallel) {
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
    }, Qwen3FfnProfileStage::count, Qwen3BoundaryProfileStage::attention_f32_to_f16,
       hidden * sizeof(__half));
    profile_gpu_call(profile, Qwen3ProfileCategory::conversion, 1, [&] {
        launch_qwen3_f16_to_f32(half_output, attention.data(), hidden);
    }, Qwen3FfnProfileStage::count, Qwen3BoundaryProfileStage::attention_f16_to_f32,
       hidden * sizeof(float));
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
    const bool shared_gate_up_q8 = !capture_trace && use_shared_gate_up_q8();
    const bool fused_ffn_norm_q8 = shared_gate_up_q8 && use_fused_ffn_norm_q8();
    if (fused_ffn_norm_q8) {
        const bool gate_exact_metadata = use_exact_q8_metadata("gate");
        const bool up_exact_metadata = use_exact_q8_metadata("up");
        const bool gate_f32_input = projection_selected("MIINFER_F32_INPUT_PROJECTIONS", "gate");
        const bool up_f32_input = projection_selected("MIINFER_F32_INPUT_PROJECTIONS", "up");
        if (!gate_exact_metadata || !up_exact_metadata || gate_f32_input || up_f32_input) {
            throw std::invalid_argument(
                "FFN norm/Q8 fusion requires exact Q8 metadata and FP16-input projections");
        }
        const bool verify = verify_fused_ffn_norm_q8();
        const auto q8_bytes = static_cast<std::size_t>(config.hidden_size)
            / kQ8_1BlockSize * sizeof(Q8ExactBlock);
        profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
            launch_qwen3_ffn_norm_to_q8_exact(
                ffn_input.data(), ffn_norm_weights,
                static_cast<Q8ExactBlock*>(buffers.input_q8_exact.data()),
                static_cast<int>(config.hidden_size), config.rms_epsilon,
                verify ? static_cast<__half*>(buffers.ffn_norm_f16_verify.data()) : nullptr);
        }, Qwen3FfnProfileStage::normalization,
           Qwen3BoundaryProfileStage::ffn_norm_to_shared_q8, q8_bytes);
        if (verify || verify_shared_gate_up_q8()) {
            // Recreate the old producer chain into verifier-only workspace.
            // This path is intentionally host-visible and is never part of
            // production execution when verification is disabled.
            launch_qwen3_rms_normalize(ffn_input.data(), ffn_rms.data(), config.hidden_size,
                                       config.rms_epsilon);
            launch_qwen3_elementwise_mul(ffn_rms.data(), ffn_norm_weights, ffn_norm.data(),
                                         config.hidden_size);
            launch_qwen3_f32_to_f16(
                ffn_norm.data(), half_input, config.hidden_size);
            launch_q8_exact_quantize(
                half_input,
                static_cast<Q8ExactBlock*>(buffers.ffn_norm_q8_verify.data()),
                config.hidden_size);
            if (verify) {
                verify_fused_ffn_norm_buffers(
                    buffers.ffn_norm_f16_verify.data(), half_input,
                    buffers.input_q8_exact.data(), buffers.ffn_norm_q8_verify.data(),
                    config.hidden_size * sizeof(__half), q8_bytes, profile);
            }
        }
    } else {
        profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
            launch_qwen3_rms_normalize(ffn_input.data(), ffn_rms.data(), config.hidden_size,
                                       config.rms_epsilon);
        }, Qwen3FfnProfileStage::normalization,
           Qwen3BoundaryProfileStage::ffn_rms_normalize, hidden * sizeof(float));
        trace.ffn_rms = capture_optional(ffn_rms.data(), hidden);
        profile_gpu_call(profile, Qwen3ProfileCategory::normalization, 1, [&] {
            launch_qwen3_elementwise_mul(ffn_rms.data(), ffn_norm_weights, ffn_norm.data(),
                                         config.hidden_size);
        }, Qwen3FfnProfileStage::normalization,
           Qwen3BoundaryProfileStage::ffn_norm_scale, hidden * sizeof(float));
        trace.ffn_norm = capture_optional(ffn_norm.data(), hidden);
    }
    if (shared_gate_up_q8) {
        const bool gate_exact_metadata = use_exact_q8_metadata("gate");
        const bool up_exact_metadata = use_exact_q8_metadata("up");
        const bool gate_f32_input = projection_selected("MIINFER_F32_INPUT_PROJECTIONS", "gate");
        const bool up_f32_input = projection_selected("MIINFER_F32_INPUT_PROJECTIONS", "up");
        if (gate_exact_metadata != up_exact_metadata || gate_f32_input != up_f32_input) {
            throw std::invalid_argument(
                "Gate/Up Q8 reuse requires identical quantization contracts");
        }
        if (!fused_ffn_norm_q8) {
            quantize_projection_input(ffn_norm.data(), config.hidden_size, half_input, q8_input,
                                      q8_exact_input, "gate", profile);
        }
        if (verify_shared_gate_up_q8()) {
            // Recreate the old two-call dataflow into a separate persistent
            // buffer.  The comparison is intentionally host-visible and only
            // enabled for the C9c diagnostic; production reuse never reads it.
            quantize_projection_input(
                ffn_norm.data(), config.hidden_size, half_input,
                static_cast<Q8_1Block*>(buffers.gate_up_q8_verify.data()),
                static_cast<Q8ExactBlock*>(buffers.gate_up_q8_verify.data()), "up", nullptr);
            const auto q8_bytes = static_cast<std::size_t>(config.hidden_size)
                / kQ8_1BlockSize * sizeof(Q8ExactBlock);
            const void* shared_buffer = gate_exact_metadata
                ? static_cast<const void*>(q8_exact_input)
                : static_cast<const void*>(q8_input);
            verify_q8_reuse_buffers(shared_buffer, buffers.gate_up_q8_verify.data(), q8_bytes,
                                    profile);
        }
        launch_projection(plan, layer.gate, fused_ffn_norm_q8 ? ffn_input.data() : ffn_norm.data(),
                          config.hidden_size, gate.data(),
                          config.intermediate_size, config.hidden_size, half_input, q8_input,
                          q8_exact_input, half_output, "gate", profile, true);
        trace.gate = capture_optional(gate.data(), intermediate);
        launch_projection(plan, layer.up, fused_ffn_norm_q8 ? ffn_input.data() : ffn_norm.data(),
                          config.hidden_size, up.data(),
                          config.intermediate_size, config.hidden_size, half_input, q8_input,
                          q8_exact_input, half_output, "up", profile, true);
        trace.up = capture_optional(up.data(), intermediate);
    } else {
        launch_projection(plan, layer.gate, ffn_norm.data(), config.hidden_size, gate.data(),
                          config.intermediate_size, config.hidden_size, half_input, q8_input,
                          q8_exact_input, half_output, "gate", profile);
        trace.gate = capture_optional(gate.data(), intermediate);
        launch_projection(plan, layer.up, ffn_norm.data(), config.hidden_size, up.data(),
                          config.intermediate_size, config.hidden_size, half_input, q8_input,
                          q8_exact_input, half_output, "up", profile);
        trace.up = capture_optional(up.data(), intermediate);
    }
    const bool fused_swiglu = !capture_trace && use_fused_swiglu_q8();
    const char* mutation = std::getenv("MIINFER_GPU_LAYER_MUTATE");
    if (fused_swiglu) {
        profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
            launch_silu_mul_q8_exact(gate.data(), up.data(), q8_exact_input,
                                     static_cast<int>(config.intermediate_size));
        }, Qwen3FfnProfileStage::swiglu_down_input_quantization);
    } else if (mutation != nullptr && std::strcmp(mutation, "swap-gate-up") == 0) {
        // Test-only discriminator.  It is opt-in through an environment
        // variable and never participates in normal execution.
        profile_gpu_call(profile, Qwen3ProfileCategory::activation, 1, [&] {
            launch_qwen3_silu_mul(up.data(), gate.data(), swiglu.data(), config.intermediate_size);
        }, Qwen3FfnProfileStage::swiglu);
    } else {
        profile_gpu_call(profile, Qwen3ProfileCategory::activation, 1, [&] {
            launch_qwen3_silu_mul(gate.data(), up.data(), swiglu.data(), config.intermediate_size);
        }, Qwen3FfnProfileStage::swiglu);
    }
    trace.swiglu = capture_optional(swiglu.data(), intermediate);
    launch_projection(plan, layer.down, swiglu.data(), config.intermediate_size, ffn_output.data(),
                      config.hidden_size, config.intermediate_size, half_input, q8_input, q8_exact_input,
                      half_output, "down", profile, fused_swiglu);
    trace.ffn_output = capture_optional(ffn_output.data(), hidden);
    profile_gpu_call(profile, Qwen3ProfileCategory::residual, 1, [&] {
        launch_qwen3_add(ffn_output.data(), ffn_input.data(), layer_output_target,
                         config.hidden_size);
    }, Qwen3FfnProfileStage::residual);
    trace.layer_output = capture_optional(layer_output_target, hidden);

    if (output_device != nullptr && !direct_output) {
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
    }, Qwen3FfnProfileStage::count, Qwen3BoundaryProfileStage::final_rms_norm,
       static_cast<std::size_t>(config.hidden_size) * sizeof(float));
    forward.final_norm = capture(workspace.final_norm.data(), config.hidden_size, profile);

    const auto* output_weight = static_cast<const Q6KDeviceBlock*>(
        plan.device_tensor_data(model.output().name()));
    profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
        launch_qwen3_q8_k_quantize(workspace.final_norm.data(), workspace.quantized_final_norm.data(),
                                   config.hidden_size);
    }, Qwen3FfnProfileStage::count, Qwen3BoundaryProfileStage::final_norm_to_q8k,
       static_cast<std::size_t>(config.hidden_size) / 256 * sizeof(Q8KDeviceBlock));
    profile_gpu_call(profile, Qwen3ProfileCategory::lm_head, 1, [&] {
        launch_qwen3_q6_k_q8_k_gemv(output_weight, workspace.quantized_final_norm.data(),
                                    workspace.logits.data(),
                                    config.vocab_size, config.hidden_size);
    });
    forward.logits = capture(workspace.logits.data(), config.vocab_size, profile);
    if (profile != nullptr) profile->finalize();
    return forward;
}

void execute_qwen3_decode_gpu_fast_impl(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    std::span<float> logits_host,
    std::uint32_t* token_host,
    Qwen3GpuProfile* profile) {
    const auto& model = plan.model();
    const auto& config = model.config();
    const bool greedy = token_host != nullptr;
    const char* lm_mmvq_env = std::getenv("MIINFER_LM_Q8_1_MMVQ");
    const bool lm_mmvq = lm_mmvq_env != nullptr && std::strcmp(lm_mmvq_env, "0") != 0;
    if (token >= config.vocab_size
        || (greedy ? !logits_host.empty() : logits_host.size() != config.vocab_size)
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
    }, Qwen3FfnProfileStage::count, Qwen3BoundaryProfileStage::final_rms_norm,
       static_cast<std::size_t>(config.hidden_size) * sizeof(float));

    const auto* output_weight = static_cast<const Q6KDeviceBlock*>(
        plan.device_tensor_data(model.output().name()));
    profile_gpu_call(profile, Qwen3ProfileCategory::quantization, 1, [&] {
        if (lm_mmvq) {
            launch_q8_1_quantize_f32(
                workspace.final_norm.data(), static_cast<Q8_1Block*>(workspace.input_q8.data()),
                config.hidden_size);
        } else {
            launch_qwen3_q8_k_quantize(workspace.final_norm.data(), workspace.quantized_final_norm.data(),
                                       config.hidden_size);
        }
    }, Qwen3FfnProfileStage::count, Qwen3BoundaryProfileStage::final_norm_to_q8k,
       static_cast<std::size_t>(config.hidden_size) / (lm_mmvq ? 32 : 256)
           * (lm_mmvq ? sizeof(Q8_1Block) : sizeof(Q8KDeviceBlock)));
    profile_gpu_call(profile, Qwen3ProfileCategory::lm_head, 1, [&] {
        if (lm_mmvq) {
            launch_qwen3_q6_k_q8_1_mmvq(
                output_weight, static_cast<const Q8_1Block*>(workspace.input_q8.data()),
                workspace.logits.data(), config.vocab_size, config.hidden_size);
        } else {
            launch_qwen3_q6_k_q8_k_gemv(output_weight, workspace.quantized_final_norm.data(),
                                        workspace.logits.data(),
                                        config.vocab_size, config.hidden_size);
        }
    });
    if (greedy) {
        profile_gpu_call(profile, Qwen3ProfileCategory::argmax, 1, [&] {
            launch_qwen3_argmax(workspace.logits.data(), workspace.argmax_token.data(),
                                config.vocab_size);
        });
        profile_copy_call(profile, sizeof(std::uint32_t), [&] {
            MIINFER_HIP_CHECK(hipMemcpy(token_host, workspace.argmax_token.data(),
                                        sizeof(std::uint32_t), hipMemcpyDeviceToHost));
        });
    } else {
        profile_copy_call(profile, logits_host.size() * sizeof(float), [&] {
            MIINFER_HIP_CHECK(hipMemcpy(logits_host.data(), workspace.logits.data(),
                                        config.vocab_size * sizeof(float), hipMemcpyDeviceToHost));
        });
    }
    if (profile != nullptr) profile->finalize();
}

void execute_qwen3_decode_gpu_fast(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    std::span<float> logits_host) {
    execute_qwen3_decode_gpu_fast_impl(plan, token, position, cache, logits_host, nullptr, nullptr);
}

void execute_qwen3_decode_gpu_fast(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    std::span<float> logits_host,
    Qwen3GpuProfile* profile) {
    execute_qwen3_decode_gpu_fast_impl(plan, token, position, cache, logits_host, nullptr, profile);
}

std::uint32_t execute_qwen3_decode_gpu_greedy(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache) {
    return execute_qwen3_decode_gpu_greedy(plan, token, position, cache, nullptr);
}

std::uint32_t execute_qwen3_decode_gpu_greedy(
    const Qwen3GpuPlan& plan,
    std::uint32_t token,
    std::size_t position,
    Qwen3GpuDecodeCache& cache,
    Qwen3GpuProfile* profile) {
    std::uint32_t result = 0;
    execute_qwen3_decode_gpu_fast_impl(plan, token, position, cache, {}, &result, profile);
    return result;
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
