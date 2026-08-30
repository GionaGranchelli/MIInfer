#include "miinfer/model_plan.hpp"

#include "miinfer/hip_check.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace miinfer {

namespace {

constexpr std::size_t kTensorAlignment = 256;

std::size_t align_up(std::size_t value, std::size_t alignment) {
    const auto remainder = value % alignment;
    if (remainder == 0) return value;
    if (alignment - remainder > std::numeric_limits<std::size_t>::max() - value) {
        throw std::runtime_error("GPU weight arena size overflows");
    }
    return value + alignment - remainder;
}

std::size_t checked_add(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::runtime_error("GPU memory plan size overflows");
    }
    return left + right;
}

const PlannedTensor& find_planned_tensor(
    const std::vector<PlannedTensor>& tensors,
    const std::string& name) {
    const auto found = std::find_if(tensors.begin(), tensors.end(),
                                    [&](const PlannedTensor& tensor) { return tensor.name == name; });
    if (found == tensors.end()) throw std::invalid_argument("unknown planned tensor: " + name);
    return *found;
}

const GgufTensor& find_source_tensor(const Qwen3Model& model, const std::string& name) {
    const auto found = std::find_if(model.tensors().begin(), model.tensors().end(),
                                    [&](const GgufTensor& tensor) { return tensor.name == name; });
    if (found == model.tensors().end()) throw std::invalid_argument("unknown source tensor: " + name);
    return *found;
}

void hip_throw(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + hipGetErrorString(error));
    }
}

}  // namespace

const char* q4_gemv_kernel_name(Q4GemvKernel kernel) noexcept {
    switch (kernel) {
    case Q4GemvKernel::zero_point_128: return "zero-point-128";
    case Q4GemvKernel::zero_point_128_wave64: return "zero-point-wave64";
    case Q4GemvKernel::zero_point_256: return "zero-point-256";
    }
    return "unknown";
}

GpuWeightArena::GpuWeightArena(std::size_t bytes) {
    allocate(bytes);
}

GpuWeightArena::GpuWeightArena(GpuWeightArena&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
}

GpuWeightArena& GpuWeightArena::operator=(GpuWeightArena&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        bytes_ = other.bytes_;
        other.data_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

GpuWeightArena::~GpuWeightArena() {
    release();
}

void GpuWeightArena::allocate(std::size_t bytes) {
    if (data_ != nullptr) throw std::logic_error("GPU weight arena already allocated");
    if (bytes == 0) throw std::invalid_argument("GPU weight arena cannot be empty");
    hip_throw(hipMalloc(&data_, bytes), "hipMalloc(weight arena)");
    bytes_ = bytes;
}

void GpuWeightArena::upload(std::size_t offset, const std::byte* data, std::size_t bytes) {
    if (data_ == nullptr || bytes > bytes_ || offset > bytes_ - bytes) {
        throw std::out_of_range("GPU weight upload is outside the arena");
    }
    hip_throw(hipMemcpy(static_cast<std::byte*>(data_) + offset, data, bytes,
                       hipMemcpyHostToDevice), "hipMemcpy(weight tensor)");
}

void GpuWeightArena::release() noexcept {
    if (data_ != nullptr) {
        (void)hipFree(data_);
        data_ = nullptr;
        bytes_ = 0;
    }
}

Qwen3GpuPlan Qwen3GpuPlan::build(const Qwen3Model& model) {
    Qwen3GpuPlan plan;
    plan.model_ = &model;
    std::string device_error;
    if (!validate_gfx906_device(-1, plan.device_, device_error)) {
        throw std::runtime_error("cannot construct GPU model plan: " + device_error);
    }

    std::size_t arena_bytes = 0;
    plan.tensors_.reserve(model.tensors().size());
    for (const auto& tensor : model.tensors()) {
        arena_bytes = align_up(arena_bytes, kTensorAlignment);
        plan.tensors_.push_back({tensor.name, tensor.type, tensor.dimensions, tensor.byte_size, arena_bytes});
        arena_bytes = checked_add(arena_bytes, tensor.byte_size);
    }
    plan.weights_.allocate(arena_bytes);
    for (std::size_t index = 0; index < model.tensors().size(); ++index) {
        plan.weights_.upload(plan.tensors_[index].offset, model.tensors()[index].data,
                            model.tensors()[index].byte_size);
    }

    std::size_t buffer_offset = 0;
    const auto add_buffer = [&](const char* name, std::size_t bytes) {
        buffer_offset = align_up(buffer_offset, 256);
        plan.buffers_.push_back({name, bytes, buffer_offset});
        buffer_offset = checked_add(buffer_offset, bytes);
    };
    add_buffer("hidden_fp16", static_cast<std::size_t>(model.config().hidden_size) * sizeof(std::uint16_t));
    add_buffer("q_output_fp16", static_cast<std::size_t>(4096) * sizeof(std::uint16_t));
    add_buffer("k_output_fp16", static_cast<std::size_t>(1024) * sizeof(std::uint16_t));
    add_buffer("v_output_fp16", static_cast<std::size_t>(1024) * sizeof(std::uint16_t));
    add_buffer("ffn_intermediate_fp16", static_cast<std::size_t>(12288) * sizeof(std::uint16_t));
    add_buffer("q8_hidden", (4096U / 32U) * 36U);
    add_buffer("q8_intermediate", (12288U / 32U) * 36U);
    plan.workspace_bytes_ = buffer_offset;
    return plan;
}

Q4GemvKernel Qwen3GpuPlan::kernel_for(const std::string& projection) const {
    if (projection == "k" || projection == "v") return Q4GemvKernel::zero_point_128_wave64;
    if (projection == "down") return Q4GemvKernel::zero_point_256;
    if (projection == "q" || projection == "o" || projection == "gate" || projection == "up") {
        return Q4GemvKernel::zero_point_128;
    }
    throw std::invalid_argument("unknown Qwen3 projection: " + projection);
}

bool Qwen3GpuPlan::verify_tensor_sample(const std::string& name, std::size_t sample_bytes) const {
    const auto& planned = find_planned_tensor(tensors_, name);
    const auto& source = find_source_tensor(*model_, name);
    const auto count = std::min(sample_bytes, planned.bytes);
    std::vector<std::byte> host(count);
    hip_throw(hipMemcpy(host.data(), static_cast<const std::byte*>(weights_.data()) + planned.offset,
                        count, hipMemcpyDeviceToHost), "hipMemcpy(weight verification)");
    if (std::memcmp(host.data(), source.data, count) != 0) return false;
    if (planned.bytes > count) {
        const auto tail_offset = planned.bytes - count;
        hip_throw(hipMemcpy(host.data(), static_cast<const std::byte*>(weights_.data())
                                      + planned.offset + tail_offset,
                            count, hipMemcpyDeviceToHost), "hipMemcpy(weight tail verification)");
        return std::memcmp(host.data(), source.data + tail_offset, count) == 0;
    }
    return true;
}

}  // namespace miinfer
