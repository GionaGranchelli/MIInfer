#pragma once

#include "miinfer/qwen3_model.hpp"
#include "miinfer/device_validation.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>

namespace miinfer {

enum class Q4GemvKernel {
    zero_point_128,
    zero_point_128_wave64,
    zero_point_256,
};

[[nodiscard]] const char* q4_gemv_kernel_name(Q4GemvKernel kernel) noexcept;

struct StaticBuffer {
    std::string name;
    std::size_t bytes = 0;
    std::size_t offset = 0;
};

struct PlannedTensor {
    std::string name;
    GgufTensorType type;
    std::vector<std::uint64_t> dimensions;
    std::size_t bytes = 0;
    std::size_t offset = 0;
};

class GpuWeightArena {
public:
    GpuWeightArena() = default;
    explicit GpuWeightArena(std::size_t bytes);
    GpuWeightArena(const GpuWeightArena&) = delete;
    GpuWeightArena& operator=(const GpuWeightArena&) = delete;
    GpuWeightArena(GpuWeightArena&& other) noexcept;
    GpuWeightArena& operator=(GpuWeightArena&& other) noexcept;
    ~GpuWeightArena();

    void allocate(std::size_t bytes);
    void upload(std::size_t offset, const std::byte* data, std::size_t bytes);
    [[nodiscard]] void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
    void release() noexcept;

    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

class Qwen3GpuPlan {
public:
    Qwen3GpuPlan() = default;
    static Qwen3GpuPlan build(const Qwen3Model& model);

    Qwen3GpuPlan(const Qwen3GpuPlan&) = delete;
    Qwen3GpuPlan& operator=(const Qwen3GpuPlan&) = delete;
    Qwen3GpuPlan(Qwen3GpuPlan&&) noexcept = default;
    Qwen3GpuPlan& operator=(Qwen3GpuPlan&&) noexcept = default;

    [[nodiscard]] const Qwen3Model& model() const noexcept { return *model_; }
    [[nodiscard]] const GpuWeightArena& weights() const noexcept { return weights_; }
    [[nodiscard]] const std::vector<PlannedTensor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const std::vector<StaticBuffer>& buffers() const noexcept { return buffers_; }
    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }
    [[nodiscard]] bool verify_tensor_sample(
        const std::string& name,
        std::size_t sample_bytes = 4096) const;
    [[nodiscard]] Q4GemvKernel kernel_for(const std::string& projection) const;
    [[nodiscard]] const DeviceInfo& device() const noexcept { return device_; }

private:
    const Qwen3Model* model_ = nullptr;
    GpuWeightArena weights_;
    std::vector<PlannedTensor> tensors_;
    std::vector<StaticBuffer> buffers_;
    std::size_t workspace_bytes_ = 0;
    DeviceInfo device_;
};

}  // namespace miinfer
