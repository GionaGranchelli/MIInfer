#include "miinfer/hip_check.hpp"
#include "miinfer/vector_add.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <vector>

namespace miinfer {

namespace {

__global__ void vector_add_kernel(const float* left, const float* right, float* result,
                                  std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        result[index] = left[index] + right[index];
    }
}

constexpr unsigned int kThreadsPerBlock = 256;

}  // namespace

void launch_vector_add(const float* left, const float* right, float* result, std::size_t count) {
    const auto blocks = static_cast<unsigned int>((count + kThreadsPerBlock - 1)
                                                  / kThreadsPerBlock);
    hipLaunchKernelGGL(
        vector_add_kernel,
        dim3(blocks),
        dim3(kThreadsPerBlock),
        0,
        nullptr,
        left,
        right,
        result,
        count);
    MIINFER_HIP_CHECK(hipGetLastError());
}

bool run_vector_add(std::size_t count, float value, float tolerance) {
    std::vector<float> left(count, value);
    std::vector<float> right(count, value * 2.0F);
    std::vector<float> result(count, 0.0F);

    float* device_left = nullptr;
    float* device_right = nullptr;
    float* device_result = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&device_left, count * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&device_right, count * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&device_result, count * sizeof(float)));
    MIINFER_HIP_CHECK(
        hipMemcpy(device_left, left.data(), count * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(
        hipMemcpy(device_right, right.data(), count * sizeof(float), hipMemcpyHostToDevice));

    launch_vector_add(device_left, device_right, device_result, count);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    MIINFER_HIP_CHECK(
        hipMemcpy(result.data(), device_result, count * sizeof(float), hipMemcpyDeviceToHost));

    MIINFER_HIP_CHECK(hipFree(device_result));
    MIINFER_HIP_CHECK(hipFree(device_right));
    MIINFER_HIP_CHECK(hipFree(device_left));

    const float expected = value * 3.0F;
    for (const float actual : result) {
        if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
            return false;
        }
    }
    return true;
}

}  // namespace miinfer
