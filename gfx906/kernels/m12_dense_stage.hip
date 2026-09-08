#include "miinfer/m12_dense_stage.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <limits>

namespace {

__device__ __forceinline__ void q4k_scale_min_m12(
    const miinfer::Q4KDeviceBlock& block, int group, int& scale, int& minimum) {
    if (group < 4) {
        scale = block.scales[group] & 63;
        minimum = block.scales[group + 4] & 63;
    } else {
        scale = (block.scales[group + 4] & 0x0f)
                | ((block.scales[group - 4] >> 6) << 4);
        minimum = (block.scales[group + 4] >> 4)
                  | ((block.scales[group] >> 6) << 4);
    }
}

__device__ __forceinline__ float q4k_value_m12(
    const miinfer::Q4KDeviceBlock& block, int index) {
    const int group = index / 32;
    const int q_index = (index / 64) * 32 + index % 32;
    const std::uint8_t packed = block.qs[q_index];
    const int quantized = (group & 1) == 0 ? packed & 0x0f : packed >> 4;
    int scale = 0;
    int minimum = 0;
    q4k_scale_min_m12(block, group, scale, minimum);
    return __half2float(block.d) * static_cast<float>(scale * quantized)
           - __half2float(block.dmin) * static_cast<float>(minimum);
}

__global__ void m12_q4k_to_fp16_kernel(
    const miinfer::Q4KDeviceBlock* source,
    __half* destination,
    std::uint32_t rows,
    std::uint32_t columns) {
    constexpr std::uint32_t kBlockSize = 256;
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t row = block / (columns / kBlockSize);
    const std::uint32_t block_in_row = block % (columns / kBlockSize);
    if (row >= rows) return;
    const auto& quant = source[block];
    const std::uint32_t index = threadIdx.x;
    destination[static_cast<std::size_t>(row) * columns
                + block_in_row * kBlockSize + index] =
        __float2half_rn(q4k_value_m12(quant, static_cast<int>(index)));
}

__device__ __forceinline__ float q6k_value_m12(
    const miinfer::Q6KDeviceBlock& block, int index) {
    const int group = index / 128;
    const int lane = index % 128;
    const int quarter = lane / 32;
    const int in_quarter = lane % 32;
    const int low_index = group * 64 + in_quarter
        + (quarter == 1 || quarter == 3 ? 32 : 0);
    const int low = quarter < 2 ? block.ql[low_index] & 0x0f : block.ql[low_index] >> 4;
    const int high = (block.qh[group * 32 + in_quarter] >> (2 * quarter)) & 0x03;
    return __half2float(block.d) * static_cast<float>(block.scales[index / 16])
        * static_cast<float>((low | (high << 4)) - 32);
}

__global__ void m12_q6k_to_fp16_kernel(
    const miinfer::Q6KDeviceBlock* source,
    __half* destination,
    std::uint32_t rows,
    std::uint32_t columns) {
    constexpr std::uint32_t kBlockSize = 256;
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t row = block / (columns / kBlockSize);
    const std::uint32_t block_in_row = block % (columns / kBlockSize);
    if (row >= rows) return;
    const auto& quant = source[block];
    const std::uint32_t index = threadIdx.x;
    destination[static_cast<std::size_t>(row) * columns
                + block_in_row * kBlockSize + index] =
        __float2half_rn(q6k_value_m12(quant, static_cast<int>(index)));
}

__global__ void m12_f32_to_fp16_kernel(
    const float* source, __half* destination, std::uint32_t elements) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) destination[index] = __float2half_rn(source[index]);
}

} // namespace

namespace miinfer {

void launch_m12_q4k_to_fp16(
    const Q4KDeviceBlock* source,
    __half* destination,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (source == nullptr || destination == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const dim3 grid(rows * (columns / 256));
    const dim3 block(256);
    hipLaunchKernelGGL(m12_q4k_to_fp16_kernel, grid, block, 0, stream,
                       source, destination, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m12_q6k_to_fp16(
    const Q6KDeviceBlock* source, __half* destination,
    std::uint32_t rows, std::uint32_t columns, hipStream_t stream) {
    if (source == nullptr || destination == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const dim3 grid(rows * (columns / 256));
    const dim3 block(256);
    hipLaunchKernelGGL(m12_q6k_to_fp16_kernel, grid, block, 0, stream,
                       source, destination, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m12_f32_to_fp16(
    const float* source, __half* destination, std::size_t elements, hipStream_t stream) {
    if (source == nullptr || destination == nullptr || elements == 0
        || elements > std::numeric_limits<std::uint32_t>::max()) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    constexpr std::uint32_t kThreads = 256;
    const auto count = static_cast<std::uint32_t>(elements);
    hipLaunchKernelGGL(m12_f32_to_fp16_kernel,
                       dim3((count + kThreads - 1) / kThreads), dim3(kThreads), 0, stream,
                       source, destination, count);
    MIINFER_HIP_CHECK(hipGetLastError());
}

} // namespace miinfer
