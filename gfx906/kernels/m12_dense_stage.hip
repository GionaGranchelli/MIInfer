#include "miinfer/m12_dense_stage.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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

} // namespace miinfer
