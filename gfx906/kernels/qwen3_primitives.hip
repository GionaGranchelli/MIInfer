#include "miinfer/qwen3_gpu_primitives.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "Qwen3 primitives are intentionally gfx906-only"
#endif

namespace miinfer {

namespace {

__device__ __forceinline__ float half_to_float(__half value) {
    return __half2float(value);
}

__device__ __forceinline__ float q4_value(const std::byte* block, int index) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(block);
    const float scale = half_to_float(*reinterpret_cast<const __half*>(bytes));
    const std::uint8_t packed = bytes[2 + (index < 16 ? index : index - 16)];
    const int nibble = index < 16 ? packed & 0x0f : packed >> 4;
    return scale * static_cast<float>(nibble - 8);
}

__device__ __forceinline__ int q6_value(const Q6KDeviceBlock& block, int index) {
    const int quarter = index / 32;
    const int in_quarter = index % 32;
    const int low_index = in_quarter + (quarter >= 2 ? 32 : 0);
    const int low = quarter % 2 == 0 ? block.ql[low_index] & 0x0f : block.ql[low_index] >> 4;
    const int high = (block.qh[in_quarter] >> (2 * quarter)) & 0x03;
    return (low | (high << 4)) - 32;
}

__global__ void qwen3_q4_embedding_kernel(
    const std::byte* weights,
    std::uint32_t token,
    std::uint32_t vocabulary,
    std::uint32_t hidden_size,
    float* output) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_size || token >= vocabulary) return;
    const std::uint32_t blocks_per_row = hidden_size / 32;
    const std::uint32_t block_index = token * blocks_per_row + index / 32;
    output[index] = q4_value(weights + static_cast<std::size_t>(block_index) * 18, index % 32);
}

__global__ void qwen3_rms_norm_kernel(
    const float* input,
    const float* weights,
    float* output,
    std::uint32_t elements,
    float epsilon) {
    __shared__ float partials[256];
    float sum = 0.0F;
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        sum += input[index] * input[index];
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(partials[0] / static_cast<float>(elements) + epsilon);
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        output[index] = input[index] * inverse_rms * weights[index];
    }
}

__global__ void qwen3_q6_k_gemv_kernel(
    const Q6KDeviceBlock* weights,
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partials[256];
    float sum = 0.0F;
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (std::uint32_t block = threadIdx.x; block < blocks_per_row; block += blockDim.x) {
        const auto& quant = row_weights[block];
        const float block_scale = half_to_float(quant.d);
        const std::uint32_t input_offset = block * 256;
        for (int index = 0; index < 256; ++index) {
            const float value = block_scale
                                * static_cast<float>(quant.scales[index / 16])
                                * static_cast<float>(q6_value(quant, index));
            sum += value * input[input_offset + index];
        }
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = partials[0];
}

}  // namespace

void launch_qwen3_q4_embedding(
    const std::byte* weights,
    std::uint32_t token,
    std::uint32_t vocabulary,
    std::uint32_t hidden_size,
    float* output,
    hipStream_t stream) {
    if (weights == nullptr || output == nullptr || vocabulary == 0 || hidden_size == 0
        || hidden_size % 32 != 0 || token >= vocabulary) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q4_embedding_kernel,
                       dim3((hidden_size + 255) / 256), dim3(256), 0, stream,
                       weights, token, vocabulary, hidden_size, output);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_rms_norm(
    const float* input,
    const float* weights,
    float* output,
    std::uint32_t elements,
    float epsilon,
    hipStream_t stream) {
    if (input == nullptr || weights == nullptr || output == nullptr || elements == 0
        || elements > 256U * 1024U || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_rms_norm_kernel, dim3(1), dim3(256), 0, stream,
                       input, weights, output, elements, epsilon);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q6_k_gemv(
    const Q6KDeviceBlock* weights,
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q6_k_gemv_kernel, dim3(rows), dim3(256), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

}  // namespace miinfer
