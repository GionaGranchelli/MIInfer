#include "miinfer/q4_q8_zero_point_dot.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "The Q4/Q8 zero-point dot candidate is intentionally gfx906-only"
#endif

namespace miinfer {

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::uint32_t kNibbleMask = 0x0F0F0F0FU;

__device__ __forceinline__ int dot4_i8(int lhs, int rhs, int accumulator) {
    // ROCm Clang's signed AMDGCN intrinsic lowers to v_dot4_i32_i8 on gfx906.
    return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
}

__device__ __forceinline__ float zero_point_block_dot(
    const Q4_0Block& weight, const Q8_1Block& activation) {
    int raw_integer_dot = 0;
    for (int group = 0; group < kQ4_0BlockSize / 2; group += 4) {
        const auto q4_word = *reinterpret_cast<const std::uint32_t*>(weight.qs + group);
        const int q4_low = static_cast<int>(q4_word & kNibbleMask);
        const int q4_high = static_cast<int>((q4_word >> 4) & kNibbleMask);
        const int q8_low = *reinterpret_cast<const int*>(activation.qs + group);
        const int q8_high = *reinterpret_cast<const int*>(activation.qs + group + 16);
        raw_integer_dot = dot4_i8(q4_low, q8_low, raw_integer_dot);
        raw_integer_dot = dot4_i8(q4_high, q8_high, raw_integer_dot);
    }
    return __half2float(weight.d)
           * (__half2float(activation.d) * static_cast<float>(raw_integer_dot)
              - 8.0F * __half2float(activation.s));
}

__global__ void q4_q8_gemv_zero_point_dot_kernel(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    __shared__ float partials[kThreadsPerBlock];
    float sum = 0.0F;
    const int blocks_per_row = columns / kQ4_0BlockSize;
    const std::size_t row_offset = static_cast<std::size_t>(row) * blocks_per_row;
    for (int block = static_cast<int>(threadIdx.x); block < blocks_per_row;
         block += static_cast<int>(blockDim.x)) {
        const auto& weight = weights[row_offset + static_cast<std::size_t>(block)];
        const auto& activation = input[static_cast<std::size_t>(block)];
        // Q4 nibbles remain unsigned; Q8_1.s supplies the zero-point correction.
        sum += zero_point_block_dot(weight, activation);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = __float2half_rn(partials[0]);
    }
}

template <unsigned int kThreads>
__global__ void q4_q8_gemv_zero_point_dot_128_kernel(
    const Q4_0Block* weights, const Q8_1Block* input, __half* output,
    int rows, int columns) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    __shared__ float partials[kThreads];
    const int blocks_per_row = columns / kQ4_0BlockSize;
    float sum = 0.0F;
    for (int block = static_cast<int>(threadIdx.x); block < blocks_per_row;
         block += static_cast<int>(kThreads)) {
        sum += zero_point_block_dot(
            weights[static_cast<std::size_t>(row) * blocks_per_row + block], input[block]);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = __float2half_rn(partials[0]);
}

__global__ void q4_q8_gemv_zero_point_dot_wave64_kernel(
    const Q4_0Block* weights, const Q8_1Block* input, __half* output,
    int rows, int columns) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    __shared__ float wave_partials[2];
    const int blocks_per_row = columns / kQ4_0BlockSize;
    float sum = 0.0F;
    for (int block = static_cast<int>(threadIdx.x); block < blocks_per_row;
         block += 128) {
        sum += zero_point_block_dot(
            weights[static_cast<std::size_t>(row) * blocks_per_row + block], input[block]);
    }
    for (int offset = 32; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 64);
    }
    if ((threadIdx.x & 63) == 0) wave_partials[threadIdx.x / 64] = sum;
    __syncthreads();
    if (threadIdx.x == 0) output[row] = __float2half_rn(wave_partials[0] + wave_partials[1]);
}

}  // namespace

void launch_q4_q8_gemv_zero_point_dot(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream) {
    if (rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        q4_q8_gemv_zero_point_dot_kernel,
        dim3(static_cast<unsigned int>(rows)),
        dim3(kThreadsPerBlock),
        kThreadsPerBlock * sizeof(float),
        stream,
        weights,
        input,
        output,
        rows,
        columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_128(
    const Q4_0Block* weights, const Q8_1Block* input, __half* output,
    int rows, int columns, hipStream_t stream) {
    if (rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(q4_q8_gemv_zero_point_dot_128_kernel<128>, dim3(rows), dim3(128),
                       128 * sizeof(float), stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_wave64(
    const Q4_0Block* weights, const Q8_1Block* input, __half* output,
    int rows, int columns, hipStream_t stream) {
    if (rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(q4_q8_gemv_zero_point_dot_wave64_kernel, dim3(rows), dim3(128), 0,
                       stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

}  // namespace miinfer
