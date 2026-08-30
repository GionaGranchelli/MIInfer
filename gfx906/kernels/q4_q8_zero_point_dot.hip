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

__device__ __forceinline__ float exact_sum_block_dot(
    const Q4_0Block& weight, const Q8_1Block& activation) {
    int raw_integer_dot = 0;
    int q8_integer_sum = 0;
    for (int group = 0; group < kQ4_0BlockSize / 2; group += 4) {
        const auto q4_word = *reinterpret_cast<const std::uint32_t*>(weight.qs + group);
        const int q4_low = static_cast<int>(q4_word & kNibbleMask);
        const int q4_high = static_cast<int>((q4_word >> 4) & kNibbleMask);
        const int q8_low = *reinterpret_cast<const int*>(activation.qs + group);
        const int q8_high = *reinterpret_cast<const int*>(activation.qs + group + 16);
        raw_integer_dot = dot4_i8(q4_low, q8_low, raw_integer_dot);
        raw_integer_dot = dot4_i8(q4_high, q8_high, raw_integer_dot);
    }
    for (int index = 0; index < kQ8_1BlockSize; ++index) {
        q8_integer_sum += static_cast<int>(activation.qs[index]);
    }
    return __half2float(weight.d) * __half2float(activation.d)
           * static_cast<float>(raw_integer_dot - 8 * q8_integer_sum);
}

__device__ __forceinline__ float exact_sum_block_dot(
    const Q4_0Block& weight, const Q8ExactBlock& activation) {
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
    return __half2float(weight.d) * __half2float(activation.d)
           * static_cast<float>(raw_integer_dot - 8 * static_cast<int>(activation.sum));
}

__device__ __forceinline__ float direct_signed_block_dot(
    const Q4_0Block& weight, const Q8_1Block& activation) {
    float sum = 0.0F;
    for (int index = 0; index < kQ4_0BlockSize; ++index) {
        const auto packed = weight.qs[index < 16 ? index : index - 16];
        const int nibble = index < 16 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
        sum += static_cast<float>(nibble - 8) * static_cast<float>(activation.qs[index]);
    }
    return __half2float(weight.d) * __half2float(activation.d) * sum;
}

__device__ __forceinline__ void store_result(__half* output, int index, float value) {
    output[index] = __float2half_rn(value);
}

__device__ __forceinline__ void store_result(float* output, int index, float value) {
    output[index] = value;
}

template <typename Input, typename Output, bool ExactSum>
__global__ void q4_q8_gemv_zero_point_dot_kernel(
    const Q4_0Block* weights,
    const Input* input,
    Output* output,
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
        // The old control uses Q8_1.s; production uses the exact integer
        // lane sum so FP16 metadata rounding cannot accumulate in deep FFN.
        if constexpr (ExactSum) {
            sum += exact_sum_block_dot(weight, activation);
        } else {
            sum += zero_point_block_dot(weight, activation);
        }
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
        store_result(output, row, partials[0]);
    }
}

template <typename BlockDot, typename Output>
__global__ void q4_q8_gemv_contract_probe_kernel(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    Output* output,
    int rows,
    int columns,
    BlockDot block_dot) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    __shared__ float partials[kThreadsPerBlock];
    const int blocks_per_row = columns / kQ4_0BlockSize;
    const std::size_t row_offset = static_cast<std::size_t>(row) * blocks_per_row;
    float sum = 0.0F;
    for (int block = static_cast<int>(threadIdx.x); block < blocks_per_row;
         block += static_cast<int>(blockDim.x)) {
        sum += block_dot(weights[row_offset + static_cast<std::size_t>(block)], input[block]);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) store_result(output, row, partials[0]);
}

struct ExactSumBlockDot {
    __device__ float operator()(const Q4_0Block& weight, const Q8_1Block& input) const {
        return exact_sum_block_dot(weight, input);
    }
};

struct DirectSignedBlockDot {
    __device__ float operator()(const Q4_0Block& weight, const Q8_1Block& input) const {
        return direct_signed_block_dot(weight, input);
    }
};

template <unsigned int kThreads, typename Input, typename Output, bool ExactSum>
__global__ void q4_q8_gemv_zero_point_dot_128_kernel(
    const Q4_0Block* weights, const Input* input, Output* output,
    int rows, int columns) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    __shared__ float partials[kThreads];
    const int blocks_per_row = columns / kQ4_0BlockSize;
    float sum = 0.0F;
    for (int block = static_cast<int>(threadIdx.x); block < blocks_per_row;
         block += static_cast<int>(kThreads)) {
        const auto& weight = weights[static_cast<std::size_t>(row) * blocks_per_row + block];
        if constexpr (ExactSum) {
            sum += exact_sum_block_dot(weight, input[block]);
        } else {
            sum += zero_point_block_dot(weight, input[block]);
        }
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) store_result(output, row, partials[0]);
}

template <typename Input, typename Output, bool ExactSum>
__global__ void q4_q8_gemv_zero_point_dot_wave64_kernel(
    const Q4_0Block* weights, const Input* input, Output* output,
    int rows, int columns) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    __shared__ float wave_partials[2];
    const int blocks_per_row = columns / kQ4_0BlockSize;
    float sum = 0.0F;
    for (int block = static_cast<int>(threadIdx.x); block < blocks_per_row;
         block += 128) {
        const auto& weight = weights[static_cast<std::size_t>(row) * blocks_per_row + block];
        if constexpr (ExactSum) {
            sum += exact_sum_block_dot(weight, input[block]);
        } else {
            sum += zero_point_block_dot(weight, input[block]);
        }
    }
    for (int offset = 32; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 64);
    }
    if ((threadIdx.x & 63) == 0) wave_partials[threadIdx.x / 64] = sum;
    __syncthreads();
    if (threadIdx.x == 0) store_result(output, row, wave_partials[0] + wave_partials[1]);
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
        (q4_q8_gemv_zero_point_dot_kernel<Q8_1Block, __half, false>),
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
    hipLaunchKernelGGL((q4_q8_gemv_zero_point_dot_128_kernel<128, Q8_1Block, __half, false>), dim3(rows), dim3(128),
                       128 * sizeof(float), stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_wave64(
    const Q4_0Block* weights, const Q8_1Block* input, __half* output,
    int rows, int columns, hipStream_t stream) {
    if (rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((q4_q8_gemv_zero_point_dot_wave64_kernel<Q8_1Block, __half, false>), dim3(rows), dim3(128), 0,
                       stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_exact_sum(
    const Q4_0Block* weights, const Q8_1Block* input, __half* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        (q4_q8_gemv_zero_point_dot_kernel<Q8_1Block, __half, true>), dim3(static_cast<unsigned int>(rows)),
        dim3(kThreadsPerBlock), kThreadsPerBlock * sizeof(float), stream, weights, input,
        output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_128_exact_sum(
    const Q4_0Block* weights, const Q8_1Block* input, __half* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((q4_q8_gemv_zero_point_dot_128_kernel<128, Q8_1Block, __half, true>),
                       dim3(static_cast<unsigned int>(rows)), dim3(128), 128 * sizeof(float),
                       stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_wave64_exact_sum(
    const Q4_0Block* weights, const Q8_1Block* input, __half* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((q4_q8_gemv_zero_point_dot_wave64_kernel<Q8_1Block, __half, true>),
                       dim3(static_cast<unsigned int>(rows)), dim3(128), 0, stream, weights,
                       input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_exact_metadata(
    const Q4_0Block* weights, const Q8ExactBlock* input, __half* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        (q4_q8_gemv_zero_point_dot_kernel<Q8ExactBlock, __half, true>),
        dim3(static_cast<unsigned int>(rows)), dim3(kThreadsPerBlock),
        kThreadsPerBlock * sizeof(float), stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_128_exact_metadata(
    const Q4_0Block* weights, const Q8ExactBlock* input, __half* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        (q4_q8_gemv_zero_point_dot_128_kernel<128, Q8ExactBlock, __half, true>),
        dim3(static_cast<unsigned int>(rows)), dim3(128), 128 * sizeof(float), stream,
        weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_wave64_exact_metadata(
    const Q4_0Block* weights, const Q8ExactBlock* input, __half* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        (q4_q8_gemv_zero_point_dot_wave64_kernel<Q8ExactBlock, __half, true>),
        dim3(static_cast<unsigned int>(rows)), dim3(128), 0, stream,
        weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_exact_metadata_f32(
    const Q4_0Block* weights, const Q8ExactBlock* input, float* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        (q4_q8_gemv_zero_point_dot_kernel<Q8ExactBlock, float, true>),
        dim3(static_cast<unsigned int>(rows)), dim3(kThreadsPerBlock),
        kThreadsPerBlock * sizeof(float), stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_128_exact_metadata_f32(
    const Q4_0Block* weights, const Q8ExactBlock* input, float* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        (q4_q8_gemv_zero_point_dot_128_kernel<128, Q8ExactBlock, float, true>),
        dim3(static_cast<unsigned int>(rows)), dim3(128), 128 * sizeof(float),
        stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_wave64_exact_metadata_f32(
    const Q4_0Block* weights, const Q8ExactBlock* input, float* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        (q4_q8_gemv_zero_point_dot_wave64_kernel<Q8ExactBlock, float, true>),
        dim3(static_cast<unsigned int>(rows)), dim3(128), 0, stream,
        weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_f32(
    const Q4_0Block* weights, const Q8_1Block* input, float* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((q4_q8_gemv_zero_point_dot_kernel<Q8_1Block, float, false>), dim3(rows), dim3(kThreadsPerBlock),
                       kThreadsPerBlock * sizeof(float), stream, weights, input, output,
                       rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_128_f32(
    const Q4_0Block* weights, const Q8_1Block* input, float* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((q4_q8_gemv_zero_point_dot_128_kernel<128, Q8_1Block, float, false>), dim3(rows), dim3(128),
                       128 * sizeof(float), stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_wave64_f32(
    const Q4_0Block* weights, const Q8_1Block* input, float* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((q4_q8_gemv_zero_point_dot_wave64_kernel<Q8_1Block, float, false>), dim3(rows), dim3(128), 0,
                       stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_exact_sum_f32(
    const Q4_0Block* weights, const Q8_1Block* input, float* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((q4_q8_gemv_contract_probe_kernel<ExactSumBlockDot, float>),
                       dim3(static_cast<unsigned int>(rows)), dim3(kThreadsPerBlock),
                       kThreadsPerBlock * sizeof(float), stream, weights, input, output,
                       rows, columns, ExactSumBlockDot{});
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_128_exact_sum_f32(
    const Q4_0Block* weights, const Q8_1Block* input, float* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((q4_q8_gemv_zero_point_dot_128_kernel<128, Q8_1Block, float, true>),
                       dim3(static_cast<unsigned int>(rows)), dim3(128), 128 * sizeof(float),
                       stream, weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_zero_point_dot_wave64_exact_sum_f32(
    const Q4_0Block* weights, const Q8_1Block* input, float* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((q4_q8_gemv_zero_point_dot_wave64_kernel<Q8_1Block, float, true>),
                       dim3(static_cast<unsigned int>(rows)), dim3(128), 0, stream, weights,
                       input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv_direct_signed_f32(
    const Q4_0Block* weights, const Q8_1Block* input, float* output,
    int rows, int columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr
        || rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((q4_q8_gemv_contract_probe_kernel<DirectSignedBlockDot, float>),
                       dim3(static_cast<unsigned int>(rows)), dim3(kThreadsPerBlock),
                       kThreadsPerBlock * sizeof(float), stream, weights, input, output,
                       rows, columns, DirectSignedBlockDot{});
    MIINFER_HIP_CHECK(hipGetLastError());
}

}  // namespace miinfer
