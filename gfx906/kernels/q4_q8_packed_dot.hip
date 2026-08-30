#include "miinfer/q4_q8_packed_dot.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "The Q4/Q8 packed-dot candidate is intentionally gfx906-only"
#endif

namespace miinfer {

namespace {

constexpr unsigned int kThreadsPerBlock = 256;

__device__ __forceinline__ int pack_i8x4(
    int first,
    int second,
    int third,
    int fourth) {
    return static_cast<int>(
        static_cast<std::uint32_t>(static_cast<std::uint8_t>(first))
        | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(second)) << 8)
        | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(third)) << 16)
        | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(fourth)) << 24));
}

__device__ __forceinline__ int q4_signed(const Q4_0Block& block, int index) {
    const int packed_index = index < kQ4_0BlockSize / 2
                                 ? index
                                 : index - kQ4_0BlockSize / 2;
    const std::uint8_t packed = block.qs[packed_index];
    const int nibble = index < kQ4_0BlockSize / 2
                           ? (packed & 0x0F)
                           : ((packed >> 4) & 0x0F);
    return nibble - 8;
}

__device__ __forceinline__ int dot4_i8(
    int lhs,
    int rhs,
    int accumulator) {
    // ROCm Clang's signed AMDGCN intrinsic lowers to v_dot4_i32_i8 for gfx906.
    return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
}

__global__ void q4_q8_gemv_packed_dot_kernel(
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
        int integer_dot = 0;
        for (int group = 0; group < kQ4_0BlockSize; group += 4) {
            const int q4_packed = pack_i8x4(
                q4_signed(weight, group + 0),
                q4_signed(weight, group + 1),
                q4_signed(weight, group + 2),
                q4_signed(weight, group + 3));
            const int q8_packed = pack_i8x4(
                activation.qs[group + 0],
                activation.qs[group + 1],
                activation.qs[group + 2],
                activation.qs[group + 3]);
            integer_dot = dot4_i8(q4_packed, q8_packed, integer_dot);
        }
        sum += static_cast<float>(integer_dot)
               * __half2float(weight.d) * __half2float(activation.d);
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

__global__ void q4_q8_dot4_probe_kernel(int lhs, int rhs, int* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = dot4_i8(lhs, rhs, 0);
    }
}

}  // namespace

void launch_q4_q8_gemv_packed_dot(
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
        q4_q8_gemv_packed_dot_kernel,
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

void launch_q4_q8_dot4_probe(
    int lhs,
    int rhs,
    int* output,
    hipStream_t stream) {
    hipLaunchKernelGGL(q4_q8_dot4_probe_kernel, dim3(1), dim3(1), 0, stream,
                       lhs, rhs, output);
    MIINFER_HIP_CHECK(hipGetLastError());
}

}  // namespace miinfer
