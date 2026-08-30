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
        int raw_integer_dot = 0;
        for (int group = 0; group < kQ4_0BlockSize / 2; group += 4) {
            // Q4_0 stores four low nibbles and four high nibbles in each
            // four-byte word. Keep both sets in the unsigned [0, 15] range;
            // the Q4 zero point is corrected once per complete block below.
            const auto q4_word = *reinterpret_cast<const std::uint32_t*>(
                weight.qs + group);
            const int q4_low = static_cast<int>(q4_word & kNibbleMask);
            const int q4_high = static_cast<int>((q4_word >> 4) & kNibbleMask);
            const int q8_low = *reinterpret_cast<const int*>(activation.qs + group);
            const int q8_high = *reinterpret_cast<const int*>(activation.qs + group + 16);
            raw_integer_dot = dot4_i8(q4_low, q8_low, raw_integer_dot);
            raw_integer_dot = dot4_i8(q4_high, q8_high, raw_integer_dot);
        }
        const float weight_scale = __half2float(weight.d);
        const float activation_scale = __half2float(activation.d);
        const float activation_scaled_sum = __half2float(activation.s);
        sum += weight_scale * (activation_scale * static_cast<float>(raw_integer_dot)
                               - 8.0F * activation_scaled_sum);
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

}  // namespace miinfer
