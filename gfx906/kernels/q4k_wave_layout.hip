#include "miinfer/q4k_wave_layout.hpp"
#include "miinfer/hip_check.hpp"
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "Q4K Wave layout requires gfx906"
#endif

namespace {
__global__ void wave_down(const Q4KWaveTile* weights,
                         const miinfer::Q8_1Block* input, float* output) {
    const unsigned tid = threadIdx.x % 128;
    const unsigned local_row = threadIdx.x / 128;
    const unsigned row = blockIdx.x * 2 + local_row;
    const unsigned lane = tid % 64;
    const unsigned in_block = lane % 16;
    const unsigned group = 2 * (in_block / 4);
    float sum = 0;
    for (unsigned tile = tid / 64; tile < 17; tile += 2) {
        const auto& w = weights[row * 17 + tile];
        const auto& m = w.metadata[lane / 16];
        const auto v0 = w.words[0][lane], v1 = w.words[1][lane];
        float sd = 0, sm = 0;
#pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const auto& q = input[(tile * 4 + lane / 16) * 8 + group + part];
            const auto* u = reinterpret_cast<const int*>(q.qs) + in_block % 4;
            const int a = (v0 >> (4 * part)) & 0x0f0f0f0f;
            const int b = (v1 >> (4 * part)) & 0x0f0f0f0f;
            const int dot = __builtin_amdgcn_sdot4(a, u[0],
                __builtin_amdgcn_sdot4(b, u[4], 0, false), false);
            const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
                __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);
            sd += __half2float(q.d) * float(dot * m.scales[group + part]);
            sm += __half2float(q.d) * float(ones * m.minimums[group + part]);
        }
        sum += __half2float(m.d) * sd - __half2float(m.dmin) * sm;
    }
    // Preserve the production two-wave row reduction and summation order.
    __shared__ float partials[2][128];
    partials[local_row][tid] = sum;
    __syncthreads();
    if (tid < 64) partials[local_row][tid] += partials[local_row][tid + 64];
    __syncthreads();
    for (unsigned stride = 32; stride; stride /= 2) {
        if (tid < stride) partials[local_row][tid] += partials[local_row][tid + stride];
        __syncthreads();
    }
    if (!tid) output[row] = partials[local_row][0];
}
}
void launch_q4k_wave_down(const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y) {
    if (!w || !x || !y) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    hipLaunchKernelGGL(wave_down, dim3(2560), dim3(256), 0, nullptr, w, x, y);
    MIINFER_HIP_CHECK(hipGetLastError());
}
