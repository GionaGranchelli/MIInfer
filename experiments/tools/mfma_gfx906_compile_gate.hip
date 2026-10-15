// Compile-only ISA probe for the gfx906 Stage1 MFMA feasibility gate.
// This deliberately does not launch: it only keeps an MFMA result observable
// so the pinned compiler must either reject the target or emit the instruction.
#include <hip/hip_runtime.h>

using mfma_acc_t = float __attribute__((ext_vector_type(4)));

extern "C" __global__ void mfma_gfx906_compile_gate(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out) {
    const unsigned lane = threadIdx.x & 63u;
    mfma_acc_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
    acc = __builtin_amdgcn_mfma_f32_16x16x4f32(
        a[lane], b[lane], acc, 0, 0, 0);
    const unsigned base = threadIdx.x * 4u;
    out[base + 0] = acc[0];
    out[base + 1] = acc[1];
    out[base + 2] = acc[2];
    out[base + 3] = acc[3];
}
