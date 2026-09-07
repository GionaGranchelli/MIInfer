#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/hip_check.hpp"
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "K-quant Wave layout requires gfx906"
#endif

#include <string>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

namespace {

// =========================================================================
// Decoder Policies
// =========================================================================

struct Q4KDecoder {
    using Tile = Q4KWaveTile;

    struct LaneState {
        const Tile::Metadata* m;
        std::uint32_t v0;
        std::uint32_t v1;
        float sd;
        float sm;
    };

    static __device__ __forceinline__ void load_tile(const Tile& w, unsigned lane, LaneState& state) {
        state.m = &w.metadata[lane / 16];
        state.v0 = w.words[0][lane];
        state.v1 = w.words[1][lane];
        state.sd = 0.0f;
        state.sm = 0.0f;
    }

    static __device__ __forceinline__ void process_part(
        LaneState& state, unsigned part, unsigned group,
        const miinfer::Q8_1Block& q, const int* u) {
        const int a = (state.v0 >> (4 * part)) & 0x0f0f0f0f;
        const int b = (state.v1 >> (4 * part)) & 0x0f0f0f0f;
        const int dot = __builtin_amdgcn_sdot4(a, u[0],
            __builtin_amdgcn_sdot4(b, u[4], 0, false), false);
        const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
            __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);
        state.sd += __half2float(q.d) * float(dot * state.m->scales[group + part]);
        state.sm += __half2float(q.d) * float(ones * state.m->minimums[group + part]);
    }

    static __device__ __forceinline__ float finalize_tile(const LaneState& state) {
        return __half2float(state.m->d) * state.sd - __half2float(state.m->dmin) * state.sm;
    }
};

struct Q5KDecoder {
    using Tile = Q5KWaveTile;

    struct LaneState {
        const Tile::Metadata* m;
        std::uint32_t v0;
        std::uint32_t v1;
        std::uint32_t h;
        float sd;
        float sm;
    };

    static __device__ __forceinline__ void load_tile(const Tile& w, unsigned lane, LaneState& state) {
        state.m = &w.metadata[lane / 16];
        state.v0 = w.words[0][lane];
        state.v1 = w.words[1][lane];
        state.h = w.qh[lane];
        state.sd = 0.0f;
        state.sm = 0.0f;
    }

    static __device__ __forceinline__ void process_part(
        LaneState& state, unsigned part, unsigned group,
        const miinfer::Q8_1Block& q, const int* u) {
        const int low_a = (state.v0 >> (4 * part)) & 0x0f0f0f0f;
        const int low_b = (state.v1 >> (4 * part)) & 0x0f0f0f0f;
        const int high_a = (part == 0) ? ((state.h << 4) & 0x10101010) : ((state.h << 3) & 0x10101010);
        const int high_b = (part == 0) ? ((state.h << 2) & 0x10101010) : ((state.h << 1) & 0x10101010);
        const int a = low_a | high_a;
        const int b = low_b | high_b;
        const int dot = __builtin_amdgcn_sdot4(a, u[0],
            __builtin_amdgcn_sdot4(b, u[4], 0, false), false);
        const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
            __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);
        state.sd += __half2float(q.d) * float(dot * state.m->scales[group + part]);
        state.sm += __half2float(q.d) * float(ones * state.m->minimums[group + part]);
    }

    static __device__ __forceinline__ float finalize_tile(const LaneState& state) {
        return __half2float(state.m->d) * state.sd - __half2float(state.m->dmin) * state.sm;
    }
};

struct Q6KDecoderLegacy {
    using Tile = Q6KWaveTile;

    struct LaneState {
        const Tile::Metadata* m;
        std::uint32_t v0;
        std::uint32_t v1;
        std::uint32_t h;
        float sd;
    };

    static __device__ __forceinline__ void load_tile(const Tile& w, unsigned lane, LaneState& state) {
        state.m = &w.metadata[lane / 16];
        state.v0 = w.words[0][lane];
        state.v1 = w.words[1][lane];
        state.h = w.qh[lane];
        state.sd = 0.0f;
    }

    static __device__ __forceinline__ void process_part(
        LaneState& state, unsigned part, unsigned group,
        const miinfer::Q8_1Block& q, const int* u) {
        std::uint32_t a = 0;
        std::uint32_t b = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t byte_v0 = (state.v0 >> (8 * j)) & 0xff;
            const std::uint32_t byte_v1 = (state.v1 >> (8 * j)) & 0xff;
            const std::uint32_t low_a = (part == 0) ? (byte_v0 & 0x0f) : (byte_v0 >> 4);
            const std::uint32_t low_b = (part == 0) ? (byte_v1 & 0x0f) : (byte_v1 >> 4);

            const std::uint32_t byte_h = (state.h >> (8 * j)) & 0xff;
            const int shift_a = (part == 0) ? 0 : 4;
            const int shift_b = (part == 0) ? 2 : 6;
            const std::uint32_t high_a = (byte_h >> shift_a) & 0x03;
            const std::uint32_t high_b = (byte_h >> shift_b) & 0x03;

            const std::int32_t q_a = static_cast<std::int32_t>(low_a | (high_a << 4)) - 32;
            const std::int32_t q_b = static_cast<std::int32_t>(low_b | (high_b << 4)) - 32;

            a |= (static_cast<std::uint32_t>(q_a & 0xff) << (8 * j));
            b |= (static_cast<std::uint32_t>(q_b & 0xff) << (8 * j));
        }

        const int dot_a = __builtin_amdgcn_sdot4(static_cast<int>(a), u[0], 0, false);
        const int dot_b = __builtin_amdgcn_sdot4(static_cast<int>(b), u[4], 0, false);
        const int sc_a = state.m->scales[2 * (group + part)];
        const int sc_b = state.m->scales[2 * (group + part) + 1];
        state.sd += __half2float(q.d) * float(dot_a * sc_a + dot_b * sc_b);
    }

    static __device__ __forceinline__ float finalize_tile(const LaneState& state) {
        return __half2float(state.m->d) * state.sd;
    }
};

struct Q6KDecoderSimd {
    using Tile = Q6KWaveTile;

    struct LaneState {
        const Tile::Metadata* m;
        std::uint32_t v0;
        std::uint32_t v1;
        std::uint32_t h;
        float sd;
    };

    static __device__ __forceinline__ void load_tile(const Tile& w, unsigned lane, LaneState& state) {
        state.m = &w.metadata[lane / 16];
        state.v0 = w.words[0][lane];
        state.v1 = w.words[1][lane];
        state.h = w.qh[lane];
        state.sd = 0.0f;
    }

    static __device__ __forceinline__ void process_part(
        LaneState& state, unsigned part, unsigned group,
        const miinfer::Q8_1Block& q, const int* u) {
        const std::uint32_t low_a = (part == 0) ? (state.v0 & 0x0f0f0f0f) : ((state.v0 >> 4) & 0x0f0f0f0f);
        const std::uint32_t low_b = (part == 0) ? (state.v1 & 0x0f0f0f0f) : ((state.v1 >> 4) & 0x0f0f0f0f);

        const std::uint32_t high_a = (part == 0) ? ((state.h & 0x03030303) << 4) : (state.h & 0x30303030);
        const std::uint32_t high_b = (part == 0) ? ((state.h & 0x0c0c0c0c) << 2) : ((state.h >> 2) & 0x30303030);

        const std::uint32_t u_a = low_a | high_a;
        const std::uint32_t u_b = low_b | high_b;

        const std::uint32_t not_b5_a = (u_a & 0x20202020) ^ 0x20202020;
        const std::uint32_t a = (u_a & 0x1f1f1f1f) | (not_b5_a | (not_b5_a << 1) | (not_b5_a << 2));

        const std::uint32_t not_b5_b = (u_b & 0x20202020) ^ 0x20202020;
        const std::uint32_t b = (u_b & 0x1f1f1f1f) | (not_b5_b | (not_b5_b << 1) | (not_b5_b << 2));

        const int dot_a = __builtin_amdgcn_sdot4(static_cast<int>(a), u[0], 0, false);
        const int dot_b = __builtin_amdgcn_sdot4(static_cast<int>(b), u[4], 0, false);
        const int sc_a = state.m->scales[2 * (group + part)];
        const int sc_b = state.m->scales[2 * (group + part) + 1];
        state.sd += __half2float(q.d) * float(dot_a * sc_a + dot_b * sc_b);
    }

    static __device__ __forceinline__ float finalize_tile(const LaneState& state) {
        return __half2float(state.m->d) * state.sd;
    }
};

using Q6KDecoder = Q6KDecoderSimd;

// =========================================================================
// Common Wave64 Tile Executor Kernel
// =========================================================================

template <typename Decoder, unsigned NumTiles>
__global__ void kquant_wave_gemv_kernel(
    const typename Decoder::Tile* weights,
    const miinfer::Q8_1Block* input,
    float* output,
    unsigned rows) {
    const unsigned wave_id = threadIdx.x / 64;
    const unsigned row = blockIdx.x * 4 + wave_id;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x % 64;
    const unsigned in_block = lane % 16;
    const unsigned group = 2 * (in_block / 4);

    float sum = 0.0f;
    for (unsigned tile = 0; tile < NumTiles; ++tile) {
        const auto& w = weights[row * NumTiles + tile];
        typename Decoder::LaneState state;
        Decoder::load_tile(w, lane, state);

        #pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const auto& q = input[(tile * 4 + lane / 16) * 8 + group + part];
            const auto* u = reinterpret_cast<const int*>(q.qs) + (in_block % 4);
            Decoder::process_part(state, part, group, q, u);
        }
        sum += Decoder::finalize_tile(state);
    }

    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 64);
    }

    if (lane == 0) {
        output[row] = sum;
    }
}

template <typename Decoder>
void launch_kquant_gemv_dispatch(
    const typename Decoder::Tile* w,
    const miinfer::Q8_1Block* x,
    float* y,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr) {
    if (!w || !x || !y) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    if (columns % 1024 != 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    const unsigned num_tiles = columns / 1024;
    const dim3 grid((rows + 3) / 4);
    const dim3 block(256);
    switch (num_tiles) {
    case 1: {
        auto kernel = kquant_wave_gemv_kernel<Decoder, 1>;
        hipLaunchKernelGGL(kernel, grid, block, 0, stream, w, x, y, rows);
        break;
    }
    case 5: {
        auto kernel = kquant_wave_gemv_kernel<Decoder, 5>;
        hipLaunchKernelGGL(kernel, grid, block, 0, stream, w, x, y, rows);
        break;
    }
    case 6: {
        auto kernel = kquant_wave_gemv_kernel<Decoder, 6>;
        hipLaunchKernelGGL(kernel, grid, block, 0, stream, w, x, y, rows);
        break;
    }
    case 12: {
        auto kernel = kquant_wave_gemv_kernel<Decoder, 12>;
        hipLaunchKernelGGL(kernel, grid, block, 0, stream, w, x, y, rows);
        break;
    }
    case 17: {
        auto kernel = kquant_wave_gemv_kernel<Decoder, 17>;
        hipLaunchKernelGGL(kernel, grid, block, 0, stream, w, x, y, rows);
        break;
    }
    default:
        throw std::runtime_error("unsupported NumTiles in launch_kquant_gemv_dispatch: " + std::to_string(num_tiles));
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

template <unsigned NumTiles, bool FastShuffle = true>
__global__ void q4k_wave_fused_gate_up_swiglu_kernel(
    const Q4KWaveTile* __restrict__ weights_gate,
    const Q4KWaveTile* __restrict__ weights_up,
    const miinfer::Q8_1Block* __restrict__ input,
    float* __restrict__ output) {
    const unsigned tid = threadIdx.x % 128;
    const unsigned local_row = threadIdx.x / 128;
    const unsigned row = blockIdx.x * 2 + local_row;
    const unsigned lane = tid % 64;
    const unsigned in_block = lane % 16;
    const unsigned group = 2 * (in_block / 4);

    float sum_gate = 0.0f;
    float sum_up = 0.0f;

    for (unsigned tile = tid / 64; tile < NumTiles; tile += 2) {
        const auto& w_g = weights_gate[row * NumTiles + tile];
        const auto& w_u = weights_up[row * NumTiles + tile];
        Q4KDecoder::LaneState state_g;
        Q4KDecoder::LaneState state_u;
        Q4KDecoder::load_tile(w_g, lane, state_g);
        Q4KDecoder::load_tile(w_u, lane, state_u);

        #pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const auto& q = input[(tile * 4 + lane / 16) * 8 + group + part];
            const auto* u = reinterpret_cast<const int*>(q.qs) + (in_block % 4);

            const int a_g = (state_g.v0 >> (4 * part)) & 0x0f0f0f0f;
            const int b_g = (state_g.v1 >> (4 * part)) & 0x0f0f0f0f;
            const int dot_g = __builtin_amdgcn_sdot4(a_g, u[0],
                __builtin_amdgcn_sdot4(b_g, u[4], 0, false), false);

            const int a_u = (state_u.v0 >> (4 * part)) & 0x0f0f0f0f;
            const int b_u = (state_u.v1 >> (4 * part)) & 0x0f0f0f0f;
            const int dot_u = __builtin_amdgcn_sdot4(a_u, u[0],
                __builtin_amdgcn_sdot4(b_u, u[4], 0, false), false);

            const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
                __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);

            const float q_d = __half2float(q.d);
            const float q_ones = q_d * float(ones);

            state_g.sd += q_d * float(dot_g * state_g.m->scales[group + part]);
            state_g.sm += q_ones * float(state_g.m->minimums[group + part]);

            state_u.sd += q_d * float(dot_u * state_u.m->scales[group + part]);
            state_u.sm += q_ones * float(state_u.m->minimums[group + part]);
        }
        sum_gate += Q4KDecoder::finalize_tile(state_g);
        sum_up += Q4KDecoder::finalize_tile(state_u);
    }

    if constexpr (FastShuffle) {
        #pragma unroll
        for (int offset = 32; offset > 0; offset /= 2) {
            sum_gate += __shfl_down(sum_gate, offset, 64);
            sum_up += __shfl_down(sum_up, offset, 64);
        }

        __shared__ float odd_gate[2];
        __shared__ float odd_up[2];

        if (tid == 64) {
            odd_gate[local_row] = sum_gate;
            odd_up[local_row] = sum_up;
        }
        __syncthreads();

        if (tid == 0) {
            const float g = sum_gate + odd_gate[local_row];
            const float u = sum_up + odd_up[local_row];
            const float silu_g = g / (1.0f + expf(-g));
            output[row] = silu_g * u;
        }
    } else {
        __shared__ float partials_gate[2][128];
        __shared__ float partials_up[2][128];
        partials_gate[local_row][tid] = sum_gate;
        partials_up[local_row][tid] = sum_up;
        __syncthreads();

        if (tid < 64) {
            partials_gate[local_row][tid] += partials_gate[local_row][tid + 64];
            partials_up[local_row][tid] += partials_up[local_row][tid + 64];
        }
        __syncthreads();

        for (unsigned stride = 32; stride; stride /= 2) {
            if (tid < stride) {
                partials_gate[local_row][tid] += partials_gate[local_row][tid + stride];
                partials_up[local_row][tid] += partials_up[local_row][tid + stride];
            }
            __syncthreads();
        }

        if (!tid) {
            const float g = partials_gate[local_row][0];
            const float u = partials_up[local_row][0];
            const float silu_g = g / (1.0f + expf(-g));
            output[row] = silu_g * u;
        }
    }
}

static bool swiglu_shuffle_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MIINFER_SWIGLU_SHUFFLE");
        return env == nullptr || (std::strcmp(env, "0") != 0 && std::strcmp(env, "OFF") != 0 && std::strcmp(env, "off") != 0);
    }();
    return enabled;
}

} // namespace

void launch_q4k_wave_gemv(const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          std::uint32_t rows, std::uint32_t columns, hipStream_t stream) {
    launch_kquant_gemv_dispatch<Q4KDecoder>(w, x, y, rows, columns, stream);
}

void launch_q4k_wave_fused_gate_up_swiglu(
    const Q4KWaveTile* w_gate,
    const Q4KWaveTile* w_up,
    const miinfer::Q8_1Block* x,
    float* y_activation,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (!w_gate || !w_up || !x || !y_activation) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    if (rows % 2 != 0 || columns % 1024 != 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    const unsigned num_tiles = columns / 1024;
    const dim3 grid(rows / 2);
    const dim3 block(256);
    switch (num_tiles) {
    case 5: {
        if (swiglu_shuffle_enabled()) {
            auto kernel = q4k_wave_fused_gate_up_swiglu_kernel<5, true>;
            hipLaunchKernelGGL(kernel, grid, block, 0, stream, w_gate, w_up, x, y_activation);
        } else {
            auto kernel = q4k_wave_fused_gate_up_swiglu_kernel<5, false>;
            hipLaunchKernelGGL(kernel, grid, block, 0, stream, w_gate, w_up, x, y_activation);
        }
        break;
    }
    default:
        throw std::runtime_error("unsupported NumTiles in launch_q4k_wave_fused_gate_up_swiglu: " + std::to_string(num_tiles));
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4k_wave_down(const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y, hipStream_t stream) {
    launch_q4k_wave_gemv(w, x, y, 5120, 17408, stream);
}

void launch_q5k_wave_gemv(const Q5KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          std::uint32_t rows, std::uint32_t columns, hipStream_t stream) {
    launch_kquant_gemv_dispatch<Q5KDecoder>(w, x, y, rows, columns, stream);
}

static bool q6k_simd_unpack_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MIINFER_Q6K_SIMD_UNPACK");
        return env == nullptr || (std::strcmp(env, "0") != 0 && std::strcmp(env, "OFF") != 0 && std::strcmp(env, "off") != 0);
    }();
    return enabled;
}

void launch_q6k_wave_gemv(const Q6KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          std::uint32_t rows, std::uint32_t columns, hipStream_t stream) {
    if (q6k_simd_unpack_enabled()) {
        launch_kquant_gemv_dispatch<Q6KDecoderSimd>(w, x, y, rows, columns, stream);
    } else {
        launch_kquant_gemv_dispatch<Q6KDecoderLegacy>(w, x, y, rows, columns, stream);
    }
}
