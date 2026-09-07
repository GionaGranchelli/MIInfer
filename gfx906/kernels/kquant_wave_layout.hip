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

struct Q4KDecoderLegacy {
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

struct Q4KDecoderFast {
    using Tile = Q4KWaveTile;

    struct LaneState {
        std::uint32_t v0;
        std::uint32_t v1;
        std::uint16_t scales2;
        std::uint16_t mins2;
        float d;
        float dmin;
        float sd;
        float sm;
    };

    static __device__ __forceinline__ void load_tile(const Tile& w, unsigned lane, LaneState& state) {
        const auto& meta = w.metadata[lane / 16];
        const unsigned in_block = lane % 16;
        const unsigned group = 2 * (in_block / 4); // 0, 2, 4, 6
        state.scales2 = *reinterpret_cast<const std::uint16_t*>(&meta.scales[group]);
        state.mins2 = *reinterpret_cast<const std::uint16_t*>(&meta.minimums[group]);
        state.d = __half2float(meta.d);
        state.dmin = __half2float(meta.dmin);
        state.v0 = w.words[0][lane];
        state.v1 = w.words[1][lane];
        state.sd = 0.0f;
        state.sm = 0.0f;
    }

    static __device__ __forceinline__ void process_part(
        LaneState& state, unsigned part, [[maybe_unused]] unsigned group,
        const miinfer::Q8_1Block& q, const int* u) {
        const int a = (part == 0) ? (state.v0 & 0x0f0f0f0f) : ((state.v0 >> 4) & 0x0f0f0f0f);
        const int b = (part == 0) ? (state.v1 & 0x0f0f0f0f) : ((state.v1 >> 4) & 0x0f0f0f0f);
        const int dot = __builtin_amdgcn_sdot4(a, u[0],
            __builtin_amdgcn_sdot4(b, u[4], 0, false), false);
        const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
            __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);
        const float q_d = __half2float(q.d);
        const int scale = (part == 0) ? (state.scales2 & 0xff) : (state.scales2 >> 8);
        const int minimum = (part == 0) ? (state.mins2 & 0xff) : (state.mins2 >> 8);
        state.sd += q_d * float(dot * scale);
        state.sm += (q_d * float(ones)) * float(minimum);
    }

    static __device__ __forceinline__ float finalize_tile(const LaneState& state) {
        return state.d * state.sd - state.dmin * state.sm;
    }
};

using Q4KDecoder = Q4KDecoderFast;

struct Q5KDecoderLegacy {
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

struct Q5KDecoderFast {
    using Tile = Q5KWaveTile;

    struct LaneState {
        std::uint32_t v0;
        std::uint32_t v1;
        std::uint32_t h;
        std::uint16_t scales2;
        std::uint16_t mins2;
        float d;
        float dmin;
        float sd;
        float sm;
    };

    static __device__ __forceinline__ void load_tile(const Tile& w, unsigned lane, LaneState& state) {
        const auto& meta = w.metadata[lane / 16];
        const unsigned in_block = lane % 16;
        const unsigned group = 2 * (in_block / 4); // 0, 2, 4, 6
        state.scales2 = *reinterpret_cast<const std::uint16_t*>(&meta.scales[group]);
        state.mins2 = *reinterpret_cast<const std::uint16_t*>(&meta.minimums[group]);
        state.d = __half2float(meta.d);
        state.dmin = __half2float(meta.dmin);
        state.v0 = w.words[0][lane];
        state.v1 = w.words[1][lane];
        state.h = w.qh[lane];
        state.sd = 0.0f;
        state.sm = 0.0f;
    }

    static __device__ __forceinline__ void process_part(
        LaneState& state, unsigned part, [[maybe_unused]] unsigned group,
        const miinfer::Q8_1Block& q, const int* u) {
        const int low_a = (part == 0) ? (state.v0 & 0x0f0f0f0f) : ((state.v0 >> 4) & 0x0f0f0f0f);
        const int low_b = (part == 0) ? (state.v1 & 0x0f0f0f0f) : ((state.v1 >> 4) & 0x0f0f0f0f);
        const int high_a = (part == 0) ? ((state.h << 4) & 0x10101010) : ((state.h << 3) & 0x10101010);
        const int high_b = (part == 0) ? ((state.h << 2) & 0x10101010) : ((state.h << 1) & 0x10101010);
        const int a = low_a | high_a;
        const int b = low_b | high_b;
        const int dot = __builtin_amdgcn_sdot4(a, u[0],
            __builtin_amdgcn_sdot4(b, u[4], 0, false), false);
        const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
            __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);
        const float q_d = __half2float(q.d);
        const int scale = (part == 0) ? (state.scales2 & 0xff) : (state.scales2 >> 8);
        const int min_val = (part == 0) ? (state.mins2 & 0xff) : (state.mins2 >> 8);
        state.sd += q_d * float(dot * scale);
        state.sm += (q_d * float(ones)) * float(min_val);
    }

    static __device__ __forceinline__ float finalize_tile(const LaneState& state) {
        return state.d * state.sd - state.dmin * state.sm;
    }
};

using Q5KDecoder = Q5KDecoderFast;

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
        std::uint32_t v0;
        std::uint32_t v1;
        std::uint32_t h;
        std::uint32_t scales_word;
        float d;
        float sd;
    };

    static __device__ __forceinline__ void load_tile(const Tile& w, unsigned lane, LaneState& state) {
        const auto& meta = w.metadata[lane / 16];
        const unsigned in_block = lane % 16;
        const unsigned group = 2 * (in_block / 4); // 0, 2, 4, 6
        state.scales_word = *reinterpret_cast<const std::uint32_t*>(&meta.scales[2 * group]);
        state.d = __half2float(meta.d);
        state.v0 = w.words[0][lane];
        state.v1 = w.words[1][lane];
        state.h = w.qh[lane];
        state.sd = 0.0f;
    }

    static __device__ __forceinline__ void process_part(
        LaneState& state, unsigned part, [[maybe_unused]] unsigned group,
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
        
        const int shift = part * 16;
        const int sc_a = static_cast<std::int8_t>((state.scales_word >> shift) & 0xff);
        const int sc_b = static_cast<std::int8_t>((state.scales_word >> (shift + 8)) & 0xff);

        state.sd += __half2float(q.d) * float(dot_a * sc_a + dot_b * sc_b);
    }

    static __device__ __forceinline__ float finalize_tile(const LaneState& state) {
        return state.d * state.sd;
    }
};

using Q6KDecoder = Q6KDecoderSimd;

// =========================================================================
// Common Wave64 Tile Executor Kernel
// =========================================================================

template <unsigned NumTiles>
__global__ void q6k_wave_gemv_2wave_512_kernel(
    const Q6KWaveTile* __restrict__ weights,
    const miinfer::Q8_1Block* __restrict__ input,
    float* __restrict__ output,
    unsigned rows) {
    const unsigned tid = threadIdx.x % 128;
    const unsigned local_row = threadIdx.x / 128;
    const unsigned row = blockIdx.x * 4 + local_row;
    if (row >= rows) return;

    const unsigned lane = tid % 64;
    const unsigned in_block = lane % 16;
    const unsigned group = 2 * (in_block / 4);

    float sum = 0.0f;
    for (unsigned tile = tid / 64; tile < NumTiles; tile += 2) {
        const auto& w = weights[row * NumTiles + tile];
        const auto& meta = w.metadata[lane / 16];
        const uint32_t scales_word = *reinterpret_cast<const uint32_t*>(&meta.scales[2 * group]);
        const float d = __half2float(meta.d);
        const uint32_t v0 = w.words[0][lane];
        const uint32_t v1 = w.words[1][lane];
        const uint32_t h = w.qh[lane];

        float sd = 0.0f;
        #pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const auto& q = input[(tile * 4 + lane / 16) * 8 + group + part];
            const auto* u = reinterpret_cast<const int*>(q.qs) + (in_block % 4);

            const uint32_t low_a = (part == 0) ? (v0 & 0x0f0f0f0f) : ((v0 >> 4) & 0x0f0f0f0f);
            const uint32_t low_b = (part == 0) ? (v1 & 0x0f0f0f0f) : ((v1 >> 4) & 0x0f0f0f0f);
            const uint32_t high_a = (part == 0) ? ((h & 0x03030303) << 4) : (h & 0x30303030);
            const uint32_t high_b = (part == 0) ? ((h & 0x0c0c0c0c) << 2) : ((h >> 2) & 0x30303030);

            const uint32_t u_a = low_a | high_a;
            const uint32_t u_b = low_b | high_b;

            const uint32_t not_b5_a = (u_a & 0x20202020) ^ 0x20202020;
            const uint32_t a = (u_a & 0x1f1f1f1f) | (not_b5_a | (not_b5_a << 1) | (not_b5_a << 2));

            const uint32_t not_b5_b = (u_b & 0x20202020) ^ 0x20202020;
            const uint32_t b = (u_b & 0x1f1f1f1f) | (not_b5_b | (not_b5_b << 1) | (not_b5_b << 2));

            const int dot_a = __builtin_amdgcn_sdot4(static_cast<int>(a), u[0], 0, false);
            const int dot_b = __builtin_amdgcn_sdot4(static_cast<int>(b), u[4], 0, false);

            const int shift = part * 16;
            const int sc_a = static_cast<int8_t>((scales_word >> shift) & 0xff);
            const int sc_b = static_cast<int8_t>((scales_word >> (shift + 8)) & 0xff);

            sd += __half2float(q.d) * float(dot_a * sc_a + dot_b * sc_b);
        }
        sum += d * sd;
    }

    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 64);
    }

    __shared__ float odd_sum[4];
    if (tid == 64) {
        odd_sum[local_row] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        output[row] = sum + odd_sum[local_row];
    }
}

template <typename Decoder, unsigned NumTiles, unsigned WavesPerBlock = 4>
__global__ void kquant_wave_gemv_kernel(
    const typename Decoder::Tile* weights,
    const miinfer::Q8_1Block* input,
    float* output,
    unsigned rows) {
    const unsigned wave_id = threadIdx.x / 64;
    const unsigned row = blockIdx.x * WavesPerBlock + wave_id;
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

static bool q6k_2wave_coop_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MIINFER_Q6K_2WAVE_COOP");
        return env == nullptr || (std::strcmp(env, "0") != 0 && std::strcmp(env, "OFF") != 0 && std::strcmp(env, "off") != 0);
    }();
    return enabled;
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

    if constexpr (std::is_same_v<Decoder, Q6KDecoderSimd>) {
        if (num_tiles == 17 && q6k_2wave_coop_enabled()) {
            const dim3 grid((rows + 3) / 4);
            const dim3 block(512);
            auto kernel = q6k_wave_gemv_2wave_512_kernel<17>;
            hipLaunchKernelGGL(kernel, grid, block, 0, stream, w, x, y, rows);
            MIINFER_HIP_CHECK(hipGetLastError());
            return;
        }
    }

    constexpr bool is_legacy = std::is_same_v<Decoder, Q4KDecoderLegacy>
                            || std::is_same_v<Decoder, Q5KDecoderLegacy>
                            || std::is_same_v<Decoder, Q6KDecoderLegacy>;

    if constexpr (is_legacy) {
        const dim3 grid((rows + 3) / 4);
        const dim3 block(256);
        switch (num_tiles) {
        case 1:  hipLaunchKernelGGL((kquant_wave_gemv_kernel<Decoder, 1, 4>), grid, block, 0, stream, w, x, y, rows); break;
        case 5:  hipLaunchKernelGGL((kquant_wave_gemv_kernel<Decoder, 5, 4>), grid, block, 0, stream, w, x, y, rows); break;
        case 6:  hipLaunchKernelGGL((kquant_wave_gemv_kernel<Decoder, 6, 4>), grid, block, 0, stream, w, x, y, rows); break;
        case 12: hipLaunchKernelGGL((kquant_wave_gemv_kernel<Decoder, 12, 4>), grid, block, 0, stream, w, x, y, rows); break;
        case 17: hipLaunchKernelGGL((kquant_wave_gemv_kernel<Decoder, 17, 4>), grid, block, 0, stream, w, x, y, rows); break;
        default:
            throw std::runtime_error("unsupported NumTiles in launch_kquant_gemv_dispatch: " + std::to_string(num_tiles));
        }
    } else {
        const dim3 grid((rows + 7) / 8);
        const dim3 block(512);
        switch (num_tiles) {
        case 1:  hipLaunchKernelGGL((kquant_wave_gemv_kernel<Decoder, 1, 8>), grid, block, 0, stream, w, x, y, rows); break;
        case 5:  hipLaunchKernelGGL((kquant_wave_gemv_kernel<Decoder, 5, 8>), grid, block, 0, stream, w, x, y, rows); break;
        case 6:  hipLaunchKernelGGL((kquant_wave_gemv_kernel<Decoder, 6, 8>), grid, block, 0, stream, w, x, y, rows); break;
        case 12: hipLaunchKernelGGL((kquant_wave_gemv_kernel<Decoder, 12, 8>), grid, block, 0, stream, w, x, y, rows); break;
        case 17: hipLaunchKernelGGL((kquant_wave_gemv_kernel<Decoder, 17, 8>), grid, block, 0, stream, w, x, y, rows); break;
        default:
            throw std::runtime_error("unsupported NumTiles in launch_kquant_gemv_dispatch: " + std::to_string(num_tiles));
        }
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

            const int a_g = (part == 0) ? (state_g.v0 & 0x0f0f0f0f) : ((state_g.v0 >> 4) & 0x0f0f0f0f);
            const int b_g = (part == 0) ? (state_g.v1 & 0x0f0f0f0f) : ((state_g.v1 >> 4) & 0x0f0f0f0f);
            const int dot_g = __builtin_amdgcn_sdot4(a_g, u[0],
                __builtin_amdgcn_sdot4(b_g, u[4], 0, false), false);

            const int a_u = (part == 0) ? (state_u.v0 & 0x0f0f0f0f) : ((state_u.v0 >> 4) & 0x0f0f0f0f);
            const int b_u = (part == 0) ? (state_u.v1 & 0x0f0f0f0f) : ((state_u.v1 >> 4) & 0x0f0f0f0f);
            const int dot_u = __builtin_amdgcn_sdot4(a_u, u[0],
                __builtin_amdgcn_sdot4(b_u, u[4], 0, false), false);

            const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
                __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);

            const float q_d = __half2float(q.d);
            const float q_ones = q_d * float(ones);

            const int scale_g = (part == 0) ? (state_g.scales2 & 0xff) : (state_g.scales2 >> 8);
            const int minimum_g = (part == 0) ? (state_g.mins2 & 0xff) : (state_g.mins2 >> 8);
            state_g.sd += q_d * float(dot_g * scale_g);
            state_g.sm += q_ones * float(minimum_g);

            const int scale_u = (part == 0) ? (state_u.scales2 & 0xff) : (state_u.scales2 >> 8);
            const int minimum_u = (part == 0) ? (state_u.mins2 & 0xff) : (state_u.mins2 >> 8);
            state_u.sd += q_d * float(dot_u * scale_u);
            state_u.sm += q_ones * float(minimum_u);
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

template <unsigned NumTiles>
__global__ void q4k_wave_fused_gate_up_swiglu_1wave_kernel(
    const Q4KWaveTile* __restrict__ weights_gate,
    const Q4KWaveTile* __restrict__ weights_up,
    const miinfer::Q8_1Block* __restrict__ input,
    float* __restrict__ output,
    unsigned rows) {
    const unsigned wave_id = threadIdx.x / 64;
    const unsigned row = blockIdx.x * 4 + wave_id;
    if (row >= rows) return;

    const unsigned lane = threadIdx.x % 64;
    const unsigned in_block = lane % 16;
    const unsigned group = 2 * (in_block / 4);

    float sum_gate = 0.0f;
    float sum_up = 0.0f;

    for (unsigned tile = 0; tile < NumTiles; ++tile) {
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

            const int a_g = (part == 0) ? (state_g.v0 & 0x0f0f0f0f) : ((state_g.v0 >> 4) & 0x0f0f0f0f);
            const int b_g = (part == 0) ? (state_g.v1 & 0x0f0f0f0f) : ((state_g.v1 >> 4) & 0x0f0f0f0f);
            const int dot_g = __builtin_amdgcn_sdot4(a_g, u[0],
                __builtin_amdgcn_sdot4(b_g, u[4], 0, false), false);

            const int a_u = (part == 0) ? (state_u.v0 & 0x0f0f0f0f) : ((state_u.v0 >> 4) & 0x0f0f0f0f);
            const int b_u = (part == 0) ? (state_u.v1 & 0x0f0f0f0f) : ((state_u.v1 >> 4) & 0x0f0f0f0f);
            const int dot_u = __builtin_amdgcn_sdot4(a_u, u[0],
                __builtin_amdgcn_sdot4(b_u, u[4], 0, false), false);

            const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
                __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);

            const float q_d = __half2float(q.d);
            const float q_ones = q_d * float(ones);

            const int scale_g = (part == 0) ? (state_g.scales2 & 0xff) : (state_g.scales2 >> 8);
            const int minimum_g = (part == 0) ? (state_g.mins2 & 0xff) : (state_g.mins2 >> 8);
            state_g.sd += q_d * float(dot_g * scale_g);
            state_g.sm += q_ones * float(minimum_g);

            const int scale_u = (part == 0) ? (state_u.scales2 & 0xff) : (state_u.scales2 >> 8);
            const int minimum_u = (part == 0) ? (state_u.mins2 & 0xff) : (state_u.mins2 >> 8);
            state_u.sd += q_d * float(dot_u * scale_u);
            state_u.sm += q_ones * float(minimum_u);
        }
        sum_gate += Q4KDecoder::finalize_tile(state_g);
        sum_up += Q4KDecoder::finalize_tile(state_u);
    }

    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        sum_gate += __shfl_down(sum_gate, offset, 64);
        sum_up += __shfl_down(sum_up, offset, 64);
    }

    if (lane == 0) {
        const float silu_g = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = silu_g * sum_up;
    }
}

template <unsigned NumTiles>
__global__ void q4k_wave_fused_gate_up_swiglu_paired_kernel(
    const Q4KWaveSwigluFusedTile* __restrict__ weights_fused,
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
        const auto& w = weights_fused[row * NumTiles + tile];
        
        // Coalesced 64-bit dwordx2 loads for plane 0 and plane 1
        const uint64_t pair0 = w.words_p0[lane];
        const int v0_g = static_cast<int>(pair0);
        const int v0_u = static_cast<int>(pair0 >> 32);

        const uint64_t pair1 = w.words_p1[lane];
        const int v1_g = static_cast<int>(pair1);
        const int v1_u = static_cast<int>(pair1 >> 32);

        const auto& m_g = w.metadata_gate[lane / 16];
        const auto& m_u = w.metadata_up[lane / 16];

        const uint16_t sc_g = *reinterpret_cast<const uint16_t*>(&m_g.scales[group]);
        const uint16_t min_g = *reinterpret_cast<const uint16_t*>(&m_g.minimums[group]);
        const float d_g = __half2float(m_g.d);
        const float dmin_g = __half2float(m_g.dmin);

        const uint16_t sc_u = *reinterpret_cast<const uint16_t*>(&m_u.scales[group]);
        const uint16_t min_u = *reinterpret_cast<const uint16_t*>(&m_u.minimums[group]);
        const float d_u = __half2float(m_u.d);
        const float dmin_u = __half2float(m_u.dmin);

        float sd_g = 0.0f, sm_g = 0.0f;
        float sd_u = 0.0f, sm_u = 0.0f;

        #pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const auto& q = input[(tile * 4 + lane / 16) * 8 + group + part];
            const auto* u = reinterpret_cast<const int*>(q.qs) + (in_block % 4);

            const int a_g = (part == 0) ? (v0_g & 0x0f0f0f0f) : ((v0_g >> 4) & 0x0f0f0f0f);
            const int b_g = (part == 0) ? (v1_g & 0x0f0f0f0f) : ((v1_g >> 4) & 0x0f0f0f0f);
            const int dot_g = __builtin_amdgcn_sdot4(a_g, u[0],
                __builtin_amdgcn_sdot4(b_g, u[4], 0, false), false);

            const int a_u = (part == 0) ? (v0_u & 0x0f0f0f0f) : ((v0_u >> 4) & 0x0f0f0f0f);
            const int b_u = (part == 0) ? (v1_u & 0x0f0f0f0f) : ((v1_u >> 4) & 0x0f0f0f0f);
            const int dot_u = __builtin_amdgcn_sdot4(a_u, u[0],
                __builtin_amdgcn_sdot4(b_u, u[4], 0, false), false);

            const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
                __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);

            const float q_d = __half2float(q.d);
            const float q_ones = q_d * float(ones);

            const int scale_g = (part == 0) ? (sc_g & 0xff) : (sc_g >> 8);
            const int minimum_g = (part == 0) ? (min_g & 0xff) : (min_g >> 8);
            sd_g += q_d * float(dot_g * scale_g);
            sm_g += q_ones * float(minimum_g);

            const int scale_u = (part == 0) ? (sc_u & 0xff) : (sc_u >> 8);
            const int minimum_u = (part == 0) ? (min_u & 0xff) : (min_u >> 8);
            sd_u += q_d * float(dot_u * scale_u);
            sm_u += q_ones * float(minimum_u);
        }
        sum_gate += d_g * sd_g - dmin_g * sm_g;
        sum_up += d_u * sd_u - dmin_u * sm_u;
    }

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
}

template <unsigned NumTiles>
__global__ void q4k_wave_fused_gate_up_swiglu_paired_legacy_kernel(
    const Q4KWaveSwigluFusedTile* __restrict__ weights_fused,
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
        const auto& w = weights_fused[row * NumTiles + tile];
        
        const uint64_t pair0 = w.words_p0[lane];
        const int v0_g = static_cast<int>(pair0);
        const int v0_u = static_cast<int>(pair0 >> 32);

        const uint64_t pair1 = w.words_p1[lane];
        const int v1_g = static_cast<int>(pair1);
        const int v1_u = static_cast<int>(pair1 >> 32);

        const auto* m_g = &w.metadata_gate[lane / 16];
        const auto* m_u = &w.metadata_up[lane / 16];

        float sd_g = 0.0f, sm_g = 0.0f;
        float sd_u = 0.0f, sm_u = 0.0f;

        #pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const auto& q = input[(tile * 4 + lane / 16) * 8 + group + part];
            const auto* u = reinterpret_cast<const int*>(q.qs) + (in_block % 4);

            const int a_g = (part == 0) ? (v0_g & 0x0f0f0f0f) : ((v0_g >> 4) & 0x0f0f0f0f);
            const int b_g = (part == 0) ? (v1_g & 0x0f0f0f0f) : ((v1_g >> 4) & 0x0f0f0f0f);
            const int dot_g = __builtin_amdgcn_sdot4(a_g, u[0],
                __builtin_amdgcn_sdot4(b_g, u[4], 0, false), false);

            const int a_u = (part == 0) ? (v0_u & 0x0f0f0f0f) : ((v0_u >> 4) & 0x0f0f0f0f);
            const int b_u = (part == 0) ? (v1_u & 0x0f0f0f0f) : ((v1_u >> 4) & 0x0f0f0f0f);
            const int dot_u = __builtin_amdgcn_sdot4(a_u, u[0],
                __builtin_amdgcn_sdot4(b_u, u[4], 0, false), false);

            const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
                __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);

            const float q_d = __half2float(q.d);
            const float q_ones = q_d * float(ones);

            sd_g += q_d * float(dot_g * m_g->scales[group + part]);
            sm_g += q_ones * float(m_g->minimums[group + part]);

            sd_u += q_d * float(dot_u * m_u->scales[group + part]);
            sm_u += q_ones * float(m_u->minimums[group + part]);
        }
        sum_gate += __half2float(m_g->d) * sd_g - __half2float(m_g->dmin) * sm_g;
        sum_up += __half2float(m_u->d) * sd_u - __half2float(m_u->dmin) * sm_u;
    }

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
}

static bool swiglu_1wave_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MIINFER_SWIGLU_1WAVE");
        return env != nullptr && (std::strcmp(env, "1") == 0 || std::strcmp(env, "ON") == 0 || std::strcmp(env, "on") == 0);
    }();
    return enabled;
}

static bool swiglu_shuffle_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MIINFER_SWIGLU_SHUFFLE");
        return env == nullptr || (std::strcmp(env, "0") != 0 && std::strcmp(env, "OFF") != 0 && std::strcmp(env, "off") != 0);
    }();
    return enabled;
}

static bool kquant_fast_arith_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MIINFER_KQUANT_FAST_ARITH");
        return env == nullptr || (std::strcmp(env, "0") != 0 && std::strcmp(env, "OFF") != 0 && std::strcmp(env, "off") != 0);
    }();
    return enabled;
}

} // namespace

void launch_q4k_wave_gemv(const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          std::uint32_t rows, std::uint32_t columns, hipStream_t stream) {
    if (kquant_fast_arith_enabled()) {
        launch_kquant_gemv_dispatch<Q4KDecoderFast>(w, x, y, rows, columns, stream);
    } else {
        launch_kquant_gemv_dispatch<Q4KDecoderLegacy>(w, x, y, rows, columns, stream);
    }
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
    switch (num_tiles) {
    case 5: {
        if (swiglu_1wave_enabled()) {
            const dim3 grid((rows + 3) / 4);
            const dim3 block(256);
            auto kernel = q4k_wave_fused_gate_up_swiglu_1wave_kernel<5>;
            hipLaunchKernelGGL(kernel, grid, block, 0, stream, w_gate, w_up, x, y_activation, rows);
        } else if (swiglu_shuffle_enabled()) {
            const dim3 grid(rows / 2);
            const dim3 block(256);
            auto kernel = q4k_wave_fused_gate_up_swiglu_kernel<5, true>;
            hipLaunchKernelGGL(kernel, grid, block, 0, stream, w_gate, w_up, x, y_activation);
        } else {
            const dim3 grid(rows / 2);
            const dim3 block(256);
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

void launch_q4k_wave_fused_gate_up_swiglu_paired(
    const Q4KWaveSwigluFusedTile* w_fused,
    const miinfer::Q8_1Block* x,
    float* y_activation,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (!w_fused || !x || !y_activation) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    if (rows % 2 != 0 || columns % 1024 != 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    const unsigned num_tiles = columns / 1024;
    switch (num_tiles) {
    case 5: {
        const dim3 grid(rows / 2);
        const dim3 block(256);
        if (kquant_fast_arith_enabled()) {
            auto kernel = q4k_wave_fused_gate_up_swiglu_paired_kernel<5>;
            hipLaunchKernelGGL(kernel, grid, block, 0, stream, w_fused, x, y_activation);
        } else {
            auto kernel = q4k_wave_fused_gate_up_swiglu_paired_legacy_kernel<5>;
            hipLaunchKernelGGL(kernel, grid, block, 0, stream, w_fused, x, y_activation);
        }
        break;
    }
    default:
        throw std::runtime_error("unsupported NumTiles in launch_q4k_wave_fused_gate_up_swiglu_paired: " + std::to_string(num_tiles));
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4k_wave_down(const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y, hipStream_t stream) {
    launch_q4k_wave_gemv(w, x, y, 5120, 17408, stream);
}

void launch_q5k_wave_gemv(const Q5KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          std::uint32_t rows, std::uint32_t columns, hipStream_t stream) {
    if (kquant_fast_arith_enabled()) {
        launch_kquant_gemv_dispatch<Q5KDecoderFast>(w, x, y, rows, columns, stream);
    } else {
        launch_kquant_gemv_dispatch<Q5KDecoderLegacy>(w, x, y, rows, columns, stream);
    }
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
