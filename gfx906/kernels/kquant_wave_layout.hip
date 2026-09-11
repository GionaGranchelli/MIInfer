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

template <typename Decoder, unsigned NumTiles, unsigned Batch = 4>
__global__ void kquant_wave_gemv_batched_kernel(
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

    float sums[Batch] = {};
    for (unsigned tile = 0; tile < NumTiles; ++tile) {
        const auto& w = weights[row * NumTiles + tile];
        typename Decoder::LaneState states[Batch];
        #pragma unroll
        for (unsigned batch = 0; batch < Batch; ++batch) {
            Decoder::load_tile(w, lane, states[batch]);
        }

        #pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            #pragma unroll
            for (unsigned batch = 0; batch < Batch; ++batch) {
                const auto& q = input[batch * NumTiles * 32 + (tile * 4 + lane / 16) * 8 + group + part];
                const auto* u = reinterpret_cast<const int*>(q.qs) + (in_block % 4);
                Decoder::process_part(states[batch], part, group, q, u);
            }
        }
        #pragma unroll
        for (unsigned batch = 0; batch < Batch; ++batch) {
            sums[batch] += Decoder::finalize_tile(states[batch]);
        }
    }

    #pragma unroll
    for (unsigned batch = 0; batch < Batch; ++batch) {
        #pragma unroll
        for (int offset = 32; offset > 0; offset /= 2) {
            sums[batch] += __shfl_down(sums[batch], offset, 64);
        }
        if (lane == 0) output[batch * rows + row] = sums[batch];
    }
}

template <unsigned NumTiles, unsigned Batch = 4>
__global__ void q4k_wave_gemv_batched4_reuse_kernel(
    const Q4KWaveTile* __restrict__ weights,
    const miinfer::Q8_1Block* __restrict__ input,
    float* __restrict__ output,
    unsigned rows,
    unsigned input_stride_blocks) {
    const unsigned wave_id = threadIdx.x / 64;
    const unsigned row = blockIdx.x * 4 + wave_id;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x % 64;
    const unsigned in_block = lane % 16;
    const unsigned group = 2 * (in_block / 4);
    float sums[Batch] = {};

    for (unsigned tile = 0; tile < NumTiles; ++tile) {
        const auto& w = weights[row * NumTiles + tile];
        const auto& meta = w.metadata[lane / 16];
        const unsigned group2 = 2 * (in_block / 4);
        const std::uint16_t scales2 = *reinterpret_cast<const std::uint16_t*>(&meta.scales[group2]);
        const std::uint16_t mins2 = *reinterpret_cast<const std::uint16_t*>(&meta.minimums[group2]);
        const float d = __half2float(meta.d);
        const float dmin = __half2float(meta.dmin);
        const std::uint32_t v0 = w.words[0][lane];
        const std::uint32_t v1 = w.words[1][lane];
        float sd[Batch] = {};
        float sm[Batch] = {};
        #pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const int a = (part == 0) ? (v0 & 0x0f0f0f0f) : ((v0 >> 4) & 0x0f0f0f0f);
            const int b = (part == 0) ? (v1 & 0x0f0f0f0f) : ((v1 >> 4) & 0x0f0f0f0f);
            #pragma unroll
            for (unsigned batch = 0; batch < Batch; ++batch) {
                const auto& q = input[batch * input_stride_blocks
                                       + (tile * 4 + lane / 16) * 8 + group + part];
                const auto* u = reinterpret_cast<const int*>(q.qs) + (in_block % 4);
                const int qdot = __builtin_amdgcn_sdot4(a, u[0],
                    __builtin_amdgcn_sdot4(b, u[4], 0, false), false);
                const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
                    __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);
                const float qd = __half2float(q.d);
                const int scale = (part == 0) ? (scales2 & 0xff) : (scales2 >> 8);
                const int minimum = (part == 0) ? (mins2 & 0xff) : (mins2 >> 8);
                sd[batch] += qd * float(qdot * scale);
                sm[batch] += (qd * float(ones)) * float(minimum);
            }
        }
        #pragma unroll
        for (unsigned batch = 0; batch < Batch; ++batch) {
            sums[batch] += d * sd[batch] - dmin * sm[batch];
        }
    }

    #pragma unroll
    for (unsigned batch = 0; batch < Batch; ++batch) {
        #pragma unroll
        for (int offset = 32; offset > 0; offset /= 2) sums[batch] += __shfl_down(sums[batch], offset, 64);
        if (lane == 0) output[batch * rows + row] = sums[batch];
    }
}


template <unsigned NumTiles>
// Unlike the generic decoder-state batch kernel, this mapping decodes each
// weight word once and keeps only the four output sums live.
__global__ void q5k_wave_gemv_batched4_reuse_kernel(
    const Q5KWaveTile* __restrict__ weights,
    const miinfer::Q8_1Block* __restrict__ input,
    float* __restrict__ output,
    unsigned rows) {
    const unsigned wave_id = threadIdx.x / 64;
    const unsigned row = blockIdx.x * 4 + wave_id;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x % 64;
    const unsigned in_block = lane % 16;
    const unsigned group = 2 * (in_block / 4);
    float sums[4] = {};

    for (unsigned tile = 0; tile < NumTiles; ++tile) {
        const auto& w = weights[row * NumTiles + tile];
        const auto& meta = w.metadata[lane / 16];
        const unsigned group2 = 2 * (in_block / 4);
        const std::uint16_t scales2 = *reinterpret_cast<const std::uint16_t*>(&meta.scales[group2]);
        const std::uint16_t mins2 = *reinterpret_cast<const std::uint16_t*>(&meta.minimums[group2]);
        const float d = __half2float(meta.d);
        const float dmin = __half2float(meta.dmin);
        const std::uint32_t v0 = w.words[0][lane];
        const std::uint32_t v1 = w.words[1][lane];
        const std::uint32_t h = w.qh[lane];
        float sd[4] = {};
        float sm[4] = {};
        #pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const std::uint32_t low_a = (part == 0) ? (v0 & 0x0f0f0f0f) : ((v0 >> 4) & 0x0f0f0f0f);
            const std::uint32_t low_b = (part == 0) ? (v1 & 0x0f0f0f0f) : ((v1 >> 4) & 0x0f0f0f0f);
            const std::uint32_t high_a = (part == 0) ? ((h << 4) & 0x10101010) : ((h << 3) & 0x10101010);
            const std::uint32_t high_b = (part == 0) ? ((h << 2) & 0x10101010) : ((h << 1) & 0x10101010);
            const int a = static_cast<int>(low_a | high_a);
            const int b = static_cast<int>(low_b | high_b);
            const int scale = (part == 0) ? (scales2 & 0xff) : (scales2 >> 8);
            const int minimum = (part == 0) ? (mins2 & 0xff) : (mins2 >> 8);
            #pragma unroll
            for (unsigned batch = 0; batch < 4; ++batch) {
                const auto& q = input[batch * NumTiles * 32 + (tile * 4 + lane / 16) * 8 + group + part];
                const auto* u = reinterpret_cast<const int*>(q.qs) + (in_block % 4);
                const int qdot = __builtin_amdgcn_sdot4(a, u[0],
                    __builtin_amdgcn_sdot4(b, u[4], 0, false), false);
                const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
                    __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);
                const float qd = __half2float(q.d);
                sd[batch] += qd * float(qdot * scale);
                sm[batch] += (qd * float(ones)) * float(minimum);
            }
        }
        #pragma unroll
        for (unsigned batch = 0; batch < 4; ++batch) {
            sums[batch] += d * sd[batch] - dmin * sm[batch];
        }
    }

    #pragma unroll
    for (unsigned batch = 0; batch < 4; ++batch) {
        #pragma unroll
        for (int offset = 32; offset > 0; offset /= 2) {
            sums[batch] += __shfl_down(sums[batch], offset, 64);
        }
        if (lane == 0) output[batch * rows + row] = sums[batch];
    }
}

template <unsigned NumTiles, unsigned Batch = 4>
__global__ void q6k_wave_gemv_batched4_reuse_kernel(
    const Q6KWaveTile* __restrict__ weights,
    const miinfer::Q8_1Block* __restrict__ input,
    float* __restrict__ output,
    unsigned rows,
    unsigned input_stride_blocks) {
    const unsigned wave_id = threadIdx.x / 64;
    const unsigned row = blockIdx.x * 4 + wave_id;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x % 64;
    const unsigned in_block = lane % 16;
    const unsigned group = 2 * (in_block / 4);
    float sums[Batch] = {};

    for (unsigned tile = 0; tile < NumTiles; ++tile) {
        const auto& w = weights[row * NumTiles + tile];
        const auto& meta = w.metadata[lane / 16];
        const unsigned group2 = 2 * (in_block / 4);
        const std::uint32_t scales_word = *reinterpret_cast<const std::uint32_t*>(&meta.scales[2 * group2]);
        const float d = __half2float(meta.d);
        const std::uint32_t v0 = w.words[0][lane];
        const std::uint32_t v1 = w.words[1][lane];
        const std::uint32_t h = w.qh[lane];
        float sd[Batch] = {};
        #pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const std::uint32_t low_a = (part == 0) ? (v0 & 0x0f0f0f0f) : ((v0 >> 4) & 0x0f0f0f0f);
            const std::uint32_t low_b = (part == 0) ? (v1 & 0x0f0f0f0f) : ((v1 >> 4) & 0x0f0f0f0f);
            const std::uint32_t high_a = (part == 0) ? ((h & 0x03030303) << 4) : (h & 0x30303030);
            const std::uint32_t high_b = (part == 0) ? ((h & 0x0c0c0c0c) << 2) : ((h >> 2) & 0x30303030);
            const std::uint32_t u_a = low_a | high_a;
            const std::uint32_t u_b = low_b | high_b;
            const std::uint32_t not_b5_a = (u_a & 0x20202020) ^ 0x20202020;
            const std::uint32_t a = (u_a & 0x1f1f1f1f) | (not_b5_a | (not_b5_a << 1) | (not_b5_a << 2));
            const std::uint32_t not_b5_b = (u_b & 0x20202020) ^ 0x20202020;
            const std::uint32_t b = (u_b & 0x1f1f1f1f) | (not_b5_b | (not_b5_b << 1) | (not_b5_b << 2));
            #pragma unroll
            for (unsigned batch = 0; batch < Batch; ++batch) {
                const auto& q = input[batch * input_stride_blocks
                                       + (tile * 4 + lane / 16) * 8 + group + part];
                const auto* u = reinterpret_cast<const int*>(q.qs) + (in_block % 4);
                const int dot_a = __builtin_amdgcn_sdot4(static_cast<int>(a), u[0], 0, false);
                const int dot_b = __builtin_amdgcn_sdot4(static_cast<int>(b), u[4], 0, false);
                const int sc_a = static_cast<std::int8_t>((scales_word >> (part * 16)) & 0xff);
                const int sc_b = static_cast<std::int8_t>((scales_word >> (part * 16 + 8)) & 0xff);
                sd[batch] += __half2float(q.d) * float(dot_a * sc_a + dot_b * sc_b);
            }
        }
        #pragma unroll
        for (unsigned batch = 0; batch < Batch; ++batch) sums[batch] += d * sd[batch];
    }

    #pragma unroll
    for (unsigned batch = 0; batch < Batch; ++batch) {
        #pragma unroll
        for (int offset = 32; offset > 0; offset /= 2) sums[batch] += __shfl_down(sums[batch], offset, 64);
        if (lane == 0) output[batch * rows + row] = sums[batch];
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
__global__ void q4k_wave_fused_gate_up_swiglu_paired_batched4_kernel(
    const Q4KWaveSwigluFusedTile* __restrict__ weights_fused,
    const miinfer::Q8_1Block* __restrict__ input,
    float* __restrict__ output,
    unsigned rows) {
    const unsigned wave_id = threadIdx.x / 64;
    const unsigned row = blockIdx.x * 4 + wave_id;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x % 64;
    const unsigned in_block = lane % 16;
    const unsigned group = 2 * (in_block / 4);
    float sum_gate[4] = {};
    float sum_up[4] = {};

    for (unsigned tile = 0; tile < NumTiles; ++tile) {
        const auto& w = weights_fused[row * NumTiles + tile];
        const uint64_t pair0 = w.words_p0[lane];
        const uint64_t pair1 = w.words_p1[lane];
        const int v0_g = static_cast<int>(pair0);
        const int v0_u = static_cast<int>(pair0 >> 32);
        const int v1_g = static_cast<int>(pair1);
        const int v1_u = static_cast<int>(pair1 >> 32);
        const auto& m_g = w.metadata_gate[lane / 16];
        const auto& m_u = w.metadata_up[lane / 16];
        const std::uint16_t sc_g = *reinterpret_cast<const std::uint16_t*>(&m_g.scales[group]);
        const std::uint16_t min_g = *reinterpret_cast<const std::uint16_t*>(&m_g.minimums[group]);
        const std::uint16_t sc_u = *reinterpret_cast<const std::uint16_t*>(&m_u.scales[group]);
        const std::uint16_t min_u = *reinterpret_cast<const std::uint16_t*>(&m_u.minimums[group]);
        const float d_g = __half2float(m_g.d);
        const float dmin_g = __half2float(m_g.dmin);
        const float d_u = __half2float(m_u.d);
        const float dmin_u = __half2float(m_u.dmin);
        float sd_g[4] = {};
        float sm_g[4] = {};
        float sd_u[4] = {};
        float sm_u[4] = {};

        #pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const int a_g = (part == 0) ? (v0_g & 0x0f0f0f0f) : ((v0_g >> 4) & 0x0f0f0f0f);
            const int b_g = (part == 0) ? (v1_g & 0x0f0f0f0f) : ((v1_g >> 4) & 0x0f0f0f0f);
            const int a_u = (part == 0) ? (v0_u & 0x0f0f0f0f) : ((v0_u >> 4) & 0x0f0f0f0f);
            const int b_u = (part == 0) ? (v1_u & 0x0f0f0f0f) : ((v1_u >> 4) & 0x0f0f0f0f);
            const int scale_g = (part == 0) ? (sc_g & 0xff) : (sc_g >> 8);
            const int minimum_g = (part == 0) ? (min_g & 0xff) : (min_g >> 8);
            const int scale_u = (part == 0) ? (sc_u & 0xff) : (sc_u >> 8);
            const int minimum_u = (part == 0) ? (min_u & 0xff) : (min_u >> 8);
            #pragma unroll
            for (unsigned batch = 0; batch < 4; ++batch) {
                const auto& q = input[batch * NumTiles * 32 + (tile * 4 + lane / 16) * 8 + group + part];
                const auto* u = reinterpret_cast<const int*>(q.qs) + (in_block % 4);
                const int dot_g = __builtin_amdgcn_sdot4(a_g, u[0],
                    __builtin_amdgcn_sdot4(b_g, u[4], 0, false), false);
                const int dot_u = __builtin_amdgcn_sdot4(a_u, u[0],
                    __builtin_amdgcn_sdot4(b_u, u[4], 0, false), false);
                const int ones = __builtin_amdgcn_sdot4(0x01010101, u[0],
                    __builtin_amdgcn_sdot4(0x01010101, u[4], 0, false), false);
                const float q_d = __half2float(q.d);
                sd_g[batch] += q_d * float(dot_g * scale_g);
                sm_g[batch] += q_d * float(ones * minimum_g);
                sd_u[batch] += q_d * float(dot_u * scale_u);
                sm_u[batch] += q_d * float(ones * minimum_u);
            }
        }
        #pragma unroll
        for (unsigned batch = 0; batch < 4; ++batch) {
            sum_gate[batch] += d_g * sd_g[batch] - dmin_g * sm_g[batch];
            sum_up[batch] += d_u * sd_u[batch] - dmin_u * sm_u[batch];
        }
    }

    #pragma unroll
    for (unsigned batch = 0; batch < 4; ++batch) {
        #pragma unroll
        for (int offset = 32; offset > 0; offset /= 2) {
            sum_gate[batch] += __shfl_down(sum_gate[batch], offset, 64);
            sum_up[batch] += __shfl_down(sum_up[batch], offset, 64);
        }
        if (lane == 0) {
            const float silu_gate = sum_gate[batch] / (1.0f + expf(-sum_gate[batch]));
            output[batch * rows + row] = silu_gate * sum_up[batch];
        }
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

static bool m23_repacked_row128_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MIINFER_M23_REPACKED_ROW128");
        return env != nullptr && std::strcmp(env, "0") != 0
            && std::strcmp(env, "OFF") != 0 && std::strcmp(env, "off") != 0;
    }();
    return enabled;
}

// Legacy 64-row fallback retained until the row-128 Q6 mapping passes GPU parity.
[[maybe_unused]] __launch_bounds__(256, 1) __global__ void m23_q6k_repacked_mmq_kernel(
    const Q6KMmqTile* __restrict__ weights,
    const miinfer::M23Q8_1MmqBlock* __restrict__ input,
    float* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count) {
    const std::uint32_t row_tile = blockIdx.x;
    const std::uint32_t tile_base = blockIdx.y * 128;
    const std::uint32_t token_base = tile_base + threadIdx.x;
    const std::uint32_t row_base = row_tile * 64 + threadIdx.y * 16;
    const std::uint32_t k_tiles = columns / 128;
    const std::uint32_t tid = threadIdx.y * 64 + threadIdx.x;
    __shared__ std::uint32_t expanded[2][4][64][4];
    __shared__ miinfer::M23Q8_1MmqBlock activation_tile[128];
    float accum[16][2] = {};

    for (std::uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const auto* source_activation = input + static_cast<std::size_t>(k_tile) * token_count + tile_base;
        constexpr std::uint32_t activation_words = sizeof(miinfer::M23Q8_1MmqBlock) / sizeof(uint4);
        const std::uint32_t active_tokens = tile_base < token_count
            ? std::min<std::uint32_t>(128, token_count - tile_base) : 0;
        for (std::uint32_t index = tid; index < active_tokens * activation_words; index += 256) {
            reinterpret_cast<uint4*>(activation_tile)[index] = reinterpret_cast<const uint4*>(source_activation)[index];
        }
        __syncthreads();
        for (std::uint32_t index = tid; index < 2 * 4 * 64 * 4; index += 256) {
            const std::uint32_t slab = index / (2 * 64 * 4);
            const std::uint32_t rem = index % (2 * 64 * 4);
            const std::uint32_t half = rem / (64 * 4);
            const std::uint32_t row = (rem % (64 * 4)) / 4;
            const std::uint32_t word = rem % 4;
            const auto* source_weight = weights
                + static_cast<std::size_t>(row_tile) * k_tiles + k_tile;
            const std::uint32_t raw = source_weight->values[slab][row][word];
            const std::uint64_t high = static_cast<std::uint64_t>(source_weight->high[slab][row][0])
                | (static_cast<std::uint64_t>(source_weight->high[slab][row][1]) << 32);
            std::uint32_t value = 0;
            for (std::uint32_t byte = 0; byte < 4; ++byte) {
                const std::uint32_t value_index = half * 16 + word * 4 + byte;
                const int q = static_cast<int>((raw >> (4 * half + 8 * byte)) & 0x0fU)
                    | static_cast<int>(((high >> (2 * value_index)) & 0x03U) << 4);
                value |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(q - 32)) << (8 * byte);
            }
            expanded[half][slab][row][word] = value;
        }
        __syncthreads();

        float input_d[2][4] = {};
        for (int output_column = 0; output_column < 2; ++output_column) {
            const std::uint32_t token = token_base + output_column * 64;
            if (token >= token_count) continue;
            const auto& x = activation_tile[threadIdx.x + output_column * 64];
#pragma unroll
            for (int slab = 0; slab < 4; ++slab) {
                input_d[output_column][slab] = __half2float(__float2half_rn(x.d[slab]));
            }
        }

#pragma unroll
        for (int row_offset = 0; row_offset < 16; ++row_offset) {
#pragma unroll
            for (int slab = 0; slab < 4; ++slab) {
                const auto* source_weight = weights
                    + static_cast<std::size_t>(row_tile) * k_tiles + k_tile;
                const std::uint32_t scale_d_lane = threadIdx.x == 0
                    ? source_weight->scale_d[slab][threadIdx.y * 16 + row_offset]
                    : 0;
                const std::uint32_t scale_d = __shfl(scale_d_lane, 0, 64);
                const std::uint16_t d_bits = static_cast<std::uint16_t>(scale_d >> 16);
                const float weight_d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
                const int scales[2] = {static_cast<std::int8_t>(scale_d),
                                       static_cast<std::int8_t>(scale_d >> 8)};
#pragma unroll
                for (int output_column = 0; output_column < 2; ++output_column) {
                    const std::uint32_t token = token_base + output_column * 64;
                    if (token >= token_count) continue;
                    const auto& x = activation_tile[threadIdx.x + output_column * 64];
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        int dot = 0;
#pragma unroll
                        for (int word = 0; word < 4; ++word) {
                            const int x_word = static_cast<int>(*reinterpret_cast<const std::uint32_t*>(
                                x.qs + slab * 32 + half * 16 + word * 4));
                            dot = __builtin_amdgcn_sdot4(
                                static_cast<int>(expanded[half][slab][threadIdx.y * 16 + row_offset][word]),
                                x_word, dot, false);
                        }
                        accum[row_offset][output_column] += weight_d * input_d[output_column][slab]
                            * static_cast<float>(scales[half] * dot);
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int row_offset = 0; row_offset < 16; ++row_offset) {
        const std::uint32_t row = row_base + row_offset;
        if (token_base < token_count) output[static_cast<std::size_t>(token_base) * rows + row] = accum[row_offset][0];
        if (token_base + 64 < token_count) output[static_cast<std::size_t>(token_base + 64) * rows + row] = accum[row_offset][1];
    }
}

[[maybe_unused]] __launch_bounds__(256, 1) __global__ void m23_q6k_repacked_mmq_row128_kernel(
    const Q6KMmqTile* __restrict__ weights,
    const miinfer::M23Q8_1MmqBlock* __restrict__ input,
    float* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count) {
    const std::uint32_t row_tile = blockIdx.x;
    const std::uint32_t token_base = blockIdx.y * 64;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t warp = threadIdx.y;
    const std::uint32_t tid = warp * 64 + lane;
    const std::uint32_t k_tiles = columns / 128;
    // Keep only expanded weights in LDS; raw tiles are read coalesced from
    // global memory during expansion so two resident workgroups fit on a CU.
    __shared__ std::uint32_t expanded[2][2][4][64][4];
    __shared__ miinfer::M23Q8_1MmqBlock activation_tile[64];
    float accum[2][16] = {};

    for (std::uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        constexpr std::uint32_t activation_words = sizeof(miinfer::M23Q8_1MmqBlock) / sizeof(uint4);
        const auto* source_activation = input + static_cast<std::size_t>(k_tile) * token_count + token_base;
        const std::uint32_t active_tokens = token_base < token_count
            ? std::min<std::uint32_t>(64, token_count - token_base) : 0;
        for (std::uint32_t index = tid; index < active_tokens * activation_words; index += 256) {
            reinterpret_cast<uint4*>(activation_tile)[index] = reinterpret_cast<const uint4*>(source_activation)[index];
        }
        __syncthreads();
        for (std::uint32_t index = tid; index < 2 * 2 * 4 * 64 * 4; index += 256) {
            const std::uint32_t row_half = index / (2 * 4 * 64 * 4);
            const std::uint32_t local_index = index % (2 * 4 * 64 * 4);
            const std::uint32_t slab = local_index / (2 * 64 * 4);
            const std::uint32_t rem = local_index % (2 * 64 * 4);
            const std::uint32_t half = rem / (64 * 4);
            const std::uint32_t row = (rem % (64 * 4)) / 4;
            const std::uint32_t word = rem % 4;
            const bool valid = row_tile * 128 + row_half * 64 < rows;
            const auto* source_weight = weights
                + static_cast<std::size_t>(row_tile * 2 + row_half) * k_tiles + k_tile;
            const std::uint32_t raw = valid ? source_weight->values[slab][row][word] : 0;
            const std::uint64_t high = valid
                ? static_cast<std::uint64_t>(source_weight->high[slab][row][0])
                    | (static_cast<std::uint64_t>(source_weight->high[slab][row][1]) << 32)
                : 0;
            std::uint32_t value = 0;
            for (std::uint32_t byte = 0; byte < 4; ++byte) {
                const std::uint32_t value_index = half * 16 + word * 4 + byte;
                const int q = static_cast<int>((raw >> (4 * half + 8 * byte)) & 0x0fU)
                    | static_cast<int>(((high >> (2 * value_index)) & 0x03U) << 4);
                value |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(q - 32)) << (8 * byte);
            }
            expanded[row_half][half][slab][row][word] = value;
        }
        __syncthreads();
        for (std::uint32_t row_half = 0; row_half < 2; ++row_half) {
            const std::uint32_t row = row_tile * 128 + row_half * 64 + lane;
            if (row >= rows) continue;
            for (int slab = 0; slab < 4; ++slab) {
                const auto* source_weight = weights
                    + static_cast<std::size_t>(row_tile * 2 + row_half) * k_tiles + k_tile;
                const std::uint32_t scale_d = source_weight->scale_d[slab][lane];
                const std::uint16_t d_bits = static_cast<std::uint16_t>(scale_d >> 16);
                const float weight_d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
                const int scales[2] = {static_cast<std::int8_t>(scale_d),
                                       static_cast<std::int8_t>(scale_d >> 8)};
                for (int j0 = 0; j0 < 64; j0 += 4) {
                    const std::uint32_t token = j0 + warp;
                    if (token >= token_count - token_base) continue;
                    const auto& x = activation_tile[token];
                    const float input_d = __half2float(__float2half_rn(x.d[slab]));
                    for (int half = 0; half < 2; ++half) {
                        int dot = 0;
                        for (int word = 0; word < 4; ++word) {
                            const int x_word = static_cast<int>(*reinterpret_cast<const std::uint32_t*>(
                                x.qs + slab * 32 + half * 16 + word * 4));
                            dot = __builtin_amdgcn_sdot4(
                                static_cast<int>(expanded[row_half][half][slab][lane][word]),
                                x_word, dot, false);
                        }
                        accum[row_half][j0 / 4] += weight_d * input_d
                            * static_cast<float>(scales[half] * dot);
                    }
                }
            }
        }
        __syncthreads();
    }
    for (std::uint32_t row_half = 0; row_half < 2; ++row_half) {
        const std::uint32_t row = row_tile * 128 + row_half * 64 + lane;
        if (row >= rows) continue;
        for (int j0 = 0; j0 < 64; j0 += 4) {
            const std::uint32_t token = token_base + j0 + warp;
            if (token < token_count) output[static_cast<std::size_t>(token) * rows + row] = accum[row_half][j0 / 4];
        }
    }
}

template <typename Tile>
__launch_bounds__(256, 1)
__global__ void m23_affine_repacked_mmq_kernel(
    const Tile* __restrict__ weights,
    const miinfer::M23Q8_1MmqBlock* __restrict__ input,
    float* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count) {
    const std::uint32_t row_tile = blockIdx.x;
    const std::uint32_t tile_base = blockIdx.y * 128;
    const std::uint32_t token_base = tile_base + threadIdx.x;
    const std::uint32_t row_base = row_tile * 64 + threadIdx.y * 16;
    const std::uint32_t k_tiles = columns / 128;
    const std::uint32_t tid = threadIdx.y * 64 + threadIdx.x;
    __shared__ std::uint32_t expanded[2][4][64][4];
    __shared__ miinfer::M23Q8_1MmqBlock activation_tile[128];
    float accum[16][2] = {};

    for (std::uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const auto* source_activation = input + static_cast<std::size_t>(k_tile) * token_count + tile_base;
        constexpr std::uint32_t activation_words = sizeof(miinfer::M23Q8_1MmqBlock) / sizeof(uint4);
        const std::uint32_t active_tokens = tile_base < token_count
            ? std::min<std::uint32_t>(128, token_count - tile_base) : 0;
        for (std::uint32_t index = tid; index < active_tokens * activation_words; index += 256) {
            reinterpret_cast<uint4*>(activation_tile)[index] = reinterpret_cast<const uint4*>(source_activation)[index];
        }
        __syncthreads();
        for (std::uint32_t index = tid; index < 2 * 4 * 64 * 4; index += 256) {
            const std::uint32_t slab = index / (2 * 64 * 4);
            const std::uint32_t rem = index % (2 * 64 * 4);
            const std::uint32_t half = rem / (64 * 4);
            const std::uint32_t row = (rem % (64 * 4)) / 4;
            const std::uint32_t word = rem % 4;
            const auto* source_weight = weights
                + static_cast<std::size_t>(row_tile) * k_tiles + k_tile;
            const std::uint32_t raw = source_weight->values[slab][row][word];
            const std::uint32_t high = source_weight->high[slab][row][0];
            std::uint32_t value = 0;
            for (std::uint32_t byte = 0; byte < 4; ++byte) {
                const std::uint32_t value_index = half * 16 + word * 4 + byte;
                const std::uint32_t nibble = (raw >> (8 * byte)) >> (half ? 4 : 0) & 0x0fU;
                const std::uint32_t q = nibble | (((high >> value_index) & 1U) << 4);
                value |= q << (8 * byte);
            }
            expanded[half][slab][row][word] = value;
        }
        __syncthreads();

        float input_d[2][4] = {};
        for (int output_column = 0; output_column < 2; ++output_column) {
            const std::uint32_t token = token_base + output_column * 64;
            if (token >= token_count) continue;
            const auto& x = activation_tile[threadIdx.x + output_column * 64];
#pragma unroll
            for (int slab = 0; slab < 4; ++slab) {
                input_d[output_column][slab] = __half2float(__float2half_rn(x.d[slab]));
            }
        }

#pragma unroll
        for (int row_offset = 0; row_offset < 16; ++row_offset) {
#pragma unroll
            for (int slab = 0; slab < 4; ++slab) {
                const auto* source_weight = weights
                    + static_cast<std::size_t>(row_tile) * k_tiles + k_tile;
                const std::uint32_t scale_d_lane = threadIdx.x == 0
                    ? source_weight->scale_d[slab][threadIdx.y * 16 + row_offset]
                    : 0;
                const std::uint32_t scale_d = __shfl(scale_d_lane, 0, 64);
                const std::uint16_t d_bits = static_cast<std::uint16_t>(scale_d >> 16);
                const std::uint16_t dmin_lane = threadIdx.x == 0
                    ? static_cast<std::uint16_t>(source_weight->dmin[slab][threadIdx.y * 16 + row_offset])
                    : 0;
                const std::uint16_t dmin_bits = __shfl(dmin_lane, 0, 64);
                const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
                const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_bits));
                const int scale = static_cast<int>(scale_d & 0xffU);
                const int minimum = static_cast<int>((scale_d >> 8) & 0xffU);
#pragma unroll
                for (int output_column = 0; output_column < 2; ++output_column) {
                    const std::uint32_t token = token_base + output_column * 64;
                    if (token >= token_count) continue;
                    const auto& x = activation_tile[threadIdx.x + output_column * 64];
                    int dot = 0;
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
#pragma unroll
                        for (int word = 0; word < 4; ++word) {
                            const int x_word = static_cast<int>(*reinterpret_cast<const std::uint32_t*>(
                                x.qs + slab * 32 + half * 16 + word * 4));
                            dot = __builtin_amdgcn_sdot4(
                                static_cast<int>(expanded[half][slab][threadIdx.y * 16 + row_offset][word]),
                                x_word, dot, false);
                        }
                    }
                    accum[row_offset][output_column] += d * input_d[output_column][slab]
                        * static_cast<float>(scale * dot)
                        - dmin * static_cast<float>(minimum) * x.qsum_scaled[slab];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int row_offset = 0; row_offset < 16; ++row_offset) {
        const std::uint32_t row = row_base + row_offset;
        if (token_base < token_count) output[static_cast<std::size_t>(token_base) * rows + row] = accum[row_offset][0];
        if (token_base + 64 < token_count) output[static_cast<std::size_t>(token_base + 64) * rows + row] = accum[row_offset][1];
    }
}

template <typename Tile>
__launch_bounds__(256, 1)
__global__ void m23_affine_repacked_mmq_row128_kernel(
    const Tile* __restrict__ weights,
    const miinfer::M23Q8_1MmqBlock* __restrict__ input,
    float* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count) {
    const std::uint32_t row_tile = blockIdx.x;
    const std::uint32_t token_base = blockIdx.y * 64;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t warp = threadIdx.y;
    const std::uint32_t tid = warp * 64 + lane;
    const std::uint32_t k_tiles = columns / 128;
    // Keep only expanded weights in LDS; raw tiles are read coalesced from
    // global memory during expansion so two resident workgroups fit on a CU.
    __shared__ std::uint32_t expanded[2][2][4][64][4];
    __shared__ miinfer::M23Q8_1MmqBlock activation_tile[64];
    float accum[2][16] = {};

    for (std::uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        constexpr std::uint32_t activation_words = sizeof(miinfer::M23Q8_1MmqBlock) / sizeof(uint4);
        const auto* source_activation = input + static_cast<std::size_t>(k_tile) * token_count + token_base;
        const std::uint32_t active_tokens = token_base < token_count
            ? std::min<std::uint32_t>(64, token_count - token_base) : 0;
        for (std::uint32_t index = tid; index < active_tokens * activation_words; index += 256) {
            reinterpret_cast<uint4*>(activation_tile)[index] = reinterpret_cast<const uint4*>(source_activation)[index];
        }
        __syncthreads();
        for (std::uint32_t index = tid; index < 2 * 2 * 4 * 64 * 4; index += 256) {
            const std::uint32_t row_half = index / (2 * 4 * 64 * 4);
            const std::uint32_t local_index = index % (2 * 4 * 64 * 4);
            const std::uint32_t slab = local_index / (2 * 64 * 4);
            const std::uint32_t rem = local_index % (2 * 64 * 4);
            const std::uint32_t half = rem / (64 * 4);
            const std::uint32_t row = (rem % (64 * 4)) / 4;
            const std::uint32_t word = rem % 4;
            const bool valid = row_tile * 128 + row_half * 64 < rows;
            const auto* source_weight = weights
                + static_cast<std::size_t>(row_tile * 2 + row_half) * k_tiles + k_tile;
            const std::uint32_t raw = valid ? source_weight->values[slab][row][word] : 0;
            const std::uint32_t high = valid ? source_weight->high[slab][row][0] : 0;
            std::uint32_t value = 0;
            for (std::uint32_t byte = 0; byte < 4; ++byte) {
                const std::uint32_t value_index = half * 16 + word * 4 + byte;
                const std::uint32_t nibble = (raw >> (8 * byte)) >> (half ? 4 : 0) & 0x0fU;
                const std::uint32_t q = nibble | (((high >> value_index) & 1U) << 4);
                value |= q << (8 * byte);
            }
            expanded[row_half][half][slab][row][word] = value;
        }
        __syncthreads();
        for (std::uint32_t row_half = 0; row_half < 2; ++row_half) {
            const std::uint32_t row = row_tile * 128 + row_half * 64 + lane;
            if (row >= rows) continue;
            for (int slab = 0; slab < 4; ++slab) {
                const auto* source_weight = weights
                    + static_cast<std::size_t>(row_tile * 2 + row_half) * k_tiles + k_tile;
                const std::uint32_t scale_d = source_weight->scale_d[slab][lane];
                const std::uint16_t d_bits = static_cast<std::uint16_t>(scale_d >> 16);
                const std::uint16_t dmin_bits = source_weight->dmin[slab][lane];
                const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
                const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_bits));
                const int scale = static_cast<int>(scale_d & 0xffU);
                const int minimum = static_cast<int>((scale_d >> 8) & 0xffU);
                for (int j0 = 0; j0 < 64; j0 += 4) {
                    const std::uint32_t token = j0 + warp;
                    if (token >= token_count - token_base) continue;
                    const auto& x = activation_tile[token];
                    const float input_d = __half2float(__float2half_rn(x.d[slab]));
                    int dot = 0;
                    for (int half = 0; half < 2; ++half) {
                        for (int word = 0; word < 4; ++word) {
                            const int x_word = static_cast<int>(*reinterpret_cast<const std::uint32_t*>(
                                x.qs + slab * 32 + half * 16 + word * 4));
                            dot = __builtin_amdgcn_sdot4(
                                static_cast<int>(expanded[row_half][half][slab][lane][word]), x_word, dot, false);
                        }
                    }
                    accum[row_half][j0 / 4] += d * input_d * static_cast<float>(scale * dot)
                        - dmin * static_cast<float>(minimum) * x.qsum_scaled[slab];
                }
            }
        }
        __syncthreads();
    }
    for (std::uint32_t row_half = 0; row_half < 2; ++row_half) {
        const std::uint32_t row = row_tile * 128 + row_half * 64 + lane;
        if (row >= rows) continue;
        for (int j0 = 0; j0 < 64; j0 += 4) {
            const std::uint32_t token = token_base + j0 + warp;
            if (token < token_count) output[static_cast<std::size_t>(token) * rows + row] = accum[row_half][j0 / 4];
        }
    }
}

static __device__ __forceinline__ std::uint32_t mx_spread1(std::uint32_t value) {
    return (value | (value << 7) | (value << 14) | (value << 21)) & 0x01010101U;
}

static __device__ __forceinline__ std::uint32_t mx_spread2(std::uint32_t value) {
    return (value | (value << 6) | (value << 12) | (value << 18)) & 0x03030303U;
}

static __device__ __forceinline__ int mx_dp4a(
    const std::uint32_t weight, const std::uint32_t input, int accumulator = 0) {
    return __builtin_amdgcn_sdot4(static_cast<int>(weight), static_cast<int>(input),
                                  accumulator, false);
}

static __device__ __forceinline__ int mx_input_swizzle(int row) {
    int local = row & 63;
    local ^= (local >> 5) << 4;
    return (row & ~63) | local;
}

struct MxWeightWords {
    uint4 lo, hi;
    uint2 slot;
};

template <bool Q5>
struct MxAffineGeometry {
    std::uint32_t qs_stride, n_sub, superblocks;
    std::size_t qs_offset, high_offset, scale_offset;

    __host__ __device__ MxAffineGeometry(std::uint32_t columns, std::uint32_t rows)
        : qs_stride(columns / 2 + (columns / 2 % 128 == 0 ? 16 : 0)),
          n_sub(columns / 32), superblocks((n_sub + 7) / 8),
          qs_offset(static_cast<std::size_t>(rows) * qs_stride),
          high_offset(qs_offset + (Q5 ? static_cast<std::size_t>(rows) * n_sub * 4 : 0)),
          scale_offset(high_offset + static_cast<std::size_t>(rows) * superblocks * 12) {}
};

template <bool Q5>
static __device__ __forceinline__ MxWeightWords mx_load_affine(
    const std::uint8_t* weights, const MxAffineGeometry<Q5>& geometry,
    std::uint32_t row, std::uint32_t subblock) {
    const auto packed = *reinterpret_cast<const uint4*>(weights
        + static_cast<std::size_t>(row) * geometry.qs_stride + subblock * 16);
    MxWeightWords result;
    if constexpr (Q5) {
        const auto high = *reinterpret_cast<const std::uint32_t*>(weights
            + geometry.qs_offset + (static_cast<std::size_t>(row) * geometry.n_sub + subblock) * 4);
        auto* lo = reinterpret_cast<std::uint32_t*>(&result.lo);
        auto* hi = reinterpret_cast<std::uint32_t*>(&result.hi);
        const auto* packed_words = reinterpret_cast<const std::uint32_t*>(&packed);
        for (int word = 0; word < 4; ++word) {
            lo[word] = (packed_words[word] & 0x0F0F0F0FU)
                | (mx_spread1((high >> (4 * word)) & 0x0FU) << 4);
            hi[word] = ((packed_words[word] >> 4) & 0x0F0F0F0FU)
                | (mx_spread1((high >> (16 + 4 * word)) & 0x0FU) << 4);
        }
    } else {
        result.lo = make_uint4(packed.x & 0x0F0F0F0FU, packed.y & 0x0F0F0F0FU,
                               packed.z & 0x0F0F0F0FU, packed.w & 0x0F0F0F0FU);
        result.hi = make_uint4((packed.x >> 4) & 0x0F0F0F0FU,
                               (packed.y >> 4) & 0x0F0F0F0FU,
                               (packed.z >> 4) & 0x0F0F0F0FU,
                               (packed.w >> 4) & 0x0F0F0F0FU);
    }

    const std::size_t superblock = static_cast<std::size_t>(row) * geometry.superblocks
                                   + subblock / 8;
    const auto* scale_words = reinterpret_cast<const std::uint32_t*>(weights
        + geometry.high_offset + superblock * 12);
    const std::uint32_t s0 = scale_words[0], s1 = scale_words[1], s2 = scale_words[2];
    const std::uint32_t index = subblock & 7;
    std::uint32_t scale, minimum;
    if (index < 4) {
        scale = (s0 >> (8 * index)) & 63U;
        minimum = (s1 >> (8 * index)) & 63U;
    } else {
        const std::uint32_t shift = 8 * (index - 4);
        scale = ((s2 >> shift) & 0x0FU) | (((s0 >> (shift + 6)) & 3U) << 4);
        minimum = ((s2 >> (shift + 4)) & 0x0FU) | (((s1 >> (shift + 6)) & 3U) << 4);
    }
    result.slot.x = scale | (minimum << 8);
    result.slot.y = *reinterpret_cast<const std::uint32_t*>(weights
        + geometry.scale_offset + superblock * 4);
    return result;
}

struct MxQ6Geometry {
    std::uint32_t qs_stride, n_sub, superblocks;
    std::size_t qs_offset, high_offset, scale_offset;

    __host__ __device__ MxQ6Geometry(std::uint32_t columns, std::uint32_t rows)
        : qs_stride(columns / 2 + (columns / 2 % 128 == 0 ? 16 : 0)),
          n_sub(columns / 32), superblocks((n_sub + 7) / 8),
          qs_offset(static_cast<std::size_t>(rows) * qs_stride),
          high_offset(qs_offset + static_cast<std::size_t>(rows) * n_sub * 8),
          scale_offset(high_offset + static_cast<std::size_t>(rows) * n_sub * 2) {}
};

static __device__ __forceinline__ MxWeightWords mx_load_q6(
    const std::uint8_t* weights, const MxQ6Geometry& geometry,
    std::uint32_t row, std::uint32_t subblock) {
    const auto packed = *reinterpret_cast<const uint4*>(weights
        + static_cast<std::size_t>(row) * geometry.qs_stride + subblock * 16);
    const auto high = *reinterpret_cast<const uint2*>(weights
        + geometry.qs_offset + (static_cast<std::size_t>(row) * geometry.n_sub + subblock) * 8);
    MxWeightWords result;
    auto* lo = reinterpret_cast<std::uint32_t*>(&result.lo);
    auto* hi = reinterpret_cast<std::uint32_t*>(&result.hi);
    const auto* packed_words = reinterpret_cast<const std::uint32_t*>(&packed);
    for (int word = 0; word < 4; ++word) {
        const std::uint32_t low_high = (high.x >> (8 * word)) & 0xFFU;
        const std::uint32_t high_high = (high.y >> (8 * word)) & 0xFFU;
        const std::uint32_t low_values = (packed_words[word] & 0x0F0F0F0FU)
            | (mx_spread2(low_high) << 4);
        const std::uint32_t high_values = ((packed_words[word] >> 4) & 0x0F0F0F0FU)
            | (mx_spread2(high_high) << 4);
        lo[word] = ((low_values << 2) & 0xFCFCFCFCU) ^ 0x80808080U;
        hi[word] = ((high_values << 2) & 0xFCFCFCFCU) ^ 0x80808080U;
    }
    const std::size_t index = static_cast<std::size_t>(row) * geometry.n_sub + subblock;
    const std::uint32_t scales = *reinterpret_cast<const std::uint16_t*>(weights
        + geometry.high_offset + index * 2);
    const std::uint32_t d = *reinterpret_cast<const std::uint16_t*>(weights
        + geometry.scale_offset + (static_cast<std::size_t>(row) * geometry.superblocks
                                   + subblock / 8) * 2);
    result.slot = make_uint2(scales | (d << 16), 0);
    return result;
}

struct MxInputSlice {
    uint4 q0, q1;
    float d, sum;
};

struct MxSharedInputRow {
    uint4 q[4][2];
    uint4 pad;
};

static __device__ __forceinline__ MxInputSlice mx_load_input(
    const miinfer::MxQ8_1MmqBlock* input, std::uint32_t token, std::uint32_t group,
    bool affine) {
    const auto& block = input[token];
    const auto* quantized = reinterpret_cast<const uint4*>(block.qs + group * 32);
    MxInputSlice result{quantized[0], quantized[1], 0.0F, 0.0F};
    if (affine) {
        result.d = __half2float(block.ds4[group].x);
        result.sum = __half2float(block.ds4[group].y);
    } else {
        result.d = block.d4[group];
    }
    return result;
}

template <bool Q5, bool Q6>
__launch_bounds__(256, 1) __global__ void mx_repacked_mmq_kernel(
    const std::uint8_t* __restrict__ weights,
    const miinfer::MxQ8_1MmqBlock* __restrict__ input,
    float* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count) {
    static_assert(!(Q5 && Q6));
    constexpr int kBm = 64;
    constexpr int kBn = 128;
    constexpr int kBk = 4;
    constexpr int kTn = 2;
    constexpr int kNrow = kBm / 4;
    constexpr int kThreads = 256;
    constexpr int kWeightElements = kBm * kBk;
    constexpr int kInputElements = kBn * kBk;
    constexpr int kWeightElementsPerThread = kWeightElements / kThreads;
    constexpr int kInputElementsPerThread = kInputElements / kThreads;
    static_assert(kWeightElementsPerThread == 1 && kInputElementsPerThread == 2);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 64 + tx;
    const std::uint32_t row_base = blockIdx.x * kBm;
    const std::uint32_t token_base = blockIdx.y * kBn;
    const std::uint32_t n_sub = columns / 32;
    const auto affine_geometry = MxAffineGeometry<Q5>(columns, rows);
    const auto q6_geometry = MxQ6Geometry(columns, rows);

    __shared__ uint4 weight_lo[64][4];
    __shared__ uint4 weight_hi[64][4];
    __shared__ uint2 weight_slot[4][64];
    __shared__ MxSharedInputRow input_q[128];
    __shared__ float input_d[128][5];
    __shared__ float input_sum[Q6 ? 1 : 128][5];
    float accum[kNrow][kTn] = {};
    uint4 prefetched_weight_lo[1], prefetched_weight_hi[1];
    uint2 prefetched_weight_slot[1];
    MxInputSlice prefetched_input[2];

    auto prefetch_weights_checked = [&](const std::uint32_t subblock_base) {
        const int local_row = tid / kBk;
        const int local_group = tid & (kBk - 1);
        const std::uint32_t row = row_base + local_row;
        const std::uint32_t subblock = subblock_base + local_group;
        MxWeightWords loaded{make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint2(0, 0)};
        if (row < rows && subblock < n_sub) {
            loaded = Q6
                ? mx_load_q6(weights, q6_geometry, row, subblock)
                : mx_load_affine<Q5>(weights, affine_geometry, row, subblock);
        }
        prefetched_weight_lo[0] = loaded.lo;
        prefetched_weight_hi[0] = loaded.hi;
        prefetched_weight_slot[0] = loaded.slot;
    };

    auto prefetch_weights_full = [&](const std::uint32_t subblock_base) {
        const int local_row = tid / kBk;
        const int local_group = tid & (kBk - 1);
        MxWeightWords loaded = Q6
            ? mx_load_q6(weights, q6_geometry, row_base + local_row,
                         subblock_base + local_group)
            : mx_load_affine<Q5>(weights, affine_geometry, row_base + local_row,
                                 subblock_base + local_group);
        prefetched_weight_lo[0] = loaded.lo;
        prefetched_weight_hi[0] = loaded.hi;
        prefetched_weight_slot[0] = loaded.slot;
    };

    auto prefetch_input_checked = [&](const std::uint32_t subblock_base) {
        const auto* group = input + static_cast<std::size_t>(subblock_base / kBk) * token_count;
        for (int i = 0; i < kInputElementsPerThread; ++i) {
            const int element = tid + i * kThreads;
            const int local_row = element / kBk;
            const int local_group = element & (kBk - 1);
            const std::uint32_t token = token_base + local_row;
            MxInputSlice loaded{make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), 0.0F, 0.0F};
            if (token < token_count) loaded = mx_load_input(
                group, token, local_group, !Q6);
            prefetched_input[i] = loaded;
        }
    };

    auto prefetch_input_full = [&](const std::uint32_t subblock_base) {
        const auto* group = input + static_cast<std::size_t>(subblock_base / kBk) * token_count;
        for (int i = 0; i < kInputElementsPerThread; ++i) {
            const int element = tid + i * kThreads;
            const int local_row = element / kBk;
            const int local_group = element & (kBk - 1);
            const auto loaded = mx_load_input(group, token_base + local_row,
                                              local_group, !Q6);
            prefetched_input[i] = loaded;
        }
    };

    auto store_stage = [&]() {
        const int local_row = tid / kBk;
        const int local_group = tid & (kBk - 1);
        weight_lo[local_row][local_group] = prefetched_weight_lo[0];
        weight_hi[local_row][local_group] = prefetched_weight_hi[0];
        weight_slot[local_group][local_row] = prefetched_weight_slot[0];
        for (int i = 0; i < kInputElementsPerThread; ++i) {
            const int element = tid + i * kThreads;
            const int local_input_row = element / kBk;
            const int local_input_group = element & (kBk - 1);
            const int shared_row = mx_input_swizzle(local_input_row);
            input_q[shared_row].q[local_input_group][0] = prefetched_input[i].q0;
            input_q[shared_row].q[local_input_group][1] = prefetched_input[i].q1;
            input_d[shared_row][local_input_group] = prefetched_input[i].d;
            if constexpr (!Q6) input_sum[shared_row][local_input_group] = prefetched_input[i].sum;
        }
    };

    auto compute_stage = [&]() {
        for (int group = 0; group < kBk; ++group) {
            const int local = ty * kNrow;
            for (int offset = 0; offset < kNrow; ++offset) {
                const auto lo = weight_lo[local + offset][group];
                const auto hi = weight_hi[local + offset][group];
                const auto slot = weight_slot[group][local + offset];
                for (int column = 0; column < kTn; ++column) {
                    const int shared_row = mx_input_swizzle(tx + column * 64);
                    const auto& x = input_q[shared_row];
                    const auto q0 = x.q[group][0];
                    const auto q1 = x.q[group][1];
                    const float dx = input_d[shared_row][group];
                    const int dot_lo = mx_dp4a(lo.x, q0.x)
                        + mx_dp4a(lo.y, q0.y) + mx_dp4a(lo.z, q0.z) + mx_dp4a(lo.w, q0.w);
                    const int dot_hi = mx_dp4a(hi.x, q1.x)
                        + mx_dp4a(hi.y, q1.y) + mx_dp4a(hi.z, q1.z) + mx_dp4a(hi.w, q1.w);
                    if constexpr (Q6) {
                        const std::uint16_t d_bits = static_cast<std::uint16_t>(slot.x >> 16);
                        const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits)) * 0.25F;
                        const int scale_lo = static_cast<std::int8_t>(slot.x & 0xFFU);
                        const int scale_hi = static_cast<std::int8_t>((slot.x >> 8) & 0xFFU);
                        accum[offset][column] += d * dx
                            * static_cast<float>(scale_lo * dot_lo + scale_hi * dot_hi);
                    } else {
                        const std::uint16_t d_bits = static_cast<std::uint16_t>(slot.y);
                        const std::uint16_t dmin_bits = static_cast<std::uint16_t>(slot.y >> 16);
                        const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
                        const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_bits));
                        const int scale = static_cast<int>(slot.x & 0xFFU);
                        const int minimum = static_cast<int>((slot.x >> 8) & 0xFFU);
                        accum[offset][column] += d * dx * static_cast<float>(scale * (dot_lo + dot_hi))
                            - dmin * static_cast<float>(minimum) * input_sum[shared_row][group];
                    }
                }
            }
        }
    };

    const bool full_tile = row_base + kBm <= rows && token_base + kBn <= token_count
        && (n_sub % kBk == 0);
    if (full_tile) {
        prefetch_weights_full(0);
        prefetch_input_full(0);
    } else {
        prefetch_weights_checked(0);
        prefetch_input_checked(0);
    }
    for (std::uint32_t subblock_base = 0; subblock_base < n_sub; subblock_base += kBk) {
        store_stage();
        __syncthreads();
        if (subblock_base + kBk < n_sub) {
            if (full_tile) {
                prefetch_weights_full(subblock_base + kBk);
                prefetch_input_full(subblock_base + kBk);
            } else {
                prefetch_weights_checked(subblock_base + kBk);
                prefetch_input_checked(subblock_base + kBk);
            }
        }
        compute_stage();
        __syncthreads();
    }

    for (int offset = 0; offset < kNrow; ++offset) {
        const std::uint32_t row = row_base + ty * kNrow + offset;
        if (row >= rows) continue;
        for (int column = 0; column < kTn; ++column) {
            const std::uint32_t token = token_base + tx + column * 64;
            if (token < token_count) output[static_cast<std::size_t>(token) * rows + row]
                = accum[offset][column];
        }
    }
}

// The register-pipeline transplant is retained behind MIINFER_MX_PIPELINE for
// isolated retests. The simpler staged kernel is the measured default.
template <bool Q5, bool Q6>
__launch_bounds__(256, 1) __global__ void mx_repacked_mmq_legacy_kernel(
    const std::uint8_t* __restrict__ weights,
    const miinfer::MxQ8_1MmqBlock* __restrict__ input,
    float* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count) {
    static_assert(!(Q5 && Q6));
    const std::uint32_t tx = threadIdx.x;
    const std::uint32_t ty = threadIdx.y;
    const std::uint32_t tid = ty * 64 + tx;
    const std::uint32_t row_base = blockIdx.x * 64;
    const std::uint32_t token_base = blockIdx.y * 128;
    const std::uint32_t n_sub = columns / 32;
    const auto affine_geometry = MxAffineGeometry<Q5>(columns, rows);
    const auto q6_geometry = MxQ6Geometry(columns, rows);

    __shared__ uint4 weight_lo[64][4];
    __shared__ uint4 weight_hi[64][4];
    __shared__ uint2 weight_slot[4][64];
    __shared__ miinfer::MxQ8_1MmqBlock input_tile[128];
    float accum[16][2] = {};

    for (std::uint32_t subblock_base = 0; subblock_base < n_sub; subblock_base += 4) {
        const std::uint32_t local_row = tid / 4;
        const std::uint32_t local_group = tid & 3;
        const std::uint32_t row = row_base + local_row;
        MxWeightWords loaded{make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint2(0, 0)};
        if (row < rows) {
            loaded = Q6
                ? mx_load_q6(weights, q6_geometry, row, subblock_base + local_group)
                : mx_load_affine<Q5>(weights, affine_geometry, row, subblock_base + local_group);
        }
        weight_lo[local_row][local_group] = loaded.lo;
        weight_hi[local_row][local_group] = loaded.hi;
        weight_slot[local_group][local_row] = loaded.slot;

        constexpr std::uint32_t input_words = sizeof(miinfer::MxQ8_1MmqBlock) / sizeof(uint4);
        for (std::uint32_t index = tid; index < 128 * input_words; index += 256) {
            const std::uint32_t token = index / input_words;
            const std::uint32_t word = index % input_words;
            auto* destination = reinterpret_cast<uint4*>(input_tile + mx_input_swizzle(token));
            if (token_base + token < token_count) {
                const auto* source = reinterpret_cast<const uint4*>(input
                    + static_cast<std::size_t>(subblock_base / 4) * token_count
                    + token_base + token);
                destination[word] = source[word];
            } else {
                destination[word] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        for (int group = 0; group < 4; ++group) {
            MxInputSlice x[2]{};
            for (int column = 0; column < 2; ++column) {
                const std::uint32_t token = token_base + tx + column * 64;
                if (token < token_count) {
                    x[column] = mx_load_input(input_tile, mx_input_swizzle(tx + column * 64),
                                              group, !Q6);
                }
            }
            const std::uint32_t local = ty * 16;
            for (int offset = 0; offset < 16; ++offset) {
                const auto lo = weight_lo[local + offset][group];
                const auto hi = weight_hi[local + offset][group];
                const auto slot = weight_slot[group][local + offset];
                for (int column = 0; column < 2; ++column) {
                    if constexpr (Q6) {
                        const std::uint16_t d_bits = static_cast<std::uint16_t>(slot.x >> 16);
                        const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits)) * 0.25F;
                        const int scale_lo = static_cast<std::int8_t>(slot.x & 0xFFU);
                        const int scale_hi = static_cast<std::int8_t>((slot.x >> 8) & 0xFFU);
                        const int dot_lo = mx_dp4a(lo.x, x[column].q0.x)
                            + mx_dp4a(lo.y, x[column].q0.y)
                            + mx_dp4a(lo.z, x[column].q0.z)
                            + mx_dp4a(lo.w, x[column].q0.w);
                        const int dot_hi = mx_dp4a(hi.x, x[column].q1.x)
                            + mx_dp4a(hi.y, x[column].q1.y)
                            + mx_dp4a(hi.z, x[column].q1.z)
                            + mx_dp4a(hi.w, x[column].q1.w);
                        accum[offset][column] += d * x[column].d
                            * static_cast<float>(scale_lo * dot_lo + scale_hi * dot_hi);
                    } else {
                        const std::uint16_t d_bits = static_cast<std::uint16_t>(slot.y);
                        const std::uint16_t dmin_bits = static_cast<std::uint16_t>(slot.y >> 16);
                        const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
                        const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_bits));
                        const int scale = static_cast<int>(slot.x & 0xFFU);
                        const int minimum = static_cast<int>((slot.x >> 8) & 0xFFU);
                        // Interleave independent halves as mx does; serializing
                        // all low then all high DP4A chains costs gfx906 issue slots.
                        int dot = 0;
                        dot = mx_dp4a(lo.x, x[column].q0.x, dot);
                        dot = mx_dp4a(hi.x, x[column].q1.x, dot);
                        dot = mx_dp4a(lo.y, x[column].q0.y, dot);
                        dot = mx_dp4a(hi.y, x[column].q1.y, dot);
                        dot = mx_dp4a(lo.z, x[column].q0.z, dot);
                        dot = mx_dp4a(hi.z, x[column].q1.z, dot);
                        dot = mx_dp4a(lo.w, x[column].q0.w, dot);
                        dot = mx_dp4a(hi.w, x[column].q1.w, dot);
                        accum[offset][column] += d * x[column].d * static_cast<float>(scale * dot)
                            - dmin * static_cast<float>(minimum) * x[column].sum;
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int offset = 0; offset < 16; ++offset) {
        const std::uint32_t row = row_base + ty * 16 + offset;
        if (row >= rows) continue;
        for (int column = 0; column < 2; ++column) {
            const std::uint32_t token = token_base + tx + column * 64;
            if (token < token_count) output[static_cast<std::size_t>(token) * rows + row]
                = accum[offset][column];
        }
    }
}

bool mx_pipeline_enabled() {
    const char* env = std::getenv("MIINFER_MX_PIPELINE");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

template <bool AFFINE>
__launch_bounds__(128, 1) __global__ void mx_q8_1_mmq_quantize_kernel(
    const float* input, miinfer::MxQ8_1MmqBlock* output, std::uint32_t elements) {
    const std::uint32_t token = blockIdx.y;
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    const std::size_t base = static_cast<std::size_t>(token) * elements + block * 128;
    __shared__ float maxima[128];
    __shared__ float sums[128];
    const float value = input[base + lane];
    maxima[lane] = fabsf(value);
    sums[lane] = value;
    __syncthreads();
    const std::uint32_t group = lane / 32;
    const std::uint32_t group_lane = lane & 31;
    for (std::uint32_t stride = 16; stride > 0; stride /= 2) {
        if (group_lane < stride) {
            maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
            sums[lane] += sums[lane + stride];
        }
        __syncthreads();
    }
    const float max_abs = maxima[group * 32];
    const float d = max_abs == 0.0F ? 0.0F : max_abs / 127.0F;
    const int quantized = d == 0.0F ? 0 : static_cast<int>(roundf(value / d));
    output[static_cast<std::size_t>(block) * gridDim.y + token].qs[lane]
        = static_cast<std::int8_t>(max(-127, min(127, quantized)));
    if (group_lane == 0) {
        auto& destination = output[static_cast<std::size_t>(block) * gridDim.y + token];
        if constexpr (AFFINE) {
            destination.ds4[group].x = __float2half_rn(d);
            destination.ds4[group].y = __float2half_rn(sums[group * 32]);
        } else {
            destination.d4[group] = d;
        }
    }
}

} // namespace

void miinfer::launch_mx_q8_1_mmq_quantize(
    const float* input, miinfer::MxQ8_1MmqBlock* output, std::uint32_t token_count,
    std::uint32_t elements, bool affine, hipStream_t stream) {
    if (input == nullptr || output == nullptr || token_count == 0 || token_count > 512
        || elements == 0 || elements % 128 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (affine) {
        hipLaunchKernelGGL((mx_q8_1_mmq_quantize_kernel<true>),
                           dim3(elements / 128, token_count), dim3(128), 0, stream,
                           input, output, elements);
    } else {
        hipLaunchKernelGGL((mx_q8_1_mmq_quantize_kernel<false>),
                           dim3(elements / 128, token_count), dim3(128), 0, stream,
                           input, output, elements);
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_mx_q4k_repacked_mmq(
    const std::uint8_t* weights, const miinfer::MxQ8_1MmqBlock* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0
        || columns == 0 || columns % 128 != 0 || token_count == 0 || token_count > 512) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (mx_pipeline_enabled()) {
        hipLaunchKernelGGL((mx_repacked_mmq_kernel<false, false>),
                           dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0,
                           stream, weights, input, output, rows, columns, token_count);
    } else {
        hipLaunchKernelGGL((mx_repacked_mmq_legacy_kernel<false, false>),
                           dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0,
                           stream, weights, input, output, rows, columns, token_count);
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_mx_q5k_repacked_mmq(
    const std::uint8_t* weights, const miinfer::MxQ8_1MmqBlock* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0
        || columns == 0 || columns % 128 != 0 || token_count == 0 || token_count > 512) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (mx_pipeline_enabled()) {
        hipLaunchKernelGGL((mx_repacked_mmq_kernel<true, false>),
                           dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0,
                           stream, weights, input, output, rows, columns, token_count);
    } else {
        hipLaunchKernelGGL((mx_repacked_mmq_legacy_kernel<true, false>),
                           dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0,
                           stream, weights, input, output, rows, columns, token_count);
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_mx_q6k_repacked_mmq(
    const std::uint8_t* weights, const miinfer::MxQ8_1MmqBlock* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0
        || columns == 0 || columns % 128 != 0 || token_count == 0 || token_count > 512) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (mx_pipeline_enabled()) {
        hipLaunchKernelGGL((mx_repacked_mmq_kernel<false, true>),
                           dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0,
                           stream, weights, input, output, rows, columns, token_count);
    } else {
        hipLaunchKernelGGL((mx_repacked_mmq_legacy_kernel<false, true>),
                           dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0,
                           stream, weights, input, output, rows, columns, token_count);
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

namespace {

template <bool Q5>
__launch_bounds__(256)
__global__ void m24_affine_mmq_to_fp16_kernel(
    const Q4KMmqTile* __restrict__ weights,
    __half* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * columns;
    if (index >= elements) return;

    const std::uint32_t row = static_cast<std::uint32_t>(index / columns);
    const std::uint32_t column = static_cast<std::uint32_t>(index % columns);
    const std::uint32_t k_tiles = columns / 128;
    const std::uint32_t slab = (column % 128) / 32;
    const std::uint32_t local = column % 32;
    const std::uint32_t half = local / 16;
    const std::uint32_t position = local % 16;
    const std::uint32_t word = position / 4;
    const std::uint32_t byte = position % 4;
    const auto& tile = weights[(row / 64) * k_tiles + column / 128];
    const std::uint32_t raw = tile.values[slab][row % 64][word];
    const std::uint32_t low = (raw >> (8 * byte + 4 * half)) & 0x0fU;
    const std::uint32_t high = tile.high[slab][row % 64][0];
    const std::uint32_t high_bit = Q5
        ? ((high >> (half * 16 + position)) & 1U) << 4 : 0U;
    const std::uint32_t scale_d = tile.scale_d[slab][row % 64];
    const std::uint16_t d_bits = static_cast<std::uint16_t>(scale_d >> 16);
    const std::uint16_t dmin_bits = tile.dmin[slab][row % 64];
    const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_bits));
    const float scale = static_cast<float>(scale_d & 0xffU);
    const float minimum = static_cast<float>((scale_d >> 8) & 0xffU);
    const float value = d * scale * static_cast<float>(low | high_bit)
        - dmin * minimum;
    output[index] = __float2half_rn(value);
}

__launch_bounds__(256)
__global__ void m24_q6k_mmq_to_fp16_kernel(
    const Q6KMmqTile* __restrict__ weights,
    __half* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * columns;
    if (index >= elements) return;

    const std::uint32_t row = static_cast<std::uint32_t>(index / columns);
    const std::uint32_t column = static_cast<std::uint32_t>(index % columns);
    const std::uint32_t k_tiles = columns / 128;
    const std::uint32_t slab = (column % 128) / 32;
    const std::uint32_t local = column % 32;
    const std::uint32_t half = local / 16;
    const std::uint32_t position = local % 16;
    const std::uint32_t word = position / 4;
    const std::uint32_t byte = position % 4;
    const auto& tile = weights[(row / 64) * k_tiles + column / 128];
    const std::uint32_t raw = tile.values[slab][row % 64][word];
    const std::uint32_t low = (raw >> (8 * byte + 4 * half)) & 0x0fU;
    const std::uint32_t bit_index = 2 * (half * 16 + position);
    const std::uint64_t high = static_cast<std::uint64_t>(tile.high[slab][row % 64][0])
        | (static_cast<std::uint64_t>(tile.high[slab][row % 64][1]) << 32);
    const std::uint32_t quantized = low | (((high >> bit_index) & 0x03U) << 4);
    const std::uint32_t scale_d = tile.scale_d[slab][row % 64];
    const std::uint16_t d_bits = static_cast<std::uint16_t>(scale_d >> 16);
    const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
    const int scale = static_cast<std::int8_t>(scale_d >> (8 * half));
    const int signed_quantized = static_cast<int>(quantized) - 32;
    output[index] = __float2half_rn(
        d * static_cast<float>(scale) * static_cast<float>(signed_quantized));
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

template <typename Decoder>
void launch_kquant_gemv_batched4_dispatch(
    const typename Decoder::Tile* w,
    const miinfer::Q8_1Block* x,
    float* y,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (!w || !x || !y) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    if (columns % 1024 != 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    const unsigned num_tiles = columns / 1024;
    const dim3 grid((rows + 3) / 4);
    const dim3 block(256);
    switch (num_tiles) {
    case 1:  hipLaunchKernelGGL((kquant_wave_gemv_batched_kernel<Decoder, 1>), grid, block, 0, stream, w, x, y, rows); break;
    case 5:  hipLaunchKernelGGL((kquant_wave_gemv_batched_kernel<Decoder, 5>), grid, block, 0, stream, w, x, y, rows); break;
    case 6:  hipLaunchKernelGGL((kquant_wave_gemv_batched_kernel<Decoder, 6>), grid, block, 0, stream, w, x, y, rows); break;
    case 12: hipLaunchKernelGGL((kquant_wave_gemv_batched_kernel<Decoder, 12>), grid, block, 0, stream, w, x, y, rows); break;
    case 17: hipLaunchKernelGGL((kquant_wave_gemv_batched_kernel<Decoder, 17>), grid, block, 0, stream, w, x, y, rows); break;
    default:
        throw std::runtime_error("unsupported batched NumTiles: " + std::to_string(num_tiles));
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4k_wave_gemv_batched4(const Q4KWaveTile* w, const miinfer::Q8_1Block* x,
                                   float* y, std::uint32_t rows, std::uint32_t columns,
                                   hipStream_t stream) {
    if (!w || !x || !y || columns % 1024 != 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    const dim3 grid((rows + 3) / 4);
    const dim3 block(256);
    switch (columns / 1024) {
    case 1: hipLaunchKernelGGL((q4k_wave_gemv_batched4_reuse_kernel<1>), grid, block, 0, stream, w, x, y, rows, columns / 32); break;
    case 5: hipLaunchKernelGGL((q4k_wave_gemv_batched4_reuse_kernel<5>), grid, block, 0, stream, w, x, y, rows, columns / 32); break;
    case 6: hipLaunchKernelGGL((q4k_wave_gemv_batched4_reuse_kernel<6>), grid, block, 0, stream, w, x, y, rows, columns / 32); break;
    case 12: hipLaunchKernelGGL((q4k_wave_gemv_batched4_reuse_kernel<12>), grid, block, 0, stream, w, x, y, rows, columns / 32); break;
    case 17: hipLaunchKernelGGL((q4k_wave_gemv_batched4_reuse_kernel<17>), grid, block, 0, stream, w, x, y, rows, columns / 32); break;
    default: throw std::runtime_error("unsupported batched Q4K NumTiles: " + std::to_string(columns / 1024));
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

// M13 lab-only dispatch. Keep it out of the runtime-facing header because the
// measured mapping was rejected against repeated B4 on the production shapes.
void launch_m13_q4k_wave_gemv_batched_mm(
    const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch,
    std::uint32_t input_stride_blocks, hipStream_t stream) {
    if (!w || !x || !y || columns % 1024 != 0 || batch == 0 || batch % 8 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (input_stride_blocks == 0) input_stride_blocks = columns / 32;
    const dim3 grid((rows + 3) / 4);
    const dim3 block(256);
    for (std::uint32_t offset = 0; offset < batch; offset += 8) {
        const auto* chunk_x = x + static_cast<std::size_t>(offset) * input_stride_blocks;
        auto* chunk_y = y + static_cast<std::size_t>(offset) * rows;
        switch (columns / 1024) {
        case 1: hipLaunchKernelGGL((q4k_wave_gemv_batched4_reuse_kernel<1, 8>), grid, block, 0, stream, w, chunk_x, chunk_y, rows, input_stride_blocks); break;
        case 5: hipLaunchKernelGGL((q4k_wave_gemv_batched4_reuse_kernel<5, 8>), grid, block, 0, stream, w, chunk_x, chunk_y, rows, input_stride_blocks); break;
        case 6: hipLaunchKernelGGL((q4k_wave_gemv_batched4_reuse_kernel<6, 8>), grid, block, 0, stream, w, chunk_x, chunk_y, rows, input_stride_blocks); break;
        case 12: hipLaunchKernelGGL((q4k_wave_gemv_batched4_reuse_kernel<12, 8>), grid, block, 0, stream, w, chunk_x, chunk_y, rows, input_stride_blocks); break;
        case 17: hipLaunchKernelGGL((q4k_wave_gemv_batched4_reuse_kernel<17, 8>), grid, block, 0, stream, w, chunk_x, chunk_y, rows, input_stride_blocks); break;
        default: throw std::runtime_error("unsupported quantized MM Q4K NumTiles: " + std::to_string(columns / 1024));
        }
    }
    MIINFER_HIP_CHECK(hipGetLastError());
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

void launch_q4k_wave_fused_gate_up_swiglu_paired_batched4(
    const Q4KWaveSwigluFusedTile* w_fused,
    const miinfer::Q8_1Block* x,
    float* y_activation,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (!w_fused || !x || !y_activation || columns % 1024 != 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    if (columns / 1024 != 5) {
        throw std::runtime_error("unsupported batched paired SwiGLU NumTiles: " + std::to_string(columns / 1024));
    }
    const dim3 grid((rows + 3) / 4);
    const dim3 block(256);
    hipLaunchKernelGGL((q4k_wave_fused_gate_up_swiglu_paired_batched4_kernel<5>),
                       grid, block, 0, stream, w_fused, x, y_activation, rows);
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

void launch_q5k_wave_gemv_batched4(const Q5KWaveTile* w, const miinfer::Q8_1Block* x,
                                   float* y, std::uint32_t rows, std::uint32_t columns,
                                   hipStream_t stream) {
    if (!w || !x || !y || columns % 1024 != 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    const dim3 grid((rows + 3) / 4);
    const dim3 block(256);
    switch (columns / 1024) {
    case 1: hipLaunchKernelGGL((q5k_wave_gemv_batched4_reuse_kernel<1>), grid, block, 0, stream, w, x, y, rows); break;
    case 5: hipLaunchKernelGGL((q5k_wave_gemv_batched4_reuse_kernel<5>), grid, block, 0, stream, w, x, y, rows); break;
    case 6: hipLaunchKernelGGL((q5k_wave_gemv_batched4_reuse_kernel<6>), grid, block, 0, stream, w, x, y, rows); break;
    case 12: hipLaunchKernelGGL((q5k_wave_gemv_batched4_reuse_kernel<12>), grid, block, 0, stream, w, x, y, rows); break;
    case 17: hipLaunchKernelGGL((q5k_wave_gemv_batched4_reuse_kernel<17>), grid, block, 0, stream, w, x, y, rows); break;
    default: throw std::runtime_error("unsupported batched Q5K NumTiles: " + std::to_string(columns / 1024));
    }
    MIINFER_HIP_CHECK(hipGetLastError());
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

void launch_q6k_wave_gemv_batched4(const Q6KWaveTile* w, const miinfer::Q8_1Block* x,
                                   float* y, std::uint32_t rows, std::uint32_t columns,
                                   hipStream_t stream) {
    if (!w || !x || !y || columns % 1024 != 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    const dim3 grid((rows + 3) / 4);
    const dim3 block(256);
    switch (columns / 1024) {
    case 1: hipLaunchKernelGGL((q6k_wave_gemv_batched4_reuse_kernel<1>), grid, block, 0, stream, w, x, y, rows, columns / 32); break;
    case 5: hipLaunchKernelGGL((q6k_wave_gemv_batched4_reuse_kernel<5>), grid, block, 0, stream, w, x, y, rows, columns / 32); break;
    case 6: hipLaunchKernelGGL((q6k_wave_gemv_batched4_reuse_kernel<6>), grid, block, 0, stream, w, x, y, rows, columns / 32); break;
    case 12: hipLaunchKernelGGL((q6k_wave_gemv_batched4_reuse_kernel<12>), grid, block, 0, stream, w, x, y, rows, columns / 32); break;
    case 17: hipLaunchKernelGGL((q6k_wave_gemv_batched4_reuse_kernel<17>), grid, block, 0, stream, w, x, y, rows, columns / 32); break;
    default: throw std::runtime_error("unsupported batched Q6K NumTiles: " + std::to_string(columns / 1024));
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m23_q6k_repacked_mmq(
    const Q6KMmqTile* weights, const miinfer::M23Q8_1MmqBlock* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count, hipStream_t stream) {
    if (!weights || !input || !output || rows == 0 || rows % 64 != 0 || columns == 0
        || columns % 128 != 0 || token_count == 0 || token_count > 512) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (m23_repacked_row128_enabled() && rows % 128 == 0) {
        hipLaunchKernelGGL(m23_q6k_repacked_mmq_row128_kernel,
                           dim3(rows / 128, (token_count + 63) / 64), dim3(64, 4), 0, stream,
                           weights, input, output, rows, columns, token_count);
    } else {
        hipLaunchKernelGGL(m23_q6k_repacked_mmq_kernel,
                           dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0, stream,
                           weights, input, output, rows, columns, token_count);
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m23_q4k_repacked_mmq(
    const Q4KMmqTile* weights, const miinfer::M23Q8_1MmqBlock* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count, hipStream_t stream) {
    if (!weights || !input || !output || rows == 0 || rows % 64 != 0 || columns == 0
        || columns % 128 != 0 || token_count == 0 || token_count > 512) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (m23_repacked_row128_enabled() && rows % 128 == 0) {
        hipLaunchKernelGGL((m23_affine_repacked_mmq_row128_kernel<Q4KMmqTile>),
                           dim3(rows / 128, (token_count + 63) / 64), dim3(64, 4), 0, stream,
                           weights, input, output, rows, columns, token_count);
    } else {
        hipLaunchKernelGGL((m23_affine_repacked_mmq_kernel<Q4KMmqTile>),
                           dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0, stream,
                           weights, input, output, rows, columns, token_count);
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m23_q5k_repacked_mmq(
    const Q5KMmqTile* weights, const miinfer::M23Q8_1MmqBlock* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count, hipStream_t stream) {
    if (!weights || !input || !output || rows == 0 || rows % 64 != 0 || columns == 0
        || columns % 128 != 0 || token_count == 0 || token_count > 512) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (m23_repacked_row128_enabled() && rows % 128 == 0) {
        hipLaunchKernelGGL((m23_affine_repacked_mmq_row128_kernel<Q5KMmqTile>),
                           dim3(rows / 128, (token_count + 63) / 64), dim3(64, 4), 0, stream,
                           weights, input, output, rows, columns, token_count);
    } else {
        hipLaunchKernelGGL((m23_affine_repacked_mmq_kernel<Q5KMmqTile>),
                           dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0, stream,
                           weights, input, output, rows, columns, token_count);
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

template <bool Q5>
void launch_m24_affine_mmq_to_fp16(
    const Q4KMmqTile* weights, __half* output, std::uint32_t rows,
    std::uint32_t columns, hipStream_t stream) {
    if (weights == nullptr || output == nullptr || rows == 0 || rows % 64 != 0
        || columns == 0 || columns % 128 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const auto elements = static_cast<std::uint64_t>(rows) * columns;
    const auto blocks = static_cast<unsigned>((elements + 255) / 256);
    hipLaunchKernelGGL((m24_affine_mmq_to_fp16_kernel<Q5>), dim3(blocks), dim3(256), 0,
                       stream, weights, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m24_q4k_mmq_to_fp16(
    const Q4KMmqTile* weights, __half* output, std::uint32_t rows,
    std::uint32_t columns, hipStream_t stream) {
    launch_m24_affine_mmq_to_fp16<false>(weights, output, rows, columns, stream);
}

void launch_m24_q5k_mmq_to_fp16(
    const Q5KMmqTile* weights, __half* output, std::uint32_t rows,
    std::uint32_t columns, hipStream_t stream) {
    launch_m24_affine_mmq_to_fp16<true>(weights, output, rows, columns, stream);
}

void launch_m24_q6k_mmq_to_fp16(
    const Q6KMmqTile* weights, __half* output, std::uint32_t rows,
    std::uint32_t columns, hipStream_t stream) {
    if (weights == nullptr || output == nullptr || rows == 0 || rows % 64 != 0
        || columns == 0 || columns % 128 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const auto elements = static_cast<std::uint64_t>(rows) * columns;
    const auto blocks = static_cast<unsigned>((elements + 255) / 256);
    hipLaunchKernelGGL(m24_q6k_mmq_to_fp16_kernel, dim3(blocks), dim3(256), 0,
                       stream, weights, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m13_q6k_wave_gemv_batched_mm(
    const Q6KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch,
    std::uint32_t input_stride_blocks, hipStream_t stream) {
    if (!w || !x || !y || columns % 1024 != 0 || batch == 0 || batch % 8 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (input_stride_blocks == 0) input_stride_blocks = columns / 32;
    const dim3 grid((rows + 3) / 4);
    const dim3 block(256);
    for (std::uint32_t offset = 0; offset < batch; offset += 8) {
        const auto* chunk_x = x + static_cast<std::size_t>(offset) * input_stride_blocks;
        auto* chunk_y = y + static_cast<std::size_t>(offset) * rows;
        switch (columns / 1024) {
        case 1: hipLaunchKernelGGL((q6k_wave_gemv_batched4_reuse_kernel<1, 8>), grid, block, 0, stream, w, chunk_x, chunk_y, rows, input_stride_blocks); break;
        case 5: hipLaunchKernelGGL((q6k_wave_gemv_batched4_reuse_kernel<5, 8>), grid, block, 0, stream, w, chunk_x, chunk_y, rows, input_stride_blocks); break;
        case 6: hipLaunchKernelGGL((q6k_wave_gemv_batched4_reuse_kernel<6, 8>), grid, block, 0, stream, w, chunk_x, chunk_y, rows, input_stride_blocks); break;
        case 12: hipLaunchKernelGGL((q6k_wave_gemv_batched4_reuse_kernel<12, 8>), grid, block, 0, stream, w, chunk_x, chunk_y, rows, input_stride_blocks); break;
        case 17: hipLaunchKernelGGL((q6k_wave_gemv_batched4_reuse_kernel<17, 8>), grid, block, 0, stream, w, chunk_x, chunk_y, rows, input_stride_blocks); break;
        default: throw std::runtime_error("unsupported quantized MM Q6K NumTiles: " + std::to_string(columns / 1024));
        }
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}
