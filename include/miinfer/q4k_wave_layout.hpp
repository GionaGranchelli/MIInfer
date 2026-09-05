#pragma once
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/gguf.hpp"
#include <vector>

// Exact 5120 x 17408 Down layout: four K blocks, two Wave64 word planes.
struct alignas(128) Q4KWaveTile {
    std::uint32_t words[2][64];
    struct Metadata {
        __half d, dmin;
        std::uint8_t scales[8], minimums[8];
    } metadata[4];
};
static_assert(sizeof(Q4KWaveTile) == 640);
void launch_q4k_wave_down(const Q4KWaveTile*, const miinfer::Q8_1Block*, float*);
std::vector<Q4KWaveTile> pack_q4k_wave_down(const miinfer::GgufTensor&);
