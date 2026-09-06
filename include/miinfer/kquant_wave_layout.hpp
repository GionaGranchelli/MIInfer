#pragma once
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/gguf.hpp"
#include <vector>
#include <cstdint>

// Gfx906-native Q4_K layout: four K blocks per tile (1024 weights), two Wave64 word planes.
struct alignas(128) Q4KWaveTile {
    std::uint32_t words[2][64];
    struct Metadata {
        __half d, dmin;
        std::uint8_t scales[8], minimums[8];
    } metadata[4];
};
static_assert(sizeof(Q4KWaveTile) == 640);

// Gfx906-native Q5_K layout: four K blocks per tile (1024 weights), two low planes + high-bit plane.
struct alignas(128) Q5KWaveTile {
    std::uint32_t words[2][64];  // 512 bytes: low 4-bit planes (plane 0: v0, plane 1: v1)
    std::uint32_t qh[64];         // 256 bytes: high 5th-bit plane (one 32-bit word per lane)
    struct Metadata {
        __half d, dmin;
        std::uint8_t scales[8], minimums[8];
    } metadata[4];                // 80 bytes: 4 blocks * 20 bytes
    std::uint8_t padding[48];     // pads to 896 bytes (aligned to 128 bytes)
};
static_assert(sizeof(Q5KWaveTile) == 896);

// Gfx906-native Q6_K layout: four K blocks per tile (1024 weights), two low planes + high 2-bit plane.
struct alignas(128) Q6KWaveTile {
    std::uint32_t words[2][64];  // 512 bytes: low 4-bit planes (plane 0: v0, plane 1: v1)
    std::uint32_t qh[64];         // 256 bytes: high 2-bit plane (one 32-bit word per lane)
    struct Metadata {
        __half d;
        std::int8_t scales[16];
        std::uint8_t pad[2];      // 20 bytes per block
    } metadata[4];                // 80 bytes: 4 blocks * 20 bytes
    std::uint8_t padding[48];     // pads to 896 bytes (aligned to 128 bytes)
};
static_assert(sizeof(Q6KWaveTile) == 896);

// Host repacking functions from canonical GGUF tensors
std::vector<Q4KWaveTile> pack_q4k_wave_tensor(const miinfer::GgufTensor& tensor);
std::vector<Q4KWaveTile> pack_q4k_wave_down(const miinfer::GgufTensor& tensor);
std::vector<Q5KWaveTile> pack_q5k_wave_tensor(const miinfer::GgufTensor& tensor);
std::vector<Q6KWaveTile> pack_q6k_wave_tensor(const miinfer::GgufTensor& tensor);

// Host reference dequantizers (each outputs 1024 floats)
void q4k_wave_tile_dequantize(const Q4KWaveTile& tile, float* output);
void q5k_wave_tile_dequantize(const Q5KWaveTile& tile, float* output);
void q6k_wave_tile_dequantize(const Q6KWaveTile& tile, float* output);

// Host reference GEMV implementations
void q4k_wave_gemv_reference(const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                             std::uint32_t rows, std::uint32_t columns);
void q5k_wave_gemv_reference(const Q5KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                             std::uint32_t rows, std::uint32_t columns);
void q6k_wave_gemv_reference(const Q6KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                             std::uint32_t rows, std::uint32_t columns);
void q4k_wave_fused_gate_up_swiglu_reference(
    const Q4KWaveTile* w_gate,
    const Q4KWaveTile* w_up,
    const miinfer::Q8_1Block* x,
    float* y_activation,
    std::uint32_t rows,
    std::uint32_t columns);

// Device kernel dispatch functions
void launch_q4k_wave_gemv(const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          std::uint32_t rows, std::uint32_t columns);
void launch_q4k_wave_fused_gate_up_swiglu(
    const Q4KWaveTile* w_gate,
    const Q4KWaveTile* w_up,
    const miinfer::Q8_1Block* x,
    float* y_activation,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);
void launch_q4k_wave_down(const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y);
void launch_q5k_wave_gemv(const Q5KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          std::uint32_t rows, std::uint32_t columns);
void launch_q6k_wave_gemv(const Q6KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          std::uint32_t rows, std::uint32_t columns);
