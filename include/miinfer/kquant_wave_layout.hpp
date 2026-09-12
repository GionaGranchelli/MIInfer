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

// Gfx906-native paired SwiGLU layout: paired Gate and Up tiles with dwordx2 coalescing.
struct alignas(128) Q4KWaveSwigluFusedTile {
    std::uint64_t words_p0[64];             // 512 bytes: paired {gate_word, up_word} for plane 0
    std::uint64_t words_p1[64];             // 512 bytes: paired {gate_word, up_word} for plane 1
    Q4KWaveTile::Metadata metadata_gate[4]; // 80 bytes
    Q4KWaveTile::Metadata metadata_up[4];   // 80 bytes
    std::uint8_t padding[96];               // 96 bytes -> total = 1280 bytes
};
static_assert(sizeof(Q4KWaveSwigluFusedTile) == 1280);

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

// M23 resident Q6_K MMQ layout. One tile is 64 output rows by four 32-value
// K slabs (128 K values). The global tile keeps packed low/high planes; the
// reader expands them once into LDS, matching the pinned mx strategy.
struct alignas(128) Q6KMmqTile {
    std::uint32_t values[4][64][4];    // packed ql bytes, 16 bytes per 32-value slab
    std::uint32_t high[4][64][2];      // two-bit high values, 8 bytes per slab
    std::uint32_t scale_d[4][64];      // scale_lo | scale_hi<<8 | d_bits<<16
};
static_assert(sizeof(Q6KMmqTile) == 7168);

// M23 resident affine K-quant tiles. Packed low nibbles remain resident and
// are expanded to byte lanes in LDS; scale_d stores scale/minimum plus d.
struct alignas(128) Q4KMmqTile {
    std::uint32_t values[4][64][4];
    std::uint32_t high[4][64][1];
    std::uint32_t scale_d[4][64];
    std::uint16_t dmin[4][64];
};
static_assert(sizeof(Q4KMmqTile) == 6656);

using Q5KMmqTile = Q4KMmqTile;

// Host repacking functions from canonical GGUF tensors
std::vector<Q4KWaveTile> pack_q4k_wave_tensor(const miinfer::GgufTensor& tensor);
std::vector<Q4KWaveTile> pack_q4k_wave_down(const miinfer::GgufTensor& tensor);
std::vector<Q5KWaveTile> pack_q5k_wave_tensor(const miinfer::GgufTensor& tensor);
std::vector<Q6KWaveTile> pack_q6k_wave_tensor(const miinfer::GgufTensor& tensor);
std::vector<Q6KMmqTile> pack_q6k_mmq_tensor(const miinfer::GgufTensor& tensor);
std::vector<Q4KMmqTile> pack_q4k_mmq_tensor(const miinfer::GgufTensor& tensor);
std::vector<Q5KMmqTile> pack_q5k_mmq_tensor(const miinfer::GgufTensor& tensor);
std::vector<std::uint8_t> pack_mx_q4k_repacked_tensor(const miinfer::GgufTensor& tensor);
std::vector<std::uint8_t> pack_mx_q5k_repacked_tensor(const miinfer::GgufTensor& tensor);
std::vector<std::uint8_t> pack_mx_q6k_repacked_tensor(const miinfer::GgufTensor& tensor);

void launch_mx_q4k_repacked_mmq(
    const std::uint8_t* weights,
    const miinfer::MxQ8_1MmqBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count,
    hipStream_t stream = nullptr);
void launch_mx_q5k_repacked_mmq(
    const std::uint8_t* weights,
    const miinfer::MxQ8_1MmqBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count,
    hipStream_t stream = nullptr);
void launch_mx_q6k_repacked_mmq(
    const std::uint8_t* weights,
    const miinfer::MxQ8_1MmqBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count,
    hipStream_t stream = nullptr);

void launch_mx_q6k_repacked_mmq_pinned(
    const std::uint8_t* weights,
    const miinfer::MxQ8_1MmqBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count,
    hipStream_t stream = nullptr);

void launch_m23_q6k_repacked_mmq(
    const Q6KMmqTile* weights,
    const miinfer::M23Q8_1MmqBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count,
    hipStream_t stream = nullptr);
void launch_m23_q4k_repacked_mmq(
    const Q4KMmqTile* weights,
    const miinfer::M23Q8_1MmqBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count,
    hipStream_t stream = nullptr);
void launch_m23_q5k_repacked_mmq(
    const Q5KMmqTile* weights,
    const miinfer::M23Q8_1MmqBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t token_count,
    hipStream_t stream = nullptr);

// M24 prefill staging: expand resident MMQ tiles directly into row-major FP16
// scratch without retaining a second canonical quantized device copy.
void launch_m24_q4k_mmq_to_fp16(
    const Q4KMmqTile* weights,
    __half* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);
void launch_m24_q5k_mmq_to_fp16(
    const Q5KMmqTile* weights,
    __half* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);
void launch_m24_q6k_mmq_to_fp16(
    const Q6KMmqTile* weights,
    __half* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);
std::vector<Q4KWaveSwigluFusedTile> pack_q4k_wave_swiglu_fused(
    const miinfer::GgufTensor& gate,
    const miinfer::GgufTensor& up);

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
                          std::uint32_t rows, std::uint32_t columns,
                          hipStream_t stream = nullptr);
void launch_q4k_wave_gemv_batched4(const Q4KWaveTile* w, const miinfer::Q8_1Block* x,
                                   float* y, std::uint32_t rows, std::uint32_t columns,
                                   hipStream_t stream = nullptr);
void launch_q4k_wave_fused_gate_up_swiglu(
    const Q4KWaveTile* w_gate,
    const Q4KWaveTile* w_up,
    const miinfer::Q8_1Block* x,
    float* y_activation,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);
void launch_q4k_wave_fused_gate_up_swiglu_paired(
    const Q4KWaveSwigluFusedTile* w_fused,
    const miinfer::Q8_1Block* x,
    float* y_activation,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);
void launch_q4k_wave_fused_gate_up_swiglu_paired_batched4(
    const Q4KWaveSwigluFusedTile* w_fused,
    const miinfer::Q8_1Block* x,
    float* y_activation,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream = nullptr);
void launch_q4k_wave_down(const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          hipStream_t stream = nullptr);
void launch_q5k_wave_gemv(const Q5KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          std::uint32_t rows, std::uint32_t columns,
                          hipStream_t stream = nullptr);
void launch_q5k_wave_gemv_batched4(const Q5KWaveTile* w, const miinfer::Q8_1Block* x,
                                   float* y, std::uint32_t rows, std::uint32_t columns,
                                   hipStream_t stream = nullptr);
void launch_q6k_wave_gemv(const Q6KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                          std::uint32_t rows, std::uint32_t columns,
                          hipStream_t stream = nullptr);
void launch_q6k_wave_gemv_batched4(const Q6KWaveTile* w, const miinfer::Q8_1Block* x,
                                   float* y, std::uint32_t rows, std::uint32_t columns,
                                   hipStream_t stream = nullptr);
