#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace miinfer {

// Host-side views of the canonical GGUF blocks.  The byte layout is kept
// explicit so the correctness oracle does not depend on HIP's __half type.
struct Q4_0HostBlock {
    std::uint16_t d_bits = 0;
    std::uint8_t qs[16]{};
};

struct Q6KHostBlock {
    std::uint8_t ql[128]{};
    std::uint8_t qh[64]{};
    std::int8_t scales[16]{};
    std::uint16_t d_bits = 0;
};

struct Q8KHostBlock {
    float d = 0.0F;
    std::int8_t qs[256]{};
    std::int16_t bsums[16]{};
};

static_assert(sizeof(Q4_0HostBlock) == 18);
static_assert(sizeof(Q6KHostBlock) == 210);
static_assert(sizeof(Q8KHostBlock) == 292);

[[nodiscard]] float fp16_bits_to_float(std::uint16_t bits) noexcept;

void rms_norm_reference(
    std::span<const float> input,
    std::span<const float> weights,
    std::span<float> output,
    float epsilon);

// Qwen3 uses NeoX pairing: each pair is input[pair] with input[head_dim/2 + pair].
void rope_qwen3_reference(
    std::span<const float> input,
    std::span<float> output,
    std::size_t head_count,
    std::size_t head_dim,
    std::size_t position,
    float theta);

void softmax_reference(std::span<const float> input, std::span<float> output);

[[nodiscard]] float silu_reference(float value) noexcept;
void silu_mul_reference(
    std::span<const float> gate,
    std::span<const float> up,
    std::span<float> output);

void q4_0_dequantize(
    const Q4_0HostBlock& block,
    std::span<float, 32> output);

void q6_k_dequantize(
    const Q6KHostBlock& block,
    std::span<float, 256> output);

void q6_k_gemv_reference(
    std::span<const std::byte> weights,
    std::span<const float> input,
    std::span<float> output,
    std::size_t rows,
    std::size_t columns);

void q6_k_q8_k_gemv_reference(
    std::span<const std::byte> weights,
    std::span<const float> input,
    std::span<float> output,
    std::size_t rows,
    std::size_t columns);

void q4_0_embedding_reference(
    std::span<const std::byte> weights,
    std::size_t vocabulary,
    std::size_t hidden_size,
    std::size_t token,
    std::span<float> output);

}  // namespace miinfer
