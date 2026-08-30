#include "miinfer/qwen3_primitives.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace miinfer {

namespace {

float load_half(std::uint16_t bits) noexcept {
    return fp16_bits_to_float(bits);
}

void require_size(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

std::int8_t signed_q6(const Q6KHostBlock& block, std::size_t index) noexcept {
    const std::size_t group = index / 128;
    const std::size_t lane = index % 128;
    const std::size_t quarter = lane / 32;
    const std::size_t in_quarter = lane % 32;
    const std::size_t low_index = in_quarter + (quarter >= 2 ? 32 : 0);
    const std::size_t high_index = in_quarter;
    const std::uint8_t low = quarter % 2 == 0
                                 ? block.ql[low_index] & 0x0fU
                                 : block.ql[low_index] >> 4U;
    const std::uint8_t high = static_cast<std::uint8_t>(
        (block.qh[high_index] >> (2U * static_cast<unsigned>(quarter))) & 0x03U);
    (void) group;
    return static_cast<std::int8_t>(static_cast<int>(low | (high << 4U)) - 32);
}

}  // namespace

float fp16_bits_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000U)) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    const std::uint32_t fraction = bits & 0x03ffU;
    std::uint32_t value = sign;
    if (exponent == 0) {
        if (fraction != 0) {
            std::uint32_t normalized = fraction;
            std::uint32_t exp = 0;
            while ((normalized & 0x0400U) == 0) {
                normalized <<= 1U;
                ++exp;
            }
            normalized &= 0x03ffU;
            value |= (127U - 15U - exp + 1U) << 23U;
            value |= normalized << 13U;
        }
    } else if (exponent == 0x1fU) {
        value |= 0x7f800000U | (fraction << 13U);
    } else {
        value |= (exponent + (127U - 15U)) << 23U;
        value |= fraction << 13U;
    }
    return std::bit_cast<float>(value);
}

void rms_norm_reference(
    std::span<const float> input,
    std::span<const float> weights,
    std::span<float> output,
    float epsilon) {
    require_size(!input.empty() && input.size() == weights.size() && input.size() == output.size()
                     && std::isfinite(epsilon) && epsilon > 0.0F,
                 "invalid RMSNorm dimensions or epsilon");
    double sum = 0.0;
    for (const float value : input) sum += static_cast<double>(value) * value;
    const float inverse_rms = 1.0F / std::sqrt(static_cast<float>(sum / input.size()) + epsilon);
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = input[index] * inverse_rms * weights[index];
    }
}

void rope_qwen3_reference(
    std::span<const float> input,
    std::span<float> output,
    std::size_t head_count,
    std::size_t head_dim,
    std::size_t position,
    float theta) {
    require_size(head_count > 0 && head_dim > 0 && head_dim % 2 == 0
                     && input.size() == head_count * head_dim
                     && output.size() == input.size() && theta > 0.0F,
                 "invalid RoPE dimensions");
    for (std::size_t head = 0; head < head_count; ++head) {
        for (std::size_t pair = 0; pair < head_dim / 2; ++pair) {
            const float frequency = std::pow(theta, -static_cast<float>(2 * pair)
                                                       / static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const std::size_t offset = head * head_dim + 2 * pair;
            const float x0 = input[offset];
            const float x1 = input[offset + 1];
            output[offset] = x0 * cosine - x1 * sine;
            output[offset + 1] = x0 * sine + x1 * cosine;
        }
    }
}

void softmax_reference(std::span<const float> input, std::span<float> output) {
    require_size(!input.empty() && input.size() == output.size(), "invalid softmax dimensions");
    const float maximum = *std::max_element(input.begin(), input.end());
    double denominator = 0.0;
    for (const float value : input) denominator += std::exp(static_cast<double>(value - maximum));
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = static_cast<float>(
            std::exp(static_cast<double>(input[index] - maximum)) / denominator);
    }
}

float silu_reference(float value) noexcept {
    return value / (1.0F + std::exp(-value));
}

void silu_mul_reference(
    std::span<const float> gate,
    std::span<const float> up,
    std::span<float> output) {
    require_size(!gate.empty() && gate.size() == up.size() && gate.size() == output.size(),
                 "invalid SwiGLU dimensions");
    for (std::size_t index = 0; index < gate.size(); ++index) {
        output[index] = silu_reference(gate[index]) * up[index];
    }
}

void q4_0_dequantize(const Q4_0HostBlock& block, std::span<float, 32> output) {
    const float scale = load_half(block.d_bits);
    for (std::size_t index = 0; index < 32; ++index) {
        const std::uint8_t packed = block.qs[index < 16 ? index : index - 16];
        const int nibble = index < 16 ? packed & 0x0fU : packed >> 4U;
        output[index] = scale * static_cast<float>(nibble - 8);
    }
}

void q6_k_dequantize(const Q6KHostBlock& block, std::span<float, 256> output) {
    const float scale = load_half(block.d_bits);
    for (std::size_t index = 0; index < 256; ++index) {
        const std::size_t scale_index = index / 16;
        output[index] = scale * static_cast<float>(block.scales[scale_index])
                        * static_cast<float>(signed_q6(block, index));
    }
}

void q6_k_gemv_reference(
    std::span<const std::byte> weights,
    std::span<const float> input,
    std::span<float> output,
    std::size_t rows,
    std::size_t columns) {
    require_size(rows > 0 && columns > 0 && columns % 256 == 0
                     && input.size() == columns && output.size() == rows
                     && weights.size() == rows * (columns / 256) * sizeof(Q6KHostBlock),
                 "invalid Q6_K GEMV dimensions");
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        const auto* blocks = reinterpret_cast<const Q6KHostBlock*>(weights.data())
                             + row * (columns / 256);
        std::array<float, 256> values{};
        for (std::size_t block = 0; block < columns / 256; ++block) {
            q6_k_dequantize(blocks[block], values);
            for (std::size_t index = 0; index < values.size(); ++index) {
                sum += values[index] * input[block * values.size() + index];
            }
        }
        output[row] = sum;
    }
}

void q4_0_embedding_reference(
    std::span<const std::byte> weights,
    std::size_t vocabulary,
    std::size_t hidden_size,
    std::size_t token,
    std::span<float> output) {
    require_size(vocabulary > 0 && hidden_size > 0 && hidden_size % 32 == 0
                     && token < vocabulary && output.size() == hidden_size
                     && weights.size() == vocabulary * (hidden_size / 32) * sizeof(Q4_0HostBlock),
                 "invalid Q4_0 embedding dimensions");
    const auto* blocks = reinterpret_cast<const Q4_0HostBlock*>(weights.data())
                         + token * (hidden_size / 32);
    for (std::size_t block = 0; block < hidden_size / 32; ++block) {
        std::span<float, 32> destination(output.data() + block * 32, 32);
        q4_0_dequantize(blocks[block], destination);
    }
}

}  // namespace miinfer
