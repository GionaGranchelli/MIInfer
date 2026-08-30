#include "miinfer/q4_q8_gemv.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace miinfer {

namespace {

void validate_block_shape(std::size_t elements, int rows, int columns) {
    if (rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0
        || elements != static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns)) {
        throw std::invalid_argument("Q4_0 GEMV shape must have positive rows and a K multiple of 32");
    }
}

}  // namespace

std::vector<Q4_0Block> quantize_q4_0(
    const std::vector<__half>& source,
    int rows,
    int columns) {
    validate_block_shape(source.size(), rows, columns);
    const int blocks_per_row = columns / kQ4_0BlockSize;
    std::vector<Q4_0Block> result(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(blocks_per_row));
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks_per_row; ++block) {
            const std::size_t source_offset =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(columns)
                + static_cast<std::size_t>(block) * kQ4_0BlockSize;
            auto& destination = result[static_cast<std::size_t>(row) * blocks_per_row + block];
            float amax = 0.0F;
            float max_value = 0.0F;
            for (int index = 0; index < kQ4_0BlockSize; ++index) {
                const float value = __half2float(source[source_offset + static_cast<std::size_t>(index)]);
                if (amax < std::fabs(value)) {
                    amax = std::fabs(value);
                    max_value = value;
                }
            }
            const float scale = max_value / -8.0F;
            const float inverse_scale = scale != 0.0F ? 1.0F / scale : 0.0F;
            destination.d = __float2half_rn(scale);
            for (int index = 0; index < kQ4_0BlockSize / 2; ++index) {
                const float first = __half2float(source[source_offset + static_cast<std::size_t>(index)])
                                    * inverse_scale;
                const float second = __half2float(
                                         source[source_offset + kQ4_0BlockSize / 2
                                                + static_cast<std::size_t>(index)])
                                     * inverse_scale;
                const auto q_first = static_cast<std::uint8_t>(std::clamp(
                    static_cast<int>(static_cast<std::int8_t>(first + 8.5F)), 0, 15));
                const auto q_second = static_cast<std::uint8_t>(std::clamp(
                    static_cast<int>(static_cast<std::int8_t>(second + 8.5F)), 0, 15));
                destination.qs[index] = static_cast<std::uint8_t>(q_first | (q_second << 4));
            }
        }
    }
    return result;
}

std::vector<Q8_1Block> quantize_q8_1(const std::vector<__half>& source) {
    if (source.empty() || source.size() % kQ8_1BlockSize != 0) {
        throw std::invalid_argument("Q8_1 input length must be a non-zero multiple of 32");
    }
    const std::size_t block_count = source.size() / kQ8_1BlockSize;
    std::vector<Q8_1Block> result(block_count);
    for (std::size_t block = 0; block < block_count; ++block) {
        const std::size_t source_offset = block * kQ8_1BlockSize;
        float amax = 0.0F;
        for (int index = 0; index < kQ8_1BlockSize; ++index) {
            amax = std::max(amax, std::fabs(__half2float(
                source[source_offset + static_cast<std::size_t>(index)])));
        }
        const float scale = amax / 127.0F;
        const float inverse_scale = scale != 0.0F ? 1.0F / scale : 0.0F;
        auto& destination = result[block];
        destination.d = __float2half_rn(scale);
        int sum = 0;
        for (int index = 0; index < kQ8_1BlockSize; ++index) {
            const int quantized = std::clamp(static_cast<int>(std::round(
                __half2float(source[source_offset + static_cast<std::size_t>(index)])
                * inverse_scale)), -127, 127);
            destination.qs[index] = static_cast<std::int8_t>(quantized);
            sum += quantized;
        }
        destination.s = __float2half_rn(static_cast<float>(sum) * scale);
    }
    return result;
}

std::vector<Q8ExactBlock> quantize_q8_exact(const std::vector<__half>& source) {
    if (source.empty() || source.size() % kQ8_1BlockSize != 0) {
        throw std::invalid_argument("exact Q8 input length must be a non-zero multiple of 32");
    }
    const std::size_t block_count = source.size() / kQ8_1BlockSize;
    std::vector<Q8ExactBlock> result(block_count);
    for (std::size_t block = 0; block < block_count; ++block) {
        const std::size_t source_offset = block * kQ8_1BlockSize;
        float amax = 0.0F;
        for (int index = 0; index < kQ8_1BlockSize; ++index) {
            amax = std::max(amax, std::fabs(__half2float(
                source[source_offset + static_cast<std::size_t>(index)])));
        }
        const float scale = amax / 127.0F;
        const float inverse_scale = scale != 0.0F ? 1.0F / scale : 0.0F;
        auto& destination = result[block];
        destination.d = __float2half_rn(scale);
        int sum = 0;
        for (int index = 0; index < kQ8_1BlockSize; ++index) {
            const int quantized = std::clamp(static_cast<int>(std::round(
                __half2float(source[source_offset + static_cast<std::size_t>(index)])
                * inverse_scale)), -127, 127);
            destination.qs[index] = static_cast<std::int8_t>(quantized);
            sum += quantized;
        }
        destination.sum = static_cast<std::int16_t>(sum);
    }
    return result;
}

std::vector<float> q4_q8_cpu_reference(
    const std::vector<Q4_0Block>& weights,
    const std::vector<Q8_1Block>& input,
    int rows,
    int columns) {
    if (rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0
        || weights.size() != static_cast<std::size_t>(rows) * columns / kQ4_0BlockSize
        || input.size() != static_cast<std::size_t>(columns) / kQ8_1BlockSize) {
        throw std::invalid_argument("invalid Q4_0/Q8_1 GEMV block dimensions");
    }
    const int blocks_per_row = columns / kQ4_0BlockSize;
    std::vector<float> result(static_cast<std::size_t>(rows), 0.0F);
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0F;
        for (int block = 0; block < blocks_per_row; ++block) {
            const auto& weight = weights[static_cast<std::size_t>(row) * blocks_per_row + block];
            const auto& activation = input[static_cast<std::size_t>(block)];
            const float weight_scale = __half2float(weight.d);
            const float activation_scale = __half2float(activation.d);
            for (int index = 0; index < kQ4_0BlockSize; ++index) {
                const std::uint8_t packed = weight.qs[index < kQ4_0BlockSize / 2
                                                           ? index
                                                           : index - kQ4_0BlockSize / 2];
                const int q4 = index < kQ4_0BlockSize / 2
                                   ? (packed & 0x0F)
                                   : ((packed >> 4) & 0x0F);
                sum += static_cast<float>(q4 - 8)
                       * weight_scale
                       * static_cast<float>(activation.qs[index])
                       * activation_scale;
            }
        }
        result[static_cast<std::size_t>(row)] = sum;
    }
    return result;
}

}  // namespace miinfer
