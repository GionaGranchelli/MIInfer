#include "miinfer/kquant_wave_layout.hpp"
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <array>

namespace {

inline float half_to_float(__half h) {
    #if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        return __half2float(h);
    #else
        // Host conversion using bit pattern
        std::uint16_t bits;
        std::memcpy(&bits, &h, sizeof(bits));
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
        float res;
        std::memcpy(&res, &value, sizeof(res));
        return res;
    #endif
}

std::int8_t signed_q6_canonical(const miinfer::Q6KDeviceBlock& block, std::size_t index) noexcept {
    const std::size_t group = index / 128;
    const std::size_t lane = index % 128;
    const std::size_t quarter = lane / 32;
    const std::size_t in_quarter = lane % 32;
    const std::size_t low_index = group * 64 + in_quarter
                                  + (quarter == 1 || quarter == 3 ? 32 : 0);
    const std::size_t high_index = group * 32 + in_quarter;
    const std::uint8_t low = quarter < 2
                                 ? block.ql[low_index] & 0x0fU
                                 : block.ql[low_index] >> 4U;
    const std::uint8_t high = static_cast<std::uint8_t>(
        (block.qh[high_index] >> (2U * static_cast<unsigned>(quarter))) & 0x03U);
    return static_cast<std::int8_t>(static_cast<int>(low | (high << 4U)) - 32);
}

} // namespace

std::vector<Q4KWaveTile> pack_q4k_wave_tensor(const miinfer::GgufTensor& tensor) {
    if (tensor.type != miinfer::GgufTensorType::q4_k ||
        tensor.dimensions.size() != 2 ||
        tensor.data == nullptr)
        throw std::runtime_error("pack_q4k_wave_tensor requires 2D Q4_K tensor: " + tensor.name);
    const std::uint64_t columns = tensor.dimensions[0];
    const std::uint64_t rows = tensor.dimensions[1];
    if (columns % 1024 != 0 || rows % 2 != 0)
        throw std::runtime_error("pack_q4k_wave_tensor requires columns % 1024 == 0 and rows % 2 == 0: " + tensor.name);
    const std::size_t blocks = columns / 256;
    const std::size_t tiles_per_row = columns / 1024;
    if (tensor.byte_size != rows * blocks * sizeof(miinfer::Q4KDeviceBlock))
        throw std::runtime_error("pack_q4k_wave_tensor byte size mismatch: " + tensor.name);

    const auto* source = reinterpret_cast<const miinfer::Q4KDeviceBlock*>(tensor.data);
    std::vector<Q4KWaveTile> native(rows * tiles_per_row);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t block = 0; block < blocks; ++block) {
            auto& tile = native[row * tiles_per_row + block / 4];
            const auto& src = source[row * blocks + block];
            auto& m = tile.metadata[block % 4];
            m.d = src.d; m.dmin = src.dmin;
            for (int g = 0; g < 8; ++g) {
                m.scales[g] = g < 4 ? src.scales[g] & 63 :
                    (src.scales[g + 4] & 15) | ((src.scales[g - 4] >> 6) << 4);
                m.minimums[g] = g < 4 ? src.scales[g + 4] & 63 :
                    (src.scales[g + 4] >> 4) | ((src.scales[g] >> 6) << 4);
            }
            std::uint8_t reconstructed[12]{};
            for (int g = 0; g < 4; ++g) {
                reconstructed[g] = m.scales[g] | ((m.scales[g+4] >> 4) << 6);
                reconstructed[g+4] = m.minimums[g] | ((m.minimums[g+4] >> 4) << 6);
                reconstructed[g+8] = (m.scales[g+4] & 15) | ((m.minimums[g+4] & 15) << 4);
            }
            if (std::memcmp(reconstructed, src.scales, 12) ||
                std::memcmp(&m.d, &src.d, sizeof(__half)) ||
                std::memcmp(&m.dmin, &src.dmin, sizeof(__half)))
                throw std::runtime_error("native Q4K metadata conversion mismatch: " + tensor.name);
            for (int i = 0; i < 16; ++i) for (int plane = 0; plane < 2; ++plane) {
                const int offset = (i / 4) * 32 + (i % 4) * 4 + plane * 16;
                std::memcpy(&tile.words[plane][(block % 4)*16+i], src.qs + offset, 4);
            }
            // Validate every source payload byte independently of the packing traversal.
            for (int byte = 0; byte < 128; ++byte) {
                const int i = (byte / 32)*4 + (byte % 16)/4;
                const auto word = tile.words[(byte % 32)/16][(block % 4)*16+i];
                if (((word >> (8*(byte%4))) & 255) != src.qs[byte])
                    throw std::runtime_error("native Q4K payload conversion mismatch: " + tensor.name);
            }
        }
    }
    return native;
}

std::vector<Q4KWaveTile> pack_q4k_wave_down(const miinfer::GgufTensor& tensor) {
    if (tensor.dimensions != std::vector<std::uint64_t>{17408, 5120})
        throw std::runtime_error("native Q4K Down requires Q4_K [17408,5120]");
    return pack_q4k_wave_tensor(tensor);
}

std::vector<Q5KWaveTile> pack_q5k_wave_tensor(const miinfer::GgufTensor& tensor) {
    if (tensor.type != miinfer::GgufTensorType::q5_k ||
        tensor.dimensions.size() != 2 ||
        tensor.data == nullptr)
        throw std::runtime_error("pack_q5k_wave_tensor requires 2D Q5_K tensor: " + tensor.name);
    const std::uint64_t columns = tensor.dimensions[0];
    const std::uint64_t rows = tensor.dimensions[1];
    if (columns % 1024 != 0 || rows % 2 != 0)
        throw std::runtime_error("pack_q5k_wave_tensor requires columns % 1024 == 0 and rows % 2 == 0: " + tensor.name);
    const std::size_t blocks = columns / 256;
    const std::size_t tiles_per_row = columns / 1024;
    if (tensor.byte_size != rows * blocks * sizeof(miinfer::Q5KDeviceBlock))
        throw std::runtime_error("pack_q5k_wave_tensor byte size mismatch: " + tensor.name);

    const auto* source = reinterpret_cast<const miinfer::Q5KDeviceBlock*>(tensor.data);
    std::vector<Q5KWaveTile> native(rows * tiles_per_row);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t block = 0; block < blocks; ++block) {
            auto& tile = native[row * tiles_per_row + block / 4];
            const auto& src = source[row * blocks + block];
            auto& m = tile.metadata[block % 4];
            m.d = src.d; m.dmin = src.dmin;
            for (int g = 0; g < 8; ++g) {
                m.scales[g] = g < 4 ? src.scales[g] & 63 :
                    (src.scales[g + 4] & 15) | ((src.scales[g - 4] >> 6) << 4);
                m.minimums[g] = g < 4 ? src.scales[g + 4] & 63 :
                    (src.scales[g + 4] >> 4) | ((src.scales[g] >> 6) << 4);
            }
            std::uint8_t reconstructed[12]{};
            for (int g = 0; g < 4; ++g) {
                reconstructed[g] = m.scales[g] | ((m.scales[g+4] >> 4) << 6);
                reconstructed[g+4] = m.minimums[g] | ((m.minimums[g+4] >> 4) << 6);
                reconstructed[g+8] = (m.scales[g+4] & 15) | ((m.minimums[g+4] & 15) << 4);
            }
            if (std::memcmp(reconstructed, src.scales, 12) ||
                std::memcmp(&m.d, &src.d, sizeof(__half)) ||
                std::memcmp(&m.dmin, &src.dmin, sizeof(__half)))
                throw std::runtime_error("native Q5K metadata conversion mismatch: " + tensor.name);

            // Pack low nibble planes (same structure as Q4_K)
            for (int i = 0; i < 16; ++i) for (int plane = 0; plane < 2; ++plane) {
                const int offset = (i / 4) * 32 + (i % 4) * 4 + plane * 16;
                std::memcpy(&tile.words[plane][(block % 4)*16+i], src.ql + offset, 4);
            }

            // Pack high 5th bit plane into tile.qh
            for (int i = 0; i < 16; ++i) {
                const int lane = (block % 4) * 16 + i;
                const int pair = i / 4;
                const int sub = i % 4;
                std::uint32_t packed_qh = 0;
                for (int j = 0; j < 4; ++j) {
                    const int idx0 = 4 * sub + j;
                    const int idx1 = 16 + 4 * sub + j;
                    const std::uint32_t h0_p0 = (src.qh[idx0] >> (2 * pair)) & 1;
                    const std::uint32_t h0_p1 = (src.qh[idx0] >> (2 * pair + 1)) & 1;
                    const std::uint32_t h1_p0 = (src.qh[idx1] >> (2 * pair)) & 1;
                    const std::uint32_t h1_p1 = (src.qh[idx1] >> (2 * pair + 1)) & 1;
                    const std::uint32_t byte_val = h0_p0 | (h0_p1 << 1) | (h1_p0 << 2) | (h1_p1 << 3);
                    packed_qh |= (byte_val << (8 * j));
                }
                tile.qh[lane] = packed_qh;
            }

            // Independent verification: reconstruct src.ql
            for (int byte = 0; byte < 128; ++byte) {
                const int i = (byte / 32)*4 + (byte % 16)/4;
                const auto word = tile.words[(byte % 32)/16][(block % 4)*16+i];
                if (((word >> (8*(byte%4))) & 255) != src.ql[byte])
                    throw std::runtime_error("native Q5K payload ql conversion mismatch: " + tensor.name);
            }

            // Independent verification: reconstruct src.qh
            std::uint8_t reconstructed_qh[32]{};
            for (int idx = 0; idx < 32; ++idx) {
                const int plane = idx / 16;
                const int in_p = idx % 16;
                const int sub = in_p / 4;
                const int j = in_p % 4;
                std::uint8_t byte_val = 0;
                for (int pair = 0; pair < 4; ++pair) {
                    const int i = pair * 4 + sub;
                    const int lane = (block % 4) * 16 + i;
                    const std::uint32_t h = (tile.qh[lane] >> (8 * j)) & 0xff;
                    if (plane == 0) {
                        const std::uint8_t p0 = h & 1;
                        const std::uint8_t p1 = (h >> 1) & 1;
                        byte_val |= (p0 << (2 * pair)) | (p1 << (2 * pair + 1));
                    } else {
                        const std::uint8_t p0 = (h >> 2) & 1;
                        const std::uint8_t p1 = (h >> 3) & 1;
                        byte_val |= (p0 << (2 * pair)) | (p1 << (2 * pair + 1));
                    }
                }
                reconstructed_qh[idx] = byte_val;
            }
            if (std::memcmp(reconstructed_qh, src.qh, 32) != 0)
                throw std::runtime_error("native Q5K payload qh conversion mismatch: " + tensor.name);
        }
    }
    return native;
}

std::vector<Q6KWaveTile> pack_q6k_wave_tensor(const miinfer::GgufTensor& tensor) {
    if (tensor.type != miinfer::GgufTensorType::q6_k ||
        tensor.dimensions.size() != 2 ||
        tensor.data == nullptr)
        throw std::runtime_error("pack_q6k_wave_tensor requires 2D Q6_K tensor: " + tensor.name);
    const std::uint64_t columns = tensor.dimensions[0];
    const std::uint64_t rows = tensor.dimensions[1];
    if (columns % 1024 != 0 || rows % 2 != 0)
        throw std::runtime_error("pack_q6k_wave_tensor requires columns % 1024 == 0 and rows % 2 == 0: " + tensor.name);
    const std::size_t blocks = columns / 256;
    const std::size_t tiles_per_row = columns / 1024;
    if (tensor.byte_size != rows * blocks * sizeof(miinfer::Q6KDeviceBlock))
        throw std::runtime_error("pack_q6k_wave_tensor byte size mismatch: " + tensor.name);

    const auto* source = reinterpret_cast<const miinfer::Q6KDeviceBlock*>(tensor.data);
    std::vector<Q6KWaveTile> native(rows * tiles_per_row);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t block = 0; block < blocks; ++block) {
            auto& tile = native[row * tiles_per_row + block / 4];
            const auto& src = source[row * blocks + block];
            auto& m = tile.metadata[block % 4];
            m.d = src.d;
            std::memcpy(m.scales, src.scales, 16);
            m.pad[0] = 0;
            m.pad[1] = 0;

            if (std::memcmp(&m.d, &src.d, sizeof(__half)) ||
                std::memcmp(m.scales, src.scales, 16))
                throw std::runtime_error("native Q6K metadata conversion mismatch: " + tensor.name);

            // Pack words[0], words[1], and qh for the 16 lanes in this block
            for (int i = 0; i < 16; ++i) {
                const int lane = (block % 4) * 16 + i;
                const int pair = i / 4;
                const int sub = i % 4;

                std::uint32_t w0 = 0;
                std::uint32_t w1 = 0;
                std::uint32_t qh_val = 0;

                const int group_0 = 2 * pair;
                const int group_1 = 2 * pair + 1;

                for (int j = 0; j < 4; ++j) {
                    const int elem_a0 = 32 * group_0 + 4 * sub + j;
                    const int elem_b0 = 32 * group_0 + 16 + 4 * sub + j;
                    const int elem_a1 = 32 * group_1 + 4 * sub + j;
                    const int elem_b1 = 32 * group_1 + 16 + 4 * sub + j;

                    auto extract_q6 = [&](int elem) {
                        const std::size_t g = elem / 128;
                        const std::size_t l = elem % 128;
                        const std::size_t q = l / 32;
                        const std::size_t in_q = l % 32;
                        const std::size_t low_idx = g * 64 + in_q + (q == 1 || q == 3 ? 32 : 0);
                        const std::size_t high_idx = g * 32 + in_q;
                        const std::uint8_t low = (q < 2) ? (src.ql[low_idx] & 0x0f) : (src.ql[low_idx] >> 4);
                        const std::uint8_t high = (src.qh[high_idx] >> (2 * q)) & 0x03;
                        return std::make_pair(low, high);
                    };

                    const auto [low_a0, high_a0] = extract_q6(elem_a0);
                    const auto [low_b0, high_b0] = extract_q6(elem_b0);
                    const auto [low_a1, high_a1] = extract_q6(elem_a1);
                    const auto [low_b1, high_b1] = extract_q6(elem_b1);

                    const std::uint32_t byte_w0 = low_a0 | (low_a1 << 4);
                    const std::uint32_t byte_w1 = low_b0 | (low_b1 << 4);
                    const std::uint32_t byte_qh = high_a0 | (high_b0 << 2) | (high_a1 << 4) | (high_b1 << 6);

                    w0 |= (byte_w0 << (8 * j));
                    w1 |= (byte_w1 << (8 * j));
                    qh_val |= (byte_qh << (8 * j));
                }

                tile.words[0][lane] = w0;
                tile.words[1][lane] = w1;
                tile.qh[lane] = qh_val;
            }

            // Independent verification: verify signed_q6 for all 256 elements in the block
            for (int elem = 0; elem < 256; ++elem) {
                const int expected = signed_q6_canonical(src, elem);
                const int q8_idx = elem / 32;
                const int in_q8 = elem % 32;
                const int pair = q8_idx / 2;
                const int part = q8_idx % 2;
                const int is_b = (in_q8 >= 16);
                const int in_half = in_q8 % 16;
                const int sub = in_half / 4;
                const int j = in_half % 4;

                const int i = pair * 4 + sub;
                const int lane = (block % 4) * 16 + i;

                std::uint32_t word = is_b ? tile.words[1][lane] : tile.words[0][lane];
                std::uint32_t qh_word = tile.qh[lane];

                std::uint32_t byte_w = (word >> (8 * j)) & 0xff;
                std::uint32_t low = (part == 0) ? (byte_w & 0x0f) : (byte_w >> 4);

                std::uint32_t byte_qh = (qh_word >> (8 * j)) & 0xff;
                int shift = 0;
                if (part == 0 && !is_b) shift = 0;
                else if (part == 0 && is_b) shift = 2;
                else if (part == 1 && !is_b) shift = 4;
                else if (part == 1 && is_b) shift = 6;
                std::uint32_t high = (byte_qh >> shift) & 0x03;

                int actual = static_cast<int>(low | (high << 4)) - 32;
                if (actual != expected)
                    throw std::runtime_error("native Q6K value mismatch at elem " + std::to_string(elem) + ": " + tensor.name);
            }
        }
    }
    return native;
}

void q4k_wave_tile_dequantize(const Q4KWaveTile& tile, float* output) {
    for (int block = 0; block < 4; ++block) {
        const auto& m = tile.metadata[block];
        const float d = half_to_float(m.d);
        const float dmin = half_to_float(m.dmin);

        for (int i = 0; i < 16; ++i) {
            const int lane = block * 16 + i;
            const int pair = i / 4;
            const int sub = i % 4;

            const std::uint32_t v0 = tile.words[0][lane];
            const std::uint32_t v1 = tile.words[1][lane];

            for (int part = 0; part < 2; ++part) {
                const int group = 2 * pair + part;
                const float scale = static_cast<float>(m.scales[group]);
                const float minimum = static_cast<float>(m.minimums[group]);

                for (int j = 0; j < 4; ++j) {
                    const int q0 = (v0 >> (4 * part + 8 * j)) & 0x0f;
                    const int q1 = (v1 >> (4 * part + 8 * j)) & 0x0f;

                    const int elem0 = block * 256 + 32 * group + 4 * sub + j;
                    const int elem1 = block * 256 + 32 * group + 16 + 4 * sub + j;

                    output[elem0] = d * scale * static_cast<float>(q0) - dmin * minimum;
                    output[elem1] = d * scale * static_cast<float>(q1) - dmin * minimum;
                }
            }
        }
    }
}

void q5k_wave_tile_dequantize(const Q5KWaveTile& tile, float* output) {
    for (int block = 0; block < 4; ++block) {
        const auto& m = tile.metadata[block];
        const float d = half_to_float(m.d);
        const float dmin = half_to_float(m.dmin);

        for (int i = 0; i < 16; ++i) {
            const int lane = block * 16 + i;
            const int pair = i / 4;
            const int sub = i % 4;

            const std::uint32_t v0 = tile.words[0][lane];
            const std::uint32_t v1 = tile.words[1][lane];
            const std::uint32_t h = tile.qh[lane];

            for (int part = 0; part < 2; ++part) {
                const int group = 2 * pair + part;
                const float scale = static_cast<float>(m.scales[group]);
                const float minimum = static_cast<float>(m.minimums[group]);

                for (int j = 0; j < 4; ++j) {
                    const int low0 = (v0 >> (4 * part + 8 * j)) & 0x0f;
                    const int low1 = (v1 >> (4 * part + 8 * j)) & 0x0f;

                    const std::uint32_t byte_h = (h >> (8 * j)) & 0xff;
                    const int h0 = (part == 0) ? (byte_h & 1) : ((byte_h >> 1) & 1);
                    const int h1 = (part == 0) ? ((byte_h >> 2) & 1) : ((byte_h >> 3) & 1);

                    const int q0 = low0 | (h0 << 4);
                    const int q1 = low1 | (h1 << 4);

                    const int elem0 = block * 256 + 32 * group + 4 * sub + j;
                    const int elem1 = block * 256 + 32 * group + 16 + 4 * sub + j;

                    output[elem0] = d * scale * static_cast<float>(q0) - dmin * minimum;
                    output[elem1] = d * scale * static_cast<float>(q1) - dmin * minimum;
                }
            }
        }
    }
}

void q6k_wave_tile_dequantize(const Q6KWaveTile& tile, float* output) {
    for (int block = 0; block < 4; ++block) {
        const auto& m = tile.metadata[block];
        const float d = half_to_float(m.d);

        for (int i = 0; i < 16; ++i) {
            const int lane = block * 16 + i;
            const int pair = i / 4;
            const int sub = i % 4;

            const std::uint32_t v0 = tile.words[0][lane];
            const std::uint32_t v1 = tile.words[1][lane];
            const std::uint32_t h = tile.qh[lane];

            for (int part = 0; part < 2; ++part) {
                const int group = 2 * pair + part;
                const float scale0 = static_cast<float>(m.scales[2 * group]);
                const float scale1 = static_cast<float>(m.scales[2 * group + 1]);

                for (int j = 0; j < 4; ++j) {
                    const int low0 = (v0 >> (4 * part + 8 * j)) & 0x0f;
                    const int low1 = (v1 >> (4 * part + 8 * j)) & 0x0f;

                    const std::uint32_t byte_h = (h >> (8 * j)) & 0xff;
                    const int shift0 = (part == 0) ? 0 : 4;
                    const int shift1 = (part == 0) ? 2 : 6;
                    const int h0 = (byte_h >> shift0) & 0x03;
                    const int h1 = (byte_h >> shift1) & 0x03;

                    const int q0 = static_cast<int>(low0 | (h0 << 4)) - 32;
                    const int q1 = static_cast<int>(low1 | (h1 << 4)) - 32;

                    const int elem0 = block * 256 + 32 * group + 4 * sub + j;
                    const int elem1 = block * 256 + 32 * group + 16 + 4 * sub + j;

                    output[elem0] = d * scale0 * static_cast<float>(q0);
                    output[elem1] = d * scale1 * static_cast<float>(q1);
                }
            }
        }
    }
}

void q4k_wave_gemv_reference(const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                             std::uint32_t rows, std::uint32_t columns) {
    const std::uint32_t num_tiles = columns / 1024;
    std::vector<float> tile_weights(1024);

    for (std::uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (std::uint32_t t = 0; t < num_tiles; ++t) {
            const auto& tile = w[r * num_tiles + t];
            q4k_wave_tile_dequantize(tile, tile_weights.data());

            for (int k = 0; k < 1024; ++k) {
                const int q8_block_idx = (t * 1024 + k) / 32;
                const int q8_in_block = (t * 1024 + k) % 32;
                const auto& q8 = x[q8_block_idx];
                const float x_val = half_to_float(q8.d) * static_cast<float>(q8.qs[q8_in_block]);
                sum += tile_weights[k] * x_val;
            }
        }
        y[r] = sum;
    }
}

void q5k_wave_gemv_reference(const Q5KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                             std::uint32_t rows, std::uint32_t columns) {
    const std::uint32_t num_tiles = columns / 1024;
    std::vector<float> tile_weights(1024);

    for (std::uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (std::uint32_t t = 0; t < num_tiles; ++t) {
            const auto& tile = w[r * num_tiles + t];
            q5k_wave_tile_dequantize(tile, tile_weights.data());

            for (int k = 0; k < 1024; ++k) {
                const int q8_block_idx = (t * 1024 + k) / 32;
                const int q8_in_block = (t * 1024 + k) % 32;
                const auto& q8 = x[q8_block_idx];
                const float x_val = half_to_float(q8.d) * static_cast<float>(q8.qs[q8_in_block]);
                sum += tile_weights[k] * x_val;
            }
        }
        y[r] = sum;
    }
}

void q6k_wave_gemv_reference(const Q6KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
                             std::uint32_t rows, std::uint32_t columns) {
    const std::uint32_t num_tiles = columns / 1024;
    std::vector<float> tile_weights(1024);

    for (std::uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (std::uint32_t t = 0; t < num_tiles; ++t) {
            const auto& tile = w[r * num_tiles + t];
            q6k_wave_tile_dequantize(tile, tile_weights.data());

            for (int k = 0; k < 1024; ++k) {
                const int q8_block_idx = (t * 1024 + k) / 32;
                const int q8_in_block = (t * 1024 + k) % 32;
                const auto& q8 = x[q8_block_idx];
                const float x_val = half_to_float(q8.d) * static_cast<float>(q8.qs[q8_in_block]);
                sum += tile_weights[k] * x_val;
            }
        }
        y[r] = sum;
    }
}
