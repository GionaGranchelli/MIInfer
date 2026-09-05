#include "miinfer/q4k_wave_layout.hpp"
#include <cstring>
#include <stdexcept>

std::vector<Q4KWaveTile> pack_q4k_wave_down(const miinfer::GgufTensor& tensor) {
    constexpr int rows = 5120, columns = 17408, blocks = columns / 256;
    if (tensor.type != miinfer::GgufTensorType::q4_k ||
        tensor.dimensions != std::vector<std::uint64_t>{columns, rows} ||
        tensor.byte_size != rows * blocks * sizeof(miinfer::Q4KDeviceBlock) ||
        tensor.data == nullptr)
        throw std::runtime_error("native Q4K Down requires Q4_K [17408,5120]");
    const auto* source = reinterpret_cast<const miinfer::Q4KDeviceBlock*>(tensor.data);
    std::vector<Q4KWaveTile> native(rows * 17);
    for (int row = 0; row < rows; ++row) for (int block = 0; block < blocks; ++block) {
        auto& tile = native[row * 17 + block / 4];
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
            throw std::runtime_error("native metadata conversion mismatch");
        for (int i = 0; i < 16; ++i) for (int plane = 0; plane < 2; ++plane) {
            const int offset = (i / 4) * 32 + (i % 4) * 4 + plane * 16;
            std::memcpy(&tile.words[plane][(block % 4)*16+i], src.qs + offset, 4);
        }
        // Validate every source payload byte independently of the packing traversal.
        for (int byte = 0; byte < 128; ++byte) {
            const int i = (byte / 32)*4 + (byte % 16)/4;
            const auto word = tile.words[(byte % 32)/16][(block % 4)*16+i];
            if (((word >> (8*(byte%4))) & 255) != src.qs[byte])
                throw std::runtime_error("native payload conversion mismatch");
        }
    }
    return native;
}

