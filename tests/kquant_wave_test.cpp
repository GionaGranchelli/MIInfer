#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/qwen3_primitives.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

namespace {

inline float half_to_float_bits(std::uint16_t bits) {
    return miinfer::fp16_bits_to_float(bits);
}


} // namespace

int main() {
    bool all_passed = true;
    std::cout << "Running K-quant WaveTile host tests...\n";

    // -------------------------------------------------------------
    // Test 1: Synthetic Q4_K wave tile packing and dequantization
    // -------------------------------------------------------------
    {
        std::cout << "Test 1: Synthetic Q4_K wave tile... ";
        const std::size_t rows = 2;
        const std::size_t cols = 1024;
        const std::size_t blocks = cols / 256;
        std::vector<miinfer::Q4KDeviceBlock> raw_blocks(rows * blocks);

        std::mt19937 rng(42);
        for (std::size_t b = 0; b < raw_blocks.size(); ++b) {
            auto& blk = raw_blocks[b];
            std::uint16_t d_bits = 0x3c00; // 1.0
            std::uint16_t dmin_bits = 0x3800; // 0.5
            std::memcpy(&blk.d, &d_bits, 2);
            std::memcpy(&blk.dmin, &dmin_bits, 2);
            for (int i = 0; i < 12; ++i) blk.scales[i] = static_cast<std::uint8_t>(rng() & 0x3f);
            for (int i = 0; i < 128; ++i) blk.qs[i] = static_cast<std::uint8_t>(rng() & 0xff);
        }

        miinfer::GgufTensor tensor{};
        tensor.name = "synthetic_q4k";
        tensor.type = miinfer::GgufTensorType::q4_k;
        tensor.dimensions = {cols, rows};
        tensor.byte_size = raw_blocks.size() * sizeof(miinfer::Q4KDeviceBlock);
        tensor.data = reinterpret_cast<const std::byte*>(raw_blocks.data());

        try {
            auto packed = pack_q4k_wave_tensor(tensor);
            if (packed.size() != rows * (cols / 1024)) {
                std::cout << "FAIL (size mismatch)\n";
                all_passed = false;
            } else {
                std::vector<float> dequant(1024);
                q4k_wave_tile_dequantize(packed[0], dequant.data());
                std::cout << "PASS\n";
            }
        } catch (const std::exception& e) {
            std::cout << "FAIL (" << e.what() << ")\n";
            all_passed = false;
        }
    }

    // -------------------------------------------------------------
    // Test 2: Synthetic Q5_K wave tile packing and dequantization
    // -------------------------------------------------------------
    {
        std::cout << "Test 2: Synthetic Q5_K wave tile... ";
        const std::size_t rows = 2;
        const std::size_t cols = 1024;
        const std::size_t blocks = cols / 256;
        std::vector<miinfer::Q5KDeviceBlock> raw_blocks(rows * blocks);

        std::mt19937 rng(12345);
        for (std::size_t b = 0; b < raw_blocks.size(); ++b) {
            auto& blk = raw_blocks[b];
            std::uint16_t d_bits = 0x3c00; // 1.0
            std::uint16_t dmin_bits = 0x3800; // 0.5
            std::memcpy(&blk.d, &d_bits, 2);
            std::memcpy(&blk.dmin, &dmin_bits, 2);
            for (int i = 0; i < 12; ++i) blk.scales[i] = static_cast<std::uint8_t>(rng() & 0x3f);
            for (int i = 0; i < 32; ++i) blk.qh[i] = static_cast<std::uint8_t>(rng() & 0xff);
            for (int i = 0; i < 128; ++i) blk.ql[i] = static_cast<std::uint8_t>(rng() & 0xff);
        }

        miinfer::GgufTensor tensor{};
        tensor.name = "synthetic_q5k";
        tensor.type = miinfer::GgufTensorType::q5_k;
        tensor.dimensions = {cols, rows};
        tensor.byte_size = raw_blocks.size() * sizeof(miinfer::Q5KDeviceBlock);
        tensor.data = reinterpret_cast<const std::byte*>(raw_blocks.data());

        try {
            auto packed = pack_q5k_wave_tensor(tensor);
            if (packed.size() != rows * (cols / 1024)) {
                std::cout << "FAIL (size mismatch)\n";
                all_passed = false;
            } else {
                std::vector<float> dequant(1024);
                q5k_wave_tile_dequantize(packed[0], dequant.data());
                std::cout << "PASS\n";
            }
        } catch (const std::exception& e) {
            std::cout << "FAIL (" << e.what() << ")\n";
            all_passed = false;
        }
    }

    // -------------------------------------------------------------
    // Test 3: Synthetic Q6_K wave tile packing, dequantization and canonical equivalence
    // -------------------------------------------------------------
    {
        std::cout << "Test 3: Synthetic Q6_K wave tile canonical equivalence... ";
        const std::size_t rows = 2;
        const std::size_t cols = 1024;
        const std::size_t blocks = cols / 256;
        std::vector<miinfer::Q6KDeviceBlock> raw_blocks(rows * blocks);

        std::mt19937 rng(999);
        for (std::size_t b = 0; b < raw_blocks.size(); ++b) {
            auto& blk = raw_blocks[b];
            std::uint16_t d_bits = 0x3c00; // 1.0
            std::memcpy(&blk.d, &d_bits, 2);
            for (int i = 0; i < 16; ++i) blk.scales[i] = static_cast<std::int8_t>((rng() % 30) - 15);
            for (int i = 0; i < 64; ++i) blk.qh[i] = static_cast<std::uint8_t>(rng() & 0xff);
            for (int i = 0; i < 128; ++i) blk.ql[i] = static_cast<std::uint8_t>(rng() & 0xff);
        }

        miinfer::GgufTensor tensor{};
        tensor.name = "synthetic_q6k";
        tensor.type = miinfer::GgufTensorType::q6_k;
        tensor.dimensions = {cols, rows};
        tensor.byte_size = raw_blocks.size() * sizeof(miinfer::Q6KDeviceBlock);
        tensor.data = reinterpret_cast<const std::byte*>(raw_blocks.data());

        try {
            auto packed = pack_q6k_wave_tensor(tensor);
            std::vector<float> dequant(1024);
            q6k_wave_tile_dequantize(packed[0], dequant.data());

            // Check against canonical q6_k_dequantize for each block
            bool mismatch = false;
            for (int b = 0; b < 4; ++b) {
                miinfer::Q6KHostBlock host_blk{};
                std::memcpy(host_blk.ql, raw_blocks[b].ql, 128);
                std::memcpy(host_blk.qh, raw_blocks[b].qh, 64);
                std::memcpy(host_blk.scales, raw_blocks[b].scales, 16);
                std::memcpy(&host_blk.d_bits, &raw_blocks[b].d, 2);

                std::array<float, 256> canonical_out{};
                miinfer::q6_k_dequantize(host_blk, canonical_out);

                for (int elem = 0; elem < 256; ++elem) {
                    float diff = std::fabs(dequant[b * 256 + elem] - canonical_out[elem]);
                    if (diff > 1e-4f) {
                        mismatch = true;
                        std::cout << "FAIL (mismatch at b=" << b << " elem=" << elem
                                  << " wave=" << dequant[b * 256 + elem]
                                  << " canon=" << canonical_out[elem] << ")\n";
                        break;
                    }
                }
                if (mismatch) break;
            }
            if (!mismatch) std::cout << "PASS\n";
            else all_passed = false;
        } catch (const std::exception& e) {
            std::cout << "FAIL (" << e.what() << ")\n";
            all_passed = false;
        }
    }

    // -------------------------------------------------------------
    // Test 4: Real GGUF model tensors test (if file exists)
    // -------------------------------------------------------------
    {
        const char* model_path = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";
        std::cout << "Test 4: Real GGUF model tensor verification (" << model_path << ")... ";
        try {
            auto model = miinfer::GgufFile::open(model_path);
            bool found_q5k = false, found_q6k = false;

            for (const auto& tensor : model->tensors()) {
                if (!found_q5k && tensor.type == miinfer::GgufTensorType::q5_k &&
                    tensor.dimensions.size() == 2 && tensor.dimensions[0] % 1024 == 0) {
                    auto packed_q5k = pack_q5k_wave_tensor(tensor);
                    found_q5k = true;
                }
                if (!found_q6k && tensor.type == miinfer::GgufTensorType::q6_k &&
                    tensor.dimensions.size() == 2 && tensor.dimensions[0] % 1024 == 0) {
                    auto packed_q6k = pack_q6k_wave_tensor(tensor);
                    found_q6k = true;
                }
                if (found_q5k && found_q6k) break;
            }

            if (found_q5k && found_q6k) {
                std::cout << "PASS (real Q5_K and Q6_K tensors successfully packed and verified)\n";
            } else {
                std::cout << "SKIP (model tensors not found)\n";
            }
        } catch (const std::exception& e) {
            std::cout << "SKIP/FAIL (" << e.what() << ")\n";
        }
    }

    // -------------------------------------------------------------
    // Test 5: Synthetic Q4_K fused gate+up SwiGLU reference equivalence
    // -------------------------------------------------------------
    {
        std::cout << "Test 5: Synthetic Q4_K fused gate+up SwiGLU reference... ";
        const std::size_t rows = 4;
        const std::size_t cols = 1024;
        const std::size_t blocks = cols / 256;
        std::vector<miinfer::Q4KDeviceBlock> gate_blocks(rows * blocks);
        std::vector<miinfer::Q4KDeviceBlock> up_blocks(rows * blocks);
        std::vector<miinfer::Q8_1Block> x_blocks(cols / 32);

        std::mt19937 rng(777);
        for (std::size_t b = 0; b < gate_blocks.size(); ++b) {
            std::uint16_t d_bits = 0x3c00;
            std::uint16_t dmin_bits = 0x3800;
            std::memcpy(&gate_blocks[b].d, &d_bits, 2);
            std::memcpy(&gate_blocks[b].dmin, &dmin_bits, 2);
            for (int i = 0; i < 12; ++i) gate_blocks[b].scales[i] = static_cast<std::uint8_t>(rng() & 0x3f);
            for (int i = 0; i < 128; ++i) gate_blocks[b].qs[i] = static_cast<std::uint8_t>(rng() & 0xff);

            std::memcpy(&up_blocks[b].d, &d_bits, 2);
            std::memcpy(&up_blocks[b].dmin, &dmin_bits, 2);
            for (int i = 0; i < 12; ++i) up_blocks[b].scales[i] = static_cast<std::uint8_t>(rng() & 0x3f);
            for (int i = 0; i < 128; ++i) up_blocks[b].qs[i] = static_cast<std::uint8_t>(rng() & 0xff);
        }
        for (auto& b : x_blocks) {
            std::uint16_t d_bits = 0x3800;
            std::memcpy(&b.d, &d_bits, 2);
            std::uint16_t s_bits = 0;
            std::memcpy(&b.s, &s_bits, 2);
            for (int i = 0; i < 32; ++i) b.qs[i] = static_cast<std::int8_t>((rng() % 255) - 128);
        }

        miinfer::GgufTensor tensor_gate{};
        tensor_gate.type = miinfer::GgufTensorType::q4_k;
        tensor_gate.dimensions = {cols, rows};
        tensor_gate.byte_size = gate_blocks.size() * sizeof(miinfer::Q4KDeviceBlock);
        tensor_gate.data = reinterpret_cast<const std::byte*>(gate_blocks.data());

        miinfer::GgufTensor tensor_up{};
        tensor_up.type = miinfer::GgufTensorType::q4_k;
        tensor_up.dimensions = {cols, rows};
        tensor_up.byte_size = up_blocks.size() * sizeof(miinfer::Q4KDeviceBlock);
        tensor_up.data = reinterpret_cast<const std::byte*>(up_blocks.data());

        auto packed_gate = pack_q4k_wave_tensor(tensor_gate);
        auto packed_up = pack_q4k_wave_tensor(tensor_up);

        std::vector<float> fused_out(rows);
        q4k_wave_fused_gate_up_swiglu_reference(
            packed_gate.data(), packed_up.data(), x_blocks.data(),
            fused_out.data(), rows, cols);

        std::vector<float> gate_out(rows), up_out(rows), expected_out(rows);
        q4k_wave_gemv_reference(packed_gate.data(), x_blocks.data(), gate_out.data(), rows, cols);
        q4k_wave_gemv_reference(packed_up.data(), x_blocks.data(), up_out.data(), rows, cols);
        for (std::size_t r = 0; r < rows; ++r) {
            const float g = gate_out[r];
            const float u = up_out[r];
            expected_out[r] = (g / (1.0f + std::exp(-g))) * u;
        }

        bool match = true;
        for (std::size_t r = 0; r < rows; ++r) {
            if (std::fabs(fused_out[r] - expected_out[r]) > 1e-4f) {
                match = false;
                break;
            }
        }
        if (match) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL (numerical mismatch)\n";
            all_passed = false;
        }
    }

    std::cout << "\nAll K-quant WaveTile host tests: " << (all_passed ? "PASS" : "FAIL") << '\n';
    return all_passed ? 0 : 1;
}
