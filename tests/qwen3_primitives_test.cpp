#include "miinfer/hip_check.hpp"
#include "miinfer/device_validation.hpp"
#include "miinfer/q4_q8_gemv.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen3_primitives.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

namespace {

bool close_enough(float actual, float expected, float tolerance = 2.0e-4F) {
    return std::fabs(actual - expected)
           <= tolerance * std::max(1.0F, std::fabs(expected));
}

bool host_tests() {
    bool passed = true;

    std::vector<float> input(128);
    std::vector<float> weights(128);
    std::iota(input.begin(), input.end(), -64.0F);
    std::fill(weights.begin(), weights.end(), 1.5F);
    std::vector<float> normalized(input.size());
    miinfer::rms_norm_reference(input, weights, normalized, 1.0e-6F);
    const double sum = std::inner_product(input.begin(), input.end(), input.begin(), 0.0);
    const float expected_scale = 1.0F / std::sqrt(static_cast<float>(sum / input.size()) + 1.0e-6F);
    for (std::size_t index = 0; index < input.size(); ++index) {
        passed = close_enough(normalized[index], input[index] * expected_scale * 1.5F) && passed;
    }
    std::cout << "rms_norm host=" << (passed ? "PASS" : "FAIL") << '\n';

    std::vector<float> rope_input(2 * 8);
    std::iota(rope_input.begin(), rope_input.end(), 1.0F);
    std::vector<float> rope_output(rope_input.size());
    miinfer::rope_qwen3_reference(rope_input, rope_output, 2, 8, 0, 1000000.0F);
    passed = rope_output == rope_input && passed;
    miinfer::rope_qwen3_reference(rope_input, rope_output, 2, 8, 7, 1000000.0F);
    passed = std::all_of(rope_output.begin(), rope_output.end(), [](float value) {
        return std::isfinite(value);
    }) && passed;
    std::vector<float> neox_input{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<float> neox_output(neox_input.size());
    miinfer::rope_qwen3_reference(neox_input, neox_output, 1, 8, 1, 1.0F);
    const float c = std::cos(1.0F);
    const float s = std::sin(1.0F);
    passed = close_enough(neox_output[0], 1.0F * c - 5.0F * s)
             && close_enough(neox_output[4], 1.0F * s + 5.0F * c)
             && close_enough(neox_output[3], 4.0F * c - 8.0F * s)
             && close_enough(neox_output[7], 4.0F * s + 8.0F * c)
             && passed;
    std::cout << "rope host=" << (passed ? "PASS" : "FAIL") << '\n';

    const std::array<float, 4> softmax_input{-1.0F, 0.0F, 1.0F, 2.0F};
    std::array<float, 4> softmax_output{};
    miinfer::softmax_reference(softmax_input, softmax_output);
    const float probability_sum = std::accumulate(softmax_output.begin(), softmax_output.end(), 0.0F);
    passed = close_enough(probability_sum, 1.0F) && softmax_output.back() > softmax_output.front() && passed;
    std::cout << "softmax host=" << (passed ? "PASS" : "FAIL") << '\n';

    std::array<float, 4> gate{-2.0F, -0.5F, 0.0F, 2.0F};
    const std::array<float, 4> up{1.0F, 2.0F, 3.0F, 4.0F};
    std::array<float, 4> swiglu{};
    miinfer::silu_mul_reference(gate, up, swiglu);
    for (std::size_t index = 0; index < gate.size(); ++index) {
        passed = close_enough(swiglu[index], miinfer::silu_reference(gate[index]) * up[index]) && passed;
    }
    std::cout << "swiglu host=" << (passed ? "PASS" : "FAIL") << '\n';

    miinfer::Q4_0HostBlock q4{};
    q4.d_bits = 0x3c00U; // 1.0
    std::fill(std::begin(q4.qs), std::end(q4.qs), 0xF0U);
    std::array<float, 32> q4_values{};
    miinfer::q4_0_dequantize(q4, q4_values);
    passed = close_enough(q4_values[0], -8.0F) && close_enough(q4_values[15], -8.0F)
             && close_enough(q4_values[16], 7.0F) && close_enough(q4_values[31], 7.0F) && passed;
    std::cout << "q4 dequant host=" << (passed ? "PASS" : "FAIL") << '\n';

    miinfer::Q6KHostBlock q6{};
    q6.d_bits = 0x3c00U;
    std::fill(std::begin(q6.scales), std::end(q6.scales), 1);
    std::fill(std::begin(q6.ql), std::end(q6.ql), 0x00U);
    std::fill(std::begin(q6.qh), std::end(q6.qh), 0x00U);
    q6.ql[64] = 0x01U;
    std::array<float, 256> q6_values{};
    miinfer::q6_k_dequantize(q6, q6_values);
    passed = q6_values[128] == -31.0F
             && std::all_of(q6_values.begin(), q6_values.begin() + 128,
                            [](float value) { return value == -32.0F; })
             && std::all_of(q6_values.begin() + 129, q6_values.end(),
                            [](float value) { return value == -32.0F; }) && passed;
    std::cout << "q6 dequant host=" << (passed ? "PASS" : "FAIL")
              << " values=" << q6_values[0] << ',' << q6_values[64]
              << ',' << q6_values[127] << ',' << q6_values[128]
              << ',' << q6_values[159] << ',' << q6_values[160] << '\n';
    return passed;
}

template <typename T>
T* device_copy(const std::vector<T>& source) {
    T* destination = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&destination), source.size() * sizeof(T)));
    MIINFER_HIP_CHECK(hipMemcpy(destination, source.data(), source.size() * sizeof(T), hipMemcpyHostToDevice));
    return destination;
}

bool gpu_tests() {
    bool passed = true;
    std::vector<miinfer::Q4_0Block> embedding_blocks(4);
    for (std::size_t block = 0; block < embedding_blocks.size(); ++block) {
        embedding_blocks[block].d = __float2half(1.0F);
        std::fill(std::begin(embedding_blocks[block].qs), std::end(embedding_blocks[block].qs),
                  static_cast<std::uint8_t>(block == 0 ? 0xF0U : 0x08U));
    }
    auto* device_embedding = device_copy(embedding_blocks);
    float* device_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), 128 * sizeof(float)));
    miinfer::launch_qwen3_q4_embedding(reinterpret_cast<const std::byte*>(device_embedding), 1, 2, 64,
                                       device_output);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> embedding_output(64);
    MIINFER_HIP_CHECK(hipMemcpy(embedding_output.data(), device_output, embedding_output.size() * sizeof(float),
                                hipMemcpyDeviceToHost));
    passed = close_enough(embedding_output[0], 0.0F)
             && close_enough(embedding_output[15], 0.0F)
             && close_enough(embedding_output[16], -8.0F)
             && passed;
    std::cout << "q4 embedding gpu=" << (passed ? "PASS" : "FAIL")
              << " values=" << embedding_output[0] << ',' << embedding_output[15]
              << ',' << embedding_output[16] << '\n';

    std::vector<float> norm_input(128, 2.0F);
    std::vector<float> norm_weights(128, 0.5F);
    auto* device_norm_input = device_copy(norm_input);
    auto* device_norm_weights = device_copy(norm_weights);
    miinfer::launch_qwen3_rms_norm(device_norm_input, device_norm_weights, device_output, 128, 1.0e-6F);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> norm_output(128);
    MIINFER_HIP_CHECK(hipMemcpy(norm_output.data(), device_output, norm_output.size() * sizeof(float),
                                hipMemcpyDeviceToHost));
    passed = close_enough(norm_output[0], 0.5F) && close_enough(norm_output[127], 0.5F) && passed;
    std::cout << "rms_norm gpu=" << (passed ? "PASS" : "FAIL")
              << " values=" << norm_output[0] << ',' << norm_output[127] << '\n';
    // Test fused add rms norm with in-register Q8_1 quantization (5120 elements)
    constexpr int kFfnNormDim = 5120;
    constexpr int kFfnNormBlocks = kFfnNormDim / 32;
    std::vector<float> fused_res_in(kFfnNormDim, 1.5F);
    std::vector<float> fused_proj(kFfnNormDim, 0.5F);
    std::vector<float> fused_weights(kFfnNormDim, 0.8F);
    for (int i = 0; i < kFfnNormDim; ++i) {
        fused_res_in[i] = 1.0F + static_cast<float>(i % 17) * 0.1F;
        fused_proj[i] = -0.5F + static_cast<float>(i % 13) * 0.2F;
        fused_weights[i] = 0.5F + static_cast<float>(i % 7) * 0.1F;
    }
    auto* d_fused_res_in = device_copy(fused_res_in);
    auto* d_fused_proj = device_copy(fused_proj);
    auto* d_fused_weights = device_copy(fused_weights);
    float* d_fused_res_out = nullptr;
    float* d_fused_norm_ref = nullptr;
    float* d_fused_norm_test = nullptr;
    miinfer::Q8_1Block* d_fused_q8_ref = nullptr;
    miinfer::Q8_1Block* d_fused_q8_test = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_fused_res_out), kFfnNormDim * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_fused_norm_ref), kFfnNormDim * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_fused_norm_test), kFfnNormDim * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_fused_q8_ref), kFfnNormBlocks * sizeof(miinfer::Q8_1Block)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_fused_q8_test), kFfnNormBlocks * sizeof(miinfer::Q8_1Block)));

    // Reference: separate fused_add_rms_norm + launch_q8_1_quantize_f32
    miinfer::launch_qwen3_fused_add_rms_norm(
        d_fused_res_in, d_fused_proj, d_fused_weights,
        d_fused_res_out, d_fused_norm_ref, kFfnNormDim, 1.0e-6F);
    miinfer::launch_q8_1_quantize_f32(d_fused_norm_ref, d_fused_q8_ref, kFfnNormDim);

    // Candidate: fused_add_rms_norm with in-register q8_out
    miinfer::launch_qwen3_fused_add_rms_norm(
        d_fused_res_in, d_fused_proj, d_fused_weights,
        d_fused_res_out, d_fused_norm_test, kFfnNormDim, 1.0e-6F, nullptr, d_fused_q8_test);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    std::vector<miinfer::Q8_1Block> h_q8_ref(kFfnNormBlocks), h_q8_test(kFfnNormBlocks);
    MIINFER_HIP_CHECK(hipMemcpy(h_q8_ref.data(), d_fused_q8_ref, sizeof(miinfer::Q8_1Block) * kFfnNormBlocks, hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(h_q8_test.data(), d_fused_q8_test, sizeof(miinfer::Q8_1Block) * kFfnNormBlocks, hipMemcpyDeviceToHost));

    bool fused_q8_exact = true;
    for (int b = 0; b < kFfnNormBlocks; ++b) {
        if (__half2float(h_q8_ref[b].d) != __half2float(h_q8_test[b].d) ||
            __half2float(h_q8_ref[b].s) != __half2float(h_q8_test[b].s)) {
            fused_q8_exact = false;
            break;
        }
        for (int j = 0; j < 32; ++j) {
            if (h_q8_ref[b].qs[j] != h_q8_test[b].qs[j]) {
                fused_q8_exact = false;
                break;
            }
        }
    }
    passed = fused_q8_exact && passed;
    std::cout << "fused_add_rms_norm_q8 gpu=" << (fused_q8_exact ? "PASS" : "FAIL") << '\n';

    MIINFER_HIP_CHECK(hipFree(d_fused_q8_test));
    MIINFER_HIP_CHECK(hipFree(d_fused_q8_ref));
    MIINFER_HIP_CHECK(hipFree(d_fused_norm_test));
    MIINFER_HIP_CHECK(hipFree(d_fused_norm_ref));
    MIINFER_HIP_CHECK(hipFree(d_fused_res_out));
    MIINFER_HIP_CHECK(hipFree(d_fused_weights));
    MIINFER_HIP_CHECK(hipFree(d_fused_proj));
    MIINFER_HIP_CHECK(hipFree(d_fused_res_in));

    const std::vector<float> argmax_input{1.0F, 5.0F, 5.0F, -2.0F, 4.0F};
    auto* device_argmax_input = device_copy(argmax_input);
    std::uint32_t* device_argmax_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_argmax_output), sizeof(std::uint32_t)));
    miinfer::launch_qwen3_argmax(
        device_argmax_input, device_argmax_output,
        static_cast<std::uint32_t>(argmax_input.size()));
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::uint32_t argmax_output = 0;
    MIINFER_HIP_CHECK(hipMemcpy(&argmax_output, device_argmax_output, sizeof(argmax_output),
                                hipMemcpyDeviceToHost));
    passed = argmax_output == 1 && passed;
    std::cout << "argmax gpu=" << (argmax_output == 1 ? "PASS" : "FAIL")
              << " selected=" << argmax_output << " (first tie)\n";

    std::vector<float> large_argmax_input(151936, 0.0F);
    large_argmax_input[42000] = 10.0F;
    large_argmax_input[105000] = 10.0F;
    auto* device_large_input = device_copy(large_argmax_input);
    miinfer::launch_qwen3_argmax(
        device_large_input, device_argmax_output,
        static_cast<std::uint32_t>(large_argmax_input.size()));
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    MIINFER_HIP_CHECK(hipMemcpy(&argmax_output, device_argmax_output, sizeof(argmax_output),
                                hipMemcpyDeviceToHost));
    passed = argmax_output == 42000 && passed;
    std::cout << "argmax large gpu=" << (argmax_output == 42000 ? "PASS" : "FAIL")
              << " selected=" << argmax_output << " (expected 42000)\n";
    MIINFER_HIP_CHECK(hipFree(device_argmax_output));
    MIINFER_HIP_CHECK(hipFree(device_argmax_input));
    MIINFER_HIP_CHECK(hipFree(device_large_input));

    std::vector<miinfer::Q6KDeviceBlock> q6_blocks(2);
    for (auto& block : q6_blocks) {
        block.d = __float2half(1.0F);
        std::fill(std::begin(block.ql), std::end(block.ql), 0);
        std::fill(std::begin(block.qh), std::end(block.qh), 0);
        std::fill(std::begin(block.scales), std::end(block.scales), 1);
        block.ql[64] = 1;
    }
    auto* device_q6 = device_copy(q6_blocks);
    std::vector<float> q6_input(256, 1.0F);
    auto* device_q6_input = device_copy(q6_input);
    float* device_q6_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_q6_output), 2 * sizeof(float)));
    miinfer::launch_qwen3_q6_k_gemv(device_q6, device_q6_input, device_q6_output, 2, 256);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::array<float, 2> q6_output{};
    MIINFER_HIP_CHECK(hipMemcpy(q6_output.data(), device_q6_output, sizeof(q6_output), hipMemcpyDeviceToHost));
    passed = close_enough(q6_output[0], -8191.0F) && close_enough(q6_output[1], -8191.0F) && passed;
    std::cout << "q6 gemv gpu=" << (passed ? "PASS" : "FAIL")
              << " values=" << q6_output[0] << ',' << q6_output[1] << '\n';

    MIINFER_HIP_CHECK(hipFree(device_q6_output));
    MIINFER_HIP_CHECK(hipFree(device_q6_input));
    MIINFER_HIP_CHECK(hipFree(device_q6));
    MIINFER_HIP_CHECK(hipFree(device_norm_weights));
    MIINFER_HIP_CHECK(hipFree(device_norm_input));
    MIINFER_HIP_CHECK(hipFree(device_output));
    MIINFER_HIP_CHECK(hipFree(device_embedding));
    return passed;
}

}  // namespace

int main() {
    if (!host_tests()) return 1;
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) {
        std::cerr << "Qwen3 primitive GPU tests unavailable: " << error << '\n';
        return 1;
    }
    return gpu_tests() ? 0 : 1;
}
