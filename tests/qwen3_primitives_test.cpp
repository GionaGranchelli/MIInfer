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
    std::array<float, 256> q6_values{};
    miinfer::q6_k_dequantize(q6, q6_values);
    passed = std::all_of(q6_values.begin(), q6_values.end(), [](float value) { return value == -32.0F; }) && passed;
    std::cout << "q6 dequant host=" << (passed ? "PASS" : "FAIL") << '\n';
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

    std::vector<miinfer::Q6KDeviceBlock> q6_blocks(2);
    for (auto& block : q6_blocks) {
        block.d = __float2half(1.0F);
        std::fill(std::begin(block.ql), std::end(block.ql), 0);
        std::fill(std::begin(block.qh), std::end(block.qh), 0);
        std::fill(std::begin(block.scales), std::end(block.scales), 1);
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
    passed = close_enough(q6_output[0], -8192.0F) && close_enough(q6_output[1], -8192.0F) && passed;
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
