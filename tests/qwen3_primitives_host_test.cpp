#include "miinfer/qwen3_primitives.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    bool passed = true;
    std::vector<float> input(128, 2.0F);
    std::vector<float> weights(128, 0.5F);
    std::vector<float> output(128);
    miinfer::rms_norm_reference(input, weights, output, 1.0e-6F);
    passed = std::all_of(output.begin(), output.end(), [](float value) {
        return std::fabs(value - 0.5F) < 1.0e-5F;
    }) && passed;

    const std::array<float, 4> softmax_input{-1.0F, 0.0F, 1.0F, 2.0F};
    std::array<float, 4> softmax_output{};
    miinfer::softmax_reference(softmax_input, softmax_output);
    passed = std::fabs(std::accumulate(softmax_output.begin(), softmax_output.end(), 0.0F) - 1.0F)
             < 1.0e-5F && passed;

    miinfer::Q6KHostBlock block{};
    block.d_bits = 0x3c00U;
    std::fill(std::begin(block.scales), std::end(block.scales), 1);
    std::array<float, 256> values{};
    miinfer::q6_k_dequantize(block, values);
    passed = std::all_of(values.begin(), values.end(), [](float value) { return value == -32.0F; })
             && passed;
    std::cout << "Qwen3 host primitive tests: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
