#include "miinfer/device_validation.hpp"
#include "miinfer/fp16_gemv.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/q4_q8_packed_dot.hpp"
#include "miinfer/q4_q8_gemv.hpp"
#include "miinfer/q4_q8_zero_point_dot.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kSeed = 0x4D493050U;

bool run_zero_point_identity_tests() {
    miinfer::Q4_0Block weights{};
    miinfer::Q8_1Block input{};
    float maximum_algebra_error = 0.0F;
    float maximum_stored_sum_error = 0.0F;
    bool passed = true;
    for (int test_case = 0; test_case < 12; ++test_case) {
        std::fill(std::begin(weights.qs), std::end(weights.qs), 0);
        weights.d = __float2half(0.2173F + 0.011F * static_cast<float>(test_case));
        input.d = __float2half(0.0137F + 0.002F * static_cast<float>(test_case));
        int q_sum = 0;
        int raw_dot = 0;
        float direct = 0.0F;
        std::mt19937 generator(0x51A7U + static_cast<unsigned int>(test_case));
        for (int index = 0; index < miinfer::kQ4_0BlockSize; ++index) {
            int nibble = 0;
            switch (test_case) {
            case 0: nibble = 0; break;
            case 1: nibble = 15; break;
            case 2: nibble = 8; break;
            case 3: nibble = 7; break;
            case 4: nibble = index % 2 == 0 ? 0 : 15; break;
            case 5: nibble = index % 3; break;
            case 6: nibble = 14 - index % 4; break;
            default: nibble = static_cast<int>(generator() % 16U); break;
            }
            auto& packed = weights.qs[index < 16 ? index : index - 16];
            if (index < 16) {
                packed = static_cast<std::uint8_t>((packed & 0xF0U) | nibble);
            } else {
                packed = static_cast<std::uint8_t>((packed & 0x0FU) | (nibble << 4));
            }
            const int activation = test_case == 5 ? 127
                                  : test_case == 6 ? -127
                                  : test_case == 7 ? (index % 2 == 0 ? 37 : -37)
                                  : test_case < 5 ? ((index % 5) - 2) * 11
                                                  : static_cast<int>(generator() % 255U) - 127;
            input.qs[index] = static_cast<std::int8_t>(activation);
            q_sum += activation;
            raw_dot += nibble * activation;
            direct += static_cast<float>(nibble - 8) * __half2float(weights.d)
                      * static_cast<float>(activation) * __half2float(input.d);
        }
        input.s = __float2half(static_cast<float>(q_sum) * __half2float(input.d));
        const float algebraic = __half2float(weights.d)
                                * (__half2float(input.d) * static_cast<float>(raw_dot)
                                   - 8.0F * __half2float(input.d)
                                   * static_cast<float>(q_sum));
        const float stored_s = __half2float(weights.d)
                               * (__half2float(input.d) * static_cast<float>(raw_dot)
                                  - 8.0F * __half2float(input.s));
        maximum_stored_sum_error = std::max(maximum_stored_sum_error,
                                            std::fabs(stored_s - direct));
        maximum_algebra_error = std::max(maximum_algebra_error,
                                         std::fabs(algebraic - direct));
        passed = std::fabs(algebraic - direct) < 1.0e-3F && passed;
    }
    passed = maximum_stored_sum_error < 0.05F && passed;
    std::cout << "q4_q8 zero-point algebra max_ideal_error=" << maximum_algebra_error
              << " max_stored_s_error=" << maximum_stored_sum_error
              << " result=" << (passed ? "PASS" : "FAIL")
              << '\n';
    return passed;
}

bool run_shape(const miinfer::GemvShape& shape, const std::string& implementation) {
    std::vector<__half> weights_fp16;
    std::vector<__half> input_fp16;
    miinfer::generate_fp16_gemv_data(shape.m, shape.k, kSeed, weights_fp16, input_fp16);
    const auto weights_q4 = miinfer::quantize_q4_0(weights_fp16, shape.m, shape.k);
    const auto input_q8 = miinfer::quantize_q8_1(input_fp16);
    const auto input_q8_exact = miinfer::quantize_q8_exact(input_fp16);
    const auto oracle = miinfer::q4_q8_cpu_reference(weights_q4, input_q8, shape.m, shape.k);

    miinfer::Q4_0Block* device_weights = nullptr;
    miinfer::Q8_1Block* device_input = nullptr;
    miinfer::Q8ExactBlock* device_input_exact = nullptr;
    __half* device_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_weights),
                                weights_q4.size() * sizeof(miinfer::Q4_0Block)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input),
                                input_q8.size() * sizeof(miinfer::Q8_1Block)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input_exact),
                                input_q8_exact.size() * sizeof(miinfer::Q8ExactBlock)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output),
                                static_cast<std::size_t>(shape.m) * sizeof(__half)));
    MIINFER_HIP_CHECK(hipMemcpy(device_weights, weights_q4.data(),
                                weights_q4.size() * sizeof(miinfer::Q4_0Block),
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(device_input, input_q8.data(),
                                input_q8.size() * sizeof(miinfer::Q8_1Block),
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(device_input_exact, input_q8_exact.data(),
                                input_q8_exact.size() * sizeof(miinfer::Q8ExactBlock),
                                hipMemcpyHostToDevice));
    if (implementation == "packed-dot") {
        miinfer::launch_q4_q8_gemv_packed_dot(
            device_weights, device_input, device_output, shape.m, shape.k);
    } else if (implementation == "zero-point-dot") {
        miinfer::launch_q4_q8_gemv_zero_point_dot(
            device_weights, device_input, device_output, shape.m, shape.k);
    } else if (implementation == "zero-point-128") {
        miinfer::launch_q4_q8_gemv_zero_point_dot_128(
            device_weights, device_input, device_output, shape.m, shape.k);
    } else if (implementation == "zero-point-wave64") {
        miinfer::launch_q4_q8_gemv_zero_point_dot_wave64(
            device_weights, device_input, device_output, shape.m, shape.k);
    } else if (implementation == "zero-point-four-wave64") {
        miinfer::launch_q4_q8_gemv_zero_point_dot_four_wave64(
            device_weights, device_input, device_output, shape.m, shape.k);
    } else if (implementation == "zero-point-four-wave64-exact-metadata") {
        miinfer::launch_q4_q8_gemv_zero_point_dot_four_wave64_exact_metadata(
            device_weights, device_input_exact, device_output, shape.m, shape.k);
    } else if (implementation == "zero-point-exact-metadata") {
        const bool use_wave64 = std::strcmp(shape.id, "k") == 0
                             || std::strcmp(shape.id, "v") == 0;
        if (use_wave64) {
            miinfer::launch_q4_q8_gemv_zero_point_dot_wave64_exact_metadata(
                device_weights, device_input_exact, device_output, shape.m, shape.k);
        } else {
            miinfer::launch_q4_q8_gemv_zero_point_dot_exact_metadata(
                device_weights, device_input_exact, device_output, shape.m, shape.k);
        }
    } else {
        miinfer::launch_q4_q8_gemv(device_weights, device_input, device_output, shape.m, shape.k);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<__half> output(static_cast<std::size_t>(shape.m));
    MIINFER_HIP_CHECK(hipMemcpy(output.data(), device_output,
                                output.size() * sizeof(__half), hipMemcpyDeviceToHost));
    const auto metrics = miinfer::evaluate_fp16_gemv(output, oracle);
    std::cout << "q4_q8 " << implementation << ' ' << shape.id
              << " max_abs=" << metrics.max_abs_error
              << " mean_abs=" << metrics.mean_abs_error
              << " cosine=" << metrics.cosine_similarity
              << " result=" << (metrics.pass ? "PASS" : "FAIL") << '\n';
    MIINFER_HIP_CHECK(hipFree(device_output));
    MIINFER_HIP_CHECK(hipFree(device_input_exact));
    MIINFER_HIP_CHECK(hipFree(device_input));
    MIINFER_HIP_CHECK(hipFree(device_weights));
    return metrics.pass;
}

}  // namespace

int main() {
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) {
        std::cerr << "Q4_0 x Q8_1 correctness unavailable: " << error << '\n';
        return 1;
    }
    bool passed = true;
    passed = run_zero_point_identity_tests() && passed;
    const std::vector<std::string> implementations = {
        "scalar", "packed-dot", "zero-point-dot", "zero-point-128", "zero-point-wave64",
        "zero-point-four-wave64", "zero-point-four-wave64-exact-metadata",
        "zero-point-exact-metadata"};
    const std::vector<miinfer::GemvShape> small_shapes = {
        {"small", "small Q4/Q8 indexing", 7, 32},
        {"tail", "small Q4/Q8 row tail", 257, 64},
    };
    for (const auto& shape : small_shapes) {
        for (const auto& implementation : implementations) {
            passed = run_shape(shape, implementation) && passed;
        }
    }
    for (const auto& shape : miinfer::qwen3_gemv_shapes()) {
        for (const auto& implementation : implementations) {
            passed = run_shape(shape, implementation) && passed;
        }
    }
    return passed ? 0 : 1;
}
