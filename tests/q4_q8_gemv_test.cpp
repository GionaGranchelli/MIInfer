#include "miinfer/device_validation.hpp"
#include "miinfer/fp16_gemv.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/q4_q8_gemv.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kSeed = 0x4D493050U;

bool run_shape(const miinfer::GemvShape& shape) {
    std::vector<__half> weights_fp16;
    std::vector<__half> input_fp16;
    miinfer::generate_fp16_gemv_data(shape.m, shape.k, kSeed, weights_fp16, input_fp16);
    const auto weights_q4 = miinfer::quantize_q4_0(weights_fp16, shape.m, shape.k);
    const auto input_q8 = miinfer::quantize_q8_1(input_fp16);
    const auto oracle = miinfer::q4_q8_cpu_reference(weights_q4, input_q8, shape.m, shape.k);

    miinfer::Q4_0Block* device_weights = nullptr;
    miinfer::Q8_1Block* device_input = nullptr;
    __half* device_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_weights),
                                weights_q4.size() * sizeof(miinfer::Q4_0Block)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input),
                                input_q8.size() * sizeof(miinfer::Q8_1Block)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output),
                                static_cast<std::size_t>(shape.m) * sizeof(__half)));
    MIINFER_HIP_CHECK(hipMemcpy(device_weights, weights_q4.data(),
                                weights_q4.size() * sizeof(miinfer::Q4_0Block),
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(device_input, input_q8.data(),
                                input_q8.size() * sizeof(miinfer::Q8_1Block),
                                hipMemcpyHostToDevice));
    miinfer::launch_q4_q8_gemv(device_weights, device_input, device_output, shape.m, shape.k);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<__half> output(static_cast<std::size_t>(shape.m));
    MIINFER_HIP_CHECK(hipMemcpy(output.data(), device_output,
                                output.size() * sizeof(__half), hipMemcpyDeviceToHost));
    const auto metrics = miinfer::evaluate_fp16_gemv(output, oracle);
    std::cout << "q4_q8 " << shape.id << " max_abs=" << metrics.max_abs_error
              << " mean_abs=" << metrics.mean_abs_error
              << " cosine=" << metrics.cosine_similarity
              << " result=" << (metrics.pass ? "PASS" : "FAIL") << '\n';
    MIINFER_HIP_CHECK(hipFree(device_output));
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
    const std::vector<miinfer::GemvShape> small_shapes = {
        {"small", "small Q4/Q8 indexing", 7, 32},
        {"tail", "small Q4/Q8 row tail", 257, 64},
    };
    for (const auto& shape : small_shapes) {
        passed = run_shape(shape) && passed;
    }
    for (const auto& shape : miinfer::qwen3_gemv_shapes()) {
        passed = run_shape(shape) && passed;
    }
    return passed ? 0 : 1;
}
