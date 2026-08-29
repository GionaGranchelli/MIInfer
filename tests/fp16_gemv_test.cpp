#include "miinfer/fp16_gemv.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/device_validation.hpp"

#include <hip/hip_runtime.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kTestSeed = 0x4D493050U;

bool check_output(
    const char* implementation,
    const miinfer::GemvShape& shape,
    const std::vector<__half>& output,
    const std::vector<float>& reference) {
    const auto metrics = miinfer::evaluate_fp16_gemv(output, reference);
    std::cout << implementation << ' ' << shape.id << " max_abs="
              << std::setprecision(8) << metrics.max_abs_error << " mean_abs="
              << metrics.mean_abs_error << " max_rel=" << metrics.max_relative_error
              << " cosine=" << metrics.cosine_similarity << " result="
              << (metrics.pass ? "PASS" : "FAIL") << '\n';
    return metrics.pass;
}

bool run_shape(const miinfer::GemvShape& shape) {
    std::vector<__half> weights;
    std::vector<__half> input;
    miinfer::generate_fp16_gemv_data(shape.m, shape.k, kTestSeed, weights, input);
    const auto reference = miinfer::fp16_gemv_cpu_reference(weights, input, shape.m, shape.k);

    __half* device_weights = nullptr;
    __half* device_input = nullptr;
    __half* device_output = nullptr;
    float* device_partials = nullptr;
    const std::size_t weight_bytes = weights.size() * sizeof(__half);
    const std::size_t input_bytes = input.size() * sizeof(__half);
    const std::size_t output_bytes = static_cast<std::size_t>(shape.m) * sizeof(__half);
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_weights), weight_bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input_bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), output_bytes));
    if (shape.k % 4 == 0) {
        MIINFER_HIP_CHECK(hipMalloc(
            reinterpret_cast<void**>(&device_partials),
            static_cast<std::size_t>(shape.m) * 4 * sizeof(float)));
    }
    MIINFER_HIP_CHECK(
        hipMemcpy(device_weights, weights.data(), weight_bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(device_input, input.data(), input_bytes, hipMemcpyHostToDevice));

    miinfer::launch_fp16_gemv_baseline(
        device_weights, device_input, device_output, shape.m, shape.k);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<__half> output(static_cast<std::size_t>(shape.m));
    MIINFER_HIP_CHECK(
        hipMemcpy(output.data(), device_output, output_bytes, hipMemcpyDeviceToHost));
    const bool miinfer_pass = check_output("miinfer-baseline", shape, output, reference);

    bool k_split_pass = true;
    if (device_partials != nullptr) {
        for (const int splits : {2, 4}) {
            miinfer::launch_fp16_gemv_k_split(
                device_weights, device_input, device_output, device_partials,
                shape.m, shape.k, splits);
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            MIINFER_HIP_CHECK(
                hipMemcpy(output.data(), device_output, output_bytes, hipMemcpyDeviceToHost));
            k_split_pass = check_output(
                              (std::string("miinfer-k-split-") + std::to_string(splits)).c_str(),
                              shape, output, reference)
                           && k_split_pass;
        }
    }

    miinfer::RocblasGemmHandle handle;
    std::string error;
    const bool handle_created =
        miinfer::create_rocblas_gemm_handle(handle, nullptr, error);
    bool rocblas_pass = false;
    if (handle_created
        && miinfer::launch_rocblas_gemm_fp16(
            handle, device_weights, device_input, device_output, shape.m, shape.k, error)) {
        MIINFER_HIP_CHECK(hipDeviceSynchronize());
        MIINFER_HIP_CHECK(
            hipMemcpy(output.data(), device_output, output_bytes, hipMemcpyDeviceToHost));
        rocblas_pass = check_output("rocblas-gemm", shape, output, reference);
    } else {
        std::cerr << "rocblas-gemm " << shape.id << " unavailable: " << error << '\n';
    }
    miinfer::destroy_rocblas_gemm_handle(handle);

    MIINFER_HIP_CHECK(hipFree(device_output));
    MIINFER_HIP_CHECK(hipFree(device_input));
    MIINFER_HIP_CHECK(hipFree(device_weights));
    if (device_partials != nullptr) {
        MIINFER_HIP_CHECK(hipFree(device_partials));
    }
    return miinfer_pass && k_split_pass && rocblas_pass;
}

}  // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::cerr << "usage: miinfer-fp16-gemv-test\n";
        return 2;
    }

    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) {
        std::cerr << "FP16 GEMV correctness test unavailable: " << error << '\n';
        return 1;
    }
    std::cout << "device: " << device.name << " (" << device.architecture << ")\n";

    bool passed = true;
    const std::vector<miinfer::GemvShape> small_shapes = {
        {"small-tail", "small indexing/tail test", 7, 13},
        {"m-tail", "non-multiple workgroup test", 257, 37},
        {"k-split-tail", "K-split indexing/tail test", 11, 20},
    };
    for (const auto& shape : small_shapes) {
        passed = run_shape(shape) && passed;
    }
    for (const auto& shape : miinfer::qwen3_gemv_shapes()) {
        passed = run_shape(shape) && passed;
    }
    return passed ? 0 : 1;
}
