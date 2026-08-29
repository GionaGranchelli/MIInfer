#include "miinfer/build_config.hpp"
#include "miinfer/device_validation.hpp"
#include "miinfer/fp16_gemv.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Options {
    int m = 0;
    int k = 0;
    int warmup = 10;
    int iterations = 1000;
    int device = -1;
};

bool positive(const char* text, int& value) {
    try { value = std::stoi(text); } catch (...) { return false; }
    return value > 0;
}

bool nonnegative(const char* text, int& value) {
    try { value = std::stoi(text); } catch (...) { return false; }
    return value >= 0;
}

bool parse(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return false;
        const std::string arg = argv[i];
        if (arg == "--m") {
            if (!positive(argv[++i], options.m)) return false;
        } else if (arg == "--k") {
            if (!positive(argv[++i], options.k)) return false;
        } else if (arg == "--warmup") {
            if (!nonnegative(argv[++i], options.warmup)) return false;
        } else if (arg == "--iterations") {
            if (!positive(argv[++i], options.iterations)) return false;
        } else if (arg == "--device") {
            if (!nonnegative(argv[++i], options.device)) return false;
        } else {
            return false;
        }
    }
    return options.m > 0 && options.k > 0;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
}

double time_full(
    const __half* weights,
    const __half* input,
    __half* output,
    int m,
    int k,
    int iterations,
    int warmup) {
    for (int i = 0; i < warmup; ++i) {
        miinfer::launch_fp16_gemv_baseline(weights, input, output, m, k);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        MIINFER_HIP_CHECK(hipEventRecord(start, nullptr));
        miinfer::launch_fp16_gemv_baseline(weights, input, output, m, k);
        MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    MIINFER_HIP_CHECK(hipEventDestroy(stop));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    return median(samples);
}

double time_dot_only(
    const __half* weights,
    const __half* input,
    float* partials,
    int m,
    int k,
    int iterations,
    int warmup) {
    for (int i = 0; i < warmup; ++i) {
        miinfer::launch_fp16_gemv_diagnostic_dot_only(weights, input, partials, m, k);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        MIINFER_HIP_CHECK(hipEventRecord(start, nullptr));
        miinfer::launch_fp16_gemv_diagnostic_dot_only(weights, input, partials, m, k);
        MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    MIINFER_HIP_CHECK(hipEventDestroy(stop));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    return median(samples);
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse(argc, argv, options)) {
        std::cerr << "usage: miinfer-fp16-gemv-reduction-diag --m N --k N [--warmup N] [--iterations N]\n";
        return 2;
    }
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(options.device, device, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::vector<__half> weights;
    std::vector<__half> input;
    miinfer::generate_fp16_gemv_data(options.m, options.k, 0x4D493050U, weights, input);
    __half* device_weights = nullptr;
    __half* device_input = nullptr;
    __half* device_output = nullptr;
    float* device_partials = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_weights), weights.size() * sizeof(__half)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(__half)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), static_cast<std::size_t>(options.m) * sizeof(__half)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_partials), static_cast<std::size_t>(options.m) * 256 * sizeof(float)));
    MIINFER_HIP_CHECK(hipMemcpy(device_weights, weights.data(), weights.size() * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(device_input, input.data(), input.size() * sizeof(__half), hipMemcpyHostToDevice));
    const double full_us = time_full(device_weights, device_input, device_output, options.m, options.k, options.iterations, options.warmup);
    const double dot_us = time_dot_only(device_weights, device_input, device_partials, options.m, options.k, options.iterations, options.warmup);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"experiment\":\"EXP-0003\",\"diagnostic\":\"reduction-overhead\","
              << "\"classification\":\"DIAGNOSTIC — NOT A PERFORMANCE CANDIDATE\","
              << "\"m\":" << options.m << ",\"k\":" << options.k
              << ",\"workgroups\":" << options.m
              << ",\"full_baseline_median_us\":" << full_us
              << ",\"dot_only_median_us\":" << dot_us
              << ",\"difference_us\":" << (full_us - dot_us)
              << ",\"difference_percent_of_full\":" << ((full_us - dot_us) / full_us * 100.0)
              << ",\"gpu\":\"" << device.name << "\",\"gfx\":\"" << device.architecture << "\"}\n";
    MIINFER_HIP_CHECK(hipFree(device_partials));
    MIINFER_HIP_CHECK(hipFree(device_output));
    MIINFER_HIP_CHECK(hipFree(device_input));
    MIINFER_HIP_CHECK(hipFree(device_weights));
    return 0;
}
