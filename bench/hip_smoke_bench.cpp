#include "miinfer/build_info.hpp"
#include "miinfer/build_config.hpp"
#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/vector_add.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    int device = -1;
    int warmup = 5;
    int iterations = 100;
    std::size_t elements = 1U << 20;
    std::string json_output;
};

void print_usage() {
    std::cerr << "usage: miinfer-bench [options]\n"
              << "  --warmup N       warm-up launches (default: 5)\n"
              << "  --iterations N   measured launches (default: 100)\n"
              << "  --elements N     vector elements (default: 1048576)\n"
              << "  --device INDEX   select a HIP device (default: first gfx906)\n"
              << "  --json-output P  write JSON result to P\n"
              << "  --version        print build information\n";
}

bool parse_positive(const char* text, int& value) {
    try {
        value = std::stoi(text);
    } catch (...) {
        return false;
    }
    return value > 0;
}

bool parse_nonnegative(const char* text, int& value) {
    try {
        value = std::stoi(text);
    } catch (...) {
        return false;
    }
    return value >= 0;
}

bool parse_size(const char* text, std::size_t& value) {
    try {
        value = static_cast<std::size_t>(std::stoull(text));
    } catch (...) {
        return false;
    }
    return value > 0;
}

bool parse_options(int argc, char** argv, Options& options, bool& informational) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--version") {
            miinfer::print_build_info(std::cout);
            informational = true;
            return false;
        }
        if (argument == "--help") {
            print_usage();
            informational = true;
            return false;
        }
        if (index + 1 >= argc) {
            std::cerr << "missing value for " << argument << '\n';
            return false;
        }
        if (argument == "--warmup" && !parse_nonnegative(argv[++index], options.warmup)) {
            std::cerr << "--warmup must be a non-negative integer\n";
            return false;
        }
        if (argument == "--iterations" && !parse_positive(argv[++index], options.iterations)) {
            std::cerr << "--iterations must be a positive integer\n";
            return false;
        }
        if (argument == "--elements" && !parse_size(argv[++index], options.elements)) {
            std::cerr << "--elements must be a positive integer\n";
            return false;
        }
        if (argument == "--device" && !parse_nonnegative(argv[++index], options.device)) {
            std::cerr << "--device must be a non-negative integer\n";
            return false;
        }
        if (argument == "--json-output") {
            options.json_output = argv[++index];
        }
        if (argument != "--warmup" && argument != "--iterations" && argument != "--elements"
            && argument != "--device" && argument != "--json-output") {
            std::cerr << "unknown option: " << argument << '\n';
            return false;
        }
    }
    return true;
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    escaped << '"';
    for (const char character : value) {
        switch (character) {
        case '"': escaped << "\\\""; break;
        case '\\': escaped << "\\\\"; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
            if (static_cast<unsigned char>(character) < 0x20U) {
                escaped << ' ';
            } else {
                escaped << character;
            }
        }
    }
    escaped << '"';
    return escaped.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    bool informational = false;
    if (!parse_options(argc, argv, options, informational)) {
        return informational ? 0 : 2;
    }

    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(options.device, device, error)) {
        std::cerr << "MIInfer benchmark unavailable: " << error << '\n';
        return 1;
    }

    float* device_left = nullptr;
    float* device_right = nullptr;
    float* device_result = nullptr;
    const std::size_t bytes = options.elements * sizeof(float);
    MIINFER_HIP_CHECK(hipMalloc(&device_left, bytes));
    MIINFER_HIP_CHECK(hipMalloc(&device_right, bytes));
    MIINFER_HIP_CHECK(hipMalloc(&device_result, bytes));

    std::vector<float> host_left(options.elements, 1.25F);
    std::vector<float> host_right(options.elements, 2.5F);
    std::vector<float> host_result(options.elements, 0.0F);
    MIINFER_HIP_CHECK(hipMemcpy(device_left, host_left.data(), bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(device_right, host_right.data(), bytes, hipMemcpyHostToDevice));

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        miinfer::launch_vector_add(device_left, device_right, device_result, options.elements);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.iterations));
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        MIINFER_HIP_CHECK(hipEventRecord(start, nullptr));
        miinfer::launch_vector_add(device_left, device_right, device_result, options.elements);
        MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }

    MIINFER_HIP_CHECK(hipMemcpy(host_result.data(), device_result, bytes, hipMemcpyDeviceToHost));
    const bool correct = std::all_of(host_result.begin(), host_result.end(), [](float value) {
        return std::isfinite(value) && std::fabs(value - 3.75F) <= 1.0e-6F;
    });
    MIINFER_HIP_CHECK(hipEventDestroy(stop));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipFree(device_result));
    MIINFER_HIP_CHECK(hipFree(device_right));
    MIINFER_HIP_CHECK(hipFree(device_left));
    if (!correct) {
        std::cerr << "MIInfer benchmark correctness check failed; timing is invalid\n";
        return 1;
    }

    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.size() % 2 == 0
                              ? (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0
                              : sorted[sorted.size() / 2];
    const double variance = std::accumulate(samples.begin(), samples.end(), 0.0,
        [mean](double total, double sample) {
            const double difference = sample - mean;
            return total + difference * difference;
        }) / samples.size();
    const double standard_deviation = std::sqrt(variance);
    const double minimum = *std::min_element(samples.begin(), samples.end());
    const double maximum = *std::max_element(samples.begin(), samples.end());

    std::ostringstream json;
    json << std::fixed << std::setprecision(6)
         << "{\n"
         << "  \"benchmark\": \"hip_smoke\",\n"
         << "  \"version\": " << json_escape(MIINFER_VERSION) << ",\n"
         << "  \"git_commit\": " << json_escape(MIINFER_GIT_COMMIT) << ",\n"
         << "  \"git_dirty\": " << json_escape(MIINFER_GIT_DIRTY) << ",\n"
         << "  \"build_type\": " << json_escape(MIINFER_BUILD_TYPE) << ",\n"
         << "  \"compiler\": " << json_escape(MIINFER_CXX_COMPILER) << ",\n"
         << "  \"hip_compiler\": " << json_escape(MIINFER_HIP_COMPILER) << ",\n"
         << "  \"hip_available\": " << json_escape(MIINFER_HIP_AVAILABLE) << ",\n"
         << "  \"gpu\": " << json_escape(device.name) << ",\n"
         << "  \"gfx\": " << json_escape(device.architecture) << ",\n"
         << "  \"vram_bytes\": " << device.total_vram_bytes << ",\n"
         << "  \"elements\": " << options.elements << ",\n"
         << "  \"warmup_iterations\": " << options.warmup << ",\n"
         << "  \"measured_iterations\": " << options.iterations << ",\n"
         << "  \"mean_us\": " << mean << ",\n"
         << "  \"median_us\": " << median << ",\n"
         << "  \"min_us\": " << minimum << ",\n"
         << "  \"max_us\": " << maximum << ",\n"
         << "  \"stddev_us\": " << standard_deviation << ",\n"
         << "  \"samples_us\": [";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) {
            json << ", ";
        }
        json << samples[index];
    }
    json << "],\n"
         << "  \"correctness\": \"PASS\"\n"
         << "}\n";
    if (options.json_output.empty()) {
        std::cout << json.str();
    } else {
        std::ofstream output(options.json_output);
        if (!output) {
            std::cerr << "could not open JSON output: " << options.json_output << '\n';
            return 1;
        }
        output << json.str();
    }
    std::cerr << "hip_smoke: " << options.iterations << " HIP-event samples, median " << median
              << " us\n";
    return 0;
}
