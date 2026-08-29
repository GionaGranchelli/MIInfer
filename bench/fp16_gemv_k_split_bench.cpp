#include "miinfer/build_config.hpp"
#include "miinfer/device_validation.hpp"
#include "miinfer/fp16_gemv.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kSeed = 0x4D493050U;

struct Options {
    std::string shape = "all";
    int workgroups_per_row = 1;
    int device = -1;
    int warmup = 5;
    int iterations = 1000;
    std::uint32_t seed = kSeed;
    std::string json_output;
};

void usage() {
    std::cerr << "usage: miinfer-fp16-gemv-k-split-bench [options]\n"
              << "  --shape k|q|all       real K/V and Q shapes (default: all)\n"
              << "  --splits 1|2|4        workgroups per output row (default: 1)\n"
              << "  --warmup N             warm-up logical GEMVs (default: 5)\n"
              << "  --iterations N         measured logical GEMVs (default: 1000)\n"
              << "  --device INDEX         select a HIP device (default: first gfx906)\n"
              << "  --json-output PATH     write JSONL output to PATH\n";
}

bool positive(const char* text, int& value) {
    try {
        value = std::stoi(text);
    } catch (...) {
        return false;
    }
    return value > 0;
}

bool nonnegative(const char* text, int& value) {
    try {
        value = std::stoi(text);
    } catch (...) {
        return false;
    }
    return value >= 0;
}

bool parse(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            usage();
            return false;
        }
        if (index + 1 >= argc) {
            std::cerr << "missing value for " << argument << '\n';
            return false;
        }
        if (argument == "--shape") {
            options.shape = argv[++index];
        } else if (argument == "--splits") {
            if (!positive(argv[++index], options.workgroups_per_row)) {
                std::cerr << "--splits must be a positive integer\n";
                return false;
            }
        } else if (argument == "--warmup") {
            if (!nonnegative(argv[++index], options.warmup)) {
                std::cerr << "--warmup must be non-negative\n";
                return false;
            }
        } else if (argument == "--iterations") {
            if (!positive(argv[++index], options.iterations)) {
                std::cerr << "--iterations must be positive\n";
                return false;
            }
        } else if (argument == "--device") {
            if (!nonnegative(argv[++index], options.device)) {
                std::cerr << "--device must be non-negative\n";
                return false;
            }
        } else if (argument == "--json-output") {
            options.json_output = argv[++index];
        } else {
            std::cerr << "unknown option: " << argument << '\n';
            return false;
        }
    }
    if ((options.shape != "k" && options.shape != "q" && options.shape != "all")
        || (options.workgroups_per_row != 1 && options.workgroups_per_row != 2
            && options.workgroups_per_row != 4)) {
        std::cerr << "shape must be k, q, or all; splits must be 1, 2, or 4\n";
        return false;
    }
    return true;
}

std::string escape(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (const char character : value) {
        if (character == '"' || character == '\\') {
            result << '\\';
        }
        result << character;
    }
    result << '"';
    return result.str();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
               ? (values[middle - 1] + values[middle]) / 2.0
               : values[middle];
}

bool run_shape(
    const Options& options,
    const miinfer::DeviceInfo& device,
    const miinfer::GemvShape& shape,
    std::ostream& output) {
    std::vector<__half> weights;
    std::vector<__half> input;
    miinfer::generate_fp16_gemv_data(shape.m, shape.k, options.seed, weights, input);
    const auto reference = miinfer::fp16_gemv_cpu_reference(weights, input, shape.m, shape.k);

    const int buffer_count = 3;
    std::vector<__half*> device_weights(buffer_count, nullptr);
    __half* device_input = nullptr;
    __half* device_output = nullptr;
    float* device_partials = nullptr;
    const std::size_t weight_bytes = weights.size() * sizeof(__half);
    const std::size_t input_bytes = input.size() * sizeof(__half);
    const std::size_t output_bytes = static_cast<std::size_t>(shape.m) * sizeof(__half);
    const std::size_t partial_bytes = static_cast<std::size_t>(shape.m)
                                      * static_cast<std::size_t>(options.workgroups_per_row)
                                      * sizeof(float);
    for (auto& pointer : device_weights) {
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&pointer), weight_bytes));
        MIINFER_HIP_CHECK(hipMemcpy(pointer, weights.data(), weight_bytes,
                                    hipMemcpyHostToDevice));
    }
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input_bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), output_bytes));
    MIINFER_HIP_CHECK(hipMemcpy(device_input, input.data(), input_bytes,
                                hipMemcpyHostToDevice));
    if (options.workgroups_per_row > 1) {
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_partials), partial_bytes));
    }

    const auto launch = [&](const __half* weights_pointer) {
        if (options.workgroups_per_row == 1) {
            miinfer::launch_fp16_gemv_baseline(
                weights_pointer, device_input, device_output, shape.m, shape.k);
        } else {
            miinfer::launch_fp16_gemv_k_split(
                weights_pointer, device_input, device_output, device_partials,
                shape.m, shape.k, options.workgroups_per_row);
        }
    };

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        launch(device_weights[static_cast<std::size_t>(iteration) % device_weights.size()]);
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
        launch(device_weights[static_cast<std::size_t>(iteration) % device_weights.size()]);
        MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }

    std::vector<__half> output_values(static_cast<std::size_t>(shape.m));
    MIINFER_HIP_CHECK(hipMemcpy(output_values.data(), device_output, output_bytes,
                                hipMemcpyDeviceToHost));
    const auto correctness = miinfer::evaluate_fp16_gemv(output_values, reference);
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double med = median(samples);
    const double variance = std::accumulate(
                                samples.begin(), samples.end(), 0.0,
                                [mean](double total, double sample) {
                                    const double difference = sample - mean;
                                    return total + difference * difference;
                                })
                            / samples.size();
    const double logical_bytes = static_cast<double>(shape.m) * shape.k * sizeof(__half)
                                 + static_cast<double>(shape.k + shape.m) * sizeof(__half);
    const double effective_gb_per_second = logical_bytes / (med * 1000.0);
    const std::string implementation = options.workgroups_per_row == 1
                                            ? "miinfer-baseline"
                                            : "miinfer-k-split-"
                                                  + std::to_string(options.workgroups_per_row);

    output << std::fixed << std::setprecision(9)
           << "{\"experiment\":\"EXP-0004\",\"shape\":\"" << shape.id
           << "\",\"projection\":" << escape(shape.projection)
           << ",\"m\":" << shape.m << ",\"k\":" << shape.k
           << ",\"implementation\":" << escape(implementation)
           << ",\"workgroups_per_row\":" << options.workgroups_per_row
           << ",\"partial_workgroups\":" << shape.m * options.workgroups_per_row
           << ",\"final_workgroups\":" << (shape.m + 255) / 256
           << ",\"input_dtype\":\"fp16\",\"weight_dtype\":\"fp16\""
           << ",\"accumulator_dtype\":\"fp32\",\"output_dtype\":\"fp16\""
           << ",\"seed\":" << options.seed
           << ",\"cache_regime\":\"streaming-rotate-3\""
           << ",\"warmup_iterations\":" << options.warmup
           << ",\"measured_iterations\":" << options.iterations
           << ",\"mean_us\":" << mean
           << ",\"median_us\":" << med
           << ",\"min_us\":" << *std::min_element(samples.begin(), samples.end())
           << ",\"max_us\":" << *std::max_element(samples.begin(), samples.end())
           << ",\"stddev_us\":" << std::sqrt(variance)
           << ",\"effective_gb_per_second\":" << effective_gb_per_second
           << ",\"logical_bytes\":" << logical_bytes
           << ",\"gpu\":" << escape(device.name)
           << ",\"gfx\":" << escape(device.architecture)
           << ",\"vram_bytes\":" << device.total_vram_bytes
           << ",\"git_commit\":" << escape(MIINFER_GIT_COMMIT)
           << ",\"git_dirty\":" << escape(MIINFER_GIT_DIRTY)
           << ",\"correctness\":{\"pass\":"
           << (correctness.pass ? "true" : "false")
           << ",\"max_abs_error\":" << correctness.max_abs_error
           << ",\"mean_abs_error\":" << correctness.mean_abs_error
           << ",\"max_relative_error\":" << correctness.max_relative_error
           << ",\"cosine_similarity\":" << correctness.cosine_similarity
           << ",\"nan_detected\":" << (correctness.nan_detected ? "true" : "false")
           << ",\"inf_detected\":" << (correctness.inf_detected ? "true" : "false")
           << "}}\n";

    MIINFER_HIP_CHECK(hipEventDestroy(stop));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    if (device_partials != nullptr) {
        MIINFER_HIP_CHECK(hipFree(device_partials));
    }
    MIINFER_HIP_CHECK(hipFree(device_output));
    MIINFER_HIP_CHECK(hipFree(device_input));
    for (auto pointer : device_weights) {
        MIINFER_HIP_CHECK(hipFree(pointer));
    }
    return correctness.pass;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse(argc, argv, options)) {
        return argc > 1 && std::string(argv[1]) == "--help" ? 0 : 2;
    }

    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(options.device, device, error)) {
        std::cerr << "K-split benchmark unavailable: " << error << '\n';
        return 1;
    }

    std::ofstream file;
    std::ostream* output = &std::cout;
    if (!options.json_output.empty()) {
        file.open(options.json_output);
        if (!file) {
            std::cerr << "could not open JSON output: " << options.json_output << '\n';
            return 1;
        }
        output = &file;
    }

    const auto& shapes = miinfer::qwen3_gemv_shapes();
    bool passed = true;
    for (const auto& shape : shapes) {
        if (options.shape == "all"
            || (options.shape == "k" && (std::string(shape.id) == "K"
                                           || std::string(shape.id) == "V"))
            || (options.shape == "q" && std::string(shape.id) == "Q")) {
            passed = run_shape(options, device, shape, *output) && passed;
        }
    }
    return passed ? 0 : 1;
}
