#include "miinfer/build_config.hpp"
#include "miinfer/build_info.hpp"
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

constexpr std::uint32_t kDefaultSeed = 0x4D493050U;

struct Options {
    std::string experiment = "EXP-0002";
    std::string shape = "all";
    std::string implementation = "all";
    std::string cache_regime = "streaming";
    std::string custom_label = "synthetic";
    int custom_m = 0;
    int custom_k = 0;
    int device = -1;
    int warmup = 5;
    int iterations = 1000;
    std::uint32_t seed = kDefaultSeed;
    std::string json_output;
};

void print_usage() {
    std::cerr << "usage: miinfer-fp16-gemv-bench [options]\n"
              << "  --experiment ID         experiment label (default: EXP-0002)\n"
              << "  --shape ID             q|k|v|o|gate|up|down|all (default: all)\n"
              << "  --m N --k N             synthetic shape dimensions (requires both)\n"
              << "  --label NAME            synthetic shape label (default: synthetic)\n"
              << "  --implementation NAME  miinfer|rocblas-gemm|all (default: all)\n"
              << "  --cache-regime NAME    hot|streaming (default: streaming)\n"
              << "  --warmup N             warm-up launches (default: 5)\n"
              << "  --iterations N         measured launches (default: 1000)\n"
              << "  --seed N               deterministic data seed (default: 1296642128)\n"
              << "  --device INDEX        select a HIP device (default: first gfx906)\n"
              << "  --json-output PATH     write JSONL results to PATH\n"
              << "  --version              print build information\n";
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

bool parse_seed(const char* text, std::uint32_t& value) {
    try {
        const auto parsed = std::stoull(text, nullptr, 0);
        if (parsed > 0xFFFFFFFFULL) {
            return false;
        }
        value = static_cast<std::uint32_t>(parsed);
    } catch (...) {
        return false;
    }
    return true;
}

bool valid_shape_name(const std::string& name) {
    return name == "q" || name == "k" || name == "v" || name == "o"
           || name == "gate" || name == "up" || name == "down" || name == "all";
}

bool valid_implementation_name(const std::string& name) {
    return name == "miinfer" || name == "rocblas-gemm" || name == "all";
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
        if (argument == "--shape") {
            options.shape = argv[++index];
        } else if (argument == "--experiment") {
            options.experiment = argv[++index];
        } else if (argument == "--m") {
            if (!parse_positive(argv[++index], options.custom_m)) {
                std::cerr << "--m must be a positive integer\n";
                return false;
            }
        } else if (argument == "--k") {
            if (!parse_positive(argv[++index], options.custom_k)) {
                std::cerr << "--k must be a positive integer\n";
                return false;
            }
        } else if (argument == "--label") {
            options.custom_label = argv[++index];
        } else if (argument == "--implementation") {
            options.implementation = argv[++index];
        } else if (argument == "--cache-regime") {
            options.cache_regime = argv[++index];
        } else if (argument == "--warmup") {
            if (!parse_nonnegative(argv[++index], options.warmup)) {
                std::cerr << "--warmup must be a non-negative integer\n";
                return false;
            }
        } else if (argument == "--iterations") {
            if (!parse_positive(argv[++index], options.iterations)) {
                std::cerr << "--iterations must be a positive integer\n";
                return false;
            }
        } else if (argument == "--seed") {
            if (!parse_seed(argv[++index], options.seed)) {
                std::cerr << "--seed must be a uint32 integer\n";
                return false;
            }
        } else if (argument == "--device") {
            if (!parse_nonnegative(argv[++index], options.device)) {
                std::cerr << "--device must be a non-negative integer\n";
                return false;
            }
        } else if (argument == "--json-output") {
            options.json_output = argv[++index];
        } else {
            std::cerr << "unknown option: " << argument << '\n';
            return false;
        }
    }
    if (!valid_shape_name(options.shape)) {
        std::cerr << "invalid --shape: " << options.shape << '\n';
        return false;
    }
    if ((options.custom_m == 0) != (options.custom_k == 0)) {
        std::cerr << "--m and --k must be supplied together\n";
        return false;
    }
    if (options.custom_m > 0 && options.shape != "all") {
        std::cerr << "--m/--k cannot be combined with a named --shape\n";
        return false;
    }
    if (!valid_implementation_name(options.implementation)) {
        std::cerr << "invalid --implementation: " << options.implementation << '\n';
        return false;
    }
    if (options.cache_regime != "hot" && options.cache_regime != "streaming") {
        std::cerr << "invalid --cache-regime: " << options.cache_regime << '\n';
        return false;
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
        default: escaped << character; break;
        }
    }
    escaped << '"';
    return escaped.str();
}

std::vector<miinfer::GemvShape> selected_shapes(const Options& options) {
    if (options.custom_m > 0) {
        return {{options.custom_label.c_str(), "synthetic diagnostic", options.custom_m,
                 options.custom_k}};
    }
    const auto& name = options.shape;
    if (name == "all") {
        return miinfer::qwen3_gemv_shapes();
    }
    for (const auto& shape : miinfer::qwen3_gemv_shapes()) {
        if ((name == "q" && std::string(shape.id) == "Q")
            || (name == "k" && std::string(shape.id) == "K")
            || (name == "v" && std::string(shape.id) == "V")
            || (name == "o" && std::string(shape.id) == "O")
            || (name == "gate" && std::string(shape.id) == "G")
            || (name == "up" && std::string(shape.id) == "U")
            || (name == "down" && std::string(shape.id) == "D")) {
            return {shape};
        }
    }
    return {};
}

std::string implementation_label(const std::string& name) {
    return name == "miinfer" ? "miinfer-baseline" : "rocblas-gemm";
}

std::vector<std::string> selected_implementations(const std::string& name) {
    if (name == "all") {
        return {"miinfer", "rocblas-gemm"};
    }
    return {name};
}

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
               ? (values[middle - 1] + values[middle]) / 2.0
               : values[middle];
}

std::string metrics_json(const miinfer::Fp16GemvMetrics& metrics) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(9)
           << "{\"pass\":" << (metrics.pass ? "true" : "false")
           << ",\"max_abs_error\":" << metrics.max_abs_error
           << ",\"mean_abs_error\":" << metrics.mean_abs_error
           << ",\"max_relative_error\":" << metrics.max_relative_error
           << ",\"cosine_similarity\":" << metrics.cosine_similarity
           << ",\"nan_detected\":" << (metrics.nan_detected ? "true" : "false")
           << ",\"inf_detected\":" << (metrics.inf_detected ? "true" : "false") << '}';
    return output.str();
}

bool run_one(
    const Options& options,
    const miinfer::DeviceInfo& device,
    const miinfer::GemvShape& shape,
    const std::string& implementation,
    std::ostream& output) {
    std::vector<__half> weights;
    std::vector<__half> input;
    miinfer::generate_fp16_gemv_data(shape.m, shape.k, options.seed, weights, input);
    const auto reference = miinfer::fp16_gemv_cpu_reference(weights, input, shape.m, shape.k);

    const int weight_buffer_count = options.cache_regime == "streaming" ? 3 : 1;
    std::vector<__half*> device_weights(static_cast<std::size_t>(weight_buffer_count), nullptr);
    __half* device_input = nullptr;
    __half* device_output = nullptr;
    const std::size_t weight_bytes = weights.size() * sizeof(__half);
    const std::size_t input_bytes = input.size() * sizeof(__half);
    const std::size_t output_bytes = static_cast<std::size_t>(shape.m) * sizeof(__half);
    for (auto& pointer : device_weights) {
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&pointer), weight_bytes));
        MIINFER_HIP_CHECK(
            hipMemcpy(pointer, weights.data(), weight_bytes, hipMemcpyHostToDevice));
    }
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input_bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), output_bytes));
    MIINFER_HIP_CHECK(hipMemcpy(device_input, input.data(), input_bytes, hipMemcpyHostToDevice));

    miinfer::RocblasGemmHandle rocblas_handle;
    std::string error;
    if (implementation == "rocblas-gemm"
        && !miinfer::create_rocblas_gemm_handle(rocblas_handle, nullptr, error)) {
        std::cerr << "cannot initialize rocblas-gemm for " << shape.id << ": " << error << '\n';
        return false;
    }

    const auto launch = [&](const __half* weights_pointer) {
        if (implementation == "miinfer") {
            miinfer::launch_fp16_gemv_baseline(
                weights_pointer, device_input, device_output, shape.m, shape.k);
            return true;
        }
        return miinfer::launch_rocblas_gemm_fp16(
            rocblas_handle, weights_pointer, device_input, device_output, shape.m, shape.k, error);
    };

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        if (!launch(device_weights[static_cast<std::size_t>(iteration) % device_weights.size()])) {
            std::cerr << "launch failed during warmup for " << shape.id << ": " << error << '\n';
            return false;
        }
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
        if (!launch(device_weights[static_cast<std::size_t>(iteration) % device_weights.size()])) {
            std::cerr << "launch failed for " << shape.id << ": " << error << '\n';
            return false;
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }

    std::vector<__half> output_values(static_cast<std::size_t>(shape.m));
    MIINFER_HIP_CHECK(
        hipMemcpy(output_values.data(), device_output, output_bytes, hipMemcpyDeviceToHost));
    const auto correctness = miinfer::evaluate_fp16_gemv(output_values, reference);
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double median = median_of(samples);
    const double variance = std::accumulate(
                                samples.begin(), samples.end(), 0.0,
                                [mean](double total, double sample) {
                                    const double difference = sample - mean;
                                    return total + difference * difference;
                                })
                            / samples.size();
    const double logical_bytes = static_cast<double>(shape.m) * shape.k * sizeof(__half)
                                 + static_cast<double>(shape.k + shape.m) * sizeof(__half);
    const double effective_gb_per_second = logical_bytes / (median * 1000.0);

    output << std::fixed << std::setprecision(9)
           << "{\"experiment\":" << json_escape(options.experiment)
           << ",\"shape\":\"" << shape.id
           << "\",\"projection\":" << json_escape(shape.projection)
           << ",\"m\":" << shape.m << ",\"k\":" << shape.k
           << ",\"implementation\":\"" << implementation_label(implementation)
           << "\",\"input_dtype\":\"fp16\",\"weight_dtype\":\"fp16\""
           << ",\"accumulator_dtype\":\"fp32\",\"output_dtype\":\"fp16\""
           << ",\"seed\":" << options.seed
           << ",\"cache_regime\":\""
           << (options.cache_regime == "hot" ? "hot" : "streaming-rotate-3") << '"'
           << ",\"warmup_iterations\":" << options.warmup
           << ",\"measured_iterations\":" << options.iterations
           << ",\"mean_us\":" << mean
           << ",\"median_us\":" << median
           << ",\"min_us\":" << *std::min_element(samples.begin(), samples.end())
           << ",\"max_us\":" << *std::max_element(samples.begin(), samples.end())
           << ",\"stddev_us\":" << std::sqrt(variance)
           << ",\"effective_gb_per_second\":" << effective_gb_per_second
           << ",\"logical_bytes\":" << logical_bytes
           << ",\"workgroups\":" << shape.m
           << ",\"shape_classification\":"
           << json_escape(options.custom_m > 0 ? "SYNTHETIC — SCALING DIAGNOSTIC" : "Qwen3-8B real shape")
           << ",\"gpu\":" << json_escape(device.name)
           << ",\"gfx\":" << json_escape(device.architecture)
           << ",\"vram_bytes\":" << device.total_vram_bytes
           << ",\"git_commit\":" << json_escape(MIINFER_GIT_COMMIT)
           << ",\"git_dirty\":" << json_escape(MIINFER_GIT_DIRTY)
           << ",\"build_type\":" << json_escape(MIINFER_BUILD_TYPE)
           << ",\"compiler\":" << json_escape(MIINFER_CXX_COMPILER)
           << ",\"hip_compiler\":" << json_escape(MIINFER_HIP_COMPILER)
           << ",\"rocm_library\":\"hipBLAS/rocBLAS\""
           << ",\"correctness\":" << metrics_json(correctness);
    if (implementation == "miinfer") {
        const auto resources = miinfer::fp16_gemv_baseline_resources();
        output << ",\"resources\":{\"registers\":" << resources.registers
               << ",\"shared_bytes\":" << resources.shared_bytes
               << ",\"local_bytes\":" << resources.local_bytes
               << ",\"max_threads_per_block\":" << resources.max_threads_per_block << '}';
    } else {
        output << ",\"resources\":null";
    }
    output << "}\n";

    MIINFER_HIP_CHECK(hipEventDestroy(stop));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    miinfer::destroy_rocblas_gemm_handle(rocblas_handle);
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
    bool informational = false;
    if (!parse_options(argc, argv, options, informational)) {
        return informational ? 0 : 2;
    }

    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(options.device, device, error)) {
        std::cerr << "FP16 GEMV benchmark unavailable: " << error << '\n';
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

    bool passed = true;
    for (const auto& shape : selected_shapes(options)) {
        for (const auto& implementation : selected_implementations(options.implementation)) {
            passed = run_one(options, device, shape, implementation, *output) && passed;
        }
    }
    return passed ? 0 : 1;
}
