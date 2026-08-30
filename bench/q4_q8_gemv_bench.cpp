#include "miinfer/build_config.hpp"
#include "miinfer/device_validation.hpp"
#include "miinfer/fp16_gemv.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/q4_q8_packed_dot.hpp"
#include "miinfer/q4_q8_gemv.hpp"

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
    std::string mode = "gemv";
    std::string implementation = "scalar";
    std::string shape = "all";
    int length = 4096;
    int device = -1;
    int warmup = 5;
    int iterations = 1000;
    std::string json_output;
};

void usage() {
    std::cerr << "usage: miinfer-q4-q8-gemv-bench [options]\n"
              << "  --mode gemv|one|quantize|fanout-attention|fanout-ffn\n"
              << "  --implementation scalar|packed-dot (default: scalar)\n"
              << "  --shape q|k|v|o|gate|up|down|all (gemv/one; default: all)\n"
              << "  --length N             activation length for quantize (default: 4096)\n"
              << "  --warmup N             warm-up operations (default: 5)\n"
              << "  --iterations N         measured operations (default: 1000)\n"
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
        if (argument == "--mode") {
            options.mode = argv[++index];
        } else if (argument == "--implementation") {
            options.implementation = argv[++index];
        } else if (argument == "--shape") {
            options.shape = argv[++index];
        } else if (argument == "--length") {
            if (!positive(argv[++index], options.length)) {
                std::cerr << "--length must be positive\n";
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
    const bool valid_mode = options.mode == "gemv" || options.mode == "one"
                            || options.mode == "quantize"
                            || options.mode == "fanout-attention"
                            || options.mode == "fanout-ffn";
    const bool valid_implementation = options.implementation == "scalar"
                                       || options.implementation == "packed-dot";
    const bool valid_shape = options.shape == "q" || options.shape == "k"
                             || options.shape == "v" || options.shape == "o"
                             || options.shape == "gate" || options.shape == "up"
                             || options.shape == "down" || options.shape == "all";
    if (!valid_mode || !valid_implementation || !valid_shape
        || options.length % miinfer::kQ8_1BlockSize != 0) {
        std::cerr << "invalid mode/shape or length is not a multiple of 32\n";
        return false;
    }
    if ((options.mode == "fanout-attention" || options.mode == "fanout-ffn")
        && options.shape != "all") {
        std::cerr << "fanout modes do not accept --shape\n";
        return false;
    }
    return true;
}

void launch_selected_gemv(
    const Options& options,
    const miinfer::Q4_0Block* weights,
    const miinfer::Q8_1Block* input,
    __half* output,
    int rows,
    int columns) {
    if (options.implementation == "packed-dot") {
        miinfer::launch_q4_q8_gemv_packed_dot(weights, input, output, rows, columns);
    } else {
        miinfer::launch_q4_q8_gemv(weights, input, output, rows, columns);
    }
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

std::vector<double> time_operations(int warmup, int iterations, const auto& operation) {
    for (int index = 0; index < warmup; ++index) {
        operation(index);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int index = 0; index < iterations; ++index) {
        MIINFER_HIP_CHECK(hipEventRecord(start, nullptr));
        operation(index);
        MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    MIINFER_HIP_CHECK(hipEventDestroy(stop));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    return samples;
}

void write_timing_fields(std::ostream& output, const std::vector<double>& samples) {
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double med = median(samples);
    const double variance = std::accumulate(
                                samples.begin(), samples.end(), 0.0,
                                [mean](double total, double sample) {
                                    const double difference = sample - mean;
                                    return total + difference * difference;
                                })
                            / samples.size();
    output << std::fixed << std::setprecision(9)
           << ",\"mean_us\":" << mean
           << ",\"median_us\":" << med
           << ",\"min_us\":" << *std::min_element(samples.begin(), samples.end())
           << ",\"max_us\":" << *std::max_element(samples.begin(), samples.end())
           << ",\"stddev_us\":" << std::sqrt(variance)
           << ",\"samples_us\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << samples[index];
    }
    output << ']';
}

void write_metrics(std::ostream& output, const miinfer::Fp16GemvMetrics& metrics) {
    output << std::fixed << std::setprecision(9)
           << "{\"pass\":" << (metrics.pass ? "true" : "false")
           << ",\"max_abs_error\":" << metrics.max_abs_error
           << ",\"mean_abs_error\":" << metrics.mean_abs_error
           << ",\"max_relative_error\":" << metrics.max_relative_error
           << ",\"cosine_similarity\":" << metrics.cosine_similarity
           << ",\"nan_detected\":" << (metrics.nan_detected ? "true" : "false")
           << ",\"inf_detected\":" << (metrics.inf_detected ? "true" : "false") << '}';
}

std::vector<miinfer::GemvShape> selected_shapes(const Options& options) {
    if (options.shape == "all") {
        return miinfer::qwen3_gemv_shapes();
    }
    for (const auto& shape : miinfer::qwen3_gemv_shapes()) {
        if ((options.shape == "q" && std::string(shape.id) == "Q")
            || (options.shape == "k" && std::string(shape.id) == "K")
            || (options.shape == "v" && std::string(shape.id) == "V")
            || (options.shape == "o" && std::string(shape.id) == "O")
            || (options.shape == "gate" && std::string(shape.id) == "G")
            || (options.shape == "up" && std::string(shape.id) == "U")
            || (options.shape == "down" && std::string(shape.id) == "D")) {
            return {shape};
        }
    }
    return {};
}

struct DeviceShapeData {
    std::vector<__half> weights_fp16;
    std::vector<__half> input_fp16;
    std::vector<miinfer::Q4_0Block> weights_q4;
    std::vector<miinfer::Q8_1Block> input_q8;
    std::vector<float> oracle_q4_q8;
    std::vector<float> oracle_fp16;
    std::vector<miinfer::Q4_0Block*> device_weights;
    miinfer::Q8_1Block* device_input_q8 = nullptr;
    __half* device_input_fp16 = nullptr;
    __half* device_output = nullptr;
};

void allocate_shape(DeviceShapeData& data, const miinfer::GemvShape& shape) {
    const std::size_t weight_bytes = data.weights_q4.size() * sizeof(miinfer::Q4_0Block);
    const std::size_t input_q8_bytes = data.input_q8.size() * sizeof(miinfer::Q8_1Block);
    const std::size_t input_fp16_bytes = data.input_fp16.size() * sizeof(__half);
    const std::size_t output_bytes = static_cast<std::size_t>(shape.m) * sizeof(__half);
    data.device_weights.assign(3, nullptr);
    for (auto& pointer : data.device_weights) {
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&pointer), weight_bytes));
        MIINFER_HIP_CHECK(hipMemcpy(pointer, data.weights_q4.data(), weight_bytes,
                                    hipMemcpyHostToDevice));
    }
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data.device_input_q8), input_q8_bytes));
    MIINFER_HIP_CHECK(hipMemcpy(data.device_input_q8, data.input_q8.data(), input_q8_bytes,
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data.device_input_fp16), input_fp16_bytes));
    MIINFER_HIP_CHECK(hipMemcpy(data.device_input_fp16, data.input_fp16.data(), input_fp16_bytes,
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data.device_output), output_bytes));
}

void free_shape(DeviceShapeData& data) {
    if (data.device_output != nullptr) {
        MIINFER_HIP_CHECK(hipFree(data.device_output));
    }
    if (data.device_input_fp16 != nullptr) {
        MIINFER_HIP_CHECK(hipFree(data.device_input_fp16));
    }
    if (data.device_input_q8 != nullptr) {
        MIINFER_HIP_CHECK(hipFree(data.device_input_q8));
    }
    for (auto pointer : data.device_weights) {
        if (pointer != nullptr) {
            MIINFER_HIP_CHECK(hipFree(pointer));
        }
    }
}

bool run_shape(
    const Options& options,
    const miinfer::DeviceInfo& device,
    const miinfer::GemvShape& shape,
    std::ostream& output) {
    DeviceShapeData data;
    miinfer::generate_fp16_gemv_data(shape.m, shape.k, kSeed, data.weights_fp16, data.input_fp16);
    data.weights_q4 = miinfer::quantize_q4_0(data.weights_fp16, shape.m, shape.k);
    data.input_q8 = miinfer::quantize_q8_1(data.input_fp16);
    data.oracle_q4_q8 = miinfer::q4_q8_cpu_reference(data.weights_q4, data.input_q8,
                                                      shape.m, shape.k);
    data.oracle_fp16 = miinfer::fp16_gemv_cpu_reference(data.weights_fp16, data.input_fp16,
                                                         shape.m, shape.k);
    allocate_shape(data, shape);

    miinfer::Q8_1Block* quantized_input = data.device_input_q8;
    const auto operation = [&](int index) {
        const auto* weights = data.device_weights[static_cast<std::size_t>(index)
                                                  % data.device_weights.size()];
        if (options.mode == "one") {
            miinfer::launch_q8_1_quantize(data.device_input_fp16, quantized_input, shape.k);
        }
        launch_selected_gemv(options, weights, quantized_input, data.device_output,
                             shape.m, shape.k);
    };
    if (options.mode == "gemv") {
        // Q8_1 already exists before the timed loop for the kernel-only view.
        operation(0);
    }
    const auto samples = time_operations(options.warmup, options.iterations, operation);
    std::vector<__half> output_values(static_cast<std::size_t>(shape.m));
    MIINFER_HIP_CHECK(hipMemcpy(output_values.data(), data.device_output,
                                output_values.size() * sizeof(__half), hipMemcpyDeviceToHost));
    const auto gpu_metrics = miinfer::evaluate_fp16_gemv(output_values, data.oracle_q4_q8);
    std::vector<__half> oracle_values(output_values.size());
    for (std::size_t index = 0; index < oracle_values.size(); ++index) {
        oracle_values[index] = __float2half_rn(data.oracle_q4_q8[index]);
    }
    const auto quantization_metrics = miinfer::evaluate_fp16_gemv(oracle_values, data.oracle_fp16);
    const double logical_bytes = static_cast<double>(data.weights_q4.size() * sizeof(miinfer::Q4_0Block))
                                 + static_cast<double>(data.input_q8.size() * sizeof(miinfer::Q8_1Block))
                                 + static_cast<double>(shape.m * sizeof(__half));
    const double med = median(samples);
    output << "{\"experiment\":\"EXP-0006\",\"mode\":" << escape(options.mode)
           << ",\"implementation\":" << escape(options.implementation)
           << ",\"shape\":\"" << shape.id << "\",\"m\":" << shape.m
           << ",\"k\":" << shape.k << ",\"input_dtype\":\"q8_1\""
           << ",\"weight_dtype\":\"q4_0\",\"accumulator_dtype\":\"fp32\""
           << ",\"output_dtype\":\"fp16\",\"seed\":" << kSeed
           << ",\"cache_regime\":\"streaming-rotate-3\""
           << ",\"warmup_iterations\":" << options.warmup
           << ",\"measured_iterations\":" << options.iterations;
    write_timing_fields(output, samples);
    output << ",\"effective_gb_per_second\":" << logical_bytes / (med * 1000.0)
           << ",\"logical_bytes\":" << logical_bytes
           << ",\"q4_0_weight_bytes\":" << data.weights_q4.size() * sizeof(miinfer::Q4_0Block)
           << ",\"fp16_weight_bytes\":" << data.weights_fp16.size() * sizeof(__half)
           << ",\"q4_0_compression_ratio\":"
           << static_cast<double>(data.weights_fp16.size() * sizeof(__half))
                  / static_cast<double>(data.weights_q4.size() * sizeof(miinfer::Q4_0Block))
           << ",\"gpu\":" << escape(device.name)
           << ",\"gfx\":" << escape(device.architecture)
           << ",\"vram_bytes\":" << device.total_vram_bytes
           << ",\"git_commit\":" << escape(MIINFER_GIT_COMMIT)
           << ",\"git_dirty\":" << escape(MIINFER_GIT_DIRTY)
           << ",\"gpu_vs_quantized_oracle\":";
    write_metrics(output, gpu_metrics);
    output << ",\"quantized_oracle_vs_fp16_oracle\":";
    write_metrics(output, quantization_metrics);
    output << "}\n";
    free_shape(data);
    return gpu_metrics.pass;
}

bool run_quantize(
    const Options& options,
    const miinfer::DeviceInfo& device,
    std::ostream& output) {
    std::vector<__half> input;
    std::vector<__half> ignored_weights;
    miinfer::generate_fp16_gemv_data(1, options.length, kSeed, ignored_weights, input);
    const auto expected = miinfer::quantize_q8_1(input);
    __half* device_input = nullptr;
    miinfer::Q8_1Block* device_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(__half)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output),
                                expected.size() * sizeof(miinfer::Q8_1Block)));
    MIINFER_HIP_CHECK(hipMemcpy(device_input, input.data(), input.size() * sizeof(__half),
                                hipMemcpyHostToDevice));
    const auto operation = [&](int) {
        miinfer::launch_q8_1_quantize(device_input, device_output, options.length);
    };
    const auto samples = time_operations(options.warmup, options.iterations, operation);
    std::vector<miinfer::Q8_1Block> actual(expected.size());
    MIINFER_HIP_CHECK(hipMemcpy(actual.data(), device_output,
                                actual.size() * sizeof(miinfer::Q8_1Block), hipMemcpyDeviceToHost));
    std::size_t mismatched_blocks = 0;
    std::size_t first_mismatch = expected.size();
    int first_expected_q = 0;
    int first_actual_q = 0;
    int first_q_mismatches = 0;
    int first_expected_q_sum = 0;
    int first_actual_q_sum = 0;
    float first_expected_d = 0.0F;
    float first_actual_d = 0.0F;
    float first_expected_s = 0.0F;
    float first_actual_s = 0.0F;
    double max_scale_sum_error = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double scale_sum_error = std::fabs(
            static_cast<double>(__half2float(actual[index].s))
            - static_cast<double>(__half2float(expected[index].s)));
        max_scale_sum_error = std::max(max_scale_sum_error, scale_sum_error);
        bool block_mismatch = __half2float(actual[index].d) != __half2float(expected[index].d);
        int block_q_mismatches = 0;
        for (int q_index = 0; q_index < miinfer::kQ8_1BlockSize; ++q_index) {
            block_q_mismatches += actual[index].qs[q_index] != expected[index].qs[q_index];
        }
        // s is FP16 metadata. CPU/GPU evaluation can differ by one FP16 ULP
        // at a rounding boundary; q-values and d remain exact requirements.
        block_mismatch = block_mismatch || block_q_mismatches != 0 || scale_sum_error > 0.0011;
        if (block_mismatch) {
            ++mismatched_blocks;
            if (first_mismatch == expected.size()) {
                first_mismatch = index;
                first_expected_d = __half2float(expected[index].d);
                first_actual_d = __half2float(actual[index].d);
                first_expected_s = __half2float(expected[index].s);
                first_actual_s = __half2float(actual[index].s);
                for (int q_index = 0; q_index < miinfer::kQ8_1BlockSize; ++q_index) {
                    first_expected_q_sum += expected[index].qs[q_index];
                    first_actual_q_sum += actual[index].qs[q_index];
                    if (expected[index].qs[q_index] != actual[index].qs[q_index]) {
                        ++first_q_mismatches;
                        first_expected_q = expected[index].qs[q_index];
                        first_actual_q = actual[index].qs[q_index];
                    }
                }
            }
        }
    }
    const bool pass = mismatched_blocks == 0;
    const double logical_bytes = static_cast<double>(input.size() * sizeof(__half))
                                 + static_cast<double>(expected.size() * sizeof(miinfer::Q8_1Block));
    const double med = median(samples);
    output << "{\"experiment\":\"EXP-0005\",\"mode\":\"quantize\",\"length\":"
           << options.length << ",\"input_dtype\":\"fp16\",\"output_dtype\":\"q8_1\""
           << ",\"warmup_iterations\":" << options.warmup
           << ",\"measured_iterations\":" << options.iterations;
    write_timing_fields(output, samples);
    output << ",\"effective_io_gb_per_second\":" << logical_bytes / (med * 1000.0)
           << ",\"logical_input_bytes\":" << input.size() * sizeof(__half)
           << ",\"logical_output_bytes\":" << expected.size() * sizeof(miinfer::Q8_1Block)
           << ",\"correctness\":{\"pass\":" << (pass ? "true" : "false")
           << ",\"mismatched_blocks\":" << mismatched_blocks
           << ",\"first_mismatch\":" << (first_mismatch == expected.size() ? -1 : static_cast<long long>(first_mismatch))
           << ",\"first_expected_d\":" << first_expected_d
           << ",\"first_actual_d\":" << first_actual_d
           << ",\"first_expected_s\":" << first_expected_s
           << ",\"first_actual_s\":" << first_actual_s
           << ",\"first_expected_q\":" << first_expected_q
           << ",\"first_actual_q\":" << first_actual_q
           << ",\"first_q_mismatches\":" << first_q_mismatches
           << ",\"first_expected_q_sum\":" << first_expected_q_sum
           << ",\"first_actual_q_sum\":" << first_actual_q_sum
           << ",\"max_scale_sum_abs_error\":" << max_scale_sum_error
           << '}'
           << ",\"gpu\":" << escape(device.name) << ",\"gfx\":" << escape(device.architecture)
           << ",\"git_commit\":" << escape(MIINFER_GIT_COMMIT)
           << ",\"git_dirty\":" << escape(MIINFER_GIT_DIRTY) << "}\n";
    MIINFER_HIP_CHECK(hipFree(device_output));
    MIINFER_HIP_CHECK(hipFree(device_input));
    return pass;
}

bool run_fanout(
    const Options& options,
    const miinfer::DeviceInfo& device,
    std::ostream& output) {
    const bool attention = options.mode == "fanout-attention";
    const std::vector<miinfer::GemvShape> shapes = attention
        ? std::vector<miinfer::GemvShape>{miinfer::qwen3_gemv_shapes()[0],
                                          miinfer::qwen3_gemv_shapes()[1],
                                          miinfer::qwen3_gemv_shapes()[2]}
        : std::vector<miinfer::GemvShape>{miinfer::qwen3_gemv_shapes()[4],
                                          miinfer::qwen3_gemv_shapes()[5]};
    std::vector<__half> input_fp16;
    std::vector<__half> ignored_weights;
    miinfer::generate_fp16_gemv_data(1, shapes.front().k, kSeed, ignored_weights, input_fp16);
    std::vector<miinfer::Q8_1Block> input_q8 = miinfer::quantize_q8_1(input_fp16);
    __half* device_input_fp16 = nullptr;
    miinfer::Q8_1Block* device_input_q8 = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input_fp16),
                                input_fp16.size() * sizeof(__half)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input_q8),
                                input_q8.size() * sizeof(miinfer::Q8_1Block)));
    MIINFER_HIP_CHECK(hipMemcpy(device_input_fp16, input_fp16.data(),
                                input_fp16.size() * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(device_input_q8, input_q8.data(),
                                input_q8.size() * sizeof(miinfer::Q8_1Block), hipMemcpyHostToDevice));

    std::vector<std::vector<miinfer::Q4_0Block>> host_weights;
    std::vector<std::vector<float>> oracles;
    std::vector<std::vector<miinfer::Q4_0Block*>> device_weights(
        shapes.size(), std::vector<miinfer::Q4_0Block*>(0));
    std::vector<__half*> device_outputs(shapes.size(), nullptr);
    for (const auto& shape : shapes) {
        std::vector<__half> weights;
        std::vector<__half> ignored_input;
        miinfer::generate_fp16_gemv_data(shape.m, shape.k, kSeed, weights, ignored_input);
        host_weights.push_back(miinfer::quantize_q4_0(weights, shape.m, shape.k));
        oracles.push_back(miinfer::q4_q8_cpu_reference(host_weights.back(), input_q8,
                                                       shape.m, shape.k));
    }
    for (std::size_t shape_index = 0; shape_index < shapes.size(); ++shape_index) {
        const auto& shape = shapes[shape_index];
        const std::size_t bytes = host_weights[shape_index].size() * sizeof(miinfer::Q4_0Block);
        device_weights[shape_index].assign(3, nullptr);
        for (auto& pointer : device_weights[shape_index]) {
            MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&pointer), bytes));
            MIINFER_HIP_CHECK(hipMemcpy(pointer, host_weights[shape_index].data(), bytes,
                                        hipMemcpyHostToDevice));
        }
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_outputs[shape_index]),
                                    static_cast<std::size_t>(shape.m) * sizeof(__half)));
    }
    const auto operation = [&](int index) {
        miinfer::launch_q8_1_quantize(device_input_fp16, device_input_q8,
                                      shapes.front().k);
        for (std::size_t shape_index = 0; shape_index < shapes.size(); ++shape_index) {
            if (options.implementation == "packed-dot") {
                miinfer::launch_q4_q8_gemv_packed_dot(
                    device_weights[shape_index][static_cast<std::size_t>(index) % 3],
                    device_input_q8, device_outputs[shape_index], shapes[shape_index].m,
                    shapes[shape_index].k);
            } else {
                miinfer::launch_q4_q8_gemv(
                device_weights[shape_index][static_cast<std::size_t>(index) % 3],
                device_input_q8, device_outputs[shape_index], shapes[shape_index].m,
                shapes[shape_index].k);
            }
        }
    };
    const auto samples = time_operations(options.warmup, options.iterations, operation);
    bool pass = true;
    for (std::size_t shape_index = 0; shape_index < shapes.size(); ++shape_index) {
        std::vector<__half> actual(static_cast<std::size_t>(shapes[shape_index].m));
        MIINFER_HIP_CHECK(hipMemcpy(actual.data(), device_outputs[shape_index],
                                    actual.size() * sizeof(__half), hipMemcpyDeviceToHost));
        pass = miinfer::evaluate_fp16_gemv(actual, oracles[shape_index]).pass && pass;
    }
    output << "{\"experiment\":\"EXP-0006\",\"mode\":" << escape(options.mode)
           << ",\"implementation\":" << escape(options.implementation)
           << ",\"fanout\":\"" << (attention ? "Q/K/V" : "gate/up") << "\""
           << ",\"activation_length\":" << shapes.front().k
           << ",\"warmup_iterations\":" << options.warmup
           << ",\"measured_iterations\":" << options.iterations;
    write_timing_fields(output, samples);
    output << ",\"correctness\":{\"pass\":" << (pass ? "true" : "false") << "}"
           << ",\"gpu\":" << escape(device.name) << ",\"gfx\":" << escape(device.architecture)
           << ",\"git_commit\":" << escape(MIINFER_GIT_COMMIT)
           << ",\"git_dirty\":" << escape(MIINFER_GIT_DIRTY) << "}\n";
    for (std::size_t shape_index = 0; shape_index < shapes.size(); ++shape_index) {
        for (auto pointer : device_weights[shape_index]) {
            MIINFER_HIP_CHECK(hipFree(pointer));
        }
        MIINFER_HIP_CHECK(hipFree(device_outputs[shape_index]));
    }
    MIINFER_HIP_CHECK(hipFree(device_input_q8));
    MIINFER_HIP_CHECK(hipFree(device_input_fp16));
    return pass;
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
        std::cerr << "Q4_0 x Q8_1 benchmark unavailable: " << error << '\n';
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
    if (options.mode == "quantize") {
        passed = run_quantize(options, device, *output);
    } else if (options.mode == "fanout-attention" || options.mode == "fanout-ffn") {
        passed = run_fanout(options, device, *output);
    } else {
        for (const auto& shape : selected_shapes(options)) {
            passed = run_shape(options, device, shape, *output) && passed;
        }
    }
    return passed ? 0 : 1;
}
