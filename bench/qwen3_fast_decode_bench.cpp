#include "miinfer/build_config.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_layer.hpp"
#include "miinfer/sha256.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

namespace {

struct TimePoint { timespec value{}; };

TimePoint now() {
    TimePoint result;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &result.value) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC_RAW) failed");
    }
    return result;
}

double elapsed_ms(TimePoint start, TimePoint end) {
    const auto seconds = static_cast<double>(end.value.tv_sec - start.value.tv_sec);
    const auto nanoseconds = static_cast<double>(end.value.tv_nsec - start.value.tv_nsec);
    return seconds * 1000.0 + nanoseconds / 1000000.0;
}

struct Options {
    std::string model_path;
    std::string prompt_ids = "14990";
    std::size_t warmup_tokens = 8;
    std::size_t decode_tokens = 64;
    std::size_t iterations = 5;
    std::string json_output;
};

struct Stats {
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
};

struct Measurement {
    double prefill_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    std::vector<std::uint32_t> generated_ids;
};

void usage() {
    std::cerr
        << "usage: miinfer-qwen3-fast-decode-bench MODEL.gguf [options]\n"
        << "  --prompt-ids CSV       explicit prompt IDs (default: 14990)\n"
        << "  --warmup N             generated tokens before measurement (default: 8)\n"
        << "  --generated-tokens N   measured decode forward calls (default: 64)\n"
        << "  --iterations N         measured runs (default: 5)\n"
        << "  --json-output PATH     write machine-readable result\n";
}

std::size_t parse_size(const char* text, const char* option, bool allow_zero = false) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoull(text, &consumed, 10);
        if (consumed != std::string(text).size()
            || (!allow_zero && value == 0)
            || value > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("invalid value");
        }
        return static_cast<std::size_t>(value);
    } catch (...) {
        throw std::invalid_argument(std::string(option) + " must be a positive integer");
    }
}

std::uint32_t parse_id(const std::string& value) {
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed, 10);
        if (consumed != value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("invalid token ID");
        }
        return static_cast<std::uint32_t>(parsed);
    } catch (...) {
        throw std::invalid_argument("invalid token ID: " + value);
    }
}

std::vector<std::uint32_t> parse_ids(const std::string& value) {
    if (value.empty()) throw std::invalid_argument("--prompt-ids must not be empty");
    std::vector<std::uint32_t> result;
    std::size_t begin = 0;
    while (begin < value.size()) {
        const auto end = value.find(',', begin);
        result.push_back(parse_id(value.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin)));
        if (end == std::string::npos) break;
        begin = end + 1;
        if (begin == value.size()) throw std::invalid_argument("--prompt-ids has an empty ID");
    }
    return result;
}

bool parse_options(int argc, char** argv, Options& options) {
    if (argc < 2) {
        usage();
        return false;
    }
    options.model_path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            usage();
            return false;
        }
        if (index + 1 >= argc) throw std::invalid_argument("missing value for " + argument);
        if (argument == "--prompt-ids") {
            options.prompt_ids = argv[++index];
        } else if (argument == "--warmup") {
            options.warmup_tokens = parse_size(argv[++index], "--warmup", true);
        } else if (argument == "--generated-tokens") {
            options.decode_tokens = parse_size(argv[++index], "--generated-tokens");
        } else if (argument == "--iterations") {
            options.iterations = parse_size(argv[++index], "--iterations");
        } else if (argument == "--json-output") {
            options.json_output = argv[++index];
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return true;
}

std::string ids_json(const std::vector<std::uint32_t>& ids) {
    std::ostringstream result;
    result << '[';
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) result << ',';
        result << ids[index];
    }
    result << ']';
    return result.str();
}

std::string json_escape(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (const unsigned char character : value) {
        if (character == '"') result << "\\\"";
        else if (character == '\\') result << "\\\\";
        else result << static_cast<char>(character);
    }
    result << '"';
    return result.str();
}

std::uint32_t argmax(const std::vector<float>& logits) {
    if (logits.empty()) throw std::runtime_error("model returned empty logits");
    for (const float value : logits) {
        if (!std::isfinite(value)) throw std::runtime_error("model returned non-finite logits");
    }
    return static_cast<std::uint32_t>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
}

Stats summarize(const std::vector<double>& samples) {
    if (samples.empty()) throw std::runtime_error("cannot summarize empty timing samples");
    Stats result;
    result.min_ms = *std::min_element(samples.begin(), samples.end());
    result.max_ms = *std::max_element(samples.begin(), samples.end());
    for (const double sample : samples) result.mean_ms += sample;
    result.mean_ms /= static_cast<double>(samples.size());
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2;
    result.median_ms = sorted.size() % 2 == 0
        ? (sorted[middle - 1] + sorted[middle]) / 2.0 : sorted[middle];
    for (const double sample : samples) {
        const double delta = sample - result.mean_ms;
        result.stddev_ms += delta * delta;
    }
    result.stddev_ms = std::sqrt(result.stddev_ms / static_cast<double>(samples.size()));
    return result;
}

double throughput(const Stats& stats, std::size_t tokens) {
    return stats.mean_ms <= 0.0 ? 0.0 : static_cast<double>(tokens) * 1000.0 / stats.mean_ms;
}

Measurement run_once(
    const Options& options,
    const miinfer::Qwen3GpuPlan& plan,
    const std::vector<std::uint32_t>& prompt_ids,
    miinfer::Qwen3GpuDecodeCache& cache) {
    cache.reset();
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> logits(plan.model().config().vocab_size);
    const auto total_start = now();
    const auto prefill_start = total_start;
    for (std::size_t position = 0; position < prompt_ids.size(); ++position) {
        miinfer::execute_qwen3_decode_gpu_fast(
            plan, prompt_ids[position], position, cache, std::span<float>(logits));
    }
    const auto prefill_end = now();
    auto generated = std::vector<std::uint32_t>{argmax(logits)};
    const auto ttft_end = now();

    generated.reserve(1 + options.warmup_tokens + options.decode_tokens);
    for (std::size_t index = 1; index < options.warmup_tokens; ++index) {
        miinfer::execute_qwen3_decode_gpu_fast(
            plan, generated.back(), cache.length(), cache, std::span<float>(logits));
        generated.push_back(argmax(logits));
    }

    const auto decode_start = now();
    for (std::size_t index = 0; index < options.decode_tokens; ++index) {
        miinfer::execute_qwen3_decode_gpu_fast(
            plan, generated.back(), cache.length(), cache, std::span<float>(logits));
        generated.push_back(argmax(logits));
    }
    const auto decode_end = now();

    Measurement result;
    result.prefill_ms = elapsed_ms(prefill_start, prefill_end);
    result.ttft_ms = elapsed_ms(total_start, ttft_end);
    result.decode_ms = elapsed_ms(decode_start, decode_end);
    result.total_ms = elapsed_ms(total_start, decode_end);
    result.generated_ids = std::move(generated);
    if (cache.length() != prompt_ids.size() + options.warmup_tokens - (options.warmup_tokens > 0 ? 1 : 0)
                              + options.decode_tokens) {
        throw std::runtime_error("trace-free decode cache length mismatch");
    }
    return result;
}

void write_stats(std::ostream& output, const Stats& stats, std::size_t tokens) {
    output << std::fixed << std::setprecision(6)
           << "{\"tokens\":" << tokens
           << ",\"mean_ms\":" << stats.mean_ms
           << ",\"median_ms\":" << stats.median_ms
           << ",\"min_ms\":" << stats.min_ms
           << ",\"max_ms\":" << stats.max_ms
           << ",\"stddev_ms\":" << stats.stddev_ms
           << ",\"mean_tokens_per_s\":" << throughput(stats, tokens) << '}';
}

std::string build_json(
    const Options& options,
    const miinfer::Qwen3Model& model,
    const miinfer::Qwen3GpuPlan& plan,
    const std::vector<std::uint32_t>& prompt_ids,
    const std::vector<Measurement>& measurements,
    const std::vector<double>& prefill_samples,
    const std::vector<double>& ttft_samples,
    const std::vector<double>& decode_samples,
    const std::vector<double>& total_samples) {
    const auto& first = measurements.front();
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"benchmark\":\"m5c0_qwen3_fast_decode\",\n"
           << "  \"experiment\":\"M5-C0\",\n"
           << "  \"git_commit\":\"" << MIINFER_GIT_COMMIT << "\",\n"
           << "  \"git_dirty\":\"" << MIINFER_GIT_DIRTY << "\",\n"
           << "  \"build_type\":\"" << MIINFER_BUILD_TYPE << "\",\n"
           << "  \"target_architecture\":\"" << MIINFER_TARGET_ARCH << "\",\n"
           << "  \"model_path\":" << json_escape(options.model_path) << ",\n"
           << "  \"model_sha256\":\"" << miinfer::sha256_file(options.model_path) << "\",\n"
           << "  \"model_name\":" << json_escape(model.model_name()) << ",\n"
           << "  \"model_weight_bytes\":" << model.total_weight_bytes() << ",\n"
           << "  \"planned_weight_bytes\":" << plan.weights().bytes() << ",\n"
           << "  \"workspace_bytes\":" << plan.workspace_bytes() << ",\n"
           << "  \"total_vram_bytes\":" << plan.device().total_vram_bytes << ",\n"
           << "  \"prompt_ids\":" << ids_json(prompt_ids) << ",\n"
           << "  \"warmup_tokens\":" << options.warmup_tokens << ",\n"
           << "  \"measured_decode_tokens\":" << options.decode_tokens << ",\n"
           << "  \"iterations\":" << options.iterations << ",\n"
           << "  \"trace_copies\":false,\n"
           << "  \"timing_method\":\"CLOCK_MONOTONIC_RAW wall time; final logits copy required for CPU argmax\",\n"
           << "  \"generated_ids\":" << ids_json(first.generated_ids) << ",\n"
           << "  \"prefill\":";
    write_stats(output, summarize(prefill_samples), prompt_ids.size());
    output << ",\n  \"ttft\":";
    write_stats(output, summarize(ttft_samples), 1);
    output << ",\n  \"decode\":";
    write_stats(output, summarize(decode_samples), options.decode_tokens);
    output << ",\n  \"total\":";
    write_stats(output, summarize(total_samples), prompt_ids.size() + options.warmup_tokens
                                                    + options.decode_tokens);
    output << ",\n  \"samples_ms\":{\n    \"prefill\":[";
    for (std::size_t index = 0; index < prefill_samples.size(); ++index) {
        if (index != 0) output << ',';
        output << prefill_samples[index];
    }
    output << "],\n    \"ttft\":[";
    for (std::size_t index = 0; index < ttft_samples.size(); ++index) {
        if (index != 0) output << ',';
        output << ttft_samples[index];
    }
    output << "],\n    \"decode\":[";
    for (std::size_t index = 0; index < decode_samples.size(); ++index) {
        if (index != 0) output << ',';
        output << decode_samples[index];
    }
    output << "],\n    \"total\":[";
    for (std::size_t index = 0; index < total_samples.size(); ++index) {
        if (index != 0) output << ',';
        output << total_samples[index];
    }
    output << "]\n  }\n}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options;
        if (!parse_options(argc, argv, options)) return argc < 2 ? 2 : 0;
        const auto model = miinfer::Qwen3Model::load(options.model_path);
        const auto prompt_ids = parse_ids(options.prompt_ids);
        if (prompt_ids.empty() || prompt_ids.size() + options.warmup_tokens
                                     + options.decode_tokens > model.config().context_length) {
            throw std::invalid_argument("prompt and decode workload exceed model context");
        }
        for (const auto token : prompt_ids) {
            if (token >= model.config().vocab_size) throw std::invalid_argument("prompt token outside vocabulary");
        }

        const auto plan = miinfer::Qwen3GpuPlan::build(model);
        miinfer::Qwen3GpuDecodeCache cache(
            model.config().layer_count, model.config().kv_heads, model.config().head_dim,
            prompt_ids.size() + options.warmup_tokens + options.decode_tokens + 1);
        std::vector<Measurement> measurements;
        measurements.reserve(options.iterations);
        for (std::size_t run = 0; run < options.iterations; ++run) {
            auto measurement = run_once(options, plan, prompt_ids, cache);
            if (!measurements.empty()
                && measurement.generated_ids != measurements.front().generated_ids) {
                throw std::runtime_error("trace-free generated sequence was not deterministic");
            }
            measurements.push_back(std::move(measurement));
        }

        if (prompt_ids == std::vector<std::uint32_t>{14990}) {
            const std::vector<std::uint32_t> expected_prefix{8, 341, 286, 470, 330, 9707, 11, 330};
            const auto& ids = measurements.front().generated_ids;
            if (ids.size() >= expected_prefix.size()
                && !std::equal(expected_prefix.begin(), expected_prefix.end(), ids.begin())) {
                throw std::runtime_error("trace-free decode changed the pinned hello prefix");
            }
        }

        std::vector<double> prefill_samples, ttft_samples, decode_samples, total_samples;
        for (const auto& measurement : measurements) {
            prefill_samples.push_back(measurement.prefill_ms);
            ttft_samples.push_back(measurement.ttft_ms);
            decode_samples.push_back(measurement.decode_ms);
            total_samples.push_back(measurement.total_ms);
        }
        const auto result = build_json(
            options, model, plan, prompt_ids, measurements, prefill_samples, ttft_samples,
            decode_samples, total_samples);
        if (!options.json_output.empty()) {
            std::ofstream output(options.json_output);
            if (!output) throw std::runtime_error("cannot open JSON output: " + options.json_output);
            output << result;
        }

        const auto prefill = summarize(prefill_samples);
        const auto ttft = summarize(ttft_samples);
        const auto decode = summarize(decode_samples);
        std::cout << "M5-C0 Qwen3 trace-free decode benchmark\n"
                  << "model: " << model.model_name() << "\n"
                  << "prompt_tokens: " << prompt_ids.size() << "\n"
                  << "warmup_tokens: " << options.warmup_tokens << "\n"
                  << "measured_decode_tokens: " << options.decode_tokens << "\n"
                  << std::fixed << std::setprecision(3)
                  << "prefill: " << prefill.mean_ms << " ms ("
                  << throughput(prefill, prompt_ids.size()) << " tok/s)\n"
                  << "ttft: " << ttft.mean_ms << " ms\n"
                  << "decode: " << decode.mean_ms << " ms ("
                  << throughput(decode, options.decode_tokens) << " tok/s)\n"
                  << "generated_prefix: " << ids_json(measurements.front().generated_ids) << '\n'
                  << "correctness: deterministic, finite, trace-free\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M5-C0 fast decode benchmark error: " << error.what() << '\n';
        return 1;
    }
}
