#include "miinfer/build_config.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_layer.hpp"
#include "miinfer/sha256.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
    std::size_t pairs = 3;
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

struct PolicySamples {
    std::string name;
    std::vector<Measurement> measurements;
};

void usage() {
    std::cerr
        << "usage: miinfer-qwen3-attention-ab-bench MODEL.gguf [options]\n"
        << "  --prompt-ids CSV       explicit prompt IDs (default: 14990)\n"
        << "  --warmup N             generated tokens before measurement (default: 8)\n"
        << "  --generated-tokens N   measured decode forward calls (default: 64)\n"
        << "  --pairs N              serial/parallel A/B pairs (default: 3)\n"
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
        } else if (argument == "--pairs") {
            options.pairs = parse_size(argv[++index], "--pairs");
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
    miinfer::Qwen3GpuDecodeCache& cache,
    const char* attention_kernel) {
    if (::setenv("MIINFER_ATTENTION_KERNEL", attention_kernel, 1) != 0) {
        throw std::runtime_error("setenv(MIINFER_ATTENTION_KERNEL) failed");
    }
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

    const auto expected_cache_length = prompt_ids.size()
        + (options.warmup_tokens > 0 ? options.warmup_tokens - 1 : 0)
        + options.decode_tokens;
    if (cache.length() != expected_cache_length) {
        throw std::runtime_error("interleaved decode cache length mismatch");
    }
    Measurement result;
    result.prefill_ms = elapsed_ms(prefill_start, prefill_end);
    result.ttft_ms = elapsed_ms(total_start, ttft_end);
    result.decode_ms = elapsed_ms(decode_start, decode_end);
    result.total_ms = elapsed_ms(total_start, decode_end);
    result.generated_ids = std::move(generated);
    return result;
}

std::vector<double> field_samples(
    const std::vector<Measurement>& measurements,
    double Measurement::*field) {
    std::vector<double> result;
    result.reserve(measurements.size());
    for (const auto& measurement : measurements) result.push_back(measurement.*field);
    return result;
}

void write_stats_json(std::ostream& output, const Stats& stats, std::size_t tokens) {
    output << std::fixed << std::setprecision(6)
           << "{\"tokens\":" << tokens
           << ",\"mean_ms\":" << stats.mean_ms
           << ",\"median_ms\":" << stats.median_ms
           << ",\"min_ms\":" << stats.min_ms
           << ",\"max_ms\":" << stats.max_ms
           << ",\"stddev_ms\":" << stats.stddev_ms
           << ",\"mean_tokens_per_s\":" << throughput(stats, tokens) << '}';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options;
        if (!parse_options(argc, argv, options)) return argc < 2 ? 2 : 0;
        const auto model = miinfer::Qwen3Model::load(options.model_path);
        const auto prompt_ids = parse_ids(options.prompt_ids);
        if (prompt_ids.size() + options.warmup_tokens + options.decode_tokens
                > model.config().context_length) {
            throw std::invalid_argument("prompt and decode workload exceed model context");
        }
        for (const auto token : prompt_ids) {
            if (token >= model.config().vocab_size) {
                throw std::invalid_argument("prompt token outside vocabulary");
            }
        }

        const auto plan = miinfer::Qwen3GpuPlan::build(model);
        miinfer::Qwen3GpuDecodeCache cache(
            model.config().layer_count, model.config().kv_heads, model.config().head_dim,
            prompt_ids.size() + options.warmup_tokens + options.decode_tokens + 1);
        PolicySamples serial{"serial", {}};
        PolicySamples parallel{"parallel", {}};
        serial.measurements.reserve(options.pairs);
        parallel.measurements.reserve(options.pairs);
        std::vector<std::string> order;
        order.reserve(options.pairs * 2);

        for (std::size_t pair = 0; pair < options.pairs; ++pair) {
            const bool serial_first = pair % 2 == 0;
            const auto run = [&](PolicySamples& samples, const char* kernel) {
                samples.measurements.push_back(
                    run_once(options, plan, prompt_ids, cache, kernel));
                order.emplace_back(kernel);
            };
            if (serial_first) {
                run(serial, "serial");
                run(parallel, "parallel");
            } else {
                run(parallel, "parallel");
                run(serial, "serial");
            }
        }

        const std::vector<std::uint32_t> expected_prefix{8, 341, 286, 470, 330, 9707, 11, 330};
        const auto validate = [&](const PolicySamples& samples) {
            if (samples.measurements.empty()) throw std::runtime_error("missing A/B samples");
            const auto& ids = samples.measurements.front().generated_ids;
            if (ids.size() < expected_prefix.size()
                || !std::equal(expected_prefix.begin(), expected_prefix.end(), ids.begin())) {
                throw std::runtime_error(samples.name + " changed the pinned hello prefix");
            }
            for (const auto& measurement : samples.measurements) {
                if (measurement.generated_ids != ids) {
                    throw std::runtime_error(samples.name + " generated sequence was not deterministic");
                }
            }
        };
        validate(serial);
        validate(parallel);
        if (serial.measurements.front().generated_ids
            != parallel.measurements.front().generated_ids) {
            throw std::runtime_error("serial and parallel generated sequences differ");
        }

        const auto serial_prefill = field_samples(serial.measurements, &Measurement::prefill_ms);
        const auto serial_ttft = field_samples(serial.measurements, &Measurement::ttft_ms);
        const auto serial_decode = field_samples(serial.measurements, &Measurement::decode_ms);
        const auto serial_total = field_samples(serial.measurements, &Measurement::total_ms);
        const auto parallel_prefill = field_samples(parallel.measurements, &Measurement::prefill_ms);
        const auto parallel_ttft = field_samples(parallel.measurements, &Measurement::ttft_ms);
        const auto parallel_decode = field_samples(parallel.measurements, &Measurement::decode_ms);
        const auto parallel_total = field_samples(parallel.measurements, &Measurement::total_ms);
        const auto serial_decode_stats = summarize(serial_decode);
        const auto parallel_decode_stats = summarize(parallel_decode);

        std::ostringstream json;
        json << std::fixed << std::setprecision(6)
             << "{\n"
             << "  \"benchmark\":\"m5c3_qwen3_attention_ab\",\n"
             << "  \"experiment\":\"M5-C3\",\n"
             << "  \"git_commit\":\"" << MIINFER_GIT_COMMIT << "\",\n"
             << "  \"git_dirty\":\"" << MIINFER_GIT_DIRTY << "\",\n"
             << "  \"build_type\":\"" << MIINFER_BUILD_TYPE << "\",\n"
             << "  \"target_architecture\":\"" << MIINFER_TARGET_ARCH << "\",\n"
             << "  \"model_path\":" << json_escape(options.model_path) << ",\n"
             << "  \"model_sha256\":\"" << miinfer::sha256_file(options.model_path) << "\",\n"
             << "  \"model_name\":" << json_escape(model.model_name()) << ",\n"
             << "  \"prompt_ids\":" << ids_json(prompt_ids) << ",\n"
             << "  \"warmup_tokens\":" << options.warmup_tokens << ",\n"
             << "  \"measured_decode_tokens\":" << options.decode_tokens << ",\n"
             << "  \"pairs\":" << options.pairs << ",\n"
             << "  \"interleave_order\":[";
        for (std::size_t index = 0; index < order.size(); ++index) {
            if (index != 0) json << ',';
            json << json_escape(order[index]);
        }
        json << "],\n"
             << "  \"generated_ids\":" << ids_json(serial.measurements.front().generated_ids) << ",\n"
             << "  \"serial_decode\":";
        write_stats_json(json, serial_decode_stats, options.decode_tokens);
        json << ",\n  \"parallel_decode\":";
        write_stats_json(json, parallel_decode_stats, options.decode_tokens);
        json << ",\n  \"samples_ms\":{\n"
             << "    \"serial_prefill\":[";
        for (std::size_t index = 0; index < serial_prefill.size(); ++index) {
            if (index != 0) json << ',';
            json << serial_prefill[index];
        }
        json << "],\n    \"serial_ttft\":[";
        for (std::size_t index = 0; index < serial_ttft.size(); ++index) {
            if (index != 0) json << ',';
            json << serial_ttft[index];
        }
        json << "],\n    \"serial_decode\":[";
        for (std::size_t index = 0; index < serial_decode.size(); ++index) {
            if (index != 0) json << ',';
            json << serial_decode[index];
        }
        json << "],\n    \"serial_total\":[";
        for (std::size_t index = 0; index < serial_total.size(); ++index) {
            if (index != 0) json << ',';
            json << serial_total[index];
        }
        json << "],\n    \"parallel_prefill\":[";
        for (std::size_t index = 0; index < parallel_prefill.size(); ++index) {
            if (index != 0) json << ',';
            json << parallel_prefill[index];
        }
        json << "],\n    \"parallel_ttft\":[";
        for (std::size_t index = 0; index < parallel_ttft.size(); ++index) {
            if (index != 0) json << ',';
            json << parallel_ttft[index];
        }
        json << "],\n    \"parallel_decode\":[";
        for (std::size_t index = 0; index < parallel_decode.size(); ++index) {
            if (index != 0) json << ',';
            json << parallel_decode[index];
        }
        json << "],\n    \"parallel_total\":[";
        for (std::size_t index = 0; index < parallel_total.size(); ++index) {
            if (index != 0) json << ',';
            json << parallel_total[index];
        }
        json << "]\n  }\n}\n";

        if (!options.json_output.empty()) {
            std::ofstream output(options.json_output);
            if (!output) throw std::runtime_error("cannot open JSON output: " + options.json_output);
            output << json.str();
        }

        const auto print = [&](const char* name, const std::vector<Measurement>& measurements) {
            const auto decode = summarize(field_samples(measurements, &Measurement::decode_ms));
            std::cout << name << " decode: " << decode.mean_ms << " ms ("
                      << throughput(decode, options.decode_tokens) << " tok/s), samples="
                      << measurements.size() << '\n';
        };
        std::cout << "M5-C3 interleaved Qwen3 attention A/B benchmark\n"
                  << "model: " << model.model_name() << "\n"
                  << "pairs: " << options.pairs << "\n"
                  << "measured_decode_tokens: " << options.decode_tokens << "\n"
                  << "order: ";
        for (std::size_t index = 0; index < order.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << order[index];
        }
        std::cout << '\n';
        print("serial", serial.measurements);
        print("parallel", parallel.measurements);
        const double serial_throughput = throughput(serial_decode_stats, options.decode_tokens);
        const double parallel_throughput = throughput(parallel_decode_stats, options.decode_tokens);
        const double speedup = serial_throughput > 0.0
            ? parallel_throughput / serial_throughput : 0.0;
        std::cout << "speedup: " << speedup << "x\n"
                  << "correctness: deterministic, finite, identical generated IDs\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M5-C3 attention A/B benchmark error: " << error.what() << '\n';
        return 1;
    }
}
