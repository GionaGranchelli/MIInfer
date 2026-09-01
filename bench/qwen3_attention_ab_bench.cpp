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
    std::string ab_mode = "attention";
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
        << "  --mode MODE            attention, layer-output, or kv-cache (default: attention)\n"
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
        } else if (argument == "--mode") {
            options.ab_mode = argv[++index];
            if (options.ab_mode != "attention" && options.ab_mode != "layer-output"
                && options.ab_mode != "kv-cache") {
                throw std::invalid_argument(
                    "--mode must be 'attention', 'layer-output', or 'kv-cache'");
            }
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
    const char* policy,
    bool layer_output_mode) {
    if (layer_output_mode) {
        if (::setenv("MIINFER_ATTENTION_KERNEL", "parallel", 1) != 0
            || ::setenv("MIINFER_KV_CACHE_WRITE", "store", 1) != 0
            || ::setenv("MIINFER_LAYER_OUTPUT_HANDOFF", policy, 1) != 0) {
            throw std::runtime_error("setenv(layer-output A/B policy) failed");
        }
    } else if (options.ab_mode == "kv-cache") {
        if (::setenv("MIINFER_ATTENTION_KERNEL", "parallel", 1) != 0
            || ::setenv("MIINFER_KV_CACHE_WRITE", policy, 1) != 0
            || ::setenv("MIINFER_LAYER_OUTPUT_HANDOFF", "direct", 1) != 0) {
            throw std::runtime_error("setenv(KV-cache A/B policy) failed");
        }
    } else if (::setenv("MIINFER_ATTENTION_KERNEL", policy, 1) != 0
               || ::setenv("MIINFER_KV_CACHE_WRITE", "store", 1) != 0
               || ::setenv("MIINFER_LAYER_OUTPUT_HANDOFF", "direct", 1) != 0) {
        throw std::runtime_error("setenv(attention A/B policy) failed");
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
        const bool layer_output_mode = options.ab_mode == "layer-output";
        const bool kv_cache_mode = options.ab_mode == "kv-cache";
        const char* first_policy = layer_output_mode || kv_cache_mode ? "copy" : "serial";
        const char* second_policy = layer_output_mode ? "direct"
            : kv_cache_mode ? "store" : "parallel";
        PolicySamples first{first_policy, {}};
        PolicySamples second{second_policy, {}};
        first.measurements.reserve(options.pairs);
        second.measurements.reserve(options.pairs);
        std::vector<std::string> order;
        order.reserve(options.pairs * 2);

        for (std::size_t pair = 0; pair < options.pairs; ++pair) {
            const bool first_policy_first = pair % 2 == 0;
            const auto run = [&](PolicySamples& samples, const char* policy) {
                samples.measurements.push_back(
                    run_once(options, plan, prompt_ids, cache, policy, layer_output_mode));
                order.emplace_back(policy);
            };
            if (first_policy_first) {
                run(first, first_policy);
                run(second, second_policy);
            } else {
                run(second, second_policy);
                run(first, first_policy);
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
        validate(first);
        validate(second);
        if (first.measurements.front().generated_ids
            != second.measurements.front().generated_ids) {
            throw std::runtime_error("A/B generated sequences differ");
        }

        const auto first_prefill = field_samples(first.measurements, &Measurement::prefill_ms);
        const auto first_ttft = field_samples(first.measurements, &Measurement::ttft_ms);
        const auto first_decode = field_samples(first.measurements, &Measurement::decode_ms);
        const auto first_total = field_samples(first.measurements, &Measurement::total_ms);
        const auto second_prefill = field_samples(second.measurements, &Measurement::prefill_ms);
        const auto second_ttft = field_samples(second.measurements, &Measurement::ttft_ms);
        const auto second_decode = field_samples(second.measurements, &Measurement::decode_ms);
        const auto second_total = field_samples(second.measurements, &Measurement::total_ms);
        const auto first_decode_stats = summarize(first_decode);
        const auto second_decode_stats = summarize(second_decode);

        std::ostringstream json;
        json << std::fixed << std::setprecision(6)
             << "{\n"
             << "  \"benchmark\":\""
             << (layer_output_mode ? "m5c6b_qwen3_layer_output_ab"
                 : kv_cache_mode ? "m5c6c_qwen3_kv_cache_ab" : "m5c3_qwen3_attention_ab")
             << "\",\n"
             << "  \"experiment\":\""
             << (layer_output_mode ? "M5-C6b" : kv_cache_mode ? "M5-C6c" : "M5-C3")
             << "\",\n"
             << "  \"ab_mode\":\"" << options.ab_mode << "\",\n"
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
             << "  \"generated_ids\":" << ids_json(first.measurements.front().generated_ids) << ",\n"
             << "  \"" << first.name << "_decode\":";
        write_stats_json(json, first_decode_stats, options.decode_tokens);
        json << ",\n  \"" << second.name << "_decode\":";
        write_stats_json(json, second_decode_stats, options.decode_tokens);
        json << ",\n  \"samples_ms\":{\n"
             << "    \"" << first.name << "_prefill\":[";
        for (std::size_t index = 0; index < first_prefill.size(); ++index) {
            if (index != 0) json << ',';
            json << first_prefill[index];
        }
        json << "],\n    \"" << first.name << "_ttft\":[";
        for (std::size_t index = 0; index < first_ttft.size(); ++index) {
            if (index != 0) json << ',';
            json << first_ttft[index];
        }
        json << "],\n    \"" << first.name << "_decode\":[";
        for (std::size_t index = 0; index < first_decode.size(); ++index) {
            if (index != 0) json << ',';
            json << first_decode[index];
        }
        json << "],\n    \"" << first.name << "_total\":[";
        for (std::size_t index = 0; index < first_total.size(); ++index) {
            if (index != 0) json << ',';
            json << first_total[index];
        }
        json << "],\n    \"" << second.name << "_prefill\":[";
        for (std::size_t index = 0; index < second_prefill.size(); ++index) {
            if (index != 0) json << ',';
            json << second_prefill[index];
        }
        json << "],\n    \"" << second.name << "_ttft\":[";
        for (std::size_t index = 0; index < second_ttft.size(); ++index) {
            if (index != 0) json << ',';
            json << second_ttft[index];
        }
        json << "],\n    \"" << second.name << "_decode\":[";
        for (std::size_t index = 0; index < second_decode.size(); ++index) {
            if (index != 0) json << ',';
            json << second_decode[index];
        }
        json << "],\n    \"" << second.name << "_total\":[";
        for (std::size_t index = 0; index < second_total.size(); ++index) {
            if (index != 0) json << ',';
            json << second_total[index];
        }
        json << "]\n  }\n}\n";

        if (!options.json_output.empty()) {
            std::ofstream output(options.json_output);
            if (!output) throw std::runtime_error("cannot open JSON output: " + options.json_output);
            output << json.str();
        }

        const auto print = [&](const std::string& name, const std::vector<Measurement>& measurements) {
            const auto decode = summarize(field_samples(measurements, &Measurement::decode_ms));
            std::cout << name << " decode: " << decode.mean_ms << " ms ("
                      << throughput(decode, options.decode_tokens) << " tok/s), samples="
                      << measurements.size() << '\n';
        };
        std::cout << (layer_output_mode
                          ? "M5-C6b interleaved Qwen3 layer-output A/B benchmark\n"
                          : kv_cache_mode
                          ? "M5-C6c interleaved Qwen3 KV-cache A/B benchmark\n"
                          : "M5-C3 interleaved Qwen3 attention A/B benchmark\n")
                  << "model: " << model.model_name() << "\n"
                  << "pairs: " << options.pairs << "\n"
                  << "measured_decode_tokens: " << options.decode_tokens << "\n"
                  << "order: ";
        for (std::size_t index = 0; index < order.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << order[index];
        }
        std::cout << '\n';
        print(first.name, first.measurements);
        print(second.name, second.measurements);
        const double first_throughput = throughput(first_decode_stats, options.decode_tokens);
        const double second_throughput = throughput(second_decode_stats, options.decode_tokens);
        const double speedup = first_throughput > 0.0
            ? second_throughput / first_throughput : 0.0;
        std::cout << "speedup: " << speedup << "x\n"
                  << "correctness: deterministic, finite, identical generated IDs\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M5-C3 attention A/B benchmark error: " << error.what() << '\n';
        return 1;
    }
}
