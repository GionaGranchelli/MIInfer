#include "miinfer/build_config.hpp"
#include "miinfer/build_info.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_layer.hpp"
#include "miinfer/qwen3_tokenizer.hpp"
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

struct TimePoint {
    timespec value{};
};

TimePoint now() {
    TimePoint result;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &result.value) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC_RAW) failed");
    }
    return result;
}

struct Options {
    std::string model_path;
    std::string prompt = "hello";
    std::string prompt_ids;
    std::size_t prompt_repeat = 1;
    std::size_t generated_tokens = 8;
    std::size_t warmup = 1;
    std::size_t iterations = 3;
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
    double reset_ms = 0.0;
    double prefill_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    std::size_t decode_forward_tokens = 0;
    std::vector<std::uint32_t> generated_ids;
};

void usage() {
    std::cerr
        << "usage: miinfer-qwen3-inference-bench MODEL.gguf [options]\n"
        << "  --prompt TEXT          text prompt (default: hello)\n"
        << "  --prompt-ids CSV       explicit prompt IDs instead of text\n"
        << "  --prompt-repeat N      repeat prompt IDs N times (default: 1)\n"
        << "  --generated-tokens N   greedy tokens to request (default: 8)\n"
        << "  --warmup N             unmeasured runs (default: 1)\n"
        << "  --iterations N         measured runs (default: 3)\n"
        << "  --json-output PATH     write machine-readable result\n"
        << "  --version              print build information\n";
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
    std::size_t start = 0;
    while (start < value.size()) {
        const auto comma = value.find(',', start);
        result.push_back(parse_id(value.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start)));
        if (comma == std::string::npos) break;
        start = comma + 1;
        if (start == value.size()) throw std::invalid_argument("--prompt-ids has an empty ID");
    }
    return result;
}

bool parse_options(int argc, char** argv, Options& options, bool& informational) {
    if (argc < 2) {
        usage();
        return false;
    }
    options.model_path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            usage();
            informational = true;
            return false;
        }
        if (argument == "--version") {
            miinfer::print_build_info(std::cout);
            informational = true;
            return false;
        }
        if (index + 1 >= argc) throw std::invalid_argument("missing value for " + argument);
        if (argument == "--prompt") {
            options.prompt = argv[++index];
        } else if (argument == "--prompt-ids") {
            options.prompt_ids = argv[++index];
        } else if (argument == "--prompt-repeat") {
            options.prompt_repeat = parse_size(argv[++index], "--prompt-repeat");
        } else if (argument == "--generated-tokens") {
            options.generated_tokens = parse_size(argv[++index], "--generated-tokens");
        } else if (argument == "--warmup") {
            options.warmup = parse_size(argv[++index], "--warmup", true);
        } else if (argument == "--iterations") {
            options.iterations = parse_size(argv[++index], "--iterations");
        } else if (argument == "--json-output") {
            options.json_output = argv[++index];
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (!options.prompt_ids.empty() && options.prompt != "hello") {
        throw std::invalid_argument("--prompt and --prompt-ids cannot be combined");
    }
    return true;
}

std::string json_escape(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': result << "\\\""; break;
        case '\\': result << "\\\\"; break;
        case '\n': result << "\\n"; break;
        case '\r': result << "\\r"; break;
        case '\t': result << "\\t"; break;
        default:
            if (character < 0x20U) {
                result << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned>(character) << std::dec << std::setfill(' ');
            } else {
                result << static_cast<char>(character);
            }
            break;
        }
    }
    result << '"';
    return result.str();
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

std::uint32_t argmax(const std::vector<float>& logits) {
    if (logits.empty()) throw std::runtime_error("model returned empty logits");
    for (const float value : logits) {
        if (!std::isfinite(value)) throw std::runtime_error("model returned non-finite logits");
    }
    return static_cast<std::uint32_t>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
}

void validate_cache(const miinfer::Qwen3GpuDecodeCache& cache, std::size_t expected) {
    if (cache.length() != expected) throw std::runtime_error("decode cache length mismatch");
    for (std::size_t layer = 0; layer < cache.layers(); ++layer) {
        if (cache.layer(layer).length() != expected) {
            throw std::runtime_error("per-layer decode cache length mismatch");
        }
    }
}

double elapsed_ms(TimePoint start, TimePoint end) {
    const auto seconds = static_cast<double>(end.value.tv_sec - start.value.tv_sec);
    const auto nanoseconds = static_cast<double>(end.value.tv_nsec - start.value.tv_nsec);
    return seconds * 1000.0 + nanoseconds / 1000000.0;
}

Measurement run_once(
    const Options& options,
    const miinfer::Qwen3GpuPlan& plan,
    const std::vector<std::uint32_t>& prompt_ids,
    miinfer::Qwen3GpuDecodeCache& cache,
    std::uint32_t eos_id) {
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    const auto reset_start = now();
    cache.reset();
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    const auto reset_end = now();

    const auto total_start = reset_start;
    const auto prefill_start = reset_end;
    miinfer::Qwen3ForwardTrace trace;
    for (std::size_t position = 0; position < prompt_ids.size(); ++position) {
        trace = miinfer::execute_qwen3_decode_gpu(plan, prompt_ids[position], position, cache);
    }
    const auto prefill_end = now();
    validate_cache(cache, prompt_ids.size());

    std::vector<std::uint32_t> generated;
    generated.reserve(options.generated_tokens);
    generated.push_back(argmax(trace.logits));
    const auto ttft_end = now();

    const auto decode_start = ttft_end;
    for (std::size_t step = 1; step < options.generated_tokens; ++step) {
        const auto token = generated.back();
        if (token == eos_id) break;
        trace = miinfer::execute_qwen3_decode_gpu(
            plan, token, prompt_ids.size() + step - 1, cache);
        generated.push_back(argmax(trace.logits));
    }
    const auto decode_end = now();
    validate_cache(cache, prompt_ids.size() + generated.size() - 1);

    Measurement measurement;
    measurement.reset_ms = elapsed_ms(reset_start, reset_end);
    measurement.prefill_ms = elapsed_ms(prefill_start, prefill_end);
    measurement.ttft_ms = elapsed_ms(total_start, ttft_end);
    measurement.decode_ms = elapsed_ms(decode_start, decode_end);
    measurement.total_ms = elapsed_ms(total_start, decode_end);
    measurement.decode_forward_tokens = generated.size() - 1;
    measurement.generated_ids = std::move(generated);
    return measurement;
}

Stats summarize(const std::vector<double>& samples) {
    if (samples.empty()) throw std::runtime_error("cannot summarize empty timing samples");
    Stats result;
    result.min_ms = *std::min_element(samples.begin(), samples.end());
    result.max_ms = *std::max_element(samples.begin(), samples.end());
    for (const double value : samples) result.mean_ms += value;
    result.mean_ms /= static_cast<double>(samples.size());
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2;
    result.median_ms = sorted.size() % 2 == 0
        ? (sorted[middle - 1] + sorted[middle]) / 2.0
        : sorted[middle];
    double squared = 0.0;
    for (const double value : samples) {
        const double delta = value - result.mean_ms;
        squared += delta * delta;
    }
    result.stddev_ms = std::sqrt(squared / static_cast<double>(samples.size()));
    return result;
}

double throughput(const Stats& stats, std::size_t tokens) {
    return tokens == 0 || stats.mean_ms <= 0.0
        ? 0.0 : static_cast<double>(tokens) * 1000.0 / stats.mean_ms;
}

void write_stats(std::ostream& output, const Stats& stats, std::size_t tokens) {
    output << std::fixed << std::setprecision(6)
           << "{\"tokens\":" << tokens
           << ",\"mean_ms\":" << stats.mean_ms
           << ",\"median_ms\":" << stats.median_ms
           << ",\"min_ms\":" << stats.min_ms
           << ",\"max_ms\":" << stats.max_ms
           << ",\"stddev_ms\":" << stats.stddev_ms
           << ",\"mean_tokens_per_s\":" << throughput(stats, tokens)
           << '}';
}

std::string build_json(
    const Options& options,
    const miinfer::Qwen3Model& model,
    const miinfer::Qwen3GpuPlan& plan,
    const std::vector<std::uint32_t>& prompt_ids,
    const std::vector<Measurement>& measurements,
    const std::vector<double>& reset_samples,
    const std::vector<double>& prefill_samples,
    const std::vector<double>& ttft_samples,
    const std::vector<double>& decode_samples,
    const std::vector<double>& total_samples) {
    const auto& config = model.config();
    const auto reset = summarize(reset_samples);
    const auto prefill = summarize(prefill_samples);
    const auto ttft = summarize(ttft_samples);
    const auto decode = summarize(decode_samples);
    const auto total = summarize(total_samples);
    const auto& first = measurements.front();
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"benchmark\":\"m5a_qwen3_inference\",\n"
           << "  \"experiment\":\"M5-A\",\n"
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
           << "  \"layer_count\":" << config.layer_count << ",\n"
           << "  \"hidden_size\":" << config.hidden_size << ",\n"
           << "  \"vocab_size\":" << config.vocab_size << ",\n"
           << "  \"prompt_source\":\"" << (options.prompt_ids.empty() ? "text" : "explicit_ids") << "\",\n"
           << "  \"prompt_text\":" << json_escape(options.prompt) << ",\n"
           << "  \"prompt_ids\":" << ids_json(prompt_ids) << ",\n"
           << "  \"prompt_repeat\":" << options.prompt_repeat << ",\n"
           << "  \"requested_generated_tokens\":" << options.generated_tokens << ",\n"
           << "  \"warmup_runs\":" << options.warmup << ",\n"
           << "  \"measured_runs\":" << options.iterations << ",\n"
           << "  \"timing_method\":\"steady_clock_wall_time; model and plan setup excluded\",\n"
           << "  \"prefill_definition\":\"sequential batch-1 prompt ingestion through decode API\",\n"
           << "  \"ttft_definition\":\"reset plus prompt ingestion through first greedy argmax\",\n"
           << "  \"decode_definition\":\"forward calls after the first selected token; argmax included\",\n"
           << "  \"generated_ids\":" << ids_json(first.generated_ids) << ",\n"
           << "  \"actual_generated_tokens\":" << first.generated_ids.size() << ",\n"
           << "  \"decode_forward_tokens\":" << first.decode_forward_tokens << ",\n"
           << "  \"reset\":";
    write_stats(output, reset, 0);
    output << ",\n  \"prefill\":";
    write_stats(output, prefill, prompt_ids.size());
    output << ",\n  \"ttft\":";
    write_stats(output, ttft, 1);
    output << ",\n  \"decode\":";
    write_stats(output, decode, first.decode_forward_tokens);
    output << ",\n  \"total\":";
    write_stats(output, total, prompt_ids.size() + first.generated_ids.size());
    output << ",\n  \"samples_ms\":{\n"
           << "    \"reset\":[";
    for (std::size_t index = 0; index < reset_samples.size(); ++index) {
        if (index != 0) output << ',';
        output << reset_samples[index];
    }
    output << "],\n    \"prefill\":[";
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
        bool informational = false;
        if (!parse_options(argc, argv, options, informational)) return informational ? 0 : 2;
        const auto model = miinfer::Qwen3Model::load(options.model_path);
        const auto tokenizer = miinfer::Qwen3Tokenizer::load(model);
        const auto base_prompt = options.prompt_ids.empty()
            ? tokenizer.encode(options.prompt) : parse_ids(options.prompt_ids);
        if (base_prompt.empty()) throw std::invalid_argument("prompt produced no token IDs");
        std::vector<std::uint32_t> prompt_ids;
        prompt_ids.reserve(base_prompt.size() * options.prompt_repeat);
        for (std::size_t repeat = 0; repeat < options.prompt_repeat; ++repeat) {
            prompt_ids.insert(prompt_ids.end(), base_prompt.begin(), base_prompt.end());
        }
        if (prompt_ids.size() + options.generated_tokens > model.config().context_length) {
            throw std::invalid_argument("prompt and generation exceed model context");
        }

        const auto plan = miinfer::Qwen3GpuPlan::build(model);
        miinfer::Qwen3GpuDecodeCache cache(
            model.config().layer_count, model.config().kv_heads, model.config().head_dim,
            prompt_ids.size() + options.generated_tokens);
        for (std::size_t run = 0; run < options.warmup; ++run) {
            (void)run_once(options, plan, prompt_ids, cache, tokenizer.eos_id());
        }

        std::vector<Measurement> measurements;
        measurements.reserve(options.iterations);
        for (std::size_t run = 0; run < options.iterations; ++run) {
            auto measurement = run_once(options, plan, prompt_ids, cache, tokenizer.eos_id());
            if (!measurements.empty()
                && measurement.generated_ids != measurements.front().generated_ids) {
                throw std::runtime_error("generated token sequence was not deterministic");
            }
            measurements.push_back(std::move(measurement));
        }

        const auto& first = measurements.front();
        std::vector<double> reset_samples;
        std::vector<double> prefill_samples;
        std::vector<double> ttft_samples;
        std::vector<double> decode_samples;
        std::vector<double> total_samples;
        for (const auto& measurement : measurements) {
            reset_samples.push_back(measurement.reset_ms);
            prefill_samples.push_back(measurement.prefill_ms);
            ttft_samples.push_back(measurement.ttft_ms);
            decode_samples.push_back(measurement.decode_ms);
            total_samples.push_back(measurement.total_ms);
        }
        const auto result = build_json(
            options, model, plan, prompt_ids, measurements, reset_samples, prefill_samples,
            ttft_samples, decode_samples, total_samples);
        if (!options.json_output.empty()) {
            std::ofstream output(options.json_output);
            if (!output) throw std::runtime_error("cannot open JSON output: " + options.json_output);
            output << result;
        }

        const auto prefill = summarize(prefill_samples);
        const auto ttft = summarize(ttft_samples);
        const auto decode = summarize(decode_samples);
        std::cout << "M5-A Qwen3 inference baseline\n"
                  << "model: " << model.model_name() << "\n"
                  << "prompt_tokens: " << prompt_ids.size() << "\n"
                  << "generated_tokens: " << first.generated_ids.size() << "\n"
                  << std::fixed << std::setprecision(3)
                  << "prefill: " << prefill.mean_ms << " ms ("
                  << throughput(prefill, prompt_ids.size()) << " tok/s)\n"
                  << "ttft: " << ttft.mean_ms << " ms\n"
                  << "decode: " << decode.mean_ms << " ms ("
                  << throughput(decode, first.decode_forward_tokens) << " tok/s)\n"
                  << "generated_ids: " << ids_json(first.generated_ids) << '\n'
                  << "correctness: deterministic and finite\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M5-A inference benchmark error: " << error.what() << '\n';
        return 1;
    }
}
