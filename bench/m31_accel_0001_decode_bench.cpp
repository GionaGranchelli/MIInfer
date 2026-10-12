#include "miinfer/build_config.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen3_tokenizer.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kKvCapacityAlignment = 16384;
constexpr std::size_t kMaxKvCapacity = 81920;

struct Options {
    std::string model_path;
    std::string json_path;
    std::size_t prompt_tokens = 0;
    std::size_t decode_tokens = 0;
    std::size_t iterations = 0;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint32_t> make_prompt(
    std::size_t count, std::uint32_t vocabulary_size,
    bool add_bos, std::uint32_t bos_id) {
    if (count == 0 || vocabulary_size == 0 || (add_bos && bos_id >= vocabulary_size)) {
        throw std::invalid_argument("invalid deterministic prompt dimensions");
    }
    std::vector<std::uint32_t> result;
    result.reserve(count);
    std::uint32_t state = 0x6d31a001U;
    for (std::size_t index = 0; index < count; ++index) {
        if (index == 0 && add_bos) {
            result.push_back(bos_id);
        } else {
            state = state * 1664525U + 1013904223U;
            result.push_back(state % vocabulary_size);
        }
    }
    return result;
}

std::string ids_sha256(std::span<const std::uint32_t> ids) {
    return miinfer::sha256_bytes(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(ids.data()), ids.size_bytes()));
}

void validate_workload(std::size_t prompt, std::size_t decode, std::size_t iterations) {
    if (prompt == 0 || decode == 0 || iterations == 0
        || prompt > kMaxKvCapacity || decode > kMaxKvCapacity - prompt
        || decode >= std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("P and TG must be positive and fit the 81920-token campaign limit");
    }
}

bool valid_timing_sample(const miinfer::prefill_v2::GenerateStats& sample) {
    return sample.used_hip_graph
        && std::isfinite(sample.ttft_ms) && sample.ttft_ms > 0.0
        && std::isfinite(sample.decode_ms) && sample.decode_ms > 0.0
        && std::isfinite(sample.decode_tok_per_sec) && sample.decode_tok_per_sec > 0.0;
}

std::size_t kv_capacity(std::size_t prompt, std::size_t decode) {
    const auto needed = prompt + decode;
    const auto aligned = ((needed + kKvCapacityAlignment - 1) / kKvCapacityAlignment)
        * kKvCapacityAlignment;
    return std::max(kKvCapacityAlignment, aligned);
}

void self_check() {
    const auto left = make_prompt(128, 32000, true, 1);
    const auto right = make_prompt(128, 32000, true, 1);
    require(left == right, "deterministic prompt generation changed between calls");
    require(left.size() == 128 && left.front() == 1, "prompt shape or BOS placement is wrong");
    require(std::all_of(left.begin(), left.end(), [](auto id) { return id < 32000; }),
            "prompt contains an out-of-vocabulary token");
    validate_workload(2048, 128, 5);
    miinfer::prefill_v2::GenerateStats valid_sample;
    valid_sample.used_hip_graph = true;
    valid_sample.ttft_ms = 1.0;
    valid_sample.decode_ms = 1.0;
    valid_sample.decode_tok_per_sec = 1.0;
    require(valid_timing_sample(valid_sample), "valid HIP Graph timing was rejected");
    valid_sample.used_hip_graph = false;
    require(!valid_timing_sample(valid_sample), "HIP Graph fallback was accepted");
    valid_sample.used_hip_graph = true;
    valid_sample.decode_ms = std::numeric_limits<double>::infinity();
    require(!valid_timing_sample(valid_sample), "non-finite decode time was accepted");
    require(kv_capacity(65536, 128) == 81920,
            "64K decode workload did not select the campaign KV capacity");
    bool rejected = false;
    try { validate_workload(81920, 1, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "KV-capacity overflow was not rejected");
    std::cout << "M31 decode benchmark self-check: PASS (deterministic prompt, bounds)\n";
}

Options parse(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        std::cerr << "usage: miinfer-m31-accel-0001-decode-bench MODEL.gguf P TG ITERATIONS [JSON]\n"
                     "       miinfer-m31-accel-0001-decode-bench --self-check\n";
        throw std::invalid_argument("usage");
    }
    if (argc < 5 || argc > 6) throw std::invalid_argument("expected MODEL.gguf P TG ITERATIONS [JSON]");
    Options result;
    result.model_path = argv[1];
    result.prompt_tokens = std::stoull(argv[2]);
    result.decode_tokens = std::stoull(argv[3]);
    result.iterations = std::stoull(argv[4]);
    if (argc == 6) result.json_path = argv[5];
    validate_workload(result.prompt_tokens, result.decode_tokens, result.iterations);
    return result;
}

void write_ids(std::ostream& out, const std::vector<std::uint32_t>& ids) {
    out << '[';
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index) out << ',';
        out << ids[index];
    }
    out << ']';
}

std::string result_json(
    const Options& options, const std::string& model_sha,
    std::span<const std::uint32_t> prompt,
    std::size_t capacity,
    const std::vector<miinfer::prefill_v2::GenerateStats>& samples) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\"benchmark\":\"m31-accel-0001-production-graph-decode\","
        << "\"source_sha\":\"" << MIINFER_GIT_COMMIT << "\","
        << "\"source_dirty\":\"" << MIINFER_GIT_DIRTY << "\","
        << "\"build_type\":\"" << MIINFER_BUILD_TYPE << "\","
        << "\"model_sha256\":\"" << model_sha << "\","
        << "\"prompt_kind\":\"seeded_token_ids\","
        << "\"prompt_tokens\":" << prompt.size() << ','
        << "\"kv_capacity\":" << capacity << ','
        << "\"decode_tokens\":" << options.decode_tokens << ','
        << "\"iterations\":" << samples.size() << ','
        << "\"requested_route\":\"hip_graph\","
        << "\"prompt_id_sha256\":\"" << ids_sha256(prompt) << "\","
        << "\"samples\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index) out << ',';
        const auto& sample = samples[index];
        out << "{\"ttft_ms\":" << sample.ttft_ms
            << ",\"decode_ms\":" << sample.decode_ms
            << ",\"decode_tok_per_sec\":" << sample.decode_tok_per_sec
            << ",\"used_hip_graph\":" << (sample.used_hip_graph ? "true" : "false")
            << ",\"generated_ids_sha256\":\"" << ids_sha256(sample.generated_tokens)
            << "\",\"generated_ids\":";
        write_ids(out, sample.generated_tokens);
        out << '}';
    }
    out << "]}\n";
    return out.str();
}

void persist_json(const std::string& path, const std::string& json) {
    const auto temporary = path + ".tmp";
    {
        std::ofstream file(temporary, std::ios::trunc);
        if (!file) throw std::runtime_error("could not open temporary JSON output path");
        file << json;
        file.flush();
        if (!file) throw std::runtime_error("failed to write benchmark JSON");
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) throw std::runtime_error("could not atomically publish benchmark JSON: " + error.message());
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-check") {
        try { self_check(); return 0; }
        catch (const std::exception& error) {
            std::cerr << "self-check failed: " << error.what() << '\n';
            return 1;
        }
    }

    try {
        const auto options = parse(argc, argv);
        if (!options.json_path.empty() && std::filesystem::exists(options.json_path)) {
            throw std::runtime_error("JSON output already exists; choose a fresh capture path");
        }
        const auto model_data = miinfer::Qwen35Model::load(options.model_path);
        const auto tokenizer = miinfer::Qwen3Tokenizer::load(*model_data.file());
        if (tokenizer.vocabulary_size() != model_data.config().vocab_size) {
            throw std::runtime_error("tokenizer/model vocabulary mismatch");
        }
        if (tokenizer.vocabulary_size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("tokenizer vocabulary exceeds the token ID range");
        }
        const auto prompt = make_prompt(options.prompt_tokens,
            static_cast<std::uint32_t>(tokenizer.vocabulary_size()),
            tokenizer.add_bos(), tokenizer.bos_id());
        const auto capacity = kv_capacity(options.prompt_tokens, options.decode_tokens);
        miinfer::prefill_v2::PrefillV2Model model(model_data, capacity, true);
        const auto model_sha = miinfer::sha256_file(options.model_path);

        miinfer::prefill_v2::GenerateOptions generation;
        generation.max_new_tokens = options.decode_tokens + 1;
        generation.use_hip_graph = true;
        generation.temperature = 0.0f;
        generation.top_p = 1.0f;
        generation.top_k = 1;
        generation.stop_token_ids.clear();
        generation.repetition_penalty = 1.0f;
        generation.presence_penalty = 0.0f;
        generation.frequency_penalty = 0.0f;
        generation.seed = 7;

        std::vector<miinfer::prefill_v2::GenerateStats> samples;
        samples.reserve(options.iterations);
        for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
            auto sample = model.generate(prompt, generation);
            require(valid_timing_sample(sample),
                    "sample did not use HIP Graph or returned invalid timing metrics");
            require(sample.generated_tokens.size() == options.decode_tokens + 1,
                    "generation ended before the requested decode count");
            if (!samples.empty()) {
                require(sample.generated_tokens == samples.front().generated_tokens,
                        "repeated deterministic generation produced different token histories");
            }
            samples.push_back(std::move(sample));
            if (!options.json_path.empty()) {
                persist_json(options.json_path,
                    result_json(options, model_sha, prompt, capacity, samples));
            }
        }

        if (options.json_path.empty())
            std::cout << result_json(options, model_sha, prompt, capacity, samples);
        return 0;
    } catch (const std::exception& error) {
        if (std::string(error.what()) != "usage") {
            std::cerr << "decode benchmark error: " << error.what() << '\n';
            return 1;
        }
        return argc >= 2 && std::string(argv[1]) == "--help" ? 0 : 2;
    }
}
