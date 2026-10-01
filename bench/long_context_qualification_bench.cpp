#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;
using namespace miinfer::prefill_v2;

namespace {

constexpr std::size_t kDecodeTokens = 128;
constexpr std::size_t kTotalRuns = 3; // one warm-up, two recorded repetitions
constexpr const char* kDefaultModelPath = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";

struct Run {
    double prefill_ms = 0.0;
    double prefill_tok_s = 0.0;
    double decode_ms_per_token = 0.0;
    double decode_tok_s = 0.0;
    double total_ms = 0.0;
    std::size_t generated_tokens = 0;
    std::uint64_t token_hash = 1469598103934665603ULL;
    bool valid = false;
    std::vector<std::uint32_t> generated_token_ids;
};

std::vector<std::uint32_t> make_prompt(std::size_t count) {
    std::vector<std::uint32_t> tokens(count);
    std::uint32_t value = 42;
    for (auto& token : tokens) {
        value = value * 1664525u + 1013904223u;
        token = value % 151643u + 1;
    }
    return tokens;
}

std::uint64_t hash_tokens(std::span<const std::uint32_t> tokens) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto token : tokens) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(token >> shift);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

std::size_t number(const char* text, const char* name) {
    std::size_t used = 0;
    const std::string value(text);
    const auto parsed = std::stoull(value, &used);
    if (used != value.size() || parsed == 0) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

} // namespace

int main(int argc, char** argv) try {
    std::string model_path = kDefaultModelPath;
    std::size_t capacity = 0;
    std::string json_output;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json-output" && i + 1 < argc) json_output = argv[++i];
        else if (capacity == 0 && arg != "--help") capacity = number(argv[i], "capacity");
        else if (arg == "--help") {
            std::cout << "usage: miinfer-long-context-qualification-bench CAPACITY [MODEL] [--json-output PATH]\n";
            return 0;
        } else if (model_path == kDefaultModelPath) model_path = arg;
        else throw std::runtime_error("unexpected argument: " + arg);
    }
    if (capacity != 16384 && capacity != 32768 && capacity != 65536 && capacity != 131072) {
        throw std::runtime_error("capacity must be one of 16384, 32768, 65536, 131072");
    }

    const std::size_t prompt_count = capacity - kDecodeTokens;
    const auto prompt = make_prompt(prompt_count);
    int device_id = 0;
    MIINFER_HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t props{};
    MIINFER_HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    if (std::string(props.gcnArchName).find("gfx906") != 0) {
        throw std::runtime_error(std::string("unsupported GPU target: ") + props.gcnArchName);
    }

    std::cout << "device=" << props.name << " arch=" << props.gcnArchName
              << " capacity=" << capacity << " prompt=" << prompt_count
              << " decode=128 final_position=" << capacity << '\n';
    const auto qwen_model = Qwen35Model::load(model_path);
    PrefillV2Model model(qwen_model, static_cast<std::uint32_t>(capacity), true);

    std::size_t free_before = 0, total_bytes = 0;
    MIINFER_HIP_CHECK(hipMemGetInfo(&free_before, &total_bytes));
    std::size_t kv_bytes = 0;
    for (const auto& kv : model.kv_caches()) kv_bytes += kv.total_bytes();
    const std::size_t recurrent_bytes = model.recurrent_states().size()
        * (RecurrentLayerState::kStateBytes + RecurrentLayerState::kConvHistoryBytes);

    std::vector<Run> runs;
    runs.reserve(kTotalRuns);
    GenerateOptions options;
    options.max_new_tokens = kDecodeTokens;
    options.reset_state_before = true;
    options.use_hip_graph = true;
    options.stop_token_ids.clear();
    options.temperature = 0.0f;
    for (std::size_t i = 0; i < kTotalRuns; ++i) {
        const auto stats = model.generate(prompt, options);
        Run run;
        run.prefill_ms = stats.prefill_ms;
        run.prefill_tok_s = stats.prefill_tok_per_sec;
        run.decode_ms_per_token = stats.avg_decode_latency_ms;
        run.decode_tok_s = stats.decode_tok_per_sec;
        run.total_ms = stats.total_ms;
        run.generated_tokens = stats.generated_tokens.size();
        run.token_hash = hash_tokens(stats.generated_tokens);
        run.generated_token_ids = stats.generated_tokens;
        run.valid = run.generated_tokens == kDecodeTokens
            && std::all_of(stats.generated_tokens.begin(), stats.generated_tokens.end(),
                [&](std::uint32_t token) { return token < qwen_model.config().vocab_size; })
            && std::isfinite(run.prefill_ms) && run.prefill_ms > 0.0
            && std::isfinite(run.prefill_tok_s) && run.prefill_tok_s > 0.0
            && std::isfinite(run.decode_ms_per_token) && run.decode_ms_per_token > 0.0
            && std::isfinite(run.decode_tok_s) && run.decode_tok_s > 0.0
            && std::isfinite(run.total_ms) && run.total_ms > 0.0;
        runs.push_back(run);
        std::cout << "run=" << i << " warmup=" << (i == 0 ? "true" : "false")
                  << " prefill_ms=" << run.prefill_ms << " prefill_tok_s=" << run.prefill_tok_s
                  << " decode_ms_per_token=" << run.decode_ms_per_token
                  << " generated=" << run.generated_tokens << " token_hash=" << run.token_hash
                  << " valid=" << (run.valid ? "true" : "false") << '\n';
        if (!run.valid) throw std::runtime_error("generation correctness gate failed");
    }
    const bool repeat_equal = runs[1].generated_token_ids == runs[2].generated_token_ids;
    std::size_t free_after = 0;
    MIINFER_HIP_CHECK(hipMemGetInfo(&free_after, &total_bytes));
    if (!repeat_equal) throw std::runtime_error("repeated generated token streams differ");

    if (!json_output.empty()) {
        std::ofstream out(json_output);
        if (!out) throw std::runtime_error("cannot write JSON output: " + json_output);
        out << std::setprecision(12)
            << "{\n  \"device\": \"" << props.name << "\",\n"
            << "  \"architecture\": \"" << props.gcnArchName << "\",\n"
            << "  \"context_capacity\": " << capacity << ",\n"
            << "  \"prompt_tokens\": " << prompt_count << ",\n"
            << "  \"decode_tokens_requested\": " << kDecodeTokens << ",\n"
            << "  \"final_active_context_position\": " << capacity << ",\n"
            << "  \"kv_bytes\": " << kv_bytes << ",\n"
            << "  \"kv_bytes_per_token\": " << (kv_bytes / capacity) << ",\n"
            << "  \"recurrent_state_bytes\": " << recurrent_bytes << ",\n"
            << "  \"weight_bytes\": " << model.persistent_weight_bytes() << ",\n"
            << "  \"workspace_bytes\": " << model.workspace_bytes() << ",\n"
            << "  \"activation_bytes\": " << model.activation_bytes() << ",\n"
            << "  \"cached_state_bytes\": " << model.cached_state_bytes() << ",\n"
            << "  \"total_vram_bytes\": " << model.total_vram_bytes() << ",\n"
            << "  \"device_total_bytes\": " << total_bytes << ",\n"
            << "  \"device_free_before_bytes\": " << free_before << ",\n"
            << "  \"device_free_after_bytes\": " << free_after << ",\n"
            << "  \"correctness\": \"PASS\",\n"
            << "  \"repeat_token_stream_match\": " << (repeat_equal ? "true" : "false") << ",\n"
            << "  \"runs\": [\n";
        for (std::size_t i = 0; i < runs.size(); ++i) {
            const auto& run = runs[i];
            out << "    {\"warmup\": " << (i == 0 ? "true" : "false")
                << ", \"prefill_ms\": " << run.prefill_ms
                << ", \"prefill_tok_s\": " << run.prefill_tok_s
                << ", \"decode_ms_per_token\": " << run.decode_ms_per_token
                << ", \"decode_tok_s\": " << run.decode_tok_s
                << ", \"total_ms\": " << run.total_ms
                << ", \"generated_tokens\": " << run.generated_tokens
                << ", \"token_hash\": " << run.token_hash
                << ", \"valid\": " << (run.valid ? "true" : "false")
                << ", \"generated_token_ids\": [";
            for (std::size_t token = 0; token < run.generated_token_ids.size(); ++token) {
                if (token != 0) out << ", ";
                out << run.generated_token_ids[token];
            }
            out << "]}" << (i + 1 == runs.size() ? "\n" : ",\n");
        }
        out << "  ]\n}\n";
        if (!out) throw std::runtime_error("failed while writing JSON output: " + json_output);
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "long-context qualification failed: " << error.what() << '\n';
    return 1;
}
