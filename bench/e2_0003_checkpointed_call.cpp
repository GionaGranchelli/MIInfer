#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/sha256.hpp"
#include "e2_0003_prompt.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Context {
    const char* name;
    std::size_t tokens;
    std::uint32_t seed;
};

std::vector<std::uint32_t> read_tokens(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open reference token file: " + path);
    std::vector<std::uint32_t> result;
    std::uint64_t token = 0;
    while (input >> token) {
        if (token > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("reference token exceeds uint32 range");
        }
        result.push_back(static_cast<std::uint32_t>(token));
    }
    if (!input.eof()) throw std::runtime_error("invalid token in reference token file");
    return result;
}

std::size_t parse_size(const std::string& value, const char* name, bool allow_zero = false) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || (!allow_zero && parsed == 0)) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

void event(std::ostream& out, const char* name, std::size_t iteration,
           const std::string& detail = {}) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    out << "event=" << name << " iteration=" << iteration << " epoch_ms=" << ms;
    if (!detail.empty()) out << ' ' << detail;
    out << '\n' << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: e2-0003-checkpointed-call MODEL.gguf "
                     "8K|32K|64K|128K [--generate N] [--iterations N] [--warmup N] "
                     "[--prompt-tokens N] [--reference-token-ids FILE] [--output FILE] [--ledger-only]\n";
        return 2;
    }

    try {
        const std::string model_path = argv[1];
        const std::string context_name = argv[2];
        const Context contexts[] = {{"8K", 8192, 77}, {"32K", 32768, 123},
                                    {"64K", 65536, 456}, {"128K", 131072, 789}};
        const auto found = std::find_if(std::begin(contexts), std::end(contexts),
            [&](const Context& c) {
                return context_name == c.name || context_name == std::to_string(c.tokens);
            });
        if (found == std::end(contexts)) throw std::runtime_error("unsupported context");

        std::size_t generate_tokens = 128;
        std::size_t prompt_tokens = found->tokens;
        std::size_t iterations = 1;
        std::size_t warmups = 0;
        std::string reference_path;
        std::string output_path;
        bool ledger_only = false;
        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--generate" && i + 1 < argc) generate_tokens = parse_size(argv[++i], "generate count");
            else if (arg == "--prompt-tokens" && i + 1 < argc) prompt_tokens = parse_size(argv[++i], "prompt token count");
            else if ((arg == "--iterations" || arg == "--repetitions") && i + 1 < argc)
                iterations = parse_size(argv[++i], "iteration count");
            else if (arg == "--warmup" && i + 1 < argc) warmups = parse_size(argv[++i], "warmup count", true);
            else if (arg == "--reference-token-ids" && i + 1 < argc) reference_path = argv[++i];
            else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
            else if (arg == "--ledger-only") ledger_only = true;
            else throw std::runtime_error("unknown or incomplete option: " + arg);
        }
        if (prompt_tokens > found->tokens || generate_tokens > prompt_tokens || iterations > 100 || warmups > 20) {
            throw std::runtime_error("requested generation/iteration/warmup count is out of range");
        }

        std::ofstream output_file;
        std::ostream* output = &std::cout;
        if (!output_path.empty()) {
            if (std::filesystem::exists(output_path)) {
                throw std::runtime_error("refusing to overwrite existing output file: " + output_path);
            }
            output_file.open(output_path);
            if (!output_file) throw std::runtime_error("cannot create output file: " + output_path);
            output = &output_file;
        }
        auto& out = *output;

        event(out, "MODEL_HASH_BEGIN", 0);
        const auto model_hash = miinfer::sha256_file(model_path);
        event(out, "MODEL_HASH_READY", 0);
        event(out, "MODEL_LOAD_BEGIN", 0);
        const auto model_data = miinfer::Qwen35Model::load(model_path);
        event(out, "MODEL_DATA_READY", 0);
        event(out, "MODEL_ALLOC_BEGIN", 0);
        const auto context_capacity_tokens = static_cast<std::uint32_t>(prompt_tokens + 256);
        miinfer::prefill_v2::PrefillV2Model model(
            model_data, context_capacity_tokens, true);
        event(out, "MODEL_ALLOC_READY", 0);
        const auto prompt = e2_0003::make_prompt(prompt_tokens, found->seed);
        const auto expected = reference_path.empty() ? std::vector<std::uint32_t>{} : read_tokens(reference_path);

        out << "benchmark=E2-0003-clean-source\ncontext="
            << (prompt_tokens == found->tokens ? found->name : "custom")
            << "\ncontext_profile=" << found->name
            << "\nprompt_tokens=" << prompt_tokens
            << "\ncontext_capacity_tokens=" << context_capacity_tokens
            << "\ngenerate_tokens=" << generate_tokens
            << "\niterations=" << iterations
            << "\nwarmup=" << warmups
            << "\nmodel_sha256=" << model_hash
            << "\nprompt_fingerprint_fnv1a64=" << std::hex << e2_0003::prompt_fingerprint(prompt) << std::dec
            << "\ntemperature=0\ntop_p=1\ntop_k=1\nrepetition_penalty=1"
            << "\nfrequency_penalty=0\npresence_penalty=0\nstop_token_ids=disabled"
            << "\nrepeat_last_n=256\nreset_state_before=true\nuse_hip_graph=true"
            << "\nseed=42\nsampler=greedy_argmax\n"
            << "\nmodel_vram_bytes=" << model.total_vram_bytes()
            << "\n" << std::flush;
        event(out, "MODEL_READY", 0);
        if (ledger_only) {
            std::size_t free_bytes = 0, total_bytes = 0;
            MIINFER_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
            out << "ledger={\"model_weights_bytes\":" << model.persistent_weight_bytes()
                << ",\"persistent_state_bytes\":" << model.persistent_state_bytes()
                << ",\"workspace_bytes\":" << model.workspace_bytes()
                << ",\"activation_bytes\":" << model.activation_bytes()
                << ",\"cached_state_bytes\":" << model.cached_state_bytes()
                << ",\"snapshot_bytes\":" << model.snapshot_bytes()
                << ",\"model_total_bytes\":" << model.total_vram_bytes()
                << ",\"splitk_splits\":" << model.workspace().splitk_splits
                << ",\"device_used_bytes\":" << total_bytes - free_bytes
                << ",\"device_free_bytes\":" << free_bytes << "}\n" << std::flush;
            return 0;
        }

        const auto execute = [&](std::size_t iteration, bool measured) {
            miinfer::prefill_v2::GenerateOptions options;
            options.max_new_tokens = generate_tokens;
            options.reset_state_before = true;
            options.use_hip_graph = true;
            options.temperature = 0.0f;
            options.top_p = 1.0f;
            options.top_k = 1;
            options.repetition_penalty = 1.0f;
            options.frequency_penalty = 0.0f;
            options.presence_penalty = 0.0f;
            options.repeat_last_n = 256;
            options.stop_token_ids.clear();
            options.seed = 42;
            std::size_t token_count = 0;
            options.on_token = [&](std::uint32_t token) {
                const bool first_token = token_count == 0;
                if (first_token) {
                    event(out, "PREFILL_END", iteration);
                    event(out, "DECODE_BEGIN", iteration);
                }
                ++token_count;
                (void)token;
                return true;
            };

            event(out, "RUN_BEGIN", iteration, measured ? "mode=measure" : "mode=warmup");
            event(out, "PREFILL_BEGIN", iteration);
            std::size_t free_before = 0, total_before = 0;
            if (measured) MIINFER_HIP_CHECK(hipMemGetInfo(&free_before, &total_before));
            const auto wall_start = std::chrono::steady_clock::now();
            const auto stats = model.generate(prompt, options);
            const auto wall_end = std::chrono::steady_clock::now();
            if (token_count == 0) event(out, "PREFILL_END", iteration);
            event(out, "DECODE_END", iteration);

            bool tokens_valid = stats.generated_tokens.size() == generate_tokens;
            for (const auto token : stats.generated_tokens) {
                tokens_valid = tokens_valid && token < model.vocab_size();
            }
            const bool parity_checked = !expected.empty();
            const bool parity = parity_checked && expected.size() >= stats.generated_tokens.size()
                && std::equal(stats.generated_tokens.begin(), stats.generated_tokens.end(), expected.begin());
            double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
            event(out, "RUN_END", iteration, "wall_ms=" + std::to_string(wall_ms));
            if (!measured) return tokens_valid;

            std::size_t free_after = 0, total_after = 0;
            MIINFER_HIP_CHECK(hipMemGetInfo(&free_after, &total_after));
            const auto device_used_before = total_before - free_before;
            const auto device_used_after = total_after - free_after;
            const auto model_vram_bytes = model.total_vram_bytes();
            const auto device_residual_after = device_used_after > model_vram_bytes
                ? device_used_after - model_vram_bytes : 0;
            out << std::fixed << std::setprecision(3)
                << "result={\"context\":\"" << found->name
                << "\",\"iteration\":" << iteration
                << ",\"wall_ms\":" << wall_ms
                << ",\"prefill_ms\":" << stats.prefill_ms
                << ",\"decode_ms\":" << stats.decode_ms
                << ",\"prefill_tok_s\":" << stats.prefill_tok_per_sec
                << ",\"decode_tok_s\":" << stats.decode_tok_per_sec
                << ",\"used_hip_graph\":" << (stats.used_hip_graph ? "true" : "false")
                << ",\"generated_tokens\":" << stats.generated_tokens.size()
                << ",\"model_vram_bytes\":" << model_vram_bytes
                << ",\"model_weights_bytes\":" << model.persistent_weight_bytes()
                << ",\"persistent_state_bytes\":" << model.persistent_state_bytes()
                << ",\"workspace_bytes\":" << model.workspace_bytes()
                << ",\"activation_bytes\":" << model.activation_bytes()
                << ",\"cached_state_bytes\":" << model.cached_state_bytes()
                << ",\"snapshot_bytes\":" << model.snapshot_bytes()
                << ",\"splitk_splits\":" << model.workspace().splitk_splits
                << ",\"device_total_bytes\":" << total_after
                << ",\"device_free_before_bytes\":" << free_before
                << ",\"device_free_after_bytes\":" << free_after
                << ",\"device_used_before_bytes\":" << device_used_before
                << ",\"device_used_after_bytes\":" << device_used_after
                << ",\"device_residual_after_bytes\":" << device_residual_after
                << ",\"device_peak_bytes\":null"
                << ",\"tokens_valid\":\"" << (tokens_valid ? "PASS" : "FAIL")
                << "\",\"output_parity\":\""
                << (parity_checked ? (parity ? "PASS" : "FAIL") : "NOT_CHECKED")
                << "\",\"token_ids\":[";
            for (std::size_t i = 0; i < stats.generated_tokens.size(); ++i) {
                if (i) out << ',';
                out << stats.generated_tokens[i];
            }
            out << "]}\n" << std::flush;
            event(out, "CALL_COMPLETE", iteration);
            return tokens_valid && (!parity_checked || parity);
        };

        for (std::size_t i = 1; i <= warmups; ++i) {
            if (!execute(i, false)) throw std::runtime_error("warmup produced invalid output length");
        }
        for (std::size_t i = 1; i <= iterations; ++i) {
            if (!execute(i, true)) throw std::runtime_error("measured call failed token validity/parity");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M31-0002 checkpointed call: FAIL: " << error.what() << '\n';
        return 1;
    }
}
