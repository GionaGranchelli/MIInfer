#include <chrono>

#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/sha256.hpp"
#include "e2_0003_prompt.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
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

std::vector<std::uint32_t> read_prompt_ids(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open prompt ID file: " + path);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.empty() || bytes.size() % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error("prompt ID file must contain little-endian uint32 IDs");
    }
    std::vector<std::uint32_t> result(bytes.size() / sizeof(std::uint32_t));
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<std::uint32_t>(bytes[i * 4])
            | (static_cast<std::uint32_t>(bytes[i * 4 + 1]) << 8)
            | (static_cast<std::uint32_t>(bytes[i * 4 + 2]) << 16)
            | (static_cast<std::uint32_t>(bytes[i * 4 + 3]) << 24);
    }
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

void write_bytes(const std::filesystem::path& path, const void* data, std::size_t size) {
    if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite " + path.string());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

void capture_state(miinfer::prefill_v2::PrefillV2Model& model,
                   const std::string& prefix, std::size_t active_tokens) {
    using miinfer::prefill_v2::RecurrentLayerState;
    std::vector<float> recurrent;
    recurrent.reserve(model.recurrent_states().size()
        * (RecurrentLayerState::kStateElements + RecurrentLayerState::kConvHistoryElements));
    for (const auto& storage : model.recurrent_states()) {
        std::vector<float> state, history;
        storage.download(state, history);
        recurrent.insert(recurrent.end(), state.begin(), state.end());
        recurrent.insert(recurrent.end(), history.begin(), history.end());
    }
    write_bytes(prefix + ".recurrent.f32", recurrent.data(), recurrent.size() * sizeof(float));

    std::vector<std::uint8_t> kv;
    for (const auto& cache : model.kv_caches()) {
        const auto size = cache.raw_tokens_bytes(active_tokens);
        const auto offset = kv.size();
        kv.resize(offset + size);
        cache.download_raw(kv.data() + offset, active_tokens);
    }
    write_bytes(prefix + ".kv.raw", kv.data(), kv.size());
    std::ofstream metadata(prefix + ".json");
    if (!metadata) throw std::runtime_error("cannot create state metadata: " + prefix + ".json");
    metadata << "{\"active_tokens\":" << active_tokens
             << ",\"recurrent_layers\":" << model.recurrent_states().size()
             << ",\"recurrent_f32_values\":" << recurrent.size()
             << ",\"recurrent_layout\":\"layer-major state then conv-history, native MIInfer float32 order\""
             << ",\"kv_layers\":" << model.kv_caches().size()
             << ",\"kv_raw_bytes\":" << kv.size()
             << ",\"kv_quant_mode\":" << static_cast<unsigned>(model.kv_quant_mode())
             << ",\"kv_layout\":\"layer-major raw cache bytes from AttentionLayerKvCacheStorage::download_raw\"}\n";
    if (!metadata) throw std::runtime_error("failed writing state metadata");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: e2-0003-checkpointed-call MODEL.gguf "
                     "8K|32K|64K|128K [--generate N] [--iterations N] [--warmup N] "
                     "[--prompt-tokens N|--prompt-ids FILE] [--forced-token-ids FILE] "
                     "[--capture-raw-logits FILE] [--capture-token-ids FILE] [--capture-state-prefix PREFIX] "
                     "[--reference-token-ids FILE] [--output FILE] [--ledger-only]\n";
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
        std::string raw_logits_path;
        std::string generated_ids_path;
        std::string state_capture_prefix;
        std::string prompt_ids_path;
        std::string forced_ids_path;
        bool ledger_only = false;
        bool prompt_tokens_set = false;
        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--generate" && i + 1 < argc) generate_tokens = parse_size(argv[++i], "generate count");
            else if (arg == "--prompt-tokens" && i + 1 < argc) {
                prompt_tokens = parse_size(argv[++i], "prompt token count");
                prompt_tokens_set = true;
            }
            else if (arg == "--prompt-ids" && i + 1 < argc) prompt_ids_path = argv[++i];
            else if (arg == "--forced-token-ids" && i + 1 < argc) forced_ids_path = argv[++i];
            else if ((arg == "--iterations" || arg == "--repetitions") && i + 1 < argc)
                iterations = parse_size(argv[++i], "iteration count");
            else if (arg == "--warmup" && i + 1 < argc) warmups = parse_size(argv[++i], "warmup count", true);
            else if (arg == "--reference-token-ids" && i + 1 < argc) reference_path = argv[++i];
            else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
            else if (arg == "--capture-raw-logits" && i + 1 < argc) raw_logits_path = argv[++i];
            else if (arg == "--capture-token-ids" && i + 1 < argc) generated_ids_path = argv[++i];
            else if (arg == "--capture-state-prefix" && i + 1 < argc) state_capture_prefix = argv[++i];
            else if (arg == "--ledger-only") ledger_only = true;
            else throw std::runtime_error("unknown or incomplete option: " + arg);
        }
        if (prompt_tokens_set && !prompt_ids_path.empty()) {
            throw std::runtime_error("--prompt-tokens and --prompt-ids cannot be used together");
        }
        if (prompt_tokens > found->tokens || iterations > 100 || warmups > 20) {
            throw std::runtime_error("requested generation/iteration/warmup count is out of range");
        }
        const auto prompt = prompt_ids_path.empty()
            ? e2_0003::make_prompt(prompt_tokens, found->seed)
            : read_prompt_ids(prompt_ids_path);
        if (prompt.empty() || prompt.size() > found->tokens) {
            throw std::runtime_error("prompt ID count is outside the selected context profile");
        }
        prompt_tokens = prompt.size();
        if (generate_tokens > prompt_tokens) {
            throw std::runtime_error("generation count must not exceed the prompt token count");
        }
        const auto forced_ids = forced_ids_path.empty() ? std::vector<std::uint32_t>{} : read_tokens(forced_ids_path);

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
        if (!raw_logits_path.empty() && (iterations != 1 || warmups != 0 || generate_tokens < 5)) {
            throw std::runtime_error("raw-logit capture requires one measured iteration, no warmup, and at least five generated tokens");
        }
        if (!generated_ids_path.empty() && (iterations != 1 || warmups != 0)) {
            throw std::runtime_error("token ID capture requires one measured iteration and no warmup");
        }
        if (!state_capture_prefix.empty()
            && (iterations != 1 || warmups != 0 || generate_tokens < 5 || forced_ids.size() != 4)) {
            throw std::runtime_error("state capture requires one run, no warmup, five decisions, and four forced IDs");
        }
        if (!raw_logits_path.empty() && !forced_ids_path.empty() && forced_ids.size() != 4) {
            throw std::runtime_error("five-decision teacher-forced capture requires four forced token IDs");
        }
        if (!raw_logits_path.empty() && std::filesystem::exists(raw_logits_path)) {
            throw std::runtime_error("refusing to overwrite existing raw-logit capture: " + raw_logits_path);
        }
        if (!generated_ids_path.empty() && std::filesystem::exists(generated_ids_path)) {
            throw std::runtime_error("refusing to overwrite existing token ID capture: " + generated_ids_path);
        }
        if (!state_capture_prefix.empty()) {
            for (const std::size_t decision : {std::size_t{0}, std::size_t{1}, std::size_t{4}}) {
                const auto path = state_capture_prefix + ".decision-" + std::to_string(decision + 1);
                if (std::filesystem::exists(path + ".recurrent.f32")
                    || std::filesystem::exists(path + ".kv.raw")
                    || std::filesystem::exists(path + ".json")) {
                    throw std::runtime_error("refusing to overwrite existing state capture: " + path);
                }
            }
        }
        auto& out = *output;

        event(out, "MODEL_HASH_BEGIN", 0);
        const auto model_hash = miinfer::sha256_file(model_path);
        event(out, "MODEL_HASH_READY", 0);
        event(out, "MODEL_LOAD_BEGIN", 0);
        const auto model_data = miinfer::Qwen35Model::load(model_path);
        event(out, "MODEL_DATA_READY", 0);
        event(out, "MODEL_ALLOC_BEGIN", 0);
        const auto context_capacity_tokens = static_cast<std::uint32_t>(
            prompt_tokens + std::max<std::size_t>(256, generate_tokens));
        miinfer::prefill_v2::PrefillV2Model model(
            model_data, context_capacity_tokens, true);
        event(out, "MODEL_ALLOC_READY", 0);
        if (std::any_of(prompt.begin(), prompt.end(),
                [&](std::uint32_t token) { return token >= model.vocab_size(); })) {
            throw std::runtime_error("prompt IDs contain a token outside the loaded model vocabulary");
        }
        const auto expected = reference_path.empty() ? std::vector<std::uint32_t>{} : read_tokens(reference_path);
        const char* graph_env = std::getenv("MIINFER_HIP_GRAPH");
        const bool use_hip_graph = graph_env == nullptr || std::string_view(graph_env) != "0";

        out << "benchmark=E2-0003-clean-source\ncontext="
            << (prompt_tokens == found->tokens ? found->name : "custom")
            << "\ncontext_profile=" << found->name
            << "\nprompt_tokens=" << prompt_tokens
            << "\ncontext_capacity_tokens=" << context_capacity_tokens
            << "\ngenerate_tokens=" << generate_tokens
            << "\niterations=" << iterations
            << "\nwarmup=" << warmups
            << "\nmodel_sha256=" << model_hash
            << "\nprompt_ids_sha256="
            << (prompt_ids_path.empty() ? "synthetic" : miinfer::sha256_file(prompt_ids_path))
            << "\nprompt_fingerprint_fnv1a64=" << std::hex << e2_0003::prompt_fingerprint(prompt) << std::dec
            << "\ntemperature=0\ntop_p=1\ntop_k=1\nrepetition_penalty=1"
            << "\nfrequency_penalty=0\npresence_penalty=0\nstop_token_ids=disabled"
            << "\nrepeat_last_n=256\nreset_state_before=true\nuse_hip_graph="
            << (use_hip_graph ? "true" : "false")
            << "\nseed=42\nsampler=greedy_argmax\n"
            << "\nforced_token_count=" << forced_ids.size() << "\nforced_token_ids=";
        for (std::size_t i = 0; i < forced_ids.size(); ++i) out << (i ? "," : "") << forced_ids[i];
        out << "\nprompt_ids_file="
            << (prompt_ids_path.empty() ? "synthetic" : std::filesystem::path(prompt_ids_path).filename().string())
            << "\nmodel_vram_bytes=" << model.total_vram_bytes() << "\n" << std::flush;
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
            options.use_hip_graph = use_hip_graph;
            options.temperature = 0.0f;
            options.top_p = 1.0f;
            options.top_k = 1;
            options.repetition_penalty = 1.0f;
            options.frequency_penalty = 0.0f;
            options.presence_penalty = 0.0f;
            options.repeat_last_n = 256;
            options.stop_token_ids.clear();
            options.seed = 42;
            options.forced_token_ids = forced_ids;
            std::vector<std::vector<float>> raw_logits;
            std::size_t logit_decision = 0;
            if (measured && (!raw_logits_path.empty() || !state_capture_prefix.empty())) {
                raw_logits.reserve(5);
                options.on_raw_logits = [&](std::span<const float> logits) {
                    if (!raw_logits_path.empty() && raw_logits.size() < 5) {
                        raw_logits.emplace_back(logits.begin(), logits.end());
                    }
                    if (!state_capture_prefix.empty()
                        && (logit_decision == 0 || logit_decision == 1 || logit_decision == 4)) {
                        const auto prefix = state_capture_prefix + ".decision-"
                            + std::to_string(logit_decision + 1);
                        capture_state(model, prefix, prompt.size() + logit_decision);
                    }
                    ++logit_decision;
                };
            }
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
            if (measured && !generated_ids_path.empty()) {
                std::ofstream ids_file(generated_ids_path);
                if (!ids_file) throw std::runtime_error("cannot create token ID capture: " + generated_ids_path);
                for (const auto token : stats.generated_tokens) ids_file << token << '\n';
                if (!ids_file) throw std::runtime_error("failed writing token ID capture: " + generated_ids_path);
            }
            if (!raw_logits_path.empty()) {
                if (raw_logits.size() != 5) throw std::runtime_error("raw-logit capture did not contain five decisions");
                std::ofstream raw_logits_file(raw_logits_path, std::ios::binary | std::ios::out);
                if (!raw_logits_file) throw std::runtime_error("cannot create raw-logit capture: " + raw_logits_path);
                for (const auto& decision : raw_logits) {
                    if (decision.size() != model.vocab_size()) throw std::runtime_error("raw-logit vocabulary size changed between decisions");
                    raw_logits_file.write(reinterpret_cast<const char*>(decision.data()),
                        static_cast<std::streamsize>(decision.size() * sizeof(float)));
                }
                if (!raw_logits_file) throw std::runtime_error("failed writing raw-logit capture: " + raw_logits_path);
                out << "raw_logits_capture={\"format\":\"native_f32\",\"decisions\":5,\"vocab_size\":"
                    << model.vocab_size() << ",\"bytes\":"
                    << 5 * model.vocab_size() * sizeof(float) << ",\"file\":\""
                    << std::filesystem::path(raw_logits_path).filename().string() << "\"}\n" << std::flush;
            }
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
                << ",\"ttft_ms\":" << stats.ttft_ms
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
                << ",\"raw_logits_decisions\":" << (raw_logits_path.empty() ? 0 : raw_logits.size())
                << ",\"raw_logits_vocab_size\":" << (raw_logits_path.empty() ? 0 : model.vocab_size())
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
