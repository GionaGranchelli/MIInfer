#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/prefill_v2/persistent_session.hpp"
#include "miinfer/qwen3_tokenizer.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

namespace {

template <typename T>
T* allocate_device(std::size_t count) {
    T* ptr = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}

struct Result {
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    double restore_seconds = 0.0;
    std::uint32_t top1 = 0;
    std::uint32_t second_token = 0;
    std::vector<float> logits;
    std::vector<float> suffix_hidden;
    std::vector<float> recurrent_state_trace;
    std::size_t recurrent_layer_count = 0;
};

Result run_mode(const miinfer::Qwen35Model& model,
                miinfer::prefill_v2::KvCacheQuantMode mode,
                const char* label, const std::vector<std::uint32_t>& tokens,
                std::uint32_t capacity) {
    using namespace miinfer::prefill_v2;
    PrefillV2Model runtime(model, capacity, true, mode);
    std::cout << label << " capacity_tokens=" << capacity
              << " static_vram_bytes=" << runtime.total_vram_bytes()
              << " weights_bytes=" << runtime.persistent_weight_bytes()
              << " state_bytes=" << runtime.persistent_state_bytes()
              << " workspace_bytes=" << runtime.workspace_bytes() << '\n';

    float* hidden = allocate_device<float>(tokens.size() * kHidden);
    float* logits = allocate_device<float>(model.config().vocab_size);
    runtime.reset_state();
    const auto start = std::chrono::steady_clock::now();
    runtime.prefill_sequence(std::span<const std::uint32_t>(tokens), hidden);
    runtime.compute_logits(hidden + (tokens.size() - 1) * kHidden, logits);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();

    Result result;
    result.prefill_seconds = std::chrono::duration<double>(stop - start).count();
    result.logits.resize(model.config().vocab_size);
    MIINFER_HIP_CHECK(hipMemcpy(result.logits.data(), logits,
        result.logits.size() * sizeof(float), hipMemcpyDeviceToHost));
    result.top1 = static_cast<std::uint32_t>(std::distance(
        result.logits.begin(), std::max_element(result.logits.begin(), result.logits.end())));
    const auto decode_start = std::chrono::steady_clock::now();
    result.second_token = runtime.decode_step(result.top1, static_cast<std::uint32_t>(tokens.size()));
    result.decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - decode_start).count();
    MIINFER_HIP_CHECK(hipFree(hidden));
    MIINFER_HIP_CHECK(hipFree(logits));
    std::cout << label << " prefill_seconds=" << std::fixed << std::setprecision(3)
              << result.prefill_seconds << " first_token=" << result.top1
              << " decode_seconds=" << result.decode_seconds
              << " second_token=" << result.second_token << '\n';
    return result;
}

Result run_session_mode(const miinfer::Qwen35Model& model,
                        miinfer::prefill_v2::KvCacheQuantMode mode,
                        const char* label,
                        const std::string& session_path,
                        const std::vector<std::uint32_t>& suffix,
                        std::uint32_t capacity,
                        bool retain_suffix_hidden = false) {
    using namespace miinfer::prefill_v2;
    PrefillV2Model runtime(model, capacity, true, mode);
    std::vector<std::uint32_t> restored_tokens;
    PersistentSessionHeader session_header{};
    auto restore_start = std::chrono::steady_clock::now();
    if (!PersistentSession::inspect_file(session_path, session_header)
        || !runtime.load_session(session_path, restored_tokens)
        || restored_tokens.size() != session_header.prefix_length) {
        throw std::runtime_error(std::string(label) + " failed to restore the prefix session");
    }
    auto restore_stop = std::chrono::steady_clock::now();

    const std::size_t hidden_count = retain_suffix_hidden
        ? suffix.size() * kHidden : static_cast<std::size_t>(kPrefillV2MacroTile) * kHidden;
    float* hidden = allocate_device<float>(hidden_count);
    float* logits = allocate_device<float>(model.config().vocab_size);
    const auto prefill_start = std::chrono::steady_clock::now();
    std::size_t offset = 0;
    std::uint32_t final_chunk = 0;
    while (offset < suffix.size()) {
        final_chunk = static_cast<std::uint32_t>(std::min<std::size_t>(
            kPrefillV2MacroTile, suffix.size() - offset));
        runtime.forward(std::span<const std::uint32_t>(suffix).subspan(offset, final_chunk),
                        static_cast<std::uint32_t>(restored_tokens.size() + offset),
                        retain_suffix_hidden ? hidden + offset * kHidden : hidden);
        offset += final_chunk;
    }
    const float* final_hidden = retain_suffix_hidden
        ? hidden + (suffix.size() - 1) * kHidden : hidden + (final_chunk - 1) * kHidden;
    runtime.compute_logits(final_hidden, logits);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    const auto prefill_stop = std::chrono::steady_clock::now();

    Result result;
    result.restore_seconds = std::chrono::duration<double>(restore_stop - restore_start).count();
    result.prefill_seconds = std::chrono::duration<double>(prefill_stop - prefill_start).count();
    result.logits.resize(model.config().vocab_size);
    MIINFER_HIP_CHECK(hipMemcpy(result.logits.data(), logits,
        result.logits.size() * sizeof(float), hipMemcpyDeviceToHost));
    if (retain_suffix_hidden) {
        result.suffix_hidden.resize(suffix.size() * kHidden);
        MIINFER_HIP_CHECK(hipMemcpy(result.suffix_hidden.data(), hidden,
            result.suffix_hidden.size() * sizeof(float), hipMemcpyDeviceToHost));
    }
    result.top1 = static_cast<std::uint32_t>(std::distance(
        result.logits.begin(), std::max_element(result.logits.begin(), result.logits.end())));
    const auto decode_start = std::chrono::steady_clock::now();
    result.second_token = runtime.decode_step(
        result.top1, static_cast<std::uint32_t>(restored_tokens.size() + suffix.size()));
    result.decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - decode_start).count();
    MIINFER_HIP_CHECK(hipFree(hidden));
    MIINFER_HIP_CHECK(hipFree(logits));
    std::cout << label << " restore_seconds=" << std::fixed << std::setprecision(3)
              << result.restore_seconds << " suffix_seconds=" << result.prefill_seconds
              << " restored_prefix_tokens=" << restored_tokens.size()
              << " position_after_suffix=" << restored_tokens.size() + suffix.size()
              << " first_token=" << result.top1 << " decode_seconds=" << result.decode_seconds
              << " second_token=" << result.second_token << '\n';
    return result;
}

Result run_session_graph_mode(const miinfer::Qwen35Model& model,
                              const char* label,
                              const std::string& session_path,
                              const std::vector<std::uint32_t>& suffix,
                              std::uint32_t capacity) {
    using namespace miinfer::prefill_v2;
    PersistentSessionHeader header{};
    std::vector<std::uint32_t> prefix;
    if (!PersistentSession::inspect_file(session_path, header, &prefix)
        || header.kv_quant_mode != static_cast<std::uint8_t>(KvCacheQuantMode::kQ8Q8)
        || prefix.size() != header.prefix_length || suffix.empty()
        || suffix.size() % kPrefillV2MacroTile != 0) {
        throw std::runtime_error(std::string(label) + " found an incompatible prefix session or empty suffix");
    }
    std::vector<std::uint32_t> prompt;
    prompt.reserve(prefix.size() + suffix.size());
    prompt.insert(prompt.end(), prefix.begin(), prefix.end());
    prompt.insert(prompt.end(), suffix.begin(), suffix.end());

    PrefillV2Model runtime(model, capacity, true, KvCacheQuantMode::kQ8Q8);
    GenerateOptions options;
    options.max_new_tokens = 1;
    options.reset_state_before = true;
    const char* block_compare_env =
        std::getenv("MIINFER_M31_Q8_GQA6_BLOCK_COMPARE_LAYERS");
    options.use_hip_graph = block_compare_env == nullptr
        || std::strcmp(block_compare_env, "0") == 0;
    options.enable_prefix_reuse = true;
    options.persistent_session_dir = std::filesystem::path(session_path).parent_path().string();
    options.temperature = 0.0F;
    options.top_k = 1;
    options.repetition_penalty = 1.0F;
    options.stop_token_ids.clear();
    const GenerateStats stats = runtime.generate(prompt, options);
    const std::uint32_t expected_tiles = options.use_hip_graph
        ? static_cast<std::uint32_t>(suffix.size() / kPrefillV2MacroTile) : 0;
    if (!stats.persistent_session_restored || stats.prefix_tokens_reused != prefix.size()
        || stats.suffix_tokens_dispatched != suffix.size()
        || stats.suffix_graph_replay_tiles != expected_tiles
        || stats.used_hip_graph != options.use_hip_graph
        || stats.generated_tokens.size() != 1) {
        throw std::runtime_error(std::string(label) + " did not restore/replay the expected full suffix tiles");
    }

    Result result;
    result.restore_seconds = stats.restore_ms / 1000.0;
    result.prefill_seconds = stats.suffix_prefill_ms / 1000.0;
    result.top1 = stats.generated_tokens.front();
    result.logits.resize(model.config().vocab_size);
    MIINFER_HIP_CHECK(hipMemcpy(result.logits.data(), runtime.logits_buffer(),
        result.logits.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (const auto& state : runtime.recurrent_states()) {
        if (state.position() != prompt.size()) {
            throw std::runtime_error(std::string(label) + " recurrent position did not reach suffix end");
        }
        std::vector<float> values, history;
        state.download(values, history);
        result.recurrent_state_trace.insert(result.recurrent_state_trace.end(), values.begin(), values.end());
        result.recurrent_state_trace.insert(result.recurrent_state_trace.end(), history.begin(), history.end());
    }
    result.recurrent_layer_count = runtime.recurrent_states().size();
    if (result.recurrent_layer_count == 0 || result.recurrent_state_trace.empty()) {
        throw std::runtime_error(std::string(label) + " restored no recurrent-state trace");
    }
    result.second_token = runtime.decode_step(result.top1, static_cast<std::uint32_t>(prompt.size()));
    std::cout << label << " restore_seconds=" << std::fixed << std::setprecision(3)
              << result.restore_seconds << " suffix_seconds=" << result.prefill_seconds
              << " restored_prefix_tokens=" << stats.prefix_tokens_reused
              << " suffix_tokens=" << stats.suffix_tokens_dispatched
              << " graph_replay_tiles=" << stats.suffix_graph_replay_tiles
              << " graph_used=" << stats.used_hip_graph
              << " position_after_suffix=" << prompt.size()
              << " first_token=" << result.top1 << " second_token=" << result.second_token << '\n';
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

bool compare_results(const Result& control, const Result& candidate, int pair) {
    if (control.logits.size() != candidate.logits.size()) return false;
    double dot = 0.0, control_norm = 0.0, candidate_norm = 0.0, max_abs = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < control.logits.size(); ++i) {
        finite = finite && std::isfinite(control.logits[i]) && std::isfinite(candidate.logits[i]);
        max_abs = std::max(max_abs, std::abs(static_cast<double>(control.logits[i]) - candidate.logits[i]));
        dot += static_cast<double>(control.logits[i]) * candidate.logits[i];
        control_norm += static_cast<double>(control.logits[i]) * control.logits[i];
        candidate_norm += static_cast<double>(candidate.logits[i]) * candidate.logits[i];
    }
    const double cosine = dot / (std::sqrt(control_norm * candidate_norm) + 1e-30);
    const bool first_equal = control.top1 == candidate.top1;
    const bool second_equal = control.second_token == candidate.second_token;
    std::cout << std::setprecision(9) << "q8_pair=" << pair
              << " logits_max_abs=" << max_abs << " logits_cosine=" << cosine
              << " first_token_equal=" << first_equal
              << " second_token_equal=" << second_equal << '\n';
    return finite && std::isfinite(cosine) && cosine >= 0.9999 && first_equal && second_equal;
}

bool compare_flash_tile_results(const Result& control, const Result& candidate, int pair) {
    struct Metrics { double max_abs; double cosine; bool finite; bool same_size; };
    const auto measure = [](const std::vector<float>& a, const std::vector<float>& b) {
        Metrics m{0.0, 1.0, true, a.size() == b.size()};
        if (!m.same_size) return m;
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            m.finite = m.finite && std::isfinite(a[i]) && std::isfinite(b[i]);
            m.max_abs = std::max(m.max_abs, std::abs(static_cast<double>(a[i]) - b[i]));
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        m.cosine = dot / (std::sqrt(norm_a * norm_b) + 1e-30);
        return m;
    };
    const Metrics logits = measure(control.logits, candidate.logits);
    const Metrics state = measure(control.recurrent_state_trace, candidate.recurrent_state_trace);
    const bool tokens_equal = control.top1 == candidate.top1
        && control.second_token == candidate.second_token;
    const bool logits_exact = control.logits == candidate.logits;
    const bool state_exact = control.recurrent_state_trace == candidate.recurrent_state_trace;
    const bool state_observed = control.recurrent_layer_count > 0
        && control.recurrent_layer_count == candidate.recurrent_layer_count
        && !control.recurrent_state_trace.empty() && !candidate.recurrent_state_trace.empty();
    const bool exact_numeric_pass = logits.same_size && state.same_size && state_observed
        && logits.finite && state.finite
        && logits.max_abs <= 0.01 && logits.cosine >= 0.9999
        && state.max_abs <= 0.01 && state.cosine >= 0.9999 && tokens_equal;
    const char* block_compare_env = std::getenv("MIINFER_M31_Q8_GQA6_BLOCK_SOFTMAX4_PAIR");
    const bool block_softmax4_pair = block_compare_env != nullptr
        && std::strcmp(block_compare_env, "0") != 0;
    const char* pair_head_reuse_env = std::getenv("MIINFER_M31_Q8_GQA6_PAIR_HEAD_REUSE_PAIR");
    const bool pair_head_reuse_pair = pair_head_reuse_env != nullptr
        && std::strcmp(pair_head_reuse_env, "0") != 0;
    const bool numeric_pass = pair_head_reuse_pair
        ? logits.same_size && state.same_size && state_observed && logits.finite && state.finite
        : exact_numeric_pass;
    std::cout << std::setprecision(9)
              << (pair_head_reuse_pair ? "pair_head_reuse_pair="
                  : block_softmax4_pair ? "block_softmax4_pair=" : "flash_tile_pair=") << pair
              << " logits_max_abs=" << logits.max_abs << " logits_cosine=" << logits.cosine
              << " state_max_abs=" << state.max_abs << " state_cosine=" << state.cosine
              << " recurrent_layers=" << candidate.recurrent_layer_count
              << " logits_exact=" << logits_exact << " state_exact=" << state_exact
              << " tokens_equal=" << tokens_equal << " finite=" << (logits.finite && state.finite)
              << " numeric_pass=" << numeric_pass
              << (pair_head_reuse_pair ? " task_behavior_unproven=1" : "") << '\n';
    return numeric_pass;
}

}  // namespace

int main(int argc, char** argv) {
    bool fp16_first = false;
    bool fp16_only = false;
    bool q8_query_tiled_pair = false;
    bool q8_native_gqa6_pair = false;
    bool q8_gqa6_session_pair = false;
    bool q8_gqa6_dpp_session_pair = false;
    bool q8_gqa6_pair_head_reuse_pair = false;
    bool q8_gqa6_wave64_kv_broadcast_pair = false;
    bool q8_only = false;
    bool fp16_fp32_attention = false;
    std::uint32_t prompt_tokens = 8192;
    std::string prompt_file;
    std::string session_path;
    std::string logits_output;
    for (int i = 2; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--fp16-first") {
            fp16_first = true;
        } else if (arg == "--fp16-only") {
            fp16_only = true;
        } else if (arg == "--q8-query-tiled-pair") {
            q8_query_tiled_pair = true;
        } else if (arg == "--q8-native-gqa6-pair") {
            q8_native_gqa6_pair = true;
        } else if (arg == "--q8-gqa6-session-pair") {
            q8_gqa6_session_pair = true;
        } else if (arg == "--q8-gqa6-dpp-session-pair") {
            q8_gqa6_dpp_session_pair = true;
        } else if (arg == "--q8-gqa6-pair-head-reuse") {
            q8_gqa6_pair_head_reuse_pair = true;
        } else if (arg == "--q8-gqa6-wave64-kv-broadcast") {
            q8_gqa6_wave64_kv_broadcast_pair = true;
        } else if (arg == "--q8-only") {
            q8_only = true;
        } else if (arg == "--fp16-fp32-attn") {
            fp16_fp32_attention = true;
        } else if (arg == "--tokens" && i + 1 < argc) {
            char* end = nullptr;
            const auto parsed = std::strtoul(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || parsed == 0 || parsed > 65536) {
                std::cerr << "--tokens must be between 1 and 65536\n";
                return 2;
            }
            prompt_tokens = static_cast<std::uint32_t>(parsed);
        } else if (arg == "--prompt-file" && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (arg == "--session" && i + 1 < argc) {
            session_path = argv[++i];
        } else if (arg == "--logits-out" && i + 1 < argc) {
            logits_output = argv[++i];
        } else {
            std::cerr << "usage: miinfer-m31-fp16-full-model MODEL.gguf "
                         "[--fp16-first] [--fp16-only] [--q8-only] [--logits-out PATH] "
                         "[--q8-query-tiled-pair] [--q8-native-gqa6-pair] [--q8-gqa6-session-pair] "
                         "[--q8-gqa6-dpp-session-pair] [--q8-gqa6-pair-head-reuse] "
                         "[--q8-gqa6-wave64-kv-broadcast] "
                         "[--fp16-fp32-attn] [--tokens N] "
                         "[--prompt-file PATH] [--session PATH]\n";
            return 2;
        }
    }
    if (argc < 2 || (fp16_first && fp16_only) || (q8_only && (fp16_first || fp16_only))
        || (q8_only && q8_query_tiled_pair)
        || (!logits_output.empty() && !(q8_only || fp16_only))
        || (q8_query_tiled_pair && (fp16_first || fp16_only))
        || (q8_native_gqa6_pair && (fp16_first || fp16_only || q8_only
            || q8_query_tiled_pair))
        || ((q8_gqa6_session_pair || q8_gqa6_dpp_session_pair || q8_gqa6_pair_head_reuse_pair
                || q8_gqa6_wave64_kv_broadcast_pair)
            && (session_path.empty() || prompt_file.empty() || fp16_first || fp16_only
                || q8_only || q8_query_tiled_pair || q8_native_gqa6_pair
                || static_cast<int>(q8_gqa6_session_pair) + static_cast<int>(q8_gqa6_dpp_session_pair)
                    + static_cast<int>(q8_gqa6_pair_head_reuse_pair)
                    + static_cast<int>(q8_gqa6_wave64_kv_broadcast_pair) > 1))) {
        std::cerr << "usage: miinfer-m31-fp16-full-model MODEL.gguf "
                     "[--fp16-first] [--fp16-only] [--q8-only] [--logits-out PATH] "
                     "[--q8-query-tiled-pair] [--q8-native-gqa6-pair] [--q8-gqa6-session-pair] "
                     "[--q8-gqa6-dpp-session-pair] [--q8-gqa6-pair-head-reuse] "
                     "[--q8-gqa6-wave64-kv-broadcast] "
                     "[--fp16-fp32-attn] [--tokens N] "
                     "[--prompt-file PATH] [--session PATH]\n";
        return 2;
    }
    miinfer::DeviceInfo device{};
    std::string error;
    if (!miinfer::validate_gfx906_device(0, device, error)) {
        std::cerr << "device validation failed: " << error << '\n';
        return 1;
    }
    char bdf[32]{};
    MIINFER_HIP_CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), 0));
    std::cout << "device_bdf=" << bdf << " architecture=" << device.architecture << '\n';

    const std::uint32_t capacity = std::max<std::uint32_t>(65536, prompt_tokens + 4096);
    const auto model = miinfer::Qwen35Model::load(argv[1]);
    std::vector<std::uint32_t> tokens(prompt_tokens);
    if (prompt_file.empty()) {
        for (std::uint32_t i = 0; i < prompt_tokens; ++i) {
            tokens[i] = (i * 37U + 101U) % model.config().vocab_size;
        }
    } else {
        std::ifstream input(prompt_file, std::ios::binary);
        if (!input) {
            std::cerr << "failed to open prompt file: " << prompt_file << '\n';
            return 2;
        }
        const std::string text(std::istreambuf_iterator<char>(input), {});
        const auto encoded = miinfer::Qwen3Tokenizer::load(*model.file()).encode(text);
        if (encoded.size() < prompt_tokens) {
            std::cerr << "prompt file encodes to " << encoded.size()
                      << " tokens, fewer than requested " << prompt_tokens << '\n';
            return 2;
        }
        std::copy_n(encoded.begin(), prompt_tokens, tokens.begin());
        std::cout << "prompt_text_bytes=" << text.size()
                  << " prompt_tokens_available=" << encoded.size()
                  << " prompt_tokens_processed=" << prompt_tokens << '\n';
    }

    if (::setenv("MIINFER_V2_0043_GQA_ATTENTION", "1", 1) != 0) {
        std::cerr << "failed to enable existing opt-in GQA attention path\n";
        return 1;
    }
    if (::setenv("MIINFER_M31_FP16_ATTN_FP32", fp16_fp32_attention ? "1" : "0", 1) != 0) {
        std::cerr << "failed to set FP16 attention accumulation mode\n";
        return 1;
    }
    if (::setenv("MIINFER_M31_Q8_TO_FP16_RESTORE", session_path.empty() ? "0" : "1", 1) != 0) {
        std::cerr << "failed to set Q8-to-FP16 session restore mode\n";
        return 1;
    }
    if (q8_only) {
        const auto q8 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                                 "q8_only", tokens, capacity);
        const bool finite = std::all_of(q8.logits.begin(), q8.logits.end(),
                                        [](float value) { return std::isfinite(value); });
        if (!logits_output.empty()) {
            std::ofstream output(logits_output, std::ios::binary);
            if (!output || !output.write(reinterpret_cast<const char*>(q8.logits.data()),
                                         static_cast<std::streamsize>(q8.logits.size() * sizeof(float)))) {
                std::cerr << "failed to write logits to " << logits_output << '\n';
                return 1;
            }
        }
        std::cout << "q8_logits_finite=" << finite << " q8_logits_count=" << q8.logits.size() << '\n';
        return finite ? 0 : 1;
    }
    if (q8_query_tiled_pair) {
        std::vector<double> control_times, candidate_times;
        bool correct = true;
        for (int pair = 0; pair < 3; ++pair) {
            Result control, candidate;
            if (pair == 1) {
                if (::setenv("MIINFER_M31_Q8_QUERY_TILED", "1", 1) != 0) return 1;
                candidate = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                                     "q8_query_tiled_candidate", tokens, capacity);
                if (::setenv("MIINFER_M31_Q8_QUERY_TILED", "0", 1) != 0) return 1;
                control = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                                   "q8_control", tokens, capacity);
            } else {
                if (::setenv("MIINFER_M31_Q8_QUERY_TILED", "0", 1) != 0) return 1;
                control = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                                   "q8_control", tokens, capacity);
                if (::setenv("MIINFER_M31_Q8_QUERY_TILED", "1", 1) != 0) return 1;
                candidate = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                                     "q8_query_tiled_candidate", tokens, capacity);
            }
            correct = compare_results(control, candidate, pair) && correct;
            control_times.push_back(control.prefill_seconds);
            candidate_times.push_back(candidate.prefill_seconds);
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "median_q8_control_seconds=" << median(control_times)
                  << " median_q8_query_tiled_seconds=" << median(candidate_times)
                  << " speedup=" << median(control_times) / median(candidate_times) << '\n';
        return correct ? 0 : 3;
    }
    if (q8_native_gqa6_pair) {
        if (::setenv("MIINFER_M31_Q8_DOT4_ATTENTION", "0", 1) != 0
            || ::setenv("MIINFER_M31_Q8_QUERY_TILED", "0", 1) != 0
            || ::setenv("MIINFER_M31_Q8_GQA6_NATIVE_TILED", "0", 1) != 0) return 1;
        const Result control = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                                        "q8_production_control", tokens, capacity);
        if (::setenv("MIINFER_M31_Q8_GQA6_NATIVE_TILED", "1", 1) != 0) return 1;
        const Result candidate = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                                          "q8_native_gqa6_candidate", tokens, capacity);
        const bool correct = compare_results(control, candidate, 0);
        std::cout << std::fixed << std::setprecision(3)
                  << "q8_production_seconds=" << control.prefill_seconds
                  << " q8_native_gqa6_seconds=" << candidate.prefill_seconds
                  << " speedup=" << control.prefill_seconds / candidate.prefill_seconds << '\n';
        return correct ? 0 : 3;
    }
    if (!session_path.empty()) {
        miinfer::prefill_v2::PersistentSessionHeader header{};
        if (!miinfer::prefill_v2::PersistentSession::inspect_file(session_path, header) ||
            header.kv_quant_mode != static_cast<std::uint8_t>(
                miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8) || prompt_tokens == 0) {
            std::cerr << "--session requires a valid Q8/Q8 session and positive --tokens suffix\n";
            return 2;
        }
        const std::uint32_t session_capacity = std::max<std::uint32_t>(
            capacity, header.prefix_length + prompt_tokens + 4096);
        std::vector<std::uint32_t> suffix = tokens;
        if (prompt_file.empty()) {
            for (std::uint32_t i = 0; i < prompt_tokens; ++i) {
                suffix[i] = (i * 37U + 101U) % model.config().vocab_size;
            }
        }
        std::cout << "session_prefix_tokens=" << header.prefix_length
                  << " suffix_tokens=" << prompt_tokens
                  << " suffix_source=" << (prompt_file.empty() ? "deterministic_synthetic" : "tokenized_text_file")
                  << " graph_requested=1 expected_replay_tiles="
                  << (prompt_tokens + miinfer::prefill_v2::kPrefillV2MacroTile - 1)
                      / miinfer::prefill_v2::kPrefillV2MacroTile << '\n';
        if (q8_gqa6_session_pair || q8_gqa6_dpp_session_pair || q8_gqa6_pair_head_reuse_pair
            || q8_gqa6_wave64_kv_broadcast_pair) {
            const bool dpp_pair = q8_gqa6_dpp_session_pair;
            const bool pair_head_reuse_pair = q8_gqa6_pair_head_reuse_pair;
            const bool wave64_kv_broadcast_pair = q8_gqa6_wave64_kv_broadcast_pair;
            if (pair_head_reuse_pair
                && ::setenv("MIINFER_M31_Q8_GQA6_PAIR_HEAD_REUSE_PAIR", "1", 1) != 0) return 1;
            const char* packed_env = std::getenv("MIINFER_M31_Q8_GQA6_PACKED_LDS_PAIR");
            const char* wide128_pair_env = std::getenv("MIINFER_M31_Q8_GQA6_PACKED_LDS128_PAIR");
            const char* paired_token_env = std::getenv("MIINFER_M31_Q8_GQA6_PAIRED_TOKEN_PAIR");
            const char* direct_global_env = std::getenv("MIINFER_M31_Q8_GQA6_DIRECT_GLOBAL_PAIR");
            const char* flash_tile_env = std::getenv("MIINFER_M31_Q8_GQA6_FLASH_TILE_PAIR");
            const char* block_softmax4_env = std::getenv("MIINFER_M31_Q8_GQA6_BLOCK_SOFTMAX4_PAIR");
            const bool wide128_pair = wide128_pair_env != nullptr && std::strcmp(wide128_pair_env, "0") != 0;
            const bool paired_token_pair = paired_token_env != nullptr && std::strcmp(paired_token_env, "0") != 0;
            const bool direct_global_pair = direct_global_env != nullptr && std::strcmp(direct_global_env, "0") != 0;
            const bool flash_tile_pair = flash_tile_env != nullptr && std::strcmp(flash_tile_env, "0") != 0;
            const bool block_softmax4_pair = block_softmax4_env != nullptr
                && std::strcmp(block_softmax4_env, "0") != 0;
            const bool direct_global_control = dpp_pair || direct_global_pair || flash_tile_pair
                || wave64_kv_broadcast_pair
                || block_softmax4_pair;
            const bool packed_pair = dpp_pair || wide128_pair || paired_token_pair || direct_global_pair
                || flash_tile_pair || block_softmax4_pair || pair_head_reuse_pair
                || wave64_kv_broadcast_pair
                || (packed_env != nullptr && std::strcmp(packed_env, "0") != 0);
            const char* reps_env = std::getenv("MIINFER_M31_Q8_GQA6_PAIR_REPS");
            char* reps_end = nullptr;
            const unsigned long parsed_reps = reps_env == nullptr ? 1 : std::strtoul(reps_env, &reps_end, 10);
            if ((reps_env != nullptr && (reps_end == reps_env || *reps_end != '\0'))
                || parsed_reps == 0 || parsed_reps > 5) {
                std::cerr << "MIINFER_M31_Q8_GQA6_PAIR_REPS must be between 1 and 5\n";
                return 2;
            }
            if (::setenv("MIINFER_HIP_GRAPH", "1", 1) != 0
                || ::setenv("MIINFER_M31_Q8_DOT4_ATTENTION", "0", 1) != 0
                || ::setenv("MIINFER_M31_Q8_GQA6_BLOCK_SOFTMAX4", "0", 1) != 0
                || ::setenv("MIINFER_M31_Q8_QUERY_TILED", "0", 1) != 0
                || ::setenv("MIINFER_M31_Q8_GQA6_NATIVE_TILED", "0", 1) != 0) return 1;
            std::vector<double> control_times, candidate_times;
            bool all_correct = true;
            for (unsigned long rep = 0; rep < parsed_reps; ++rep) {
                if (::setenv("MIINFER_M31_Q8_GQA6_PACKED_LDS",
                        (wide128_pair || paired_token_pair || direct_global_pair || dpp_pair || pair_head_reuse_pair) ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_PACKED_LDS128",
                        (wide128_pair || paired_token_pair || direct_global_pair || dpp_pair || pair_head_reuse_pair) ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_PROD_ARITH", packed_pair ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_BLOCK_SOFTMAX4", "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_PAIRED_TOKEN", "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_PAIR_HEAD_REUSE", "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_DIRECT_GLOBAL",
                        direct_global_control ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_DPP_REDUCE", "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_WAVE64_KV_BROADCAST", "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_FLASH_TILE", "0", 1) != 0) return 1;
                const Result control = run_session_graph_mode(model,
                    direct_global_control ? "q8_gqa6_direct_global_control_session"
                        : (paired_token_pair || direct_global_pair) ? "q8_gqa6_packed_u128_session"
                        : (wide128_pair || pair_head_reuse_pair) ? "q8_gqa6_packed_u4_u128_session"
                        : packed_pair ? "q8_gqa6_production_arith_session" : "q8_production_session",
                    session_path, suffix, session_capacity);
                if (::setenv("MIINFER_M31_Q8_GQA6_BLOCK_SOFTMAX4", block_softmax4_pair ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_PROD_ARITH", "1", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_PACKED_LDS",
                        (pair_head_reuse_pair || (packed_pair && !flash_tile_pair && !block_softmax4_pair)) ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_PACKED_LDS128",
                        (pair_head_reuse_pair || wide128_pair) ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_PAIRED_TOKEN", paired_token_pair ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_PAIR_HEAD_REUSE", pair_head_reuse_pair ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_DIRECT_GLOBAL",
                        direct_global_control ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_DPP_REDUCE", dpp_pair ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_WAVE64_KV_BROADCAST",
                        wave64_kv_broadcast_pair ? "1" : "0", 1) != 0
                    || ::setenv("MIINFER_M31_Q8_GQA6_FLASH_TILE", flash_tile_pair ? "1" : "0", 1) != 0) return 1;
                const Result candidate = run_session_graph_mode(model,
                    pair_head_reuse_pair ? "q8_gqa6_pair_head_reuse_session"
                        : wave64_kv_broadcast_pair ? "q8_gqa6_wave64_kv_broadcast_session"
                        : block_softmax4_pair ? "q8_gqa6_block_softmax4_session"
                        : flash_tile_pair ? "q8_gqa6_flash_tile_session"
                        : dpp_pair ? "q8_gqa6_dpp_session"
                        : direct_global_pair ? "q8_gqa6_direct_global_session"
                        : paired_token_pair ? "q8_gqa6_paired_token_session"
                        : wide128_pair ? "q8_gqa6_packed_u4_u128_session"
                        : packed_pair ? "q8_gqa6_packed_lds_session" : "q8_gqa6_production_arith_session",
                    session_path, suffix, session_capacity);
                const bool logits_exact = control.logits == candidate.logits;
                const bool state_exact = control.recurrent_state_trace == candidate.recurrent_state_trace;
                const bool generated_equal = control.top1 == candidate.top1
                    && control.second_token == candidate.second_token;
                const bool finite = std::all_of(control.logits.begin(), control.logits.end(),
                        [](float value) { return std::isfinite(value); })
                    && std::all_of(candidate.logits.begin(), candidate.logits.end(),
                        [](float value) { return std::isfinite(value); })
                    && std::all_of(control.suffix_hidden.begin(), control.suffix_hidden.end(),
                        [](float value) { return std::isfinite(value); })
                    && std::all_of(candidate.suffix_hidden.begin(), candidate.suffix_hidden.end(),
                        [](float value) { return std::isfinite(value); })
                    && std::all_of(control.recurrent_state_trace.begin(), control.recurrent_state_trace.end(),
                        [](float value) { return std::isfinite(value); })
                    && std::all_of(candidate.recurrent_state_trace.begin(), candidate.recurrent_state_trace.end(),
                        [](float value) { return std::isfinite(value); });
                control_times.push_back(control.prefill_seconds);
                candidate_times.push_back(candidate.prefill_seconds);
                const bool numeric_comparison = flash_tile_pair || block_softmax4_pair || pair_head_reuse_pair
                    ? compare_flash_tile_results(control, candidate, static_cast<int>(rep + 1))
                    : logits_exact && state_exact && generated_equal && finite;
                const bool sample_correct = pair_head_reuse_pair ? finite : numeric_comparison;
                all_correct = all_correct && sample_correct;
                std::cout << "pair_sample=" << (rep + 1)
                          << " session_logits_exact=" << logits_exact
                          << " recurrent_state_trace_exact=" << state_exact
                          << " generated_tokens_equal=" << generated_equal
                          << " finite=" << finite << " accepted=" << sample_correct << '\n';
            }
            std::cout << std::fixed << std::setprecision(3)
                      << (flash_tile_pair || dpp_pair || block_softmax4_pair ? "q8_gqa6_direct_global_control_median_seconds="
                          : (paired_token_pair || direct_global_pair) ? "q8_gqa6_packed_u128_median_seconds="
                          : (wide128_pair || pair_head_reuse_pair) ? "q8_gqa6_packed_u4_u128_median_seconds="
                          : packed_pair ? "q8_gqa6_suffix_median_seconds=" : "q8_production_suffix_median_seconds=")
                      << median(control_times)
                      << (pair_head_reuse_pair ? " q8_gqa6_pair_head_reuse_median_seconds="
                          : wave64_kv_broadcast_pair ? " q8_gqa6_wave64_kv_broadcast_median_seconds="
                          : block_softmax4_pair ? " q8_gqa6_block_softmax4_median_seconds="
                          : flash_tile_pair ? " q8_gqa6_flash_tile_median_seconds="
                          : dpp_pair ? " q8_gqa6_dpp_median_seconds="
                          : direct_global_pair ? " q8_gqa6_direct_global_median_seconds="
                          : paired_token_pair ? " q8_gqa6_paired_token_median_seconds="
                          : wide128_pair ? " q8_gqa6_packed_u4_u128_median_seconds="
                          : packed_pair ? " q8_gqa6_packed_lds_suffix_median_seconds=" : " q8_gqa6_suffix_median_seconds=")
                      << median(candidate_times)
                      << " suffix_median_speedup=" << median(control_times) / median(candidate_times)
                      << " samples=" << parsed_reps << " all_samples_passed_screen=" << all_correct
                      << (pair_head_reuse_pair ? " task_behavior_unproven=1" : "") << '\n';
            return all_correct ? 0 : 3;
        }
        const auto q8 = run_session_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                                         "q8_session", session_path, suffix, session_capacity);
        const auto fp16 = run_session_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kFp16Fp16,
                                           fp16_fp32_attention ? "fp16_session_fp32" : "fp16_session",
                                           session_path, suffix, session_capacity);
        const bool correct = compare_results(q8, fp16, 0);
        std::cout << std::fixed << std::setprecision(3)
                  << "q8_total_resume_seconds=" << (q8.restore_seconds + q8.prefill_seconds + q8.decode_seconds)
                  << " fp16_total_resume_seconds=" << (fp16.restore_seconds + fp16.prefill_seconds + fp16.decode_seconds)
                  << " q8_to_fp16_suffix_speedup=" << q8.prefill_seconds / fp16.prefill_seconds
                  << " session_match=" << (correct ? "yes" : "no") << '\n';
        return correct ? 0 : 3;
    }
    if (fp16_only) {
        const auto fp16 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kFp16Fp16,
                                   "fp16_tiled_candidate", tokens, capacity);
        const bool finite = std::all_of(fp16.logits.begin(), fp16.logits.end(),
                                        [](float value) { return std::isfinite(value); });
        if (!logits_output.empty()) {
            std::ofstream output(logits_output, std::ios::binary);
            if (!output || !output.write(reinterpret_cast<const char*>(fp16.logits.data()),
                                         static_cast<std::streamsize>(fp16.logits.size() * sizeof(float)))) {
                std::cerr << "failed to write logits to " << logits_output << '\n';
                return 1;
            }
        }
        std::cout << "logits_finite=" << finite << '\n';
        return finite ? 0 : 1;
    }
    Result q8;
    Result fp16;
    if (fp16_first) {
        fp16 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kFp16Fp16,
                        "fp16_tiled_candidate", tokens, capacity);
        q8 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                      "q8_control", tokens, capacity);
    } else {
        q8 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kQ8Q8,
                      "q8_control", tokens, capacity);
        fp16 = run_mode(model, miinfer::prefill_v2::KvCacheQuantMode::kFp16Fp16,
                        "fp16_tiled_candidate", tokens, capacity);
    }
    if (q8.logits.size() != fp16.logits.size()) return 1;
    double dot = 0.0, q8_norm = 0.0, fp16_norm = 0.0;
    double max_abs = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < q8.logits.size(); ++i) {
        finite = finite && std::isfinite(q8.logits[i]) && std::isfinite(fp16.logits[i]);
        max_abs = std::max(max_abs, std::abs(static_cast<double>(q8.logits[i]) - fp16.logits[i]));
        dot += static_cast<double>(q8.logits[i]) * fp16.logits[i];
        q8_norm += static_cast<double>(q8.logits[i]) * q8.logits[i];
        fp16_norm += static_cast<double>(fp16.logits[i]) * fp16.logits[i];
    }
    const double cosine = dot / (std::sqrt(q8_norm * fp16_norm) + 1e-30);
    std::cout << std::setprecision(9) << "logits_max_abs=" << max_abs
              << " logits_cosine=" << cosine
              << " first_token_equal=" << (q8.top1 == fp16.top1)
              << " second_token_equal=" << (q8.second_token == fp16.second_token) << '\n';
    return finite && std::isfinite(max_abs) && std::isfinite(cosine) ? 0 : 1;
}
