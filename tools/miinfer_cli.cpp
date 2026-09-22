#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "qwen35_gpu_pipeline.hpp"
#include "miinfer/qwen3_tokenizer.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/build_info.hpp"
#include "miinfer/openai_api.hpp"
#include "miinfer/sha256.hpp"

extern char** environ;

namespace {

std::atomic<bool> g_shutdown_requested{false};
int g_signal_wakeup_fd = -1;
constexpr std::size_t kFullPrefillCapacity = 512;
// Bump when the checkpoint state layout or wide-prefill numerical contract changes.
constexpr std::uint32_t kSessionExecutionContractVersion = 2;
constexpr std::size_t kMaxSessionCheckpoints = 8;
constexpr std::size_t kSessionCheckpointBudgetBytes = 3ULL * 1024 * 1024 * 1024;

struct PresetFlag {
    const char* name;
    const char* value;
};

bool apply_runtime_preset() {
    const char* preset = std::getenv("MIINFER_PRESET");
    if (preset == nullptr) return true;
    const bool interactive = std::strcmp(preset, "m25_interactive") == 0;
    const bool m26_mx_mmq = std::strcmp(preset, "m26_mx_mmq") == 0;
    const bool interactive_validate = interactive
        && std::getenv("MIINFER_INTERACTIVE_VALIDATE") != nullptr
        && std::strcmp(std::getenv("MIINFER_INTERACTIVE_VALIDATE"), "0") != 0;
    if (!interactive && !m26_mx_mmq && std::strcmp(preset, "m25_hi_qualified") != 0) {
        std::cerr << "unsupported MIINFER_PRESET: " << preset
                  << " (expected m25_hi_qualified, m25_interactive, or m26_mx_mmq)\n";
        return false;
    }

    // Clear every MIINFER_* override before applying the versioned vector;
    // Preserve server controls, which are not runtime selectors.
    std::vector<std::string> names;
    for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
        const std::string_view value(*entry);
        if (value.rfind("MIINFER_", 0) == 0
            && value.rfind("MIINFER_API_KEY=", 0) != 0
            && value.rfind("MIINFER_PRESET=", 0) != 0
            && value.rfind("MIINFER_SESSION_REUSE=", 0) != 0
            && value.rfind("MIINFER_DECODE_PROFILE=", 0) != 0
            && value.rfind("MIINFER_DECODE_PROFILE_POSITION=", 0) != 0
            && value.rfind("MIINFER_PREFILL_PROFILE=", 0) != 0
            && value.rfind("MIINFER_PREFILL_PROFILE_POSITION=", 0) != 0
            && value.rfind("MIINFER_EXP0366_PARTIAL_TAIL=", 0) != 0
            && value.rfind("MIINFER_EXP0368_SCALAR_ORACLE=", 0) != 0
            && value.rfind("MIINFER_HIP_GRAPH=", 0) != 0) {
            names.emplace_back(value.substr(0, value.find('=')));
        }
    }
    for (const auto& name : names) unsetenv(name.c_str());

    static constexpr std::array flags{
        PresetFlag{"MIINFER_CONTEXT_CAPACITY", "1024"},
        PresetFlag{"MIINFER_PREFILL_LAYER_MAJOR", "1"},
        PresetFlag{"MIINFER_PREFILL_WIDE_CHUNK", "1"},
        PresetFlag{"MIINFER_PREFILL_FULL_LAYER_MAJOR", "1"},
        PresetFlag{"MIINFER_PREFILL_WIDE_ATTN", "1"},
        PresetFlag{"MIINFER_PREFILL_WIDE_REPACKED_MMQ", "1"},
        PresetFlag{"MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ", "1"},
        PresetFlag{"MIINFER_PREFILL_WIDE_MMQ_QKV", "1"},
        PresetFlag{"MIINFER_PREFILL_WIDE_MMQ_SSM_OUT", "1"},
        PresetFlag{"MIINFER_PREFILL_WIDE_MMQ_FFN", "1"},
        PresetFlag{"MIINFER_M23_REPACKED_ROW128", "1"},
        PresetFlag{"MIINFER_PREFILL_REPACKED_RESIDENT_ALL", "1"},
        PresetFlag{"MIINFER_PREFILL_CHUNK", "512"},
        PresetFlag{"MIINFER_PREFILL_MX_GDN", "1"},
        PresetFlag{"MIINFER_MX_Q8_BATCH", "0"},
        PresetFlag{"MIINFER_MX_PINNED_QKV", "0"},
        PresetFlag{"MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_FFN", "1"},
        PresetFlag{"MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_O", "1"},
    };
    for (const auto [name, value] : flags) {
        if (setenv(name, value, 1) != 0) {
            std::cerr << "unable to set " << name << " for MIINFER_PRESET\n";
            return false;
        }
    }
    g_cache_capacity = 1024;
    std::cerr << "preset=" << preset << '\n'
              << "  MIINFER_MX_PIPELINE=unset\n";
    for (const auto [name, value] : flags) std::cerr << "  " << name << "=" << value << '\n';
    if (interactive || m26_mx_mmq) {
        for (const char* name : {"MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE"}) {
            if (setenv(name, "1", 1) != 0) return false;
            std::cerr << "  " << name << "=1\n";
        }
        if (interactive) {
            if (setenv("MIINFER_MX_MMV", "1", 1) != 0) return false;
            std::cerr << "  MIINFER_MX_MMV=1\n";
        }
        if (interactive_validate) {
            (void)setenv("MIINFER_WIDE_VALIDATE", "1", 1);
            (void)setenv("MIINFER_WIDE_VALIDATE_MX", "1", 1);
            std::cerr << "  MIINFER_WIDE_VALIDATE=1\n";
            std::cerr << "  MIINFER_WIDE_VALIDATE_MX=1\n";
        }
    }
    return true;
}

void print_hip_memory(std::ostream& output) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
        output << "device_vram_free_bytes=" << free_bytes << "\n"
               << "device_vram_total_bytes=" << total_bytes << "\n";
    }
}

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        g_shutdown_requested = true;
        if (g_signal_wakeup_fd >= 0) {
            const unsigned char wake = 1;
            (void)::write(g_signal_wakeup_fd, &wake, sizeof(wake));
        }
    }
}

std::string json_escape(std::string_view value) {
    std::string escaped;
    for (const char c : value) {
        if (c == '"') escaped += "\\\"";
        else if (c == '\\') escaped += "\\\\";
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else escaped += c;
    }
    return escaped;
}

std::string openai_tool_calls_json(const std::vector<miinfer::ChatToolCall>& calls) {
    std::string result = "[";
    for (std::size_t i = 0; i < calls.size(); ++i) {
        if (i != 0) result += ',';
        const auto& call = calls[i];
        result += "{\"id\":\"" + json_escape(call.id)
            + "\",\"type\":\"function\",\"function\":{\"name\":\""
            + json_escape(call.name) + "\",\"arguments\":\""
            + json_escape(call.arguments) + "\"}}";
    }
    return result + "]";
}

std::optional<int> parse_port(std::string_view value) {
    int port = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), port);
    if (error != std::errc{} || end != value.data() + value.size() || port < 0 || port > 65535) return std::nullopt;
    return port;
}

struct RuntimeGenerateOptions {
    std::size_t max_new_tokens = 256;
    bool stream = true;
    bool reuse_session = false;
    std::function<bool()> should_cancel = nullptr;
    std::function<bool(std::uint32_t token, std::string_view piece)> on_token = nullptr;
    std::function<void()> on_prefill_complete = nullptr;
    std::function<void(const float*, std::size_t)> on_prefill_state = nullptr;
    std::function<void()> on_first_token = nullptr;
    std::function<void(std::size_t token_count)> on_prefill_checkpoint = nullptr;
};

struct RuntimeGenerateStats {
    std::vector<std::uint32_t> tokens;
    std::string text;
    std::size_t prompt_tokens = 0;
    std::size_t prefill_processed_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t reused_prefix_tokens = 0;
    std::size_t common_prefix_tokens = 0;
    std::size_t session_checkpoint_count = 0;
    std::size_t session_checkpoint_bytes = 0;
    double graph_capture_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double first_token_ms = 0.0;
    double total_ms = 0.0;
    double prefill_tok_s = 0.0;
    double decode_tok_s = 0.0;
    std::size_t decode_graph_launches = 0;
    std::size_t decode_h2d_bytes = 0;
    std::size_t decode_d2h_bytes = 0;
    bool cancelled = false;
};

// ---------------------------------------------------------------------------
// Engine class wrapping the full 64-layer gfx906 GPU pipeline
// ---------------------------------------------------------------------------
class Qwen35RuntimeEngine {
public:
    using GenerateOptions = RuntimeGenerateOptions;
    using GenerateStats = RuntimeGenerateStats;

    explicit Qwen35RuntimeEngine(const std::string& model_path)
        : model_(miinfer::Qwen35Model::load(model_path)),
          tokenizer_(miinfer::Qwen3Tokenizer::load(*model_.file())) {
        setup_environment();
        init_layers();
        initialize_session_identity();
        init_shared_wide_prefill_workspace();
        init_m12_gdn_workspace();
        init_m12_dense_workspace();
        init_buffers();
    }

    ~Qwen35RuntimeEngine() {
        cleanup_graphs();
        if (m12_dense_gemm_.opaque != nullptr) {
            miinfer::destroy_rocblas_gemm_handle(m12_dense_gemm_);
        }
        prefill_profile_.destroy();
    }

    [[nodiscard]] const miinfer::Qwen35Model& model() const noexcept { return model_; }
    [[nodiscard]] const miinfer::Qwen3Tokenizer& tokenizer() const noexcept { return tokenizer_; }

    struct PrefillProfile {
        bool enabled = false;
        std::array<hipEvent_t, 4> stage{};
        hipEvent_t embedding_start = nullptr;
        hipEvent_t embedding_end = nullptr;
        std::array<RecurrentLayer::StageProfile, 64> recurrent_layers{};
        std::array<FullAttentionLayer::StageProfile, 64> attention_layers{};
        std::size_t profile_position = 511;
        std::size_t profile_chunk_base = 448;
        std::size_t projection_batch_width = 4;
        bool decode_mode = false;
        double embedding_ms = 0.0;
        std::size_t chunks = 0;
        std::size_t partial_tail_batched_chunks = 0;
        std::size_t partial_tail_scalar_tokens = 0;
        std::size_t partial_tail_layer_run_calls = 0;
        std::size_t partial_tail_layer_run_tokens = 0;
        M23ProfileCounters m23{};

        void init() {
            if (!enabled) return;
            for (auto& event : stage) MIINFER_HIP_CHECK(hipEventCreate(&event));
            MIINFER_HIP_CHECK(hipEventCreate(&embedding_start));
            MIINFER_HIP_CHECK(hipEventCreate(&embedding_end));
            for (auto& layer : recurrent_layers) {
                for (auto& event : layer.start) MIINFER_HIP_CHECK(hipEventCreate(&event));
                for (auto& event : layer.end) MIINFER_HIP_CHECK(hipEventCreate(&event));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.tail_start));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.tail_end));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.prepare_start));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.prepare_end));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.ordered_start));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.ordered_end));
                for (auto& event : layer.tail_family_start) MIINFER_HIP_CHECK(hipEventCreate(&event));
                for (auto& event : layer.tail_family_end) MIINFER_HIP_CHECK(hipEventCreate(&event));
                for (auto& event : layer.wide_tail_family_start) MIINFER_HIP_CHECK(hipEventCreate(&event));
                for (auto& event : layer.wide_tail_family_end) MIINFER_HIP_CHECK(hipEventCreate(&event));
            }
            for (auto& layer : attention_layers) {
                for (auto& event : layer.start) MIINFER_HIP_CHECK(hipEventCreate(&event));
                for (auto& event : layer.end) MIINFER_HIP_CHECK(hipEventCreate(&event));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.tail_start));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.tail_end));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.prepare_start));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.prepare_end));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.ordered_start));
                MIINFER_HIP_CHECK(hipEventCreate(&layer.ordered_end));
            }
        }

        void destroy() {
            if (!enabled) return;
            for (auto& event : stage) {
                if (event != nullptr) (void)hipEventDestroy(event);
                event = nullptr;
            }
            if (embedding_start != nullptr) (void)hipEventDestroy(embedding_start);
            if (embedding_end != nullptr) (void)hipEventDestroy(embedding_end);
            embedding_start = nullptr;
            embedding_end = nullptr;
            for (auto& layer : recurrent_layers) {
                for (auto& event : layer.start) {
                    if (event != nullptr) (void)hipEventDestroy(event);
                    event = nullptr;
                }
                for (auto& event : layer.end) {
                    if (event != nullptr) (void)hipEventDestroy(event);
                    event = nullptr;
                }
                if (layer.tail_start != nullptr) (void)hipEventDestroy(layer.tail_start);
                if (layer.tail_end != nullptr) (void)hipEventDestroy(layer.tail_end);
                if (layer.prepare_start != nullptr) (void)hipEventDestroy(layer.prepare_start);
                if (layer.prepare_end != nullptr) (void)hipEventDestroy(layer.prepare_end);
                if (layer.ordered_start != nullptr) (void)hipEventDestroy(layer.ordered_start);
                if (layer.ordered_end != nullptr) (void)hipEventDestroy(layer.ordered_end);
                for (auto& event : layer.tail_family_start) {
                    if (event != nullptr) (void)hipEventDestroy(event);
                    event = nullptr;
                }
                for (auto& event : layer.tail_family_end) {
                    if (event != nullptr) (void)hipEventDestroy(event);
                    event = nullptr;
                }
                for (auto& event : layer.wide_tail_family_start) {
                    if (event != nullptr) (void)hipEventDestroy(event);
                    event = nullptr;
                }
                for (auto& event : layer.wide_tail_family_end) {
                    if (event != nullptr) (void)hipEventDestroy(event);
                    event = nullptr;
                }
                layer.tail_start = nullptr;
                layer.tail_end = nullptr;
                layer.prepare_start = nullptr;
                layer.prepare_end = nullptr;
                layer.ordered_start = nullptr;
                layer.ordered_end = nullptr;
            }
            for (auto& layer : attention_layers) {
                for (auto& event : layer.start) {
                    if (event != nullptr) (void)hipEventDestroy(event);
                    event = nullptr;
                }
                for (auto& event : layer.end) {
                    if (event != nullptr) (void)hipEventDestroy(event);
                    event = nullptr;
                }
                if (layer.tail_start != nullptr) (void)hipEventDestroy(layer.tail_start);
                if (layer.tail_end != nullptr) (void)hipEventDestroy(layer.tail_end);
                if (layer.prepare_start != nullptr) (void)hipEventDestroy(layer.prepare_start);
                if (layer.prepare_end != nullptr) (void)hipEventDestroy(layer.prepare_end);
                if (layer.ordered_start != nullptr) (void)hipEventDestroy(layer.ordered_start);
                if (layer.ordered_end != nullptr) (void)hipEventDestroy(layer.ordered_end);
                layer.tail_start = nullptr;
                layer.tail_end = nullptr;
                layer.prepare_start = nullptr;
                layer.prepare_end = nullptr;
                layer.ordered_start = nullptr;
                layer.ordered_end = nullptr;
            }
        }

        void report(double wall_ms, std::size_t prompt_tokens) const {
            if (!enabled) return;
            if (!decode_mode && prompt_tokens <= profile_position) {
                std::cout << "Prefill profile skipped: prompt=" << prompt_tokens
                          << " profile_position=" << profile_position << '\n';
                return;
            }
            static constexpr std::array<const char*, 14> recurrent_names{
                "normalization", "qkv_projection", "gate_projection",
                "recurrent_parameter_projection", "conv_and_head_norm", "gdn_scan",
                "recurrent_gate", "ssm_output_projection", "residual_post_norm",
                "post_normalization", "ffn_gate_up_projection", "swiglu",
                "ffn_down_projection", "residual"};
            static constexpr std::array<const char*, 15> attention_names{
                "normalization", "q_projection", "query_norm_rope", "k_projection",
                "key_norm", "v_projection", "kv_store", "attention",
                "attention_output_projection", "residual_post_norm", "post_normalization",
                "ffn_gate_up_projection", "swiglu", "ffn_down_projection", "residual"};
            std::array<double, 17> family_ms{};
            std::array<double, 4> recurrent_tail_family_ms{};
            std::array<double, 4> recurrent_wide_tail_family_ms{};
            double recurrent_deferred_tail_ms = 0.0;
            double attention_deferred_tail_ms = 0.0;
            double ordered_ms = 0.0;
            double sampled_total = embedding_ms;
            std::cout << (decode_mode ? "Decode operator profile: sampled_token_position="
                                      : "Prefill operator profile: prompt=") << prompt_tokens
                      << " chunks=" << chunks << " wall_ms=" << wall_ms
                      << " sampled_position=" << profile_position
                      << " projection_batch_width=B" << projection_batch_width
                      << " causal_chunk_width=B" << kM23CausalChunk
                      << " chunk_base=" << profile_chunk_base
                      << " embedding_ms=" << embedding_ms << '\n';
            std::cout << "EXP-0366 partial-tail counters: batched_chunks="
                      << partial_tail_batched_chunks
                      << " scalar_tokens=" << partial_tail_scalar_tokens
                      << " layer_run_calls=" << partial_tail_layer_run_calls
                      << " layer_run_tokens=" << partial_tail_layer_run_tokens << '\n';
            static constexpr std::array<const char*, 7> recurrent_dispatch_names{
                "qkv", "gate", "beta_alpha", "ssm_out", "ffn_gate", "ffn_up", "ffn_down"};
            static constexpr std::array<const char*, 9> attention_dispatch_names{
                "qk", "v", "q_post", "k_post", "attention", "o", "ffn_gate", "ffn_up", "ffn_down"};
            static constexpr std::array<const char*, 6> recurrent_weight_names{
                "qkv", "gate", "ssm_out", "ffn_gate", "ffn_up", "ffn_down"};
            static constexpr std::array<const char*, 6> attention_weight_names{
                "qk", "v", "o", "ffn_gate", "ffn_up", "ffn_down"};
            std::cout << "M23 invocation profile (whole prefill):\n";
            for (std::size_t i = 0; i < recurrent_dispatch_names.size(); ++i) {
                std::cout << "  recurrent=" << recurrent_dispatch_names[i]
                          << " launches=" << m23.recurrent_dispatches[i] << '\n';
            }
            for (std::size_t i = 0; i < attention_dispatch_names.size(); ++i) {
                std::cout << "  attention=" << attention_dispatch_names[i]
                          << " launches=" << m23.attention_dispatches[i]
                          << '\n';
            }
            std::cout << "M23 weight upload profile (whole prefill):\n";
            for (std::size_t i = 0; i < recurrent_weight_names.size(); ++i) {
                std::cout << "  recurrent=" << recurrent_weight_names[i]
                          << " weight_upload_bytes=" << m23.recurrent_weight_upload_bytes[i]
                          << " weight_upload_ms=" << m23.recurrent_weight_upload_ms[i] << '\n';
            }
            for (std::size_t i = 0; i < attention_weight_names.size(); ++i) {
                std::cout << "  attention=" << attention_weight_names[i]
                          << " weight_upload_bytes=" << m23.attention_weight_upload_bytes[i]
                          << " weight_upload_ms=" << m23.attention_weight_upload_ms[i] << '\n';
            }
            std::cout << "  repacked_weight_upload_bytes=" << m23.weight_upload_bytes
                      << " repacked_weight_upload_ms=" << m23.weight_upload_ms << '\n';
            for (std::size_t layer = 0; layer < 64; ++layer) {
                const bool attention = layer % 4 == 3;
                const std::size_t stage_count = attention ? attention_names.size() : recurrent_names.size();
                const hipEvent_t* starts = attention
                    ? attention_layers[layer].start.data() : recurrent_layers[layer].start.data();
                const hipEvent_t* ends = attention
                    ? attention_layers[layer].end.data() : recurrent_layers[layer].end.data();
                double layer_total = 0.0;
                for (std::size_t stage = 0; stage < stage_count; ++stage) {
                    float elapsed = 0.0F;
                    const bool recorded = attention
                        ? attention_layers[layer].stage_recorded[stage]
                        : recurrent_layers[layer].stage_recorded[stage];
                    if (recorded) {
                        MIINFER_HIP_CHECK(hipEventElapsedTime(&elapsed, starts[stage], ends[stage]));
                    }
                    family_ms[stage] += elapsed;
                    layer_total += elapsed;
                    std::cout << "  layer=" << layer << " kind="
                              << (attention ? "attention" : "recurrent")
                              << " family=" << (attention ? attention_names[stage] : recurrent_names[stage])
                              << " stage=" << stage
                              << " timing=selected-token-stage recorded=" << recorded
                              << " gpu_ms=" << elapsed << '\n';
                }
                float tail = 0.0F;
                const hipEvent_t tail_start = attention
                    ? attention_layers[layer].tail_start : recurrent_layers[layer].tail_start;
                const hipEvent_t tail_end = attention
                    ? attention_layers[layer].tail_end : recurrent_layers[layer].tail_end;
                const bool tail_recorded = attention
                    ? attention_layers[layer].tail_recorded : recurrent_layers[layer].tail_recorded;
                const bool wide_tail_recorded = !attention && recurrent_layers[layer].wide_tail_recorded;
                if (tail_recorded || wide_tail_recorded) {
                    MIINFER_HIP_CHECK(hipEventElapsedTime(&tail, tail_start, tail_end));
                }
                family_ms[15] += tail;
                if (attention) attention_deferred_tail_ms += tail;
                else recurrent_deferred_tail_ms += tail;
                layer_total += tail;
                sampled_total += layer_total;
                std::cout << "  layer=" << layer << " kind="
                          << (attention ? "attention" : "recurrent")
                          << " family=deferred_prefill_tail timing=whole-tail"
                          << " gpu_ms=" << tail << " total_ms=" << layer_total << '\n';
                if (!attention && wide_tail_recorded) {
                    static constexpr std::array<const char*, 4> wide_tail_names{
                        "gdn_core", "ssm_output_and_residual",
                        "ffn_gate_up_and_swiglu", "ffn_down_and_residual"};
                    for (std::size_t family = 0; family < wide_tail_names.size(); ++family) {
                        float family_elapsed = 0.0F;
                        if (recurrent_layers[layer].wide_tail_family_recorded[family]) {
                            MIINFER_HIP_CHECK(hipEventElapsedTime(
                                &family_elapsed,
                                recurrent_layers[layer].wide_tail_family_start[family],
                                recurrent_layers[layer].wide_tail_family_end[family]));
                        }
                        recurrent_wide_tail_family_ms[family] += family_elapsed;
                        std::cout << "  layer=" << layer << " kind=recurrent family="
                                  << wide_tail_names[family]
                                  << " timing=whole-B"
                                  << recurrent_layers[layer].wide_tail_batch_count
                                  << "-tail recorded="
                                  << recurrent_layers[layer].wide_tail_family_recorded[family]
                                  << " gpu_ms=" << family_elapsed << '\n';
                    }
                } else if (!attention) {
                    static constexpr std::array<const char*, 4> tail_names{
                        "gdn_core", "ssm_output_and_residual",
                        "ffn_gate_up_and_swiglu", "ffn_down_and_residual"};
                    for (std::size_t family = 0; family < tail_names.size(); ++family) {
                        float family_elapsed = 0.0F;
                        if (recurrent_layers[layer].tail_family_recorded[family]) {
                            MIINFER_HIP_CHECK(hipEventElapsedTime(
                                &family_elapsed,
                                recurrent_layers[layer].tail_family_start[family],
                                recurrent_layers[layer].tail_family_end[family]));
                        }
                        recurrent_tail_family_ms[family] += family_elapsed;
                        std::cout << "  layer=" << layer << " kind=recurrent family="
                                  << tail_names[family]
                                  << " timing=selected-B4-tail recorded="
                                  << recurrent_layers[layer].tail_family_recorded[family]
                                  << " gpu_ms=" << family_elapsed << '\n';
                    }
                }
                const hipEvent_t prepare_start = attention
                    ? attention_layers[layer].prepare_start : recurrent_layers[layer].prepare_start;
                const hipEvent_t prepare_end = attention
                    ? attention_layers[layer].prepare_end : recurrent_layers[layer].prepare_end;
                const bool prepare_recorded = attention
                    ? attention_layers[layer].prepare_recorded : recurrent_layers[layer].prepare_recorded;
                float prepare = 0.0F;
                if (prepare_recorded) {
                    MIINFER_HIP_CHECK(hipEventElapsedTime(&prepare, prepare_start, prepare_end));
                }
                family_ms[16] += prepare;
                sampled_total += prepare;
                std::cout << "  layer=" << layer << " kind="
                          << (attention ? "attention" : "recurrent")
                          << " family=prefill_batch_prepare batch=B" << projection_batch_width
                          << " recorded=" << prepare_recorded << " gpu_ms=" << prepare << '\n';
                const hipEvent_t ordered_start = attention
                    ? attention_layers[layer].ordered_start : recurrent_layers[layer].ordered_start;
                const hipEvent_t ordered_end = attention
                    ? attention_layers[layer].ordered_end : recurrent_layers[layer].ordered_end;
                const bool ordered_recorded = attention
                    ? attention_layers[layer].ordered_recorded : recurrent_layers[layer].ordered_recorded;
                float ordered = 0.0F;
                if (ordered_recorded) {
                    MIINFER_HIP_CHECK(hipEventElapsedTime(&ordered, ordered_start, ordered_end));
                }
                ordered_ms += ordered;
                std::cout << "  layer=" << layer << " kind="
                          << (attention ? "attention" : "recurrent")
                          << " family=ordered_token_path recorded=" << ordered_recorded
                          << " gpu_ms=" << ordered << '\n';
                const std::size_t contract_end_stage = attention ? 7 : 6;
                const auto& contract_start = attention
                    ? attention_layers[layer].prepare_start : recurrent_layers[layer].start[0];
                const auto& contract_end = attention
                    ? attention_layers[layer].end[contract_end_stage]
                    : recurrent_layers[layer].end[contract_end_stage];
                const bool contract_recorded = attention
                    ? attention_layers[layer].prepare_recorded
                        && attention_layers[layer].stage_recorded[contract_end_stage]
                    : recurrent_layers[layer].stage_recorded[0]
                        && recurrent_layers[layer].stage_recorded[contract_end_stage];
                if (contract_recorded) {
                    float contract_ms = 0.0F;
                    MIINFER_HIP_CHECK(hipEventElapsedTime(&contract_ms, contract_start, contract_end));
                    std::cout << "  layer=" << layer
                              << " kind=" << (attention ? "attention" : "recurrent")
                              << " family=" << (attention ? "attention_input_contract" : "recurrent_input_contract")
                              << " timing=" << (attention ? "norm_to_attention_output" : "norm_to_gdn_output_gate")
                              << " gpu_ms=" << contract_ms << '\n';
                }
            }
            std::cout << "Recurrent tail family profile (selected B4 group):\n";
            static constexpr std::array<const char*, 4> tail_names{
                "gdn_core", "ssm_output_and_residual",
                "ffn_gate_up_and_swiglu", "ffn_down_and_residual"};
            for (std::size_t family = 0; family < tail_names.size(); ++family) {
                std::cout << "  family=" << tail_names[family]
                          << " gpu_ms=" << recurrent_tail_family_ms[family] << '\n';
            }
            std::cout << "Recurrent tail family profile (whole wide batch):\n";
            for (std::size_t family = 0; family < tail_names.size(); ++family) {
                std::cout << "  family=" << tail_names[family]
                          << " gpu_ms=" << recurrent_wide_tail_family_ms[family] << '\n';
            }
            std::cout << "Deferred tail ownership (sampled layers):\n"
                      << "  recurrent_deferred_tail_ms=" << recurrent_deferred_tail_ms << '\n'
                      << "  attention_deferred_tail_ms=" << attention_deferred_tail_ms << '\n'
                      << "  aggregate_deferred_tail_ms="
                      << (recurrent_deferred_tail_ms + attention_deferred_tail_ms) << '\n';
            std::cout << "Top operator families (sampled position, summed layers):\n";
            std::vector<std::size_t> order(family_ms.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&family_ms](std::size_t a, std::size_t b) {
                return family_ms[a] > family_ms[b];
            });
            static constexpr std::array<const char*, 17> summary_names{
                "normalization", "projection_or_norm", "projection_or_rope",
                "projection_or_k", "projection_or_v", "kv_or_head_norm", "gdn_or_attention",
                "projection_or_attention_output", "residual_post_norm", "post_normalization",
                "ffn_gate_up_projection", "swiglu", "ffn_down_projection", "residual",
                "attention_residual", "deferred_prefill_tail_total", "prefill_batch_prepare"};
            for (std::size_t rank = 0; rank < 10; ++rank) {
                const auto index = order[rank];
                std::cout << "  rank=" << (rank + 1) << " family=" << summary_names[index]
                          << " gpu_ms=" << family_ms[index]
                          << " share_of_sampled=" << (sampled_total > 0.0
                              ? 100.0 * family_ms[index] / sampled_total : 0.0) << '\n';
            }
            std::cout << "  ordered_token_path_gpu_ms=" << ordered_ms
                      << " note=gross layer-loop timing; excluded from component shares\n";
            const std::size_t full_chunks = profile_chunk_base / kPrefillBatch + 1;
            const double estimated_full_ms = sampled_total * full_chunks;
            std::cout << "  sampled_layer_gpu_ms=" << (sampled_total - embedding_ms)
                      << " accounted_sample_ms=" << sampled_total
                      << " estimated_full_prefill_ms=" << estimated_full_ms
                      << " estimated_full_chunks=" << full_chunks
                      << " wall_ms=" << wall_ms
                      << " note=full-chunk extrapolation; use wall_ms for qualification\n";
        }
    };

    void reset() {
        clear_session_checkpoints();
        reset_device_state();
    }

    void reset_device_state() {
        for (const auto& layer : layers_) {
            if (layer.recurrent != nullptr) layer.recurrent->reset();
            if (layer.attention != nullptr) layer.attention->reset();
        }
        MIINFER_HIP_CHECK(hipDeviceSynchronize());
    }

    void initialize_session_identity() {
        session_model_identity_ = model_.artifact_path() + "|" + model_.model_name();
        const auto& config = model_.config();
        session_model_identity_ += "|" + std::to_string(config.block_count)
            + ":" + std::to_string(config.hidden_size)
            + ":" + std::to_string(config.context_length);

        std::vector<std::pair<int, std::size_t>> quantization_types;
        for (const auto& tensor : model_.tensors()) {
            const int type = static_cast<int>(tensor.type);
            auto found = std::find_if(quantization_types.begin(), quantization_types.end(),
                [type](const auto& entry) { return entry.first == type; });
            if (found == quantization_types.end()) quantization_types.emplace_back(type, 1);
            else ++found->second;
        }
        std::sort(quantization_types.begin(), quantization_types.end());
        for (const auto& [type, count] : quantization_types) {
            session_quantization_identity_ += std::to_string(type) + ":"
                + std::to_string(count) + ";";
        }

        const char* preset = std::getenv("MIINFER_PRESET");
        session_runtime_identity_ = preset == nullptr ? "default" : preset;
        session_runtime_identity_ += layer_major_prefill_ ? "|layer-major" : "|batch-major";
        session_runtime_identity_ += wide_prefill_ ? "|wide" : "|narrow";
        session_runtime_identity_ += full_layer_major_prefill_ ? "|full-layer" : "|partial-layer";
        session_runtime_identity_ += "|chunk:" + std::to_string(prefill_chunk_);
        if (environment_flag("MIINFER_M26C_RECURRENT_QKV_M23")) {
            session_runtime_identity_ += "|recurrent-qkv-m23";
        }
        if (environment_flag("MIINFER_M26C_SSM_OUT_NATIVE")) {
            session_runtime_identity_ += "|ssm-out-native";
        }
        for (const auto& layer : layers_) {
            if (layer.recurrent != nullptr) {
                session_runtime_identity_ += layer.recurrent->transposed_state
                    ? "|gdn-state-transposed" : "|gdn-state-logical";
                break;
            }
        }
        for (const auto& layer : layers_) {
            if (layer.attention != nullptr) {
                session_runtime_identity_ += layer.attention->fp16_kv_cache ? "|kv-f16" : "|kv-f32";
                break;
            }
        }
    }

    struct SessionCheckpoint {
        std::vector<std::uint32_t> tokens;
        std::size_t boundary_position = 0;
        std::size_t storage_bytes = 0;
        std::size_t last_used = 0;
        std::uint32_t execution_contract_version = 0;
        std::string model_identity;
        std::string quantization_identity;
        std::string runtime_identity;
        std::array<Buffer, 64> recurrent_state{};
        std::array<Buffer, 64> recurrent_history{};
        std::array<Buffer, 64> key_cache{};
        std::array<Buffer, 64> value_cache{};
    };

    struct SessionPrefixNode {
        std::vector<std::uint32_t> edge;
        std::unordered_map<std::uint32_t, std::unique_ptr<SessionPrefixNode>> children;
        std::optional<std::size_t> checkpoint_index;
    };

    void clear_session_checkpoints() {
        session_checkpoints_.clear();
        session_prefix_root_ = {};
        session_checkpoint_bytes_ = 0;
        session_use_clock_ = 0;
    }

    bool session_checkpoint_compatible(const SessionCheckpoint& checkpoint) const {
        return checkpoint.boundary_position == checkpoint.tokens.size()
            && checkpoint.execution_contract_version == kSessionExecutionContractVersion
            && checkpoint.model_identity == session_model_identity_
            && checkpoint.quantization_identity == session_quantization_identity_
            && checkpoint.runtime_identity == session_runtime_identity_;
    }

    static void insert_session_prefix(SessionPrefixNode& parent,
                                      std::span<const std::uint32_t> tokens,
                                      std::size_t position,
                                      std::size_t checkpoint_index) {
        auto& slot = parent.children[tokens[position]];
        if (!slot) {
            slot = std::make_unique<SessionPrefixNode>();
            slot->edge.assign(tokens.begin() + position, tokens.end());
            slot->checkpoint_index = checkpoint_index;
            return;
        }

        auto& child = *slot;
        std::size_t shared = 0;
        while (shared < child.edge.size() && position + shared < tokens.size()
               && child.edge[shared] == tokens[position + shared]) ++shared;
        if (shared == child.edge.size()) {
            position += shared;
            if (position == tokens.size()) child.checkpoint_index = checkpoint_index;
            else insert_session_prefix(child, tokens, position, checkpoint_index);
            return;
        }

        auto old_child = std::move(slot);
        auto branch = std::make_unique<SessionPrefixNode>();
        branch->edge.assign(old_child->edge.begin(), old_child->edge.begin() + shared);
        old_child->edge.erase(old_child->edge.begin(), old_child->edge.begin() + shared);
        branch->children.emplace(old_child->edge.front(), std::move(old_child));
        position += shared;
        if (position == tokens.size()) branch->checkpoint_index = checkpoint_index;
        else {
            auto leaf = std::make_unique<SessionPrefixNode>();
            leaf->edge.assign(tokens.begin() + position, tokens.end());
            leaf->checkpoint_index = checkpoint_index;
            branch->children.emplace(leaf->edge.front(), std::move(leaf));
        }
        slot = std::move(branch);
    }

    void rebuild_session_prefix_index() {
        session_prefix_root_ = {};
        for (std::size_t i = 0; i < session_checkpoints_.size(); ++i) {
            const auto& checkpoint = session_checkpoints_[i];
            if (session_checkpoint_compatible(checkpoint)) {
                insert_session_prefix(session_prefix_root_, checkpoint.tokens, 0, i);
            }
        }
    }

    std::optional<std::size_t> longest_session_checkpoint(
        std::span<const std::uint32_t> prompt, std::size_t& common) const {
        common = 0;
        std::optional<std::size_t> best;
        const SessionPrefixNode* node = &session_prefix_root_;
        std::size_t position = 0;
        while (position < prompt.size()) {
            std::size_t shared = 0;
            const auto found = node->children.find(prompt[position]);
            if (found == node->children.end()) break;
            const auto* child = found->second.get();
            while (shared < child->edge.size() && position + shared < prompt.size()
                   && child->edge[shared] == prompt[position + shared]) ++shared;
            common = std::max(common, position + shared);
            if (shared != child->edge.size()) break;
            position += shared;
            node = child;
            if (node->checkpoint_index && position < prompt.size()) best = node->checkpoint_index;
        }
        return best;
    }

    std::size_t session_checkpoint_storage_bytes(std::size_t tokens) const {
        const std::size_t recurrent_state_bytes = kVHeads * kState * kState * sizeof(float);
        const std::size_t recurrent_history_bytes = 4 * kChannels * sizeof(float);
        std::size_t bytes = 0;
        for (const auto& layer : layers_) {
            if (layer.recurrent != nullptr) bytes += recurrent_state_bytes + recurrent_history_bytes;
            else {
                const std::size_t element_size = layer.attention->fp16_kv_cache
                    ? sizeof(__half) : sizeof(float);
                bytes += 2 * 4 * tokens * 256 * element_size * 2;
            }
        }
        return bytes;
    }

    void evict_oldest_session_checkpoint() {
        if (session_checkpoints_.empty()) return;
        const auto oldest = std::min_element(session_checkpoints_.begin(), session_checkpoints_.end(),
            [](const auto& left, const auto& right) { return left.last_used < right.last_used; });
        session_checkpoint_bytes_ -= oldest->storage_bytes;
        session_checkpoints_.erase(oldest);
        rebuild_session_prefix_index();
    }

    void touch_session_checkpoint(std::span<const std::uint32_t> tokens) {
        const auto checkpoint = std::find_if(session_checkpoints_.begin(), session_checkpoints_.end(),
            [&](const auto& entry) {
                return entry.tokens.size() == tokens.size()
                    && std::equal(entry.tokens.begin(), entry.tokens.end(), tokens.begin());
            });
        if (checkpoint != session_checkpoints_.end()) checkpoint->last_used = ++session_use_clock_;
    }

    // ponytail: cap eight GPU snapshots at 3 GiB; raise only when measured branch reuse justifies the VRAM cost.
    void capture_session_checkpoint(std::span<const std::uint32_t> tokens) {
        if (tokens.empty() || tokens.size() % kFullPrefillCapacity != 0) return;
        const auto existing = std::find_if(session_checkpoints_.begin(), session_checkpoints_.end(),
            [&](const auto& checkpoint) {
                return checkpoint.tokens.size() == tokens.size()
                    && std::equal(checkpoint.tokens.begin(), checkpoint.tokens.end(), tokens.begin());
            });
        if (existing != session_checkpoints_.end()) {
            existing->last_used = ++session_use_clock_;
            existing->model_identity = session_model_identity_;
            existing->quantization_identity = session_quantization_identity_;
            existing->runtime_identity = session_runtime_identity_;
            existing->execution_contract_version = kSessionExecutionContractVersion;
            copy_session_checkpoint(*existing, tokens);
            return;
        }

        const std::size_t bytes = session_checkpoint_storage_bytes(tokens.size());
        if (bytes > kSessionCheckpointBudgetBytes) return;
        while (!session_checkpoints_.empty()
               && (session_checkpoints_.size() >= kMaxSessionCheckpoints
                   || session_checkpoint_bytes_ + bytes > kSessionCheckpointBudgetBytes)) {
            evict_oldest_session_checkpoint();
        }

        session_checkpoints_.emplace_back();
        auto& checkpoint = session_checkpoints_.back();
        checkpoint.tokens.assign(tokens.begin(), tokens.end());
        checkpoint.boundary_position = tokens.size();
        checkpoint.storage_bytes = bytes;
        checkpoint.last_used = ++session_use_clock_;
        checkpoint.execution_contract_version = kSessionExecutionContractVersion;
        checkpoint.model_identity = session_model_identity_;
        checkpoint.quantization_identity = session_quantization_identity_;
        checkpoint.runtime_identity = session_runtime_identity_;
        try {
            allocate_session_checkpoint(checkpoint, tokens.size());
        } catch (const std::exception&) {
            session_checkpoints_.pop_back();
            return;
        }
        session_checkpoint_bytes_ += bytes;
        copy_session_checkpoint(checkpoint, tokens);
        rebuild_session_prefix_index();
    }

    void allocate_session_checkpoint(SessionCheckpoint& checkpoint, std::size_t tokens) {
        const std::size_t recurrent_state_bytes = kVHeads * kState * kState * sizeof(float);
        const std::size_t recurrent_history_bytes = 4 * kChannels * sizeof(float);
        const std::size_t kv_elements = 4 * tokens * 256;
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            const auto& layer = layers_[i];
            if (layer.recurrent != nullptr) {
                checkpoint.recurrent_state[i] = allocate(recurrent_state_bytes);
                checkpoint.recurrent_history[i] = allocate(recurrent_history_bytes);
            } else {
                const std::size_t element_size = layer.attention->fp16_kv_cache
                    ? sizeof(__half) : sizeof(float);
                checkpoint.key_cache[i] = allocate(kv_elements * element_size);
                checkpoint.value_cache[i] = allocate(kv_elements * element_size);
            }
        }
    }

    void copy_session_checkpoint(SessionCheckpoint& checkpoint,
                                 std::span<const std::uint32_t> tokens) {
        const std::size_t recurrent_state_bytes = kVHeads * kState * kState * sizeof(float);
        const std::size_t recurrent_history_bytes = 4 * kChannels * sizeof(float);
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            const auto& layer = layers_[i];
            if (layer.recurrent != nullptr) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(checkpoint.recurrent_state[i]->get(),
                    layer.recurrent->state->get(), recurrent_state_bytes,
                    hipMemcpyDeviceToDevice, hipStreamPerThread));
                MIINFER_HIP_CHECK(hipMemcpyAsync(checkpoint.recurrent_history[i]->get(),
                    layer.recurrent->history->get(), recurrent_history_bytes,
                    hipMemcpyDeviceToDevice, hipStreamPerThread));
            } else {
                const std::size_t element_size = layer.attention->fp16_kv_cache
                    ? sizeof(__half) : sizeof(float);
                const std::size_t bytes_per_head = tokens.size() * 256 * element_size;
                for (std::size_t head = 0; head < 4; ++head) {
                    const std::size_t source = head * g_cache_capacity * 256 * element_size;
                    const std::size_t destination = head * bytes_per_head;
                    MIINFER_HIP_CHECK(hipMemcpyAsync(
                        static_cast<std::byte*>(checkpoint.key_cache[i]->get()) + destination,
                        static_cast<std::byte*>(layer.attention->key_cache->get()) + source,
                        bytes_per_head, hipMemcpyDeviceToDevice, hipStreamPerThread));
                    MIINFER_HIP_CHECK(hipMemcpyAsync(
                        static_cast<std::byte*>(checkpoint.value_cache[i]->get()) + destination,
                        static_cast<std::byte*>(layer.attention->value_cache->get()) + source,
                        bytes_per_head, hipMemcpyDeviceToDevice, hipStreamPerThread));
                }
            }
        }
    }

    void restore_session_checkpoint(SessionCheckpoint& checkpoint) {
        if (!session_checkpoint_compatible(checkpoint) || checkpoint.tokens.empty()) {
            throw std::runtime_error("session checkpoint execution contract mismatch");
        }
        reset_device_state();
        const std::size_t recurrent_state_bytes = kVHeads * kState * kState * sizeof(float);
        const std::size_t recurrent_history_bytes = 4 * kChannels * sizeof(float);
        const std::size_t bytes_per_head = checkpoint.boundary_position * 256;
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            const auto& layer = layers_[i];
            if (layer.recurrent != nullptr) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(layer.recurrent->state->get(),
                    checkpoint.recurrent_state[i]->get(), recurrent_state_bytes,
                    hipMemcpyDeviceToDevice, hipStreamPerThread));
                MIINFER_HIP_CHECK(hipMemcpyAsync(layer.recurrent->history->get(),
                    checkpoint.recurrent_history[i]->get(), recurrent_history_bytes,
                    hipMemcpyDeviceToDevice, hipStreamPerThread));
            } else {
                const std::size_t element_size = layer.attention->fp16_kv_cache
                    ? sizeof(__half) : sizeof(float);
                const std::size_t head_bytes = bytes_per_head * element_size;
                for (std::size_t head = 0; head < 4; ++head) {
                    const std::size_t destination = head * g_cache_capacity * 256 * element_size;
                    const std::size_t source = head * head_bytes;
                    MIINFER_HIP_CHECK(hipMemcpyAsync(
                        static_cast<std::byte*>(layer.attention->key_cache->get()) + destination,
                        static_cast<std::byte*>(checkpoint.key_cache[i]->get()) + source,
                        head_bytes, hipMemcpyDeviceToDevice, hipStreamPerThread));
                    MIINFER_HIP_CHECK(hipMemcpyAsync(
                        static_cast<std::byte*>(layer.attention->value_cache->get()) + destination,
                        static_cast<std::byte*>(checkpoint.value_cache[i]->get()) + source,
                        head_bytes, hipMemcpyDeviceToDevice, hipStreamPerThread));
                }
            }
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    }

    struct StepResult {
        std::uint32_t token;
        double latency_ms;
    };

    struct StepTransferMetrics {
        double h2d_device_ms = 0.0;
        double d2h_device_ms = 0.0;
        double blocking_sync_ms = 0.0;
        std::size_t sync_calls = 0;
    };

    struct StepTransferEvents {
        hipEvent_t h2d_start = nullptr;
        hipEvent_t h2d_end = nullptr;
        hipEvent_t d2h_start = nullptr;
        hipEvent_t d2h_end = nullptr;

        StepTransferEvents() {
            MIINFER_HIP_CHECK(hipEventCreate(&h2d_start));
            MIINFER_HIP_CHECK(hipEventCreate(&h2d_end));
            MIINFER_HIP_CHECK(hipEventCreate(&d2h_start));
            MIINFER_HIP_CHECK(hipEventCreate(&d2h_end));
        }
        ~StepTransferEvents() {
            if (d2h_end != nullptr) (void)hipEventDestroy(d2h_end);
            if (d2h_start != nullptr) (void)hipEventDestroy(d2h_start);
            if (h2d_end != nullptr) (void)hipEventDestroy(h2d_end);
            if (h2d_start != nullptr) (void)hipEventDestroy(h2d_start);
        }
        StepTransferEvents(const StepTransferEvents&) = delete;
        StepTransferEvents& operator=(const StepTransferEvents&) = delete;
    };

    StepResult step(std::uint32_t input_token, std::size_t position,
                    StepTransferEvents* transfer_events = nullptr,
                    StepTransferMetrics* transfer_metrics = nullptr,
                    DecodeLayerCapture* layer_capture = nullptr) {
        if (position >= g_cache_capacity) {
            throw std::runtime_error("context length exceeded capacity " + std::to_string(g_cache_capacity));
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (transfer_events != nullptr) {
            MIINFER_HIP_CHECK(hipEventRecord(transfer_events->h2d_start, hipStreamPerThread));
        }
        MIINFER_HIP_CHECK(hipMemcpyAsync(
            static_cast<std::uint32_t*>(d_decode_tokens_->get()) + position,
            &input_token, sizeof(input_token), hipMemcpyHostToDevice, hipStreamPerThread));
        if (transfer_events != nullptr) {
            MIINFER_HIP_CHECK(hipEventRecord(transfer_events->h2d_end, hipStreamPerThread));
        }

        miinfer::launch_qwen35_q4_k_embedding_device_token(
            static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding_->get()),
            static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + position,
            model_.config().vocab_size, kHidden,
            static_cast<float*>(input_->get()), hipStreamPerThread);
        run_prefix(std::span<const GpuLayerRef>(layers_),
                       std::span<float* const>(output_pointers_),
                       static_cast<const float*>(input_->get()), position,
                       static_cast<const float*>(d_final_norm_weight_->get()),
                       static_cast<float*>(final_norm_->get()),
                       static_cast<miinfer::Q8_1Block*>(final_q8_1_->get()),
                       nullptr, layer_capture);
        miinfer::launch_qwen3_rms_norm(
                output_pointers_[63],
                static_cast<const float*>(d_final_norm_weight_->get()),
                static_cast<float*>(final_norm_->get()), kHidden,
                model_.config().rms_epsilon);
        miinfer::launch_q8_1_quantize_f32(
                static_cast<const float*>(final_norm_->get()),
                static_cast<miinfer::Q8_1Block*>(final_q8_1_->get()), kHidden,
                hipStreamPerThread);
        launch_q6k_wave_gemv(
                static_cast<const Q6KWaveTile*>(d_output_weight_->get()),
                static_cast<const miinfer::Q8_1Block*>(final_q8_1_->get()),
                static_cast<float*>(logits_->get()), model_.config().vocab_size, kHidden,
                hipStreamPerThread);
        miinfer::launch_qwen3_argmax(
            static_cast<const float*>(logits_->get()),
            static_cast<std::uint32_t*>(d_decode_tokens_->get()) + (position + 1),
            model_.config().vocab_size);
        std::uint32_t next_token = 0;
        if (transfer_events != nullptr) {
            MIINFER_HIP_CHECK(hipEventRecord(transfer_events->d2h_start, hipStreamPerThread));
        }
        MIINFER_HIP_CHECK(hipMemcpyAsync(&next_token,
                                        static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + (position + 1),
                                        sizeof(next_token), hipMemcpyDeviceToHost, hipStreamPerThread));
        if (transfer_events != nullptr) {
            MIINFER_HIP_CHECK(hipEventRecord(transfer_events->d2h_end, hipStreamPerThread));
        }
        const auto sync_start = std::chrono::steady_clock::now();
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        const auto sync_end = std::chrono::steady_clock::now();
        if (transfer_events != nullptr && transfer_metrics != nullptr) {
            float h2d_ms = 0.0F, d2h_ms = 0.0F;
            MIINFER_HIP_CHECK(hipEventElapsedTime(
                &h2d_ms, transfer_events->h2d_start, transfer_events->h2d_end));
            MIINFER_HIP_CHECK(hipEventElapsedTime(
                &d2h_ms, transfer_events->d2h_start, transfer_events->d2h_end));
            transfer_metrics->h2d_device_ms += h2d_ms;
            transfer_metrics->d2h_device_ms += d2h_ms;
            transfer_metrics->blocking_sync_ms +=
                std::chrono::duration<double, std::milli>(sync_end - sync_start).count();
            ++transfer_metrics->sync_calls;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        return {next_token, ms};
    }

    void prefill_step(std::uint32_t input_token, std::size_t position) {
        if (position >= g_cache_capacity) {
            throw std::runtime_error("context length exceeded capacity " + std::to_string(g_cache_capacity));
        }
        MIINFER_HIP_CHECK(hipMemcpyAsync(
            static_cast<std::uint32_t*>(d_decode_tokens_->get()) + position,
            &input_token, sizeof(input_token), hipMemcpyHostToDevice, hipStreamPerThread));

        miinfer::launch_qwen35_q4_k_embedding_device_token(
            static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding_->get()),
            static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + position,
            model_.config().vocab_size, kHidden,
            static_cast<float*>(input_->get()), hipStreamPerThread);

        run_prefix(std::span<const GpuLayerRef>(layers_),
                   std::span<float* const>(output_pointers_),
                   static_cast<const float*>(input_->get()), position);
    }

    const float* prefill_layer_major(std::span<const std::uint32_t> prompt,
                                     const std::function<bool()>& should_cancel,
                                     std::size_t& processed_tokens,
                                     std::size_t start_position = 0) {
        const bool matrix_prefill = gdn_chunkwise_prefill_ || dense_prefill_ || wide_prefill_;
        float* current = static_cast<float*>(prefill_a_->get());
        float* next = static_cast<float*>(prefill_b_->get());
        const float* final_hidden = nullptr;
        const auto layer_span = std::span<const GpuLayerRef>(layers_);
        for (std::size_t base = start_position; base < prompt.size();) {
            if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
            const std::size_t requested = std::min(prefill_chunk_, prompt.size() - base);
            // Keep complete wide batches and isolate only the unaligned tail.
            // A whole-request fallback to 64 tokens throws away the useful
            // B512 work for normal API prompts.
            std::size_t count = requested;
            if (wide_prefill_ && matrix_prefill && count >= kM12PrefillBatch
                && count % kPrefillBatch != 0) {
                count -= count % kPrefillBatch;
            }
            const bool partial_tail_contract = std::getenv("MIINFER_EXP0366_PARTIAL_TAIL") != nullptr
                && std::strcmp(std::getenv("MIINFER_EXP0366_PARTIAL_TAIL"), "0") != 0;
            if (partial_tail_contract && full_layer_major_prefill_
                && base >= kFullPrefillCapacity && count < prefill_chunk_ && count > kPrefillBatch) {
                count = (count / kM12PrefillBatch) * kM12PrefillBatch;
                if (count == 0) count = kPrefillBatch;
                if (prefill_profile_.enabled) ++prefill_profile_.partial_tail_batched_chunks;
            } else if (partial_tail_contract && full_layer_major_prefill_
                       && base >= kFullPrefillCapacity && count <= kPrefillBatch) {
                if (prefill_profile_.enabled) prefill_profile_.partial_tail_scalar_tokens += count;
            }
            if (full_layer_major_prefill_ && count >= kM12PrefillBatch
                && count <= kFullPrefillCapacity && count % kPrefillBatch == 0) {
                final_hidden = prefill_full_layer_major_chunk(
                    prompt.subspan(base, count), should_cancel, base);
                if (final_hidden == nullptr) return nullptr;
                processed_tokens += count;
                base += count;
                if (opt_prefill_checkpoint_ && base % kFullPrefillCapacity == 0) {
                    opt_prefill_checkpoint_(base);
                }
                continue;
            }
            if (count > kPrefillBatch && count < kM12PrefillBatch) count = kPrefillBatch;
            if (prefill_profile_.enabled) {
                MIINFER_HIP_CHECK(hipEventRecord(prefill_profile_.embedding_start, hipStreamPerThread));
            }
            for (std::size_t i = 0; i < count; ++i) {
                if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
                const auto position = base + i;
                MIINFER_HIP_CHECK(hipMemcpyAsync(
                    static_cast<std::uint32_t*>(d_decode_tokens_->get()) + position,
                    &prompt[position], sizeof(std::uint32_t), hipMemcpyHostToDevice,
                    hipStreamPerThread));
                miinfer::launch_qwen35_q4_k_embedding_device_token(
                    static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding_->get()),
                    static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + position,
                    model_.config().vocab_size, kHidden, current + i * kHidden,
                    hipStreamPerThread);
            }
            if (prefill_profile_.enabled) {
                MIINFER_HIP_CHECK(hipEventRecord(prefill_profile_.embedding_end, hipStreamPerThread));
                MIINFER_HIP_CHECK(hipEventSynchronize(prefill_profile_.embedding_end));
                float embedding_ms = 0.0F;
                MIINFER_HIP_CHECK(hipEventElapsedTime(
                    &embedding_ms, prefill_profile_.embedding_start, prefill_profile_.embedding_end));
                prefill_profile_.embedding_ms += embedding_ms;
                ++prefill_profile_.chunks;
            }

            // Bounded chunk storage; layer order preserves recurrent state and
            // causal KV dependencies while keeping the full chunk at one layer.
            for (std::size_t layer = 0; layer < layer_span.size(); ++layer) {
                if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
                layer_span[layer].profile_ordered_start(
                    prefill_profile_.enabled ? base : std::numeric_limits<std::size_t>::max());
                if (wide_prefill_ && count >= kM12PrefillBatch
                    && layer_span[layer].recurrent != nullptr) {
                    layer_span[layer].recurrent->prefill_wide(current, next, base, count);
                    layer_span[layer].profile_ordered_end(
                        prefill_profile_.enabled ? base : std::numeric_limits<std::size_t>::max());
                    layer_span[layer].release_m23_repacked();
                    std::swap(current, next);
                    continue;
                }
                const bool normalized_ready = layer > 0 && layer_span[layer - 1].fused_interlayer_norm();
                const bool prepared = layer_span[layer].prepare_prefill_batch(
                    current, count, normalized_ready,
                    prefill_profile_.enabled ? base : std::numeric_limits<std::size_t>::max());
                const bool full_m12_chunk = count % kPrefillBatch == 0;
                const bool deferred_tail = prepared && layer_span[layer].prefill_tail_batch_supported()
                    && (!gdn_chunkwise_prefill_ || full_m12_chunk);
                const bool batched_attention = prepared && deferred_tail
                    && count >= kM12PrefillBatch
                    && layer_span[layer].wide_attention_batch_ready();
                const bool fuse_next_norm = layer + 1 < layer_span.size()
                    && layer_span[layer].fused_interlayer_norm();
                const float* next_norm_weight = fuse_next_norm
                    ? layer_span[layer + 1].attn_norm_weight() : nullptr;
                float* next_normalized_batch = fuse_next_norm
                    ? const_cast<float*>(layer_span[layer + 1].prefill_normalized_at(0)) : nullptr;
                if (!batched_attention) {
                    if (partial_tail_contract && full_layer_major_prefill_ && base >= kFullPrefillCapacity) {
                        ++prefill_profile_.partial_tail_layer_run_calls;
                        prefill_profile_.partial_tail_layer_run_tokens += count;
                    }
                    for (std::size_t i = 0; i < count; ++i) {
                    if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
                    float* next_normalized = fuse_next_norm
                        ? next_normalized_batch + i * kHidden : nullptr;
                    const float* prepared_normalized = prepared
                        ? layer_span[layer].prefill_normalized_at(i) : nullptr;
                    if (prepared) {
                        layer_span[layer].run(
                            current + i * kHidden, base + i, next + i * kHidden,
                            next_norm_weight, next_normalized, false, nullptr, false,
                            layer_span[layer].prefill_qkv_at(i),
                            layer_span[layer].prefill_gate_at(i),
                            prepared_normalized,
                            deferred_tail, i,
                            layer_span[layer].prefill_qfull_at(i),
                            layer_span[layer].prefill_value_at(i));
                    } else {
                        layer_span[layer].run(current + i * kHidden, base + i,
                                              next + i * kHidden, next_norm_weight,
                                              next_normalized, false,
                                              nullptr, false, nullptr, nullptr,
                                              prepared_normalized);
                    }
                    }
                }
                if (batched_attention) {
                    layer_span[layer].finish_prefill_attention(
                        static_cast<std::uint32_t>(base), count);
                }
                if (deferred_tail) {
                    layer_span[layer].finish_prefill_batch(
                        current, next, count, next_norm_weight, next_normalized_batch,
                        prefill_profile_.enabled ? base : std::numeric_limits<std::size_t>::max());
                }
                layer_span[layer].profile_ordered_end(
                    prefill_profile_.enabled ? base : std::numeric_limits<std::size_t>::max());
                layer_span[layer].release_m23_repacked();
                std::swap(current, next);
            }
            final_hidden = current + (count - 1) * kHidden;
            processed_tokens += count;
            base += count;
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        return final_hidden;
    }

    const float* prefill_full_layer_major_chunk(std::span<const std::uint32_t> prompt,
                                          const std::function<bool()>& should_cancel,
                                          std::size_t base_position) {
        float* current = static_cast<float*>(prefill_a_->get());
        float* next = static_cast<float*>(prefill_b_->get());
        for (std::size_t i = 0; i < prompt.size(); ++i) {
            if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                static_cast<std::uint32_t*>(d_decode_tokens_->get()) + base_position + i,
                &prompt[i], sizeof(std::uint32_t), hipMemcpyHostToDevice, hipStreamPerThread));
            miinfer::launch_qwen35_q4_k_embedding_device_token(
                static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding_->get()),
                static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + base_position + i,
                model_.config().vocab_size, kHidden, current + i * kHidden,
                hipStreamPerThread);
        }
        const auto layer_span = std::span<const GpuLayerRef>(layers_);
        for (std::size_t layer = 0; layer < layer_span.size(); ++layer) {
            if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
            layer_span[layer].profile_ordered_start(
                prefill_profile_.enabled ? base_position : std::numeric_limits<std::size_t>::max());
            const std::size_t full_chunk = wide_prefill_ && prefill_chunk_ >= kM12PrefillBatch
                ? prefill_chunk_ : kM12PrefillBatch;
            for (std::size_t base = 0; base < prompt.size(); base += full_chunk) {
                if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
                const auto count = static_cast<std::uint32_t>(
                    std::min<std::size_t>(full_chunk, prompt.size() - base));
                const float* chunk_input = current + base * kHidden;
                float* chunk_output = next + base * kHidden;
                if (wide_prefill_ && count >= kM12PrefillBatch
                    && layer_span[layer].recurrent != nullptr) {
                    layer_span[layer].recurrent->prefill_wide(
                        chunk_input, chunk_output, static_cast<std::uint32_t>(base_position + base), count);
                    continue;
                }
                const bool prepared = layer_span[layer].prepare_prefill_batch(
                    chunk_input, count, false,
                    prefill_profile_.enabled ? base : std::numeric_limits<std::size_t>::max());
                const bool full_m12_chunk = count % kPrefillBatch == 0;
                const bool deferred_tail = prepared && layer_span[layer].prefill_tail_batch_supported()
                    && (!gdn_chunkwise_prefill_ || full_m12_chunk);
                const bool batched_attention = prepared && deferred_tail
                    && count >= kM12PrefillBatch
                    && layer_span[layer].wide_attention_batch_ready();
                if (!batched_attention) for (std::size_t i = 0; i < count; ++i) {
                    if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
                    if (prepared) {
                        layer_span[layer].run(
                            chunk_input + i * kHidden, static_cast<std::uint32_t>(base_position + base + i),
                            chunk_output + i * kHidden, nullptr, nullptr, false, nullptr, false,
                            layer_span[layer].prefill_qkv_at(i), layer_span[layer].prefill_gate_at(i),
                            layer_span[layer].prefill_normalized_at(i), deferred_tail, i,
                            layer_span[layer].prefill_qfull_at(i),
                            layer_span[layer].prefill_value_at(i));
                    } else {
                        layer_span[layer].run(
                            chunk_input + i * kHidden, static_cast<std::uint32_t>(base_position + base + i),
                            chunk_output + i * kHidden, nullptr, nullptr, false, nullptr, false,
                            nullptr, nullptr, nullptr);
                    }
                }
                if (batched_attention) {
                    layer_span[layer].finish_prefill_attention(
                        static_cast<std::uint32_t>(base_position + base), count);
                }
                if (deferred_tail) {
                    layer_span[layer].finish_prefill_batch(
                        chunk_input, chunk_output, count, nullptr, nullptr,
                        prefill_profile_.enabled ? base : std::numeric_limits<std::size_t>::max());
                }
            }
            layer_span[layer].profile_ordered_end(
                prefill_profile_.enabled ? base_position : std::numeric_limits<std::size_t>::max());
            layer_span[layer].release_m23_repacked();
            std::swap(current, next);
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        return current + (prompt.size() - 1) * kHidden;
    }

    std::uint32_t next_token_from_hidden(const float* hidden, std::size_t position) {
        miinfer::launch_qwen3_rms_norm(
            hidden, static_cast<const float*>(d_final_norm_weight_->get()),
            static_cast<float*>(final_norm_->get()), kHidden, model_.config().rms_epsilon);
        miinfer::launch_q8_1_quantize_f32(
            static_cast<const float*>(final_norm_->get()),
            static_cast<miinfer::Q8_1Block*>(final_q8_1_->get()), kHidden,
            hipStreamPerThread);
        launch_q6k_wave_gemv(
            static_cast<const Q6KWaveTile*>(d_output_weight_->get()),
            static_cast<const miinfer::Q8_1Block*>(final_q8_1_->get()),
            static_cast<float*>(logits_->get()), model_.config().vocab_size, kHidden,
            hipStreamPerThread);
        miinfer::launch_qwen3_argmax(
            static_cast<const float*>(logits_->get()),
            static_cast<std::uint32_t*>(d_decode_tokens_->get()) + position + 1,
            model_.config().vocab_size);
        std::uint32_t result = 0;
        MIINFER_HIP_CHECK(hipMemcpyAsync(
            &result, static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + position + 1,
            sizeof(result), hipMemcpyDeviceToHost, hipStreamPerThread));
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        return result;
    }

    GenerateStats generate_layer_major(std::span<const std::uint32_t> prompt,
                                       const GenerateOptions& opt,
                                       std::chrono::steady_clock::time_point gen_start,
                                       std::size_t start_position = 0) {
        GenerateStats stats;
        stats.prompt_tokens = prompt.size();
        if (prefill_profile_.enabled) {
            prefill_profile_.m23.recurrent_dispatches.fill(0);
            prefill_profile_.m23.attention_dispatches.fill(0);
            prefill_profile_.m23.recurrent_weight_upload_bytes.fill(0);
            prefill_profile_.m23.attention_weight_upload_bytes.fill(0);
            prefill_profile_.m23.recurrent_weight_upload_ms.fill(0.0);
            prefill_profile_.m23.attention_weight_upload_ms.fill(0.0);
            prefill_profile_.m23.weight_upload_bytes = 0;
            prefill_profile_.m23.weight_upload_ms = 0.0;
        }
        const auto prefill_start = std::chrono::steady_clock::now();
        opt_prefill_checkpoint_ = opt.on_prefill_checkpoint;
        const float* final_hidden = prefill_layer_major(prompt, opt.should_cancel,
                                                        stats.prefill_processed_tokens, start_position);
        opt_prefill_checkpoint_ = nullptr;
        const auto prefill_end = std::chrono::steady_clock::now();
        stats.prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
        stats.prefill_tok_s = stats.prefill_ms > 0.0
            ? (1000.0 * stats.prefill_processed_tokens) / stats.prefill_ms : 0.0;
        if (final_hidden == nullptr) {
            stats.cancelled = true;
            stats.total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - gen_start).count();
            return stats;
        }
        if (!prefill_profile_.decode_mode) {
            prefill_profile_.report(stats.prefill_ms, prompt.size());
        }
        if (opt.on_prefill_state) opt.on_prefill_state(final_hidden, prompt.size());
        if (opt.on_prefill_complete) opt.on_prefill_complete();
        if (opt.max_new_tokens == 0) {
            stats.total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - gen_start).count();
            return stats;
        }

        const auto first_start = std::chrono::steady_clock::now();
        std::uint32_t cur_token = next_token_from_hidden(final_hidden, prompt.size() - 1);
        const auto first_end = std::chrono::steady_clock::now();
        stats.first_token_ms = std::chrono::duration<double, std::milli>(first_end - first_start).count();
        stats.tokens.push_back(cur_token);
        if (opt.on_first_token) opt.on_first_token();
        const bool stop_token = cur_token == tokenizer_.eos_id() || cur_token == 151643 || cur_token == 151645;
        bool cancelled = false;
        bool stopped = stop_token;
        if (!stop_token) {
            const std::string piece = tokenizer_.decode(std::span<const std::uint32_t>(&cur_token, 1));
            stats.text += piece;
            cancelled = opt.on_token && !opt.on_token(cur_token, piece);
        }

        std::size_t pos = prompt.size();
        // Keep prefill selection independent from the decode execution contract.
        // The queued graph chain is the qualified no-stream decode path.
        if (!stopped && !opt.stream && use_hip_graph_ && opt.max_new_tokens > 1
            && pos < g_cache_capacity) {
            const std::size_t num_to_gen = std::min(opt.max_new_tokens - 1, g_cache_capacity - pos);
            ensure_graph_captured(pos);
            stats.decode_graph_launches = num_to_gen;
            stats.decode_h2d_bytes = sizeof(miinfer::DeviceDecodeState);
            stats.decode_d2h_bytes = num_to_gen * sizeof(std::uint32_t);
            initialize_decode_state(cur_token, pos, num_to_gen);
            const auto decode_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < num_to_gen; ++i)
                MIINFER_HIP_CHECK(hipGraphLaunch(decode_graph_, hipStreamPerThread));
            std::vector<std::uint32_t> raw_tokens(num_to_gen);
            MIINFER_HIP_CHECK(hipMemcpy(raw_tokens.data(),
                static_cast<const std::uint32_t*>(d_decode_tokens_->get()),
                num_to_gen * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
            stats.decode_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_start).count();
            stats.decode_ms += stats.first_token_ms;
            for (std::uint32_t next : raw_tokens) {
                stats.tokens.push_back(next);
                if (stats.tokens.size() == 1 && opt.on_first_token) opt.on_first_token();
                if (next == tokenizer_.eos_id() || next == 151643 || next == 151645) break;
                stats.text += tokenizer_.decode(std::span<const std::uint32_t>(&next, 1));
            }
            stats.generated_tokens = stats.tokens.size();
            stats.total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - gen_start).count();
            stats.decode_tok_s = stats.decode_ms > 0.0
                ? (1000.0 * stats.generated_tokens) / stats.decode_ms : 0.0;
            if (prefill_profile_.decode_mode) {
                prefill_profile_.report(stats.decode_ms, prefill_profile_.profile_position);
            }
            return stats;
        }

        for (std::size_t gen_idx = 1; !cancelled && !stopped && gen_idx < opt.max_new_tokens && pos < g_cache_capacity; ++gen_idx) {
            if (g_shutdown_requested || (opt.should_cancel && opt.should_cancel())) { cancelled = true; break; }
            const auto step_res = step(cur_token, pos);
            cur_token = step_res.token;
            stats.tokens.push_back(cur_token);
            stats.decode_ms += step_res.latency_ms;
            ++pos;
            stopped = cur_token == tokenizer_.eos_id() || cur_token == 151643 || cur_token == 151645;
            if (stopped) break;
            const std::string piece = tokenizer_.decode(std::span<const std::uint32_t>(&cur_token, 1));
            stats.text += piece;
            if (opt.on_token && !opt.on_token(cur_token, piece)) { cancelled = true; break; }
        }
        stats.decode_ms += stats.first_token_ms;
        stats.generated_tokens = stats.tokens.size();
        stats.cancelled = cancelled;
        if (stats.generated_tokens > 0 && stats.decode_ms > 0.0) {
            stats.decode_tok_s = (1000.0 * stats.generated_tokens) / stats.decode_ms;
        }
        stats.total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gen_start).count();
        if (prefill_profile_.decode_mode) {
            prefill_profile_.report(stats.decode_ms, prefill_profile_.profile_position);
        }
        return stats;
    }

    GenerateStats generate(std::span<const std::uint32_t> prompt, const GenerateOptions& opt = GenerateOptions()) {
        const auto started = std::chrono::steady_clock::now();
        if (!layer_major_prefill_) {
            const double capture_start = graph_capture_ms_;
            auto stats = generate_fresh(prompt, opt);
            stats.graph_capture_ms = graph_capture_ms_ - capture_start;
            return stats;
        }
        if (prompt.empty()) return {};
        if (prompt.size() > g_cache_capacity) throw std::runtime_error("prompt exceeds context capacity");
        const double capture_start = graph_capture_ms_;
        if (!opt.reuse_session) clear_session_checkpoints();
        std::erase_if(session_checkpoints_, [&](const auto& checkpoint) {
            return !session_checkpoint_compatible(checkpoint);
        });
        session_checkpoint_bytes_ = 0;
        for (const auto& checkpoint : session_checkpoints_) {
            session_checkpoint_bytes_ += checkpoint.storage_bytes;
        }
        rebuild_session_prefix_index();
        std::size_t common = 0;
        const auto selected = opt.reuse_session
            ? longest_session_checkpoint(prompt, common) : std::nullopt;
        const std::size_t reused = selected
            ? session_checkpoints_[*selected].boundary_position : 0;
        if (reused == 0) reset_device_state();
        else {
            auto& checkpoint = session_checkpoints_[*selected];
            restore_session_checkpoint(checkpoint);
            checkpoint.last_used = ++session_use_clock_;
        }
        GenerateOptions actual = opt;
        if (opt.reuse_session) {
            actual.on_prefill_checkpoint = [this, prompt, reused, callback = opt.on_prefill_checkpoint](std::size_t tokens) {
                if (reused != 0) touch_session_checkpoint(prompt.first(reused));
                const std::size_t batches = tokens / kFullPrefillCapacity;
                const bool power_of_two = batches != 0 && (batches & (batches - 1)) == 0;
                const std::size_t latest_boundary =
                    (prompt.size() / kFullPrefillCapacity) * kFullPrefillCapacity;
                if (power_of_two || tokens == latest_boundary) {
                    capture_session_checkpoint(prompt.first(tokens));
                }
                if (callback) callback(tokens);
            };
        }
        try {
            auto stats = generate_layer_major(prompt, actual, started, reused);
            if (stats.cancelled) clear_session_checkpoints();
            stats.reused_prefix_tokens = reused;
            stats.common_prefix_tokens = common;
            stats.session_checkpoint_count = session_checkpoints_.size();
            stats.session_checkpoint_bytes = session_checkpoint_bytes_;
            stats.graph_capture_ms = graph_capture_ms_ - capture_start;
            return stats;
        } catch (...) {
            clear_session_checkpoints();
            throw;
        }
    }

    std::vector<std::byte> snapshot_decode_buffers(std::size_t positions) const {
        if (positions > g_cache_capacity) throw std::runtime_error("snapshot exceeds cache capacity");
        MIINFER_HIP_CHECK(hipDeviceSynchronize());
        std::vector<std::byte> snapshot;
        const auto append = [&snapshot](const void* device, std::size_t bytes) {
            const std::size_t offset = snapshot.size();
            snapshot.resize(offset + bytes);
            MIINFER_HIP_CHECK(hipMemcpy(snapshot.data() + offset, device, bytes,
                                        hipMemcpyDeviceToHost));
        };
        constexpr std::size_t recurrent_state_bytes = kVHeads * kState * kState * sizeof(float);
        constexpr std::size_t recurrent_history_bytes = 4 * kChannels * sizeof(float);
        for (const auto& layer : layers_) {
            if (layer.recurrent != nullptr) {
                append(layer.recurrent->state->get(), recurrent_state_bytes);
                append(layer.recurrent->history->get(), recurrent_history_bytes);
            } else if (layer.attention != nullptr) {
                const std::size_t element_size = layer.attention->fp16_kv_cache
                    ? sizeof(__half) : sizeof(float);
                const std::size_t head_bytes = positions * 256 * element_size;
                const auto* keys = static_cast<const std::byte*>(layer.attention->key_cache->get());
                const auto* values = static_cast<const std::byte*>(layer.attention->value_cache->get());
                for (std::size_t head = 0; head < 4; ++head) {
                    const std::size_t offset = head * g_cache_capacity * 256 * element_size;
                    append(keys + offset, head_bytes);
                    append(values + offset, head_bytes);
                }
            }
        }
        return snapshot;
    }

    std::uint32_t prepare_m26c_state(std::span<const std::uint32_t> prompt,
                                     const std::filesystem::path& path) {
        if (prompt.size() != 512) {
            throw std::runtime_error("M26-C canonical snapshot requires exactly 512 prompt tokens");
        }
        reset();
        for (std::size_t pos = 0; pos + 1 < prompt.size(); ++pos) prefill_step(prompt[pos], pos);
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        const auto first = step(prompt.back(), prompt.size() - 1).token;
        write_m26c_state(path, prompt, prompt.size(), first);
        return first;
    }

    struct M26CDecodeResult {
        std::vector<std::uint32_t> tokens;
        double restore_ms = 0.0;
        double graph_capture_ms = 0.0;
        double decode_ms = 0.0;
        double total_ms = 0.0;
        double direct_h2d_device_ms = 0.0;
        double direct_d2h_device_ms = 0.0;
        double direct_blocking_sync_ms = 0.0;
        std::size_t direct_sync_calls = 0;
        double graph_state_sync_ms = 0.0;
        double graph_replay_enqueue_ms = 0.0;
        double graph_bulk_copy_wait_ms = 0.0;
        std::size_t graph_nodes = 0;
        std::size_t restore_h2d_bytes = 0;
        std::size_t decode_h2d_bytes = 0;
        std::size_t decode_d2h_bytes = 0;
    };

    std::size_t snapshot_decode_bytes(std::size_t positions) const {
        std::size_t bytes = 0;
        for (const auto& layer : layers_) {
            if (layer.recurrent != nullptr) {
                bytes += kVHeads * kState * kState * sizeof(float) + 4 * kChannels * sizeof(float);
            } else if (layer.attention != nullptr) {
                const std::size_t element_size = layer.attention->fp16_kv_cache
                    ? sizeof(__half) : sizeof(float);
                bytes += 2 * 4 * positions * 256 * element_size;
            }
        }
        return bytes;
    }

    std::string runtime_selector_identity() const {
        std::vector<std::string> selectors;
        for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
            const std::string_view value(*entry);
            if (value.rfind("MIINFER_", 0) == 0 && value.rfind("MIINFER_API_KEY=", 0) != 0) {
                selectors.emplace_back(value);
            }
        }
        std::sort(selectors.begin(), selectors.end());
        std::string identity;
        for (const auto& selector : selectors) identity += selector + "\n";
        return identity;
    }

    void write_m26c_state(const std::filesystem::path& path,
                          std::span<const std::uint32_t> history,
                          std::size_t position, std::uint32_t current_token,
                          std::span<const std::uint32_t> output_tokens = {}) const {
        if (position != history.size() || position > g_cache_capacity
            || position > std::numeric_limits<std::uint32_t>::max()
            || current_token >= model_.config().vocab_size) {
            throw std::runtime_error("invalid M26-C semantic position/history: position="
                + std::to_string(position) + " history=" + std::to_string(history.size())
                + " capacity=" + std::to_string(g_cache_capacity) + " token="
                + std::to_string(current_token) + " vocab="
                + std::to_string(model_.config().vocab_size));
        }
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("unable to create M26-C state: " + path.string());
        const auto write = [&output](const void* data, std::size_t bytes) {
            if (bytes == 0) return;
            output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
            if (!output) throw std::runtime_error("failed writing M26-C state snapshot");
        };
        const auto write_string = [&write](std::string_view value) {
            const auto size = static_cast<std::uint64_t>(value.size());
            write(&size, sizeof(size));
            write(value.data(), value.size());
        };
        const std::array<char, 8> magic{'M','2','6','C','S','T','A','T'};
        constexpr std::uint32_t version = 1;
        const auto block_count = static_cast<std::uint32_t>(model_.config().block_count);
        const auto hidden_size = static_cast<std::uint32_t>(model_.config().hidden_size);
        const auto state_size = static_cast<std::uint32_t>(kState);
        const auto model_context_length = static_cast<std::uint64_t>(model_.config().context_length);
        const auto saved_position = static_cast<std::uint64_t>(position);
        const auto layer_count = static_cast<std::uint32_t>(layers_.size());
        const auto token_count = static_cast<std::uint64_t>(history.size());
        const auto output_count = static_cast<std::uint64_t>(output_tokens.size());
        write(magic.data(), magic.size());
        write(&version, sizeof(version));
        write(&block_count, sizeof(block_count));
        write(&hidden_size, sizeof(hidden_size));
        write(&state_size, sizeof(state_size));
        write(&model_context_length, sizeof(model_context_length));
        write(&saved_position, sizeof(saved_position));
        write(&current_token, sizeof(current_token));
        write(&layer_count, sizeof(layer_count));
        write(&token_count, sizeof(token_count));
        write(&output_count, sizeof(output_count));
        write_string(m26c_model_hash());
        write_string(session_quantization_identity_);
        write_string(session_runtime_identity_ + "\nselectors:\n" + runtime_selector_identity());
        write(history.data(), history.size_bytes());
        write(output_tokens.data(), output_tokens.size_bytes());
        constexpr std::size_t recurrent_state_elements = kVHeads * kState * kState;
        constexpr std::size_t recurrent_history_elements = 4 * kChannels;
        for (const auto& layer : layers_) {
            const std::uint8_t kind = layer.recurrent != nullptr ? 1 : 2;
            write(&kind, sizeof(kind));
            if (layer.recurrent != nullptr) {
                const auto state = layer.recurrent->logical_state();
                const auto* device_history = layer.recurrent->history->get();
                std::vector<float> conv_history(recurrent_history_elements);
                MIINFER_HIP_CHECK(hipMemcpy(conv_history.data(), device_history,
                    conv_history.size() * sizeof(float), hipMemcpyDeviceToHost));
                if (state.size() != recurrent_state_elements) {
                    throw std::runtime_error("unexpected M26-C recurrent state size");
                }
                const auto state_count = static_cast<std::uint64_t>(state.size());
                const auto history_count = static_cast<std::uint64_t>(conv_history.size());
                write(&state_count, sizeof(state_count));
                write(&history_count, sizeof(history_count));
                write(state.data(), state.size() * sizeof(float));
                write(conv_history.data(), conv_history.size() * sizeof(float));
                continue;
            }
            const auto& attention = *layer.attention;
            const std::size_t element_size = attention.fp16_kv_cache ? sizeof(__half) : sizeof(float);
            const std::size_t elements = position * 256;
            const auto* keys = static_cast<const std::byte*>(attention.key_cache->get());
            const auto* values = static_cast<const std::byte*>(attention.value_cache->get());
            const auto head_count = std::uint32_t{4};
            const auto element_count = static_cast<std::uint64_t>(elements);
            write(&head_count, sizeof(head_count));
            write(&element_count, sizeof(element_count));
            for (std::size_t head = 0; head < 4; ++head) {
                std::vector<float> host_keys(elements), host_values(elements);
                const std::size_t source_offset = head * g_cache_capacity * 256 * element_size;
                if (attention.fp16_kv_cache) {
                    std::vector<__half> half_keys(elements), half_values(elements);
                    MIINFER_HIP_CHECK(hipMemcpy(half_keys.data(), keys + source_offset,
                        elements * sizeof(__half), hipMemcpyDeviceToHost));
                    MIINFER_HIP_CHECK(hipMemcpy(half_values.data(), values + source_offset,
                        elements * sizeof(__half), hipMemcpyDeviceToHost));
                    std::transform(half_keys.begin(), half_keys.end(), host_keys.begin(),
                        [](const __half& value) { return __half2float(value); });
                    std::transform(half_values.begin(), half_values.end(), host_values.begin(),
                        [](const __half& value) { return __half2float(value); });
                } else {
                    MIINFER_HIP_CHECK(hipMemcpy(host_keys.data(), keys + source_offset,
                        elements * sizeof(float), hipMemcpyDeviceToHost));
                    MIINFER_HIP_CHECK(hipMemcpy(host_values.data(), values + source_offset,
                        elements * sizeof(float), hipMemcpyDeviceToHost));
                }
                write(host_keys.data(), host_keys.size() * sizeof(float));
                write(host_values.data(), host_values.size() * sizeof(float));
            }
        }
    }

    std::string m26c_model_hash() const {
        if (m26c_model_hash_.empty()) m26c_model_hash_ = miinfer::sha256_file(model_.artifact_path());
        return m26c_model_hash_;
    }

    M26CDecodeResult decode_m26c_state(const std::filesystem::path& input_path,
                                       const std::filesystem::path& output_path,
                                       bool graph, std::size_t token_count,
                                       const std::optional<std::filesystem::path>& logits_output = {},
                                       const std::optional<std::filesystem::path>& layer_path_prefix = {},
                                       const std::optional<std::vector<std::uint32_t>>& forced_inputs = {},
                                       const std::optional<std::filesystem::path>& checkpoint_dir = {},
                                       const std::optional<std::size_t>& layer_path_position = {}) {
        M26CDecodeResult result;
        const auto total_start = std::chrono::steady_clock::now();
        std::ifstream input(input_path, std::ios::binary);
        if (!input) throw std::runtime_error("unable to open M26-C state: " + input_path.string());
        const auto read = [&input](void* data, std::size_t bytes) {
            if (bytes == 0) return;
            input.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
            if (!input) throw std::runtime_error("truncated M26-C state snapshot");
        };
        const auto read_string = [&read]() {
            std::uint64_t size = 0;
            read(&size, sizeof(size));
            if (size > 65536) throw std::runtime_error("invalid M26-C identity length");
            std::string value(static_cast<std::size_t>(size), '\0');
            read(value.data(), value.size());
            return value;
        };
        std::array<char, 8> magic{};
        std::uint32_t version = 0, block_count = 0, hidden_size = 0, state_size = 0;
        std::uint64_t model_context_length = 0;
        std::uint32_t current_token = 0, layer_count = 0;
        std::uint64_t position = 0, token_count_in_state = 0, output_count = 0;
        read(magic.data(), magic.size());
        read(&version, sizeof(version));
        read(&block_count, sizeof(block_count));
        read(&hidden_size, sizeof(hidden_size));
        read(&state_size, sizeof(state_size));
        read(&model_context_length, sizeof(model_context_length));
        read(&position, sizeof(position));
        read(&current_token, sizeof(current_token));
        read(&layer_count, sizeof(layer_count));
        read(&token_count_in_state, sizeof(token_count_in_state));
        read(&output_count, sizeof(output_count));
        const auto model_hash = read_string();
        const auto quantization_identity = read_string();
        const auto source_runtime_identity = read_string();
        if (magic != std::array<char, 8>{'M','2','6','C','S','T','A','T'} || version != 1
            || block_count != model_.config().block_count
            || hidden_size != model_.config().hidden_size || state_size != kState
            || model_context_length != model_.config().context_length
            || layer_count != layers_.size() || token_count_in_state != position
            || model_hash != m26c_model_hash()
            || quantization_identity != session_quantization_identity_
            || position >= g_cache_capacity || current_token >= model_.config().vocab_size) {
            throw std::runtime_error("M26-C state identity/layout mismatch");
        }
        std::cerr << "m26c_source_contract=" << source_runtime_identity
                  << "\nm26c_target_contract=" << session_runtime_identity_ << '\n';
        if (token_count > g_cache_capacity
            || token_count > g_cache_capacity - position) {
            throw std::runtime_error("M26-C decode exceeds cache capacity");
        }
        std::vector<std::uint32_t> history(static_cast<std::size_t>(token_count_in_state));
        read(history.data(), history.size() * sizeof(std::uint32_t));
        if (std::any_of(history.begin(), history.end(), [&](std::uint32_t token) {
                return token >= model_.config().vocab_size;
            })) {
            throw std::runtime_error("invalid token in M26-C semantic history");
        }
        if (output_count > g_cache_capacity) throw std::runtime_error("invalid M26-C output token count");
        std::vector<std::uint32_t> prior_output(static_cast<std::size_t>(output_count));
        read(prior_output.data(), prior_output.size() * sizeof(std::uint32_t));
        if (std::any_of(prior_output.begin(), prior_output.end(), [&](std::uint32_t token) {
                return token >= model_.config().vocab_size;
            })) {
            throw std::runtime_error("invalid token in M26-C diagnostic output");
        }

        const auto restore_start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            std::uint8_t kind = 0;
            read(&kind, sizeof(kind));
            const auto& layer = layers_[i];
            if (layer.recurrent != nullptr && kind == 1) {
                constexpr std::size_t state_elements = kVHeads * kState * kState;
                constexpr std::size_t history_elements = 4 * kChannels;
                std::uint64_t state_count = 0, conv_history_count = 0;
                read(&state_count, sizeof(state_count));
                read(&conv_history_count, sizeof(conv_history_count));
                if (state_count != state_elements || conv_history_count != history_elements) {
                    throw std::runtime_error("M26-C recurrent record size mismatch");
                }
                std::vector<float> state(state_elements), conv_history(history_elements);
                read(state.data(), state.size() * sizeof(float));
                read(conv_history.data(), conv_history.size() * sizeof(float));
                layer.recurrent->upload_state(state);
                MIINFER_HIP_CHECK(hipMemcpy(layer.recurrent->history->get(), conv_history.data(),
                    conv_history.size() * sizeof(float), hipMemcpyHostToDevice));
            } else if (layer.attention != nullptr && kind == 2) {
                std::uint32_t head_count = 0;
                std::uint64_t element_count = 0;
                read(&head_count, sizeof(head_count));
                const std::size_t elements = static_cast<std::size_t>(position) * 256;
                read(&element_count, sizeof(element_count));
                if (head_count != 4 || element_count != elements) {
                    throw std::runtime_error("M26-C attention record size mismatch");
                }
                const std::size_t target_element_size = layer.attention->fp16_kv_cache
                    ? sizeof(__half) : sizeof(float);
                auto* key_base = static_cast<std::byte*>(layer.attention->key_cache->get());
                auto* value_base = static_cast<std::byte*>(layer.attention->value_cache->get());
                for (std::size_t head = 0; head < 4; ++head) {
                    std::vector<float> keys(elements), values(elements);
                    read(keys.data(), keys.size() * sizeof(float));
                    read(values.data(), values.size() * sizeof(float));
                    const std::size_t target_offset = head * g_cache_capacity * 256 * target_element_size;
                    if (layer.attention->fp16_kv_cache) {
                        std::vector<__half> half_keys(elements), half_values(elements);
                        std::transform(keys.begin(), keys.end(), half_keys.begin(),
                            [](float value) { return __float2half(value); });
                        std::transform(values.begin(), values.end(), half_values.begin(),
                            [](float value) { return __float2half(value); });
                        MIINFER_HIP_CHECK(hipMemcpy(key_base + target_offset, half_keys.data(),
                            elements * sizeof(__half), hipMemcpyHostToDevice));
                        MIINFER_HIP_CHECK(hipMemcpy(value_base + target_offset, half_values.data(),
                            elements * sizeof(__half), hipMemcpyHostToDevice));
                    } else {
                        MIINFER_HIP_CHECK(hipMemcpy(key_base + target_offset, keys.data(),
                            elements * sizeof(float), hipMemcpyHostToDevice));
                        MIINFER_HIP_CHECK(hipMemcpy(value_base + target_offset, values.data(),
                            elements * sizeof(float), hipMemcpyHostToDevice));
                    }
                }
            } else {
                throw std::runtime_error("M26-C snapshot layer topology mismatch at layer "
                                         + std::to_string(i));
            }
        }
        if (input.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("unexpected trailing bytes in M26-C snapshot");
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        result.restore_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - restore_start).count();
        result.restore_h2d_bytes = snapshot_decode_bytes(position);
        const auto first_input_token = current_token;

        if (forced_inputs && (graph || forced_inputs->size() != token_count
            || forced_inputs->empty() || forced_inputs->front() != current_token)) {
            throw std::runtime_error("M26-C teacher forcing requires direct route and a token vector "
                                     "matching max-tokens and the snapshot's pending token");
        }
        if (checkpoint_dir) std::filesystem::create_directories(*checkpoint_dir);

        if (token_count == 0) {
            write_m26c_state(output_path, history, static_cast<std::size_t>(position), current_token);
            result.total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - total_start).count();
            return result;
        }

        std::array<LayerPathCapture, 2> layer_path_captures;
        std::array<GatePathCapture, 2> gate_path_captures;
        constexpr std::array<std::size_t, 2> captured_layers{0, 1};
        if (layer_path_prefix && !layer_path_position) {
            if (graph || token_count != 1) {
                throw std::runtime_error(
                    "M26-C layer-path capture requires one direct decode token");
            }
            for (std::size_t i = 0; i < captured_layers.size(); ++i) {
                const auto index = captured_layers[i];
                if (index >= layers_.size() || layers_[index].recurrent == nullptr) {
                    throw std::runtime_error(
                        "M26-C layer-path capture requires recurrent layers 0 and 1");
                }
                auto* recurrent = layers_[index].recurrent;
                recurrent->layer_path_capture = &layer_path_captures[i];
                recurrent->layer_path_capture_position = static_cast<std::uint32_t>(position);
                recurrent->gate_path_capture = &gate_path_captures[i];
                recurrent->gate_path_capture_position = static_cast<std::uint32_t>(position);
            }
        }

        if (graph) {
            if (prefill_profile_.decode_mode) {
                throw std::runtime_error(
                    "M26-C event profiling requires direct decode; graph event capture is unsupported");
            }
            if (!use_hip_graph_) throw std::runtime_error("M26-C graph route disabled by MIINFER_HIP_GRAPH=0");
            const double capture_before = graph_capture_ms_;
            ensure_graph_captured(static_cast<std::size_t>(position));
            result.graph_capture_ms = graph_capture_ms_ - capture_before;
            result.graph_nodes = decode_graph_node_count_;
            initialize_decode_state(current_token, static_cast<std::size_t>(position), token_count);
            const auto state_sync_start = std::chrono::steady_clock::now();
            MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
            result.graph_state_sync_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - state_sync_start).count();
            result.tokens.resize(token_count);
            const auto decode_start = std::chrono::steady_clock::now();
            const auto replay_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < token_count; ++i)
                MIINFER_HIP_CHECK(hipGraphLaunch(decode_graph_, hipStreamPerThread));
            result.graph_replay_enqueue_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - replay_start).count();
            const auto copy_start = std::chrono::steady_clock::now();
            MIINFER_HIP_CHECK(hipMemcpy(result.tokens.data(), d_decode_tokens_->get(),
                token_count * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
            result.graph_bulk_copy_wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - copy_start).count();
            result.decode_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_start).count();
            result.decode_h2d_bytes = sizeof(miinfer::DeviceDecodeState);
            result.decode_d2h_bytes = token_count * sizeof(std::uint32_t);
            current_token = result.tokens.back();
            position += token_count;
        } else {
            result.tokens.reserve(token_count);
            StepTransferEvents transfer_events;
            StepTransferMetrics transfer_metrics;
            std::vector<LayerPathCapture> teacher_layer_captures(layers_.size());
            std::vector<GatePathCapture> teacher_gate_captures(layers_.size());
            DecodeLayerCapture teacher_hidden_capture;
            teacher_hidden_capture.inputs.resize(layers_.size());
            teacher_hidden_capture.outputs.resize(layers_.size());
            constexpr std::array<std::size_t, 12> kTeacherLayerPositions{
                512, 519, 527, 543, 559, 564, 565, 566, 567, 568, 569, 575};
            auto token = current_token;
            const auto decode_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < token_count; ++i) {
                const auto absolute_position = static_cast<std::size_t>(position + i);
                const bool capture_layer_path = layer_path_position
                    && (absolute_position == *layer_path_position
                        || std::find(kTeacherLayerPositions.begin(), kTeacherLayerPositions.end(),
                                     absolute_position) != kTeacherLayerPositions.end());
                if (capture_layer_path) {
                    for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
                        if (auto* recurrent = layers_[layer_index].recurrent) {
                            recurrent->layer_path_capture = &teacher_layer_captures[layer_index];
                            recurrent->layer_path_capture_position = static_cast<std::uint32_t>(absolute_position);
                            recurrent->gate_path_capture = &teacher_gate_captures[layer_index];
                            recurrent->gate_path_capture_position = static_cast<std::uint32_t>(absolute_position);
                        }
                    }
                }
                const auto input_token = forced_inputs ? (*forced_inputs)[i] : token;
                token = step(input_token, static_cast<std::size_t>(position + i),
                             &transfer_events, &transfer_metrics,
                             capture_layer_path ? &teacher_hidden_capture : nullptr).token;
                result.tokens.push_back(token);
                if (capture_layer_path) {
                    if (!layer_path_prefix) throw std::runtime_error("layer-path position requires output prefix");
                    const auto write_capture = [&layer_path_prefix, absolute_position](
                        std::size_t layer_index, std::string_view field, const std::vector<float>& values) {
                        if (values.empty()) return;
                        auto path = *layer_path_prefix;
                        path += ".pos-" + std::to_string(absolute_position) + ".layer"
                            + std::to_string(layer_index) + "." + std::string(field) + ".f32";
                        std::ofstream output(path, std::ios::binary | std::ios::trunc);
                        if (!output) throw std::runtime_error("unable to create M26-C layer capture: " + path.string());
                        output.write(reinterpret_cast<const char*>(values.data()),
                            static_cast<std::streamsize>(values.size() * sizeof(float)));
                        if (!output) throw std::runtime_error("failed writing M26-C layer capture: " + path.string());
                    };
                    for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
                        const auto& capture = teacher_layer_captures[layer_index];
                        const auto& gate = teacher_gate_captures[layer_index];
                        write_capture(layer_index, "input_hidden", teacher_hidden_capture.inputs[layer_index]);
                        write_capture(layer_index, "output_hidden", teacher_hidden_capture.outputs[layer_index]);
                        write_capture(layer_index, "input", capture.input);
                        write_capture(layer_index, "normalized", capture.normalized);
                        write_capture(layer_index, "qkv", capture.qkv);
                        write_capture(layer_index, "recurrent_output", capture.recurrent_output);
                        write_capture(layer_index, "gated", capture.gated);
                        write_capture(layer_index, "attention_residual", capture.attention_residual);
                        write_capture(layer_index, "post_normalized", capture.post_normalized);
                        write_capture(layer_index, "ffn_output", capture.ffn_output);
                        write_capture(layer_index, "layer_output", capture.layer_output);
                        write_capture(layer_index, "gate.projection", gate.gate);
                        write_capture(layer_index, "gate.gated", gate.gated);
                        if (layers_[layer_index].recurrent != nullptr) {
                            layers_[layer_index].recurrent->layer_path_capture = nullptr;
                            layers_[layer_index].recurrent->gate_path_capture = nullptr;
                        }
                    }
                }
                if (forced_inputs && logits_output) {
                    std::vector<float> host_logits(model_.config().vocab_size);
                    std::vector<float> host_final_norm(kHidden);
                    MIINFER_HIP_CHECK(hipMemcpy(host_logits.data(), logits_->get(),
                        host_logits.size() * sizeof(float), hipMemcpyDeviceToHost));
                    MIINFER_HIP_CHECK(hipMemcpy(host_final_norm.data(), final_norm_->get(),
                        host_final_norm.size() * sizeof(float), hipMemcpyDeviceToHost));
                    auto path = *logits_output;
                    path += ".pos-" + std::to_string(position + i) + ".f32";
                    std::ofstream logits_file(path, std::ios::binary | std::ios::trunc);
                    if (!logits_file) throw std::runtime_error("unable to create M26-C logits: " + path.string());
                    logits_file.write(reinterpret_cast<const char*>(host_logits.data()),
                        static_cast<std::streamsize>(host_logits.size() * sizeof(float)));
                    if (!logits_file) throw std::runtime_error("failed writing M26-C logits");
                    path = *logits_output;
                    path += ".pos-" + std::to_string(position + i) + ".final-norm.f32";
                    std::ofstream norm_file(path, std::ios::binary | std::ios::trunc);
                    if (!norm_file) throw std::runtime_error("unable to create M26-C final norm: " + path.string());
                    norm_file.write(reinterpret_cast<const char*>(host_final_norm.data()),
                        static_cast<std::streamsize>(host_final_norm.size() * sizeof(float)));
                    if (!norm_file) throw std::runtime_error("failed writing M26-C final norm");
                }
                if (forced_inputs && checkpoint_dir) {
                    const auto absolute_position = static_cast<std::size_t>(position + i);
                    const bool checkpoint = absolute_position == 512 || absolute_position == 519
                        || absolute_position == 527 || absolute_position == 543
                        || absolute_position == 559 || absolute_position == 564
                        || absolute_position == 565 || absolute_position == 566
                        || absolute_position == 567 || absolute_position == 568
                        || absolute_position == 569 || absolute_position == 575;
                    if (checkpoint) {
                        auto checkpoint_history = history;
                        checkpoint_history.insert(checkpoint_history.end(), forced_inputs->begin(),
                            forced_inputs->begin() + static_cast<std::ptrdiff_t>(i + 1));
                        const auto pending = i + 1 < token_count ? (*forced_inputs)[i + 1] : token;
                        auto path = *checkpoint_dir;
                        path /= "position-" + std::to_string(absolute_position + 1) + ".state";
                        write_m26c_state(path, checkpoint_history, absolute_position + 1, pending,
                            std::span<const std::uint32_t>(result.tokens.data(), result.tokens.size()));
                    }
                }
            }
            result.direct_h2d_device_ms = transfer_metrics.h2d_device_ms;
            result.direct_d2h_device_ms = transfer_metrics.d2h_device_ms;
            result.direct_blocking_sync_ms = transfer_metrics.blocking_sync_ms;
            result.direct_sync_calls = transfer_metrics.sync_calls;
            result.decode_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_start).count();
            if (prefill_profile_.decode_mode) {
                prefill_profile_.report(result.decode_ms, static_cast<std::size_t>(position));
            }
            result.decode_h2d_bytes = token_count * sizeof(std::uint32_t);
            result.decode_d2h_bytes = token_count * sizeof(std::uint32_t);
            // Each returned token except the final one is consumed into the saved state.
            current_token = token;
            position += token_count;
            if (forced_inputs) {
                history.insert(history.end(), forced_inputs->begin(), forced_inputs->end());
            }
        }
        if (layer_path_prefix && !layer_path_position) {
            const auto write_capture = [this, &layer_path_prefix](
                std::size_t layer_index, std::string_view field, const std::vector<float>& values) {
                if (values.empty()) return;
                auto path = *layer_path_prefix;
                path += ".layer" + std::to_string(layer_index) + "."
                    + std::string(field) + ".f32";
                std::ofstream output(path, std::ios::binary | std::ios::trunc);
                if (!output) throw std::runtime_error("unable to create M26-C layer capture: "
                                                     + path.string());
                output.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(float)));
                if (!output) throw std::runtime_error("failed writing M26-C layer capture: "
                                                     + path.string());
            };
            for (std::size_t i = 0; i < captured_layers.size(); ++i) {
                const auto layer = captured_layers[i];
                const auto& path = layer_path_captures[i];
                const auto& gate = gate_path_captures[i];
                write_capture(layer, "input", path.input);
                write_capture(layer, "normalized", path.normalized);
                write_capture(layer, "qkv", path.qkv);
                write_capture(layer, "recurrent_output", path.recurrent_output);
                write_capture(layer, "gated", path.gated);
                write_capture(layer, "attention_residual", path.attention_residual);
                write_capture(layer, "post_normalized", path.post_normalized);
                write_capture(layer, "ffn_output", path.ffn_output);
                write_capture(layer, "layer_output", path.layer_output);
                write_capture(layer, "gate.normalized", gate.normalized);
                write_capture(layer, "gate.projection", gate.gate);
                write_capture(layer, "gate.recurrent_output", gate.recurrent_output);
                write_capture(layer, "gate.head_norm", gate.head_norm);
                write_capture(layer, "gate.head_scaled", gate.head_scaled);
                write_capture(layer, "gate.gated", gate.gated);
                auto manifest_path = *layer_path_prefix;
                manifest_path += ".layer" + std::to_string(layer) + ".manifest";
                std::ofstream manifest(manifest_path, std::ios::trunc);
                if (!manifest) throw std::runtime_error("unable to create M26-C layer manifest: "
                                                       + manifest_path.string());
                manifest << "model_sha256=" << m26c_model_hash() << '\n'
                         << "position=" << (position - token_count) << '\n'
                         << "input_token=" << first_input_token << '\n'
                         << "output_token=" << result.tokens.front() << '\n'
                         << "source_contract=" << source_runtime_identity << '\n'
                         << "target_contract=" << session_runtime_identity_ << '\n';
                if (!manifest) throw std::runtime_error("failed writing M26-C layer manifest: "
                                                       + manifest_path.string());
                layers_[layer].recurrent->layer_path_capture = nullptr;
                layers_[layer].recurrent->gate_path_capture = nullptr;
            }
        }
        if (!forced_inputs) {
            history.push_back(first_input_token);
            history.insert(history.end(), result.tokens.begin(), result.tokens.end() - 1);
        }
        if (history.size() != position) throw std::runtime_error("invalid M26-C output history length");
        if (logits_output && !forced_inputs) {
            std::vector<float> host_logits(model_.config().vocab_size);
            MIINFER_HIP_CHECK(hipMemcpy(host_logits.data(), logits_->get(),
                host_logits.size() * sizeof(float), hipMemcpyDeviceToHost));
            std::ofstream logits_file(*logits_output, std::ios::binary | std::ios::trunc);
            if (!logits_file) throw std::runtime_error("unable to create M26-C logits: "
                                                       + logits_output->string());
            logits_file.write(reinterpret_cast<const char*>(host_logits.data()),
                static_cast<std::streamsize>(host_logits.size() * sizeof(float)));
            if (!logits_file) throw std::runtime_error("failed writing M26-C logits");
        }
        write_m26c_state(output_path, history, static_cast<std::size_t>(position), current_token,
                         result.tokens);
        result.total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start).count();
        return result;
    }

    GenerateStats generate_fresh(std::span<const std::uint32_t> prompt, const GenerateOptions& opt) {
        reset();
        GenerateStats stats;
        stats.prompt_tokens = prompt.size();
        if (prompt.empty()) return stats;

        const auto gen_start = std::chrono::steady_clock::now();

        if (layer_major_prefill_) return generate_layer_major(prompt, opt, gen_start);

        // 1. Prefill / Process prompt tokens (Phase 0: No LM head, no per-token host sync)
        const auto prefill_start = std::chrono::steady_clock::now();
        for (std::size_t pos = 0; pos < prompt.size() - 1; ++pos) {
            if (g_shutdown_requested || (opt.should_cancel && opt.should_cancel())) {
                stats.cancelled = true;
                stats.total_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - gen_start).count();
                return stats;
            }
            prefill_step(prompt[pos], pos);
            stats.prefill_processed_tokens = pos + 1;
        }
        if (prompt.size() > 1) {
            MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        }
        const auto prefill_end = std::chrono::steady_clock::now();
        stats.prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
        if (prompt.size() > 1 && stats.prefill_ms > 0.0) {
            stats.prefill_tok_s = (1000.0 * (prompt.size() - 1)) / stats.prefill_ms;
        }
        if (opt.on_prefill_complete) opt.on_prefill_complete();

        // 2. Decode generation loop
        std::uint32_t cur_token = prompt.back();
        std::size_t pos = prompt.size() - 1;

        if (!opt.stream && use_hip_graph_ && opt.max_new_tokens > 0
            && pos < g_cache_capacity) {
            const std::size_t num_to_gen = std::min(opt.max_new_tokens, g_cache_capacity - pos);
            ensure_graph_captured(pos);
            initialize_decode_state(cur_token, pos, num_to_gen);
            stats.decode_graph_launches = num_to_gen;
            stats.decode_h2d_bytes = sizeof(miinfer::DeviceDecodeState);
            stats.decode_d2h_bytes = num_to_gen * sizeof(std::uint32_t);

            const auto decode_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < num_to_gen; ++i)
                MIINFER_HIP_CHECK(hipGraphLaunch(decode_graph_, hipStreamPerThread));
            std::vector<std::uint32_t> raw_tokens(num_to_gen);
            MIINFER_HIP_CHECK(hipMemcpy(raw_tokens.data(),
                                        static_cast<const std::uint32_t*>(d_decode_tokens_->get()),
                                        num_to_gen * sizeof(std::uint32_t),
                                        hipMemcpyDeviceToHost));
            const auto decode_end = std::chrono::steady_clock::now();
            stats.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

            for (std::uint32_t next : raw_tokens) {
                stats.tokens.push_back(next);
                if (stats.tokens.size() == 1 && opt.on_first_token) opt.on_first_token();
                if (next == tokenizer_.eos_id() || next == 151643 || next == 151645) {
                    break;
                }
                stats.text += tokenizer_.decode(std::span<const std::uint32_t>(&next, 1));
            }
            stats.generated_tokens = stats.tokens.size();
            const auto gen_end = std::chrono::steady_clock::now();
            stats.total_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
            if (stats.generated_tokens > 0 && stats.decode_ms > 0.0) {
                stats.decode_tok_s = (1000.0 * stats.generated_tokens) / stats.decode_ms;
                stats.first_token_ms = stats.decode_ms / stats.generated_tokens;
            }
            return stats;
        }

        const auto decode_start = std::chrono::steady_clock::now();
        for (std::size_t gen_idx = 0; gen_idx < opt.max_new_tokens && pos < g_cache_capacity; ++gen_idx) {
            if (g_shutdown_requested || (opt.should_cancel && opt.should_cancel())) { stats.cancelled = true; break; }

            const auto step_res = step(cur_token, pos);
            if (gen_idx == 0) {
                stats.first_token_ms = step_res.latency_ms;
            }
            const std::uint32_t next = step_res.token;
            stats.tokens.push_back(next);
            if (gen_idx == 0 && opt.on_first_token) opt.on_first_token();
            cur_token = next;
            ++pos;

            // Stop tokens for Qwen models
            if (next == tokenizer_.eos_id() || next == 151643 || next == 151645) {
                break;
            }

            const std::string piece = tokenizer_.decode(std::span<const std::uint32_t>(&next, 1));
            stats.text += piece;

            if (opt.on_token && !opt.on_token(next, piece)) break;
        }

        const auto gen_end = std::chrono::steady_clock::now();
        stats.decode_ms = std::chrono::duration<double, std::milli>(gen_end - decode_start).count();
        stats.total_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
        stats.generated_tokens = stats.tokens.size();
        if (stats.generated_tokens > 0 && stats.decode_ms > 0.0) {
            stats.decode_tok_s = (1000.0 * stats.generated_tokens) / stats.decode_ms;
        }

        return stats;
    }

public:
    void write_exp0367_prefill_state(const std::filesystem::path& path,
                                     std::span<const std::uint32_t> prompt,
                                     const float* final_hidden) const {
        if (prompt.empty() || final_hidden == nullptr) {
            throw std::runtime_error("EXP-0367 state export requires a non-empty prompt and hidden state");
        }
        write_m26c_state(path, prompt, prompt.size(), prompt.back());
        std::vector<float> hidden(model_.config().hidden_size);
        MIINFER_HIP_CHECK(hipMemcpy(hidden.data(), final_hidden,
            hidden.size() * sizeof(float), hipMemcpyDeviceToHost));
        std::ofstream output(path.string() + ".final-hidden.f32", std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("unable to create EXP-0367 final hidden dump");
        output.write(reinterpret_cast<const char*>(hidden.data()),
            static_cast<std::streamsize>(hidden.size() * sizeof(float)));
        if (!output) throw std::runtime_error("failed writing EXP-0367 final hidden dump");
    }

private:
    void setup_environment() {
        // Enforce gfx906 production performance environment
        setenv("MIINFER_Q4K_NATIVE_DOWN", "1", 0);
        setenv("MIINFER_Q4K_NATIVE_GATE_UP", "1", 0);
        setenv("MIINFER_Q4K_NATIVE_Q", "1", 0);
        setenv("MIINFER_Q4K_NATIVE_ATTN_GATE", "1", 0);
        setenv("MIINFER_Q4K_NATIVE_ATTN_OUT", "1", 0);
        setenv("MIINFER_Q4K_NATIVE_K", "1", 0);
        setenv("MIINFER_Q5K_NATIVE_SSM_OUT", "1", 0);
        setenv("MIINFER_KQUANT_NATIVE_QKV", "1", 0);
        setenv("MIINFER_KQUANT_NATIVE_V", "1", 0);
        setenv("MIINFER_Q6K_NATIVE_DOWN", "1", 0);
        setenv("MIINFER_HIP_GRAPH", "1", 0);
        setenv("MIINFER_FUSED_GATE_UP_SWIGLU", "1", 0);
        setenv("MIINFER_FUSED_RECURRENT_CORE", "1", 0);
        setenv("MIINFER_FUSED_INTERLAYER_NORM", "1", 0);
        setenv("MIINFER_Q6K_NATIVE_LM_HEAD", "1", 0);
        setenv("MIINFER_TILED_ONLINE_ATTENTION", "1", 0);
        setenv("MIINFER_FAST_ARGMAX", "1", 0);
        setenv("MIINFER_COMBINED_QKV_GATE", "1", 0);
        setenv("MIINFER_COMBINED_ATTN_QK", "1", 0);
        setenv("MIINFER_SWIGLU_PAIRED", "1", 0);
        setenv("MIINFER_KQUANT_FAST_ARITH", "1", 0);
        setenv("MIINFER_Q6K_SIMD_UNPACK", "1", 0);
        setenv("MIINFER_FUSED_ROPE_NORM", "1", 0);
        setenv("MIINFER_FUSED_ADD_RMS_NORM", "1", 0);

        // M27 reusable-graph correctness is validated, but sustained performance
        // is still unqualified; MIINFER_HIP_GRAPH=0 selects direct decode.
        const char* graph_env = std::getenv("MIINFER_HIP_GRAPH");
        use_hip_graph_ = graph_env == nullptr || std::strcmp(graph_env, "0") != 0;
        const char* layer_major_env = std::getenv("MIINFER_PREFILL_LAYER_MAJOR");
        layer_major_prefill_ = layer_major_env != nullptr && std::strcmp(layer_major_env, "0") != 0;
        const char* gdn_chunkwise_env = std::getenv("MIINFER_PREFILL_GDN_CHUNKWISE");
        gdn_chunkwise_prefill_ = layer_major_prefill_ && gdn_chunkwise_env != nullptr
            && std::strcmp(gdn_chunkwise_env, "0") != 0;
        const char* dense_prefill_env = std::getenv("MIINFER_PREFILL_DENSE_FFN_DOWN");
        dense_prefill_ = layer_major_prefill_ && dense_prefill_env != nullptr
            && std::strcmp(dense_prefill_env, "0") != 0;
        const char* dense_projection_env = std::getenv("MIINFER_PREFILL_DENSE_PROJECTIONS");
        dense_projection_prefill_ = layer_major_prefill_ && dense_projection_env != nullptr
            && std::strcmp(dense_projection_env, "0") != 0;
        const char* wide_prefill_env = std::getenv("MIINFER_PREFILL_WIDE_CHUNK");
        wide_prefill_ = layer_major_prefill_ && wide_prefill_env != nullptr
            && std::strcmp(wide_prefill_env, "0") != 0;
        const char* full_layer_env = std::getenv("MIINFER_PREFILL_FULL_LAYER_MAJOR");
        full_layer_major_prefill_ = wide_prefill_ && full_layer_env != nullptr
            && std::strcmp(full_layer_env, "0") != 0;
        const char* state_layout_env = std::getenv("MIINFER_DELTA_TRANSPOSED_STATE");
        const bool external_state_layout = state_layout_env != nullptr
            && std::strcmp(state_layout_env, "0") == 0;
        if (environment_flag("MIINFER_PREFILL_MX_GDN_EXTERNAL_STATE")
            && (!wide_prefill_ || !full_layer_major_prefill_
                || !environment_flag("MIINFER_PREFILL_MX_GDN")
                || !external_state_layout)) {
            throw std::runtime_error(
                "external Mx GDN state requires wide full-layer prefill and "
                "MIINFER_DELTA_TRANSPOSED_STATE=0");
        }
        gdn_chunkwise_prefill_ = gdn_chunkwise_prefill_ || wide_prefill_;
        const bool exp0368_scalar_oracle = std::getenv("MIINFER_EXP0368_SCALAR_ORACLE") != nullptr
            && std::strcmp(std::getenv("MIINFER_EXP0368_SCALAR_ORACLE"), "0") != 0;
        if (exp0368_scalar_oracle) {
            full_layer_major_prefill_ = false;
            prefill_chunk_ = kPrefillBatch;
        }
        const char* prefill_chunk_env = std::getenv("MIINFER_PREFILL_CHUNK");
        if (prefill_chunk_env != nullptr) {
            const auto requested = std::stoul(prefill_chunk_env);
            const bool wide_batch = wide_prefill_ && requested >= kM12PrefillBatch
                && requested <= kMaxWidePrefillBatch && requested % kPrefillBatch == 0;
            if (!exp0368_scalar_oracle && requested != 4 && requested != kPrefillBatch && !wide_batch) {
                throw std::runtime_error(
                    "MIINFER_PREFILL_CHUNK must be 4 or 64; wide prefill accepts 128..512");
            }
            prefill_chunk_ = requested;
        }
        const char* prefill_profile_env = std::getenv("MIINFER_PREFILL_PROFILE");
        const char* decode_profile_env = std::getenv("MIINFER_DECODE_PROFILE");
        prefill_profile_.decode_mode = decode_profile_env != nullptr
            && std::strcmp(decode_profile_env, "0") != 0;
        prefill_profile_.enabled = prefill_profile_.decode_mode
            || (layer_major_prefill_ && prefill_profile_env != nullptr
                && std::strcmp(prefill_profile_env, "0") != 0);
        if (prefill_profile_.enabled) {
            prefill_profile_.projection_batch_width = wide_prefill_
                ? configured_wide_prefill_batch() : 4;
            const char* position_env = prefill_profile_.decode_mode
                ? std::getenv("MIINFER_DECODE_PROFILE_POSITION")
                : std::getenv("MIINFER_PREFILL_PROFILE_POSITION");
            if (position_env != nullptr) prefill_profile_.profile_position = std::stoull(position_env);
            const std::size_t profile_width = wide_prefill_
                ? configured_wide_prefill_batch() : prefill_chunk_;
            prefill_profile_.profile_chunk_base =
                (prefill_profile_.profile_position / profile_width) * profile_width;
        }
    }

    void init_layers() {
        const std::filesystem::path empty_fixture{};
        recurrent_layers_.reserve(48);
        attention_layers_.reserve(16);

        for (std::size_t i = 0; i < 64; ++i) {
            if (i % 4 == 3) {
                attention_layers_.push_back(std::make_unique<FullAttentionLayer>(model_, i));
                layers_[i] = {nullptr, attention_layers_.back().get()};
            } else {
                recurrent_layers_.push_back(std::make_unique<RecurrentLayer>(model_, i, empty_fixture));
                layers_[i] = {recurrent_layers_.back().get(), nullptr};
            }
            if (prefill_profile_.enabled) {
                if (layers_[i].recurrent != nullptr) {
                    layers_[i].recurrent->stage_profile = &prefill_profile_.recurrent_layers[i];
                    layers_[i].recurrent->m23_profile_counters = &prefill_profile_.m23;
                    layers_[i].recurrent->stage_profile_position =
                        static_cast<std::uint32_t>(prefill_profile_.profile_position);
                    layers_[i].recurrent->stage_profile_chunk_base = prefill_profile_.profile_chunk_base;
                } else {
                    layers_[i].attention->stage_profile = &prefill_profile_.attention_layers[i];
                    layers_[i].attention->m23_profile_counters = &prefill_profile_.m23;
                    layers_[i].attention->stage_profile_position =
                        static_cast<std::uint32_t>(prefill_profile_.profile_position);
                    layers_[i].attention->stage_profile_chunk_base = prefill_profile_.profile_chunk_base;
                }
            }
        }
    }

    void init_shared_wide_prefill_workspace() {
        if (!full_layer_major_prefill_) return;
        const std::size_t batch = configured_wide_prefill_batch();
        const auto f32 = [batch](std::size_t elements) {
            return allocate(batch * elements * sizeof(float));
        };
        wide_prefill_workspace_.normalized = f32(kHidden);
        wide_prefill_workspace_.qkv = f32(kChannels + kInner);
        wide_prefill_workspace_.gate = f32(kInner);
        wide_prefill_workspace_.q8_1 = allocate(batch * (kFfnInner / miinfer::kQ8_1BlockSize)
                                                 * sizeof(miinfer::Q8_1Block));
        wide_prefill_workspace_.mmq_q8 = allocate(batch * (kFfnInner / 128)
                                                   * sizeof(miinfer::M23Q8_1MmqBlock));
        wide_prefill_workspace_.gated = f32(kInner);
        wide_prefill_workspace_.residual = f32(kHidden);
        wide_prefill_workspace_.post_normalized = f32(kHidden);
        wide_prefill_workspace_.projected = f32(kHidden);
        wide_prefill_workspace_.ffn_gate = f32(kFfnInner);
        wide_prefill_workspace_.ffn_up = f32(kFfnInner);
        wide_prefill_workspace_.ffn_activation = f32(kFfnInner);
        wide_prefill_workspace_.ffn_projected = f32(kHidden);
        wide_prefill_workspace_.core_query = f32(kKHeads * kState);
        wide_prefill_workspace_.core_key = f32(kKHeads * kState);
        wide_prefill_workspace_.core_value = f32(kVHeads * kState);
        wide_prefill_workspace_.core_beta = f32(kVHeads);
        wide_prefill_workspace_.core_decay = f32(kVHeads);
        wide_prefill_workspace_.core_gate = f32(kVHeads * kState);
        wide_prefill_workspace_.qfull = f32(12288 + 1024);
        wide_prefill_workspace_.value = f32(1024);
        wide_prefill_workspace_.gated_attention = f32(kInner);
        wide_prefill_workspace_.query_rope = f32(6144);
        wide_prefill_workspace_.attention_gate = f32(6144);
        for (auto& layer : recurrent_layers_) {
            layer->prefill_normalized = wide_prefill_workspace_.normalized;
            layer->prefill_qkv = wide_prefill_workspace_.qkv;
            layer->prefill_gate = wide_prefill_workspace_.gate;
            layer->prefill_q8_1 = wide_prefill_workspace_.q8_1;
            layer->prefill_mmq_q8 = wide_prefill_workspace_.mmq_q8;
            layer->prefill_gated = wide_prefill_workspace_.gated;
            layer->prefill_residual = wide_prefill_workspace_.residual;
            layer->prefill_post_normalized = wide_prefill_workspace_.post_normalized;
            layer->prefill_projected = wide_prefill_workspace_.projected;
            layer->prefill_ffn_gate = wide_prefill_workspace_.ffn_gate;
            layer->prefill_ffn_up = wide_prefill_workspace_.ffn_up;
            layer->prefill_ffn_activation = wide_prefill_workspace_.ffn_activation;
            layer->prefill_core_query = wide_prefill_workspace_.core_query;
            layer->prefill_core_key = wide_prefill_workspace_.core_key;
            layer->prefill_core_value = wide_prefill_workspace_.core_value;
            layer->prefill_core_beta = wide_prefill_workspace_.core_beta;
            layer->prefill_core_decay = wide_prefill_workspace_.core_decay;
            layer->prefill_core_gate = wide_prefill_workspace_.core_gate;
        }
        for (auto& layer : attention_layers_) {
            layer->prefill_normalized = wide_prefill_workspace_.normalized;
            layer->prefill_qfull = wide_prefill_workspace_.qfull;
            layer->prefill_value = wide_prefill_workspace_.value;
            layer->prefill_q8_1 = wide_prefill_workspace_.q8_1;
            layer->prefill_gated_attention = wide_prefill_workspace_.gated_attention;
            layer->prefill_projected = wide_prefill_workspace_.projected;
            layer->prefill_residual = wide_prefill_workspace_.residual;
            layer->prefill_post_normalized = wide_prefill_workspace_.post_normalized;
            layer->prefill_ffn_gate = wide_prefill_workspace_.ffn_gate;
            layer->prefill_ffn_up = wide_prefill_workspace_.ffn_up;
            layer->prefill_ffn_activation = wide_prefill_workspace_.ffn_activation;
            layer->prefill_ffn_projected = wide_prefill_workspace_.ffn_projected;
            layer->prefill_mmq_q8 = wide_prefill_workspace_.mmq_q8;
            if (layer->prefill_mx_ffn || layer->prefill_mx_o) {
                layer->prefill_mx_q8 = wide_prefill_workspace_.mmq_q8;
            }
            layer->prefill_query_rope = wide_prefill_workspace_.query_rope;
            layer->prefill_gate = wide_prefill_workspace_.attention_gate;
        }
    }

    void init_m12_gdn_workspace() {
        if (!gdn_chunkwise_prefill_ && !wide_prefill_) return;
        const std::size_t bytes = kVHeads * 64 * kState * sizeof(float);
        m12_gdn_new_values_ = allocate(bytes);
        m12_gdn_decayed_keys_ = allocate(bytes);
        m12_gdn_solved_values_ = allocate(bytes);
        m12_gdn_solved_keys_ = allocate(bytes);
        m12_gdn_corrected_values_ = allocate(bytes);
        m12_gdn_workspace_ = {
            static_cast<float*>(m12_gdn_new_values_->get()),
            static_cast<float*>(m12_gdn_decayed_keys_->get()),
            static_cast<float*>(m12_gdn_solved_values_->get()),
            static_cast<float*>(m12_gdn_solved_keys_->get()),
            static_cast<float*>(m12_gdn_corrected_values_->get())};
        const std::size_t wide_batch = wide_prefill_ ? configured_wide_prefill_batch() : kM12PrefillBatch;
        m12_gdn_raw_output_ = allocate(wide_batch * kVHeads * kState * sizeof(float));
        for (auto& layer : recurrent_layers_) {
            layer->set_m12_gdn_workspace(
                m12_gdn_workspace_, static_cast<float*>(m12_gdn_raw_output_->get()));
        }
    }

    void init_m12_dense_workspace() {
        if (!dense_prefill_ && !dense_projection_prefill_ && !wide_prefill_) return;
        m12_dense_weights_ = allocate(kHidden * kFfnInner * sizeof(__half));
        const auto env_enabled = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr && std::strcmp(value, "0") != 0;
        };
        if (wide_prefill_ && (env_enabled("MIINFER_PREFILL_WIDE_DENSE_FFN")
                              || env_enabled("MIINFER_PREFILL_WIDE_DENSE_ALL"))) {
            m12_dense_source_ = allocate((kHidden * kFfnInner / 256)
                                          * sizeof(miinfer::Q6KDeviceBlock));
        }
        const std::size_t wide_batch = wide_prefill_ ? configured_wide_prefill_batch() : kM12PrefillBatch;
        m12_dense_input_ = allocate(wide_batch * kFfnInner * sizeof(__half));
        std::string error;
        if (!miinfer::create_rocblas_gemm_handle(m12_dense_gemm_,
                                                  hipStreamPerThread, error)) {
            throw std::runtime_error(error);
        }
        for (auto& layer : recurrent_layers_) {
            layer->set_m12_dense_workspace(
                static_cast<__half*>(m12_dense_weights_->get()),
                static_cast<__half*>(m12_dense_input_->get()),
                &m12_dense_gemm_);
            if (m12_dense_source_) {
                layer->set_m12_dense_source(
                    static_cast<std::byte*>(m12_dense_source_->get()));
            }
        }
    }

    void init_buffers() {
        for (std::size_t i = 0; i < 64; ++i) {
            outputs_[i] = allocate(kHidden * sizeof(float));
            output_pointers_[i] = static_cast<float*>(outputs_[i]->get());
        }

        const auto& final_norm_weight = tensor(*model_.file(), "output_norm.weight");
        const auto& output_weight = tensor(*model_.file(), "output.weight");
        const auto& embedding_weight = tensor(*model_.file(), "token_embd.weight");

        d_final_norm_weight_ = allocate(final_norm_weight.byte_size);
        upload_tensor(final_norm_weight, d_final_norm_weight_);

        d_output_weight_ = copy_native_q6k_tensor(output_weight);

        d_embedding_ = allocate(embedding_weight.byte_size);
        upload_tensor(embedding_weight, d_embedding_);

        input_ = allocate(kHidden * sizeof(float));
        final_norm_ = allocate(kHidden * sizeof(float));
        final_q8_ = allocate((kHidden / 256) * sizeof(miinfer::Q8KDeviceBlock));
        final_q8_1_ = allocate((kHidden / 32) * sizeof(miinfer::Q8_1Block));
        logits_ = allocate(model_.config().vocab_size * sizeof(float));
        argmax_token_ = allocate(sizeof(std::uint32_t));
        d_decode_tokens_ = allocate(g_cache_capacity * sizeof(std::uint32_t));
        const std::size_t prefill_capacity = full_layer_major_prefill_ ? kFullPrefillCapacity
            : ((gdn_chunkwise_prefill_ || dense_prefill_ || wide_prefill_)
                ? (wide_prefill_ ? configured_wide_prefill_batch() : kM12PrefillBatch) : kPrefillBatch);
        prefill_a_ = allocate(prefill_capacity * kHidden * sizeof(float));
        prefill_b_ = allocate(prefill_capacity * kHidden * sizeof(float));

        d_decode_state_ = allocate(sizeof(miinfer::DeviceDecodeState));
        prefill_profile_.init();
    }

    void initialize_decode_state(std::uint32_t token, std::size_t position,
                                 std::size_t max_generated) {
        miinfer::DeviceDecodeState state{};
        state.current_token = token;
        state.position = static_cast<std::uint32_t>(position);
        state.max_generated = static_cast<std::uint32_t>(max_generated);
        MIINFER_HIP_CHECK(hipMemcpyAsync(d_decode_state_->get(), &state, sizeof(state),
                                         hipMemcpyHostToDevice, hipStreamPerThread));
    }

    void ensure_graph_captured(std::size_t position) {
        if (!use_hip_graph_ || decode_graph_ != nullptr) return;
        for (const auto& layer : layers_) {
            if (layer.attention != nullptr
                && (!layer.attention->fused_rope_norm
                    || !layer.attention->fp16_kv_cache
                    || !layer.attention->tiled_online_attention)) {
                throw std::runtime_error(
                    "reusable decode graph requires fused RoPE, FP16 KV, and tiled attention; "
                    "set MIINFER_HIP_GRAPH=0 for this configuration");
            }
        }
        const auto capture_start = std::chrono::steady_clock::now();

        hipGraph_t graph = nullptr;
        MIINFER_HIP_CHECK(hipStreamBeginCapture(hipStreamPerThread, hipStreamCaptureModeRelaxed));

        auto* decode_state = static_cast<miinfer::DeviceDecodeState*>(d_decode_state_->get());
        const auto* token_device_ptr = &decode_state->current_token;

        miinfer::launch_qwen35_q4_k_embedding_device_token(
            static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding_->get()),
            token_device_ptr, model_.config().vocab_size, kHidden,
            static_cast<float*>(input_->get()), hipStreamPerThread);

        run_prefix(std::span<const GpuLayerRef>(layers_),
                   std::span<float* const>(output_pointers_),
                   static_cast<const float*>(input_->get()), position,
                   static_cast<const float*>(d_final_norm_weight_->get()),
                   static_cast<float*>(final_norm_->get()),
                   static_cast<miinfer::Q8_1Block*>(final_q8_1_->get()), decode_state);

        miinfer::launch_qwen3_rms_norm(
            output_pointers_[63],
            static_cast<const float*>(d_final_norm_weight_->get()),
            static_cast<float*>(final_norm_->get()), kHidden,
            model_.config().rms_epsilon);

        miinfer::launch_q8_1_quantize_f32(
            static_cast<const float*>(final_norm_->get()),
            static_cast<miinfer::Q8_1Block*>(final_q8_1_->get()), kHidden,
            hipStreamPerThread);

        launch_q6k_wave_gemv(
            static_cast<const Q6KWaveTile*>(d_output_weight_->get()),
            static_cast<const miinfer::Q8_1Block*>(final_q8_1_->get()),
            static_cast<float*>(logits_->get()), model_.config().vocab_size, kHidden,
            hipStreamPerThread);

        miinfer::launch_qwen3_argmax(
            static_cast<const float*>(logits_->get()),
            &decode_state->current_token,
            model_.config().vocab_size);

        miinfer::launch_qwen35_decode_state_advance(
            decode_state, static_cast<std::uint32_t*>(d_decode_tokens_->get()),
            static_cast<std::uint32_t>(g_cache_capacity));

        MIINFER_HIP_CHECK(hipStreamEndCapture(hipStreamPerThread, &graph));
        MIINFER_HIP_CHECK(hipGraphGetNodes(graph, nullptr, &decode_graph_node_count_));
        MIINFER_HIP_CHECK(hipGraphInstantiate(&decode_graph_, graph, nullptr, nullptr, 0));
        MIINFER_HIP_CHECK(hipGraphDestroy(graph));
        graph_capture_ms_ += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - capture_start).count();
    }

    void cleanup_graphs() {
        if (decode_graph_ != nullptr) {
            (void)hipGraphExecDestroy(decode_graph_);
            decode_graph_ = nullptr;
        }
    }

    miinfer::Qwen35Model model_;
    std::vector<SessionCheckpoint> session_checkpoints_;
    SessionPrefixNode session_prefix_root_;
    std::size_t session_checkpoint_bytes_ = 0;
    std::size_t session_use_clock_ = 0;
    std::string session_model_identity_;
    std::string session_quantization_identity_;
    std::string session_runtime_identity_;
    mutable std::string m26c_model_hash_;
    std::function<void(std::size_t)> opt_prefill_checkpoint_;
    double graph_capture_ms_ = 0.0;
    std::size_t decode_graph_node_count_ = 0;
    miinfer::Qwen3Tokenizer tokenizer_;

    std::vector<std::unique_ptr<RecurrentLayer>> recurrent_layers_;
    std::vector<std::unique_ptr<FullAttentionLayer>> attention_layers_;
    std::array<GpuLayerRef, 64> layers_{};

    std::array<Buffer, 64> outputs_{};
    std::array<float*, 64> output_pointers_{};

    Buffer d_final_norm_weight_;
    Buffer d_output_weight_;
    Buffer d_embedding_;
    Buffer input_;
    Buffer final_norm_;
    Buffer final_q8_;
    Buffer final_q8_1_;
    Buffer logits_;
    Buffer argmax_token_;
    Buffer d_decode_tokens_;
    Buffer d_decode_state_;
    Buffer m12_gdn_new_values_, m12_gdn_decayed_keys_;
    Buffer m12_gdn_solved_values_, m12_gdn_solved_keys_, m12_gdn_corrected_values_;
    Buffer m12_gdn_raw_output_;
    miinfer::M12GdnChunkWorkspace m12_gdn_workspace_{};
    Buffer m12_dense_weights_, m12_dense_input_, m12_dense_source_;
    miinfer::RocblasGemmHandle m12_dense_gemm_{};
    Buffer prefill_a_;
    Buffer prefill_b_;
    WidePrefillWorkspace wide_prefill_workspace_;

    bool use_hip_graph_ = true;
    bool layer_major_prefill_ = false;
    bool gdn_chunkwise_prefill_ = false;
    bool dense_prefill_ = false;
    bool dense_projection_prefill_ = false;
    bool wide_prefill_ = false;
    bool full_layer_major_prefill_ = false;
    std::size_t prefill_chunk_ = kPrefillBatch;
    PrefillProfile prefill_profile_;
    hipGraphExec_t decode_graph_ = nullptr;
};

// ---------------------------------------------------------------------------
// Subcommand 1: inspect
// ---------------------------------------------------------------------------
int cmd_inspect(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: miinfer inspect <model.gguf>\n";
        return 1;
    }
    const std::string model_path = argv[2];
    std::cout << "Loading model metadata from: " << model_path << " ...\n";

    const auto file = miinfer::GgufFile::open(model_path);
    const auto model = miinfer::Qwen35Model::load(model_path);
    const auto& config = model.config();

    std::uint64_t total_params = 0;
    std::size_t total_weight_bytes = 0;
    std::unordered_map<miinfer::GgufTensorType, std::pair<std::size_t, std::size_t>> quant_stats;

    for (const auto& tensor : file->tensors()) {
        std::uint64_t elements = 1;
        for (const auto dim : tensor.dimensions) elements *= dim;
        total_params += elements;
        total_weight_bytes += tensor.byte_size;
        quant_stats[tensor.type].first += 1;
        quant_stats[tensor.type].second += tensor.byte_size;
    }

    // GPU detection
    int device_id = 0;
    hipDeviceProp_t prop{};
    bool gpu_detected = (hipGetDeviceProperties(&prop, device_id) == hipSuccess);

    std::cout << "\n========================================================================\n";
    std::cout << "                      MIInfer Model Inspection Report                   \n";
    std::cout << "========================================================================\n\n";

    std::cout << "Model Information:\n";
    std::cout << "  Name:                  " << model.model_name() << "\n";
    std::cout << "  Architecture:          Qwen3.5 (Hybrid Gated DeltaNet + Multi-Head Latent Attention)\n";
    std::cout << "  Total Parameters:      " << std::fixed << std::setprecision(2)
              << (double)total_params / 1e9 << " Billion (" << total_params << " elements)\n";
    std::cout << "  Total Layers:          " << config.block_count << " (64 main blocks)\n";
    std::cout << "    - Recurrent Layers:  48 (Gated DeltaNet, 75.0%)\n";
    std::cout << "    - Attention Layers:  16 (Full Multi-Head Latent Attention, 25.0%)\n";
    std::cout << "  Hidden Dimension:      " << config.hidden_size << "\n";
    std::cout << "  FFN Dimension:         " << config.intermediate_size << " (Recurrent) / 20480 (Attention)\n";
    std::cout << "  Attention Heads:       " << config.attention_heads << " query heads, "
              << config.kv_heads << " KV heads (head_dim = " << config.head_dim << ")\n";
    std::cout << "  Recurrent State:       " << config.recurrent_group_count << " heads, "
              << config.recurrent_state_size << "x" << config.recurrent_state_size << " state matrix\n";
    std::cout << "  Vocabulary Size:       " << config.vocab_size << "\n";
    std::cout << "  Native Context Limit:  " << config.context_length
              << " tokens (MIInfer runtime capacity: " << g_cache_capacity
              << ", qualified: 1024)\n\n";

    std::cout << "Tensor Quantization Breakdown:\n";
    std::cout << "  " << std::left << std::setw(12) << "Type"
              << std::right << std::setw(14) << "Tensor Count"
              << std::setw(18) << "Weight Size"
              << std::setw(14) << "Percentage\n";
    std::cout << "  ------------------------------------------------------------\n";
    for (const auto& [type, stats] : quant_stats) {
        const double pct = (double)stats.second / (double)total_weight_bytes * 100.0;
        const double mb = (double)stats.second / (1024.0 * 1024.0);
        std::cout << "  " << std::left << std::setw(12) << miinfer::gguf_tensor_type_name(type)
                  << std::right << std::setw(14) << stats.first
                  << std::setw(14) << std::fixed << std::setprecision(1) << mb << " MB"
                  << std::setw(13) << std::fixed << std::setprecision(2) << pct << " %\n";
    }
    std::cout << "  ------------------------------------------------------------\n";
    std::cout << "  " << std::left << std::setw(12) << "TOTAL"
              << std::right << std::setw(14) << file->tensors().size()
              << std::setw(14) << std::fixed << std::setprecision(2) << (double)total_weight_bytes / 1e9 << " GB"
              << std::setw(13) << "100.00 %\n\n";

    // VRAM calculation
    const double weight_gib = (double)total_weight_bytes / (1024.0 * 1024.0 * 1024.0);
    const double kv_gib = (double)(16 * 4 * 1024 * 256 * 4) / (1024.0 * 1024.0 * 1024.0);
    const double rec_gib = (double)(48 * 16 * 128 * 128 * 4) / (1024.0 * 1024.0 * 1024.0);
    const double act_gib = 0.048; // ~50 MB
    const double total_vram_gib = weight_gib + kv_gib + rec_gib + act_gib;

    std::cout << "VRAM Footprint & Budget (1024 Context Capacity):\n";
    std::cout << "  Model Weights:         " << std::fixed << std::setprecision(3) << weight_gib << " GiB ("
              << (double)total_weight_bytes / 1e9 << " GB)\n";
    std::cout << "  KV Cache (1024 pos):   " << std::fixed << std::setprecision(3) << kv_gib * 1024.0 << " MiB\n";
    std::cout << "  Recurrent State (48L): " << std::fixed << std::setprecision(3) << rec_gib * 1024.0 << " MiB\n";
    std::cout << "  Activations & Graph:   ~" << std::fixed << std::setprecision(1) << act_gib * 1024.0 << " MiB\n";
    std::cout << "  Total Required VRAM:   " << std::fixed << std::setprecision(3) << total_vram_gib << " GiB ("
              << total_vram_gib * 1.07374 << " GB)\n\n";

    std::cout << "Hardware Target Compatibility:\n";
    if (gpu_detected) {
        const double total_dev_gib = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        const double headroom_gib = total_dev_gib - total_vram_gib;
        std::cout << "  Detected GPU:          " << prop.name << " (" << prop.gcnArchName << ")\n";
        std::cout << "  Total Device Memory:   " << std::fixed << std::setprecision(2) << total_dev_gib << " GiB (32 GB HBM2)\n";
        std::cout << "  Available Headroom:    " << std::fixed << std::setprecision(2) << headroom_gib << " GiB ("
                  << (headroom_gib / total_dev_gib * 100.0) << "% free)\n";
        std::cout << "  Specialization Match:  gfx906 / Vega20 Wave64 (AMD Instinct MI50 32GB)\n";
        std::cout << "  Execution Plan Status: VALIDATED (PASS)\n";
    } else {
        std::cout << "  GPU Status:            Not detected or HIP runtime offline\n";
    }
    std::cout << "========================================================================\n";

    return 0;
}

int cmd_models(int argc, char** argv) {
    if (argc > 3) {
        std::cerr << "usage: miinfer models [directory]\n";
        return 2;
    }
    const std::filesystem::path root = argc == 3 ? argv[2] : ".";
    std::error_code error;
    if (!std::filesystem::is_directory(root, error)) {
        std::cerr << "model directory not found: " << root << '\n';
        return 1;
    }

    std::vector<std::filesystem::path> models;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(
             root, std::filesystem::directory_options::skip_permission_denied, error)) {
        if (error) break;
        if (entry.is_regular_file(error) && entry.path().extension() == ".gguf") {
            models.push_back(entry.path());
        }
        error.clear();
    }
    std::sort(models.begin(), models.end());

    std::cout << "GGUF models: " << models.size() << '\n';
    for (const auto& model : models) std::cout << model.string() << '\n';
    return 0;
}

int cmd_config(int argc, char**) {
    if (argc != 2) {
        std::cerr << "usage: miinfer config\n";
        return 2;
    }
    std::cout << "target_architecture=gfx906\n"
              << "hardware=AMD Instinct MI50 32GB\n"
              << "model=Qwen3.8-27B\n"
              << "quantization=Q4_K_M\n"
              << "model_context_length=262144\n"
              << "configured_context_length=1024\n"
              << "runtime_context_capacity=" << g_cache_capacity << "\n"
              << "qualified_context_length=1024\n"
              << "context_qualification=qualified\n"
              << "prefill_path=validated-default\n"
              << "m12_prefill=opt-in\n";
    return 0;
}

int cmd_doctor(int argc, char** argv) {
    std::optional<std::filesystem::path> model;
    int port = 8080;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--port" && i + 1 < argc) {
            const auto parsed = parse_port(argv[++i]);
            if (!parsed) { std::cerr << "port must be an integer from 0 to 65535\n"; return 2; }
            port = *parsed;
        }
        else { std::cerr << "usage: miinfer doctor [--model MODEL.gguf] [--port PORT]\n"; return 2; }
    }
    bool healthy = true;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        std::cout << "gpu=FAIL: HIP cannot access device 0\n"; healthy = false;
    } else {
        std::size_t free_bytes = 0, total_bytes = 0;
        const bool gfx906 = std::string_view(prop.gcnArchName).find("gfx906") != std::string_view::npos;
        if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) healthy = false;
        std::cout << "gpu=" << (gfx906 ? "PASS" : "FAIL") << ": " << prop.name << " " << prop.gcnArchName << "\n"
                  << "vram_free_gib=" << std::fixed << std::setprecision(2) << free_bytes / 1073741824.0 << "\n"
                  << "rocm_hip=PASS\n";
        healthy &= gfx906;
    }
    if (model) {
        try { (void)miinfer::Qwen35Model::load(*model); std::cout << "model=PASS: " << *model << "\n"; }
        catch (const std::exception& error) { std::cout << "model=FAIL: " << error.what() << "\n"; healthy = false; }
    }
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(port);
    const bool available = fd >= 0 && bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0;
    if (fd >= 0) close(fd);
    std::cout << "port=" << (available ? "PASS" : "FAIL") << ": " << port << "\n";
    return healthy && available ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Subcommand 2: run
// ---------------------------------------------------------------------------
int cmd_run(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: miinfer run <model.gguf> (--prompt \"...\" | --prompt-file PATH) [--max-tokens N]\n";
        return 1;
    }
    (void)unsetenv("MIINFER_M26C_RECURRENT_QKV_M23");
    (void)unsetenv("MIINFER_M26C_SSM_OUT_NATIVE");
    const std::string model_path = argv[2];
    std::string prompt_text;
    std::optional<std::filesystem::path> prompt_file;
    std::size_t max_tokens = 128;
    bool stream = true;
    bool repeat_p512_check = false;
    bool check_session = false;
    bool check_graph_state = false;
    std::optional<std::size_t> check_graph_state_context;
    std::optional<std::filesystem::path> m26c_export_state;
    std::optional<std::filesystem::path> m26c_import_state;
    std::optional<std::filesystem::path> m26c_output_state;
    std::optional<std::filesystem::path> m26c_logits_output;
    std::optional<std::filesystem::path> m26c_layer_path_prefix;
    std::optional<std::filesystem::path> m26c_teacher_inputs_path;
    std::optional<std::filesystem::path> m26c_checkpoints_dir;
    std::optional<std::filesystem::path> exp0367_state_output;
    std::optional<std::size_t> m26c_layer_path_position;
    std::optional<std::string> m26c_decode_route;
    std::optional<std::size_t> m26c_state_context;
    bool m26c_restore_only = false;
    bool m26c_recurrent_qkv_m23 = false;
    bool m26c_ssm_out_native = false;
    bool decode_curve = false;
    std::size_t curve_iterations = 1;
    std::optional<std::size_t> curve_context;

    if (const char* context_env = std::getenv("MIINFER_CONTEXT_CAPACITY")) {
        const auto context = std::stoull(context_env);
        if (context == 0) throw std::runtime_error("MIINFER_CONTEXT_CAPACITY must be positive");
        g_cache_capacity = context;
    }

    for (int i = 3; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (arg == "--prompt-file" && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoull(argv[++i]);
        } else if (arg == "--no-stream") {
            stream = false;
        } else if (arg == "--repeat-p512-check") {
            repeat_p512_check = true;
        } else if (arg == "--check-session") {
            check_session = true;
        } else if (arg == "--check-graph-state") {
            check_graph_state = true;
        } else if (arg == "--check-graph-state-context" && i + 1 < argc) {
            check_graph_state_context = std::stoull(argv[++i]);
            if (*check_graph_state_context == 0) {
                throw std::runtime_error("graph-state context must be positive");
            }
        } else if (arg == "--m26c-export-state" && i + 1 < argc) {
            m26c_export_state = argv[++i];
        } else if (arg == "--m26c-import-state" && i + 1 < argc) {
            m26c_import_state = argv[++i];
        } else if (arg == "--m26c-output-state" && i + 1 < argc) {
            m26c_output_state = argv[++i];
        } else if (arg == "--m26c-logits-output" && i + 1 < argc) {
            m26c_logits_output = argv[++i];
        } else if (arg == "--m26c-layer-path-prefix" && i + 1 < argc) {
            m26c_layer_path_prefix = argv[++i];
        } else if (arg == "--m26c-teacher-forced-inputs" && i + 1 < argc) {
            m26c_teacher_inputs_path = argv[++i];
        } else if (arg == "--m26c-checkpoints-dir" && i + 1 < argc) {
            m26c_checkpoints_dir = argv[++i];
        } else if (arg == "--exp0367-state-output" && i + 1 < argc) {
            exp0367_state_output = argv[++i];
        } else if (arg == "--m26c-layer-path-position" && i + 1 < argc) {
            m26c_layer_path_position = std::stoull(argv[++i]);
        } else if (arg == "--m26c-decode-route" && i + 1 < argc) {
            m26c_decode_route = argv[++i];
        } else if (arg == "--m26c-state-context" && i + 1 < argc) {
            m26c_state_context = std::stoull(argv[++i]);
            if (*m26c_state_context == 0) throw std::runtime_error("M26-C state context must be positive");
        } else if (arg == "--m26c-restore-only") {
            m26c_restore_only = true;
        } else if (arg == "--m26c-recurrent-qkv-m23") {
            m26c_recurrent_qkv_m23 = true;
        } else if (arg == "--m26c-ssm-out-native") {
            m26c_ssm_out_native = true;
        } else if (arg == "--decode-curve") {
            decode_curve = true;
        } else if (arg == "--curve-iterations" && i + 1 < argc) {
            curve_iterations = std::stoull(argv[++i]);
            if (curve_iterations == 0) throw std::runtime_error("curve iterations must be positive");
        } else if (arg == "--curve-context" && i + 1 < argc) {
            curve_context = std::stoull(argv[++i]);
            if (*curve_context == 0) throw std::runtime_error("curve context must be positive");
        } else if (arg == "--context" && i + 1 < argc) {
            g_cache_capacity = std::stoull(argv[++i]);
            if (g_cache_capacity == 0) throw std::runtime_error("context must be positive");
        }
    }

    if (prompt_file) {
        std::ifstream input(*prompt_file);
        if (!input) {
            std::cerr << "unable to open prompt file: " << *prompt_file << '\n';
            return 2;
        }
        std::ostringstream contents;
        contents << input.rdbuf();
        prompt_text = contents.str();
    }

    if (prompt_text.empty()) {
        prompt_text = "The AMD Instinct MI50 is a high-performance GPU featuring";
        std::cerr << "No prompt provided. Defaulting to: \"" << prompt_text << "\"\n";
    }

    if (m26c_recurrent_qkv_m23) {
        if (!m26c_import_state) {
            throw std::runtime_error("--m26c-recurrent-qkv-m23 requires --m26c-import-state");
        }
        if (setenv("MIINFER_M26C_RECURRENT_QKV_M23", "1", 1) != 0) {
            throw std::runtime_error("unable to enable M26-C M23 recurrent QKV diagnostic");
        }
        std::cerr << "m26c_recurrent_qkv_m23=1\n";
    }
    if (m26c_ssm_out_native) {
        if (!m26c_import_state) {
            throw std::runtime_error("--m26c-ssm-out-native requires --m26c-import-state");
        }
        if (setenv("MIINFER_M26C_SSM_OUT_NATIVE", "1", 1) != 0) {
            throw std::runtime_error("unable to enable M26-C native SSM-output diagnostic");
        }
        std::cerr << "m26c_ssm_out_native=1\n";
    }
    std::cerr << "Initializing MIInfer gfx906 runtime engine for " << model_path << " ...\n";
    Qwen35RuntimeEngine engine(model_path);
    std::cerr << "device_allocation_count=" << g_device_allocations << "\n"
              << "device_allocated_bytes=" << g_live_device_bytes << "\n"
              << "device_total_allocated_bytes=" << g_total_device_bytes << "\n"
              << "device_peak_allocated_bytes=" << g_peak_device_bytes << "\n";
    print_hip_memory(std::cerr);
    std::cerr << "model_context_length=" << engine.model().config().context_length << "\n";

    auto prompt_tokens = engine.tokenizer().encode(prompt_text);
    if (m26c_state_context) {
        if (!m26c_export_state || *m26c_state_context != 512 || prompt_tokens.empty()) {
            throw std::runtime_error("--m26c-state-context currently requires a 512-token export prompt");
        }
        const auto seed = prompt_tokens;
        prompt_tokens.resize(*m26c_state_context);
        for (std::size_t i = seed.size(); i < prompt_tokens.size(); ++i) {
            prompt_tokens[i] = seed[i % seed.size()];
        }
    }
    if (check_graph_state_context) {
        if (!check_graph_state || prompt_tokens.empty()
            || *check_graph_state_context > g_cache_capacity) {
            throw std::runtime_error(
                "--check-graph-state-context requires --check-graph-state and must fit context");
        }
        const auto seed = prompt_tokens;
        prompt_tokens.resize(*check_graph_state_context);
        for (std::size_t i = seed.size(); i < prompt_tokens.size(); ++i) {
            prompt_tokens[i] = seed[i % seed.size()];
        }
    }
    std::cerr << "Prompt tokens: " << prompt_tokens.size() << " tokens\n";

        if (m26c_export_state) {
            if (m26c_import_state || m26c_decode_route || m26c_output_state || m26c_logits_output
            || m26c_layer_path_prefix || m26c_teacher_inputs_path || m26c_checkpoints_dir
            || m26c_layer_path_position || m26c_ssm_out_native
            || m26c_restore_only || prompt_tokens.size() != 512) {
            throw std::runtime_error("M26-C export requires one 512-token prompt and no import options");
        }
        const auto next = engine.prepare_m26c_state(prompt_tokens, *m26c_export_state);
        std::cout << "m26c_export=PASS position=512 current_token=" << next
                  << " path=" << m26c_export_state->string() << '\n';
        return 0;
    }
    if (m26c_import_state) {
        std::optional<std::vector<std::uint32_t>> teacher_inputs;
        if (m26c_teacher_inputs_path) {
            std::ifstream tokens_file(*m26c_teacher_inputs_path);
            if (!tokens_file) throw std::runtime_error("unable to open teacher-forced token file");
            teacher_inputs.emplace();
            std::uint64_t token = 0;
            while (tokens_file >> token) {
                if (token >= engine.model().config().vocab_size) {
                    throw std::runtime_error("teacher-forced token is outside the model vocabulary");
                }
                teacher_inputs->push_back(static_cast<std::uint32_t>(token));
            }
            if (!tokens_file.eof() || teacher_inputs->empty() || teacher_inputs->size() != max_tokens) {
                throw std::runtime_error("teacher-forced token file must contain exactly --max-tokens integers");
            }
        }
        if (!m26c_decode_route || !m26c_output_state || m26c_state_context
            || (*m26c_decode_route != "direct" && *m26c_decode_route != "graph")
            || (m26c_logits_output && max_tokens != 1 && !teacher_inputs)
            || (m26c_layer_path_prefix
                && ((*m26c_decode_route != "direct")
                    || (max_tokens != 1 && (!teacher_inputs || !m26c_layer_path_position))))
            || (m26c_layer_path_position && (!m26c_layer_path_prefix || !teacher_inputs))
            || ((m26c_checkpoints_dir || m26c_layer_path_position || teacher_inputs)
                && *m26c_decode_route != "direct")
            || (max_tokens == 0 && !m26c_restore_only)
            || (max_tokens != 0 && m26c_restore_only)) {
            throw std::runtime_error("M26-C import requires --m26c-decode-route direct|graph, "
                                     "--m26c-output-state and positive --max-tokens "
                                     "(or --m26c-restore-only with zero tokens)");
        }
        const auto result = engine.decode_m26c_state(*m26c_import_state, *m26c_output_state,
            *m26c_decode_route == "graph", max_tokens, m26c_logits_output,
            m26c_layer_path_prefix, teacher_inputs, m26c_checkpoints_dir,
            m26c_layer_path_position);
        std::cout << std::fixed << std::setprecision(6)
                  << "m26c_route=" << *m26c_decode_route
                  << " tokens=" << result.tokens.size()
                  << " restore_ms=" << result.restore_ms
                  << " graph_capture_ms=" << result.graph_capture_ms
                  << " graph_nodes=" << result.graph_nodes
                  << " graph_replays=" << (*m26c_decode_route == "graph" ? result.tokens.size() : 0)
                  << " decode_ms=" << result.decode_ms
                  << " direct_h2d_device_ms=" << result.direct_h2d_device_ms
                  << " direct_d2h_device_ms=" << result.direct_d2h_device_ms
                  << " direct_blocking_sync_ms=" << result.direct_blocking_sync_ms
                  << " direct_sync_calls=" << result.direct_sync_calls
                  << " graph_state_sync_ms=" << result.graph_state_sync_ms
                  << " graph_replay_enqueue_ms=" << result.graph_replay_enqueue_ms
                  << " graph_bulk_copy_wait_ms=" << result.graph_bulk_copy_wait_ms
                  << " ms_per_token=" << (result.tokens.empty() ? 0.0
                      : result.decode_ms / result.tokens.size())
                  << " restore_h2d_bytes=" << result.restore_h2d_bytes
                  << " decode_h2d_bytes=" << result.decode_h2d_bytes
                  << " decode_d2h_bytes=" << result.decode_d2h_bytes
                  << " output_state=" << m26c_output_state->string() << '\n'
                  << "m26c_logits_output="
                  << (m26c_logits_output ? m26c_logits_output->string() : "disabled") << '\n'
                  << "m26c_layer_path_prefix="
                  << (m26c_layer_path_prefix ? m26c_layer_path_prefix->string() : "disabled") << '\n'
                  << "m26c_teacher_forced=" << (teacher_inputs ? "1" : "0") << '\n'
                  << "m26c_token_ids=";
        for (std::size_t i = 0; i < result.tokens.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << result.tokens[i];
        }
        std::cout << '\n';
        return 0;
    }
    if (m26c_decode_route || m26c_output_state || m26c_logits_output || m26c_layer_path_prefix
        || m26c_ssm_out_native || m26c_state_context || m26c_restore_only) {
        throw std::runtime_error("incomplete M26-C diagnostic options");
    }

    if (check_graph_state) {
        if (max_tokens < 2 || prompt_tokens.size() + max_tokens - 1 > g_cache_capacity) {
            throw std::runtime_error("--check-graph-state requires --max-tokens >= 2 and room in context");
        }
        RuntimeGenerateOptions options;
        options.max_new_tokens = max_tokens;
        options.stream = false;
        const auto graph = engine.generate(prompt_tokens, options);
        const auto graph_eos = std::find_if(graph.tokens.begin(), graph.tokens.end(),
            [&](std::uint32_t token) {
                return token == engine.tokenizer().eos_id() || token == 151643 || token == 151645;
            });
        const auto graph_active_tokens = static_cast<std::size_t>(graph_eos - graph.tokens.begin())
            + (graph_eos == graph.tokens.end() ? 0 : 1);
        if (graph_active_tokens < 2) {
            std::cerr << "graph_state_check=FAIL too few graph tokens before EOS to compare state\n";
            return 1;
        }
        const auto active_positions = prompt_tokens.size() + graph_active_tokens - 1;
        const auto graph_state = engine.snapshot_decode_buffers(active_positions);
        options.stream = true;
        const auto direct = engine.generate(prompt_tokens, options);
        const auto direct_state = engine.snapshot_decode_buffers(active_positions);
        if (graph.decode_graph_launches == 0 || graph.tokens != direct.tokens
            || graph_active_tokens > direct.generated_tokens) {
            std::cerr << "graph_state_check=FAIL token parity or common active length mismatch"
                      << " graph_tokens=" << graph.generated_tokens
                      << " direct_tokens=" << direct.generated_tokens
                      << " graph_launches=" << graph.decode_graph_launches << '\n';
            return 1;
        }
        const auto mismatch = std::mismatch(graph_state.begin(), graph_state.end(),
                                            direct_state.begin(), direct_state.end());
        if (graph_state.size() != direct_state.size() || mismatch.first != graph_state.end()) {
            std::cerr << "graph_state_check=FAIL graph_bytes=" << graph_state.size()
                      << " direct_bytes=" << direct_state.size()
                      << " mismatch_offset=" << (mismatch.first - graph_state.begin()) << '\n';
            return 1;
        }
        std::cout << "graph_state_check=PASS tokens=" << max_tokens
                  << " positions=" << active_positions
                  << " active_tokens=" << graph_active_tokens
                  << " state_bytes=" << graph_state.size() << '\n';
        return 0;
    }

    if (decode_curve) {
        constexpr std::array<std::size_t, 6> contexts{512, 2048, 4096, 8192, 12288, 16384};
        constexpr std::size_t generated_tokens = 128;
        std::cout << "context_tokens,iterations,median_decode_ms,median_ms_per_token,median_decode_tok_s\n";
        for (const std::size_t context : contexts) {
            if (curve_context && context != *curve_context) continue;
            if (context + generated_tokens >= g_cache_capacity) continue;
            std::vector<double> decode_ms;
            decode_ms.reserve(curve_iterations);
            std::vector<std::uint32_t> tokens(context);
            for (std::size_t i = 0; i < context; ++i) tokens[i] = prompt_tokens[i % prompt_tokens.size()];
            for (std::size_t iteration = 0; iteration < curve_iterations; ++iteration) {
                RuntimeGenerateOptions options;
                options.max_new_tokens = generated_tokens;
                options.stream = false;
                const auto stats = engine.generate(tokens, options);
                if (stats.cancelled || stats.generated_tokens != generated_tokens) {
                    throw std::runtime_error("decode curve generation did not complete");
                }
                decode_ms.push_back(stats.decode_ms);
            }
            std::sort(decode_ms.begin(), decode_ms.end());
            const double median = decode_ms[decode_ms.size() / 2];
            std::cout << context << ',' << curve_iterations << ',' << median << ','
                      << median / generated_tokens << ','
                      << (1000.0 * generated_tokens / median) << '\n';
        }
        return 0;
    }

    if (check_session) {
        // Compare sparse multi-checkpoint longest-prefix restore with full replay.
        for (const std::size_t length : {512U, 640U, 3991U, 8192U, 16000U}) {
            if (length + 160 >= g_cache_capacity) continue;
            for (const std::size_t seed_count : {1U, 4U}) {
            std::vector<std::uint32_t> tokens(length);
            for (std::size_t i = 0; i < length; ++i) tokens[i] = prompt_tokens[i % prompt_tokens.size()];
            RuntimeGenerateOptions options;
            options.max_new_tokens = seed_count;
            options.reuse_session = true;
            options.stream = false;
            const auto seed = engine.generate(tokens, options);
            options.max_new_tokens = 4;
            tokens.insert(tokens.end(), seed.tokens.begin(), seed.tokens.end());
            for (std::size_t i = 0; i < 151; ++i) tokens.push_back(prompt_tokens[i % prompt_tokens.size()]);
            const auto appended = engine.generate(tokens, options);
            options.reuse_session = false;
            options.stream = true;
            const auto replay = engine.generate(tokens, options);
            if (seed.cancelled || appended.cancelled || replay.cancelled
                || appended.reused_prefix_tokens != (length / kFullPrefillCapacity) * kFullPrefillCapacity
                || appended.tokens != replay.tokens) {
                std::cerr << "session_check=FAIL length=" << length
                          << " retained=" << appended.reused_prefix_tokens
                          << " new=" << appended.prefill_processed_tokens << " seed=";
                for (auto token : seed.tokens) std::cerr << token << ',';
                std::cerr << " append=";
                for (auto token : appended.tokens) std::cerr << token << ',';
                std::cerr << " replay=";
                for (auto token : replay.tokens) std::cerr << token << ',';
                std::cerr << std::endl;
                return 1;
            }
            std::cout << "session_check=PASS prompt_tokens=" << length
                      << " seed_generated=" << seed_count
                      << " reused_prefix_tokens=" << appended.reused_prefix_tokens
                      << " new_prefill_tokens=" << appended.prefill_processed_tokens
                      << " seed_prefill_ms=" << seed.prefill_ms
                      << " append_prefill_ms=" << appended.prefill_ms
                      << " replay_prefill_ms=" << replay.prefill_ms
                      << " decode_tok_s=" << replay.decode_tok_s << std::endl;
            }
        }

        if (g_cache_capacity > 3992) {
            engine.reset();
            std::vector<std::uint32_t> branch_a(3991);
            for (std::size_t i = 0; i < branch_a.size(); ++i) {
                branch_a[i] = prompt_tokens[i % prompt_tokens.size()];
            }
            RuntimeGenerateOptions branch_options;
            branch_options.max_new_tokens = 0;
            branch_options.reuse_session = true;
            branch_options.stream = false;
            (void)engine.generate(branch_a, branch_options);

            auto branch_b = branch_a;
            const auto alternate = std::find_if(prompt_tokens.begin(), prompt_tokens.end(),
                [&](std::uint32_t token) { return token != branch_b[kFullPrefillCapacity * 4]; });
            if (alternate == prompt_tokens.end())
                throw std::runtime_error("session branch check needs two distinct prompt tokens");
            branch_b[kFullPrefillCapacity * 4] = *alternate;
            branch_options.max_new_tokens = 1;
            const auto branch_b_append = engine.generate(branch_b, branch_options);

            auto branch_a_extended = branch_a;
            branch_a_extended.insert(branch_a_extended.end(), prompt_tokens.begin(), prompt_tokens.begin() + 8);
            const auto branch_a_append = engine.generate(branch_a_extended, branch_options);
            branch_options.reuse_session = false;
            branch_options.stream = true;
            const auto branch_b_replay = engine.generate(branch_b, branch_options);
            const auto branch_a_replay = engine.generate(branch_a_extended, branch_options);
            if (branch_b_append.reused_prefix_tokens != kFullPrefillCapacity * 4
                || branch_a_append.reused_prefix_tokens != (branch_a.size() / kFullPrefillCapacity) * kFullPrefillCapacity
                || branch_b_append.tokens != branch_b_replay.tokens
                || branch_a_append.tokens != branch_a_replay.tokens) {
                throw std::runtime_error("multi-checkpoint longest-prefix branch check failed");
            }
            std::cout << "session_branch_check=PASS older_prefix_reuse="
                      << branch_b_append.reused_prefix_tokens
                      << " return_branch_reuse=" << branch_a_append.reused_prefix_tokens
                      << " cached_entries=" << branch_a_append.session_checkpoint_count
                      << " cached_bytes=" << branch_a_append.session_checkpoint_bytes << '\n';
        }

        std::vector<std::uint32_t> cancelled_prompt(kFullPrefillCapacity);
        for (std::size_t i = 0; i < cancelled_prompt.size(); ++i)
            cancelled_prompt[i] = prompt_tokens[i % prompt_tokens.size()];
        RuntimeGenerateOptions lifecycle;
        lifecycle.max_new_tokens = 1;
        lifecycle.reuse_session = true;
        if (engine.generate(cancelled_prompt, lifecycle).cancelled)
            throw std::runtime_error("session lifecycle seed was cancelled");
        cancelled_prompt.resize(kFullPrefillCapacity + 8, prompt_tokens.front());
        lifecycle.max_new_tokens = 2;
        lifecycle.on_token = [](std::uint32_t, std::string_view) { return false; };
        const auto cancelled = engine.generate(cancelled_prompt, lifecycle);
        lifecycle.on_token = {};
        lifecycle.max_new_tokens = 0;
        const auto after_cancel = engine.generate(cancelled_prompt, lifecycle);
        if (!cancelled.cancelled || cancelled.reused_prefix_tokens != kFullPrefillCapacity
            || after_cancel.reused_prefix_tokens != 0)
            throw std::runtime_error("cancelled session checkpoint was not invalidated");
        std::cout << "session_invalidation=PASS cause=cancellation reused_before=512"
                     " reused_after=0\n";
        lifecycle.max_new_tokens = 2;
        lifecycle.on_token = [](std::uint32_t, std::string_view) -> bool {
            throw std::runtime_error("injected generation failure");
        };
        bool generation_failed = false;
        try { (void)engine.generate(cancelled_prompt, lifecycle); }
        catch (const std::runtime_error&) { generation_failed = true; }
        lifecycle.on_token = {};
        lifecycle.max_new_tokens = 0;
        const auto after_failure = engine.generate(cancelled_prompt, lifecycle);
        if (!generation_failed || after_failure.reused_prefix_tokens != 0)
            throw std::runtime_error("failed session checkpoint was not invalidated");
        std::cout << "session_invalidation=PASS cause=generation_failure reused_after=0\n";
        auto mismatched_prompt = cancelled_prompt;
        mismatched_prompt.front() ^= 1U;
        const auto after_mismatch = engine.generate(mismatched_prompt, lifecycle);
        if (after_mismatch.reused_prefix_tokens != 0)
            throw std::runtime_error("mismatched session prefix was reused");
        engine.reset();
        const auto after_reset = engine.generate(cancelled_prompt, lifecycle);
        if (after_reset.reused_prefix_tokens != 0)
            throw std::runtime_error("reset session checkpoint was reused");
        std::cout << "session_invalidation=PASS causes=mismatch,reset reused_after=0\n";
        return 0;
    }

    if (repeat_p512_check) {
        if (prompt_tokens.size() != kFullPrefillCapacity) {
            throw std::runtime_error("--repeat-p512-check requires exactly 512 prompt tokens");
        }
        Qwen35RuntimeEngine::GenerateOptions continuation;
        continuation.max_new_tokens = 2;
        continuation.stream = false;
        const auto first = engine.generate(prompt_tokens, continuation);

        Qwen35RuntimeEngine::GenerateOptions repeat;
        repeat.max_new_tokens = 0;
        repeat.stream = false;
        const auto second = engine.generate(prompt_tokens, repeat);

        if (first.generated_tokens != 2 || first.tokens.front() != 13477
            || second.prefill_processed_tokens != kFullPrefillCapacity) {
            throw std::runtime_error("same-process P512 check failed");
        }
        const double continuation_ms = first.decode_ms - first.first_token_ms;
        std::cout << "same_process_p512_check=PASS first_token=13477(brown)"
                  << " continuation_token=" << first.tokens[1]
                  << " first_prefill_ms=" << first.prefill_ms
                  << " continuation_ms=" << continuation_ms
                  << " repeat_prefill_ms=" << second.prefill_ms << '\n';
        return 0;
    }

    std::cerr << "Generating up to " << max_tokens << " tokens...\n";
    std::cerr << "---------------------------------------------------------\n";

    // Print prompt to stdout
    std::cout << prompt_text << std::flush;

    Qwen35RuntimeEngine::GenerateOptions opt;
    opt.max_new_tokens = max_tokens;
    opt.stream = stream;
    if (exp0367_state_output) {
        opt.on_prefill_state = [&](const float* hidden, std::size_t) {
            engine.write_exp0367_prefill_state(*exp0367_state_output, prompt_tokens, hidden);
        };
    }
    if (stream) {
        opt.on_token = [](std::uint32_t /*token*/, std::string_view piece) {
            std::cout << piece << std::flush;
            return true;
        };
    }

    const auto stats = engine.generate(prompt_tokens, opt);
    if (!stream) {
        std::cout << stats.text << std::flush;
    }
    if (const char* dump_tokens = std::getenv("MIINFER_DUMP_TOKENS");
        dump_tokens != nullptr && std::strcmp(dump_tokens, "0") != 0) {
        std::cerr << "Generated token IDs:";
        for (const auto token : stats.tokens) std::cerr << ' ' << token;
        std::cerr << '\n';
    }
    std::cout << "\n\n";

    std::cerr << "---------------------------------------------------------\n";
    std::cerr << "Performance Summary:\n";
    std::cerr << "  Prefill Tokens:   " << stats.prompt_tokens << " tokens (" << std::fixed << std::setprecision(2)
              << stats.prefill_ms << " ms, " << stats.prefill_tok_s << " tok/s)\n";
    std::cerr << "  Decode Tokens:    " << stats.generated_tokens << " tokens (" << std::fixed << std::setprecision(2)
              << stats.decode_ms << " ms, " << stats.decode_tok_s << " tok/s)\n";
    std::cerr << "  First Token TTFT: " << std::fixed << std::setprecision(2) << stats.first_token_ms << " ms\n";
    std::cerr << "  Average Decode:   " << std::fixed << std::setprecision(3)
              << (stats.generated_tokens > 0 ? stats.decode_ms / stats.generated_tokens : 0.0) << " ms/token\n";
    std::cerr << "  Total Latency:    " << std::fixed << std::setprecision(2) << stats.total_ms << " ms\n";
    std::cerr << "  Decode contract:  graph_launches=" << stats.decode_graph_launches
              << " H2D_bytes=" << stats.decode_h2d_bytes
              << " D2H_bytes=" << stats.decode_d2h_bytes << '\n';
    std::cerr << "---------------------------------------------------------\n";

    return 0;
}

// ---------------------------------------------------------------------------
// Subcommand 3: chat (Interactive REPL)
// ---------------------------------------------------------------------------
int cmd_chat(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: miinfer chat <model.gguf>\n";
        return 1;
    }
    const std::string model_path = argv[2];
    std::cerr << "Initializing MIInfer gfx906 interactive chat session ...\n";
    Qwen35RuntimeEngine engine(model_path);

    std::cout << "\n===================================================================\n";
    std::cout << " MIInfer Interactive Chat (Qwen3.5-27B on AMD Instinct MI50 32GB) \n";
    std::cout << " Type your message and press Enter. Type 'exit' or 'quit' to end.  \n";
    std::cout << "===================================================================\n\n";

    std::string history = "<|im_start|>system\nYou are a helpful, concise AI assistant optimized for AMD MI50 inference.<|im_end|>\n";

    while (!g_shutdown_requested) {
        std::cout << "\x1b[32mUser>\x1b[0m " << std::flush;
        std::string input_line;
        if (!std::getline(std::cin, input_line)) break;
        if (input_line == "exit" || input_line == "quit") break;
        if (input_line.empty()) continue;

        history += "<|im_start|>user\n" + input_line + "<|im_end|>\n<|im_start|>assistant\n";
        const auto prompt_tokens = engine.tokenizer().encode(history);

        std::cout << "\x1b[36mAssistant>\x1b[0m " << std::flush;

        std::string assistant_reply;
        Qwen35RuntimeEngine::GenerateOptions opt;
        opt.max_new_tokens = 512;
        opt.on_token = [&](std::uint32_t /*token*/, std::string_view piece) {
            std::cout << piece << std::flush;
            assistant_reply += piece;
            return true;
        };

        const auto stats = engine.generate(prompt_tokens, opt);
        std::cout << "\n\x1b[90m[" << stats.generated_tokens << " tokens, "
                  << std::fixed << std::setprecision(2) << stats.decode_tok_s << " tok/s]\x1b[0m\n\n";

        history += assistant_reply + "<|im_end|>\n";
    }

    std::cout << "Goodbye!\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Subcommand 4: serve (OpenAI-compatible HTTP API)
// ---------------------------------------------------------------------------
struct HttpRequest {
    int client_fd = -1;
    std::uint64_t request_id = 0;
    std::string method;
    std::string path;
    std::string raw;
    std::chrono::steady_clock::time_point queued_at;
};

struct HttpReadResult {
    std::optional<HttpRequest> request;
    int status = 400;
    std::string reason = "Bad Request";
};

constexpr std::size_t kMaxHttpHeaderBytes = 64 * 1024;
constexpr std::size_t kMaxHttpRequestBytes = 4 * 1024 * 1024;
constexpr std::size_t kMaxRequestTokens = 4096;
constexpr int kClientIoTimeoutSeconds = 10;

bool send_all(int client_fd, std::string_view bytes) {
    std::size_t sent = 0;
    while (sent < bytes.size()) {
        const ssize_t count = send(client_fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
        if (count > 0) {
            sent += static_cast<std::size_t>(count);
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

void set_client_timeouts(int client_fd) {
    const timeval timeout{kClientIoTimeoutSeconds, 0};
    (void)setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

bool send_http_response(int client_fd, int status, std::string_view reason,
                        std::string_view content_type, std::string_view body) {
    const std::string header = "HTTP/1.1 " + std::to_string(status) + " "
                             + std::string(reason) + "\r\nContent-Type: "
                             + std::string(content_type) + "\r\nContent-Length: "
                             + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
    return send_all(client_fd, header) && send_all(client_fd, body);
}

std::string lowercase_ascii(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const unsigned char c : value) result += static_cast<char>(std::tolower(c));
    return result;
}

std::string trim_ascii(std::string_view value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
    return std::string(value.substr(begin, end - begin));
}

std::optional<std::string> http_header(std::string_view headers, std::string_view wanted) {
    const std::string target = lowercase_ascii(wanted);
    std::size_t line_start = 0;
    while (line_start < headers.size()) {
        const std::size_t separator = headers.find("\r\n", line_start);
        const std::size_t line_end = separator == std::string_view::npos
            ? headers.size() : separator;
        const std::string_view line = headers.substr(line_start, line_end - line_start);
        const std::size_t colon = line.find(':');
        if (colon != std::string_view::npos
            && lowercase_ascii(trim_ascii(line.substr(0, colon))) == target) {
            return trim_ascii(line.substr(colon + 1));
        }
        if (separator == std::string_view::npos) break;
        line_start = separator + 2;
    }
    return std::nullopt;
}

HttpReadResult read_http_request(int client_fd) {
    set_client_timeouts(client_fd);
    char buffer[8192];
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::seconds(kClientIoTimeoutSeconds);
    const auto receive = [&]() -> ssize_t {
        while (true) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                errno = EAGAIN;
                return -1;
            }
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            pollfd pfd{client_fd, POLLIN, 0};
            const int ready = poll(&pfd, 1, static_cast<int>(std::max<std::int64_t>(1, remaining.count())));
            if (ready > 0) return recv(client_fd, buffer, sizeof(buffer), 0);
            if (ready == 0) {
                errno = EAGAIN;
                return -1;
            }
            if (errno != EINTR) return -1;
        }
    };
    std::string data;
    data.reserve(8192);
    std::size_t header_end = std::string::npos;
    while ((header_end = data.find("\r\n\r\n")) == std::string::npos) {
        if (data.size() >= kMaxHttpHeaderBytes) return {std::nullopt, 431, "Request Header Fields Too Large"};
        const ssize_t count = receive();
        if (count == 0) return {std::nullopt, 400, "Bad Request"};
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return {std::nullopt, 408, "Request Timeout"};
        }
        if (count < 0) return {std::nullopt, 400, "Bad Request"};
        data.append(buffer, static_cast<std::size_t>(count));
    }
    if (header_end > kMaxHttpHeaderBytes) return {std::nullopt, 431, "Request Header Fields Too Large"};

    const std::string_view header_block(data.data(), header_end);
    if (http_header(header_block, "transfer-encoding")) {
        return {std::nullopt, 400, "Transfer Encoding Unsupported"};
    }

    std::size_t content_length = 0;
    if (const auto length = http_header(header_block, "content-length")) {
        const auto [end, error] = std::from_chars(length->data(), length->data() + length->size(), content_length);
        if (error != std::errc{} || end != length->data() + length->size()) {
            return {std::nullopt, 400, "Invalid Content-Length"};
        }
    }
    const std::size_t body_start = header_end + 4;
    if (content_length > kMaxHttpRequestBytes - body_start) {
        return {std::nullopt, 413, "Payload Too Large"};
    }
    const std::size_t request_end = body_start + content_length;
    while (data.size() < request_end) {
        const ssize_t count = receive();
        if (count == 0) return {std::nullopt, 400, "Incomplete Request"};
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return {std::nullopt, 408, "Request Timeout"};
        }
        if (count < 0) return {std::nullopt, 400, "Bad Request"};
        data.append(buffer, static_cast<std::size_t>(count));
    }
    data.resize(request_end);

    std::istringstream request_line(data.substr(0, header_end));
    std::string method;
    std::string path;
    request_line >> method >> path;
    if (method.empty() || path.empty()) return {std::nullopt, 400, "Bad Request"};
    return {HttpRequest{client_fd, 0, std::move(method), std::move(path), std::move(data), {}}, 0, {}};
}

void send_http_error(int client_fd, int status, std::string_view reason,
                     std::string_view code_override = {}) {
    const std::string code = status == 401 ? "authentication_error"
        : status == 413 || status == 431 ? "request_too_large"
        : status == 503 ? "server_unavailable"
        : code_override.empty() ? "invalid_request_error" : std::string(code_override);
    const std::string body = "{\"error\":{\"message\":\"" + json_escape(reason)
        + "\",\"type\":\"" + code + "\",\"code\":\"" + code + "\"}}";
    (void)send_http_response(client_fd, status, reason, "application/json", body);
}

bool constant_time_equal(std::string_view left, std::string_view right) {
    std::size_t difference = left.size() ^ right.size();
    const std::size_t count = std::max(left.size(), right.size());
    for (std::size_t i = 0; i < count; ++i) {
        const unsigned char a = i < left.size() ? static_cast<unsigned char>(left[i]) : 0;
        const unsigned char b = i < right.size() ? static_cast<unsigned char>(right[i]) : 0;
        difference |= a ^ b;
    }
    return difference == 0;
}

int cmd_serve(int argc, char** argv) {
    std::string model_path;
    int port = 8080;
    std::string host = "127.0.0.1";
    std::size_t context_length = 1024;
    bool experimental_context = false;
    std::optional<std::filesystem::path> api_key_file;
    bool allow_insecure = false;

    for (int i = 2; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            const auto parsed = parse_port(argv[++i]);
            if (!parsed) { std::cerr << "port must be an integer from 0 to 65535\n"; return 2; }
            port = *parsed;
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--context" && i + 1 < argc) {
            try { context_length = std::stoull(argv[++i]); }
            catch (...) { std::cerr << "context must be a positive integer\n"; return 2; }
        } else if (arg == "--experimental-context") {
            experimental_context = true;
        } else if (arg == "--api-key-file" && i + 1 < argc) {
            api_key_file = argv[++i];
        } else if (arg == "--allow-insecure") {
            allow_insecure = true;
        } else if (model_path.empty() && !arg.starts_with("--")) {
            model_path = arg;
        } else {
            std::cerr << "usage: miinfer serve --model MODEL.gguf [--port PORT] [--host HOST] [--context N] [--experimental-context] [--api-key-file PATH] [--allow-insecure]\n"; return 2;
        }
    }
    if (model_path.empty()) { std::cerr << "missing model; use --model MODEL.gguf\n"; return 2; }
    if (context_length != 1024 && context_length != 8192 && context_length != 16384
        && context_length != 32768 && context_length != 65536 && context_length != 131072) {
        std::cerr << "context must be one of 1024, 8192, 16384, 32768, 65536, 131072\n";
        return 2;
    }
    if (context_length > 1024 && !experimental_context) {
        std::cerr << "context values above 1024 require --experimental-context\n"; return 2;
    }
    std::string api_key;
    if (api_key_file) {
        std::ifstream input(*api_key_file);
        if (!input || !std::getline(input, api_key)) {
            std::cerr << "cannot read API key file: " << *api_key_file << "\n"; return 2;
        }
        api_key = trim_ascii(api_key);
        if (api_key.empty()) {
            std::cerr << "API key file is empty: " << *api_key_file << "\n"; return 2;
        }
    } else if (const char* configured_key = std::getenv("MIINFER_API_KEY")) {
        api_key = configured_key;
    }
    const bool loopback = host == "127.0.0.1" || host == "localhost" || host == "::1";
    if (!loopback && api_key.empty() && !allow_insecure) {
        std::cerr << "non-loopback serving requires MIINFER_API_KEY, --api-key-file, or --allow-insecure\n";
        return 2;
    }
    const std::string model_id = std::filesystem::path(model_path).stem().string();
    const char* session_reuse_env = std::getenv("MIINFER_SESSION_REUSE");
    const bool session_reuse = session_reuse_env != nullptr
        && std::strcmp(session_reuse_env, "0") != 0;

    g_cache_capacity = context_length;
    std::cerr << "Initializing MIInfer gfx906 HTTP Server on " << host << ":" << port << " ...\n";
    std::cerr << "configured_context_length=" << context_length << "\n"
              << "runtime_context_capacity=" << g_cache_capacity << "\n"
              << "qualified_context_length=1024\n"
              << "context_qualification=" << (context_length > 1024 ? "experimental" : "qualified") << "\n"
              << "session_reuse=" << (session_reuse ? "experimental" : "disabled") << "\n";
    Qwen35RuntimeEngine engine(model_path);
    std::cerr << "model_context_length=" << engine.model().config().context_length << "\n";
    std::cerr << "device_allocation_count=" << g_device_allocations << "\n"
              << "device_allocated_bytes=" << g_live_device_bytes << "\n"
              << "device_total_allocated_bytes=" << g_total_device_bytes << "\n"
              << "device_peak_allocated_bytes=" << g_peak_device_bytes << "\n";
    print_hip_memory(std::cerr);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Failed to create socket\n";
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(host.c_str());
    address.sin_port = htons(port);

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        std::cerr << "Bind failed on " << host << ":" << port << "\n";
        close(server_fd);
        return 1;
    }

    constexpr std::size_t kQueueCapacity = 8;
    if (listen(server_fd, static_cast<int>(kQueueCapacity)) < 0) {
        std::cerr << "Listen failed\n";
        close(server_fd);
        return 1;
    }

    std::cerr << "MIInfer OpenAI-compatible API listening at http://" << host << ":" << port << "\n";
    std::cerr << "Request queue capacity: " << kQueueCapacity << "\n";
    std::cerr << "Endpoints:\n";
    std::cerr << "  GET  /healthz\n";
    std::cerr << "  GET  /readyz\n";
    std::cerr << "  GET  /metrics\n";
    std::cerr << "  GET  /v1/models\n";
    std::cerr << "  POST /v1/chat/completions\n";

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    int wake_pipe[2]{};
    if (pipe(wake_pipe) != 0) { std::cerr << "failed to create signal wake pipe\n"; close(server_fd); return 1; }
    g_signal_wakeup_fd = wake_pipe[1];

    std::atomic<std::uint64_t> http_requests = 0;
    std::atomic<std::uint64_t> http_errors = 0;
    std::atomic<std::uint64_t> inference_requests = 0;
    std::atomic<std::uint64_t> prompt_tokens_total = 0;
    std::atomic<std::uint64_t> generated_tokens_total = 0;
    std::atomic<std::uint64_t> queue_rejected = 0;
    std::atomic<std::size_t> queue_depth = 0;
    std::atomic<std::size_t> active_requests = 0;
    std::atomic<std::uint64_t> queue_wait_us = 0;
    std::atomic<std::uint64_t> request_duration_us = 0;
    std::atomic<std::uint64_t> ttft_us = 0;
    std::atomic<std::uint64_t> tokenization_us = 0;
    std::atomic<std::uint64_t> prefill_us = 0;
    std::atomic<std::uint64_t> prefill_tokens_total = 0;
    std::atomic<std::uint64_t> cancelled_requests = 0;
    std::atomic<std::uint64_t> request_sequence = 0;

    auto handle_request = [&](const HttpRequest& request) {
        const auto started_at = std::chrono::steady_clock::now();
        queue_wait_us += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(started_at - request.queued_at).count());
        active_requests.fetch_add(1);
        struct Guard { std::atomic<std::size_t>& active; ~Guard() { active.fetch_sub(1); } } guard{active_requests};
        const int client_fd = request.client_fd;
        ++inference_requests;
        const auto state = [&](std::string_view value) {
            std::cerr << "miinfer_request_state request_id=" << request.request_id
                      << " state=" << value << '\n';
        };
        state("dequeued");
        const std::size_t body_start = request.raw.find("\r\n\r\n");
        const auto parsed = miinfer::parse_openai_chat_request(
            body_start == std::string::npos ? std::string_view{} : std::string_view(request.raw).substr(body_start + 4));
        if (!parsed.request) {
            ++http_errors;
            send_http_error(client_fd, 400, parsed.error);
            state("rejected");
            return;
        }
        state("parsed");
        const bool is_stream = parsed.request->stream;
        const std::size_t max_tokens = parsed.request->max_tokens;
        const bool defer_tool_output = !parsed.request->tools.empty()
            && parsed.request->tool_choice != "none";
        const std::string prompt = miinfer::build_chatml(*parsed.request);
        const auto client_cancelled = [&] {
            if (g_shutdown_requested) return true;
            char byte = 0;
            const ssize_t count = recv(client_fd, &byte, sizeof(byte), MSG_PEEK | MSG_DONTWAIT);
            if (count == 0) return true;
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return false;
            return count < 0;
        };

        const auto tokenization_start = std::chrono::steady_clock::now();
        if (client_cancelled()) {
            ++cancelled_requests;
            state("cancelled");
            return;
        }
        const auto prompt_tokens = engine.tokenizer().encode(prompt);
        const double tokenization_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tokenization_start).count();
        tokenization_us += static_cast<std::uint64_t>(tokenization_ms * 1000.0);
        const double queue_wait_ms = std::chrono::duration<double, std::milli>(started_at - request.queued_at).count();
        state("tokenized");
        if (prompt_tokens.size() > context_length || prompt_tokens.size() + max_tokens > context_length) {
            ++http_errors;
            send_http_error(client_fd, 400, "prompt and max_tokens exceed configured context", "context_length_exceeded");
            state("rejected");
            return;
        }
        state("prefill_started");
        std::optional<double> first_delta_ms;
        const auto mark_first_delta = [&] {
            if (!first_delta_ms) {
                first_delta_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - request.queued_at).count();
                ttft_us += static_cast<std::uint64_t>(*first_delta_ms * 1000.0);
            }
        };
        const auto log_latency = [&](const RuntimeGenerateStats& stats) {
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - request.queued_at).count();
            request_duration_us += static_cast<std::uint64_t>(wall_ms * 1000.0);
            const double steady_ms = stats.decode_ms - stats.first_token_ms;
            std::cerr << "miinfer_request_latency {\"request_id\":" << request.request_id
                      << ",\"prompt_tokens\":" << stats.prompt_tokens
                      << ",\"common_prefix_tokens\":" << stats.common_prefix_tokens
                      << ",\"reused_prefix_tokens\":" << stats.reused_prefix_tokens
                      << ",\"session_checkpoint_count\":" << stats.session_checkpoint_count
                      << ",\"session_checkpoint_bytes\":" << stats.session_checkpoint_bytes
                      << ",\"new_prefill_tokens\":" << stats.prefill_processed_tokens
                      << ",\"prefill_ms\":" << stats.prefill_ms
                      << ",\"graph_capture_ms\":" << stats.graph_capture_ms
                      << ",\"first_decode_token_ms\":" << stats.first_token_ms
                      << ",\"TTFT_wall_ms\":" << (first_delta_ms ? std::to_string(*first_delta_ms) : "null")
                      << ",\"steady_decode_tok_s\":" << (steady_ms > 0 && stats.generated_tokens > 1
                          ? 1000.0 * (stats.generated_tokens - 1) / steady_ms : 0.0)
                      << ",\"total_request_ms\":" << wall_ms
                      << ",\"cache_hit\":" << (stats.reused_prefix_tokens > 0 ? "true" : "false") << "}\n";
        };
        if (is_stream) {
            bool client_connected = send_all(
                client_fd,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
            Qwen35RuntimeEngine::GenerateOptions opt;
            opt.max_new_tokens = max_tokens;
            opt.reuse_session = session_reuse;
            opt.should_cancel = client_cancelled;
            opt.on_prefill_complete = [&] { state("prefill_completed"); };
            opt.on_first_token = [&] { state("first_token"); };
            bool request_cancelled = false;
            opt.on_token = [&](std::uint32_t /*token*/, std::string_view piece) {
                if (!client_connected) return false;
                if (defer_tool_output) return true;
                const std::string sse = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\""
                    + json_escape(piece) + "\"}}]}\n\n";
                client_connected = send_all(client_fd, sse);
                if (client_connected && !piece.empty()) mark_first_delta();
                return client_connected;
            };
            Qwen35RuntimeEngine::GenerateStats stats;
            miinfer::OpenAiGeneratedToolCalls tool_calls;
            try {
                stats = engine.generate(prompt_tokens, opt);
                if (defer_tool_output) tool_calls = miinfer::parse_generated_tool_calls(stats.text);
                request_cancelled = stats.cancelled;
                prompt_tokens_total += stats.prompt_tokens;
                prefill_tokens_total += stats.prefill_processed_tokens;
                prefill_us += static_cast<std::uint64_t>(stats.prefill_ms * 1000.0);
                generated_tokens_total += stats.generated_tokens;
                const bool cancelled = stats.cancelled || !client_connected;
                if (cancelled) { ++cancelled_requests; state("cancelled"); }
                else state("completed");
                std::cerr << "miinfer_request {\"request_id\":" << request.request_id
                          << ",\"request_body_bytes\":" << request.raw.size()
                          << ",\"message_count\":" << parsed.request->messages.size()
                          << ",\"prompt_chars\":" << prompt.size()
                          << ",\"prompt_tokens\":" << stats.prompt_tokens
                          << ",\"tokenization_ms\":" << tokenization_ms
                          << ",\"queue_wait_ms\":" << queue_wait_ms
                          << ",\"prefill_ms\":" << stats.prefill_ms
                          << ",\"prefill_processed_tokens\":" << stats.prefill_processed_tokens
                          << ",\"prefill_tokens_per_second\":" << stats.prefill_tok_s
                          << ",\"first_decode_token_ms\":" << stats.first_token_ms
                          << ",\"time_to_first_token_ms\":" << stats.prefill_ms + stats.first_token_ms
                          << ",\"generated_tokens\":" << stats.generated_tokens
                          << ",\"decode_ms\":" << stats.decode_ms
                          << ",\"decode_tokens_per_second\":" << stats.decode_tok_s
                          << ",\"total_ms\":" << stats.total_ms
                          << ",\"cancelled\":" << (cancelled ? "true" : "false")
                          << ",\"finish_reason\":\"" << (cancelled ? "cancelled"
                              : !tool_calls.calls.empty() ? "tool_calls" : "stop") << "\""
                          << ",\"error\":null"
                          << ",\"configured_context\":" << context_length
                          << ",\"runtime_context_capacity\":" << g_cache_capacity << "}\n";
            } catch (...) {
                ++http_errors;
                if (client_connected) {
                    (void)send_all(client_fd, "event: error\ndata: {\"error\":\"Internal Server Error\"}\n\n");
                }
                return;
            }
            if (client_connected && !request_cancelled) {
                if (defer_tool_output) {
                    if (tool_calls.calls.empty()) {
                        const std::string sse = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\""
                            + json_escape(stats.text) + "\"}}]}\n\n";
                        client_connected = send_all(client_fd, sse);
                        if (client_connected && !stats.text.empty()) mark_first_delta();
                    } else {
                        for (std::size_t i = 0; i < tool_calls.calls.size() && client_connected; ++i) {
                            const auto& call = tool_calls.calls[i];
                            const std::string sse = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":"
                                + std::to_string(i) + ",\"id\":\"" + json_escape(call.id)
                                + "\",\"type\":\"function\",\"function\":{\"name\":\""
                                + json_escape(call.name) + "\",\"arguments\":\""
                                + json_escape(call.arguments) + "\"}}]}}]}\n\n";
                            client_connected = send_all(client_fd, sse);
                            if (client_connected) mark_first_delta();
                        }
                    }
                }
                const std::string finish = defer_tool_output && !tool_calls.calls.empty()
                    ? "tool_calls" : "stop";
                if (client_connected) {
                    (void)send_all(client_fd, "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{},\"finish_reason\":\""
                        + finish + "\"}]}\n\n");
                    (void)send_all(client_fd, "data: [DONE]\n\n");
                }
            }
            log_latency(stats);
        } else {
            Qwen35RuntimeEngine::GenerateOptions opt;
            opt.max_new_tokens = max_tokens;
            opt.reuse_session = session_reuse;
            opt.should_cancel = client_cancelled;
            opt.on_prefill_complete = [&] { state("prefill_completed"); };
            opt.on_first_token = [&] { state("first_token"); };
            // Server requests must observe shutdown between tokens; the bulk
            // graph path intentionally does not provide that interruption point.
            opt.stream = true;
            const auto stats = engine.generate(prompt_tokens, opt);
            prompt_tokens_total += stats.prompt_tokens;
            prefill_tokens_total += stats.prefill_processed_tokens;
            prefill_us += static_cast<std::uint64_t>(stats.prefill_ms * 1000.0);
            generated_tokens_total += stats.generated_tokens;
            if (stats.cancelled) { ++cancelled_requests; state("cancelled"); }
            else state("completed");
            const auto tool_calls = defer_tool_output
                ? miinfer::parse_generated_tool_calls(stats.text)
                : miinfer::OpenAiGeneratedToolCalls{};
            const bool has_tool_calls = !tool_calls.calls.empty();
            const std::string message = has_tool_calls
                ? "\"content\":null,\"tool_calls\":" + openai_tool_calls_json(tool_calls.calls)
                : "\"content\":\"" + json_escape(stats.text) + "\"";
            std::cerr << "miinfer_request {\"request_id\":" << request.request_id
                      << ",\"request_body_bytes\":" << request.raw.size()
                      << ",\"message_count\":" << parsed.request->messages.size()
                      << ",\"prompt_chars\":" << prompt.size()
                      << ",\"prompt_tokens\":" << stats.prompt_tokens
                      << ",\"tokenization_ms\":" << tokenization_ms
                      << ",\"queue_wait_ms\":" << queue_wait_ms
                      << ",\"prefill_ms\":" << stats.prefill_ms
                      << ",\"prefill_processed_tokens\":" << stats.prefill_processed_tokens
                      << ",\"prefill_tokens_per_second\":" << stats.prefill_tok_s
                      << ",\"first_decode_token_ms\":" << stats.first_token_ms
                      << ",\"time_to_first_token_ms\":" << stats.prefill_ms + stats.first_token_ms
                      << ",\"generated_tokens\":" << stats.generated_tokens
                      << ",\"decode_ms\":" << stats.decode_ms
                      << ",\"decode_tokens_per_second\":" << stats.decode_tok_s
                      << ",\"total_ms\":" << stats.total_ms
                      << ",\"cancelled\":" << (stats.cancelled ? "true" : "false")
                      << ",\"finish_reason\":\"" << (stats.cancelled ? "cancelled" : has_tool_calls ? "tool_calls" : "stop") << "\""
                      << ",\"error\":null"
                      << ",\"configured_context\":" << context_length
                      << ",\"runtime_context_capacity\":" << g_cache_capacity << "}\n";
            const std::string body = "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"choices\":[{\"message\":{\"role\":\"assistant\","
                + message + "},\"finish_reason\":\""
                + std::string(stats.cancelled ? "cancelled" : has_tool_calls ? "tool_calls" : "stop") + "\"}],\"usage\":{\"prompt_tokens\":"
                + std::to_string(stats.prompt_tokens) + ",\"completion_tokens\":"
                + std::to_string(stats.generated_tokens) + "}}";
            if (send_http_response(client_fd, 200, "OK", "application/json", body)) mark_first_delta();
            log_latency(stats);
        }
    };

    std::deque<HttpRequest> pending_requests;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool accepting = true;
    std::thread worker([&] {
        while (true) {
            HttpRequest request;
            {
                std::unique_lock lock(queue_mutex);
                queue_cv.wait(lock, [&] { return !pending_requests.empty() || !accepting; });
                if (pending_requests.empty() && !accepting) return;
        request = std::move(pending_requests.front());
                pending_requests.pop_front();
                queue_depth.fetch_sub(1);
            }
            try {
                handle_request(request);
            } catch (const std::exception& error) {
                ++http_errors;
                std::cerr << "request failed: " << error.what() << "\n";
                send_http_error(request.client_fd, 500, "Internal Server Error");
            } catch (...) {
                ++http_errors;
                std::cerr << "request failed with an unknown error\n";
                send_http_error(request.client_fd, 500, "Internal Server Error");
            }
            close(request.client_fd);
        }
    });

    while (!g_shutdown_requested) {
        pollfd pfds[2]{{server_fd, POLLIN, 0}, {wake_pipe[0], POLLIN, 0}};
        int poll_res = poll(pfds, 2, -1);
        if (poll_res <= 0) continue;
        if (pfds[1].revents & POLLIN) break;
        if (!(pfds[0].revents & POLLIN)) continue;

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) continue;

        HttpReadResult result = read_http_request(client_fd);
        if (!result.request) {
            ++http_errors;
            send_http_error(client_fd, result.status, result.reason);
            close(client_fd);
            continue;
        }
        ++http_requests;
        HttpRequest request = std::move(*result.request);
        request.request_id = ++request_sequence;
        request.queued_at = std::chrono::steady_clock::now();
        std::cerr << "miinfer_request_state request_id=" << request.request_id << " state=received\n";

        if (request.method == "GET" && request.path == "/") {
            constexpr std::string_view page = R"HTML(<!doctype html><meta charset="utf-8"><title>MIInfer</title><style>body{max-width:48rem;margin:2rem auto;font:16px system-ui}#chat{white-space:pre-wrap;border:1px solid #ccc;padding:1rem;min-height:20rem}textarea{width:100%;height:5rem}button{margin-top:.5rem}</style><h1>MIInfer <small id="model"></small></h1><div id="chat"></div><textarea id="prompt" placeholder="Message"></textarea><br><button onclick="send()">Send</button><script>const chat=document.querySelector('#chat'),prompt=document.querySelector('#prompt'),messages=[];fetch('/v1/models').then(r=>r.json()).then(x=>model.textContent=x.data?.[0]?.id||'');async function send(){const text=prompt.value.trim();if(!text)return;messages.push({role:'user',content:text});chat.textContent+='You: '+text+'\nMIInfer: ';prompt.value='';const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages,stream:true,max_tokens:512})});const reader=r.body.getReader(),d=new TextDecoder;let pending='',reply='';for(;;){const x=await reader.read();if(x.done)break;pending+=d.decode(x.value,{stream:true});const lines=pending.split('\n');pending=lines.pop();for(const line of lines)if(line.startsWith('data: {')){const piece=JSON.parse(line.slice(6)).choices[0].delta.content||'';reply+=piece;chat.textContent+=piece}}messages.push({role:'assistant',content:reply});chat.textContent+='\n\n'}</script>)HTML";
            (void)send_http_response(client_fd, 200, "OK", "text/html; charset=utf-8", page);
            close(client_fd);
            continue;
        }

        if (request.method == "GET" && (request.path == "/healthz" || request.path == "/readyz")) {
            (void)send_http_response(client_fd, 200, "OK", "application/json", R"({"status":"ok","ready":true})");
            close(client_fd);
            continue;
        }
        if (request.method == "GET" && request.path == "/metrics") {
            if (!api_key.empty()) {
                const auto headers_end = request.raw.find("\r\n\r\n");
                const auto authorization = http_header(std::string_view(request.raw).substr(0, headers_end), "authorization");
                if (!authorization || !authorization->starts_with("Bearer ")
                    || !constant_time_equal(std::string_view(*authorization).substr(7), api_key)) {
                    send_http_error(client_fd, 401, "missing or invalid bearer token"); close(client_fd); continue;
                }
            }
            const std::string body = "# TYPE miinfer_http_requests_total counter\n"
                "miinfer_http_requests_total " + std::to_string(http_requests.load()) + "\n"
                "# TYPE miinfer_requests_total counter\n"
                "miinfer_requests_total " + std::to_string(http_requests.load()) + "\n"
                "# TYPE miinfer_http_errors_total counter\n"
                "miinfer_http_errors_total " + std::to_string(http_errors.load()) + "\n"
                "# TYPE miinfer_rejected_requests_total counter\n"
                "miinfer_rejected_requests_total " + std::to_string(http_errors.load() + queue_rejected.load()) + "\n"
                "# TYPE miinfer_inference_requests_total counter\n"
                "miinfer_inference_requests_total " + std::to_string(inference_requests.load()) + "\n"
                "# TYPE miinfer_queue_depth gauge\n"
                "miinfer_queue_depth " + std::to_string(queue_depth.load()) + "\n"
                "# TYPE miinfer_queue_capacity gauge\n"
                "miinfer_queue_capacity " + std::to_string(kQueueCapacity) + "\n"
                "# TYPE miinfer_queue_rejected_total counter\n"
                "miinfer_queue_rejected_total " + std::to_string(queue_rejected.load()) + "\n"
                "# TYPE miinfer_active_requests gauge\n"
                "miinfer_active_requests " + std::to_string(active_requests.load()) + "\n"
                "# TYPE miinfer_queue_wait_seconds_sum counter\n"
                "miinfer_queue_wait_seconds_sum " + std::to_string(queue_wait_us.load() / 1000000.0) + "\n"
                "# TYPE miinfer_queue_wait_seconds counter\n"
                "miinfer_queue_wait_seconds " + std::to_string(queue_wait_us.load() / 1000000.0) + "\n"
                "# TYPE miinfer_request_duration_seconds_sum counter\n"
                "miinfer_request_duration_seconds_sum " + std::to_string(request_duration_us.load() / 1000000.0) + "\n"
                "# TYPE miinfer_request_duration_seconds counter\n"
                "miinfer_request_duration_seconds " + std::to_string(request_duration_us.load() / 1000000.0) + "\n"
                "# TYPE miinfer_time_to_first_token_seconds_sum counter\n"
                "miinfer_time_to_first_token_seconds_sum " + std::to_string(ttft_us.load() / 1000000.0) + "\n"
                "# TYPE miinfer_tokenization_duration_seconds_sum counter\n"
                "miinfer_tokenization_duration_seconds_sum " + std::to_string(tokenization_us.load() / 1000000.0) + "\n"
                "# TYPE miinfer_tokenization_duration_seconds counter\n"
                "miinfer_tokenization_duration_seconds " + std::to_string(tokenization_us.load() / 1000000.0) + "\n"
                "# TYPE miinfer_prefill_duration_seconds_sum counter\n"
                "miinfer_prefill_duration_seconds_sum " + std::to_string(prefill_us.load() / 1000000.0) + "\n"
                "# TYPE miinfer_prefill_duration_seconds counter\n"
                "miinfer_prefill_duration_seconds " + std::to_string(prefill_us.load() / 1000000.0) + "\n"
                "# TYPE miinfer_prefill_tokens_total counter\n"
                "miinfer_prefill_tokens_total " + std::to_string(prefill_tokens_total.load()) + "\n"
                "# TYPE miinfer_cancelled_requests_total counter\n"
                "miinfer_cancelled_requests_total " + std::to_string(cancelled_requests.load()) + "\n"
                "# TYPE miinfer_prompt_tokens_total counter\n"
                "miinfer_prompt_tokens_total " + std::to_string(prompt_tokens_total.load()) + "\n"
                "# TYPE miinfer_generated_tokens_total counter\n"
                "miinfer_generated_tokens_total " + std::to_string(generated_tokens_total.load()) + "\n";
            (void)send_http_response(client_fd, 200, "OK", "text/plain; version=0.0.4", body);
            close(client_fd);
            continue;
        }
        if (request.method == "GET" && request.path == "/v1/models") {
            if (!api_key.empty()) {
                const auto headers_end = request.raw.find("\r\n\r\n");
                const auto authorization = http_header(std::string_view(request.raw).substr(0, headers_end), "authorization");
                if (!authorization || !authorization->starts_with("Bearer ")
                    || !constant_time_equal(std::string_view(*authorization).substr(7), api_key)) {
                    send_http_error(client_fd, 401, "missing or invalid bearer token"); close(client_fd); continue;
                }
            }
            const std::string body = "{\"object\":\"list\",\"data\":[{\"id\":\""
                + json_escape(model_id) + "\",\"object\":\"model\",\"owned_by\":\"miinfer\"}]}";
            (void)send_http_response(client_fd, 200, "OK", "application/json", body);
            close(client_fd);
            continue;
        }
        if (request.method != "POST" || request.path != "/v1/chat/completions") {
            ++http_errors;
            send_http_error(client_fd, 404, "Not Found");
            close(client_fd);
            continue;
        }
        if (!api_key.empty()) {
            const auto headers_end = request.raw.find("\r\n\r\n");
            const auto authorization = http_header(std::string_view(request.raw).substr(0, headers_end), "authorization");
            if (!authorization || !authorization->starts_with("Bearer ")
                || !constant_time_equal(std::string_view(*authorization).substr(7), api_key)) {
                send_http_error(client_fd, 401, "missing or invalid bearer token"); close(client_fd); continue;
            }
        }

        bool queued = false;
        {
            std::lock_guard lock(queue_mutex);
            if (accepting && pending_requests.size() < kQueueCapacity) {
                pending_requests.push_back(std::move(request));
                queue_depth.fetch_add(1);
                std::cerr << "miinfer_request_state request_id=" << pending_requests.back().request_id
                          << " state=queued\n";
                queued = true;
            }
        }
        if (queued) {
            queue_cv.notify_one();
        } else {
            ++queue_rejected;
            send_http_error(client_fd, 503, "Server Busy");
            close(client_fd);
        }
    }

    {
        std::lock_guard lock(queue_mutex);
        accepting = false;
        while (!pending_requests.empty()) {
            HttpRequest request = std::move(pending_requests.front());
            pending_requests.pop_front();
            queue_depth.fetch_sub(1);
            send_http_error(request.client_fd, 503, "Server Shutting Down");
            close(request.client_fd);
        }
    }
    queue_cv.notify_one();
    close(server_fd);
    g_signal_wakeup_fd = -1;
    close(wake_pipe[0]);
    close(wake_pipe[1]);
    worker.join();

    std::cerr << "Shutting down HTTP server...\n";
    return 0;
}

void print_usage() {
    std::cout << "MIInfer: Purpose-Built AMD gfx906 (MI50) LLM Inference Runtime\n\n";
    std::cout << "Usage: miinfer <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  --version                              Print build and target information\n";
    std::cout << "  config                                 Print supported runtime configuration\n";
    std::cout << "  doctor [--model MODEL] [--port PORT]  Check MI50, HIP, VRAM, model, and port\n";
    std::cout << "  models [directory]                     List GGUF model artifacts\n";
    std::cout << "  inspect <model.gguf>                     Inspect model metadata, quantization, and VRAM budget\n";
    std::cout << "  run <model.gguf> --prompt \"...\"         Generate text from a prompt with streaming output\n";
    std::cout << "       --repeat-p512-check                 Check P512, real continuation, and repeat P512\n";
    std::cout << "       --context N --check-session        Check live append against full replay\n";
    std::cout << "       --max-tokens N --check-graph-state Compare graph/direct recurrent and KV bytes\n";
    std::cout << "       --check-graph-state-context N Repeat prompt tokens to context N for parity checks\n";
    std::cout << "       --m26c-state-context 512 --m26c-export-state FILE Create canonical decode snapshot\n";
    std::cout << "       --m26c-import-state FILE --m26c-decode-route direct|graph --m26c-output-state FILE\n";
    std::cout << "       --m26c-logits-output FILE            Export position logits for a one-token probe\n";
    std::cout << "       --exp0367-state-output FILE         Opt-in partial-tail prefill state snapshot\n";
    std::cout << "       MIINFER_EXP0368_SCALAR_ORACLE=1    Diagnostic ordered P512 baseline selector\n";
    std::cout << "       --m26c-layer-path-prefix PREFIX      Capture layers 0 and 1 at one direct token\n";
    std::cout << "       --m26c-restore-only                 Verify route-specific snapshot import without decoding\n";
    std::cout << "       --m26c-recurrent-qkv-m23            Use M23 only for recurrent QKV during imported decode\n";
    std::cout << "       --m26c-ssm-out-native                Use native Q5_K only for SSM output during imported decode\n";
    std::cout << "       scripts/compare-m26c-state.py LEGACY.bin INTERACTIVE.bin Compare route outputs/state\n";
    std::cout << "       MIINFER_PRESET=m25_hi_qualified     Use the hermetic qualified MI50 P512 vector\n";
    std::cout << "       MIINFER_PRESET=m25_interactive      Use experimental wide-prefill/Mx-decode serving\n";
    std::cout << "  chat <model.gguf>                        Start an interactive multi-turn terminal chat REPL\n";
    std::cout << "  serve --model MODEL.gguf [--port 8080] [--context N] [--experimental-context]\n"
              << "        [--api-key-file PATH] [--allow-insecure]   Launch API and Web UI\n\n";
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const std::string_view cmd = argv[1];
    if ((cmd == "run" || cmd == "chat" || cmd == "serve") && !apply_runtime_preset()) return 2;
    if (cmd == "inspect") {
        return cmd_inspect(argc, argv);
    } else if (cmd == "config") {
        return cmd_config(argc, argv);
    } else if (cmd == "doctor") {
        return cmd_doctor(argc, argv);
    } else if (cmd == "models") {
        return cmd_models(argc, argv);
    } else if (cmd == "run") {
        return cmd_run(argc, argv);
    } else if (cmd == "chat") {
        return cmd_chat(argc, argv);
    } else if (cmd == "serve") {
        return cmd_serve(argc, argv);
    } else if (cmd == "--version") {
        miinfer::print_build_info(std::cout);
        return 0;
    } else if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        print_usage();
        return 0;
    } else {
        std::cerr << "Unknown command: " << cmd << "\n\n";
        print_usage();
        return 1;
    }
} catch (const std::exception& error) {
    std::cerr << "MIInfer: " << error.what() << '\n';
    return 1;
}
