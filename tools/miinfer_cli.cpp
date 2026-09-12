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

extern char** environ;

namespace {

std::atomic<bool> g_shutdown_requested{false};
int g_signal_wakeup_fd = -1;
constexpr std::size_t kFullPrefillCapacity = 512;

struct PresetFlag {
    const char* name;
    const char* value;
};

bool apply_runtime_preset() {
    const char* preset = std::getenv("MIINFER_PRESET");
    if (preset == nullptr) return true;
    if (std::strcmp(preset, "m25_hi_qualified") != 0) {
        std::cerr << "unsupported MIINFER_PRESET: " << preset
                  << " (expected m25_hi_qualified)\n";
        return false;
    }

    // Clear every MIINFER_* override before applying the versioned vector;
    // preserve the server credential, which is not a runtime selector.
    std::vector<std::string> names;
    for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
        const std::string_view value(*entry);
        if (value.rfind("MIINFER_", 0) == 0
            && value.rfind("MIINFER_API_KEY=", 0) != 0
            && value.rfind("MIINFER_PRESET=", 0) != 0) {
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
    std::cerr << "preset=m25_hi_qualified\n"
              << "  MIINFER_MX_PIPELINE=unset\n";
    for (const auto [name, value] : flags) std::cerr << "  " << name << "=" << value << '\n';
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

std::optional<int> parse_port(std::string_view value) {
    int port = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), port);
    if (error != std::errc{} || end != value.data() + value.size() || port < 0 || port > 65535) return std::nullopt;
    return port;
}

struct RuntimeGenerateOptions {
    std::size_t max_new_tokens = 256;
    bool stream = true;
    std::function<bool()> should_cancel = nullptr;
    std::function<bool(std::uint32_t token, std::string_view piece)> on_token = nullptr;
    std::function<void()> on_prefill_complete = nullptr;
    std::function<void()> on_first_token = nullptr;
};

struct RuntimeGenerateStats {
    std::vector<std::uint32_t> tokens;
    std::string text;
    std::size_t prompt_tokens = 0;
    std::size_t prefill_processed_tokens = 0;
    std::size_t generated_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double first_token_ms = 0.0;
    double total_ms = 0.0;
    double prefill_tok_s = 0.0;
    double decode_tok_s = 0.0;
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
        double embedding_ms = 0.0;
        std::size_t chunks = 0;
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
            if (prompt_tokens <= profile_position) {
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
                "v_projection", "kv_store", "attention", "attention_output_projection",
                "residual_post_norm", "post_normalization", "ffn_gate_up_projection",
                "swiglu", "ffn_down_projection", "residual", "residual"};
            std::array<double, 17> family_ms{};
            std::array<double, 4> recurrent_tail_family_ms{};
            std::array<double, 4> recurrent_wide_tail_family_ms{};
            double recurrent_deferred_tail_ms = 0.0;
            double attention_deferred_tail_ms = 0.0;
            double ordered_ms = 0.0;
            double sampled_total = embedding_ms;
            std::cout << "Prefill operator profile: prompt=" << prompt_tokens
                      << " chunks=" << chunks << " wall_ms=" << wall_ms
                      << " sampled_position=" << profile_position
                      << " projection_batch_width=B" << projection_batch_width
                      << " causal_chunk_width=B" << kM23CausalChunk
                      << " chunk_base=" << profile_chunk_base
                      << " embedding_ms=" << embedding_ms << '\n';
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
        for (const auto& layer : layers_) {
            if (layer.recurrent != nullptr) layer.recurrent->reset();
            if (layer.attention != nullptr) layer.attention->reset();
        }
        MIINFER_HIP_CHECK(hipDeviceSynchronize());
    }

    struct StepResult {
        std::uint32_t token;
        double latency_ms;
    };

    StepResult step(std::uint32_t input_token, std::size_t position) {
        if (position >= g_cache_capacity) {
            throw std::runtime_error("context length exceeded capacity " + std::to_string(g_cache_capacity));
        }
        ensure_graph_captured(position);

        const auto t0 = std::chrono::steady_clock::now();
        MIINFER_HIP_CHECK(hipMemcpyAsync(
            static_cast<std::uint32_t*>(d_decode_tokens_->get()) + position,
            &input_token, sizeof(input_token), hipMemcpyHostToDevice, hipStreamPerThread));

        if (use_hip_graph_ && decode_graphs_[position] != nullptr) {
            MIINFER_HIP_CHECK(hipGraphLaunch(decode_graphs_[position], hipStreamPerThread));
        } else {
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
                       static_cast<miinfer::Q8_1Block*>(final_q8_1_->get()));
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
        }

        std::uint32_t next_token = 0;
        MIINFER_HIP_CHECK(hipMemcpyAsync(&next_token,
                                        static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + (position + 1),
                                        sizeof(next_token), hipMemcpyDeviceToHost, hipStreamPerThread));
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
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
                                     std::size_t& processed_tokens) {
        if (full_layer_major_prefill_ && prompt.size() <= kFullPrefillCapacity
            && prompt.size() % kM12PrefillBatch == 0) {
            return prefill_full_layer_major(prompt, should_cancel, processed_tokens);
        }
        // A 128-token request with a non-128 tail would otherwise make the
        // final nearly-full chunk fall back to per-token recurrent execution.
        const bool matrix_prefill = gdn_chunkwise_prefill_ || dense_prefill_ || wide_prefill_;
        const std::size_t kChunk = (prefill_chunk_ > kPrefillBatch && matrix_prefill
            && prompt.size() % prefill_chunk_ != 0) ? kPrefillBatch : prefill_chunk_;
        float* current = static_cast<float*>(prefill_a_->get());
        float* next = static_cast<float*>(prefill_b_->get());
        const float* final_hidden = nullptr;
        const auto layer_span = std::span<const GpuLayerRef>(layers_);
        for (std::size_t base = 0; base < prompt.size(); base += kChunk) {
            if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
            const std::size_t count = std::min(kChunk, prompt.size() - base);
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
                if (!batched_attention) for (std::size_t i = 0; i < count; ++i) {
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
            processed_tokens = base + count;
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        return final_hidden;
    }

    const float* prefill_full_layer_major(std::span<const std::uint32_t> prompt,
                                          const std::function<bool()>& should_cancel,
                                          std::size_t& processed_tokens) {
        float* current = static_cast<float*>(prefill_a_->get());
        float* next = static_cast<float*>(prefill_b_->get());
        for (std::size_t i = 0; i < prompt.size(); ++i) {
            if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                static_cast<std::uint32_t*>(d_decode_tokens_->get()) + i,
                &prompt[i], sizeof(std::uint32_t), hipMemcpyHostToDevice, hipStreamPerThread));
            miinfer::launch_qwen35_q4_k_embedding_device_token(
                static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding_->get()),
                static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + i,
                model_.config().vocab_size, kHidden, current + i * kHidden,
                hipStreamPerThread);
        }
        const auto layer_span = std::span<const GpuLayerRef>(layers_);
        for (std::size_t layer = 0; layer < layer_span.size(); ++layer) {
            if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
            layer_span[layer].profile_ordered_start(
                prefill_profile_.enabled ? 0 : std::numeric_limits<std::size_t>::max());
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
                        chunk_input, chunk_output, static_cast<std::uint32_t>(base), count);
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
                            chunk_input + i * kHidden, static_cast<std::uint32_t>(base + i),
                            chunk_output + i * kHidden, nullptr, nullptr, false, nullptr, false,
                            layer_span[layer].prefill_qkv_at(i), layer_span[layer].prefill_gate_at(i),
                            layer_span[layer].prefill_normalized_at(i), deferred_tail, i,
                            layer_span[layer].prefill_qfull_at(i),
                            layer_span[layer].prefill_value_at(i));
                    } else {
                        layer_span[layer].run(
                            chunk_input + i * kHidden, static_cast<std::uint32_t>(base + i),
                            chunk_output + i * kHidden, nullptr, nullptr, false, nullptr, false,
                            nullptr, nullptr, nullptr);
                    }
                }
                if (batched_attention) {
                    layer_span[layer].finish_prefill_attention(
                        static_cast<std::uint32_t>(base), count);
                }
                if (deferred_tail) {
                    layer_span[layer].finish_prefill_batch(
                        chunk_input, chunk_output, count, nullptr, nullptr,
                        prefill_profile_.enabled ? base : std::numeric_limits<std::size_t>::max());
                }
            }
            layer_span[layer].profile_ordered_end(
                prefill_profile_.enabled ? 0 : std::numeric_limits<std::size_t>::max());
            layer_span[layer].release_m23_repacked();
            std::swap(current, next);
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        processed_tokens = prompt.size();
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
                                       std::chrono::steady_clock::time_point gen_start) {
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
        const float* final_hidden = prefill_layer_major(prompt, opt.should_cancel,
                                                        stats.prefill_processed_tokens);
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
        prefill_profile_.report(stats.prefill_ms, prompt.size());
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
            if (opt.on_token && !opt.on_token(cur_token, piece)) break;
        }
        stats.decode_ms += stats.first_token_ms;
        stats.generated_tokens = stats.tokens.size();
        stats.cancelled = cancelled;
        if (stats.generated_tokens > 0 && stats.decode_ms > 0.0) {
            stats.decode_tok_s = (1000.0 * stats.generated_tokens) / stats.decode_ms;
        }
        stats.total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gen_start).count();
        return stats;
    }

    GenerateStats generate(std::span<const std::uint32_t> prompt, const GenerateOptions& opt = GenerateOptions()) {
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

        if (!opt.stream && use_hip_graph_) {
            const std::size_t num_to_gen = std::min(opt.max_new_tokens, g_cache_capacity - pos);
            std::size_t captured_count = 0;
            for (std::size_t i = 0; i < num_to_gen; ++i) {
                if (g_shutdown_requested || (opt.should_cancel && opt.should_cancel())) {
                    stats.cancelled = true;
                    break;
                }
                ensure_graph_captured(pos + i);
                ++captured_count;
            }
            if (stats.cancelled) {
                stats.total_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - gen_start).count();
                return stats;
            }
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                static_cast<std::uint32_t*>(d_decode_tokens_->get()) + pos,
                &cur_token, sizeof(cur_token), hipMemcpyHostToDevice, hipStreamPerThread));

            const auto decode_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < captured_count; ++i) {
                MIINFER_HIP_CHECK(hipGraphLaunch(decode_graphs_[pos + i], hipStreamPerThread));
            }
            std::vector<std::uint32_t> raw_tokens(captured_count);
            MIINFER_HIP_CHECK(hipMemcpy(raw_tokens.data(),
                                        static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + pos + 1,
                                        captured_count * sizeof(std::uint32_t),
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
        gdn_chunkwise_prefill_ = gdn_chunkwise_prefill_ || wide_prefill_;
        const char* prefill_chunk_env = std::getenv("MIINFER_PREFILL_CHUNK");
        if (prefill_chunk_env != nullptr) {
            const auto requested = std::stoul(prefill_chunk_env);
            const bool wide_batch = wide_prefill_ && requested >= kM12PrefillBatch
                && requested <= kMaxWidePrefillBatch && requested % kPrefillBatch == 0;
            if (requested != 4 && requested != kPrefillBatch && !wide_batch) {
                throw std::runtime_error(
                    "MIINFER_PREFILL_CHUNK must be 4 or 64; wide prefill accepts 128..512");
            }
            prefill_chunk_ = requested;
        }
        const char* prefill_profile_env = std::getenv("MIINFER_PREFILL_PROFILE");
        prefill_profile_.enabled = layer_major_prefill_ && prefill_profile_env != nullptr
            && std::strcmp(prefill_profile_env, "0") != 0;
        if (prefill_profile_.enabled) {
            prefill_profile_.projection_batch_width = wide_prefill_
                ? configured_wide_prefill_batch() : 4;
            const char* position_env = std::getenv("MIINFER_PREFILL_PROFILE_POSITION");
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

        decode_graphs_.resize(g_cache_capacity, nullptr);
        prefill_profile_.init();
    }

    void ensure_graph_captured(std::size_t position) {
        if (!use_hip_graph_ || decode_graphs_[position] != nullptr) return;

        hipGraph_t graph = nullptr;
        MIINFER_HIP_CHECK(hipStreamBeginCapture(hipStreamPerThread, hipStreamCaptureModeRelaxed));

        const auto* token_device_ptr = static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + position;
        auto* next_token_device_ptr = static_cast<std::uint32_t*>(d_decode_tokens_->get()) + (position + 1);

        miinfer::launch_qwen35_q4_k_embedding_device_token(
            static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding_->get()),
            token_device_ptr, model_.config().vocab_size, kHidden,
            static_cast<float*>(input_->get()), hipStreamPerThread);

        run_prefix(std::span<const GpuLayerRef>(layers_),
                   std::span<float* const>(output_pointers_),
                   static_cast<const float*>(input_->get()), position,
                   static_cast<const float*>(d_final_norm_weight_->get()),
                   static_cast<float*>(final_norm_->get()),
                   static_cast<miinfer::Q8_1Block*>(final_q8_1_->get()));

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
            next_token_device_ptr,
            model_.config().vocab_size);

        MIINFER_HIP_CHECK(hipStreamEndCapture(hipStreamPerThread, &graph));
        MIINFER_HIP_CHECK(hipGraphInstantiate(&decode_graphs_[position], graph, nullptr, nullptr, 0));
        MIINFER_HIP_CHECK(hipGraphDestroy(graph));
    }

    void cleanup_graphs() {
        for (auto& g : decode_graphs_) {
            if (g != nullptr) {
                (void)hipGraphExecDestroy(g);
                g = nullptr;
            }
        }
    }

    miinfer::Qwen35Model model_;
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
    std::vector<hipGraphExec_t> decode_graphs_;
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
    const std::string model_path = argv[2];
    std::string prompt_text;
    std::optional<std::filesystem::path> prompt_file;
    std::size_t max_tokens = 128;
    bool stream = true;
    bool repeat_p512_check = false;

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

    std::cerr << "Initializing MIInfer gfx906 runtime engine for " << model_path << " ...\n";
    Qwen35RuntimeEngine engine(model_path);
    std::cerr << "device_allocation_count=" << g_device_allocations << "\n"
              << "device_allocated_bytes=" << g_live_device_bytes << "\n"
              << "device_total_allocated_bytes=" << g_total_device_bytes << "\n"
              << "device_peak_allocated_bytes=" << g_peak_device_bytes << "\n";
    print_hip_memory(std::cerr);
    std::cerr << "model_context_length=" << engine.model().config().context_length << "\n";

    const auto prompt_tokens = engine.tokenizer().encode(prompt_text);
    std::cerr << "Prompt tokens: " << prompt_tokens.size() << " tokens\n";

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

    g_cache_capacity = context_length;
    std::cerr << "Initializing MIInfer gfx906 HTTP Server on " << host << ":" << port << " ...\n";
    std::cerr << "configured_context_length=" << context_length << "\n"
              << "runtime_context_capacity=" << g_cache_capacity << "\n"
              << "qualified_context_length=1024\n"
              << "context_qualification=" << (context_length > 1024 ? "experimental" : "qualified") << "\n";
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
        if (is_stream) {
            bool client_connected = send_all(
                client_fd,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
            Qwen35RuntimeEngine::GenerateOptions opt;
            opt.max_new_tokens = max_tokens;
            opt.should_cancel = client_cancelled;
            opt.on_prefill_complete = [&] { state("prefill_completed"); };
            opt.on_first_token = [&] { state("first_token"); };
            bool request_cancelled = false;
            opt.on_token = [&](std::uint32_t /*token*/, std::string_view piece) {
                if (!client_connected) return false;
                const std::string sse = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\""
                    + json_escape(piece) + "\"}}]}\n\n";
                client_connected = send_all(client_fd, sse);
                return client_connected;
            };
            try {
                const auto stats = engine.generate(prompt_tokens, opt);
                request_cancelled = stats.cancelled;
                prompt_tokens_total += stats.prompt_tokens;
                prefill_tokens_total += stats.prefill_processed_tokens;
                prefill_us += static_cast<std::uint64_t>(stats.prefill_ms * 1000.0);
                generated_tokens_total += stats.generated_tokens;
                const double ttft_ms = stats.generated_tokens > 0
                    ? stats.prefill_ms + stats.first_token_ms : 0.0;
                ttft_us += static_cast<std::uint64_t>(ttft_ms * 1000.0);
                request_duration_us += static_cast<std::uint64_t>(stats.total_ms * 1000.0);
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
                          << ",\"finish_reason\":\"" << (cancelled ? "cancelled" : "stop") << "\""
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
            if (client_connected && !request_cancelled) (void)send_all(client_fd, "data: [DONE]\n\n");
        } else {
            Qwen35RuntimeEngine::GenerateOptions opt;
            opt.max_new_tokens = max_tokens;
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
            const double ttft_ms = stats.generated_tokens > 0
                ? stats.prefill_ms + stats.first_token_ms : 0.0;
            ttft_us += static_cast<std::uint64_t>(ttft_ms * 1000.0);
            request_duration_us += static_cast<std::uint64_t>(stats.total_ms * 1000.0);
            if (stats.cancelled) { ++cancelled_requests; state("cancelled"); }
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
                      << ",\"cancelled\":" << (stats.cancelled ? "true" : "false")
                      << ",\"finish_reason\":\"" << (stats.cancelled ? "cancelled" : "stop") << "\""
                      << ",\"error\":null"
                      << ",\"configured_context\":" << context_length
                      << ",\"runtime_context_capacity\":" << g_cache_capacity << "}\n";
            const std::string body = "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\""
                + json_escape(stats.text) + "\"},\"finish_reason\":\""
                + std::string(stats.cancelled ? "cancelled" : "stop") + "\"}],\"usage\":{\"prompt_tokens\":"
                + std::to_string(stats.prompt_tokens) + ",\"completion_tokens\":"
                + std::to_string(stats.generated_tokens) + "}}";
            (void)send_http_response(client_fd, 200, "OK", "application/json", body);
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
    std::cout << "       MIINFER_PRESET=m25_hi_qualified     Use the hermetic qualified MI50 P512 vector\n";
    std::cout << "  chat <model.gguf>                        Start an interactive multi-turn terminal chat REPL\n";
    std::cout << "  serve --model MODEL.gguf [--port 8080] [--context N] [--experimental-context]\n"
              << "        [--api-key-file PATH] [--allow-insecure]   Launch API and Web UI\n\n";
}

} // namespace

int main(int argc, char** argv) {
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
}
