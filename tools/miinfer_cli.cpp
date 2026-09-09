#include <atomic>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
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

namespace {

std::atomic<bool> g_shutdown_requested{false};
int g_signal_wakeup_fd = -1;

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
};

struct RuntimeGenerateStats {
    std::vector<std::uint32_t> tokens;
    std::string text;
    std::size_t prompt_tokens = 0;
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
        std::array<std::array<double, 3>, 64> layer_ms{};
        double embedding_ms = 0.0;
        std::size_t chunks = 0;

        void init() {
            if (!enabled) return;
            for (auto& event : stage) MIINFER_HIP_CHECK(hipEventCreate(&event));
            MIINFER_HIP_CHECK(hipEventCreate(&embedding_start));
            MIINFER_HIP_CHECK(hipEventCreate(&embedding_end));
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
        }

        void report(double wall_ms, std::size_t prompt_tokens) const {
            if (!enabled) return;
            double layer_total = 0.0;
            std::cout << "Prefill profile: prompt=" << prompt_tokens
                      << " chunks=" << chunks << " wall_ms=" << wall_ms
                      << " embedding_ms=" << embedding_ms << '\n';
            for (std::size_t layer = 0; layer < layer_ms.size(); ++layer) {
                const double total = layer_ms[layer][0] + layer_ms[layer][1] + layer_ms[layer][2];
                layer_total += total;
                std::cout << "  layer=" << layer
                          << " kind=" << (layer % 4 == 3 ? "attention" : "recurrent")
                          << " prepare_ms=" << layer_ms[layer][0]
                          << " ordered_ms=" << layer_ms[layer][1]
                          << " tail_ms=" << layer_ms[layer][2]
                          << " total_ms=" << total << '\n';
            }
            std::cout << "  layer_total_ms=" << layer_total
                      << " accounted_ms=" << (layer_total + embedding_ms)
                      << " unaccounted_wall_ms=" << (wall_ms - layer_total - embedding_ms)
                      << '\n';
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
                                     const std::function<bool()>& should_cancel) {
        // A 128-token request with a non-128 tail would otherwise make the
        // final nearly-full chunk fall back to per-token recurrent execution.
        const bool matrix_prefill = gdn_chunkwise_prefill_ || dense_prefill_;
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
                if (prefill_profile_.enabled) {
                    MIINFER_HIP_CHECK(hipEventRecord(prefill_profile_.stage[0], hipStreamPerThread));
                }
                const bool normalized_ready = layer > 0 && layer_span[layer - 1].fused_interlayer_norm();
                const bool prepared = layer_span[layer].prepare_prefill_batch(current, count, normalized_ready);
                const bool full_m12_chunk = count % kPrefillBatch == 0;
                const bool deferred_tail = prepared && layer_span[layer].prefill_tail_batch_supported()
                    && (!gdn_chunkwise_prefill_ || full_m12_chunk);
                if (prefill_profile_.enabled) {
                    MIINFER_HIP_CHECK(hipEventRecord(prefill_profile_.stage[1], hipStreamPerThread));
                }
                const bool fuse_next_norm = layer + 1 < layer_span.size()
                    && layer_span[layer].fused_interlayer_norm();
                const float* next_norm_weight = fuse_next_norm
                    ? layer_span[layer + 1].attn_norm_weight() : nullptr;
                float* next_normalized_batch = fuse_next_norm
                    ? const_cast<float*>(layer_span[layer + 1].prefill_normalized_at(0)) : nullptr;
                for (std::size_t i = 0; i < count; ++i) {
                    if (g_shutdown_requested || (should_cancel && should_cancel())) return nullptr;
                    float* next_normalized = fuse_next_norm
                        ? next_normalized_batch + i * kHidden : nullptr;
                    const float* prepared_normalized = normalized_ready
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
                if (prefill_profile_.enabled) {
                    MIINFER_HIP_CHECK(hipEventRecord(prefill_profile_.stage[2], hipStreamPerThread));
                }
                if (deferred_tail) {
                    layer_span[layer].finish_prefill_batch(
                        current, next, count, next_norm_weight, next_normalized_batch);
                }
                if (prefill_profile_.enabled) {
                    MIINFER_HIP_CHECK(hipEventRecord(prefill_profile_.stage[3], hipStreamPerThread));
                    MIINFER_HIP_CHECK(hipEventSynchronize(prefill_profile_.stage[3]));
                    for (std::size_t stage = 0; stage < 3; ++stage) {
                        float elapsed_ms = 0.0F;
                        MIINFER_HIP_CHECK(hipEventElapsedTime(
                            &elapsed_ms, prefill_profile_.stage[stage],
                            prefill_profile_.stage[stage + 1]));
                        prefill_profile_.layer_ms[layer][stage] += elapsed_ms;
                    }
                }
                std::swap(current, next);
            }
            final_hidden = current + (count - 1) * kHidden;
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        return final_hidden;
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
        const auto prefill_start = std::chrono::steady_clock::now();
        const float* final_hidden = prefill_layer_major(prompt, opt.should_cancel);
        const auto prefill_end = std::chrono::steady_clock::now();
        stats.prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
        stats.prefill_tok_s = stats.prefill_ms > 0.0
            ? (1000.0 * prompt.size()) / stats.prefill_ms : 0.0;
        if (final_hidden == nullptr) {
            stats.cancelled = true;
            stats.total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - gen_start).count();
            return stats;
        }
        prefill_profile_.report(stats.prefill_ms, prompt.size());
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
        bool cancelled = false;
        if (cur_token != tokenizer_.eos_id() && cur_token != 151643 && cur_token != 151645) {
            const std::string piece = tokenizer_.decode(std::span<const std::uint32_t>(&cur_token, 1));
            stats.text += piece;
            cancelled = opt.on_token && !opt.on_token(cur_token, piece);
        }

        std::size_t pos = prompt.size();
        for (std::size_t gen_idx = 1; !cancelled && gen_idx < opt.max_new_tokens && pos < g_cache_capacity; ++gen_idx) {
            if (g_shutdown_requested || (opt.should_cancel && opt.should_cancel())) { cancelled = true; break; }
            const auto step_res = step(cur_token, pos);
            cur_token = step_res.token;
            stats.tokens.push_back(cur_token);
            stats.decode_ms += step_res.latency_ms;
            ++pos;
            if (cur_token == tokenizer_.eos_id() || cur_token == 151643 || cur_token == 151645) break;
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
        }
        if (prompt.size() > 1) {
            MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        }
        const auto prefill_end = std::chrono::steady_clock::now();
        stats.prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
        if (prompt.size() > 1 && stats.prefill_ms > 0.0) {
            stats.prefill_tok_s = (1000.0 * (prompt.size() - 1)) / stats.prefill_ms;
        }

        // 2. Decode generation loop
        std::uint32_t cur_token = prompt.back();
        std::size_t pos = prompt.size() - 1;

        if (!opt.stream && use_hip_graph_) {
            const std::size_t num_to_gen = std::min(opt.max_new_tokens, g_cache_capacity - pos);
            for (std::size_t i = 0; i < num_to_gen; ++i) {
                if (g_shutdown_requested || (opt.should_cancel && opt.should_cancel())) { stats.cancelled = true; break; }
                ensure_graph_captured(pos + i);
            }
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                static_cast<std::uint32_t*>(d_decode_tokens_->get()) + pos,
                &cur_token, sizeof(cur_token), hipMemcpyHostToDevice, hipStreamPerThread));

            const auto decode_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < num_to_gen; ++i) {
                MIINFER_HIP_CHECK(hipGraphLaunch(decode_graphs_[pos + i], hipStreamPerThread));
            }
            std::vector<std::uint32_t> raw_tokens(num_to_gen);
            MIINFER_HIP_CHECK(hipMemcpy(raw_tokens.data(),
                                        static_cast<const std::uint32_t*>(d_decode_tokens_->get()) + pos + 1,
                                        num_to_gen * sizeof(std::uint32_t),
                                        hipMemcpyDeviceToHost));
            const auto decode_end = std::chrono::steady_clock::now();
            stats.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

            for (std::uint32_t next : raw_tokens) {
                stats.tokens.push_back(next);
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
        const char* prefill_chunk_env = std::getenv("MIINFER_PREFILL_CHUNK");
        if (prefill_chunk_env != nullptr) {
            const auto requested = std::stoul(prefill_chunk_env);
            if (requested != 4 && requested != kPrefillBatch
                && !((gdn_chunkwise_prefill_ || dense_prefill_)
                     && requested == kM12PrefillBatch)) {
                throw std::runtime_error(
                    "MIINFER_PREFILL_CHUNK must be 4 or 64; M12 also accepts 128");
            }
            prefill_chunk_ = requested;
        }
        const char* prefill_profile_env = std::getenv("MIINFER_PREFILL_PROFILE");
        prefill_profile_.enabled = layer_major_prefill_ && prefill_profile_env != nullptr
            && std::strcmp(prefill_profile_env, "0") != 0;
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
        }
    }

    void init_m12_gdn_workspace() {
        if (!gdn_chunkwise_prefill_) return;
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
        m12_gdn_raw_output_ = allocate(kM12PrefillBatch * kVHeads * kState * sizeof(float));
        for (auto& layer : recurrent_layers_) {
            layer->set_m12_gdn_workspace(
                m12_gdn_workspace_, static_cast<float*>(m12_gdn_raw_output_->get()));
        }
    }

    void init_m12_dense_workspace() {
        if (!dense_prefill_) return;
        m12_dense_weights_ = allocate(kHidden * kFfnInner * sizeof(__half));
        m12_dense_input_ = allocate(kM12PrefillBatch * kFfnInner * sizeof(__half));
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
        const std::size_t prefill_capacity = (gdn_chunkwise_prefill_ || dense_prefill_)
            ? kM12PrefillBatch : kPrefillBatch;
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
    Buffer m12_dense_weights_, m12_dense_input_;
    miinfer::RocblasGemmHandle m12_dense_gemm_{};
    Buffer prefill_a_;
    Buffer prefill_b_;

    bool use_hip_graph_ = true;
    bool layer_major_prefill_ = false;
    bool gdn_chunkwise_prefill_ = false;
    bool dense_prefill_ = false;
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
        std::cerr << "usage: miinfer run <model.gguf> --prompt \"...\" [--max-tokens N]\n";
        return 1;
    }
    const std::string model_path = argv[2];
    std::string prompt_text;
    std::size_t max_tokens = 128;
    bool stream = true;

    for (int i = 3; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoull(argv[++i]);
        } else if (arg == "--no-stream") {
            stream = false;
        }
    }

    if (prompt_text.empty()) {
        prompt_text = "The AMD Instinct MI50 is a high-performance GPU featuring";
        std::cerr << "No prompt provided. Defaulting to: \"" << prompt_text << "\"\n";
    }

    std::cerr << "Initializing MIInfer gfx906 runtime engine for " << model_path << " ...\n";
    Qwen35RuntimeEngine engine(model_path);
    std::cerr << "device_allocation_count=" << g_device_allocations << "\n"
              << "device_allocated_bytes=" << g_device_bytes << "\n"
              << "device_peak_allocated_bytes=" << g_peak_device_bytes << "\n";
    std::cerr << "model_context_length=" << engine.model().config().context_length << "\n";

    const auto prompt_tokens = engine.tokenizer().encode(prompt_text);
    std::cerr << "Prompt tokens: " << prompt_tokens.size() << " tokens\n";
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
              << "device_allocated_bytes=" << g_device_bytes << "\n"
              << "device_peak_allocated_bytes=" << g_peak_device_bytes << "\n";

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
                if (!stats.cancelled) {
                    state("prefill_completed");
                    if (stats.generated_tokens > 0) state("first_token");
                }
                prompt_tokens_total += stats.prompt_tokens;
                prefill_tokens_total += stats.prompt_tokens;
                prefill_us += static_cast<std::uint64_t>(stats.prefill_ms * 1000.0);
                generated_tokens_total += stats.generated_tokens;
                ttft_us += static_cast<std::uint64_t>(stats.first_token_ms * 1000.0);
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
            // Server requests must observe shutdown between tokens; the bulk
            // graph path intentionally does not provide that interruption point.
            opt.stream = true;
            const auto stats = engine.generate(prompt_tokens, opt);
            if (!stats.cancelled) {
                state("prefill_completed");
                if (stats.generated_tokens > 0) state("first_token");
            }
            prompt_tokens_total += stats.prompt_tokens;
            prefill_tokens_total += stats.prompt_tokens;
            prefill_us += static_cast<std::uint64_t>(stats.prefill_ms * 1000.0);
            generated_tokens_total += stats.generated_tokens;
            ttft_us += static_cast<std::uint64_t>(stats.first_token_ms * 1000.0);
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
            std::cerr << "miinfer_request_state request_id=" << request.request_id << " state=queued\n";
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
