#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
                        std::uint32_t capacity) {
    using namespace miinfer::prefill_v2;
    PrefillV2Model runtime(model, capacity, true, mode);
    std::vector<std::uint32_t> restored_tokens;
    auto restore_start = std::chrono::steady_clock::now();
    if (!runtime.load_session(session_path, restored_tokens) || restored_tokens.empty()) {
        throw std::runtime_error(std::string(label) + " failed to restore the prefix session");
    }
    auto restore_stop = std::chrono::steady_clock::now();

    float* hidden = allocate_device<float>(kPrefillV2MacroTile * kHidden);
    float* logits = allocate_device<float>(model.config().vocab_size);
    const auto prefill_start = std::chrono::steady_clock::now();
    std::size_t offset = 0;
    std::uint32_t final_chunk = 0;
    while (offset < suffix.size()) {
        final_chunk = static_cast<std::uint32_t>(std::min<std::size_t>(
            kPrefillV2MacroTile, suffix.size() - offset));
        runtime.forward(std::span<const std::uint32_t>(suffix).subspan(offset, final_chunk),
                        static_cast<std::uint32_t>(restored_tokens.size() + offset), hidden);
        offset += final_chunk;
    }
    runtime.compute_logits(hidden + (final_chunk - 1) * kHidden, logits);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    const auto prefill_stop = std::chrono::steady_clock::now();

    Result result;
    result.restore_seconds = std::chrono::duration<double>(restore_stop - restore_start).count();
    result.prefill_seconds = std::chrono::duration<double>(prefill_stop - prefill_start).count();
    result.logits.resize(model.config().vocab_size);
    MIINFER_HIP_CHECK(hipMemcpy(result.logits.data(), logits,
        result.logits.size() * sizeof(float), hipMemcpyDeviceToHost));
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
              << " first_token=" << result.top1 << " decode_seconds=" << result.decode_seconds
              << " second_token=" << result.second_token << '\n';
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

}  // namespace

int main(int argc, char** argv) {
    bool fp16_first = false;
    bool fp16_only = false;
    bool q8_query_tiled_pair = false;
    bool q8_only = false;
    bool fp16_fp32_attention = false;
    bool production_fp16_only = false;
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
        } else if (arg == "--q8-only") {
            q8_only = true;
        } else if (arg == "--fp16-fp32-attn") {
            fp16_fp32_attention = true;
        } else if (arg == "--production-fp16-only") {
            production_fp16_only = true;
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
                         "[--fp16-first] [--fp16-only [--production-fp16-only]] [--q8-only] [--logits-out PATH] "
                         "[--q8-query-tiled-pair] [--fp16-fp32-attn] [--tokens N] "
                         "[--prompt-file PATH] [--session PATH]\n";
            return 2;
        }
    }
    if (argc < 2 || (fp16_first && fp16_only) || (production_fp16_only && !fp16_only)
        || (q8_only && (fp16_first || fp16_only))
        || (q8_only && q8_query_tiled_pair)
        || (!logits_output.empty() && !(q8_only || fp16_only))
        || (q8_query_tiled_pair && (fp16_first || fp16_only))) {
        std::cerr << "usage: miinfer-m31-fp16-full-model MODEL.gguf "
                     "[--fp16-first] [--fp16-only [--production-fp16-only]] [--q8-only] [--logits-out PATH] "
                     "[--q8-query-tiled-pair] [--fp16-fp32-attn] [--tokens N] "
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

    if (::setenv("MIINFER_V2_0043_GQA_ATTENTION", production_fp16_only ? "0" : "1", 1) != 0) {
        std::cerr << "failed to set GQA attention path\n";
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
        std::vector<std::uint32_t> suffix(prompt_tokens);
        for (std::uint32_t i = 0; i < prompt_tokens; ++i) {
            suffix[i] = (i * 37U + 101U) % model.config().vocab_size;
        }
        std::cout << "session_prefix_tokens=" << header.prefix_length
                  << " suffix_tokens=" << prompt_tokens
                  << " suffix_source=deterministic_synthetic\n";
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
