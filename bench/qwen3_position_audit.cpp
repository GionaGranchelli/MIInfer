#include "miinfer/build_config.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_layer.hpp"
#include "miinfer/sha256.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

struct TimePoint { timespec value{}; };

TimePoint now() {
    TimePoint result;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &result.value) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC_RAW) failed");
    }
    return result;
}

double elapsed_ms(TimePoint start, TimePoint end) {
    const auto seconds = static_cast<double>(end.value.tv_sec - start.value.tv_sec);
    const auto nanoseconds = static_cast<double>(end.value.tv_nsec - start.value.tv_nsec);
    return seconds * 1000.0 + nanoseconds / 1000000.0;
}

struct Options {
    std::string model_path;
    std::uint32_t prompt_token = 14990;
    std::vector<std::size_t> positions{1, 8, 16, 32, 64};
    std::string json_output;
    bool gpu_argmax = false;
};

struct PositionResult {
    std::size_t position = 0;
    double production_wall_ms = 0.0;
    double production_gpu_ms = 0.0;
    double wall_ms = 0.0;
    miinfer::Qwen3GpuProfile profile;
};

void usage() {
    std::cerr
        << "usage: miinfer-qwen3-position-audit MODEL.gguf [options]\n"
        << "  --prompt-token N        initial token (default: 14990)\n"
        << "  --positions CSV         decode positions to audit (default: 1,8,16,32,64)\n"
        << "  --gpu-argmax            keep logits on device and copy only the selected ID\n"
        << "  --json-output PATH      write machine-readable result\n";
}

std::size_t parse_size(const std::string& text, const char* option) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoull(text, &consumed, 10);
        if (consumed != text.size() || value == 0
            || value > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("invalid value");
        }
        return static_cast<std::size_t>(value);
    } catch (...) {
        throw std::invalid_argument(std::string(option) + " must contain positive integers");
    }
}

std::uint32_t parse_token(const char* text) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoull(text, &consumed, 10);
        if (consumed != std::string(text).size()
            || value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("invalid token");
        }
        return static_cast<std::uint32_t>(value);
    } catch (...) {
        throw std::invalid_argument("--prompt-token must be a token ID");
    }
}

std::vector<std::size_t> parse_positions(const std::string& text) {
    if (text.empty()) throw std::invalid_argument("--positions must not be empty");
    std::vector<std::size_t> positions;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const auto end = text.find(',', begin);
        positions.push_back(parse_size(text.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin), "--positions"));
        if (end == std::string::npos) break;
        begin = end + 1;
        if (begin == text.size()) throw std::invalid_argument("--positions has an empty value");
    }
    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    return positions;
}

bool parse_options(int argc, char** argv, Options& options) {
    if (argc < 2) {
        usage();
        return false;
    }
    options.model_path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            usage();
            return false;
        }
        if (argument == "--gpu-argmax") {
            options.gpu_argmax = true;
            continue;
        }
        if (index + 1 >= argc) throw std::invalid_argument("missing value for " + argument);
        if (argument == "--prompt-token") {
            options.prompt_token = parse_token(argv[++index]);
        } else if (argument == "--positions") {
            options.positions = parse_positions(argv[++index]);
        } else if (argument == "--json-output") {
            options.json_output = argv[++index];
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return true;
}

std::size_t category_index(miinfer::Qwen3ProfileCategory category) {
    return static_cast<std::size_t>(category);
}

double total(const std::array<double, miinfer::qwen3_profile_category_count>& values) {
    double result = 0.0;
    for (const double value : values) result += value;
    return result;
}

std::size_t total(const std::array<std::size_t, miinfer::qwen3_profile_category_count>& values) {
    std::size_t result = 0;
    for (const std::size_t value : values) result += value;
    return result;
}

std::string json_escape(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (const unsigned char character : value) {
        if (character == '"') result << "\\\"";
        else if (character == '\\') result << "\\\\";
        else result << static_cast<char>(character);
    }
    result << '"';
    return result.str();
}

void write_profile_json(std::ostream& output, const miinfer::Qwen3GpuProfile& profile) {
    output << "{\"gpu_ms_total\":" << total(profile.gpu_ms)
           << ",\"copy_ms_total\":" << total(profile.copy_ms)
           << ",\"dispatches\":" << total(profile.dispatches)
           << ",\"copy_bytes\":" << total(profile.copy_bytes)
           << ",\"synchronizations\":"
           << (total(profile.synchronizations) + profile.finalization_synchronizations)
           << ",\"temporary_allocations\":" << profile.temporary_allocations
           << ",\"gate_up_q8_reuse_checks\":" << profile.gate_up_q8_reuse_checks
           << ",\"gate_up_q8_reuse_mismatches\":" << profile.gate_up_q8_reuse_mismatches
           << ",\"ffn_norm_q8_fusion_f16_checks\":"
           << profile.ffn_norm_q8_fusion_f16_checks
           << ",\"ffn_norm_q8_fusion_f16_mismatches\":"
           << profile.ffn_norm_q8_fusion_f16_mismatches
           << ",\"ffn_norm_q8_fusion_q8_checks\":"
           << profile.ffn_norm_q8_fusion_q8_checks
           << ",\"ffn_norm_q8_fusion_q8_mismatches\":"
           << profile.ffn_norm_q8_fusion_q8_mismatches
           << ",\"categories\":[";
    for (std::size_t index = 0; index < miinfer::qwen3_profile_category_count; ++index) {
        if (index != 0) output << ',';
        output << "{\"name\":\""
               << miinfer::qwen3_profile_category_name(
                      static_cast<miinfer::Qwen3ProfileCategory>(index))
               << "\",\"gpu_ms\":" << profile.gpu_ms[index]
               << ",\"copy_ms\":" << profile.copy_ms[index]
               << ",\"dispatches\":" << profile.dispatches[index]
               << ",\"copy_bytes\":" << profile.copy_bytes[index]
               << ",\"synchronizations\":" << profile.synchronizations[index] << '}';
    }
    output << "],\"ffn_stages\":[";
    for (std::size_t index = 0; index < miinfer::qwen3_ffn_profile_stage_count; ++index) {
        if (index != 0) output << ',';
        output << "{\"name\":\""
               << miinfer::qwen3_ffn_profile_stage_name(
                      static_cast<miinfer::Qwen3FfnProfileStage>(index))
               << "\",\"gpu_ms\":" << profile.ffn_gpu_ms[index]
               << ",\"dispatches\":" << profile.ffn_dispatches[index] << '}';
    }
    output << "],\"boundaries\":[";
    for (std::size_t index = 0; index < miinfer::qwen3_boundary_profile_stage_count; ++index) {
        if (index != 0) output << ',';
        output << "{\"name\":\""
               << miinfer::qwen3_boundary_profile_stage_name(
                      static_cast<miinfer::Qwen3BoundaryProfileStage>(index))
               << "\",\"gpu_ms\":" << profile.boundary_gpu_ms[index]
               << ",\"dispatches\":" << profile.boundary_dispatches[index]
               << ",\"bytes\":" << profile.boundary_bytes[index] << '}';
    }
    output << "]}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options;
        if (!parse_options(argc, argv, options)) return argc < 2 ? 2 : 0;
        const auto model = miinfer::Qwen3Model::load(options.model_path);
        const auto& config = model.config();
        if (options.prompt_token >= config.vocab_size) {
            throw std::invalid_argument("prompt token is outside the vocabulary");
        }
        if (options.positions.empty() || options.positions.back() >= config.context_length) {
            throw std::invalid_argument("audit position exceeds model context");
        }

        const auto plan = miinfer::Qwen3GpuPlan::build(model);
        const char* reuse_policy = std::getenv("MIINFER_FFN_Q8_REUSE");
        const bool shared_gate_up_q8 = reuse_policy != nullptr
            && std::string(reuse_policy) == "shared";
        miinfer::Qwen3GpuDecodeCache cache(
            config.layer_count, config.kv_heads, config.head_dim, options.positions.back() + 1);
        std::vector<float> logits(config.vocab_size);
        std::vector<double> production_wall_ms(options.positions.size(), 0.0);
        std::vector<double> production_gpu_ms(options.positions.size(), 0.0);
        std::vector<std::uint32_t> production_tokens;
        production_tokens.reserve(options.positions.back() + 1);
        auto token = options.prompt_token;

        const auto decode_next = [&](std::uint32_t input, std::size_t position,
                                     miinfer::Qwen3GpuProfile* profile = nullptr) {
            if (options.gpu_argmax) {
                return miinfer::execute_qwen3_decode_gpu_greedy(
                    plan, input, position, cache, profile);
            }
            miinfer::execute_qwen3_decode_gpu_fast(
                plan, input, position, cache, std::span<float>(logits), profile);
            for (const float value : logits) {
                if (!std::isfinite(value)) throw std::runtime_error("non-finite audit logits");
            }
            return static_cast<std::uint32_t>(std::distance(
                logits.begin(), std::max_element(logits.begin(), logits.end())));
        };

        // Collect clean wall times separately from the event-instrumented
        // pass. Event recording is intentionally excluded from these values.
        cache.reset();
        std::size_t selected_index = 0;
        for (std::size_t position = 0; position <= options.positions.back(); ++position) {
            const bool selected = selected_index < options.positions.size()
                && options.positions[selected_index] == position;
            const auto start = selected ? now() : TimePoint{};
            const auto next = decode_next(token, position);
            if (selected) {
                production_wall_ms[selected_index] = elapsed_ms(start, now());
                ++selected_index;
            }
            production_tokens.push_back(next);
            token = next;
        }

        // Measure the trace-free device timeline separately from the clean
        // wall pass and the per-operation deferred profile. The decode API's
        // result copy completes the default stream before the stop event is
        // recorded, so this adds only two events per selected token and avoids the
        // thousands of event records used by the detailed audit profile.
        cache.reset();
        token = options.prompt_token;
        selected_index = 0;
        for (std::size_t position = 0; position <= options.positions.back(); ++position) {
            const bool selected = selected_index < options.positions.size()
                && options.positions[selected_index] == position;
            if (!selected) {
                token = decode_next(token, position);
                continue;
            }
            hipEvent_t start = nullptr;
            hipEvent_t stop = nullptr;
            MIINFER_HIP_CHECK(hipEventCreate(&start));
            MIINFER_HIP_CHECK(hipEventCreate(&stop));
            MIINFER_HIP_CHECK(hipEventRecord(start));
            const auto next = decode_next(token, position);
            MIINFER_HIP_CHECK(hipEventRecord(stop));
            MIINFER_HIP_CHECK(hipEventSynchronize(stop));
            float milliseconds = 0.0F;
            MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
            production_gpu_ms[selected_index++] = milliseconds;
            (void)hipEventDestroy(start);
            (void)hipEventDestroy(stop);
            token = next;
        }

        std::vector<PositionResult> results;
        results.reserve(options.positions.size());
        std::vector<std::uint32_t> selected_tokens;
        selected_tokens.reserve(options.positions.back() + 1);
        token = options.prompt_token;
        cache.reset();
        selected_index = 0;

        for (std::size_t position = 0; position <= options.positions.back(); ++position) {
            const bool selected = std::binary_search(
                options.positions.begin(), options.positions.end(), position);
            if (selected) {
                PositionResult result;
                result.position = position;
                result.production_wall_ms = production_wall_ms[selected_index++];
                result.production_gpu_ms = production_gpu_ms[results.size()];
                result.profile.reset();
                result.profile.enable_deferred_timing();
                const auto start = now();
                const auto next = decode_next(token, position, &result.profile);
                result.wall_ms = elapsed_ms(start, now());
                results.push_back(std::move(result));
                selected_tokens.push_back(next);
                token = next;
            } else {
                const auto next = decode_next(token, position);
                selected_tokens.push_back(next);
                token = next;
            }
        }

        if (production_tokens != selected_tokens) {
            throw std::runtime_error("production and profiled audit token trajectories differ");
        }

        std::ostringstream json;
        json << std::fixed << std::setprecision(6)
             << "{\n"
             << "  \"benchmark\":\"" << (options.gpu_argmax
                 ? "m5c6d_qwen3_argmax_position_audit"
                 : "m5c1_qwen3_position_audit") << "\",\n"
             << "  \"experiment\":\"" << (options.gpu_argmax ? "M5-C6d" : "M5-C1") << "\",\n"
             << "  \"gpu_argmax\":" << (options.gpu_argmax ? "true" : "false") << ",\n"
             << "  \"ffn_q8_reuse\":\"" << (shared_gate_up_q8 ? "shared" : "separate") << "\",\n"
             << "  \"git_commit\":\"" << MIINFER_GIT_COMMIT << "\",\n"
             << "  \"git_dirty\":\"" << MIINFER_GIT_DIRTY << "\",\n"
             << "  \"build_type\":\"" << MIINFER_BUILD_TYPE << "\",\n"
             << "  \"target_architecture\":\"" << MIINFER_TARGET_ARCH << "\",\n"
             << "  \"model_path\":" << json_escape(options.model_path) << ",\n"
             << "  \"model_sha256\":\"" << miinfer::sha256_file(options.model_path) << "\",\n"
             << "  \"model_name\":" << json_escape(model.model_name()) << ",\n"
             << "  \"prompt_token\":" << options.prompt_token << ",\n"
             << "  \"positions\":[";
        for (std::size_t index = 0; index < options.positions.size(); ++index) {
            if (index != 0) json << ',';
            json << options.positions[index];
        }
        json << "],\n  \"timing_method\":\"clean CLOCK_MONOTONIC_RAW wall time; lightweight whole-token HIP events; deferred per-operation HIP events\",\n"
             << "  \"selected_tokens\":[";
        for (std::size_t index = 0; index < selected_tokens.size(); ++index) {
            if (index != 0) json << ',';
            json << selected_tokens[index];
        }
        json << "],\n  \"samples\":[\n";
        for (std::size_t index = 0; index < results.size(); ++index) {
            const auto& result = results[index];
            if (index != 0) json << ",\n";
            json << "    {\"position\":" << result.position
                 << ",\"cache_length_before\":" << result.position
                 << ",\"production_wall_ms\":" << result.production_wall_ms
                 << ",\"production_gpu_ms\":" << result.production_gpu_ms
                 << ",\"audit_wall_ms\":" << result.wall_ms << ",\"profile\":";
            write_profile_json(json, result.profile);
            json << '}';
        }
        json << "\n  ]\n}\n";

        if (!options.json_output.empty()) {
            std::ofstream output(options.json_output);
            if (!output) throw std::runtime_error("cannot open JSON output: " + options.json_output);
            output << json.str();
        }

        const auto index_of = [](miinfer::Qwen3ProfileCategory category) {
            return category_index(category);
        };
        std::cout << (options.gpu_argmax
            ? "M5-C6d Qwen3 GPU-argmax position audit\n"
            : "M5-C1 Qwen3 position-scaled decode audit\n")
                  << "FFN Gate/Up Q8 input policy: "
                  << (shared_gate_up_q8 ? "shared" : "separate") << "\n"
                  << "model: " << model.model_name() << "\n"
                  << "prompt token: " << options.prompt_token << "\n"
                  << "timing: clean wall plus lightweight whole-token HIP events and deferred per-operation events\n"
                  << "position cache_before production_wall_ms production_gpu_ms audit_wall_ms gpu_ms attention_ms kv_cache_ms quant_ms "
                     "ffn_ms copy_ms copy_bytes dispatches syncs temporary_allocations reuse_checks reuse_mismatches "
                     "fusion_f16_checks fusion_f16_mismatches fusion_q8_checks fusion_q8_mismatches\n";
        for (const auto& result : results) {
            const auto& profile = result.profile;
            std::cout << std::fixed << std::setprecision(3)
                      << result.position << ' ' << result.position << ' '
                      << result.production_wall_ms << ' ' << result.production_gpu_ms << ' '
                      << result.wall_ms << ' '
                      << total(profile.gpu_ms) << ' '
                      << profile.gpu_ms[index_of(miinfer::Qwen3ProfileCategory::attention)] << ' '
                      << profile.copy_ms[index_of(miinfer::Qwen3ProfileCategory::kv_cache)] << ' '
                      << profile.gpu_ms[index_of(miinfer::Qwen3ProfileCategory::quantization)] << ' '
                      << profile.gpu_ms[index_of(miinfer::Qwen3ProfileCategory::ffn_projection)] << ' '
                      << total(profile.copy_ms) << ' ' << total(profile.copy_bytes) << ' '
                      << total(profile.dispatches) << ' '
                      << (total(profile.synchronizations) + profile.finalization_synchronizations) << ' '
                      << profile.temporary_allocations << ' '
                      << profile.gate_up_q8_reuse_checks << ' '
                      << profile.gate_up_q8_reuse_mismatches << ' '
                      << profile.ffn_norm_q8_fusion_f16_checks << ' '
                      << profile.ffn_norm_q8_fusion_f16_mismatches << ' '
                      << profile.ffn_norm_q8_fusion_q8_checks << ' '
                      << profile.ffn_norm_q8_fusion_q8_mismatches << '\n';
            std::cout << "  FFN stage GPU ms / dispatches:\n";
            for (std::size_t stage = 0; stage < miinfer::qwen3_ffn_profile_stage_count; ++stage) {
                std::cout << "    "
                          << miinfer::qwen3_ffn_profile_stage_name(
                                 static_cast<miinfer::Qwen3FfnProfileStage>(stage))
                          << ": " << profile.ffn_gpu_ms[stage] << " ms / "
                          << profile.ffn_dispatches[stage] << '\n';
            }
            std::cout << "  Normalization/conversion/quantization boundary GPU ms / dispatches / bytes:\n";
            for (std::size_t stage = 0; stage < miinfer::qwen3_boundary_profile_stage_count; ++stage) {
                const auto boundary = static_cast<miinfer::Qwen3BoundaryProfileStage>(stage);
                if (profile.boundary_dispatches[stage] == 0) continue;
                std::cout << "    " << miinfer::qwen3_boundary_profile_stage_name(boundary)
                          << ": " << profile.boundary_gpu_ms[stage] << " ms / "
                          << profile.boundary_dispatches[stage] << " / "
                          << profile.boundary_bytes[stage] << " bytes\n";
            }
        }
        std::cout << "selected next-token IDs through audited range: [";
        for (std::size_t index = 0; index < selected_tokens.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << selected_tokens[index];
        }
        std::cout << "]\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen3 position audit error: " << error.what() << '\n';
        return 1;
    }
}
