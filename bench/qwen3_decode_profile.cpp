#include "miinfer/build_config.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_layer.hpp"
#include "miinfer/sha256.hpp"

#include <hip/hip_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <time.h>

namespace {

struct TimePoint {
    timespec value{};
};

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
    std::uint32_t decode_token = 8;
    std::string json_output;
};

void usage() {
    std::cerr
        << "usage: miinfer-qwen3-decode-profile MODEL.gguf [options]\n"
        << "  --prompt-token N        warmed token at position 0 (default: 14990)\n"
        << "  --decode-token N        profiled token at position 1 (default: 8)\n"
        << "  --json-output PATH      write machine-readable profile\n";
}

std::uint32_t parse_token(const char* text, const char* option) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoull(text, &consumed, 10);
        if (consumed != std::string(text).size()
            || value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("invalid token");
        }
        return static_cast<std::uint32_t>(value);
    } catch (...) {
        throw std::invalid_argument(std::string(option) + " must be a token ID");
    }
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
        if (index + 1 >= argc) throw std::invalid_argument("missing value for " + argument);
        if (argument == "--prompt-token") {
            options.prompt_token = parse_token(argv[++index], "--prompt-token");
        } else if (argument == "--decode-token") {
            options.decode_token = parse_token(argv[++index], "--decode-token");
        } else if (argument == "--json-output") {
            options.json_output = argv[++index];
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return true;
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

std::string build_json(
    const Options& options,
    const miinfer::Qwen3Model& model,
    const miinfer::Qwen3GpuPlan& plan,
    const miinfer::Qwen3GpuProfile& profile,
    double wall_ms) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"benchmark\":\"m5b_qwen3_decode_profile\",\n"
           << "  \"experiment\":\"M5-B\",\n"
           << "  \"git_commit\":\"" << MIINFER_GIT_COMMIT << "\",\n"
           << "  \"git_dirty\":\"" << MIINFER_GIT_DIRTY << "\",\n"
           << "  \"build_type\":\"" << MIINFER_BUILD_TYPE << "\",\n"
           << "  \"target_architecture\":\"" << MIINFER_TARGET_ARCH << "\",\n"
           << "  \"model_path\":" << json_escape(options.model_path) << ",\n"
           << "  \"model_sha256\":\"" << miinfer::sha256_file(options.model_path) << "\",\n"
           << "  \"model_name\":" << json_escape(model.model_name()) << ",\n"
           << "  \"model_weight_bytes\":" << model.total_weight_bytes() << ",\n"
           << "  \"planned_weight_bytes\":" << plan.weights().bytes() << ",\n"
           << "  \"workspace_bytes\":" << plan.workspace_bytes() << ",\n"
           << "  \"total_vram_bytes\":" << plan.device().total_vram_bytes << ",\n"
           << "  \"prompt_token\":" << options.prompt_token << ",\n"
           << "  \"decode_token\":" << options.decode_token << ",\n"
           << "  \"profile_position\":1,\n"
           << "  \"wall_ms\":" << wall_ms << ",\n"
           << "  \"timing_method\":\"HIP events per operation; wall clock includes profiling copies\",\n"
           << "  \"categories\":[\n";
    for (std::size_t index = 0; index < miinfer::qwen3_profile_category_count; ++index) {
        if (index != 0) output << ",\n";
        const auto category = static_cast<miinfer::Qwen3ProfileCategory>(index);
        output << "    {\"name\":\"" << miinfer::qwen3_profile_category_name(category)
               << "\",\"gpu_ms\":" << profile.gpu_ms[index]
               << ",\"copy_ms\":" << profile.copy_ms[index]
               << ",\"dispatches\":" << profile.dispatches[index] << '}';
    }
    output << "\n  ],\n"
           << "  \"total_gpu_ms\":";
    double total_gpu_ms = 0.0;
    double total_copy_ms = 0.0;
    std::size_t total_dispatches = 0;
    for (std::size_t index = 0; index < miinfer::qwen3_profile_category_count; ++index) {
        total_gpu_ms += profile.gpu_ms[index];
        total_copy_ms += profile.copy_ms[index];
        total_dispatches += profile.dispatches[index];
    }
    output << total_gpu_ms << ",\n"
           << "  \"total_copy_ms\":" << total_copy_ms << ",\n"
           << "  \"total_dispatches\":" << total_dispatches << ",\n"
           << "  \"unaccounted_wall_ms\":" << (wall_ms - total_gpu_ms - total_copy_ms) << "\n"
           << "}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options;
        if (!parse_options(argc, argv, options)) return argc < 2 ? 2 : 0;

        const auto model = miinfer::Qwen3Model::load(options.model_path);
        const auto plan = miinfer::Qwen3GpuPlan::build(model);
        if (options.prompt_token >= model.config().vocab_size
            || options.decode_token >= model.config().vocab_size) {
            throw std::invalid_argument("profile token is outside the vocabulary");
        }
        miinfer::Qwen3GpuDecodeCache cache(
            model.config().layer_count, model.config().kv_heads, model.config().head_dim, 2);
        (void)miinfer::execute_qwen3_decode_gpu(plan, options.prompt_token, 0, cache);
        if (cache.length() != 1) throw std::runtime_error("warmup did not append one KV position");

        miinfer::Qwen3GpuProfile profile;
        profile.reset();
        MIINFER_HIP_CHECK(hipDeviceSynchronize());
        const auto start = now();
        const auto trace = miinfer::execute_qwen3_decode_gpu(
            plan, options.decode_token, 1, cache, &profile);
        MIINFER_HIP_CHECK(hipDeviceSynchronize());
        const auto wall_ms = elapsed_ms(start, now());
        if (cache.length() != 2) throw std::runtime_error("profiled decode did not append one KV position");
        if (trace.logits.empty()) throw std::runtime_error("profiled decode returned empty logits");
        for (const float value : trace.logits) {
            if (!std::isfinite(value)) throw std::runtime_error("profiled decode returned non-finite logits");
        }

        const auto result = build_json(options, model, plan, profile, wall_ms);
        if (!options.json_output.empty()) {
            std::ofstream output(options.json_output);
            if (!output) throw std::runtime_error("cannot open JSON output: " + options.json_output);
            output << result;
        }

        double total_gpu_ms = 0.0;
        double total_copy_ms = 0.0;
        std::size_t total_dispatches = 0;
        std::cout << "M5-B Qwen3 steady-state decode profile\n"
                  << "model: " << model.model_name() << "\n"
                  << "position: 1 (warm cache length 1)\n"
                  << std::fixed << std::setprecision(3)
                  << "wall: " << wall_ms << " ms\n";
        for (std::size_t index = 0; index < miinfer::qwen3_profile_category_count; ++index) {
            total_gpu_ms += profile.gpu_ms[index];
            total_copy_ms += profile.copy_ms[index];
            total_dispatches += profile.dispatches[index];
            std::cout << "  " << std::setw(20)
                      << miinfer::qwen3_profile_category_name(
                          static_cast<miinfer::Qwen3ProfileCategory>(index))
                      << ": " << std::setw(10) << profile.gpu_ms[index] << " ms GPU, "
                      << std::setw(10) << profile.copy_ms[index] << " ms copies, "
                      << profile.dispatches[index] << " dispatches\n";
        }
        std::cout << "total GPU: " << total_gpu_ms << " ms\n"
                  << "total copies: " << total_copy_ms << " ms\n"
                  << "dispatches: " << total_dispatches << "\n"
                  << "unaccounted wall: " << (wall_ms - total_gpu_ms - total_copy_ms) << " ms\n"
                  << "correctness: finite output and KV length 2\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M5-B decode profile error: " << error.what() << '\n';
        return 1;
    }
}
