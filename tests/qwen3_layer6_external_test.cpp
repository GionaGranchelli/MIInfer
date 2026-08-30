#include "miinfer/qwen3_gpu_layer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Metrics {
    float max_abs = 0.0F;
    float mean_abs = 0.0F;
    float rmse = 0.0F;
    float max_rel = 0.0F;
    std::size_t max_index = 0;
    float actual_at_max = 0.0F;
    float expected_at_max = 0.0F;
    bool finite = true;
};

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open external layer-6 trace: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid external layer-6 trace: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file) throw std::runtime_error("short external layer-6 trace: " + path.string());
    return values;
}

Metrics metrics(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("layer-6 checkpoint size mismatch");
    }
    Metrics result;
    double sum_abs = 0.0;
    double sum_squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            result.finite = false;
        }
        const float error = std::fabs(actual[index] - expected[index]);
        sum_abs += error;
        sum_squared += static_cast<double>(error) * error;
        if (error > result.max_abs) {
            result.max_abs = error;
            result.max_index = index;
            result.actual_at_max = actual[index];
            result.expected_at_max = expected[index];
        }
        result.max_rel = std::max(result.max_rel,
            error / std::max(1.0F, std::fabs(expected[index])));
    }
    if (!actual.empty()) {
        result.mean_abs = static_cast<float>(sum_abs / actual.size());
        result.rmse = static_cast<float>(std::sqrt(sum_squared / actual.size()));
    }
    return result;
}

struct Checkpoint {
    const char* name;
    const char* file;
    std::vector<float> miinfer::Qwen3LayerTrace::*field;
    float tolerance;
};

constexpr std::array kCheckpoints{
    Checkpoint{"attn_norm", "attn-norm.f32", &miinfer::Qwen3LayerTrace::attn_norm, 2.0e-4F},
    Checkpoint{"q_projection", "q-projection.f32", &miinfer::Qwen3LayerTrace::q_projection, 2.0e-4F},
    Checkpoint{"q_reshape", "q-reshape.f32", &miinfer::Qwen3LayerTrace::q_reshape, 2.0e-4F},
    Checkpoint{"q_normed", "q-normed.f32", &miinfer::Qwen3LayerTrace::q_normed, 5.0e-3F},
    Checkpoint{"q_rope", "q-rope.f32", &miinfer::Qwen3LayerTrace::q_rope, 2.5e-2F},
    Checkpoint{"v_projection", "v-projection.f32", &miinfer::Qwen3LayerTrace::v_projection, 2.0e-4F},
    Checkpoint{"v_reshape", "v-reshape.f32", &miinfer::Qwen3LayerTrace::v_reshape, 2.0e-4F},
    Checkpoint{"k_projection", "k-projection.f32", &miinfer::Qwen3LayerTrace::k_projection, 2.0e-4F},
    Checkpoint{"k_reshape", "k-reshape.f32", &miinfer::Qwen3LayerTrace::k_reshape, 2.0e-4F},
    Checkpoint{"k_normed", "k-normed.f32", &miinfer::Qwen3LayerTrace::k_normed, 5.0e-3F},
    Checkpoint{"k_rope", "k-rope.f32", &miinfer::Qwen3LayerTrace::k_rope, 2.5e-2F},
    Checkpoint{"attention_output", "attention-output.f32", &miinfer::Qwen3LayerTrace::attention_output, 2.0e-2F},
    Checkpoint{"ffn_input", "ffn-input.f32", &miinfer::Qwen3LayerTrace::ffn_input, 2.0e-2F},
    Checkpoint{"ffn_norm", "ffn-norm.f32", &miinfer::Qwen3LayerTrace::ffn_norm, 2.0e-2F},
    Checkpoint{"gate", "gate.f32", &miinfer::Qwen3LayerTrace::gate, 2.0e-2F},
    Checkpoint{"up", "up.f32", &miinfer::Qwen3LayerTrace::up, 2.0e-2F},
    Checkpoint{"swiglu", "swiglu.f32", &miinfer::Qwen3LayerTrace::swiglu, 2.0e-2F},
    Checkpoint{"ffn_output", "ffn-output.f32", &miinfer::Qwen3LayerTrace::ffn_output, 2.0e-2F},
    Checkpoint{"layer_output", "layer-output.f32", &miinfer::Qwen3LayerTrace::layer_output, 2.0e-2F},
};

void compare_authority(const char* authority, const miinfer::Qwen3LayerTrace& actual,
                       const std::vector<std::vector<float>>& expected,
                       bool& passed) {
    std::cout << authority << " vs external layer-6 trace:\n";
    bool prior_pass = true;
    for (std::size_t index = 0; index < kCheckpoints.size(); ++index) {
        const auto& checkpoint = kCheckpoints[index];
        const auto result = metrics(actual.*(checkpoint.field), expected[index]);
        const bool checkpoint_pass = result.finite && result.max_abs <= checkpoint.tolerance;
        std::cout << "  " << checkpoint.name << ": "
                  << (checkpoint_pass ? "PASS" : "FAIL")
                  << " max_abs=" << result.max_abs
                  << " mean_abs=" << result.mean_abs
                  << " rmse=" << result.rmse
                  << " max_rel=" << result.max_rel
                  << " max_index=" << result.max_index
                  << " actual=" << result.actual_at_max
                  << " external=" << result.expected_at_max << '\n';
        if (!checkpoint_pass && prior_pass) {
            std::cout << "  FIRST DIVERGENCE: " << checkpoint.name << '\n';
        }
        prior_pass = prior_pass && checkpoint_pass;
        passed = passed && checkpoint_pass;
    }
}

void compare_host_gpu(const miinfer::Qwen3LayerTrace& host,
                      const miinfer::Qwen3LayerTrace& gpu) {
    std::cout << "GPU vs host layer-6 trace:\n";
    for (const auto& checkpoint : kCheckpoints) {
        const auto result = metrics(gpu.*(checkpoint.field), host.*(checkpoint.field));
        std::cout << "  " << checkpoint.name << ": max_abs=" << result.max_abs
                  << " mean_abs=" << result.mean_abs
                  << " max_index=" << result.max_index << '\n';
    }
}

int run(const std::filesystem::path& model_path,
        const std::filesystem::path& trace_path) {
    std::vector<std::vector<float>> external;
    external.reserve(kCheckpoints.size());
    for (const auto& checkpoint : kCheckpoints) {
        external.push_back(read_f32(trace_path / checkpoint.file));
    }

    const auto input = read_f32(trace_path / "layer-input.f32");
    const auto model = miinfer::Qwen3Model::load(model_path.string());
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    const auto host = miinfer::execute_qwen3_layer_host_teacher_forced(model, 6, input);
    const auto gpu = miinfer::execute_qwen3_layer_gpu_teacher_forced(plan, 6, input);

    bool passed = true;
    compare_authority("HOST", host, external, passed);
    compare_authority("MI50 GPU", gpu, external, passed);
    compare_host_gpu(host, gpu);

    // Prove that this test consumes the external fixture rather than merely
    // checking internal host/GPU agreement.  A deliberate mutation must fail
    // the same stage tolerance as the real authority comparison.
    auto mutated_external = external.front();
    if (mutated_external.empty()) {
        throw std::runtime_error("empty external attn-norm checkpoint");
    }
    mutated_external[0] += 1.0F;
    const auto mutation_result = metrics(host.attn_norm, mutated_external);
    const bool mutation_detected =
        !mutation_result.finite || mutation_result.max_abs > kCheckpoints.front().tolerance;
    std::cout << "external trace mutation discriminator: "
              << (mutation_detected ? "PASS" : "FAIL")
              << " max_abs=" << mutation_result.max_abs << '\n';
    passed = passed && mutation_detected;

    std::cout << "M4-B6 external layer-6 comparison: "
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-qwen3-layer6-external-test MODEL LAYER6_TRACE\n";
        return 2;
    }
    try {
        return run(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "M4-B6 external layer-6 test error: " << error.what() << '\n';
        return 1;
    }
}
