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
#include <utility>
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
    if (!file) throw std::runtime_error("cannot open layer-35 trace: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid F32 layer-35 trace: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file) throw std::runtime_error("short layer-35 trace: " + path.string());
    return values;
}

Metrics metrics(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("layer-35 checkpoint size mismatch");
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
    Checkpoint{"attn_norm", "attn-norm.f32", &miinfer::Qwen3LayerTrace::attn_norm, 5.0e-2F},
    Checkpoint{"q_projection", "q-projection.f32", &miinfer::Qwen3LayerTrace::q_projection, 5.0e-2F},
    Checkpoint{"q_reshape", "q-reshape.f32", &miinfer::Qwen3LayerTrace::q_reshape, 5.0e-2F},
    Checkpoint{"q_normed", "q-normed.f32", &miinfer::Qwen3LayerTrace::q_normed, 5.0e-2F},
    Checkpoint{"q_rope", "q-rope.f32", &miinfer::Qwen3LayerTrace::q_rope, 5.0e-2F},
    Checkpoint{"v_projection", "v-projection.f32", &miinfer::Qwen3LayerTrace::v_projection, 5.0e-2F},
    Checkpoint{"k_projection", "k-projection.f32", &miinfer::Qwen3LayerTrace::k_projection, 5.0e-2F},
    Checkpoint{"k_reshape", "k-reshape.f32", &miinfer::Qwen3LayerTrace::k_reshape, 5.0e-2F},
    Checkpoint{"k_normed", "k-normed.f32", &miinfer::Qwen3LayerTrace::k_normed, 5.0e-2F},
    Checkpoint{"k_rope", "k-rope.f32", &miinfer::Qwen3LayerTrace::k_rope, 5.0e-2F},
    Checkpoint{"attention_output", "attention-output.f32", &miinfer::Qwen3LayerTrace::attention_output, 5.0e-2F},
    Checkpoint{"ffn_input", "ffn-input.f32", &miinfer::Qwen3LayerTrace::ffn_input, 5.0e-2F},
    Checkpoint{"ffn_norm", "ffn-norm.f32", &miinfer::Qwen3LayerTrace::ffn_norm, 5.0e-2F},
    Checkpoint{"gate", "gate.f32", &miinfer::Qwen3LayerTrace::gate, 5.0e-2F},
    Checkpoint{"up", "up.f32", &miinfer::Qwen3LayerTrace::up, 5.0e-2F},
    Checkpoint{"swiglu", "swiglu.f32", &miinfer::Qwen3LayerTrace::swiglu, 5.0e-2F},
    Checkpoint{"ffn_output", "ffn-output.f32", &miinfer::Qwen3LayerTrace::ffn_output, 5.0e-2F},
    Checkpoint{"layer_output", "layer-output.f32", &miinfer::Qwen3LayerTrace::layer_output, 5.0e-2F},
};

bool report(const char* authority, const miinfer::Qwen3LayerTrace& actual,
            const std::vector<std::vector<float>>& external) {
    std::cout << authority << " vs external layer-35 trace:\n";
    bool passed = true;
    bool previous_passed = true;
    for (const auto& checkpoint : kCheckpoints) {
        const auto result = metrics(actual.*(checkpoint.field),
                                    external[&checkpoint - kCheckpoints.data()]);
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
        if (!checkpoint_pass && previous_passed) {
            std::cout << "  FIRST DIVERGENCE: " << checkpoint.name << '\n';
        }
        previous_passed = previous_passed && checkpoint_pass;
        passed = passed && checkpoint_pass;
    }
    return passed;
}

void report_pair(const char* label, const miinfer::Qwen3LayerTrace& actual,
                 const miinfer::Qwen3LayerTrace& expected) {
    std::cout << label << ":\n";
    for (const auto& checkpoint : kCheckpoints) {
        const auto result = metrics(actual.*(checkpoint.field), expected.*(checkpoint.field));
        std::cout << "  " << checkpoint.name << " max_abs=" << result.max_abs
                  << " mean_abs=" << result.mean_abs
                  << " max_rel=" << result.max_rel << '\n';
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
    const auto terminal = read_f32(trace_path / "layer-output.f32");
    const auto final_norm_input = read_f32(trace_path / "final-norm-input.f32");
    const auto terminal_boundary = metrics(final_norm_input, terminal);
    const bool terminal_boundary_pass = terminal_boundary.max_abs == 0.0F;
    std::cout << "terminal layer-output -> final-norm-input boundary: "
              << (terminal_boundary_pass ? "PASS" : "FAIL")
              << " max_abs=" << terminal_boundary.max_abs << '\n';

    const auto model = miinfer::Qwen3Model::load(model_path.string());
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    const auto host = miinfer::execute_qwen3_layer_host_teacher_forced(model, 35, input);
    const auto gpu = miinfer::execute_qwen3_layer_gpu_teacher_forced(plan, 35, input);

    bool passed = terminal_boundary_pass;
    passed = report("HOST", host, external) && passed;
    passed = report("MI50 GPU", gpu, external) && passed;
    report_pair("GPU vs host", gpu, host);

    // Hybrid injections make the terminal-layer diagnosis causal.  The
    // inputs are independent external checkpoints; only the named GPU stage
    // is allowed to compute a new value.
    using Precision = miinfer::Qwen3ProjectionPrecision;
    const auto external_gate_up_swiglu = miinfer::execute_qwen3_ffn_gpu_probe(
        plan, 35, external[11], external[12], external[13], external[14], {}, {},
        Precision::f32_input_q8_f32_output);
    const auto external_swiglu_down = miinfer::execute_qwen3_ffn_gpu_probe(
        plan, 35, external[11], external[12], external[13], external[14],
        external[15], {}, Precision::f32_input_q8_f32_output);
    const auto external_residual = miinfer::execute_qwen3_ffn_gpu_probe(
        plan, 35, external[11], external[12], external[13], external[14],
        external[15], external[16], Precision::f32_input_q8_f32_output);
    const auto swiglu_hybrid = metrics(external_gate_up_swiglu.swiglu, external[15]);
    const auto down_hybrid = metrics(external_swiglu_down.ffn_output, external[16]);
    const auto residual_hybrid = metrics(external_residual.layer_output, external[17]);
    std::cout << "hybrid external-injection diagnostics:\n"
              << "  GPU SwiGLU (external gate/up): max_abs=" << swiglu_hybrid.max_abs
              << " mean_abs=" << swiglu_hybrid.mean_abs
              << " max_rel=" << swiglu_hybrid.max_rel << '\n'
              << "  GPU Down (external SwiGLU): max_abs=" << down_hybrid.max_abs
              << " mean_abs=" << down_hybrid.mean_abs
              << " max_rel=" << down_hybrid.max_rel << '\n'
              << "  GPU residual (external Down): max_abs=" << residual_hybrid.max_abs
              << " mean_abs=" << residual_hybrid.mean_abs
              << " max_rel=" << residual_hybrid.max_rel << '\n';
    std::cout << "M4-B9 external layer-35 comparison: "
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-qwen3-layer35-external-test MODEL LAYER35_TRACE\n";
        return 2;
    }
    try {
        return run(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "M4-B9 external layer-35 test error: " << error.what() << '\n';
        return 1;
    }
}
