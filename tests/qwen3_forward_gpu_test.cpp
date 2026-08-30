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

constexpr std::uint32_t kToken = 14990U;

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open forward trace: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid forward trace: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> result(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(result.data()), size);
    if (!file) throw std::runtime_error("short forward trace: " + path.string());
    return result;
}

struct Metrics {
    float max_abs = 0.0F;
    float max_rel = 0.0F;
    std::size_t max_index = 0;
    float actual_at_max = 0.0F;
    float expected_at_max = 0.0F;
};

Metrics metrics(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) return Metrics{INFINITY};
    Metrics result;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            result.max_abs = INFINITY;
            return result;
        }
        const float error = std::fabs(actual[index] - expected[index]);
        if (error > result.max_abs) {
            result.max_abs = error;
            result.max_index = index;
            result.actual_at_max = actual[index];
            result.expected_at_max = expected[index];
        }
        result.max_rel = std::max(result.max_rel,
                                  error / std::max(1.0F, std::fabs(expected[index])));
    }
    return result;
}

bool compare_checkpoint(const char* name, const std::vector<float>& actual,
                        const std::vector<float>& expected, float tolerance) {
    const auto result = metrics(actual, expected);
    const bool pass = result.max_abs <= tolerance;
    std::cout << name << ": " << (pass ? "PASS" : "FAIL")
              << " max_abs=" << result.max_abs
              << " max_index=" << result.max_index
              << " actual_at_max=" << result.actual_at_max
              << " expected_at_max=" << result.expected_at_max
              << " max_rel=" << result.max_rel << '\n';
    return pass;
}

void report_gpu_host_trace(const miinfer::Qwen3LayerTrace& gpu,
                           const miinfer::Qwen3LayerTrace& host) {
    using Field = std::pair<const char*, std::vector<float> miinfer::Qwen3LayerTrace::*>;
    static constexpr std::array fields{
        Field{"embedding", &miinfer::Qwen3LayerTrace::embedding},
        Field{"attn_rms", &miinfer::Qwen3LayerTrace::attn_rms},
        Field{"attn_norm", &miinfer::Qwen3LayerTrace::attn_norm},
        Field{"q_projection", &miinfer::Qwen3LayerTrace::q_projection},
        Field{"q_rms", &miinfer::Qwen3LayerTrace::q_rms},
        Field{"q_normed", &miinfer::Qwen3LayerTrace::q_normed},
        Field{"q_rope", &miinfer::Qwen3LayerTrace::q_rope},
        Field{"v_projection", &miinfer::Qwen3LayerTrace::v_projection},
        Field{"k_projection", &miinfer::Qwen3LayerTrace::k_projection},
        Field{"k_rms", &miinfer::Qwen3LayerTrace::k_rms},
        Field{"k_normed", &miinfer::Qwen3LayerTrace::k_normed},
        Field{"k_rope", &miinfer::Qwen3LayerTrace::k_rope},
        Field{"attention_output", &miinfer::Qwen3LayerTrace::attention_output},
        Field{"ffn_input", &miinfer::Qwen3LayerTrace::ffn_input},
        Field{"ffn_rms", &miinfer::Qwen3LayerTrace::ffn_rms},
        Field{"ffn_norm", &miinfer::Qwen3LayerTrace::ffn_norm},
        Field{"gate", &miinfer::Qwen3LayerTrace::gate},
        Field{"up", &miinfer::Qwen3LayerTrace::up},
        Field{"swiglu", &miinfer::Qwen3LayerTrace::swiglu},
        Field{"ffn_output", &miinfer::Qwen3LayerTrace::ffn_output},
        Field{"layer_output", &miinfer::Qwen3LayerTrace::layer_output},
    };
    for (const auto& field : fields) {
        const auto& gpu_values = gpu.*(field.second);
        const auto& host_values = host.*(field.second);
        if (gpu_values.size() != host_values.size()) {
            std::cout << "gpu-host " << field.first << ": SIZE MISMATCH\n";
            continue;
        }
        const auto result = metrics(gpu_values, host_values);
        std::cout << "gpu-host " << field.first
                  << ": max_abs=" << result.max_abs
                  << " max_index=" << result.max_index
                  << " actual=" << result.actual_at_max
                  << " host=" << result.expected_at_max << '\n';
    }
}

std::size_t argmax(const std::vector<float>& values) {
    return static_cast<std::size_t>(std::distance(values.begin(),
        std::max_element(values.begin(), values.end())));
}

int run(const std::filesystem::path& model_path, const std::filesystem::path& trace_path) {
    const auto model = miinfer::Qwen3Model::load(model_path.string());
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    const auto forward = miinfer::execute_qwen3_forward_gpu(plan, kToken);
    std::cout << "MI50 GPU 36-layer forward: PASS layers=" << forward.layer_outputs.size()
              << " logits=" << forward.logits.size()
              << " argmax=" << argmax(forward.logits) << '\n';
    if (trace_path.empty()) return 0;

    bool passed = compare_checkpoint("embedding", forward.embedding,
                                     read_f32(trace_path / "embedding.f32"), 2.5e-2F);
    for (std::size_t layer = 0; layer < forward.layer_outputs.size(); ++layer) {
        const auto name = std::string("layer-") + std::to_string(layer) + ".f32";
        passed = compare_checkpoint(name.c_str(), forward.layer_outputs[layer],
                                    read_f32(trace_path / name), 5.0e-2F) && passed;
    }
    passed = compare_checkpoint("final-norm", forward.final_norm,
                                read_f32(trace_path / "final-norm.f32"), 5.0e-2F) && passed;
    passed = compare_checkpoint("logits", forward.logits,
                                read_f32(trace_path / "logits.f32"), 1.0e-1F) && passed;
    std::cout << "MI50 GPU full-forward reference comparison: "
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

int run_teacher_forced(const std::filesystem::path& model_path,
                       const std::filesystem::path& trace_path,
                       bool all_layers) {
    if (trace_path.empty()) throw std::invalid_argument("teacher-forced replay requires a trace path");
    const auto model = miinfer::Qwen3Model::load(model_path.string());
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    const auto embedding = read_f32(trace_path / "embedding.f32");
    std::vector<std::size_t> selected_layers;
    if (all_layers) {
        selected_layers.resize(model.config().layer_count);
        for (std::size_t layer = 0; layer < selected_layers.size(); ++layer) {
            selected_layers[layer] = layer;
        }
    } else {
        selected_layers = {1, 2, 6};
    }
    bool passed = true;
    std::cout << "MI50 GPU teacher-forced replay (reference input -> selected layer):\n";
    for (const auto layer : selected_layers) {
        const auto input = layer == 0
            ? embedding
            : read_f32(trace_path / (std::string("layer-") + std::to_string(layer - 1) + ".f32"));
        const auto expected = read_f32(
            trace_path / (std::string("layer-") + std::to_string(layer) + ".f32"));
        const auto gpu_trace = miinfer::execute_qwen3_layer_gpu_teacher_forced(
            plan, layer, input);
        const auto host_trace = miinfer::execute_qwen3_layer_host_teacher_forced(
            model, layer, input);
        report_gpu_host_trace(gpu_trace, host_trace);
        const auto& actual = gpu_trace.layer_output;
        const auto name = std::string("teacher-layer-") + std::to_string(layer);
        passed = compare_checkpoint(name.c_str(), actual, expected, 5.0e-2F) && passed;
    }
    std::cout << "MI50 GPU teacher-forced replay: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cout << "qwen3 full-forward GPU test: SKIP (model path not supplied)\n";
        return 0;
    }
    try {
        if (argc == 4 && (std::string(argv[3]) == "--teacher-forced"
                          || std::string(argv[3]) == "--teacher-forced-all")) {
            return run_teacher_forced(argv[1], argv[2],
                                      std::string(argv[3]) == "--teacher-forced-all");
        }
        return run(argv[1], argc == 3 ? std::filesystem::path(argv[2])
                                      : std::filesystem::path{});
    } catch (const std::exception& error) {
        std::cerr << "qwen3 full-forward GPU test error: " << error.what() << '\n';
        return 1;
    }
}
