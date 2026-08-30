#include "miinfer/qwen3_gpu_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cout << "qwen3 full-forward GPU test: SKIP (model path not supplied)\n";
        return 0;
    }
    try {
        return run(argv[1], argc == 3 ? std::filesystem::path(argv[2])
                                      : std::filesystem::path{});
    } catch (const std::exception& error) {
        std::cerr << "qwen3 full-forward GPU test error: " << error.what() << '\n';
        return 1;
    }
}
