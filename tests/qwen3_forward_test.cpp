#include "miinfer/qwen3_layer.hpp"
#include "miinfer/qwen3_primitives.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kToken = 14990U;

struct Metrics {
    float max_abs = 0.0F;
    float max_rel = 0.0F;
    float mean_abs = 0.0F;
    float rmse = 0.0F;
    std::size_t max_index = 0;
    std::size_t max_rel_index = 0;
    float actual_at_max = 0.0F;
    float expected_at_max = 0.0F;
    bool finite = true;
};

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open forward trace: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("forward trace is not F32: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> result(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(result.data()), size);
    if (!file) throw std::runtime_error("short forward trace: " + path.string());
    return result;
}

void write_f32(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("cannot create forward trace: " + path.string());
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) throw std::runtime_error("cannot write forward trace: " + path.string());
}

Metrics compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("forward checkpoint size mismatch");
    Metrics result;
    double absolute = 0.0;
    double squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) result.finite = false;
        const float error = std::fabs(actual[index] - expected[index]);
        if (error > result.max_abs) {
            result.max_abs = error;
            result.max_index = index;
            result.actual_at_max = actual[index];
            result.expected_at_max = expected[index];
        }
        // Relative error is intentionally floored at an absolute scale of
        // one so values near zero do not manufacture meaningless ratios.
        const float relative = error / std::max(1.0F, std::fabs(expected[index]));
        if (relative > result.max_rel) {
            result.max_rel = relative;
            result.max_rel_index = index;
        }
        absolute += error;
        squared += static_cast<double>(error) * error;
    }
    result.mean_abs = static_cast<float>(absolute / actual.size());
    result.rmse = static_cast<float>(std::sqrt(squared / actual.size()));
    return result;
}

std::size_t argmax(const std::vector<float>& values) {
    return static_cast<std::size_t>(std::distance(values.begin(),
        std::max_element(values.begin(), values.end())));
}

bool compare_checkpoint(const char* name, const std::vector<float>& actual,
                        const std::vector<float>& expected, float tolerance) {
    const auto metrics = compare(actual, expected);
    const bool pass = metrics.finite && metrics.max_abs <= tolerance;
    std::cout << name << ": " << (pass ? "PASS" : "FAIL")
              << " max_abs=" << metrics.max_abs
              << " mean_abs=" << metrics.mean_abs
              << " rmse=" << metrics.rmse
              << " max_index=" << metrics.max_index
              << " actual_at_max=" << metrics.actual_at_max
              << " expected_at_max=" << metrics.expected_at_max
              << " max_rel=" << metrics.max_rel
              << " max_rel_index=" << metrics.max_rel_index << '\n';
    return pass;
}

int run(const std::filesystem::path& model_path, const std::filesystem::path& trace_path,
        bool write_trace) {
    const auto model = miinfer::Qwen3Model::load(model_path.string());
    const auto forward = miinfer::execute_qwen3_forward_host(model, kToken);
    if (forward.layer_outputs.size() != model.config().layer_count) {
        throw std::runtime_error("full forward returned an incomplete layer trace");
    }
    std::cout << "host 36-layer forward: PASS"
              << " layers=" << forward.layer_outputs.size()
              << " logits=" << forward.logits.size()
              << " argmax=" << argmax(forward.logits) << '\n';

    if (write_trace) {
        std::filesystem::create_directories(trace_path);
        write_f32(trace_path / "embedding.f32", forward.embedding);
        for (std::size_t layer = 0; layer < forward.layer_outputs.size(); ++layer) {
            const auto name = std::string("layer-") + std::to_string(layer) + ".f32";
            write_f32(trace_path / name, forward.layer_outputs[layer]);
        }
        write_f32(trace_path / "final-norm.f32", forward.final_norm);
        write_f32(trace_path / "logits.f32", forward.logits);
        std::cout << "wrote host forward trace: " << trace_path << '\n';
        return 0;
    }
    if (trace_path.empty()) return 0;

    bool passed = true;
    passed = compare_checkpoint("embedding", forward.embedding,
                                read_f32(trace_path / "embedding.f32"), 2.5e-2F) && passed;
    for (std::size_t layer = 0; layer < forward.layer_outputs.size(); ++layer) {
        const auto name = std::string("layer-") + std::to_string(layer) + ".f32";
        // Depth checkpoints use a deliberately explicit, frozen bound.  This
        // is a host/reference gate, not a performance comparison.
        passed = compare_checkpoint(name.c_str(), forward.layer_outputs[layer],
                                    read_f32(trace_path / name), 5.0e-2F) && passed;
    }
    passed = compare_checkpoint("final-norm", forward.final_norm,
                                read_f32(trace_path / "final-norm.f32"), 5.0e-2F) && passed;
    const auto expected_final_norm = read_f32(trace_path / "final-norm.f32");
    const auto expected_logits = read_f32(trace_path / "logits.f32");
    passed = compare_checkpoint("logits", forward.logits, expected_logits, 1.0e-1F) && passed;
    std::vector<float> logits_from_reference_norm(expected_logits.size());
    miinfer::q6_k_q8_k_gemv_reference(
        {model.output().data(), model.output().bytes()}, expected_final_norm,
        logits_from_reference_norm, model.config().vocab_size, model.config().hidden_size);
    passed = compare_checkpoint("logits (reference final norm through MIInfer Q6_K)",
                                logits_from_reference_norm, expected_logits, 1.0e-1F) && passed;
    const auto expected_argmax = argmax(expected_logits);
    const auto actual_argmax = argmax(forward.logits);
    std::cout << "argmax: " << (actual_argmax == expected_argmax ? "PASS" : "FAIL")
              << " actual=" << actual_argmax << " reference=" << expected_argmax << '\n';
    passed = actual_argmax == expected_argmax && passed;
    std::cout << "host full-forward reference comparison: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cout << "qwen3 full-forward host test: SKIP (model path not supplied)\n";
        return 0;
    }
    try {
        const bool write_trace = argc == 4 && std::string(argv[3]) == "--write-trace";
        return run(argv[1], argc >= 3 ? std::filesystem::path(argv[2]) : std::filesystem::path{},
                   write_trace);
    } catch (const std::exception& error) {
        std::cerr << "qwen3 full-forward host test error: " << error.what() << '\n';
        return 1;
    }
}
