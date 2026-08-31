#include "miinfer/qwen3_layer.hpp"
#include "miinfer/qwen3_primitives.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <stdexcept>
#include <utility>
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

using TraceField = std::pair<const char*, std::vector<float> miinfer::Qwen3LayerTrace::*>;

static constexpr std::array kCompositionFields{
    TraceField{"input", &miinfer::Qwen3LayerTrace::embedding},
    TraceField{"attn_rms", &miinfer::Qwen3LayerTrace::attn_rms},
    TraceField{"attn_norm", &miinfer::Qwen3LayerTrace::attn_norm},
    TraceField{"q_projection", &miinfer::Qwen3LayerTrace::q_projection},
    TraceField{"k_projection", &miinfer::Qwen3LayerTrace::k_projection},
    TraceField{"v_projection", &miinfer::Qwen3LayerTrace::v_projection},
    TraceField{"attention_output", &miinfer::Qwen3LayerTrace::attention_output},
    TraceField{"ffn_input", &miinfer::Qwen3LayerTrace::ffn_input},
    TraceField{"ffn_norm", &miinfer::Qwen3LayerTrace::ffn_norm},
    TraceField{"gate", &miinfer::Qwen3LayerTrace::gate},
    TraceField{"up", &miinfer::Qwen3LayerTrace::up},
    TraceField{"swiglu", &miinfer::Qwen3LayerTrace::swiglu},
    TraceField{"ffn_output", &miinfer::Qwen3LayerTrace::ffn_output},
    TraceField{"layer_output", &miinfer::Qwen3LayerTrace::layer_output},
};

static constexpr std::array kLayer0ReferenceFields{
    TraceField{"embedding", &miinfer::Qwen3LayerTrace::embedding},
    TraceField{"attn_rms", &miinfer::Qwen3LayerTrace::attn_rms},
    TraceField{"attn_norm", &miinfer::Qwen3LayerTrace::attn_norm},
    TraceField{"q_projection", &miinfer::Qwen3LayerTrace::q_projection},
    TraceField{"q_reshape", &miinfer::Qwen3LayerTrace::q_reshape},
    TraceField{"q_rms", &miinfer::Qwen3LayerTrace::q_rms},
    TraceField{"q_normed", &miinfer::Qwen3LayerTrace::q_normed},
    TraceField{"q_rope", &miinfer::Qwen3LayerTrace::q_rope},
    TraceField{"v_projection", &miinfer::Qwen3LayerTrace::v_projection},
    TraceField{"v_reshape", &miinfer::Qwen3LayerTrace::v_reshape},
    TraceField{"k_projection", &miinfer::Qwen3LayerTrace::k_projection},
    TraceField{"k_reshape", &miinfer::Qwen3LayerTrace::k_reshape},
    TraceField{"k_rms", &miinfer::Qwen3LayerTrace::k_rms},
    TraceField{"k_normed", &miinfer::Qwen3LayerTrace::k_normed},
    TraceField{"k_rope", &miinfer::Qwen3LayerTrace::k_rope},
    TraceField{"k_view", &miinfer::Qwen3LayerTrace::k_view},
    TraceField{"v_view", &miinfer::Qwen3LayerTrace::v_view},
    TraceField{"q_view", &miinfer::Qwen3LayerTrace::q_view},
    TraceField{"q_permuted", &miinfer::Qwen3LayerTrace::q_permuted},
    TraceField{"attention_output", &miinfer::Qwen3LayerTrace::attention_output},
    TraceField{"ffn_input", &miinfer::Qwen3LayerTrace::ffn_input},
    TraceField{"ffn_rms", &miinfer::Qwen3LayerTrace::ffn_rms},
    TraceField{"ffn_norm", &miinfer::Qwen3LayerTrace::ffn_norm},
    TraceField{"gate", &miinfer::Qwen3LayerTrace::gate},
    TraceField{"up", &miinfer::Qwen3LayerTrace::up},
    TraceField{"swiglu", &miinfer::Qwen3LayerTrace::swiglu},
    TraceField{"ffn_output", &miinfer::Qwen3LayerTrace::ffn_output},
    TraceField{"layer_output", &miinfer::Qwen3LayerTrace::layer_output},
};

void print_composition_delta(const char* prefix, const Metrics& result) {
    std::cout << prefix << " max_abs=" << result.max_abs
              << " mean_abs=" << result.mean_abs
              << " rmse=" << result.rmse
              << " max_rel=" << result.max_rel
              << " max_index=" << result.max_index
              << " actual=" << result.actual_at_max
              << " expected=" << result.expected_at_max << '\n';
}

int run_composition_diagnostic(const std::filesystem::path& model_path,
                              const std::filesystem::path& trace_path) {
    if (trace_path.empty()) {
        throw std::invalid_argument("composition diagnostic requires a trace path");
    }
    const auto model = miinfer::Qwen3Model::load(model_path.string());
    const auto embedding = read_f32(trace_path / "embedding.f32");
    const auto full = miinfer::execute_qwen3_forward_host(model, kToken);
    std::vector<float> sequential_input = embedding;
    bool full_matches_reconstructed = true;
    bool isolated_matches_reference = true;
    bool sequential_matches_reference = true;
    bool reported_first_nonzero = false;

    std::cout << "M4-B15 host sequential composition diagnostic (layers 0..2):\n";
    for (std::size_t layer = 0; layer < 3; ++layer) {
        const auto expected_input = layer == 0
            ? embedding
            : read_f32(trace_path / (std::string("layer-")
                                    + std::to_string(layer - 1) + ".f32"));
        const auto expected_output = read_f32(trace_path / (std::string("layer-")
                                                           + std::to_string(layer) + ".f32"));
        const auto isolated = miinfer::execute_qwen3_layer_host_teacher_forced(
            model, layer, expected_input);
        const auto sequential = miinfer::execute_qwen3_layer_host_teacher_forced(
            model, layer, sequential_input);

        std::cout << "layer " << layer << ":\n";
        const auto input_error = compare(sequential.embedding, expected_input);
        print_composition_delta("  sequential-input vs external-input", input_error);
        const auto isolated_error = compare(isolated.layer_output, expected_output);
        const auto sequential_error = compare(sequential.layer_output, expected_output);
        print_composition_delta("  isolated-output vs external-output", isolated_error);
        print_composition_delta("  sequential-output vs external-output", sequential_error);
        isolated_matches_reference = isolated_matches_reference && isolated_error.max_abs <= 5.0e-2F;
        sequential_matches_reference = sequential_matches_reference && sequential_error.max_abs <= 5.0e-2F;

        for (const auto& field : kCompositionFields) {
            const auto delta = compare(sequential.*(field.second), isolated.*(field.second));
            print_composition_delta((std::string("  sequential-vs-isolated ") + field.first).c_str(), delta);
            if (!reported_first_nonzero && delta.max_abs != 0.0F) {
                std::cout << "  FIRST NONZERO DIVERGENCE: layer=" << layer
                          << " stage=" << field.first << '\n';
                reported_first_nonzero = true;
            }
        }

        const auto full_error = compare(full.layer_outputs[layer], sequential.layer_output);
        print_composition_delta("  execute_forward_host vs reconstructed-sequential", full_error);
        full_matches_reconstructed = full_matches_reconstructed && full_error.max_abs == 0.0F;
        sequential_input = sequential.layer_output;
    }

    std::cout << "M4-B15 host summary:"
              << " isolated_vs_external=" << (isolated_matches_reference ? "PASS" : "FAIL")
              << " sequential_vs_external=" << (sequential_matches_reference ? "PASS" : "FAIL")
              << " full_vs_reconstructed=" << (full_matches_reconstructed ? "PASS" : "FAIL")
              << "\n";
    if (!reported_first_nonzero) {
        std::cout << "  no nonzero sequential-vs-isolated divergence in layers 0..2\n";
    }
    return isolated_matches_reference && sequential_matches_reference
        && full_matches_reconstructed ? 0 : 1;
}

std::vector<float> fp16_round_trip(const std::vector<float>& values) {
    const auto half_bits = [](float value) {
        const _Float16 half = static_cast<_Float16>(value);
        std::uint16_t bits = 0;
        static_assert(sizeof(half) == sizeof(bits));
        std::memcpy(&bits, &half, sizeof(bits));
        return bits;
    };
    std::vector<float> result;
    result.reserve(values.size());
    for (const float value : values) {
        result.push_back(miinfer::fp16_bits_to_float(half_bits(value)));
    }
    return result;
}

int run_layer0_precision_diagnostic(const std::filesystem::path& model_path,
                                   const std::filesystem::path& trace_path) {
    if (trace_path.empty()) {
        throw std::invalid_argument("layer-0 precision diagnostic requires a trace path");
    }
    const auto model = miinfer::Qwen3Model::load(model_path.string());
    const auto read_checkpoint = [&](std::size_t index) {
        return read_f32(trace_path / (std::string("pos-0-")
                                      + std::to_string(index) + ".f32"));
    };
    const auto embedding = read_checkpoint(0);
    const auto actual = miinfer::execute_qwen3_layer_host_teacher_forced(
        model, 0, embedding);
    std::cout << "M4-B16 host layer-0 precision diagnostic:\n";
    std::size_t first_nonzero = kLayer0ReferenceFields.size();
    for (std::size_t index = 0; index < kLayer0ReferenceFields.size(); ++index) {
        const auto& field = kLayer0ReferenceFields[index];
        const auto expected = read_checkpoint(index);
        const auto result = compare(actual.*(field.second), expected);
        print_composition_delta((std::string("  ") + field.first).c_str(), result);
        if (first_nonzero == kLayer0ReferenceFields.size() && result.max_abs != 0.0F) {
            first_nonzero = index;
        }
    }
    const auto external_layer_output = read_checkpoint(27);
    const auto rounded = fp16_round_trip(actual.layer_output);
    const auto current = compare(actual.layer_output, external_layer_output);
    const auto rounded_metrics = compare(rounded, external_layer_output);
    std::cout << "  layer-output FP16 round-trip:\n";
    print_composition_delta("    current F32", current);
    print_composition_delta("    round-tripped F16", rounded_metrics);
    if (first_nonzero == kLayer0ReferenceFields.size()) {
        std::cout << "  first nonzero checkpoint: none\n";
    } else {
        std::cout << "  first nonzero checkpoint: "
                  << kLayer0ReferenceFields[first_nonzero].first << '\n';
    }
    std::cout << "  note: this is diagnostic-only; no production precision changed\n";
    return 0;
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

int run_teacher_forced(const std::filesystem::path& model_path,
                       const std::filesystem::path& trace_path,
                       bool all_layers) {
    if (trace_path.empty()) throw std::invalid_argument("teacher-forced replay requires a trace path");
    const auto model = miinfer::Qwen3Model::load(model_path.string());
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
    std::cout << "host teacher-forced replay (reference input -> selected layer):\n";
    for (const auto layer : selected_layers) {
        const auto input = layer == 0
            ? embedding
            : read_f32(trace_path / (std::string("layer-") + std::to_string(layer - 1) + ".f32"));
        const auto expected = read_f32(
            trace_path / (std::string("layer-") + std::to_string(layer) + ".f32"));
        const auto actual = miinfer::execute_qwen3_layer_host_teacher_forced(
            model, layer, input).layer_output;
        const auto name = std::string("teacher-layer-") + std::to_string(layer);
        passed = compare_checkpoint(name.c_str(), actual, expected, 5.0e-2F) && passed;
    }
    std::cout << "host teacher-forced replay: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cout << "qwen3 full-forward host test: SKIP (model path not supplied)\n";
        return 0;
    }
    try {
        const auto trace_path = argc >= 3 ? std::filesystem::path(argv[2]) : std::filesystem::path{};
        if (argc == 4 && (std::string(argv[3]) == "--teacher-forced"
                          || std::string(argv[3]) == "--teacher-forced-all")) {
            return run_teacher_forced(argv[1], trace_path,
                                      std::string(argv[3]) == "--teacher-forced-all");
        }
        if (argc == 4 && std::string(argv[3]) == "--composition-diagnostic") {
            return run_composition_diagnostic(argv[1], trace_path);
        }
        if (argc == 5 && std::string(argv[3]) == "--layer0-precision-diagnostic") {
            return run_layer0_precision_diagnostic(argv[1], argv[4]);
        }
        const bool write_trace = argc == 4 && std::string(argv[3]) == "--write-trace";
        return run(argv[1], trace_path, write_trace);
    } catch (const std::exception& error) {
        std::cerr << "qwen3 full-forward host test error: " << error.what() << '\n';
        return 1;
    }
}
