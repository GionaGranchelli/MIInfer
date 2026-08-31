#include "miinfer/qwen3_gpu_layer.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/q4_q8_gemv.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen3_primitives.hpp"

#include <hip/hip_runtime.h>

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
    float mean_abs = 0.0F;
    float rmse = 0.0F;
    float max_rel = 0.0F;
    std::size_t max_index = 0;
    float actual_at_max = 0.0F;
    float expected_at_max = 0.0F;
};

Metrics metrics(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) return Metrics{INFINITY};
    Metrics result;
    double sum_abs = 0.0;
    double sum_squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            result.max_abs = INFINITY;
            return result;
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
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    const auto embedding = read_f32(trace_path / "embedding.f32");
    const auto host_full = miinfer::execute_qwen3_forward_host(model, kToken);
    const auto gpu_full = miinfer::execute_qwen3_forward_gpu(plan, kToken);
    std::vector<float> host_sequential_input = embedding;
    std::vector<float> gpu_sequential_input = embedding;
    bool reported_first_host_nonzero = false;
    bool reported_first_gpu_nonzero = false;
    bool passed = true;
    bool full_matches_reconstructed = true;

    std::cout << "M4-B15 MI50 sequential composition diagnostic (layers 0..2):\n";
    for (std::size_t layer = 0; layer < 3; ++layer) {
        const auto expected_input = layer == 0
            ? embedding
            : read_f32(trace_path / (std::string("layer-")
                                    + std::to_string(layer - 1) + ".f32"));
        const auto expected_output = read_f32(trace_path / (std::string("layer-")
                                                           + std::to_string(layer) + ".f32"));
        const auto host_isolated = miinfer::execute_qwen3_layer_host_teacher_forced(
            model, layer, expected_input);
        const auto host_sequential = miinfer::execute_qwen3_layer_host_teacher_forced(
            model, layer, host_sequential_input);
        const auto gpu_isolated = miinfer::execute_qwen3_layer_gpu_teacher_forced(
            plan, layer, expected_input);
        const auto gpu_sequential = miinfer::execute_qwen3_layer_gpu_teacher_forced(
            plan, layer, gpu_sequential_input);

        std::cout << "layer " << layer << ":\n";
        const auto host_iso_error = metrics(host_isolated.layer_output, expected_output);
        const auto host_seq_error = metrics(host_sequential.layer_output, expected_output);
        const auto gpu_iso_error = metrics(gpu_isolated.layer_output, expected_output);
        const auto gpu_seq_error = metrics(gpu_sequential.layer_output, expected_output);
        const auto host_gpu_seq_error = metrics(host_sequential.layer_output,
                                                gpu_sequential.layer_output);
        const auto full_host_error = metrics(host_full.layer_outputs[layer],
                                             host_sequential.layer_output);
        const auto full_gpu_error = metrics(gpu_full.layer_outputs[layer],
                                            gpu_sequential.layer_output);
        compare_checkpoint((std::string("  host-isolated vs external layer-")
                            + std::to_string(layer)).c_str(),
                           host_isolated.layer_output, expected_output, 5.0e-2F);
        compare_checkpoint((std::string("  host-sequential vs external layer-")
                            + std::to_string(layer)).c_str(),
                           host_sequential.layer_output, expected_output, 5.0e-2F);
        compare_checkpoint((std::string("  gpu-isolated vs external layer-")
                            + std::to_string(layer)).c_str(),
                           gpu_isolated.layer_output, expected_output, 5.0e-2F);
        compare_checkpoint((std::string("  gpu-sequential vs external layer-")
                            + std::to_string(layer)).c_str(),
                           gpu_sequential.layer_output, expected_output, 5.0e-2F);
        print_composition_delta("  sequential GPU-vs-host", host_gpu_seq_error);
        print_composition_delta("  execute_forward_host vs reconstructed-sequential",
                                full_host_error);
        print_composition_delta("  execute_forward_gpu vs reconstructed-sequential",
                                full_gpu_error);
        full_matches_reconstructed = full_matches_reconstructed
            && full_host_error.max_abs == 0.0F
            && full_gpu_error.max_abs == 0.0F;
        passed = passed && host_iso_error.max_abs <= 5.0e-2F
            && host_seq_error.max_abs <= 5.0e-2F
            && gpu_iso_error.max_abs <= 5.0e-2F
            && gpu_seq_error.max_abs <= 5.0e-2F;

        for (const auto& field : kCompositionFields) {
            const auto host_delta = metrics(host_sequential.*(field.second),
                                            host_isolated.*(field.second));
            const auto gpu_delta = metrics(gpu_sequential.*(field.second),
                                           gpu_isolated.*(field.second));
            print_composition_delta((std::string("  host sequential-vs-isolated ")
                                     + field.first).c_str(), host_delta);
            print_composition_delta((std::string("  gpu sequential-vs-isolated ")
                                     + field.first).c_str(), gpu_delta);
            if (!reported_first_host_nonzero && host_delta.max_abs != 0.0F) {
                std::cout << "  FIRST NONZERO HOST DIVERGENCE: layer=" << layer
                          << " stage=" << field.first << '\n';
                reported_first_host_nonzero = true;
            }
            if (!reported_first_gpu_nonzero && gpu_delta.max_abs != 0.0F) {
                std::cout << "  FIRST NONZERO GPU DIVERGENCE: layer=" << layer
                          << " stage=" << field.first << '\n';
                reported_first_gpu_nonzero = true;
            }
        }

        host_sequential_input = host_sequential.layer_output;
        gpu_sequential_input = gpu_sequential.layer_output;
    }

    std::cout << "M4-B15 full-forward controls: host layers=" << host_full.layer_outputs.size()
              << " gpu layers=" << gpu_full.layer_outputs.size()
              << " reconstructed parity=" << (full_matches_reconstructed ? "PASS" : "FAIL") << '\n';
    std::cout << "M4-B15 MI50 summary: "
              << (passed && full_matches_reconstructed ? "layer-output gates PASS" : "layer-output gates FAIL")
              << "; first nonzero divergences are reported above\n";
    return passed && full_matches_reconstructed ? 0 : 1;
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

void report_ffn_result(const char* label,
                      const miinfer::Qwen3FfnProbeTrace& actual,
                      const miinfer::Qwen3LayerTrace& expected) {
    const auto gate = metrics(actual.gate, expected.gate);
    const auto up = metrics(actual.up, expected.up);
    const auto swiglu = metrics(actual.swiglu, expected.swiglu);
    const auto down = metrics(actual.ffn_output, expected.ffn_output);
    const auto output = metrics(actual.layer_output, expected.layer_output);
    std::cout << "ffn-probe " << label
              << ": gate=" << gate.max_abs
              << " up=" << up.max_abs
              << " swiglu=" << swiglu.max_abs
              << " down=" << down.max_abs
              << " layer=" << output.max_abs << '\n';
}

std::vector<miinfer::Q8_1Block> capture_q8_input(
    const std::vector<float>& input, const char* label) {
    const auto elements = input.size();
    const auto blocks = elements / miinfer::kQ8_1BlockSize;
    float* device_f32 = nullptr;
    __half* device_f16 = nullptr;
    miinfer::Q8_1Block* device_from_f16 = nullptr;
    miinfer::Q8_1Block* device_from_f32 = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&device_f32, elements * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&device_f16, elements * sizeof(__half)));
    MIINFER_HIP_CHECK(hipMalloc(&device_from_f16, blocks * sizeof(miinfer::Q8_1Block)));
    MIINFER_HIP_CHECK(hipMalloc(&device_from_f32, blocks * sizeof(miinfer::Q8_1Block)));
    MIINFER_HIP_CHECK(hipMemcpy(device_f32, input.data(), elements * sizeof(float),
                                hipMemcpyHostToDevice));
    miinfer::launch_qwen3_f32_to_f16(device_f32, device_f16,
                                     static_cast<std::uint32_t>(elements));
    miinfer::launch_q8_1_quantize(device_f16, device_from_f16,
                                  static_cast<int>(elements));
    miinfer::launch_q8_1_quantize_f32(device_f32, device_from_f32,
                                      static_cast<int>(elements));
    std::vector<miinfer::Q8_1Block> from_f16(blocks), from_f32(blocks);
    MIINFER_HIP_CHECK(hipMemcpy(from_f16.data(), device_from_f16,
                                blocks * sizeof(miinfer::Q8_1Block), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(from_f32.data(), device_from_f32,
                                blocks * sizeof(miinfer::Q8_1Block), hipMemcpyDeviceToHost));
    std::size_t differing_blocks = 0;
    std::size_t differing_qs = 0;
    float max_scale_delta = 0.0F;
    for (std::size_t block = 0; block < blocks; ++block) {
        bool differs = __half2float(from_f16[block].d) != __half2float(from_f32[block].d)
            || __half2float(from_f16[block].s) != __half2float(from_f32[block].s);
        max_scale_delta = std::max(max_scale_delta,
            std::fabs(__half2float(from_f16[block].d) - __half2float(from_f32[block].d)));
        for (int index = 0; index < miinfer::kQ8_1BlockSize; ++index) {
            if (from_f16[block].qs[index] != from_f32[block].qs[index]) {
                ++differing_qs;
                differs = true;
            }
        }
        if (differs) ++differing_blocks;
    }
    std::cout << label << " Q8 blocks (F16-input vs F32-input): differing_blocks="
              << differing_blocks << '/' << blocks
              << " differing_qs=" << differing_qs
              << " max_d_delta=" << max_scale_delta << '\n';
    MIINFER_HIP_CHECK(hipFree(device_f32));
    MIINFER_HIP_CHECK(hipFree(device_f16));
    MIINFER_HIP_CHECK(hipFree(device_from_f16));
    MIINFER_HIP_CHECK(hipFree(device_from_f32));
    return from_f16;
}

void report_down_contract_blocks(
    const miinfer::Qwen3GpuPlan& plan,
    std::size_t layer,
    const std::vector<miinfer::Q8_1Block>& input,
    const std::vector<float>& expected,
    const std::vector<float>& current) {
    const auto current_metrics = metrics(current, expected);
    const auto& config = plan.model().config();
    const int blocks_per_row = static_cast<int>(config.intermediate_size / 32U);
    const auto* weights = reinterpret_cast<const miinfer::Q4_0HostBlock*>(
        plan.model().layers()[layer].down.data());
    struct BlockReport {
        int block = 0;
        int q8_sum = 0;
        float activation_d = 0.0F;
        float stored_s = 0.0F;
        float exact_sum_scaled = 0.0F;
        float correction_delta = 0.0F;
        float current = 0.0F;
        float exact = 0.0F;
    };
    std::vector<BlockReport> reports;
    reports.reserve(blocks_per_row);
    const auto row = static_cast<int>(current_metrics.max_index);
    float absolute_correction_error = 0.0F;
    float signed_correction_error = 0.0F;
    for (int block = 0; block < blocks_per_row; ++block) {
        const auto& weight = weights[static_cast<std::size_t>(row) * blocks_per_row + block];
        const auto& activation = input[static_cast<std::size_t>(block)];
        const float weight_d = miinfer::fp16_bits_to_float(weight.d_bits);
        const float activation_d = __half2float(activation.d);
        const float stored_s = __half2float(activation.s);
        int q8_sum = 0;
        int raw_dot = 0;
        for (int index = 0; index < 32; ++index) {
            const auto packed = weight.qs[index < 16 ? index : index - 16];
            const int nibble = index < 16 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            const int q8 = static_cast<int>(activation.qs[index]);
            q8_sum += q8;
            raw_dot += nibble * q8;
        }
        const float exact_sum_scaled = activation_d * static_cast<float>(q8_sum);
        const float correction_delta = stored_s - exact_sum_scaled;
        const float current_value = weight_d * (activation_d * static_cast<float>(raw_dot)
                                                - 8.0F * stored_s);
        const float exact_value = weight_d * activation_d
                                  * static_cast<float>(raw_dot - 8 * q8_sum);
        absolute_correction_error += std::fabs(weight_d * 8.0F * correction_delta);
        signed_correction_error += weight_d * -8.0F * correction_delta;
        reports.push_back({block, q8_sum, activation_d, stored_s, exact_sum_scaled,
                           correction_delta, current_value, exact_value});
    }
    std::sort(reports.begin(), reports.end(), [](const auto& lhs, const auto& rhs) {
        return std::fabs(lhs.current - lhs.exact) > std::fabs(rhs.current - rhs.exact);
    });
    std::cout << "down-contract row=" << row
              << " expected=" << expected[row]
              << " current=" << current[row]
              << " absolute_correction_error=" << absolute_correction_error
              << " signed_correction_error=" << signed_correction_error << '\n';
    const auto count = std::min<std::size_t>(reports.size(), 8);
    for (std::size_t index = 0; index < count; ++index) {
        const auto& report = reports[index];
        std::cout << "  block=" << report.block
                  << " q8_sum=" << report.q8_sum
                  << " d=" << report.activation_d
                  << " stored_s=" << report.stored_s
                  << " exact_d_sum=" << report.exact_sum_scaled
                  << " correction_delta=" << report.correction_delta
                  << " current=" << report.current
                  << " exact=" << report.exact << '\n';
    }
}

void report_hybrid_ffn_probes(const miinfer::Qwen3GpuPlan& plan,
                              std::size_t layer,
                              const miinfer::Qwen3LayerTrace& reference) {
    using Precision = miinfer::Qwen3ProjectionPrecision;
    const std::array<std::pair<const char*, Precision>, 4> variants{
        std::pair{"f16-q8-f16", Precision::f16_input_q8_f16_output},
        std::pair{"f32-q8-f16", Precision::f32_input_q8_f16_output},
        std::pair{"f16-q8-f32", Precision::f16_input_q8_f32_output},
        std::pair{"f32-q8-f32", Precision::f32_input_q8_f32_output},
    };
    std::cout << "layer-6 FFN projection precision probes (GPU vs host reference):\n";
    (void)capture_q8_input(reference.ffn_norm, "layer-6 gate/up");
    const auto down_input = capture_q8_input(reference.swiglu, "layer-6 down");
    for (const auto& variant : variants) {
        const auto result = miinfer::execute_qwen3_ffn_gpu_probe(
            plan, layer, reference.ffn_input, reference.ffn_norm,
            {}, {}, {}, {}, variant.second);
        report_ffn_result(variant.first, result, reference);
    }

    std::cout << "layer-6 FFN hybrid probes (host tensor injection):\n";
    const auto gate_up = miinfer::execute_qwen3_ffn_gpu_probe(
        plan, layer, reference.ffn_input, reference.ffn_norm,
        reference.gate, reference.up);
    report_ffn_result("reference-gate-up -> GPU-SwiGLU", gate_up, reference);
    const auto swiglu = miinfer::execute_qwen3_ffn_gpu_probe(
        plan, layer, reference.ffn_input, reference.ffn_norm,
        {}, {}, reference.swiglu);
    report_ffn_result("reference-SwiGLU -> GPU-down", swiglu, reference);
    const auto down = miinfer::execute_qwen3_ffn_gpu_probe(
        plan, layer, reference.ffn_input, reference.ffn_norm,
        {}, {}, {}, reference.ffn_output);
    report_ffn_result("reference-down -> GPU-residual", down, reference);

    const auto report_contract = [](const char* name, const Metrics& result) {
        std::cout << "down-contract " << name
                  << ": max_abs=" << result.max_abs
                  << " mean_abs=" << result.mean_abs
                  << " rmse=" << result.rmse
                  << " max_rel=" << result.max_rel
                  << " max_index=" << result.max_index
                  << " actual_at_max=" << result.actual_at_max
                  << " expected_at_max=" << result.expected_at_max << '\n';
    };
    const auto report_contract_set = [&](const char* input_label,
                                         const miinfer::Qwen3DownProjectionContractTrace& contract) {
        const auto current = metrics(contract.current_s_correction, reference.ffn_output);
        const auto exact = metrics(contract.exact_sum_correction, reference.ffn_output);
        const auto direct = metrics(contract.direct_signed_oracle, reference.ffn_output);
        const auto exact_vs_direct = metrics(contract.exact_sum_correction,
                                             contract.direct_signed_oracle);
        std::cout << "layer-6 down-projection contracts (F32 output, "
                   << input_label << " Q8):\n";
        report_contract("current-fp16-s", current);
        report_contract("exact-integer-sum", exact);
        report_contract("direct-signed-oracle", direct);
        report_contract("exact-vs-direct", exact_vs_direct);
        return current;
    };
    const auto contract = miinfer::execute_qwen3_down_projection_contract_probe(
        plan, layer, reference.swiglu, false);
    (void)report_contract_set("F16-input", contract);
    const auto f32_contract = miinfer::execute_qwen3_down_projection_contract_probe(
        plan, layer, reference.swiglu, true);
    (void)report_contract_set("F32-input", f32_contract);
    report_down_contract_blocks(plan, layer, down_input, reference.ffn_output,
                                contract.current_s_correction);
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
        if (layer == 6) report_hybrid_ffn_probes(plan, layer, host_trace);
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
        if (argc == 4 && std::string(argv[3]) == "--composition-diagnostic") {
            return run_composition_diagnostic(argv[1], argv[2]);
        }
        return run(argv[1], argc == 3 ? std::filesystem::path(argv[2])
                                      : std::filesystem::path{});
    } catch (const std::exception& error) {
        std::cerr << "qwen3 full-forward GPU test error: " << error.what() << '\n';
        return 1;
    }
}
