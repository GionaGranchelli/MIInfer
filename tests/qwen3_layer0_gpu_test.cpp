#include "miinfer/qwen3_gpu_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Metrics {
    float max_abs = 0.0F;
    float mean_abs = 0.0F;
    std::size_t index = 0;
};

Metrics compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("checkpoint size mismatch");
    Metrics result;
    double sum = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            result.max_abs = std::numeric_limits<float>::infinity();
            result.index = i;
            return result;
        }
        const float error = std::fabs(actual[i] - expected[i]);
        sum += error;
        if (error > result.max_abs) {
            result.max_abs = error;
            result.index = i;
        }
    }
    result.mean_abs = static_cast<float>(sum / actual.size());
    return result;
}

std::string trace_filename(const std::filesystem::path& directory, std::size_t index) {
    const std::array prefixes{
        std::to_string(index) + '-',
        std::string("pos-0-") + std::to_string(index) + ".",
    };
    for (const auto& prefix : prefixes) {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.path().filename().string().rfind(prefix, 0) == 0) {
                return entry.path().string();
            }
        }
    }
    throw std::runtime_error("missing reference checkpoint " + std::to_string(index));
}

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid F32 checkpoint " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file) throw std::runtime_error("short read " + path.string());
    return values;
}

std::vector<const std::vector<float>*> checkpoints(const miinfer::Qwen3LayerTrace& trace) {
    return {
        &trace.embedding, &trace.attn_rms, &trace.attn_norm,
        &trace.q_projection, &trace.q_reshape, &trace.q_rms, &trace.q_normed,
        &trace.q_rope, &trace.v_projection, &trace.v_reshape,
        &trace.k_projection, &trace.k_reshape, &trace.k_rms, &trace.k_normed,
        &trace.k_rope, &trace.k_view, &trace.v_view, &trace.q_view,
        &trace.q_permuted, &trace.attention_output, &trace.ffn_input,
        &trace.ffn_rms, &trace.ffn_norm, &trace.gate, &trace.up,
        &trace.swiglu, &trace.ffn_output, &trace.layer_output,
    };
}

const char* names[] = {
    "embedding", "attn_rms", "attn_norm", "q_projection", "q_reshape", "q_rms",
    "q_normed", "q_rope", "v_projection", "v_reshape", "k_projection", "k_reshape",
    "k_rms", "k_normed", "k_rope", "k_view", "v_view", "q_view", "q_permuted",
    "attention_output", "ffn_input", "ffn_rms", "ffn_norm", "gate", "up", "swiglu",
    "ffn_output", "layer_output",
};

float abs_tolerance(std::size_t i) {
    if (i <= 2) return 2.0e-4F;
    if ((i >= 3 && i <= 4) || (i >= 8 && i <= 11)) return 3.0e-4F;
    if (i == 5 || i == 12) return 6.0e-3F;
    if ((i >= 6 && i <= 7) || (i >= 13 && i <= 18)) return 6.0e-2F;
    return 2.0e-2F;
}

float relative_tolerance(std::size_t i) {
    if (i <= 2) return 2.0e-4F;
    if ((i >= 3 && i <= 4) || (i >= 8 && i <= 11)) return 2.0e-2F;
    if (i == 5 || i == 12) return 1.0e-1F;
    if ((i >= 6 && i <= 7) || (i >= 13 && i <= 18)) return 1.5e-1F;
    if (i == 22 || i == 25) return 2.5e-1F;
    if (i >= 26) return 4.0e-1F;
    return 5.0e-1F;
}

bool within(const Metrics& metrics, const std::vector<float>& expected, std::size_t i) {
    const float reference_magnitude = std::fabs(expected[metrics.index]);
    const float meaningful_rel = reference_magnitude >= 1.0e-2F
        ? metrics.max_abs / reference_magnitude : 0.0F;
    return std::isfinite(metrics.max_abs)
        && metrics.max_abs <= abs_tolerance(i)
        && meaningful_rel <= relative_tolerance(i);
}

int run(const std::filesystem::path& model_path, const std::filesystem::path* reference_path,
        bool mutation_test) {
    if (mutation_test) setenv("MIINFER_GPU_LAYER_MUTATE", "swap-gate-up", 1);
    const auto model = miinfer::Qwen3Model::load(model_path.string());
    const auto host = miinfer::execute_qwen3_layer0_host(model, 14990);
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    const auto gpu = miinfer::execute_qwen3_layer0_gpu(plan, 14990);
    const auto host_points = checkpoints(host);
    const auto gpu_points = checkpoints(gpu);
    bool checkpoints_pass = true;
    std::size_t first_bad_checkpoint = host_points.size();
    for (std::size_t i = 0; i < host_points.size(); ++i) {
        const auto metrics = compare(*gpu_points[i], *host_points[i]);
        const bool pass = within(metrics, *host_points[i], i);
        std::cout << std::left << std::setw(18) << names[i]
                  << (pass ? "PASS" : "FAIL")
                  << " gpu_vs_host_max_abs=" << std::scientific << metrics.max_abs
                  << " mean_abs=" << metrics.mean_abs
                  << " index=" << std::dec << metrics.index << '\n';
        checkpoints_pass = checkpoints_pass && pass;
        if (!pass && first_bad_checkpoint == host_points.size()) first_bad_checkpoint = i;
        if (reference_path != nullptr) {
            const auto reference = read_f32(trace_filename(*reference_path, i));
            const auto reference_metrics = compare(*gpu_points[i], reference);
            const bool reference_pass = within(reference_metrics, reference, i);
            std::cout << std::left << std::setw(18) << names[i]
                      << (reference_pass ? "PASS" : "FAIL")
                      << " gpu_vs_reference_max_abs=" << std::scientific
                      << reference_metrics.max_abs << " index=" << std::dec
                      << reference_metrics.index << '\n';
            checkpoints_pass = checkpoints_pass && reference_pass;
            if (!reference_pass && first_bad_checkpoint == host_points.size()) {
                first_bad_checkpoint = i;
            }
        }
    }
    const auto score_metrics = compare(gpu.attention_scores, host.attention_scores);
    const auto probability_metrics = compare(gpu.attention_probabilities, host.attention_probabilities);
    std::cout << "attention scores max_abs=" << std::scientific << score_metrics.max_abs
              << " probabilities max_abs=" << probability_metrics.max_abs << '\n';
    const bool attention_pass =
        score_metrics.max_abs <= 2.0e-2F && probability_metrics.max_abs <= 1.0e-6F;
    if (mutation_test) {
        if (first_bad_checkpoint < host_points.size()) {
            std::cout << "first divergent checkpoint: " << names[first_bad_checkpoint] << '\n';
        }
        std::cout << "GPU composition mutation discriminator: "
                  << (!checkpoints_pass ? "PASS (mutation detected)\n" : "FAIL (mutation not detected)\n");
        unsetenv("MIINFER_GPU_LAYER_MUTATE");
        return checkpoints_pass ? 1 : 0;
    }
    const bool all_pass = checkpoints_pass && attention_pass;
    std::cout << "attention score/probability parity: " << (attention_pass ? "PASS" : "FAIL") << '\n';
    if (!all_pass && first_bad_checkpoint < host_points.size()) {
        std::cout << "first divergent checkpoint: " << names[first_bad_checkpoint] << '\n';
    }
    std::cout << (all_pass ? "layer-0 GPU comparison: PASS\n" : "layer-0 GPU comparison: FAIL\n");
    return all_pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cout << "qwen3 layer-0 GPU comparison: SKIP (model path not supplied)\n";
        return 0;
    }
    try {
        if (argc == 3 && std::string(argv[2]) == "--self-test-mutation") {
            return run(argv[1], nullptr, true);
        }
        const std::filesystem::path reference = argc == 3 ? argv[2] : "";
        return run(argv[1], argc == 3 ? &reference : nullptr, false);
    } catch (const std::exception& error) {
        std::cerr << "layer-0 GPU comparison error: " << error.what() << '\n';
        return 1;
    }
}
