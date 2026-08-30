#include "miinfer/qwen3_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Checkpoint {
    const char* name;
    std::size_t file_index;
    float abs_tolerance;
    float rel_tolerance;
    const std::vector<float>* actual = nullptr;
};

struct Metrics {
    float max_abs = 0.0F;
    float mean_abs = 0.0F;
    float max_rel = 0.0F;
    float rmse = 0.0F;
    std::size_t max_index = 0;
    float reference_at_max = 0.0F;
    float actual_at_max = 0.0F;
    bool finite = true;
};

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open trace file: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("trace file is not a whole number of F32 values: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file) throw std::runtime_error("short read from trace file: " + path.string());
    return values;
}

Metrics compare(const std::vector<float>& actual, const std::vector<float>& reference) {
    if (actual.size() != reference.size()) throw std::runtime_error("checkpoint size mismatch");
    Metrics result;
    double squared = 0.0;
    double absolute = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) result.finite = false;
        const float abs_error = std::fabs(actual[i] - reference[i]);
        // Relative error is only meaningful away from zero.  Small reference
        // values are governed by the absolute-error gate instead of creating
        // an unbounded ratio from harmless quantization noise.
        const float denominator = std::fabs(reference[i]);
        const float rel_error = abs_error / denominator;
        if (abs_error > result.max_abs) {
            result.max_abs = abs_error;
            result.max_index = i;
            result.reference_at_max = reference[i];
            result.actual_at_max = actual[i];
        }
        if (std::fabs(reference[i]) >= 1.0e-2F) result.max_rel = std::max(result.max_rel, rel_error);
        absolute += abs_error;
        squared += static_cast<double>(abs_error) * abs_error;
    }
    result.mean_abs = static_cast<float>(absolute / actual.size());
    result.rmse = static_cast<float>(std::sqrt(squared / actual.size()));
    return result;
}

bool passes(const Metrics& metrics, const Checkpoint& checkpoint) {
    return metrics.finite
        && metrics.max_abs <= checkpoint.abs_tolerance
        && metrics.max_rel <= checkpoint.rel_tolerance;
}

std::string trace_filename(const std::filesystem::path& directory, std::size_t index) {
    std::ostringstream prefix;
    prefix << index << '-';
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto filename = entry.path().filename().string();
        if (filename.rfind(prefix.str(), 0) == 0) return entry.path().string();
    }
    throw std::runtime_error("missing checkpoint " + std::to_string(index));
}

void print_metrics(const Checkpoint& checkpoint, const Metrics& metrics, bool pass) {
    std::cout << std::left << std::setw(18) << checkpoint.name
              << (pass ? "PASS" : "FAIL")
              << " max_abs=" << std::scientific << metrics.max_abs
              << " mean_abs=" << metrics.mean_abs
              << " max_rel=" << metrics.max_rel
              << " rmse=" << metrics.rmse
              << " index=" << std::dec << metrics.max_index
              << " ref=" << metrics.reference_at_max
              << " actual=" << metrics.actual_at_max << '\n';
}

int run(const std::filesystem::path& model_path, const std::filesystem::path& trace_path,
        std::uint32_t token) {
    const auto model = miinfer::Qwen3Model::load(model_path.string());
    const auto actual = miinfer::execute_qwen3_layer0_host(model, token);

    const std::vector<const std::vector<float>*> vectors = {
        &actual.embedding, &actual.attn_rms, &actual.attn_norm,
        &actual.q_projection, &actual.q_reshape, &actual.q_rms,
        &actual.q_normed, &actual.q_rope, &actual.v_projection,
        &actual.v_reshape, &actual.k_projection, &actual.k_reshape,
        &actual.k_rms, &actual.k_normed, &actual.k_rope, &actual.k_view,
        &actual.v_view, &actual.q_view, &actual.q_permuted,
        &actual.attention_output, &actual.ffn_input, &actual.ffn_rms,
        &actual.ffn_norm, &actual.gate, &actual.up, &actual.swiglu,
        &actual.ffn_output, &actual.layer_output,
    };
    const char* names[] = {
        "embedding", "attn_rms", "attn_norm", "q_projection", "q_reshape", "q_rms",
        "q_normed", "q_rope", "v_projection", "v_reshape", "k_projection", "k_reshape",
        "k_rms", "k_normed", "k_rope", "k_view", "v_view", "q_view", "q_permuted",
        "attention_output", "ffn_input", "ffn_rms", "ffn_norm", "gate", "up", "swiglu",
        "ffn_output", "layer_output",
    };
    bool all_pass = true;
    bool reported_first_failure = false;
    std::vector<std::vector<float>> references;
    references.reserve(vectors.size());
    for (std::size_t i = 0; i < vectors.size(); ++i) references.push_back(read_f32(trace_filename(trace_path, i)));

    for (std::size_t i = 0; i < vectors.size(); ++i) {
        // Tolerances are frozen per checkpoint class for this CPU composition
        // gate; they are intentionally not a single universal epsilon.
        float abs_tolerance = 2.0e-2F;
        float rel_tolerance = 5.0e-1F;
        if (i <= 2) {
            abs_tolerance = 2.0e-4F;
            rel_tolerance = 2.0e-4F;
        } else if ((i >= 3 && i <= 4) || (i >= 8 && i <= 11)) {
            abs_tolerance = 2.0e-4F;
            rel_tolerance = 1.0e-2F;
        } else if (i == 5 || i == 12) {
            abs_tolerance = 5.0e-3F;
            rel_tolerance = 1.0e-1F;
        } else if ((i >= 6 && i <= 7) || (i >= 13 && i <= 18)) {
            abs_tolerance = 2.5e-2F;
            rel_tolerance = 1.5e-1F;
        } else if (i == 22 || i == 25) {
            rel_tolerance = 1.5e-1F;
        } else if (i >= 26) {
            rel_tolerance = 4.0e-1F;
        }
        const Checkpoint checkpoint{names[i], i, abs_tolerance, rel_tolerance, vectors[i]};
        const auto metrics = compare(*vectors[i], references[i]);
        const bool pass = passes(metrics, checkpoint);
        print_metrics(checkpoint, metrics, pass);
        if (!pass && !reported_first_failure) {
            std::cout << "first divergent checkpoint: " << checkpoint.name << '\n';
            reported_first_failure = true;
        }
        all_pass = all_pass && pass;
    }
    std::cout << (all_pass ? "layer-0 host comparison: PASS\n" : "layer-0 host comparison: FAIL\n");
    return all_pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test-mutation") {
        const std::vector<float> reference{1.0F, 2.0F, 3.0F};
        std::vector<float> mutated = reference;
        mutated[1] += 1.0F;
        const Checkpoint checkpoint{"mutation", 0, 1.0e-4F, 1.0e-4F, &mutated};
        const auto metrics = compare(mutated, reference);
        if (passes(metrics, checkpoint)) {
            std::cerr << "comparator mutation test failed to detect a semantic change\n";
            return 1;
        }
        std::cout << "comparator mutation test: PASS\n";
        return 0;
    }
    if (argc < 3 || argc > 4) {
        std::cout << "qwen3 layer-0 host comparison: SKIP (model and trace paths not supplied)\n";
        return 0;
    }
    try {
        const auto token = argc == 4 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 14990U;
        return run(argv[1], argv[2], token);
    } catch (const std::exception& error) {
        std::cerr << "layer-0 host comparison error: " << error.what() << '\n';
        return 1;
    }
}
