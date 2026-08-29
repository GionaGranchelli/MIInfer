#include "miinfer/fp16_gemv.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace miinfer {

const std::vector<GemvShape>& qwen3_gemv_shapes() {
    static const std::vector<GemvShape> shapes = {
        {"Q", "Q projection", 4096, 4096},
        {"K", "K projection", 1024, 4096},
        {"V", "V projection", 1024, 4096},
        {"O", "Output projection", 4096, 4096},
        {"G", "FFN gate projection", 12288, 4096},
        {"U", "FFN up projection", 12288, 4096},
        {"D", "FFN down projection", 4096, 12288},
    };
    return shapes;
}

void generate_fp16_gemv_data(
    int m,
    int k,
    std::uint32_t seed,
    std::vector<__half>& weights,
    std::vector<__half>& input) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
    weights.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
    input.resize(static_cast<std::size_t>(k));
    for (auto& value : weights) {
        value = __float2half_rn(distribution(generator));
    }
    for (auto& value : input) {
        value = __float2half_rn(distribution(generator));
    }
}

std::vector<float> fp16_gemv_cpu_reference(
    const std::vector<__half>& weights,
    const std::vector<__half>& input,
    int m,
    int k) {
    std::vector<float> output(static_cast<std::size_t>(m), 0.0F);
    for (int row = 0; row < m; ++row) {
        float sum = 0.0F;
        const auto offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(k);
        for (int column = 0; column < k; ++column) {
            sum += __half2float(weights[offset + static_cast<std::size_t>(column)])
                   * __half2float(input[static_cast<std::size_t>(column)]);
        }
        output[static_cast<std::size_t>(row)] = sum;
    }
    return output;
}

Fp16GemvMetrics evaluate_fp16_gemv(
    const std::vector<__half>& output,
    const std::vector<float>& reference,
    float absolute_tolerance,
    float relative_tolerance) {
    Fp16GemvMetrics metrics;
    if (output.size() != reference.size() || output.empty()) {
        return metrics;
    }

    double absolute_sum = 0.0;
    double dot = 0.0;
    double output_norm = 0.0;
    double reference_norm = 0.0;
    bool within_tolerance = true;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const float actual = __half2float(output[index]);
        const float expected = reference[index];
        if (std::isnan(actual) || std::isnan(expected)) {
            metrics.nan_detected = true;
        }
        if (std::isinf(actual) || std::isinf(expected)) {
            metrics.inf_detected = true;
        }
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            within_tolerance = false;
            continue;
        }
        const double absolute_error = std::fabs(static_cast<double>(actual) - expected);
        const double relative_error = absolute_error
                                      / std::max(std::fabs(static_cast<double>(expected)), 1.0e-6);
        metrics.max_abs_error = std::max(metrics.max_abs_error, absolute_error);
        metrics.max_relative_error = std::max(metrics.max_relative_error, relative_error);
        absolute_sum += absolute_error;
        dot += static_cast<double>(actual) * expected;
        output_norm += static_cast<double>(actual) * actual;
        reference_norm += static_cast<double>(expected) * expected;
        if (absolute_error > static_cast<double>(absolute_tolerance)
            && relative_error > static_cast<double>(relative_tolerance)) {
            within_tolerance = false;
        }
    }
    metrics.mean_abs_error = absolute_sum / static_cast<double>(output.size());
    const double denominator = std::sqrt(output_norm) * std::sqrt(reference_norm);
    metrics.cosine_similarity = denominator > 0.0 ? dot / denominator : 0.0;
    metrics.pass = within_tolerance && !metrics.nan_detected && !metrics.inf_detected;
    return metrics;
}

}  // namespace miinfer
