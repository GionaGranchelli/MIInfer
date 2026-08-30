#include "miinfer/qwen3_gpu_layer.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kCheckpointCount = 28;
const std::vector<std::uint32_t> kTokens{14990U, 42U, 31415U, 2718U};

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

float tolerance(std::size_t checkpoint) {
    if (checkpoint <= 2) return 2.0e-4F;
    if ((checkpoint >= 3 && checkpoint <= 4) || (checkpoint >= 8 && checkpoint <= 11)) {
        return 1.0e-3F;
    }
    if (checkpoint == 5 || checkpoint == 12) return 1.5e-2F;
    if ((checkpoint >= 6 && checkpoint <= 7) || (checkpoint >= 13 && checkpoint <= 18)) {
        return 1.0e-1F;
    }
    return 1.0e-1F;
}

float max_abs(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) return INFINITY;
    float result = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) return INFINITY;
        result = std::max(result, std::fabs(actual[i] - expected[i]));
    }
    return result;
}

bool compare_trace(const miinfer::Qwen3LayerTrace& gpu, const miinfer::Qwen3LayerTrace& host,
                   std::size_t* first_bad = nullptr) {
    const auto gpu_points = checkpoints(gpu);
    const auto host_points = checkpoints(host);
    bool passed = true;
    for (std::size_t i = 0; i < kCheckpointCount; ++i) {
        const float error = max_abs(*gpu_points[i], *host_points[i]);
        const bool current = error <= tolerance(i);
        if (!current && first_bad != nullptr && *first_bad == kCheckpointCount) *first_bad = i;
        passed = passed && current;
    }
    return passed;
}

bool cache_slots_preserved(
    std::span<const float> before, std::span<const float> after,
    std::size_t kv_heads, std::size_t head_dim, std::size_t capacity, std::size_t written_position) {
    if (before.size() != after.size()) return false;
    for (std::size_t index = 0; index < after.size(); ++index) {
        const std::size_t slot = (index / head_dim) % capacity;
        if (slot != written_position && before[index] != after[index]) return false;
    }
    (void) kv_heads;
    return true;
}

bool attention_contract(const miinfer::Qwen3LayerTrace& trace, std::size_t heads, std::size_t length) {
    if (trace.attention_scores.size() != heads * length
        || trace.attention_probabilities.size() != heads * length) return false;
    for (std::size_t head = 0; head < heads; ++head) {
        float sum = 0.0F;
        for (std::size_t position = 0; position < length; ++position) {
            const auto probability = trace.attention_probabilities[head * length + position];
            if (!std::isfinite(probability) || probability < 0.0F) return false;
            sum += probability;
        }
        if (std::fabs(sum - 1.0F) > 2.0e-5F) return false;
    }
    return true;
}

bool run_sequence(const std::string& model_path, bool mutation_test) {
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    miinfer::Qwen3Layer0KvCache host_cache(config.kv_heads, config.head_dim, kTokens.size());
    miinfer::Qwen3Layer0GpuKvCache gpu_cache(config.kv_heads, config.head_dim, kTokens.size());
    std::vector<float> previous_host_keys(host_cache.keys().begin(), host_cache.keys().end());
    std::vector<float> previous_host_values(host_cache.values().begin(), host_cache.values().end());
    std::vector<float> previous_gpu_keys;
    std::vector<float> previous_gpu_values;
    bool passed = true;

    for (std::size_t position = 0; position < kTokens.size(); ++position) {
        const auto host = miinfer::execute_qwen3_layer0_host(
            model, kTokens[position], position, host_cache);
        const auto gpu = miinfer::execute_qwen3_layer0_gpu(
            plan, kTokens[position], position, gpu_cache);
        std::size_t first_bad = kCheckpointCount;
        const bool trace_pass = compare_trace(gpu, host, &first_bad);
        if (!trace_pass) {
            const auto gpu_points = checkpoints(gpu);
            const auto host_points = checkpoints(host);
            std::cout << "position " << position << " first_bad=" << first_bad
                      << " error=" << max_abs(*gpu_points[first_bad], *host_points[first_bad])
                      << " attention_score_error=" << max_abs(gpu.attention_scores, host.attention_scores)
                      << " attention_probability_error="
                      << max_abs(gpu.attention_probabilities, host.attention_probabilities) << '\n';
        }
        passed = passed && trace_pass && host_cache.length() == position + 1
                 && gpu_cache.length() == position + 1
                 && attention_contract(gpu, config.attention_heads, position + 1);

        const auto current_gpu_keys = gpu_cache.snapshot_keys();
        const auto current_gpu_values = gpu_cache.snapshot_values();
        passed = passed && cache_slots_preserved(
            previous_gpu_keys.empty() ? current_gpu_keys : std::span<const float>(previous_gpu_keys),
            current_gpu_keys, config.kv_heads, config.head_dim, gpu_cache.capacity(), position);
        passed = passed && cache_slots_preserved(
            previous_gpu_values.empty() ? current_gpu_values : std::span<const float>(previous_gpu_values),
            current_gpu_values, config.kv_heads, config.head_dim, gpu_cache.capacity(), position);
        const auto host_keys = host_cache.keys();
        const auto host_values = host_cache.values();
        passed = passed && cache_slots_preserved(
            previous_host_keys, std::vector<float>(host_keys.begin(), host_keys.end()),
            config.kv_heads, config.head_dim, host_cache.capacity(), position);
        passed = passed && cache_slots_preserved(
            previous_host_values, std::vector<float>(host_values.begin(), host_values.end()),
            config.kv_heads, config.head_dim, host_cache.capacity(), position);
        previous_host_keys.assign(host_keys.begin(), host_keys.end());
        previous_host_values.assign(host_values.begin(), host_values.end());
        previous_gpu_keys = current_gpu_keys;
        previous_gpu_values = current_gpu_values;
        passed = passed && max_abs(
            std::span<const float>(current_gpu_keys),
            std::span<const float>(previous_host_keys)) <= 1.0e-1F;
        passed = passed && max_abs(
            std::span<const float>(current_gpu_values),
            std::span<const float>(previous_host_values)) <= 1.0e-1F;
        if (!trace_pass && mutation_test) {
            std::cout << "first divergent checkpoint: "
                      << (first_bad < kCheckpointCount ? std::to_string(first_bad) : "unknown") << '\n';
        }
    }
    return passed;
}

bool reset_test(const std::string& model_path) {
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    miinfer::Qwen3Layer0KvCache host_cache(config.kv_heads, config.head_dim, kTokens.size());
    miinfer::Qwen3Layer0GpuKvCache gpu_cache(config.kv_heads, config.head_dim, kTokens.size());
    std::vector<std::vector<float>> first_outputs;
    for (std::size_t position = 0; position < kTokens.size(); ++position) {
        (void) miinfer::execute_qwen3_layer0_host(model, kTokens[position], position, host_cache);
        first_outputs.push_back(miinfer::execute_qwen3_layer0_gpu(
            plan, kTokens[position], position, gpu_cache).layer_output);
    }
    host_cache.reset();
    gpu_cache.reset();
    bool passed = host_cache.length() == 0 && gpu_cache.length() == 0;
    for (std::size_t position = 0; position < kTokens.size(); ++position) {
        (void) miinfer::execute_qwen3_layer0_host(model, kTokens[position], position, host_cache);
        const auto trace = miinfer::execute_qwen3_layer0_gpu(
            plan, kTokens[position], position, gpu_cache);
        passed = passed && max_abs(trace.layer_output, first_outputs[position]) <= 1.0e-6F;
    }
    return passed;
}

bool mutation_test(const std::string& model_path, const char* mutation) {
    setenv("MIINFER_GPU_KV_MUTATE", mutation, 1);
    const bool detected = !run_sequence(model_path, true);
    unsetenv("MIINFER_GPU_KV_MUTATE");
    return detected;
}

bool invalid_append_test() {
    miinfer::Qwen3Layer0GpuKvCache cache(2, 4, 4);
    try {
        cache.append(1, nullptr, nullptr);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

bool cache_corruption_test(const std::string& model_path) {
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    miinfer::Qwen3Layer0GpuKvCache gpu_cache(config.kv_heads, config.head_dim, kTokens.size());
    for (std::size_t position = 0; position < 2; ++position) {
        (void) miinfer::execute_qwen3_layer0_gpu(plan, kTokens[position], position, gpu_cache);
    }
    float corrupt = 1.0F;
    MIINFER_HIP_CHECK(hipMemcpy(gpu_cache.device_keys(), &corrupt, sizeof(corrupt), hipMemcpyHostToDevice));
    const auto corrupted = miinfer::execute_qwen3_layer0_gpu(plan, kTokens[2], 2, gpu_cache);

    miinfer::Qwen3Layer0KvCache host_cache(config.kv_heads, config.head_dim, kTokens.size());
    for (std::size_t position = 0; position <= 2; ++position) {
        const auto expected = miinfer::execute_qwen3_layer0_host(
            model, kTokens[position], position, host_cache);
        if (position == 2) {
            return max_abs(corrupted.attention_output, expected.attention_output) > 1.0e-3F;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "qwen3 KV-cache GPU test: SKIP (model path not supplied)\n";
        return 0;
    }
    if (argc != 2) return 1;
    try {
        const std::string model_path = argv[1];
        const bool sequence = run_sequence(model_path, false);
        const bool reset = reset_test(model_path);
        const bool wrong_rope = mutation_test(model_path, "wrong-rope-position");
        const bool ignore_cache = mutation_test(model_path, "ignore-earlier-cache");
        const bool invalid_append = invalid_append_test();
        const bool corruption = cache_corruption_test(model_path);
        std::cout << "GPU four-position layer-0 sequence: " << (sequence ? "PASS" : "FAIL") << '\n';
        std::cout << "GPU cache reset determinism: " << (reset ? "PASS" : "FAIL") << '\n';
        std::cout << "GPU wrong-RoPE mutation: " << (wrong_rope ? "PASS" : "FAIL") << '\n';
        std::cout << "GPU ignored-cache mutation: " << (ignore_cache ? "PASS" : "FAIL") << '\n';
        std::cout << "GPU invalid-append guard: " << (invalid_append ? "PASS" : "FAIL") << '\n';
        std::cout << "GPU cache-corruption mutation: " << (corruption ? "PASS" : "FAIL") << '\n';
        return sequence && reset && wrong_rope && ignore_cache && invalid_append && corruption ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "KV-cache GPU test error: " << error.what() << '\n';
        return 1;
    }
}
