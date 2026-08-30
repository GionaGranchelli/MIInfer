#include "miinfer/qwen3_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
        &trace.q_rope, &trace.v_projection, &trace.v_reshape, &trace.k_projection,
        &trace.k_reshape, &trace.k_rms, &trace.k_normed, &trace.k_rope,
        &trace.k_view, &trace.v_view, &trace.q_view, &trace.q_permuted,
        &trace.attention_output, &trace.ffn_input, &trace.ffn_rms, &trace.ffn_norm,
        &trace.gate, &trace.up, &trace.swiglu, &trace.ffn_output, &trace.layer_output,
    };
}

float checkpoint_tolerance(std::size_t checkpoint) {
    if (checkpoint <= 2) return 2.0e-4F;
    if ((checkpoint >= 3 && checkpoint <= 4) || (checkpoint >= 8 && checkpoint <= 11)) {
        return 2.0e-4F;
    }
    if (checkpoint == 5 || checkpoint == 12) return 5.0e-3F;
    if ((checkpoint >= 6 && checkpoint <= 7) || (checkpoint >= 13 && checkpoint <= 18)) {
        return 2.5e-2F;
    }
    return 2.0e-2F;
}

float max_abs(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) return std::numeric_limits<float>::infinity();
    float result = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return std::numeric_limits<float>::infinity();
        }
        result = std::max(result, std::fabs(actual[i] - expected[i]));
    }
    return result;
}

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open external trace: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid external trace tensor: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file) throw std::runtime_error("short external trace tensor: " + path.string());
    return values;
}

std::filesystem::path external_checkpoint(const std::filesystem::path& directory,
                                          std::size_t position, std::size_t checkpoint) {
    return directory / ("pos-" + std::to_string(position) + "-" + std::to_string(checkpoint) + ".f32");
}

bool equal_vectors(std::span<const float> left, std::span<const float> right, float tolerance = 0.0F) {
    if (left.size() != right.size()) return false;
    return std::equal(left.begin(), left.end(), right.begin(), [tolerance](float a, float b) {
        return std::fabs(a - b) <= tolerance;
    });
}

bool cache_contract_test() {
    miinfer::Qwen3Layer0KvCache cache(2, 4, 4);
    const std::vector<float> keys{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> values{9, 10, 11, 12, 13, 14, 15, 16};
    cache.append(0, keys, values);
    const auto first_keys = std::vector<float>(cache.keys().begin(), cache.keys().end());
    bool passed = cache.length() == 1 && first_keys[0] == 1.0F && first_keys[3] == 4.0F;
    try {
        cache.append(0, keys, values);
        passed = false;
    } catch (const std::invalid_argument&) {
    }
    cache.append(1, keys, values);
    passed = passed && cache.length() == 2;
    passed = passed && equal_vectors(
        cache.keys().subspan(0, 4), std::span<const float>(keys).subspan(0, 4));
    const auto second_slot = 1 * 4;
    passed = passed && equal_vectors(
        cache.keys().subspan(second_slot, 4), std::span<const float>(keys).subspan(0, 4));
    cache.reset();
    passed = passed && cache.length() == 0
             && std::all_of(cache.keys().begin(), cache.keys().end(), [](float value) { return value == 0.0F; });
    std::cout << "kv cache host contract: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool layer_sequence_test(const char* model_path) {
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    miinfer::Qwen3Layer0KvCache cache(config.kv_heads, config.head_dim, kTokens.size());
    std::vector<std::vector<float>> outputs;
    std::vector<float> previous_keys(cache.keys().begin(), cache.keys().end());
    std::vector<float> previous_values(cache.values().begin(), cache.values().end());
    bool passed = true;
    for (std::size_t position = 0; position < kTokens.size(); ++position) {
        const auto trace = miinfer::execute_qwen3_layer0_host(model, kTokens[position], position, cache);
        passed = passed && cache.length() == position + 1;
        const auto current_keys = cache.keys();
        const auto current_values = cache.values();
        for (std::size_t head = 0; head < config.kv_heads; ++head) {
            const auto base = (head * cache.capacity() + position) * config.head_dim;
            passed = passed && equal_vectors(
                current_keys.subspan(base, config.head_dim),
                std::span<const float>(trace.k_rope).subspan(head * config.head_dim, config.head_dim));
            passed = passed && equal_vectors(
                current_values.subspan(base, config.head_dim),
                std::span<const float>(trace.v_view).subspan(head * config.head_dim, config.head_dim));
        }
        for (std::size_t i = 0; i < current_keys.size(); ++i) {
            const auto current_slot = (i / config.head_dim) % cache.capacity();
            if (current_slot != position) {
                passed = passed && current_keys[i] == previous_keys[i];
                passed = passed && current_values[i] == previous_values[i];
            }
        }
        passed = passed && trace.attention_scores.size() == config.attention_heads * (position + 1);
        passed = passed && trace.attention_probabilities.size() == trace.attention_scores.size();
        for (std::size_t head = 0; head < config.attention_heads; ++head) {
            const auto base = head * (position + 1);
            float sum = 0.0F;
            for (std::size_t i = 0; i <= position; ++i) sum += trace.attention_probabilities[base + i];
            passed = passed && std::fabs(sum - 1.0F) < 1.0e-5F;
        }
        outputs.push_back(trace.layer_output);
        previous_keys.assign(current_keys.begin(), current_keys.end());
        previous_values.assign(current_values.begin(), current_values.end());
    }

    cache.reset();
    for (std::size_t position = 0; position < kTokens.size(); ++position) {
        const auto trace = miinfer::execute_qwen3_layer0_host(model, kTokens[position], position, cache);
        passed = passed && equal_vectors(trace.layer_output, outputs[position]);
    }
    std::cout << "host four-position layer-0 sequence: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool external_trace_test(const char* model_path, const std::filesystem::path& directory) {
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    miinfer::Qwen3Layer0KvCache cache(config.kv_heads, config.head_dim, kTokens.size());
    bool passed = true;
    for (std::size_t position = 0; position < kTokens.size(); ++position) {
        const auto trace = miinfer::execute_qwen3_layer0_host(model, kTokens[position], position, cache);
        const auto actual_points = checkpoints(trace);
        for (std::size_t checkpoint = 0; checkpoint < kCheckpointCount; ++checkpoint) {
            const auto expected = read_f32(external_checkpoint(directory, position, checkpoint));
            const float error = max_abs(*actual_points[checkpoint], expected);
            const bool current = error <= checkpoint_tolerance(checkpoint);
            if (!current) {
                std::cout << "external position " << position << " checkpoint " << checkpoint
                          << " FAIL max_abs=" << std::scientific << error << '\n';
            }
            passed = passed && current;
        }

        const auto expected_keys = read_f32(directory / ("pos-" + std::to_string(position) + "-cache-k.f32"));
        const auto expected_values = read_f32(directory / ("pos-" + std::to_string(position) + "-cache-v.f32"));
        std::vector<float> current_keys(config.kv_heads * config.head_dim);
        std::vector<float> current_values(config.kv_heads * config.head_dim);
        for (std::size_t head = 0; head < config.kv_heads; ++head) {
            const auto cache_offset = (head * cache.capacity() + position) * config.head_dim;
            std::copy_n(cache.keys().begin() + cache_offset, config.head_dim,
                        current_keys.begin() + head * config.head_dim);
            std::copy_n(cache.values().begin() + cache_offset, config.head_dim,
                        current_values.begin() + head * config.head_dim);
        }
        const float key_error = max_abs(current_keys, expected_keys);
        const float value_error = max_abs(current_values, expected_values);
        const bool key_pass = key_error <= 2.5e-2F;
        const bool value_pass = value_error <= 2.0e-2F;
        std::cout << "external position " << position << " post-RoPE K cache write: "
                  << (key_pass ? "PASS" : "FAIL") << " max_abs=" << key_error << '\n';
        std::cout << "external position " << position << " V cache write: "
                  << (value_pass ? "PASS" : "FAIL") << " max_abs=" << value_error << '\n';
        passed = passed && key_pass && value_pass;
    }

    const auto actual = miinfer::execute_qwen3_layer0_host(model, kTokens[0]);
    auto mutated = read_f32(external_checkpoint(directory, 0, 0));
    mutated[0] += 1.0F;
    const bool mutation_detected = max_abs(actual.embedding, mutated) > checkpoint_tolerance(0);
    std::cout << "external trace comparator mutation: "
              << (mutation_detected ? "PASS" : "FAIL") << '\n';
    passed = passed && mutation_detected;
    std::cout << "host four-position external trace: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    bool passed = cache_contract_test();
    if (argc == 1) return passed ? 0 : 1;
    if (argc < 2 || argc > 3) return 1;
    try {
        const std::filesystem::path external = argc == 3 ? argv[2] : "";
        return layer_sequence_test(argv[1])
                   && (argc == 2 || external_trace_test(argv[1], external))
                   && passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "KV-cache host test error: " << error.what() << '\n';
        return 1;
    }
}
