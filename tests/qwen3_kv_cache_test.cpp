#include "miinfer/qwen3_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

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
    const std::vector<std::uint32_t> tokens{14990U, 42U, 31415U, 2718U};
    miinfer::Qwen3Layer0KvCache cache(config.kv_heads, config.head_dim, tokens.size());
    std::vector<std::vector<float>> outputs;
    std::vector<float> previous_keys(cache.keys().begin(), cache.keys().end());
    std::vector<float> previous_values(cache.values().begin(), cache.values().end());
    bool passed = true;
    for (std::size_t position = 0; position < tokens.size(); ++position) {
        const auto trace = miinfer::execute_qwen3_layer0_host(model, tokens[position], position, cache);
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
    for (std::size_t position = 0; position < tokens.size(); ++position) {
        const auto trace = miinfer::execute_qwen3_layer0_host(model, tokens[position], position, cache);
        passed = passed && equal_vectors(trace.layer_output, outputs[position]);
    }
    std::cout << "host four-position layer-0 sequence: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    bool passed = cache_contract_test();
    if (argc == 1) return passed ? 0 : 1;
    if (argc != 2) return 1;
    try {
        return layer_sequence_test(argv[1]) && passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "KV-cache host test error: " << error.what() << '\n';
        return 1;
    }
}
