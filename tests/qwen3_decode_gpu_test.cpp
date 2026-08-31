#include "miinfer/qwen3_gpu_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kPromptToken = 14990U;
constexpr std::uint32_t kExpectedFirstToken = 8U;
constexpr std::size_t kCapacity = 2;

std::size_t argmax(const std::vector<float>& values) {
    if (values.empty()) throw std::runtime_error("cannot select from empty logits");
    return static_cast<std::size_t>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
}

bool finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool exact(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    return lhs == rhs;
}

bool exact(const miinfer::Qwen3ForwardTrace& lhs,
           const miinfer::Qwen3ForwardTrace& rhs) {
    if (!exact(lhs.embedding, rhs.embedding) || !exact(lhs.final_norm, rhs.final_norm)
        || !exact(lhs.logits, rhs.logits) || lhs.layer_outputs.size() != rhs.layer_outputs.size()) {
        return false;
    }
    for (std::size_t layer = 0; layer < lhs.layer_outputs.size(); ++layer) {
        if (!exact(lhs.layer_outputs[layer], rhs.layer_outputs[layer])) return false;
    }
    return true;
}

bool cache_lengths(const miinfer::Qwen3DecodeCache& cache, std::size_t expected) {
    if (cache.length() != expected) return false;
    for (std::size_t layer = 0; layer < cache.layers(); ++layer) {
        if (cache.layer(layer).length() != expected) return false;
    }
    return true;
}

bool cache_lengths(const miinfer::Qwen3GpuDecodeCache& cache, std::size_t expected) {
    if (cache.length() != expected) return false;
    for (std::size_t layer = 0; layer < cache.layers(); ++layer) {
        if (cache.layer(layer).length() != expected) return false;
    }
    return true;
}

bool finite(const miinfer::Qwen3ForwardTrace& trace) {
    if (!finite(trace.embedding) || !finite(trace.final_norm) || !finite(trace.logits)) return false;
    return std::all_of(trace.layer_outputs.begin(), trace.layer_outputs.end(),
                       [](const auto& output) { return finite(output); });
}

bool run(const std::string& model_path) {
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    miinfer::Qwen3DecodeCache host_cache(
        config.layer_count, config.kv_heads, config.head_dim, kCapacity);
    miinfer::Qwen3GpuDecodeCache gpu_cache(
        config.layer_count, config.kv_heads, config.head_dim, kCapacity);

    const auto host_zero = miinfer::execute_qwen3_decode_host(
        model, kPromptToken, 0, host_cache);
    const auto gpu_zero = miinfer::execute_qwen3_decode_gpu(
        plan, kPromptToken, 0, gpu_cache);
    const auto single = miinfer::execute_qwen3_forward_gpu(plan, kPromptToken);
    const auto generated = static_cast<std::uint32_t>(argmax(gpu_zero.logits));

    bool passed = generated == kExpectedFirstToken
        && argmax(host_zero.logits) == kExpectedFirstToken
        && exact(gpu_zero.logits, single.logits)
        && finite(host_zero) && finite(gpu_zero)
        && cache_lengths(host_cache, 1) && cache_lengths(gpu_cache, 1);
    std::cout << "M4-C1 prompt token=" << kPromptToken
              << " first generated token=" << generated
              << " expected=" << kExpectedFirstToken
              << " initial-cache=" << (cache_lengths(gpu_cache, 1) ? "PASS" : "FAIL") << '\n';

    const auto host_one = miinfer::execute_qwen3_decode_host(
        model, generated, 1, host_cache);
    const auto gpu_one = miinfer::execute_qwen3_decode_gpu(
        plan, generated, 1, gpu_cache);
    passed = passed && finite(host_one) && finite(gpu_one)
        && cache_lengths(host_cache, 2) && cache_lengths(gpu_cache, 2);
    std::cout << "M4-C1 persistent KV position=1: "
              << (cache_lengths(gpu_cache, 2) ? "PASS" : "FAIL")
              << " next-host-argmax=" << argmax(host_one.logits)
              << " next-gpu-argmax=" << argmax(gpu_one.logits) << '\n';

    host_cache.reset();
    gpu_cache.reset();
    const auto host_zero_repeat = miinfer::execute_qwen3_decode_host(
        model, kPromptToken, 0, host_cache);
    const auto host_one_repeat = miinfer::execute_qwen3_decode_host(
        model, generated, 1, host_cache);
    const auto gpu_zero_repeat = miinfer::execute_qwen3_decode_gpu(
        plan, kPromptToken, 0, gpu_cache);
    const auto gpu_one_repeat = miinfer::execute_qwen3_decode_gpu(
        plan, generated, 1, gpu_cache);
    const bool reset = exact(host_zero, host_zero_repeat) && exact(host_one, host_one_repeat)
        && exact(gpu_zero, gpu_zero_repeat) && exact(gpu_one, gpu_one_repeat);
    passed = passed && reset;
    std::cout << "M4-C1 reset determinism: " << (reset ? "PASS" : "FAIL") << '\n';

    bool invalid_position = false;
    miinfer::Qwen3DecodeCache invalid_cache(
        config.layer_count, config.kv_heads, config.head_dim, kCapacity);
    try {
        (void) miinfer::execute_qwen3_decode_host(model, kPromptToken, 1, invalid_cache);
    } catch (const std::invalid_argument&) {
        invalid_position = true;
    }
    passed = passed && invalid_position;
    std::cout << "M4-C1 invalid-position guard: "
              << (invalid_position ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "qwen3 decode GPU test: SKIP (model path not supplied)\n";
        return 0;
    }
    if (argc != 2) return 1;
    try {
        const bool passed = run(argv[1]);
        std::cout << "M4-C1 deterministic first-token decode: "
                  << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "M4-C1 decode test error: " << error.what() << '\n';
        return 1;
    }
}
