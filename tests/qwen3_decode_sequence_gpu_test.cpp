#include "miinfer/qwen3_gpu_layer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Captured from the pinned independent gfx906 reference with:
// llama-simple -m Qwen3-8B-q4_0-b968826d.gguf -n 8 -ngl 99 hello
constexpr std::array<std::uint32_t, 9> kExpectedSequence{
    14990U, 8U, 341U, 286U, 470U, 330U, 9707U, 11U, 330U};

bool finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool finite(const miinfer::Qwen3ForwardTrace& trace) {
    if (!finite(trace.embedding) || !finite(trace.final_norm) || !finite(trace.logits)) {
        return false;
    }
    return std::all_of(trace.layer_outputs.begin(), trace.layer_outputs.end(),
                       [](const auto& output) { return finite(output); });
}

std::uint32_t argmax(const std::vector<float>& values) {
    if (values.empty()) throw std::runtime_error("cannot select from empty logits");
    return static_cast<std::uint32_t>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
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

bool run_sequence(const std::string& model_path, bool report) {
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    miinfer::Qwen3DecodeCache host_cache(
        config.layer_count, config.kv_heads, config.head_dim, kExpectedSequence.size());
    miinfer::Qwen3GpuDecodeCache gpu_cache(
        config.layer_count, config.kv_heads, config.head_dim, kExpectedSequence.size());

    bool passed = true;
    std::uint32_t input_token = kExpectedSequence.front();
    for (std::size_t position = 0; position + 1 < kExpectedSequence.size(); ++position) {
        const auto host = miinfer::execute_qwen3_decode_host(
            model, input_token, position, host_cache);
        const auto gpu = miinfer::execute_qwen3_decode_gpu(
            plan, input_token, position, gpu_cache);
        const auto host_next = argmax(host.logits);
        const auto gpu_next = argmax(gpu.logits);
        const auto expected = kExpectedSequence[position + 1];
        const bool step_pass = host_next == expected && gpu_next == expected
            && host_next == gpu_next && finite(host) && finite(gpu)
            && cache_lengths(host_cache, position + 1)
            && cache_lengths(gpu_cache, position + 1);
        passed = passed && step_pass;
        if (report || !step_pass) {
            std::cout << "M4-C2 position=" << position
                      << " input=" << input_token
                      << " expected-next=" << expected
                      << " host-next=" << host_next
                      << " gpu-next=" << gpu_next
                      << " gpu-logit470=" << gpu.logits[470]
                      << " gpu-logit419=" << gpu.logits[419]
                      << " cache=" << (cache_lengths(gpu_cache, position + 1) ? "PASS" : "FAIL")
                      << " status=" << (step_pass ? "PASS" : "FAIL") << '\n';
        }
        if (!step_pass) break;
        input_token = gpu_next;
    }
    return passed;
}

bool run_gpu_sequence_only(const std::string& model_path) {
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    miinfer::Qwen3GpuDecodeCache cache(
        config.layer_count, config.kv_heads, config.head_dim, kExpectedSequence.size());

    bool passed = true;
    std::uint32_t input_token = kExpectedSequence.front();
    for (std::size_t position = 0; position + 1 < kExpectedSequence.size(); ++position) {
        const auto trace = miinfer::execute_qwen3_decode_gpu(
            plan, input_token, position, cache);
        const auto selected = argmax(trace.logits);
        const auto expected = kExpectedSequence[position + 1];
        const bool step_pass = selected == expected && finite(trace)
            && cache_lengths(cache, position + 1);
        std::cout << "M4-C2 GPU position=" << position
                  << " input=" << input_token
                  << " expected-next=" << expected
                  << " selected=" << selected
                  << " logit470=" << trace.logits[470]
                  << " logit419=" << trace.logits[419]
                  << " cache=" << (cache_lengths(cache, position + 1) ? "PASS" : "FAIL")
                  << " status=" << (step_pass ? "PASS" : "FAIL") << '\n';
        passed = passed && step_pass;
        if (!step_pass) break;
        input_token = selected;
    }
    return passed;
}

bool deterministic_replay(const std::string& model_path) {
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    miinfer::Qwen3GpuDecodeCache first(
        config.layer_count, config.kv_heads, config.head_dim, kExpectedSequence.size());
    miinfer::Qwen3GpuDecodeCache second(
        config.layer_count, config.kv_heads, config.head_dim, kExpectedSequence.size());

    std::vector<std::vector<float>> first_logits;
    first_logits.reserve(kExpectedSequence.size() - 1);
    for (std::size_t position = 0; position + 1 < kExpectedSequence.size(); ++position) {
        const auto trace = miinfer::execute_qwen3_decode_gpu(
            plan, kExpectedSequence[position], position, first);
        first_logits.push_back(trace.logits);
    }
    bool passed = true;
    for (std::size_t position = 0; position + 1 < kExpectedSequence.size(); ++position) {
        const auto trace = miinfer::execute_qwen3_decode_gpu(
            plan, kExpectedSequence[position], position, second);
        passed = passed && trace.logits == first_logits[position]
            && cache_lengths(second, position + 1);
    }
    return passed;
}

bool position3_diagnostic(const std::string& model_path) {
    constexpr std::array<std::uint32_t, 4> kPrefix{14990U, 8U, 341U, 286U};
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    miinfer::Qwen3GpuDecodeCache cache(
        config.layer_count, config.kv_heads, config.head_dim, kPrefix.size());
    miinfer::Qwen3ForwardTrace trace;
    for (std::size_t position = 0; position < kPrefix.size(); ++position) {
        trace = miinfer::execute_qwen3_decode_gpu(plan, kPrefix[position], position, cache);
    }
    const auto best = argmax(trace.logits);
    std::cout << "M4-C2 position-3 GPU diagnostic: argmax=" << best
              << " logit[470]=" << trace.logits[470]
              << " logit[419]=" << trace.logits[419]
              << " margin419-470=" << (trace.logits[419] - trace.logits[470])
              << " cache=" << (cache_lengths(cache, kPrefix.size()) ? "PASS" : "FAIL") << '\n';
    return finite(trace) && cache_lengths(cache, kPrefix.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "qwen3 decode sequence GPU test: SKIP (model path not supplied)\n";
        return 0;
    }
    if (argc == 3 && std::string(argv[2]) == "--gpu-only") {
        try {
            const bool sequence = run_gpu_sequence_only(argv[1]);
            const bool deterministic = deterministic_replay(argv[1]);
            std::cout << "M4-C2 GPU-only greedy sequence: "
                      << (sequence ? "PASS" : "FAIL") << '\n';
            std::cout << "M4-C2 GPU replay determinism: "
                      << (deterministic ? "PASS" : "FAIL") << '\n';
            return sequence && deterministic ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << "M4-C2 GPU-only sequence error: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc == 3 && std::string(argv[2]) == "--position-3-diagnostic") {
        try {
            return position3_diagnostic(argv[1]) ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << "M4-C2 position-3 diagnostic error: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 2) return 1;
    try {
        const std::string model_path = argv[1];
        const bool sequence = run_sequence(model_path, true);
        const bool deterministic = deterministic_replay(model_path);
        std::cout << "M4-C2 eight-token greedy sequence: "
                  << (sequence ? "PASS" : "FAIL") << '\n';
        std::cout << "M4-C2 GPU replay determinism: "
                  << (deterministic ? "PASS" : "FAIL") << '\n';
        return sequence && deterministic ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "M4-C2 decode sequence test error: " << error.what() << '\n';
        return 1;
    }
}
