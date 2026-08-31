#include "miinfer/qwen3_gpu_layer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
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

void write_f32(const std::filesystem::path& path, std::span<const float> values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot create diagnostic dump: " + path.string());
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size_bytes()));
    if (!file) throw std::runtime_error("cannot write diagnostic dump: " + path.string());
}

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open diagnostic dump: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid diagnostic dump: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file) throw std::runtime_error("short diagnostic dump: " + path.string());
    return values;
}

float max_abs(std::span<const float> lhs, std::span<const float> rhs) {
    if (lhs.size() != rhs.size()) return INFINITY;
    float result = 0.0F;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i])) return INFINITY;
        result = std::max(result, std::fabs(lhs[i] - rhs[i]));
    }
    return result;
}

std::vector<float> snapshot_keys(const miinfer::Qwen3GpuDecodeCache& cache) {
    std::vector<float> result;
    for (std::size_t layer = 0; layer < cache.layers(); ++layer) {
        const auto values = cache.layer(layer).snapshot_keys();
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}

std::vector<float> snapshot_values(const miinfer::Qwen3GpuDecodeCache& cache) {
    std::vector<float> result;
    for (std::size_t layer = 0; layer < cache.layers(); ++layer) {
        const auto values = cache.layer(layer).snapshot_values();
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}

void write_layer_outputs(const std::filesystem::path& directory,
                         std::size_t position,
                         const miinfer::Qwen3ForwardTrace& trace) {
    for (std::size_t layer = 0; layer < trace.layer_outputs.size(); ++layer) {
        write_f32(directory / ("pos" + std::to_string(position) + "-layer-"
                               + std::to_string(layer) + "-output.f32"),
                  trace.layer_outputs[layer]);
    }
}

void dump_position3(const std::string& model_path, const std::filesystem::path& directory) {
    constexpr std::array<std::uint32_t, 4> kPrefix{14990U, 8U, 341U, 286U};
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    miinfer::Qwen3GpuDecodeCache cache(
        config.layer_count, config.kv_heads, config.head_dim, kPrefix.size());
    std::filesystem::create_directories(directory);
    for (std::size_t position = 0; position < kPrefix.size(); ++position) {
        const auto trace = miinfer::execute_qwen3_decode_gpu(plan, kPrefix[position], position, cache);
        if (position < kPrefix.size() - 1) {
            const auto keys = snapshot_keys(cache);
            const auto values = snapshot_values(cache);
            write_f32(directory / ("cache-k-after-pos" + std::to_string(position) + ".f32"), keys);
            write_f32(directory / ("cache-v-after-pos" + std::to_string(position) + ".f32"), values);
            write_layer_outputs(directory, position, trace);
        }
        if (position == kPrefix.size() - 2) {
            const auto keys = snapshot_keys(cache);
            const auto values = snapshot_values(cache);
            write_f32(directory / "cache-k-before-pos3.f32", keys);
            write_f32(directory / "cache-v-before-pos3.f32", values);
        }
        if (position == kPrefix.size() - 1) {
            for (std::size_t layer = 0; layer < trace.layer_outputs.size(); ++layer) {
                write_f32(directory / ("layer-" + std::to_string(layer) + "-output.f32"),
                          trace.layer_outputs[layer]);
            }
            write_f32(directory / "final-norm.f32", trace.final_norm);
            write_f32(directory / "logits.f32", trace.logits);
        }
    }
}

bool compare_dumps(const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
    bool valid = true;
    const auto compare = [&](const std::string& name) {
        const auto left = read_f32(lhs / name);
        const auto right = read_f32(rhs / name);
        const float error = max_abs(left, right);
        std::cout << "M4-C2 Debug/Release " << name << " max_abs=" << error << '\n';
        valid = valid && std::isfinite(error);
    };
    const auto compare_cache = [&](const std::string& name) {
        const auto left = read_f32(lhs / name);
        const auto right = read_f32(rhs / name);
        const float error = max_abs(left, right);
        std::cout << "M4-C2 Debug/Release " << name << " max_abs=" << error << '\n';
        valid = valid && std::isfinite(error);
        if (left.size() != right.size() || left.size() % 36 != 0) return;
        const auto layer_elements = left.size() / 36;
        for (std::size_t layer = 0; layer < 36; ++layer) {
            const auto begin = layer * layer_elements;
            const auto layer_error = max_abs(
                std::span<const float>(left).subspan(begin, layer_elements),
                std::span<const float>(right).subspan(begin, layer_elements));
            std::cout << "M4-C2 Debug/Release " << name
                      << " layer=" << layer << " max_abs=" << layer_error << '\n';
        }
    };
    compare_cache("cache-k-before-pos3.f32");
    compare_cache("cache-v-before-pos3.f32");
    for (std::size_t position = 0; position < 3; ++position) {
        compare_cache("cache-k-after-pos" + std::to_string(position) + ".f32");
        compare_cache("cache-v-after-pos" + std::to_string(position) + ".f32");
        for (std::size_t layer = 0; layer < 36; ++layer) {
            compare("pos" + std::to_string(position) + "-layer-"
                    + std::to_string(layer) + "-output.f32");
        }
    }
    for (std::size_t layer = 0; layer < 36; ++layer) {
        compare("layer-" + std::to_string(layer) + "-output.f32");
    }
    compare("final-norm.f32");
    compare("logits.f32");
    return valid;
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

bool run_gpu_sequence_diagnostic(const std::string& model_path) {
    const auto model = miinfer::Qwen3Model::load(model_path);
    const auto& config = model.config();
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    miinfer::Qwen3GpuDecodeCache cache(
        config.layer_count, config.kv_heads, config.head_dim, kExpectedSequence.size());

    bool passed = true;
    for (std::size_t position = 0; position + 1 < kExpectedSequence.size(); ++position) {
        const auto trace = miinfer::execute_qwen3_decode_gpu(
            plan, kExpectedSequence[position], position, cache);
        const auto selected = argmax(trace.logits);
        const auto expected = kExpectedSequence[position + 1];
        const bool finite_state = finite(trace);
        const bool cache_valid = cache_lengths(cache, position + 1);
        const bool token_match = selected == expected;
        passed = passed && finite_state && cache_valid;
        std::cout << "M4-C2 Debug diagnostic position=" << position
                  << " input=" << kExpectedSequence[position]
                  << " expected-next=" << expected
                  << " selected=" << selected
                  << " token=" << (token_match ? "PASS" : "DIFF")
                  << " finite=" << (finite_state ? "PASS" : "FAIL")
                  << " cache=" << (cache_valid ? "PASS" : "FAIL")
                  << " status=" << ((finite_state && cache_valid) ? "PASS" : "FAIL") << '\n';
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
    if (argc == 3 && std::string(argv[2]) == "--diagnostic") {
        try {
            const bool sequence = run_gpu_sequence_diagnostic(argv[1]);
            const bool deterministic = deterministic_replay(argv[1]);
            std::cout << "M4-C2 Debug diagnostic sequence: "
                      << (sequence ? "PASS" : "FAIL") << '\n';
            std::cout << "M4-C2 Debug diagnostic replay: "
                      << (deterministic ? "PASS" : "FAIL") << '\n';
            return sequence && deterministic ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << "M4-C2 Debug diagnostic error: " << error.what() << '\n';
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
    if (argc == 4 && std::string(argv[1]) == "--compare-dumps") {
        try {
            return compare_dumps(argv[2], argv[3]) ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << "M4-C2 dump comparison error: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc == 4 && std::string(argv[2]) == "--dump-position-3") {
        try {
            dump_position3(argv[1], argv[3]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "M4-C2 dump error: " << error.what() << '\n';
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
