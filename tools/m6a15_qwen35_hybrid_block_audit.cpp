#define MIINFER_M6A5_HELPERS_ONLY
#include "m6a5_qwen35_hybrid_block.cpp"

#include "miinfer/sha256.hpp"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <initializer_list>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kA15RecurrentLayers = 3;
constexpr std::size_t kA15FullLayer = 3;
constexpr std::size_t kA15MaxPosition = 16;

using State = std::array<std::vector<float>, kA15RecurrentLayers>;
using History = std::array<std::vector<std::array<float, kChannels>>, kA15RecurrentLayers>;

struct RuntimeState {
    State recurrent{};
    History convolution{};
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
};

std::string dimensions(std::initializer_list<std::size_t> values) {
    std::ostringstream output;
    output << '[';
    bool first = true;
    for (const auto value : values) {
        if (!first) output << ',';
        first = false;
        output << value;
    }
    output << ']';
    return output.str();
}

std::string hash_floats(std::span<const float> values) {
    return miinfer::sha256_bytes(std::as_bytes(values));
}

std::string fingerprint(std::string_view type, std::size_t layer, std::size_t position,
                        std::string_view dims, std::span<const float> values) {
    std::ostringstream output;
    output << "state type=" << type << " layer=" << layer << " position=" << position
           << " dimensions=" << dims << " bytes=" << values.size_bytes()
           << " sha256=" << hash_floats(values);
    return output.str();
}

std::vector<float> flatten(const std::vector<std::vector<float>>& values) {
    std::vector<float> result;
    for (const auto& row : values) result.insert(result.end(), row.begin(), row.end());
    return result;
}

std::string cache_fingerprint(std::string_view type, std::size_t position,
                              const std::vector<std::vector<float>>& cache) {
    const auto flat = flatten(cache);
    return fingerprint(type, kA15FullLayer, position,
                       dimensions({cache.size(), kKvHeads, kHeadDim}), flat);
}

State initial_states(const std::filesystem::path& fixture) {
    State result;
    for (std::size_t layer = 0; layer < kA15RecurrentLayers; ++layer) {
        result[layer] = read_f32(
            checkpoint(fixture, 0, "state_predelta-" + std::to_string(layer)),
            kVHeads * kState * kState);
    }
    return result;
}

void reset(RuntimeState& runtime, const State& initial) {
    runtime.recurrent = initial;
    for (auto& history : runtime.convolution) history.clear();
    runtime.keys.clear();
    runtime.values.clear();
}

bool is_checkpoint_position(std::size_t position) noexcept {
    constexpr std::array<std::size_t, 6> positions{0, 1, 2, 4, 8, 16};
    return std::find(positions.begin(), positions.end(), position) != positions.end();
}

void report_timing(std::size_t layer, bool full, double milliseconds) {
    std::cout << "timing layer=" << layer << " type=" << (full ? "full" : "recurrent")
              << " ms=" << milliseconds << '\n';
}

void run_block(const GgufFile& model, const std::filesystem::path& fixture,
               RuntimeState& runtime, const std::vector<std::uint32_t>& generated,
               std::uint32_t prompt_token) {
    if (generated.size() <= kA15MaxPosition) {
        throw std::runtime_error("fixture lacks tokens through position 16");
    }
    for (std::size_t position = 0; position <= kA15MaxPosition; ++position) {
        const bool check = is_checkpoint_position(position);
        const auto token = position == 0 ? prompt_token : generated[position - 1];
        auto current = embedding(tensor(model, "token_embd.weight"), token);
        if (check) {
            require_match("block embedding", compare(
                current, read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden)),
                1.0e-6F);
        }
        const auto block_start = std::chrono::steady_clock::now();
        for (std::size_t layer = 0; layer < kA15RecurrentLayers; ++layer) {
            std::cout << fingerprint("recurrent_before", layer, position,
                                      dimensions({kVHeads, kState, kState}), runtime.recurrent[layer])
                      << '\n';
            if (check) {
                require_match("recurrent state input", compare(
                    runtime.recurrent[layer], read_f32(
                        checkpoint(fixture, position,
                                   "state_predelta-" + std::to_string(layer)),
                        kVHeads * kState * kState)), 2.0e1F);
            }
            const auto start = std::chrono::steady_clock::now();
            auto state_input = runtime.recurrent[layer];
            current = recurrent_block(model, fixture, layer, position, current, state_input,
                                      runtime.convolution[layer], runtime.recurrent[layer], check);
            const auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            report_timing(layer, false, elapsed);
            std::cout << fingerprint("recurrent_after", layer, position,
                                      dimensions({kVHeads, kState, kState}), runtime.recurrent[layer])
                      << '\n';
            std::cout << fingerprint("hidden_after", layer, position,
                                      dimensions({kHidden}), current) << '\n';
        }

        std::cout << cache_fingerprint("kv_k_before", position, runtime.keys) << '\n';
        std::cout << cache_fingerprint("kv_v_before", position, runtime.values) << '\n';
        const auto start = std::chrono::steady_clock::now();
        current = full_block(model, fixture, kA15FullLayer, position, current,
                             runtime.keys, runtime.values, check);
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        report_timing(kA15FullLayer, true, elapsed);
        std::cout << cache_fingerprint("kv_k_after", position, runtime.keys) << '\n';
        std::cout << cache_fingerprint("kv_v_after", position, runtime.values) << '\n';
        std::cout << fingerprint("hidden_after", kA15FullLayer, position,
                                  dimensions({kHidden}), current) << '\n';
        const auto block_elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - block_start).count();
        std::cout << "timing hybrid_block position=" << position
                  << " ms=" << block_elapsed << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a15-qwen35-hybrid-block-audit MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = GgufFile::open(argv[1]);
        const std::filesystem::path fixture = argv[2];
        const auto prompt_tokens = read_tokens(fixture / "prompt_tokens.txt");
        const auto generated_tokens = read_tokens(fixture / "generated_tokens.txt");
        if (prompt_tokens.empty()) throw std::runtime_error("fixture has no prompt token");

        RuntimeState runtime;
        reset(runtime, initial_states(fixture));
        run_block(*model, fixture, runtime, generated_tokens, prompt_tokens.front());
        std::cout << "M6-A15 qwen35 layers-0-3 hybrid block audit PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A15 failed: " << error.what() << '\n';
        return 1;
    }
}
