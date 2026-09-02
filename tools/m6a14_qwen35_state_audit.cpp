#define MIINFER_M6A5_HELPERS_ONLY
#include "m6a5_qwen35_hybrid_block.cpp"

#include "miinfer/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kA14RecurrentLayers = 3;
constexpr std::size_t kA14FullLayer = 3;
constexpr std::size_t kMaxPosition = 2;
constexpr std::size_t kStateAlignment = 256;
constexpr std::size_t kConvHistoryCapacity = 3;

using State = std::array<std::vector<float>, kA14RecurrentLayers>;
using History = std::array<std::vector<std::array<float, kChannels>>, kA14RecurrentLayers>;

struct RuntimeState {
    State recurrent{};
    History convolution{};
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
};

struct ManifestEntry {
    std::string name;
    std::string dimensions;
    std::size_t bytes = 0;
    std::size_t offset = 0;
};

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void print_manifest(const std::array<std::vector<float>, kRecurrentLayers>& initial_states) {
    std::vector<ManifestEntry> entries;
    std::size_t offset = 0;
    const auto add = [&](std::string name, std::string dimensions, std::size_t bytes) {
        offset = align_up(offset, kStateAlignment);
        entries.push_back({std::move(name), std::move(dimensions), bytes, offset});
        offset += bytes;
    };
    const std::size_t delta_bytes = initial_states.front().size() * sizeof(float);
    const std::size_t conv_bytes = kConvHistoryCapacity * kChannels * sizeof(float);
    const std::size_t cache_bytes = (kMaxPosition + 1) * kKvWidth * sizeof(float);
    for (std::size_t layer = 0; layer < kA14RecurrentLayers; ++layer) {
        add("L" + std::to_string(layer) + " delta state", "[48,128,128]", delta_bytes);
        add("L" + std::to_string(layer) + " convolution history", "[3,10240]", conv_bytes);
    }
    add("L3 K cache", "[17,4,256]", cache_bytes);
    add("L3 V cache", "[17,4,256]", cache_bytes);

    std::cout << "state_plan alignment=" << kStateAlignment
              << " capacity_positions=" << (kMaxPosition + 1) << '\n';
    for (const auto& entry : entries) {
        std::cout << "state_plan name=\"" << entry.name << "\" dimensions="
                  << entry.dimensions << " offset=" << entry.offset
                  << " bytes=" << entry.bytes << '\n';
    }
    std::cout << "state_plan total_bytes=" << align_up(offset, kStateAlignment) << '\n';
}

std::string dimensions(std::span<const std::size_t> values) {
    std::ostringstream output;
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) output << ',';
        output << values[i];
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
    const std::array<std::size_t, 3> dims_value{cache.size(), kKvHeads, kHeadDim};
    return fingerprint(type, kA14FullLayer, position, dimensions(dims_value), flat);
}

void poison(std::vector<float>& values, unsigned char pattern) {
    if (!values.empty()) std::memset(values.data(), pattern, values.size() * sizeof(float));
}

void poison_recurrent(RuntimeState& runtime, unsigned char pattern) {
    for (auto& state : runtime.recurrent) poison(state, pattern);
    for (auto& layer : runtime.convolution) {
        if (layer.empty()) layer.resize(kConvHistoryCapacity);
        for (auto& entry : layer) std::memset(entry.data(), pattern, sizeof(entry));
    }
}

void poison_kv(RuntimeState& runtime, unsigned char pattern) {
    if (runtime.keys.empty()) runtime.keys.resize(kMaxPosition + 1, std::vector<float>(kKvWidth));
    if (runtime.values.empty()) runtime.values.resize(kMaxPosition + 1, std::vector<float>(kKvWidth));
    for (auto& entry : runtime.keys) poison(entry, pattern);
    for (auto& entry : runtime.values) poison(entry, pattern);
}

void poison(RuntimeState& runtime, unsigned char pattern) {
    poison_recurrent(runtime, pattern);
    poison_kv(runtime, pattern);
}

void reset(RuntimeState& runtime, const State& initial) {
    runtime.recurrent = initial;
    for (auto& history : runtime.convolution) history.clear();
    runtime.keys.clear();
    runtime.values.clear();
}

void require_same(std::string_view label, const std::vector<std::string>& expected,
                  const std::vector<std::string>& actual) {
    if (expected != actual) {
        const auto count = std::min(expected.size(), actual.size());
        for (std::size_t i = 0; i < count; ++i) {
            if (expected[i] != actual[i]) {
                throw std::runtime_error(std::string(label) + " diverges at record "
                                         + std::to_string(i));
            }
        }
        throw std::runtime_error(std::string(label) + " record count differs");
    }
}

std::vector<std::string> run_sequence(const GgufFile& model,
                                      const std::filesystem::path& fixture,
                                      RuntimeState& runtime,
                                      const std::vector<std::size_t>& positions,
                                      bool validate, bool print_records) {
    std::vector<std::string> records;
    constexpr std::array<std::size_t, 3> checked_positions{0, 1, 2};
    const auto prompt_tokens = read_tokens(fixture / "prompt_tokens.txt");
    const auto generated_tokens = read_tokens(fixture / "generated_tokens.txt");
    if (prompt_tokens.empty() || generated_tokens.size() <= kMaxPosition) {
        throw std::runtime_error("fixture lacks the token sequence required by A14");
    }
    for (const auto position : positions) {
        const bool check = validate && std::find(checked_positions.begin(), checked_positions.end(), position)
            != checked_positions.end();
        const auto token = position == 0 ? prompt_tokens.front() : generated_tokens[position - 1];
        auto current = embedding(tensor(model, "token_embd.weight"), token);
        if (check) {
            require_match("embedding", compare(
                current, read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden)), 1.0e-6F);
        }
        for (std::size_t layer = 0; layer < kA14RecurrentLayers; ++layer) {
            const std::array<std::size_t, 3> state_dims{kVHeads, kState, kState};
            records.push_back(fingerprint("recurrent_before", layer, position,
                                          dimensions(state_dims), runtime.recurrent[layer]));
            if (check) {
                const auto expected = read_f32(
                    checkpoint(fixture, position, "state_predelta-" + std::to_string(layer)),
                    kVHeads * kState * kState);
                require_match("state input", compare(runtime.recurrent[layer], expected), 2.0e1F);
            }
            auto state_input = runtime.recurrent[layer];
            current = recurrent_block(model, fixture, layer, position, current, state_input,
                                      runtime.convolution[layer], runtime.recurrent[layer], check);
            records.push_back(fingerprint("recurrent_after", layer, position,
                                          dimensions(state_dims), runtime.recurrent[layer]));
            const std::array<std::size_t, 1> hidden_dims{kHidden};
            records.push_back(fingerprint("hidden_after", layer, position,
                                          dimensions(hidden_dims), current));
        }

        records.push_back(cache_fingerprint("kv_k_before", position, runtime.keys));
        records.push_back(cache_fingerprint("kv_v_before", position, runtime.values));
        current = full_block(model, fixture, kA14FullLayer, position, current,
                             runtime.keys, runtime.values, check);
        records.push_back(cache_fingerprint("kv_k_after", position, runtime.keys));
        records.push_back(cache_fingerprint("kv_v_after", position, runtime.values));
        const std::array<std::size_t, 1> hidden_dims{kHidden};
        records.push_back(fingerprint("hidden_after", kA14FullLayer, position,
                                      dimensions(hidden_dims), current));
    }
    if (print_records) {
        for (const auto& record : records) std::cout << record << '\n';
    }
    return records;
}

State initial_states(const std::filesystem::path& fixture) {
    State initial;
    for (std::size_t layer = 0; layer < kA14RecurrentLayers; ++layer) {
        initial[layer] = read_f32(
            checkpoint(fixture, 0, "state_predelta-" + std::to_string(layer)),
            kVHeads * kState * kState);
    }
    return initial;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a14-qwen35-state-audit MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = GgufFile::open(argv[1]);
        const std::filesystem::path fixture = argv[2];
        const auto initial = initial_states(fixture);
        print_manifest(initial);
        std::vector<std::size_t> positions(kMaxPosition + 1);
        std::iota(positions.begin(), positions.end(), 0);
        const std::vector<std::size_t> probe_positions{0};

        RuntimeState runtime;
        reset(runtime, initial);
        const auto reference = run_sequence(*model, fixture, runtime, positions, true, true);

        reset(runtime, initial);
        const auto replay = run_sequence(*model, fixture, runtime, positions, true, false);
        require_same("clean replay", reference, replay);
        std::cout << "replay clean=PASS records=" << replay.size() << '\n';

        reset(runtime, initial);
        const auto reference_probe = run_sequence(*model, fixture, runtime, probe_positions, true, false);

        poison(runtime, 0xFF);
        reset(runtime, initial);
        const auto poisoned = run_sequence(*model, fixture, runtime, probe_positions, true, false);
        require_same("full poison reset", reference_probe, poisoned);
        std::cout << "replay full_poison_ff=PASS\n";

        reset(runtime, initial);
        poison(runtime, 0xA5);
        reset(runtime, initial);
        const auto poisoned_a5 = run_sequence(*model, fixture, runtime, probe_positions, true, false);
        require_same("a5 poison reset", reference_probe, poisoned_a5);
        std::cout << "replay full_poison_a5=PASS\n";

        reset(runtime, initial);
        poison_recurrent(runtime, 0xFF);
        reset(runtime, initial);
        const auto partial_recurrent = run_sequence(*model, fixture, runtime, probe_positions, true, false);
        require_same("partial recurrent poison reset", reference_probe, partial_recurrent);
        std::cout << "replay partial_recurrent_poison=PASS\n";

        reset(runtime, initial);
        poison_kv(runtime, 0xA5);
        reset(runtime, initial);
        const auto partial_kv = run_sequence(*model, fixture, runtime, probe_positions, true, false);
        require_same("partial KV poison reset", reference_probe, partial_kv);
        std::cout << "replay partial_kv_poison=PASS\n";

        reset(runtime, initial);
        run_sequence(*model, fixture, runtime, probe_positions, false, false);
        reset(runtime, initial);
        const auto after_other_sequence = run_sequence(*model, fixture, runtime,
                                                        probe_positions, true, false);
        require_same("cross-sequence reset", reference_probe, after_other_sequence);
        std::cout << "replay cross_sequence_reset=PASS\n";

        std::cout << "M6-A14 qwen35 state fingerprint/reset audit PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A14 failed: " << error.what() << '\n';
        return 1;
    }
}
