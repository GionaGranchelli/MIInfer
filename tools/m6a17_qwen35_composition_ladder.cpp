#define MIINFER_M6A7_STATEFUL_GENERATION
#define MIINFER_M6A15_HELPERS_ONLY
#include "m6a15_qwen35_hybrid_block_audit.cpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

constexpr std::size_t kLayers = 64;
constexpr std::size_t kVocab = 248320;
constexpr std::array<std::size_t, 4> kBoundaries{7, 15, 31, 63};

bool is_full_attention_layer(std::size_t layer) noexcept {
    return layer % 4 == 3;
}

bool is_boundary(std::size_t layer) noexcept {
    return std::find(kBoundaries.begin(), kBoundaries.end(), layer) != kBoundaries.end();
}

std::size_t argmax(std::span<const float> values) {
    return static_cast<std::size_t>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
}

void print_ledger(const std::filesystem::path& model_path, std::size_t context) {
    const auto weights = std::filesystem::file_size(model_path);
    const auto recurrent = 48 * kVHeads * kState * kState * sizeof(float);
    const auto convolution = 48 * context * kChannels * sizeof(float);
    const auto kv = 16 * context * kKvHeads * kHeadDim * 2 * sizeof(float);
    const auto hidden = 4 * kHidden * sizeof(float);
    const auto total = weights + recurrent + convolution + kv + hidden;
    std::cout << "ledger model_weight_bytes=" << weights << '\n'
              << "ledger recurrent_state_bytes=" << recurrent << '\n'
              << "ledger convolution_history_bytes_context=" << context
              << " value=" << convolution << '\n'
              << "ledger full_attention_kv_bytes_context=" << context
              << " value=" << kv << '\n'
              << "ledger hidden_workspace_bytes=" << hidden << '\n'
              << "ledger tracked_bytes_context=" << total << '\n';
}

void run_ladder(const GgufFile& model, const std::filesystem::path& fixture,
                std::uint32_t token) {
    std::array<std::vector<std::array<float, kChannels>>, kLayers> histories;
    std::array<std::vector<float>, kLayers> states;
    std::array<std::vector<std::vector<float>>, kLayers> keys;
    std::array<std::vector<std::vector<float>>, kLayers> values;
    auto current = embedding(tensor(model, "token_embd.weight"), token);
    const auto prefix_start = std::chrono::steady_clock::now();
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        const auto start = std::chrono::steady_clock::now();
        if (is_full_attention_layer(layer)) {
            current = full_block(model, fixture, layer, 0, current,
                                 keys[layer], values[layer], true);
        } else {
            states[layer] = read_f32(
                checkpoint(fixture, 0, "state_predelta-" + std::to_string(layer)),
                kVHeads * kState * kState);
            const auto state_input = states[layer];
            current = recurrent_block(model, fixture, layer, 0, current, state_input,
                                      histories[layer], states[layer], true);
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (is_boundary(layer)) {
            const auto prefix_elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prefix_start).count();
            require_match("ladder layer output", compare(
                current, read_f32(checkpoint(fixture, 0,
                                             "l_out-" + std::to_string(layer)), kHidden)),
                1.0e2F);
            std::cout << fingerprint("ladder_hidden_after", layer, 0,
                                      dimensions({kHidden}), current) << '\n';
            std::cout << "ladder prefix_layers=" << layer + 1
                      << " boundary_layer_ms=" << elapsed
                      << " prefix_elapsed_ms=" << prefix_elapsed << '\n';
        }
    }

    std::vector<float> final_norm(kHidden);
    rms_rows(current, f32_values(tensor(model, "output_norm.weight"), kHidden),
             final_norm, kHidden);
    const auto logits = gemv(tensor(model, "output.weight"), final_norm, kVocab, kHidden);
    require_match("ladder final norm", compare(
        final_norm, read_f32(checkpoint(fixture, 0, "result_norm"), kHidden)), 1.0F);
    require_match("ladder logits", compare(
        logits, read_f32(fixture / "logits" / "logits-0.f32", kVocab)), 1.0F);
    const auto expected = read_tokens(fixture / "generated_tokens.txt");
    if (expected.empty() || argmax(logits) != expected.front()) {
        throw std::runtime_error("composition ladder argmax mismatch");
    }
    std::cout << "ladder final_argmax=" << argmax(logits) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a17-qwen35-composition-ladder MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = GgufFile::open(argv[1]);
        const std::filesystem::path fixture = argv[2];
        const auto prompt_tokens = read_tokens(fixture / "prompt_tokens.txt");
        if (prompt_tokens.empty()) throw std::runtime_error("fixture has no prompt token");
        print_ledger(argv[1], 1);
        run_ladder(*model, fixture, prompt_tokens.front());
        std::cout << "M6-A17 qwen35 composition ladder PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A17 failed: " << error.what() << '\n';
        return 1;
    }
}
