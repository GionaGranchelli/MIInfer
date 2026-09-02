#define MIINFER_M6A5_HELPERS_ONLY
#define MIINFER_M6A6_FULL_FORWARD
#include "m6a5_qwen35_hybrid_block.cpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kLayers = 64;
constexpr std::size_t kVocab = 248320;

bool is_full_attention_layer(std::size_t layer) noexcept {
    return layer % 4 == 3;
}

std::size_t argmax(std::span<const float> values) {
    if (values.empty()) throw std::runtime_error("cannot argmax empty logits");
    return static_cast<std::size_t>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a6-qwen35-full-forward MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = GgufFile::open(argv[1]);
        const std::filesystem::path fixture = argv[2];
        const auto prompt_tokens = read_tokens(fixture / "prompt_tokens.txt");
        if (prompt_tokens.size() != 1) throw std::runtime_error("A6 expects one prompt token");

        auto current = embedding(tensor(*model, "token_embd.weight"), prompt_tokens.front());
        report("embedding", compare(current,
                                    read_f32(checkpoint(fixture, 0, "model_input_embed"), kHidden)),
               1.0e-1F);

        std::array<std::vector<std::array<float, kChannels>>, kLayers> recurrent_histories;
        std::array<std::vector<float>, kLayers> recurrent_states;
        std::array<std::vector<std::vector<float>>, kLayers> attention_keys;
        std::array<std::vector<std::vector<float>>, kLayers> attention_values;

        for (std::size_t layer = 0; layer < kLayers; ++layer) {
            if (is_full_attention_layer(layer)) {
                current = full_block(*model, fixture, layer, 0, current,
                                     attention_keys[layer], attention_values[layer]);
            } else {
                const auto expected_state = read_f32(
                    checkpoint(fixture, 0, "state_predelta-" + std::to_string(layer)),
                    kVHeads * kState * kState);
                recurrent_states[layer] = expected_state;
                const auto state_input = recurrent_states[layer];
                current = recurrent_block(*model, fixture, layer, 0, current, state_input,
                                          recurrent_histories[layer], recurrent_states[layer]);
            }
            std::cout << "layer=" << layer << " type="
                      << (is_full_attention_layer(layer) ? "full" : "recurrent")
                      << " output_checked\n";
        }

        std::vector<float> final_norm(kHidden);
        rms_rows(current, f32_values(tensor(*model, "output_norm.weight"), kHidden),
                 final_norm, kHidden);
        const auto final_norm_error = compare(
            final_norm, read_f32(checkpoint(fixture, 0, "result_norm"), kHidden));
        report("final norm", final_norm_error, 1.0F);

        const auto logits = gemv(tensor(*model, "output.weight"), final_norm, kVocab, kHidden);
        const auto logits_error = compare(logits, read_f32(fixture / "logits" / "logits-0.f32", kVocab));
        report("logits", logits_error, 1.0F);
        const auto expected_tokens = read_tokens(fixture / "generated_tokens.txt");
        if (expected_tokens.empty() || argmax(logits) != expected_tokens.front()) {
            throw std::runtime_error("full-forward argmax disagrees with external reference");
        }
        std::cout << "final argmax=" << argmax(logits) << " external=" << expected_tokens.front() << '\n';
        std::cout << "M6-A6 64-layer full forward reference harness complete\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A6 failed: " << error.what() << '\n';
        return 1;
    }
}
