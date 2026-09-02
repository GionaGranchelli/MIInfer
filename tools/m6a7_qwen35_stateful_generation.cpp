#define MIINFER_M6A5_HELPERS_ONLY
#define MIINFER_M6A7_STATEFUL_GENERATION
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

bool is_checkpoint_position(std::size_t position) noexcept {
    constexpr std::array<std::size_t, 8> positions{0, 1, 2, 4, 8, 16, 32, 64};
    return std::find(positions.begin(), positions.end(), position) != positions.end();
}

std::size_t argmax(std::span<const float> values) {
    if (values.empty()) throw std::runtime_error("cannot argmax empty logits");
    return static_cast<std::size_t>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: miinfer-m6a7-qwen35-stateful-generation MODEL.gguf FIXTURE_DIR [MAX_POSITION]\n";
        return 2;
    }
    try {
        const auto model = GgufFile::open(argv[1]);
        const std::filesystem::path fixture = argv[2];
        const auto prompt_tokens = read_tokens(fixture / "prompt_tokens.txt");
        const auto expected_tokens = read_tokens(fixture / "generated_tokens.txt");
        if (prompt_tokens.size() != 1 || expected_tokens.size() < 8) {
            throw std::runtime_error("fixture does not contain the required generation");
        }
        const std::size_t max_position = argc == 4 ? std::stoul(argv[3]) : 64;
        if (max_position > 64) throw std::runtime_error("MAX_POSITION must be <= 64");

        std::array<std::vector<std::array<float, kChannels>>, kLayers> recurrent_histories;
        std::array<std::vector<float>, kLayers> recurrent_states;
        std::array<std::vector<std::vector<float>>, kLayers> attention_keys;
        std::array<std::vector<std::vector<float>>, kLayers> attention_values;
        std::uint32_t token = prompt_tokens.front();

        for (std::size_t position = 0; position <= max_position; ++position) {
            const bool validate = is_checkpoint_position(position);
            auto current = embedding(tensor(*model, "token_embd.weight"), token);
            if (validate) {
                report("position=" + std::to_string(position) + " embedding",
                       compare(current, read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden)),
                       1.0e-1F);
            }

            for (std::size_t layer = 0; layer < kLayers; ++layer) {
                if (is_full_attention_layer(layer)) {
                    current = full_block(*model, fixture, layer, position, current,
                                         attention_keys[layer], attention_values[layer], validate);
                } else {
                    if (recurrent_states[layer].empty()) {
                        recurrent_states[layer] = read_f32(
                            checkpoint(fixture, 0, "state_predelta-" + std::to_string(layer)),
                            kVHeads * kState * kState);
                    } else if (validate) {
                        report("position=" + std::to_string(position) + " layer="
                                   + std::to_string(layer) + " state input",
                               compare(recurrent_states[layer], read_f32(
                                   checkpoint(fixture, position,
                                              "state_predelta-" + std::to_string(layer)),
                                   kVHeads * kState * kState)),
                               2.0e1F);
                    }
                    const auto state_input = recurrent_states[layer];
                    current = recurrent_block(*model, fixture, layer, position, current,
                                              state_input, recurrent_histories[layer],
                                              recurrent_states[layer], validate);
                }
            }

            const bool check_logits = validate || position < 8;
            if (check_logits) {
                std::vector<float> final_norm(kHidden);
                rms_rows(current, f32_values(tensor(*model, "output_norm.weight"), kHidden),
                         final_norm, kHidden);
                const auto logits = gemv(tensor(*model, "output.weight"), final_norm, kVocab, kHidden);
                if (validate) {
                    report("position=" + std::to_string(position) + " final norm",
                           compare(final_norm, read_f32(checkpoint(fixture, position, "result_norm"), kHidden)),
                           1.0F);
                    report("position=" + std::to_string(position) + " logits",
                           compare(logits, read_f32(fixture / "logits" /
                                                    ("logits-" + std::to_string(position) + ".f32"), kVocab)),
                           1.0F);
                }
                if (position < expected_tokens.size()) {
                    const auto next = static_cast<std::uint32_t>(argmax(logits));
                    if (next != expected_tokens[position]) {
                        throw std::runtime_error("greedy token mismatch at position "
                                                 + std::to_string(position) + ": got "
                                                 + std::to_string(next) + ", expected "
                                                 + std::to_string(expected_tokens[position]));
                    }
                    if (position < 8 || validate) token = next;
                }
            } else if (position < expected_tokens.size()) {
                token = expected_tokens[position];
            }
            std::cout << "position=" << position << " stateful_forward_checked\n";
        }
        std::cout << "M6-A7 stateful Qwen3.8-27B generation harness complete\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A7 failed: " << error.what() << '\n';
        return 1;
    }
}
