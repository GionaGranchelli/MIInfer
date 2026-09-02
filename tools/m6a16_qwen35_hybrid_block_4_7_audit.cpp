#define MIINFER_M6A7_STATEFUL_GENERATION
#define MIINFER_M6A15_HELPERS_ONLY
#include "m6a15_qwen35_hybrid_block_audit.cpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void run_combined(const GgufFile& model, const std::filesystem::path& fixture,
                  RuntimeState& first, RuntimeState& second,
                  const std::vector<std::uint32_t>& generated,
                  std::uint32_t prompt_token) {
    if (generated.size() <= kA15MaxPosition) {
        throw std::runtime_error("fixture lacks tokens through position 16");
    }
    for (std::size_t position = 0; position <= kA15MaxPosition; ++position) {
        const bool check = is_checkpoint_position(position);
        const auto token = position == 0 ? prompt_token : generated[position - 1];
        auto current = embedding(tensor(model, "token_embd.weight"), token);
        if (check) {
            require_match("combined embedding", compare(
                current, read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden)),
                1.0e-6F);
        }
        for (std::size_t layer = 0; layer < kA15RecurrentLayers; ++layer) {
            const auto state_input = first.recurrent[layer];
            current = recurrent_block(model, fixture, layer, position, current, state_input,
                                      first.convolution[layer], first.recurrent[layer], check);
        }
        current = full_block(model, fixture, 3, position, current, first.keys, first.values, check);
        for (std::size_t offset = 0; offset < kA15RecurrentLayers; ++offset) {
            const auto layer = 4 + offset;
            std::cout << fingerprint("recurrent_before", layer, position,
                                      dimensions({kVHeads, kState, kState}), second.recurrent[offset])
                      << '\n';
            if (check) {
                require_match("recurrent state input", compare(
                    second.recurrent[offset], read_f32(
                        checkpoint(fixture, position,
                                   "state_predelta-" + std::to_string(layer)),
                        kVHeads * kState * kState)), 2.0e1F);
            }
            const auto state_input = second.recurrent[offset];
            current = recurrent_block(model, fixture, layer, position, current, state_input,
                                      second.convolution[offset], second.recurrent[offset], check);
            std::cout << fingerprint("recurrent_after", layer, position,
                                      dimensions({kVHeads, kState, kState}), second.recurrent[offset])
                      << '\n';
            std::cout << fingerprint("hidden_after", layer, position,
                                      dimensions({kHidden}), current) << '\n';
        }
        std::cout << cache_fingerprint("kv_k_before", 7, position, second.keys) << '\n';
        std::cout << cache_fingerprint("kv_v_before", 7, position, second.values) << '\n';
        current = full_block(model, fixture, 7, position, current,
                             second.keys, second.values, check);
        std::cout << cache_fingerprint("kv_k_after", 7, position, second.keys) << '\n';
        std::cout << cache_fingerprint("kv_v_after", 7, position, second.values) << '\n';
        std::cout << fingerprint("hidden_after", 7, position,
                                  dimensions({kHidden}), current) << '\n';
        std::cout << "position=" << position << " hybrid_block_4_7_checked\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a16-qwen35-hybrid-block-4-7-audit MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = GgufFile::open(argv[1]);
        const std::filesystem::path fixture = argv[2];
        const auto prompt_tokens = read_tokens(fixture / "prompt_tokens.txt");
        const auto generated_tokens = read_tokens(fixture / "generated_tokens.txt");
        if (prompt_tokens.empty()) throw std::runtime_error("fixture has no prompt token");

        RuntimeState first;
        RuntimeState second;
        reset(first, initial_states(fixture, 0));
        reset(second, initial_states(fixture, 4));
        run_combined(*model, fixture, first, second, generated_tokens, prompt_tokens.front());
        std::cout << "M6-A16 qwen35 layers-4-7 hybrid block audit PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A16 failed: " << error.what() << '\n';
        return 1;
    }
}
