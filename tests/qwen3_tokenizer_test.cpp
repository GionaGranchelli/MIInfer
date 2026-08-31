#include "miinfer/qwen3_model.hpp"
#include "miinfer/qwen3_tokenizer.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "qwen3 tokenizer test: SKIP (model path not supplied)\n";
        return 0;
    }
    try {
        const auto model = miinfer::Qwen3Model::load(argv[1]);
        const auto tokenizer = miinfer::Qwen3Tokenizer::load(model);
        const auto prompt = tokenizer.encode("hello");
        if (prompt != std::vector<std::uint32_t>{14990U}) {
            throw std::runtime_error("Qwen3 tokenizer encoded hello unexpectedly");
        }
        if (tokenizer.encode("hello world") != std::vector<std::uint32_t>{14990U, 1879U}) {
            throw std::runtime_error("Qwen3 tokenizer encoded hello world unexpectedly");
        }
        if (tokenizer.encode("Hello, world!")
            != std::vector<std::uint32_t>{9707U, 11U, 1879U, 0U}) {
            throw std::runtime_error("Qwen3 tokenizer encoded punctuation unexpectedly");
        }
        constexpr std::array<std::uint32_t, 8> generated{
            8U, 341U, 286U, 470U, 330U, 9707U, 11U, 330U};
        const auto text = tokenizer.decode(generated);
        const std::string expected = ") {\n        return \"Hello, \"";
        if (text != expected) {
            throw std::runtime_error("Qwen3 tokenizer detokenized the pinned sequence unexpectedly");
        }
        std::cout << "M4-C3 tokenizer: PASS\n"
                  << "prompt_ids: 14990\n"
                  << "generated_text: " << text << '\n';
    } catch (const std::exception& error) {
        std::cerr << "M4-C3 tokenizer test error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
