#include "miinfer/openai_api.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/qwen3_tokenizer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-m31-agent-token-count MODEL.gguf < request.jsonl\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto tokenizer = miinfer::Qwen3Tokenizer::load(*model.file());
        std::string body;
        while (std::getline(std::cin, body)) {
            try {
                const auto parsed = miinfer::parse_openai_chat_request(body);
                if (!parsed.request) throw std::runtime_error(parsed.error);
                const auto prompt = miinfer::build_chatml(*parsed.request);
                std::cout << tokenizer.encode(prompt).size() << '\n';
            } catch (const std::exception& error) {
                std::cout << "ERROR " << error.what() << '\n';
            }
            std::cout.flush();
        }
    } catch (const std::exception& error) {
        std::cerr << "tokenizer initialization failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
