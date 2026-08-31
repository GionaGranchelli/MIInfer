#include "miinfer/qwen3_gpu_layer.hpp"
#include "miinfer/qwen3_tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t parse_id(const std::string& value) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()
        || parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid token ID: " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

std::size_t parse_count(const std::string& value) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) throw std::invalid_argument("invalid token count: " + value);
    return static_cast<std::size_t>(parsed);
}

std::vector<std::uint32_t> parse_ids(const std::string& value) {
    std::vector<std::uint32_t> result;
    std::size_t start = 0;
    while (start < value.size()) {
        const auto comma = value.find(',', start);
        result.push_back(parse_id(value.substr(start, comma == std::string::npos
                                                       ? std::string::npos : comma - start)));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return result;
}

std::uint32_t argmax(const std::vector<float>& logits) {
    if (logits.empty()) throw std::runtime_error("model returned empty logits");
    const auto it = std::max_element(logits.begin(), logits.end());
    if (!std::isfinite(*it)) throw std::runtime_error("model returned non-finite logits");
    return static_cast<std::uint32_t>(std::distance(logits.begin(), it));
}

void print_ids(const char* label, const std::vector<std::uint32_t>& ids) {
    std::cout << label;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << ids[index];
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: miinfer-qwen3-generate MODEL.gguf --prompt TEXT "
                     "[--max-tokens N] [--expect-generated-ids CSV] "
                     "[--expect-generated-text TEXT]\n";
        return 2;
    }
    try {
        const std::string model_path = argv[1];
        std::string prompt;
        std::string expected_text;
        std::size_t max_tokens = 8;
        std::vector<std::uint32_t> expected;
        for (int index = 2; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--prompt" && index + 1 < argc) prompt = argv[++index];
            else if (option == "--max-tokens" && index + 1 < argc) max_tokens = parse_count(argv[++index]);
            else if (option == "--expect-generated-ids" && index + 1 < argc) expected = parse_ids(argv[++index]);
            else if (option == "--expect-generated-text" && index + 1 < argc) expected_text = argv[++index];
            else throw std::invalid_argument("unknown or incomplete option: " + option);
        }
        if (prompt.empty()) throw std::invalid_argument("prompt must not be empty");

        const auto model = miinfer::Qwen3Model::load(model_path);
        const auto tokenizer = miinfer::Qwen3Tokenizer::load(model);
        const auto prompt_ids = tokenizer.encode(prompt);
        if (prompt_ids.empty()) throw std::runtime_error("prompt produced no tokens");
        if (prompt_ids.size() > model.config().context_length
            || max_tokens > model.config().context_length - prompt_ids.size()) {
            throw std::invalid_argument("prompt and generation exceed model context");
        }

        const auto plan = miinfer::Qwen3GpuPlan::build(model);
        miinfer::Qwen3GpuDecodeCache cache(
            model.config().layer_count, model.config().kv_heads, model.config().head_dim,
            prompt_ids.size() + max_tokens);
        miinfer::Qwen3ForwardTrace last;
        for (std::size_t position = 0; position < prompt_ids.size(); ++position) {
            last = miinfer::execute_qwen3_decode_gpu(plan, prompt_ids[position], position, cache);
        }

        std::vector<std::uint32_t> generated;
        generated.reserve(max_tokens);
        for (std::size_t step = 0; step < max_tokens; ++step) {
            const auto token = argmax(last.logits);
            generated.push_back(token);
            if (token == tokenizer.eos_id()) break;
            if (step + 1 < max_tokens) {
                last = miinfer::execute_qwen3_decode_gpu(
                    plan, token, prompt_ids.size() + step, cache);
            }
        }

        const auto generated_text = tokenizer.decode(generated);
        print_ids("prompt_ids: ", prompt_ids);
        print_ids("generated_ids: ", generated);
        std::cout << "generated_text: " << generated_text << '\n';
        if (!expected.empty() && generated != expected) {
            print_ids("expected_ids: ", expected);
            throw std::runtime_error("generated token IDs do not match the pinned expectation");
        }
        if (!expected_text.empty() && generated_text != expected_text) {
            throw std::runtime_error("generated text does not match the pinned expectation");
        }
        std::cout << "M4-C3 greedy text generation: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M4-C3 generation error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
