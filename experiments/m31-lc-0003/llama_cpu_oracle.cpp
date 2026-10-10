#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kLlamaRevision[] = "91c631b21d6e5d09e9c6659efdf6baeef5a44ddb";
constexpr std::size_t kExpectedPromptTokens = 2048;
constexpr std::size_t kPromptTile = 512;
constexpr std::size_t kExpectedVocab = 248320;
constexpr std::size_t kDecisions = 5;
constexpr std::size_t kExpectedForcedHistory = kDecisions - 1;

std::vector<llama_token> read_ids(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open token IDs: " + path);
    std::vector<llama_token> ids;
    std::uint64_t value = 0;
    while (input >> value) {
        if (value > static_cast<std::uint64_t>(std::numeric_limits<llama_token>::max())) {
            throw std::runtime_error("token ID exceeds llama_token range: " + path);
        }
        ids.push_back(static_cast<llama_token>(value));
    }
    if (!input.eof()) throw std::runtime_error("invalid token ID: " + path);
    return ids;
}

void fill_batch(llama_batch& batch, const llama_token* tokens, std::size_t count,
                llama_pos first_position, bool request_last_logits) {
    if (count > kPromptTile) throw std::runtime_error("batch exceeds the 512-token tile");
    batch.n_tokens = static_cast<std::int32_t>(count);
    for (std::size_t i = 0; i < count; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = first_position + static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = request_last_logits && i + 1 == count ? 1 : 0;
    }
}

void append_logits(llama_context* context, std::size_t vocab,
                   std::vector<float>& output) {
    const float* logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr) throw std::runtime_error("llama_get_logits_ith returned null");
    for (std::size_t i = 0; i < vocab; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("CPU reference produced a non-finite logit");
    }
    output.insert(output.end(), logits, logits + vocab);
}

struct BatchOwner {
    explicit BatchOwner(std::int32_t capacity) : value(llama_batch_init(capacity, 0, 1)) {
        if (value.token == nullptr || value.pos == nullptr || value.n_seq_id == nullptr
            || value.seq_id == nullptr || value.logits == nullptr) {
            throw std::runtime_error("llama_batch_init failed");
        }
    }
    ~BatchOwner() { llama_batch_free(value); }
    llama_batch value;
};

struct ContextDeleter { void operator()(llama_context* value) const { llama_free(value); } };
using ContextOwner = std::unique_ptr<llama_context, ContextDeleter>;

std::vector<float> capture_pass(llama_model* model, const std::vector<llama_token>& prompt,
                                const std::vector<llama_token>& forced, std::size_t vocab,
                                std::int32_t threads) {
    auto params = llama_context_default_params();
    params.n_ctx = 2304;
    params.n_batch = kPromptTile;
    params.n_ubatch = kPromptTile;
    params.n_seq_max = 1;
    params.n_outputs_max = 1;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.type_k = GGML_TYPE_F16;
    params.type_v = GGML_TYPE_F16;
    params.offload_kqv = false;
    params.op_offload = false;
    params.no_perf = true;

    ContextOwner context(llama_init_from_model(model, params));
    if (!context) throw std::runtime_error("llama_init_from_model failed");
    BatchOwner batch(static_cast<std::int32_t>(kPromptTile));
    std::vector<float> logits;
    logits.reserve(kDecisions * vocab);

    for (std::size_t offset = 0; offset < prompt.size(); offset += kPromptTile) {
        const auto count = std::min(kPromptTile, prompt.size() - offset);
        fill_batch(batch.value, prompt.data() + offset, count,
                   static_cast<llama_pos>(offset), offset + count == prompt.size());
        const auto result = llama_decode(context.get(), batch.value);
        if (result != 0) throw std::runtime_error("llama_decode failed on prompt tile at " + std::to_string(offset)
                                                   + " with code " + std::to_string(result));
    }
    append_logits(context.get(), vocab, logits); // decision 1

    for (std::size_t i = 0; i < forced.size(); ++i) {
        fill_batch(batch.value, &forced[i], 1,
                   static_cast<llama_pos>(prompt.size() + i), true);
        const auto result = llama_decode(context.get(), batch.value);
        if (result != 0) throw std::runtime_error("llama_decode failed for forced token " + std::to_string(i)
                                                   + " with code " + std::to_string(result));
        append_logits(context.get(), vocab, logits); // decisions 2-5
    }
    return logits;
}

void write_f32(const std::filesystem::path& path, const std::vector<float>& values) {
    if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite " + path.string());
    std::ofstream output(path, std::ios::binary | std::ios::out);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

llama_token top_id(const std::vector<float>& values, std::size_t vocab, std::size_t decision) {
    const auto start = values.begin() + decision * vocab;
    return static_cast<llama_token>(std::distance(start, std::max_element(start, start + vocab)));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: m31-lc-0003-cpu-oracle MODEL.gguf PROMPT.ids FORCED.ids OUTPUT_PREFIX\n";
        return 2;
    }
    try {
        const std::string model_path = argv[1];
        const std::string output_prefix = argv[4];
        const auto prompt = read_ids(argv[2]);
        const auto forced = read_ids(argv[3]);
        if (prompt.size() != kExpectedPromptTokens || forced.size() != kExpectedForcedHistory) {
            throw std::runtime_error("expected 2,048 prompt IDs and four forced history IDs");
        }

        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.use_mmap = true;
        model_params.use_mlock = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
        if (!model) throw std::runtime_error("llama_model_load_from_file failed");

        const auto* vocab = llama_model_get_vocab(model.get());
        const auto vocab_size = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));
        if (vocab_size != kExpectedVocab || llama_model_n_layer(model.get()) != 64
            || llama_model_n_embd(model.get()) != 5120) {
            throw std::runtime_error("loaded GGUF architecture dimensions do not match the 27B Qwen3.5 model contract");
        }

        constexpr std::int32_t threads = 24;
        std::array<std::vector<float>, 2> captures;
        for (std::size_t pass = 0; pass < captures.size(); ++pass) {
            captures[pass] = capture_pass(model.get(), prompt, forced, vocab_size, threads);
            write_f32(output_prefix + ".run" + std::to_string(pass + 1) + ".logits.f32", captures[pass]);
        }
        const bool reproducible = captures[0].size() == captures[1].size()
            && std::equal(captures[0].begin(), captures[0].end(), captures[1].begin());

        std::array<char, 256> description{};
        llama_model_desc(model.get(), description.data(), description.size());
        std::ofstream metadata(output_prefix + ".metadata.json", std::ios::out);
        if (!metadata) throw std::runtime_error("cannot write CPU reference metadata");
        metadata << "{\n"
                 << "  \"reference\": \"llama.cpp CPU\",\n"
                 << "  \"revision\": \"" << kLlamaRevision << "\",\n"
                 << "  \"model_desc\": \"" << description.data() << "\",\n"
                 << "  \"prompt_tokens\": " << prompt.size() << ",\n"
                 << "  \"forced_history\": [";
        for (std::size_t i = 0; i < forced.size(); ++i) metadata << (i ? ", " : "") << forced[i];
        metadata << "],\n"
                 << "  \"decisions\": " << kDecisions << ",\n"
                 << "  \"vocab_size\": " << vocab_size << ",\n"
                 << "  \"layers\": " << llama_model_n_layer(model.get()) << ",\n"
                 << "  \"embedding_size\": " << llama_model_n_embd(model.get()) << ",\n"
                 << "  \"trained_context\": " << llama_model_n_ctx_train(model.get()) << ",\n"
                 << "  \"context\": 2304,\n"
                 << "  \"prompt_tile\": " << kPromptTile << ",\n"
                 << "  \"threads\": " << threads << ",\n"
                 << "  \"gpu_layers\": 0,\n"
                 << "  \"kv_type_k\": \"f16\",\n"
                 << "  \"kv_type_v\": \"f16\",\n"
                 << "  \"bitwise_reproducible\": " << (reproducible ? "true" : "false") << ",\n"
                 << "  \"run1_top1\": [";
        for (std::size_t d = 0; d < kDecisions; ++d) {
            if (d) metadata << ", ";
            metadata << top_id(captures[0], vocab_size, d);
        }
        metadata << "]\n}\n";
        if (!metadata) throw std::runtime_error("failed writing CPU reference metadata");
        std::cout << "reference_revision=" << kLlamaRevision
                  << " model=" << description.data()
                  << " vocab=" << vocab_size
                  << " layers=" << llama_model_n_layer(model.get())
                  << " reproducible=" << (reproducible ? "PASS" : "FAIL") << '\n';
        llama_backend_free();
        return reproducible ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "M31-LC-0003 CPU oracle failed: " << error.what() << '\n';
        return 1;
    }
}
