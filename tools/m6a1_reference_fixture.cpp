#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Record {
    std::string name;
    std::string file;
    std::string type;
    std::vector<int64_t> shape;
    std::size_t bytes = 0;
    int position = -1;
};

struct Capture {
    fs::path root;
    int position = -1;
    int serial = 0;
    std::vector<int> positions;
    std::vector<Record> records;

    bool selected_position() const {
        return std::find(positions.begin(), positions.end(), position) != positions.end();
    }

    bool wanted(const char * raw_name) const {
        const std::string_view name(raw_name == nullptr ? "" : raw_name);
        constexpr std::string_view needles[] = {
            "model.input_embed", "attn_norm-", "linear_attn_qkv_mixed-", "attn_residual-", "attn_post_norm-",
            "ffn_out-", "post_ffn-", "l_out-", "state_predelta-", "conv_output_raw-",
            "final_output-", "attn_q-", "attn_k-", "attn_v-", "attn_output-", "result_norm",
            "Qcur_full-", "Qcur_reshaped-", "Qcur_normed-", "Kcur-", "Kcur_normed-", "Vcur-",
            "gate_reshaped-", "Qcur-", "attn_pregate-", "gate_sigmoid-", "attn_gated-", "z-"
        };
        return std::any_of(std::begin(needles), std::end(needles),
                           [name](const auto needle) { return name.find(needle) != std::string_view::npos; });
    }
};

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned>(ch) << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    return out.str();
}

std::string safe_name(std::string value) {
    for (char & ch : value) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-')) ch = '_';
    }
    return value;
}

bool callback(ggml_tensor * tensor, bool ask, void * userdata) {
    auto & capture = *static_cast<Capture *>(userdata);
    if (!capture.selected_position()) return false;
    if (!capture.wanted(tensor->name)) return false;
    if (!ggml_is_contiguous(tensor)) return false;
    if (ask) return true;

    if (tensor->type != GGML_TYPE_F32) {
        std::cerr << "M6-A1 skip non-F32 checkpoint name=" << tensor->name
                  << " type=" << ggml_type_name(tensor->type) << '\n';
        return true;
    }

    const std::size_t bytes = ggml_nbytes(tensor);
    std::vector<std::uint8_t> data(bytes);
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        std::copy_n(static_cast<const std::uint8_t *>(tensor->data), bytes, data.data());
    } else {
        ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
    }

    const std::string base = std::to_string(capture.position) + "-" + safe_name(tensor->name)
                           + "-" + std::to_string(capture.serial++);
    const fs::path file = capture.root / "tensors" / (base + ".f32");
    std::ofstream output(file, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create checkpoint: " + file.string());
    output.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(bytes));
    if (!output) throw std::runtime_error("cannot write checkpoint: " + file.string());

    Record record;
    record.name = tensor->name;
    record.file = fs::relative(file, capture.root).generic_string();
    record.type = ggml_type_name(tensor->type);
    record.bytes = bytes;
    record.position = capture.position;
    for (int i = 0; i < ggml_n_dims(tensor); ++i) record.shape.push_back(tensor->ne[i]);
    capture.records.push_back(std::move(record));
    return true;
}

void write_bytes(const fs::path & path, const std::uint8_t * data, std::size_t bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create state snapshot: " + path.string());
    output.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    if (!output) throw std::runtime_error("cannot write state snapshot: " + path.string());
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & prompt) {
    const int32_t reported = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                            nullptr, 0, true, false);
    if (reported == std::numeric_limits<int32_t>::min() || reported == 0) {
        throw std::runtime_error("tokenizer returned no tokens or failed");
    }
    const int32_t needed = reported < 0 ? -reported : reported;
    std::vector<llama_token> tokens(static_cast<std::size_t>(needed));
    const int32_t actual = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                          tokens.data(), needed, true, false);
    if (actual != needed) throw std::runtime_error("tokenizer size changed between calls");
    return tokens;
}

llama_token first_argmax(const float * logits, int32_t count) {
    if (logits == nullptr || count <= 0) throw std::runtime_error("missing logits");
    int32_t best = 0;
    for (int32_t i = 1; i < count; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return static_cast<llama_token>(best);
}

void write_f32(const fs::path & path, const float * values, std::size_t count) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create logits file: " + path.string());
    output.write(reinterpret_cast<const char *>(values), static_cast<std::streamsize>(count * sizeof(float)));
    if (!output) throw std::runtime_error("cannot write logits file: " + path.string());
}

void write_tokens(const fs::path & path, const std::vector<llama_token> & tokens) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create token file: " + path.string());
    for (const llama_token token : tokens) output << token << '\n';
}

void write_text(const fs::path & path, const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::vector<char> text(std::max<std::size_t>(4096, tokens.size() * 256));
    int32_t length = llama_detokenize(vocab, tokens.data(), static_cast<int32_t>(tokens.size()),
                                      text.data(), static_cast<int32_t>(text.size()), false, false);
    if (length < 0) {
        text.resize(static_cast<std::size_t>(-length) + 1);
        length = llama_detokenize(vocab, tokens.data(), static_cast<int32_t>(tokens.size()),
                                  text.data(), static_cast<int32_t>(text.size()), false, false);
    }
    if (length < 0) throw std::runtime_error("detokenization failed");
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create text artifact: " + path.string());
    output.write(text.data(), length);
}

void write_manifest(const fs::path & root,
                    const std::string & model_path,
                    const std::string & prompt,
                    const std::vector<llama_token> & prompt_tokens,
                    const std::vector<llama_token> & generated,
                    const Capture & capture,
                    const std::vector<int> & positions,
                    int32_t vocab_size,
                    const llama_model * model) {
    std::ofstream output(root / "manifest.json");
    if (!output) throw std::runtime_error("cannot create manifest");
    output << "{\n"
           << "  \"format_version\": 1,\n"
           << "  \"reference\": {\n"
           << "    \"implementation\": \"llama.cpp\",\n"
           << "    \"commit\": \"c0bc8591e8815c63cb01dd3f051a8b0df02501c9\",\n"
           << "    \"state_format\": \"llama_state_seq_v" << LLAMA_STATE_SEQ_VERSION << "\"\n"
           << "  },\n"
           << "  \"model\": {\n"
           << "    \"path\": \"" << json_escape(model_path) << "\",\n"
           << "    \"description\": \"";
    char description[256] = {};
    llama_model_desc(model, description, sizeof(description));
    output << json_escape(description) << "\",\n"
           << "    \"architecture\": \"qwen35\",\n"
           << "    \"hybrid\": " << (llama_model_is_hybrid(model) ? "true" : "false") << ",\n"
           << "    \"n_embd\": " << llama_model_n_embd(model) << ",\n"
           << "    \"n_layer\": " << llama_model_n_layer(model) << ",\n"
           << "    \"n_vocab\": " << vocab_size << "\n"
           << "  },\n"
           << "  \"workload\": {\n"
           << "    \"prompt\": \"" << json_escape(prompt) << "\",\n"
           << "    \"prompt_tokens\": [";
    for (std::size_t i = 0; i < prompt_tokens.size(); ++i) output << (i ? ", " : "") << prompt_tokens[i];
    output << "],\n    \"generated_tokens\": [";
    for (std::size_t i = 0; i < generated.size(); ++i) output << (i ? ", " : "") << generated[i];
    output << "],\n    \"checkpoint_positions\": [";
    for (std::size_t i = 0; i < positions.size(); ++i) output << (i ? ", " : "") << positions[i];
    output << "]\n  },\n  \"artifacts\": {\n"
           << "    \"prompt_tokens\": \"prompt_tokens.txt\",\n"
           << "    \"generated_tokens\": \"generated_tokens.txt\",\n"
           << "    \"generated_text\": \"generated_text.txt\",\n"
           << "    \"states\": [";
    for (std::size_t i = 0; i < positions.size(); ++i) {
        if (i) output << ", ";
        output << "{\"position\": " << positions[i] << ", \"file\": \"states/state-"
               << positions[i] << ".bin\", \"opaque\": true}";
    }
    output << "],\n    \"logits\": [";
    for (std::size_t i = 0; i < positions.size(); ++i) {
        if (i) output << ", ";
        output << "{\"position\": " << positions[i] << ", \"file\": \"logits/logits-"
               << positions[i] << ".f32\", \"elements\": " << vocab_size << "}";
    }
    output << "],\n    \"tensors\": [\n";
    for (std::size_t i = 0; i < capture.records.size(); ++i) {
        const auto & record = capture.records[i];
        output << "      {\"position\": " << record.position << ", \"name\": \""
               << json_escape(record.name) << "\", \"file\": \"" << record.file
               << "\", \"type\": \"" << record.type << "\", \"bytes\": " << record.bytes
               << ", \"shape\": [";
        for (std::size_t j = 0; j < record.shape.size(); ++j) output << (j ? ", " : "") << record.shape[j];
        output << "]}" << (i + 1 == capture.records.size() ? "\n" : ",\n");
    }
    output << "    ]\n  }\n}\n";
}

int run(const fs::path & model_path, const fs::path & output_root, const std::string & prompt, int max_generated) {
    if (max_generated < 1) throw std::runtime_error("max_generated must be positive");
    fs::create_directories(output_root / "tensors");
    fs::create_directories(output_root / "states");
    fs::create_directories(output_root / "logits");

    llama_backend_init();
    const auto cleanup_backend = [] { llama_backend_free(); };
    try {
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model) throw std::runtime_error("llama.cpp could not load model");

        auto context_params = llama_context_default_params();
        context_params.n_ctx = std::max<uint32_t>(256, static_cast<uint32_t>(max_generated + 64));
        context_params.n_batch = 1;
        context_params.n_ubatch = 1;
        context_params.n_seq_max = 1;
        context_params.n_threads = 0;
        context_params.n_threads_batch = 0;
        context_params.no_perf = true;

        const std::vector<int> positions{0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64};
        Capture capture{output_root};
        capture.positions = positions;
        context_params.cb_eval = callback;
        context_params.cb_eval_user_data = &capture;
        llama_context * context = llama_init_from_model(model, context_params);
        if (!context) {
            llama_model_free(model);
            throw std::runtime_error("llama.cpp could not create context");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const auto prompt_tokens = tokenize(vocab, prompt);
        std::vector<llama_token> generated;
        std::vector<llama_token> input = prompt_tokens;
        int logical_position = 0;
        int generated_count = 0;
        while (!input.empty() && generated_count <= max_generated) {
            const llama_token token = input.front();
            input.clear();
            capture.position = logical_position;
            capture.serial = 0;
            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&token), 1);
            const int rc = llama_decode(context, batch);
            if (rc != 0) throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));

            const bool selected = std::find(positions.begin(), positions.end(), logical_position) != positions.end();
            const float * logits = llama_get_logits_ith(context, -1);
            if (selected) {
                write_f32(output_root / "logits" / ("logits-" + std::to_string(logical_position) + ".f32"),
                          logits, static_cast<std::size_t>(llama_vocab_n_tokens(vocab)));
                const std::size_t state_size = llama_state_seq_get_size(context, 0);
                std::vector<std::uint8_t> state(state_size);
                if (llama_state_seq_get_data(context, state.data(), state.size(), 0) != state_size) {
                    throw std::runtime_error("incomplete state snapshot");
                }
                write_bytes(output_root / "states" / ("state-" + std::to_string(logical_position) + ".bin"),
                            state.data(), state.size());
            }

            if (logical_position >= static_cast<int>(prompt_tokens.size()) - 1) {
                if (generated_count == max_generated) break;
                const llama_token next = first_argmax(logits, llama_vocab_n_tokens(vocab));
                generated.push_back(next);
                input.push_back(next);
                ++generated_count;
            } else {
                input.push_back(prompt_tokens[static_cast<std::size_t>(logical_position + 1)]);
            }
            ++logical_position;
            if (logical_position > max_generated + static_cast<int>(prompt_tokens.size())) break;
        }

        write_tokens(output_root / "prompt_tokens.txt", prompt_tokens);
        write_tokens(output_root / "generated_tokens.txt", generated);
        std::vector<llama_token> all_tokens = prompt_tokens;
        all_tokens.insert(all_tokens.end(), generated.begin(), generated.end());
        write_text(output_root / "generated_text.txt", vocab, all_tokens);
        write_manifest(output_root, model_path.string(), prompt, prompt_tokens, generated, capture, positions,
                       llama_vocab_n_tokens(vocab), model);
        std::cout << "M6-A1 fixture=" << output_root << " prompt_tokens=" << prompt_tokens.size()
                  << " generated_tokens=" << generated.size() << " checkpoints=" << capture.records.size() << '\n';
        llama_free(context);
        llama_model_free(model);
        cleanup_backend();
        return 0;
    } catch (...) {
        cleanup_backend();
        throw;
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: m6a1_reference_fixture MODEL.gguf OUTPUT_DIR PROMPT [MAX_GENERATED]\n";
        return 2;
    }
    try {
        return run(argv[1], argv[2], argv[3], argc == 5 ? std::stoi(argv[4]) : 64);
    } catch (const std::exception & error) {
        std::cerr << "m6a1 reference fixture error: " << error.what() << '\n';
        return 1;
    }
}
