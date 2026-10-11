#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::size_t kHidden = 5120;
constexpr std::size_t kTile = 512;

struct CpuOpTrace {
    std::string prefix;
    bool enabled = false;
    std::size_t token_index = 0;
    std::size_t tile_start = 0;
    bool final_prompt_tile = false;
    std::unordered_set<std::string> captured;
    std::vector<float> normalized_input;
    std::string error;
};

void write_f32(const std::string& path, const std::vector<float>& values) {
    if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite " + path);
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create " + path);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) throw std::runtime_error("failed writing " + path);
}

bool capture_cpu_op(ggml_tensor* tensor, bool ask, void* user_data) {
    auto& trace = *static_cast<CpuOpTrace*>(user_data);
    if (!trace.enabled) return false;
    static const std::unordered_set<std::string> names = {
        "attn_norm-0", "linear_attn_qkv_mixed-0", "z-0", "beta-0", "beta_sigmoid-0",
        "alpha-0", "a_softplus-0", "gate-0", "state_predelta-0", "conv_output_raw-0",
        "conv_output_silu-0", "q_conv-0", "k_conv-0", "v_conv-0", "q_conv_predelta-0",
        "k_conv_predelta-0", "v_conv_predelta-0", "final_output-0", "linear_attn_out-0",
        "attn_residual-0", "attn_post_norm-0", "ffn_out-0", "post_ffn-0", "l_out-0",
    };
    const std::string name(tensor->name);
    if (names.find(name) == names.end()) return false;
    const bool state = name == "state_predelta-0";
    if (ask) return state || trace.final_prompt_tile;
    if (!state && !trace.final_prompt_tile) return false;
    if (!state && !trace.captured.insert(name).second) return true;
    if (tensor->type != GGML_TYPE_F32 || tensor->data == nullptr) {
        trace.error = "unexpected type or null data for " + name;
        return true;
    }

    std::size_t elements = 1;
    int token_axis = -1;
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        if (axis > 0 && tensor->ne[axis] == static_cast<std::int64_t>(kTile)) {
            token_axis = axis;
            break;
        }
        elements *= static_cast<std::size_t>(tensor->ne[axis]);
    }
    const std::size_t row_offset = token_axis >= 0 ? trace.token_index * elements : 0;
    const auto* source = static_cast<const float*>(tensor->data) + row_offset;
    try {
        std::vector<float> values(elements);
        std::copy_n(source, elements, values.data());
        const std::string path = state
            ? trace.prefix + ".cpu.state-before-token-" + std::to_string(trace.tile_start) + ".f32"
            : trace.prefix + ".cpu." + name + ".f32";
        write_f32(path, values);
        if (name == "attn_norm-0") trace.normalized_input = values;

        if (name == "linear_attn_qkv_mixed-0" || name == "beta-0" || name == "alpha-0") {
            const ggml_tensor* matmul = tensor;
            while (matmul->op == GGML_OP_RESHAPE || matmul->op == GGML_OP_VIEW) matmul = matmul->src[0];
            if (matmul->op != GGML_OP_MUL_MAT || matmul->src[0] == nullptr
                    || matmul->src[1] == nullptr || matmul->src[0]->ne[0] != kHidden
                    || matmul->src[1]->ne[0] != kHidden || trace.normalized_input.size() != kHidden) {
                throw std::runtime_error("unexpected projection graph or input for " + name);
            }
            const auto* weight = matmul->src[0];
            const auto* traits = ggml_get_type_traits(weight->type);
            if (weight->data == nullptr || (weight->type != GGML_TYPE_F32
                    && (traits == nullptr || traits->to_float == nullptr))) {
                throw std::runtime_error("projection weight cannot be dequantized for " + name);
            }
            const auto outputs = static_cast<std::size_t>(weight->ne[1]);
            if (name == "linear_attn_qkv_mixed-0" || name == "beta-0") {
                const std::size_t bytes = static_cast<std::size_t>(weight->nb[1]) * outputs;
                const std::string path = trace.prefix + ".cpu.weight-" + name + ".bin";
                if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite " + path);
                std::ofstream output(path, std::ios::binary);
                output.write(static_cast<const char*>(weight->data), static_cast<std::streamsize>(bytes));
                if (!output) throw std::runtime_error("failed writing " + path);
                std::ofstream meta(trace.prefix + ".cpu.weight-" + name + ".json");
                meta << "{\"name\":\"" << weight->name << "\",\"type\":" << weight->type
                     << ",\"shape\":[" << weight->ne[0] << "," << weight->ne[1]
                     << "],\"strides\":[" << weight->nb[0] << "," << weight->nb[1]
                     << "],\"bytes\":" << bytes << "}\n";
                if (!meta) throw std::runtime_error("failed writing weight metadata for " + name);
            }
            std::vector<float> scalar(outputs), weight_row(kHidden);
            for (std::size_t out = 0; out < outputs; ++out) {
                const auto* row = static_cast<const std::uint8_t*>(weight->data) + out * weight->nb[1];
                if (weight->type == GGML_TYPE_F32) std::copy_n(reinterpret_cast<const float*>(row), kHidden, weight_row.data());
                else traits->to_float(row, weight_row.data(), kHidden);
                double sum = 0.0;
                for (std::size_t in = 0; in < kHidden; ++in)
                    sum += static_cast<double>(weight_row[in]) * trace.normalized_input[in];
                scalar[out] = static_cast<float>(sum);
            }
            write_f32(trace.prefix + ".cpu.scalar-" + name + ".f32", scalar);
            if (trace.captured.insert("scalar-input").second)
                write_f32(trace.prefix + ".cpu.scalar-input.f32", trace.normalized_input);
        }
        std::ofstream meta(trace.prefix + ".cpu-ops.jsonl", std::ios::app);
        meta << "{\"name\":\"" << name << "\",\"type\":\"f32\",\"token_axis\":"
             << token_axis << ",\"elements\":" << elements << ",\"shape\":["
             << tensor->ne[0] << "," << tensor->ne[1] << "," << tensor->ne[2] << "," << tensor->ne[3] << "]}\n";
        if (!meta) trace.error = "failed writing operation metadata for " + name;
    } catch (const std::exception& error) {
        trace.error = error.what();
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    std::ifstream ids_file(argv[2]);
    std::vector<llama_token> ids;
    int id;
    while (ids_file >> id) ids.push_back(id);
    if (ids.empty()) return 3;
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(argv[1], model_params);
    if (!model) return 4;
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 2304;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_threads = 48;
    context_params.n_threads_batch = 48;
    context_params.offload_kqv = false;
    context_params.op_offload = false;
    CpuOpTrace trace;
    if (const char* prefix = std::getenv("MIINFER_LC_CPU_OP_TRACE_PREFIX")) {
        trace.prefix = prefix;
        context_params.cb_eval = capture_cpu_op;
        context_params.cb_eval_user_data = &trace;
    }
    auto* context = llama_init_from_model(model, context_params);
    if (!context) return 5;
    auto batch = llama_batch_init(512, 0, 1);
    std::ofstream output(argv[3], std::ios::binary);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> row(vocab);
    int generated[5];
    for (std::size_t pos = 0; pos < ids.size();) {
        const int count = static_cast<int>(std::min<std::size_t>(512, ids.size() - pos));
        for (int j = 0; j < count; ++j) {
            batch.token[j] = ids[pos + j];
            batch.pos[j] = pos + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = pos + j == ids.size() - 1;
        }
        batch.n_tokens = count;
        trace.enabled = !trace.prefix.empty();
        trace.token_index = count - 1;
        trace.tile_start = pos;
        trace.final_prompt_tile = pos + count == ids.size();
        if (llama_decode(context, batch) != 0) return 6;
        trace.enabled = false;
        if (!trace.error.empty()) throw std::runtime_error(trace.error);
        pos += count;
    }
    for (int step = 0; step < 5; ++step) {
        float* logits = llama_get_logits_ith(context, -1);
        if (!logits) return 7;
        std::copy(logits, logits + vocab, row.begin());
        output.write(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
        generated[step] = static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin());
        if (step < 4) {
            batch.token[0] = generated[step];
            batch.pos[0] = ids.size() + step;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            batch.n_tokens = 1;
            if (llama_decode(context, batch) != 0) return 8;
        }
    }
    output.close();
    for (int i = 0; i < 5; ++i) std::cout << (i ? "," : "") << generated[i];
    std::cout << " vocab=" << vocab << "\n";
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
}
