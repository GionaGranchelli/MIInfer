#include "miinfer/build_info.hpp"
#include "miinfer/model_plan.hpp"
#include "miinfer/qwen3_model.hpp"
#include "miinfer/sha256.hpp"

#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr char kExpectedSha256[] =
    "458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628";

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << character; break;
        }
    }
    return output.str();
}

std::map<std::string, std::size_t> type_counts(const miinfer::Qwen3Model& model) {
    std::map<std::string, std::size_t> counts;
    for (const auto& tensor : model.tensors()) ++counts[miinfer::gguf_tensor_type_name(tensor.type)];
    return counts;
}

void print_text(
    const miinfer::Qwen3Model& model,
    const std::string& hash,
    const miinfer::Qwen3GpuPlan* plan,
    bool integrity_ok) {
    const auto& config = model.config();
    std::cout << "model: Qwen3-8B\n"
              << "model_name: " << model.model_name() << '\n'
              << "architecture: qwen3\n"
              << "quantization: Q4_0 projections, Q6_K output\n"
              << "artifact_sha256: " << hash << '\n'
              << "layers: " << config.layer_count << '\n'
              << "hidden: " << config.hidden_size << '\n'
              << "intermediate: " << config.intermediate_size << '\n'
              << "attention_heads: " << config.attention_heads << '\n'
              << "kv_heads: " << config.kv_heads << '\n'
              << "head_dim: " << config.head_dim << '\n'
              << "vocab: " << config.vocab_size << '\n'
              << "context: " << config.context_length << '\n'
              << "tensor_count: " << model.tensors().size() << '\n'
              << "host_weight_bytes: " << model.total_weight_bytes() << '\n';
    std::cout << "tensor_types:";
    for (const auto& [type, count] : type_counts(model)) std::cout << ' ' << type << '=' << count;
    std::cout << '\n';
    if (plan != nullptr) {
        std::cout << "gpu: " << plan->device().name << " (" << plan->device().architecture << ")\n"
                  << "gpu_weight_arena_bytes: " << plan->weights().bytes() << '\n'
                  << "workspace_bytes: " << plan->workspace_bytes() << '\n'
                  << "weight_integrity_samples: " << (integrity_ok ? "PASS" : "FAIL") << '\n'
                  << "kernel_plan: Q=" << miinfer::q4_gemv_kernel_name(plan->kernel_for("q"))
                  << " K=" << miinfer::q4_gemv_kernel_name(plan->kernel_for("k"))
                  << " V=" << miinfer::q4_gemv_kernel_name(plan->kernel_for("v"))
                  << " O=" << miinfer::q4_gemv_kernel_name(plan->kernel_for("o"))
                  << " Gate=" << miinfer::q4_gemv_kernel_name(plan->kernel_for("gate"))
                  << " Up=" << miinfer::q4_gemv_kernel_name(plan->kernel_for("up"))
                  << " Down=" << miinfer::q4_gemv_kernel_name(plan->kernel_for("down")) << '\n';
        std::cout << "static_buffers:\n";
        for (const auto& buffer : plan->buffers()) {
            std::cout << "  " << buffer.name << " offset=" << buffer.offset
                      << " bytes=" << buffer.bytes << '\n';
        }
    }
}

void print_json(
    const miinfer::Qwen3Model& model,
    const std::string& hash,
    const miinfer::Qwen3GpuPlan* plan,
    bool integrity_ok) {
    const auto& config = model.config();
    std::cout << "{\n"
              << "  \"model\": \"Qwen3-8B\",\n"
              << "  \"model_name\": \"" << json_escape(model.model_name()) << "\",\n"
              << "  \"revision\": \"b968826d9c46dd6066d109eabc6255188de91218\",\n"
              << "  \"artifact_sha256\": \"" << hash << "\",\n"
              << "  \"architecture\": \"qwen3\",\n"
              << "  \"quantization\": \"Q4_0 projections, Q6_K output\",\n"
              << "  \"configuration\": {\n"
              << "    \"layers\": " << config.layer_count << ",\n"
              << "    \"hidden_size\": " << config.hidden_size << ",\n"
              << "    \"intermediate_size\": " << config.intermediate_size << ",\n"
              << "    \"attention_heads\": " << config.attention_heads << ",\n"
              << "    \"kv_heads\": " << config.kv_heads << ",\n"
              << "    \"head_dim\": " << config.head_dim << ",\n"
              << "    \"vocab_size\": " << config.vocab_size << ",\n"
              << "    \"context_length\": " << config.context_length << ",\n"
              << "    \"rope_theta\": " << config.rope_theta << ",\n"
              << "    \"rms_epsilon\": " << config.rms_epsilon << "\n"
              << "  },\n"
              << "  \"tensor_count\": " << model.tensors().size() << ",\n"
              << "  \"host_weight_bytes\": " << model.total_weight_bytes() << ",\n"
              << "  \"tensor_types\": {";
    bool first = true;
    for (const auto& [type, count] : type_counts(model)) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << "\n    \"" << type << "\": " << count;
    }
    if (!first) std::cout << '\n' << "  ";
    std::cout << "},\n  \"kernel_plan\": {\n"
              << "    \"Q\": \"zero-point-128\",\n"
              << "    \"K\": \"zero-point-wave64\",\n"
              << "    \"V\": \"zero-point-wave64\",\n"
              << "    \"O\": \"zero-point-128\",\n"
              << "    \"Gate\": \"zero-point-128\",\n"
              << "    \"Up\": \"zero-point-128\",\n"
              << "    \"Down\": \"zero-point-256\"\n"
              << "  },\n"
              << "  \"workspace_bytes\": " << (plan == nullptr ? 0 : plan->workspace_bytes()) << ",\n"
              << "  \"weight_integrity_samples\": " << (plan == nullptr ? "null" : (integrity_ok ? "true" : "false")) << ",\n"
              << "  \"gpu\": {\n"
              << "    \"validated\": " << (plan != nullptr ? "true" : "false");
    if (plan != nullptr) {
        std::cout << ",\n    \"name\": \"" << json_escape(plan->device().name)
                  << "\",\n    \"architecture\": \"" << json_escape(plan->device().architecture)
                  << "\",\n    \"total_vram_bytes\": " << plan->device().total_vram_bytes
                  << ",\n    \"weight_arena_bytes\": " << plan->weights().bytes();
    }
    std::cout << "\n  }\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: miinfer-model-info MODEL.gguf [--no-gpu] [--json]\n";
        return 2;
    }
    bool use_gpu = true;
    bool json = false;
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--no-gpu") use_gpu = false;
        else if (option == "--json") json = true;
        else {
            std::cerr << "unknown option: " << option << '\n';
            return 2;
        }
    }
    try {
        const std::string path = argv[1];
        const auto model = miinfer::Qwen3Model::load(path);
        const auto hash = miinfer::sha256_file(path);
        if (hash != kExpectedSha256) {
            throw std::runtime_error("artifact SHA256 mismatch; expected pinned Qwen3-8B Q4_0 artifact");
        }
        std::unique_ptr<miinfer::Qwen3GpuPlan> plan;
        if (use_gpu) plan = std::make_unique<miinfer::Qwen3GpuPlan>(miinfer::Qwen3GpuPlan::build(model));
        bool integrity_ok = true;
        if (plan != nullptr) {
            for (const auto& name : {"token_embd.weight", "blk.0.attn_k.weight",
                                     "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight",
                                     "output_norm.weight"}) {
                integrity_ok = plan->verify_tensor_sample(name) && integrity_ok;
            }
            if (!integrity_ok) throw std::runtime_error("uploaded weight integrity sample failed");
        }
        if (json) print_json(model, hash, plan.get(), integrity_ok);
        else print_text(model, hash, plan.get(), integrity_ok);
    } catch (const std::exception& error) {
        std::cerr << "MIInfer model inspection failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
