#include "miinfer/gguf.hpp"

#include <cctype>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace {

struct Group {
    std::size_t count = 0;
    std::size_t bytes = 0;
    std::set<std::string> types;
    std::set<std::string> shapes;
};

std::string layer_pattern(const std::string & name) {
    if (name.rfind("blk.", 0) != 0) return name;
    std::size_t end = 4;
    while (end < name.size() && std::isdigit(static_cast<unsigned char>(name[end]))) ++end;
    if (end == 4 || end >= name.size() || name[end] != '.') return name;
    return "blk.<layer>." + name.substr(end + 1);
}

std::string shape_string(const std::vector<std::uint64_t> & dimensions) {
    std::ostringstream result;
    result << '[';
    for (std::size_t i = 0; i < dimensions.size(); ++i) result << (i ? ',' : ' ') << dimensions[i];
    return result.str() + ']';
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-m6a2-compatibility-audit MODEL.gguf\n";
        return 2;
    }
    try {
        const auto model = miinfer::GgufFile::open(argv[1]);
        std::map<std::string, Group> groups;
        for (const auto & tensor : model->tensors()) {
            auto & group = groups[layer_pattern(tensor.name)];
            ++group.count;
            group.bytes += tensor.byte_size;
            group.types.insert(miinfer::gguf_tensor_type_name(tensor.type));
            group.shapes.insert(shape_string(tensor.dimensions));
        }
        std::cout << "artifact=" << argv[1] << " tensors=" << model->tensors().size() << '\n';
        for (const char * key : {
                 "general.architecture", "qwen35.block_count", "qwen35.embedding_length",
                 "qwen35.feed_forward_length", "qwen35.attention.head_count",
                 "qwen35.attention.head_count_kv", "qwen35.attention.key_length",
                 "qwen35.attention.value_length", "qwen35.full_attention_interval",
                 "qwen35.ssm.inner_size", "qwen35.ssm.state_size", "qwen35.ssm.group_count",
                 "qwen35.ssm.time_step_rank"}) {
            try {
                const auto value = model->metadata_string(key);
                std::cout << "metadata " << key << '=' << value << '\n';
                continue;
            } catch (const std::exception &) {
            }
            try {
                const auto value = model->metadata_unsigned(key);
                std::cout << "metadata " << key << '=' << value << '\n';
            } catch (const std::exception &) {
                std::cout << "metadata " << key << "=<unavailable>\n";
            }
        }
        std::cout << "pattern\tcount\tbytes\ttypes\tshapes\n";
        for (const auto & [pattern, group] : groups) {
            std::cout << pattern << '\t' << group.count << '\t' << group.bytes << '\t';
            bool first = true;
            for (const auto & type : group.types) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << type;
            }
            std::cout << '\t';
            first = true;
            for (const auto & shape : group.shapes) {
                if (!first) std::cout << ';';
                first = false;
                std::cout << shape;
            }
            std::cout << '\n';
        }
        std::cout << "\n--- RECURRENT LAYER SIGNATURES ---\n";
        std::unordered_map<std::string, const miinfer::GgufTensor*> tensor_map;
        for (const auto & t : model->tensors()) {
            tensor_map[t.name] = &t;
        }
        std::map<std::string, std::vector<std::size_t>> signatures;
        for (std::size_t layer = 0; layer < 65; ++layer) {
            std::string ssm_out_name = "blk." + std::to_string(layer) + ".ssm_out.weight";
            if (tensor_map.find(ssm_out_name) == tensor_map.end()) continue;
            std::string qkv_name = "blk." + std::to_string(layer) + ".attn_qkv.weight";
            std::string gate_name = "blk." + std::to_string(layer) + ".attn_gate.weight";
            std::string ffn_g_name = "blk." + std::to_string(layer) + ".ffn_gate.weight";
            std::string ffn_u_name = "blk." + std::to_string(layer) + ".ffn_up.weight";
            std::string ffn_d_name = "blk." + std::to_string(layer) + ".ffn_down.weight";
            
            std::string qkv_t = tensor_map.count(qkv_name) ? miinfer::gguf_tensor_type_name(tensor_map[qkv_name]->type) : "NONE";
            std::string gate_t = tensor_map.count(gate_name) ? miinfer::gguf_tensor_type_name(tensor_map[gate_name]->type) : "NONE";
            std::string ssm_t = miinfer::gguf_tensor_type_name(tensor_map[ssm_out_name]->type);
            std::string ffn_g_t = tensor_map.count(ffn_g_name) ? miinfer::gguf_tensor_type_name(tensor_map[ffn_g_name]->type) : "NONE";
            std::string ffn_u_t = tensor_map.count(ffn_u_name) ? miinfer::gguf_tensor_type_name(tensor_map[ffn_u_name]->type) : "NONE";
            std::string ffn_d_t = tensor_map.count(ffn_d_name) ? miinfer::gguf_tensor_type_name(tensor_map[ffn_d_name]->type) : "NONE";
            
            std::string sig = "QKV:" + qkv_t + " | Gate:" + gate_t + " | SSM-out:" + ssm_t + " | FFN-g/u:" + ffn_g_t + "/" + ffn_u_t + " | FFN-down:" + ffn_d_t;
            signatures[sig].push_back(layer);
        }
        for (const auto & [sig, layers] : signatures) {
            std::cout << "Signature: [" << sig << "]\n";
            std::cout << "  Count: " << layers.size() << "\n  Layers: ";
            for (size_t l : layers) std::cout << l << " ";
            std::cout << "\n";
        }

        std::cout << "\n--- ATTENTION LAYER SIGNATURES ---\n";
        std::map<std::string, std::vector<std::size_t>> attn_signatures;
        for (std::size_t layer = 0; layer < 65; ++layer) {
            std::string q_name = "blk." + std::to_string(layer) + ".attn_q.weight";
            if (tensor_map.find(q_name) == tensor_map.end()) continue;
            std::string k_name = "blk." + std::to_string(layer) + ".attn_k.weight";
            std::string v_name = "blk." + std::to_string(layer) + ".attn_v.weight";
            std::string o_name = "blk." + std::to_string(layer) + ".attn_output.weight";
            std::string ffn_g_name = "blk." + std::to_string(layer) + ".ffn_gate.weight";
            std::string ffn_u_name = "blk." + std::to_string(layer) + ".ffn_up.weight";
            std::string ffn_d_name = "blk." + std::to_string(layer) + ".ffn_down.weight";
            
            std::string q_t = tensor_map.count(q_name) ? miinfer::gguf_tensor_type_name(tensor_map[q_name]->type) : "NONE";
            std::string k_t = tensor_map.count(k_name) ? miinfer::gguf_tensor_type_name(tensor_map[k_name]->type) : "NONE";
            std::string v_t = tensor_map.count(v_name) ? miinfer::gguf_tensor_type_name(tensor_map[v_name]->type) : "NONE";
            std::string o_t = tensor_map.count(o_name) ? miinfer::gguf_tensor_type_name(tensor_map[o_name]->type) : "NONE";
            std::string ffn_g_t = tensor_map.count(ffn_g_name) ? miinfer::gguf_tensor_type_name(tensor_map[ffn_g_name]->type) : "NONE";
            std::string ffn_u_t = tensor_map.count(ffn_u_name) ? miinfer::gguf_tensor_type_name(tensor_map[ffn_u_name]->type) : "NONE";
            std::string ffn_d_t = tensor_map.count(ffn_d_name) ? miinfer::gguf_tensor_type_name(tensor_map[ffn_d_name]->type) : "NONE";
            
            std::string sig = "Q:" + q_t + " | K:" + k_t + " | V:" + v_t + " | O:" + o_t + " | FFN-g/u:" + ffn_g_t + "/" + ffn_u_t + " | FFN-down:" + ffn_d_t;
            attn_signatures[sig].push_back(layer);
        }
        for (const auto & [sig, layers] : attn_signatures) {
            std::cout << "Signature: [" << sig << "]\n";
            std::cout << "  Count: " << layers.size() << "\n  Layers: ";
            for (size_t l : layers) std::cout << l << " ";
            std::cout << "\n";
        }
    } catch (const std::exception & error) {
        std::cerr << "M6-A2 audit failed: " << error.what() << '\n';
        return 1;
    }
}
