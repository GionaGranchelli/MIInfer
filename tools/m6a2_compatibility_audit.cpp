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
    } catch (const std::exception & error) {
        std::cerr << "M6-A2 audit failed: " << error.what() << '\n';
        return 1;
    }
}
