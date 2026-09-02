#include "miinfer/qwen35_model.hpp"

#include <cmath>
#include <limits>

namespace miinfer {

namespace {

[[noreturn]] void unsupported(const std::string& message) {
    throw GgufError("UNSUPPORTED QWEN35 MODEL CONFIGURATION: " + message);
}

std::uint32_t narrow(std::uint64_t value, const char* name) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        unsupported(std::string(name) + " does not fit uint32");
    }
    return static_cast<std::uint32_t>(value);
}

void require_equal(std::uint64_t actual, std::uint64_t expected, const char* name) {
    if (actual != expected) {
        unsupported(std::string(name) + "=" + std::to_string(actual)
                    + ", expected " + std::to_string(expected));
    }
}

const GgufTensor& find_tensor(const GgufFile& file, const std::string& name) {
    for (const auto& tensor : file.tensors()) {
        if (tensor.name == name) return tensor;
    }
    throw GgufError("missing required qwen35 tensor: " + name);
}

void require_f32_vector(const GgufFile& file, const std::string& name, std::size_t elements) {
    const auto& tensor = find_tensor(file, name);
    if (tensor.type != GgufTensorType::f32
        || tensor.dimensions != std::vector<std::uint64_t>{elements}) {
        unsupported(name + " must be F32[" + std::to_string(elements) + "]");
    }
}

}  // namespace

Qwen35Model Qwen35Model::load(const std::string& path) {
    Qwen35Model model;
    model.file_ = GgufFile::open(path);
    model.artifact_path_ = path;
    model.model_name_ = model.file_->metadata_string("general.name");
    if (model.file_->metadata_string("general.architecture") != "qwen35") {
        unsupported("general.architecture is not qwen35");
    }

    const auto meta = [&](const char* key) { return model.file_->metadata_unsigned(key); };
    require_equal(meta("qwen35.block_count"), 65, "qwen35.block_count");
    require_equal(meta("qwen35.nextn_predict_layers"), 1, "qwen35.nextn_predict_layers");
    require_equal(meta("qwen35.embedding_length"), 5120, "qwen35.embedding_length");
    require_equal(meta("qwen35.feed_forward_length"), 17408, "qwen35.feed_forward_length");
    require_equal(meta("qwen35.attention.head_count"), 24, "qwen35.attention.head_count");
    require_equal(meta("qwen35.attention.head_count_kv"), 4, "qwen35.attention.head_count_kv");
    require_equal(meta("qwen35.attention.key_length"), 256, "qwen35.attention.key_length");
    require_equal(meta("qwen35.attention.value_length"), 256, "qwen35.attention.value_length");
    require_equal(meta("qwen35.full_attention_interval"), 4, "qwen35.full_attention_interval");
    require_equal(meta("qwen35.ssm.inner_size"), 6144, "qwen35.ssm.inner_size");
    require_equal(meta("qwen35.ssm.state_size"), 128, "qwen35.ssm.state_size");
    require_equal(meta("qwen35.ssm.group_count"), 16, "qwen35.ssm.group_count");
    require_equal(meta("qwen35.ssm.conv_kernel"), 4, "qwen35.ssm.conv_kernel");
    require_equal(meta("qwen35.ssm.time_step_rank"), 48, "qwen35.ssm.time_step_rank");
    if (model.file_->metadata_array_size("tokenizer.ggml.tokens") != 248320) {
        unsupported("tokenizer.ggml.tokens has unexpected vocabulary size");
    }

    model.config_.block_count = narrow(meta("qwen35.block_count"), "qwen35.block_count");
    model.config_.nextn_layer_count = narrow(meta("qwen35.nextn_predict_layers"), "qwen35.nextn_predict_layers");
    model.config_.main_layer_count = model.config_.block_count - model.config_.nextn_layer_count;
    model.config_.hidden_size = narrow(meta("qwen35.embedding_length"), "qwen35.embedding_length");
    model.config_.intermediate_size = narrow(meta("qwen35.feed_forward_length"), "qwen35.feed_forward_length");
    model.config_.attention_heads = narrow(meta("qwen35.attention.head_count"), "qwen35.attention.head_count");
    model.config_.kv_heads = narrow(meta("qwen35.attention.head_count_kv"), "qwen35.attention.head_count_kv");
    model.config_.head_dim = narrow(meta("qwen35.attention.key_length"), "qwen35.attention.key_length");
    model.config_.vocab_size = 248320;
    model.config_.context_length = narrow(meta("qwen35.context_length"), "qwen35.context_length");
    model.config_.full_attention_interval = narrow(meta("qwen35.full_attention_interval"), "qwen35.full_attention_interval");
    model.config_.recurrent_inner_size = narrow(meta("qwen35.ssm.inner_size"), "qwen35.ssm.inner_size");
    model.config_.recurrent_state_size = narrow(meta("qwen35.ssm.state_size"), "qwen35.ssm.state_size");
    model.config_.recurrent_group_count = narrow(meta("qwen35.ssm.group_count"), "qwen35.ssm.group_count");
    model.config_.recurrent_conv_kernel = narrow(meta("qwen35.ssm.conv_kernel"), "qwen35.ssm.conv_kernel");
    model.config_.recurrent_time_step_rank = narrow(meta("qwen35.ssm.time_step_rank"), "qwen35.ssm.time_step_rank");
    model.config_.rope_theta = static_cast<float>(model.file_->metadata_float("qwen35.rope.freq_base"));
    model.config_.rms_epsilon = static_cast<float>(
        model.file_->metadata_float("qwen35.attention.layer_norm_rms_epsilon"));
    if (!std::isfinite(model.config_.rope_theta) || !std::isfinite(model.config_.rms_epsilon)
        || model.config_.rope_theta <= 0.0F || model.config_.rms_epsilon <= 0.0F) {
        unsupported("invalid rope or RMSNorm metadata");
    }

    require_f32_vector(*model.file_, "blk.0.attn_norm.weight", model.config_.hidden_size);
    require_f32_vector(*model.file_, "output_norm.weight", model.config_.hidden_size);
    (void)find_tensor(*model.file_, "token_embd.weight");
    const std::string output_name = "output.weight";
    const auto& output = find_tensor(*model.file_, output_name);
    if (output.type != GgufTensorType::q6_k
        || output.dimensions != std::vector<std::uint64_t>{model.config_.hidden_size,
                                                            model.config_.vocab_size}) {
        unsupported("output.weight must be Q6_K[5120,248320]");
    }
    return model;
}

Qwen35TensorView Qwen35Model::tensor(const std::string& name) const {
    return {&find_tensor(*file_, name)};
}

}  // namespace miinfer
