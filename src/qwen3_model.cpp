#include "miinfer/qwen3_model.hpp"

#include <cmath>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace miinfer {

namespace {

constexpr std::uint32_t kExpectedLayers = 36;
constexpr std::uint32_t kExpectedHidden = 4096;
constexpr std::uint32_t kExpectedIntermediate = 12288;
constexpr std::uint32_t kExpectedHeads = 32;
constexpr std::uint32_t kExpectedKvHeads = 8;
constexpr std::uint32_t kExpectedHeadDim = 128;
constexpr std::uint32_t kExpectedVocab = 151936;
constexpr std::uint32_t kExpectedContext = 40960;

[[noreturn]] void unsupported(const std::string& message) {
    throw GgufError("UNSUPPORTED MODEL CONFIGURATION: " + message);
}

std::uint32_t narrow_u32(std::uint64_t value, const char* name) {
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
        if (tensor.name == name) {
            return tensor;
        }
    }
    throw GgufError("missing required tensor: " + name);
}

Qwen3TensorView require_tensor(
    const GgufFile& file,
    const std::string& name,
    GgufTensorType type,
    std::initializer_list<std::uint64_t> dimensions) {
    const auto& tensor = find_tensor(file, name);
    if (tensor.type != type) {
        unsupported(name + " has type " + gguf_tensor_type_name(tensor.type)
                    + ", expected " + gguf_tensor_type_name(type));
    }
    if (tensor.dimensions.size() != dimensions.size()
        || !std::equal(tensor.dimensions.begin(), tensor.dimensions.end(), dimensions.begin())) {
        std::ostringstream message;
        message << name << " has unexpected dimensions";
        unsupported(message.str());
    }
    return {&tensor};
}

std::size_t add_bytes(std::size_t total, std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - total) {
        throw GgufError("model weight byte accounting overflow");
    }
    return total + bytes;
}

}  // namespace

Qwen3Model Qwen3Model::load(const std::string& path) {
    Qwen3Model model;
    model.file_ = GgufFile::open(path);
    model.artifact_path_ = path;
    model.model_name_ = model.file_->metadata_string("general.name");
    if (model.file_->metadata_string("general.architecture") != "qwen3") {
        unsupported("general.architecture is not qwen3");
    }

    const auto require_meta = [&](const char* key, std::uint64_t expected) {
        require_equal(model.file_->metadata_unsigned(key), expected, key);
    };
    require_meta("qwen3.block_count", kExpectedLayers);
    require_meta("qwen3.embedding_length", kExpectedHidden);
    require_meta("qwen3.feed_forward_length", kExpectedIntermediate);
    require_meta("qwen3.attention.head_count", kExpectedHeads);
    require_meta("qwen3.attention.head_count_kv", kExpectedKvHeads);
    require_meta("qwen3.attention.key_length", kExpectedHeadDim);
    require_meta("qwen3.attention.value_length", kExpectedHeadDim);
    require_meta("qwen3.context_length", kExpectedContext);
    if (model.file_->metadata_array_size("tokenizer.ggml.tokens") != kExpectedVocab
        || !model.file_->metadata_array_is_string("tokenizer.ggml.tokens")) {
        unsupported("tokenizer.ggml.tokens has unexpected vocabulary size");
    }
    model.config_.layer_count = narrow_u32(kExpectedLayers, "qwen3.block_count");
    model.config_.hidden_size = narrow_u32(kExpectedHidden, "qwen3.embedding_length");
    model.config_.intermediate_size = narrow_u32(kExpectedIntermediate, "qwen3.feed_forward_length");
    model.config_.attention_heads = narrow_u32(kExpectedHeads, "qwen3.attention.head_count");
    model.config_.kv_heads = narrow_u32(kExpectedKvHeads, "qwen3.attention.head_count_kv");
    model.config_.head_dim = narrow_u32(kExpectedHeadDim, "qwen3.attention.key_length");
    model.config_.vocab_size = narrow_u32(kExpectedVocab, "tokenizer.ggml.tokens");
    model.config_.context_length = narrow_u32(kExpectedContext, "qwen3.context_length");
    model.config_.rope_theta = static_cast<float>(model.file_->metadata_float("qwen3.rope.freq_base"));
    model.config_.rms_epsilon = static_cast<float>(
        model.file_->metadata_float("qwen3.attention.layer_norm_rms_epsilon"));
    if (!std::isfinite(model.config_.rope_theta) || !std::isfinite(model.config_.rms_epsilon)
        || model.config_.rope_theta <= 0.0F || model.config_.rms_epsilon <= 0.0F) {
        unsupported("invalid rope or RMSNorm metadata");
    }

    model.token_embeddings_ = require_tensor(
        *model.file_, "token_embd.weight", GgufTensorType::q4_0, {kExpectedHidden, kExpectedVocab});
    model.final_norm_ = require_tensor(
        *model.file_, "output_norm.weight", GgufTensorType::f32, {kExpectedHidden});
    model.output_ = require_tensor(
        *model.file_, "output.weight", GgufTensorType::q6_k, {kExpectedHidden, kExpectedVocab});

    model.layers_.reserve(kExpectedLayers);
    for (std::uint32_t layer = 0; layer < kExpectedLayers; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        Qwen3LayerWeights weights;
        weights.attention_norm = require_tensor(
            *model.file_, prefix + "attn_norm.weight", GgufTensorType::f32, {kExpectedHidden});
        weights.q = require_tensor(
            *model.file_, prefix + "attn_q.weight", GgufTensorType::q4_0,
            {kExpectedHidden, kExpectedHidden});
        weights.k = require_tensor(
            *model.file_, prefix + "attn_k.weight", GgufTensorType::q4_0,
            {kExpectedHidden, kExpectedKvHeads * kExpectedHeadDim});
        weights.v = require_tensor(
            *model.file_, prefix + "attn_v.weight", GgufTensorType::q4_0,
            {kExpectedHidden, kExpectedKvHeads * kExpectedHeadDim});
        weights.output = require_tensor(
            *model.file_, prefix + "attn_output.weight", GgufTensorType::q4_0,
            {kExpectedHidden, kExpectedHidden});
        weights.q_norm = require_tensor(
            *model.file_, prefix + "attn_q_norm.weight", GgufTensorType::f32, {kExpectedHeadDim});
        weights.k_norm = require_tensor(
            *model.file_, prefix + "attn_k_norm.weight", GgufTensorType::f32, {kExpectedHeadDim});
        weights.ffn_norm = require_tensor(
            *model.file_, prefix + "ffn_norm.weight", GgufTensorType::f32, {kExpectedHidden});
        weights.gate = require_tensor(
            *model.file_, prefix + "ffn_gate.weight", GgufTensorType::q4_0,
            {kExpectedHidden, kExpectedIntermediate});
        weights.up = require_tensor(
            *model.file_, prefix + "ffn_up.weight", GgufTensorType::q4_0,
            {kExpectedHidden, kExpectedIntermediate});
        weights.down = require_tensor(
            *model.file_, prefix + "ffn_down.weight", GgufTensorType::q4_0,
            {kExpectedIntermediate, kExpectedHidden});
        model.layers_.push_back(weights);
    }

    for (const auto& tensor : model.file_->tensors()) {
        model.total_weight_bytes_ = add_bytes(model.total_weight_bytes_, tensor.byte_size);
    }
    return model;
}

}  // namespace miinfer
