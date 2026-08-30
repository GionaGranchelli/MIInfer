#pragma once

#include "miinfer/gguf.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace miinfer {

struct Qwen3Config {
    std::uint32_t layer_count = 0;
    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    std::uint32_t attention_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t vocab_size = 0;
    std::uint32_t context_length = 0;
    float rope_theta = 0.0F;
    float rms_epsilon = 0.0F;
};

struct Qwen3TensorView {
    const GgufTensor* source = nullptr;

    [[nodiscard]] const std::string& name() const { return source->name; }
    [[nodiscard]] GgufTensorType type() const { return source->type; }
    [[nodiscard]] const std::vector<std::uint64_t>& dimensions() const { return source->dimensions; }
    [[nodiscard]] std::size_t bytes() const { return source->byte_size; }
    [[nodiscard]] const std::byte* data() const { return source->data; }
};

struct Qwen3LayerWeights {
    Qwen3TensorView attention_norm;
    Qwen3TensorView q;
    Qwen3TensorView k;
    Qwen3TensorView v;
    Qwen3TensorView output;
    Qwen3TensorView q_norm;
    Qwen3TensorView k_norm;
    Qwen3TensorView ffn_norm;
    Qwen3TensorView gate;
    Qwen3TensorView up;
    Qwen3TensorView down;
};

class Qwen3Model {
public:
    static Qwen3Model load(const std::string& path);

    [[nodiscard]] const Qwen3Config& config() const noexcept { return config_; }
    [[nodiscard]] const std::string& artifact_path() const noexcept { return artifact_path_; }
    [[nodiscard]] const std::string& model_name() const noexcept { return model_name_; }
    [[nodiscard]] const std::vector<GgufTensor>& tensors() const noexcept { return file_->tensors(); }
    [[nodiscard]] const std::vector<Qwen3LayerWeights>& layers() const noexcept { return layers_; }
    [[nodiscard]] const Qwen3TensorView& token_embeddings() const noexcept { return token_embeddings_; }
    [[nodiscard]] const Qwen3TensorView& final_norm() const noexcept { return final_norm_; }
    [[nodiscard]] const Qwen3TensorView& output() const noexcept { return output_; }
    [[nodiscard]] std::size_t total_weight_bytes() const noexcept { return total_weight_bytes_; }
    [[nodiscard]] std::shared_ptr<const GgufFile> file() const noexcept { return file_; }

private:
    std::shared_ptr<GgufFile> file_;
    std::string artifact_path_;
    std::string model_name_;
    Qwen3Config config_;
    std::vector<Qwen3LayerWeights> layers_;
    Qwen3TensorView token_embeddings_;
    Qwen3TensorView final_norm_;
    Qwen3TensorView output_;
    std::size_t total_weight_bytes_ = 0;
};

}  // namespace miinfer
