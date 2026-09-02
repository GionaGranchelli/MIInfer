#pragma once

#include "miinfer/gguf.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace miinfer {

struct Qwen35Config {
    std::uint32_t block_count = 0;
    std::uint32_t main_layer_count = 0;
    std::uint32_t nextn_layer_count = 0;
    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    std::uint32_t attention_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t vocab_size = 0;
    std::uint32_t context_length = 0;
    std::uint32_t full_attention_interval = 0;
    std::uint32_t recurrent_inner_size = 0;
    std::uint32_t recurrent_state_size = 0;
    std::uint32_t recurrent_group_count = 0;
    std::uint32_t recurrent_conv_kernel = 0;
    std::uint32_t recurrent_time_step_rank = 0;
    float rope_theta = 0.0F;
    float rms_epsilon = 0.0F;
};

struct Qwen35TensorView {
    const GgufTensor* source = nullptr;

    [[nodiscard]] const std::string& name() const { return source->name; }
    [[nodiscard]] GgufTensorType type() const { return source->type; }
    [[nodiscard]] const std::vector<std::uint64_t>& dimensions() const { return source->dimensions; }
    [[nodiscard]] std::size_t bytes() const { return source->byte_size; }
    [[nodiscard]] const std::byte* data() const { return source->data; }
};

class Qwen35Model {
public:
    static Qwen35Model load(const std::string& path);

    [[nodiscard]] const Qwen35Config& config() const noexcept { return config_; }
    [[nodiscard]] const std::string& artifact_path() const noexcept { return artifact_path_; }
    [[nodiscard]] const std::string& model_name() const noexcept { return model_name_; }
    [[nodiscard]] const std::vector<GgufTensor>& tensors() const noexcept { return file_->tensors(); }
    [[nodiscard]] Qwen35TensorView tensor(const std::string& name) const;
    [[nodiscard]] std::shared_ptr<const GgufFile> file() const noexcept { return file_; }

private:
    std::shared_ptr<GgufFile> file_;
    std::string artifact_path_;
    std::string model_name_;
    Qwen35Config config_;
};

}  // namespace miinfer
