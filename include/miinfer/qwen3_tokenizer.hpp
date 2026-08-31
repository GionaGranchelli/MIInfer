#pragma once

#include "miinfer/qwen3_model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace miinfer {

// Minimal, model-owned Qwen2 byte-level BPE tokenizer.  It intentionally
// supports only the tokenizer contract embedded in the pinned Qwen3 GGUF;
// other tokenizer families fail during construction instead of silently
// producing different token IDs.
class Qwen3Tokenizer {
public:
    static Qwen3Tokenizer load(const Qwen3Model& model);

    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(std::span<const std::uint32_t> tokens) const;

    [[nodiscard]] std::uint32_t bos_id() const noexcept { return bos_id_; }
    [[nodiscard]] std::uint32_t eos_id() const noexcept { return eos_id_; }
    [[nodiscard]] bool add_bos() const noexcept { return add_bos_; }
    [[nodiscard]] bool add_eos() const noexcept { return add_eos_; }
    [[nodiscard]] std::size_t vocabulary_size() const noexcept { return tokens_.size(); }

private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, std::uint32_t> token_to_id_;
    std::unordered_map<std::string, std::uint32_t> merge_ranks_;
    std::uint32_t bos_id_ = 0;
    std::uint32_t eos_id_ = 0;
    bool add_bos_ = false;
    bool add_eos_ = false;
};

}  // namespace miinfer
