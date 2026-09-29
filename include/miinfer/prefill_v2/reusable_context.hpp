#pragma once

#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/prefill_v2/state.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace miinfer::prefill_v2 {

// Version identifier for the GPU state memory layout
constexpr std::uint32_t kStateLayoutVersion = 20260925;

// Deterministic fingerprint uniquely identifying the prefix tokens and model compatibility.
struct PrefixFingerprint {
    std::string model_id;
    std::string quantization;
    std::uint32_t prefix_length = 0;
    std::uint64_t token_hash = 0;
    std::uint32_t state_layout_version = kStateLayoutVersion;

    [[nodiscard]] bool matches(const PrefixFingerprint& other) const noexcept {
        return prefix_length == other.prefix_length &&
               token_hash == other.token_hash &&
               state_layout_version == other.state_layout_version &&
               model_id == other.model_id &&
               quantization == other.quantization;
    }
};

// Compute 64-bit FNV-1a hash of token span
inline std::uint64_t compute_token_sequence_hash(std::span<const std::uint32_t> tokens) noexcept {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (std::uint32_t tok : tokens) {
        hash ^= static_cast<std::uint64_t>(tok);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// Checkpoint storage for 48 GDN SSM layers on GPU.
// Each GDN layer has:
//   - Recurrent state: 48 * 128 * 128 * sizeof(float) = 3,145,728 bytes (3.0 MiB)
//   - Conv history: 4 * 10240 * sizeof(float) = 163,840 bytes (160 KiB)
// Total per GDN layer = 3,309,568 bytes (3.15625 MiB)
// Across 48 GDN layers: 48 * 3,309,568 = 158,859,264 bytes (151.50 MiB)
class GdnCheckpointStorage {
public:
    GdnCheckpointStorage();
    ~GdnCheckpointStorage();

    GdnCheckpointStorage(const GdnCheckpointStorage&) = delete;
    GdnCheckpointStorage& operator=(const GdnCheckpointStorage&) = delete;
    GdnCheckpointStorage(GdnCheckpointStorage&& other) noexcept;
    GdnCheckpointStorage& operator=(GdnCheckpointStorage&& other) noexcept;

    // Capture GDN states from the 48 active layers into this contiguous checkpoint buffer
    void capture(const std::vector<RecurrentLayerStateStorage>& active_states, hipStream_t stream = nullptr);

    // Restore GDN states from this checkpoint buffer into the 48 active layers
    void restore(std::vector<RecurrentLayerStateStorage>& active_states, hipStream_t stream = nullptr) const;

    [[nodiscard]] std::size_t total_bytes() const noexcept { return total_bytes_; }
    [[nodiscard]] bool is_valid() const noexcept { return is_valid_; }
    void invalidate() noexcept { is_valid_ = false; }

private:
    float* d_checkpoint_states_ = nullptr;        // 48 * kStateElements floats
    float* d_checkpoint_conv_history_ = nullptr; // 48 * kConvHistoryElements floats
    std::size_t total_bytes_ = 0;
    bool is_valid_ = false;
};

// ReusableContext represents a valid prefix checkpoint that can be attached to or owned by PrefillV2Model.
class ReusableContext {
public:
    ReusableContext(std::string model_id = "Qwen3.8-27B", std::string quantization = "Q4_K_M");
    ~ReusableContext() = default;

    ReusableContext(const ReusableContext&) = delete;
    ReusableContext& operator=(const ReusableContext&) = delete;
    ReusableContext(ReusableContext&&) noexcept = default;
    ReusableContext& operator=(ReusableContext&&) noexcept = default;

    // Save prefix state
    void save(
        std::span<const std::uint32_t> prefix_tokens,
        const std::vector<RecurrentLayerStateStorage>& active_states,
        hipStream_t stream = nullptr);

    // Match evaluation results
    enum class MatchResult {
        ExactMatch,
        PrefixMismatch,
        ModelMismatch,
        QuantizationMismatch,
        EmptyCache,
        PromptShorterThanPrefix,
        VersionMismatch,
    };

    [[nodiscard]] MatchResult check_match(
        std::span<const std::uint32_t> full_prompt,
        const std::string& model_id,
        const std::string& quantization) const;

    // Restore GDN states to active model
    void restore_gdn_states(std::vector<RecurrentLayerStateStorage>& active_states, hipStream_t stream = nullptr) const;

    // Reset / clear cache
    void clear();

    [[nodiscard]] bool has_valid_prefix() const noexcept { return gdn_checkpoint_.is_valid() && fingerprint_.prefix_length > 0; }
    [[nodiscard]] std::uint32_t prefix_length() const noexcept { return fingerprint_.prefix_length; }
    [[nodiscard]] const PrefixFingerprint& fingerprint() const noexcept { return fingerprint_; }
    [[nodiscard]] std::size_t memory_bytes() const noexcept { return gdn_checkpoint_.total_bytes(); }
    [[nodiscard]] const std::vector<std::uint32_t>& cached_prefix_tokens() const noexcept { return cached_tokens_; }

private:
    PrefixFingerprint fingerprint_;
    std::vector<std::uint32_t> cached_tokens_;
    GdnCheckpointStorage gdn_checkpoint_;
};

} // namespace miinfer::prefill_v2
