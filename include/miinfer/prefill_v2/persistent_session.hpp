#pragma once

#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/prefill_v2/kv_cache.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace miinfer::prefill_v2 {

class PrefillV2Model;

#pragma pack(push, 1)
struct PersistentSessionHeader {
    char magic[4] = {'M', 'I', 'S', 'S'}; // MIInfer Session State
    std::uint32_t version = 20260928;
    char model_id[64] = {};
    char quantization[32] = {};
    std::uint32_t prefix_length = 0;
    std::uint64_t token_hash = 0;
    std::uint32_t gdn_layers = 48;
    std::uint64_t gdn_state_bytes = 0;
    std::uint32_t gqa_layers = 16;
    std::uint8_t kv_quant_mode = 0;
    std::uint64_t kv_bytes = 0;
    std::uint64_t tokens_bytes = 0;
    std::uint64_t created_timestamp = 0;
};
#pragma pack(pop)

class PersistentSession {
public:
    // Generate a deterministic session file name from token hash and length
    static std::string format_session_filename(std::uint64_t token_hash, std::uint32_t prefix_length);

    // Save active model state (48 GDN layers + 16 GQA KV caches) to disk
    static void save_to_file(
        const std::string& file_path,
        const PrefillV2Model& model,
        std::span<const std::uint32_t> prefix_tokens,
        hipStream_t stream = nullptr);

    // Inspect session header on disk without loading full payload
    static bool inspect_file(
        const std::string& file_path,
        PersistentSessionHeader& out_header,
        std::vector<std::uint32_t>* out_tokens = nullptr);

    // Load session state from disk into active model (both GDN and KV cache)
    static bool load_from_file(
        const std::string& file_path,
        PrefillV2Model& model,
        std::vector<std::uint32_t>& out_prefix_tokens,
        hipStream_t stream = nullptr);

    // Search directory for a session matching the prefix of full_prompt
    static bool find_matching_session(
        const std::string& session_dir,
        const std::string& model_id,
        const std::string& quantization,
        std::span<const std::uint32_t> full_prompt,
        std::string& out_session_path,
        std::uint32_t& out_prefix_length);
};

} // namespace miinfer::prefill_v2
