#include <chrono>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <hip/hip_runtime.h>

#include "miinfer/prefill_v2/persistent_session.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/prefill_v2/reusable_context.hpp"
#include "miinfer/hip_check.hpp"

namespace miinfer::prefill_v2 {

namespace {

constexpr std::uint32_t kExpectedGdnLayers = 48;
constexpr std::uint32_t kExpectedGqaLayers = 16;

} // namespace

std::string PersistentSession::format_session_filename(std::uint64_t token_hash, std::uint32_t prefix_length) {
    std::ostringstream oss;
    oss << "sess_" << std::hex << std::setfill('0') << std::setw(16) << token_hash
        << "_p" << std::dec << prefix_length << ".miinfer";
    return oss.str();
}

void PersistentSession::save_to_file(
    const std::string& file_path,
    const PrefillV2Model& model,
    std::span<const std::uint32_t> prefix_tokens,
    hipStream_t stream) {

    if (prefix_tokens.empty()) {
        throw std::runtime_error("PersistentSession::save_to_file: cannot save empty prefix");
    }

    const std::uint32_t P = static_cast<std::uint32_t>(prefix_tokens.size());
    const std::uint64_t hash = compute_token_sequence_hash(prefix_tokens);

    PersistentSessionHeader header{};
    std::strncpy(header.model_id, model.model_name().c_str(), sizeof(header.model_id) - 1);
    std::strncpy(header.quantization, model.quantization().c_str(), sizeof(header.quantization) - 1);
    header.prefix_length = P;
    header.token_hash = hash;
    header.gdn_layers = kExpectedGdnLayers;
    header.gqa_layers = kExpectedGqaLayers;
    header.kv_quant_mode = static_cast<std::uint8_t>(model.kv_quant_mode());
    header.created_timestamp = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    // 1. Calculate and gather GDN bytes
    const std::size_t gdn_layer_bytes = RecurrentLayerState::kStateBytes + RecurrentLayerState::kConvHistoryBytes;
    header.gdn_state_bytes = kExpectedGdnLayers * gdn_layer_bytes;

    std::vector<std::uint8_t> gdn_host_buf(header.gdn_state_bytes);
    std::size_t gdn_offset = 0;
    for (std::size_t i = 0; i < kExpectedGdnLayers; ++i) {
        auto view = model.recurrent_storage(i).view();
        if (stream != nullptr) {
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                gdn_host_buf.data() + gdn_offset,
                view.d_state,
                RecurrentLayerState::kStateBytes,
                hipMemcpyDeviceToHost,
                stream));
            gdn_offset += RecurrentLayerState::kStateBytes;

            MIINFER_HIP_CHECK(hipMemcpyAsync(
                gdn_host_buf.data() + gdn_offset,
                view.d_conv_history,
                RecurrentLayerState::kConvHistoryBytes,
                hipMemcpyDeviceToHost,
                stream));
            gdn_offset += RecurrentLayerState::kConvHistoryBytes;
        } else {
            MIINFER_HIP_CHECK(hipMemcpy(
                gdn_host_buf.data() + gdn_offset,
                view.d_state,
                RecurrentLayerState::kStateBytes,
                hipMemcpyDeviceToHost));
            gdn_offset += RecurrentLayerState::kStateBytes;

            MIINFER_HIP_CHECK(hipMemcpy(
                gdn_host_buf.data() + gdn_offset,
                view.d_conv_history,
                RecurrentLayerState::kConvHistoryBytes,
                hipMemcpyDeviceToHost));
            gdn_offset += RecurrentLayerState::kConvHistoryBytes;
        }
    }

    // 2. Calculate and gather GQA KV cache bytes
    std::size_t single_layer_kv_bytes = model.kv_storage(0).raw_tokens_bytes(P);
    header.kv_bytes = kExpectedGqaLayers * single_layer_kv_bytes;

    std::vector<std::uint8_t> kv_host_buf(header.kv_bytes);
    for (std::size_t j = 0; j < kExpectedGqaLayers; ++j) {
        model.kv_storage(j).download_raw(
            kv_host_buf.data() + j * single_layer_kv_bytes,
            P,
            stream);
    }

    if (stream != nullptr) {
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
    }

    header.tokens_bytes = P * sizeof(std::uint32_t);

    // 3. Write atomic temp file and rename
    std::filesystem::path final_path(file_path);
    if (final_path.has_parent_path()) {
        std::filesystem::create_directories(final_path.parent_path());
    }
    std::filesystem::path temp_path = final_path.string() + ".tmp";

    std::ofstream out(temp_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("PersistentSession::save_to_file: failed to open " + temp_path.string());
    }

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(prefix_tokens.data()), header.tokens_bytes);
    out.write(reinterpret_cast<const char*>(gdn_host_buf.data()), header.gdn_state_bytes);
    out.write(reinterpret_cast<const char*>(kv_host_buf.data()), header.kv_bytes);

    out.flush();
    if (!out.good()) {
        throw std::runtime_error("PersistentSession::save_to_file: write failure to " + temp_path.string());
    }
    out.close();

    std::filesystem::rename(temp_path, final_path);
}

bool PersistentSession::inspect_file(
    const std::string& file_path,
    PersistentSessionHeader& out_header,
    std::vector<std::uint32_t>* out_tokens) {

    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) return false;

    if (!in.read(reinterpret_cast<char*>(&out_header), sizeof(out_header))) return false;

    if (std::memcmp(out_header.magic, "MISS", 4) != 0 || out_header.version != 20260928) {
        return false;
    }

    if (out_tokens != nullptr && out_header.prefix_length > 0) {
        out_tokens->resize(out_header.prefix_length);
        if (!in.read(reinterpret_cast<char*>(out_tokens->data()), out_header.tokens_bytes)) {
            return false;
        }
    }

    return true;
}

bool PersistentSession::load_from_file(
    const std::string& file_path,
    PrefillV2Model& model,
    std::vector<std::uint32_t>& out_prefix_tokens,
    hipStream_t stream) {

    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) return false;

    PersistentSessionHeader header{};
    if (!in.read(reinterpret_cast<char*>(&header), sizeof(header))) return false;

    if (std::memcmp(header.magic, "MISS", 4) != 0 || header.version != 20260928) {
        return false;
    }
    if (std::string_view(header.model_id) != model.model_name() ||
        std::string_view(header.quantization) != model.quantization()) {
        return false;
    }
    if (header.gdn_layers != kExpectedGdnLayers || header.gqa_layers != kExpectedGqaLayers) {
        return false;
    }

    const std::uint32_t P = header.prefix_length;
    out_prefix_tokens.resize(P);
    if (!in.read(reinterpret_cast<char*>(out_prefix_tokens.data()), header.tokens_bytes)) {
        return false;
    }

    // Verify token hash
    if (compute_token_sequence_hash(out_prefix_tokens) != header.token_hash) {
        return false;
    }

    // 1. Load GDN states
    std::vector<std::uint8_t> gdn_host_buf(header.gdn_state_bytes);
    if (!in.read(reinterpret_cast<char*>(gdn_host_buf.data()), header.gdn_state_bytes)) {
        return false;
    }

    std::size_t gdn_offset = 0;
    for (std::size_t i = 0; i < kExpectedGdnLayers; ++i) {
        auto view = model.recurrent_storage(i).view();
        if (stream != nullptr) {
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                view.d_state,
                gdn_host_buf.data() + gdn_offset,
                RecurrentLayerState::kStateBytes,
                hipMemcpyHostToDevice,
                stream));
            gdn_offset += RecurrentLayerState::kStateBytes;

            MIINFER_HIP_CHECK(hipMemcpyAsync(
                view.d_conv_history,
                gdn_host_buf.data() + gdn_offset,
                RecurrentLayerState::kConvHistoryBytes,
                hipMemcpyHostToDevice,
                stream));
            gdn_offset += RecurrentLayerState::kConvHistoryBytes;
        } else {
            MIINFER_HIP_CHECK(hipMemcpy(
                view.d_state,
                gdn_host_buf.data() + gdn_offset,
                RecurrentLayerState::kStateBytes,
                hipMemcpyHostToDevice));
            gdn_offset += RecurrentLayerState::kStateBytes;

            MIINFER_HIP_CHECK(hipMemcpy(
                view.d_conv_history,
                gdn_host_buf.data() + gdn_offset,
                RecurrentLayerState::kConvHistoryBytes,
                hipMemcpyHostToDevice));
            gdn_offset += RecurrentLayerState::kConvHistoryBytes;
        }
        model.recurrent_storage(i).set_position(P);
    }

    // 2. Load GQA KV cache
    std::vector<std::uint8_t> kv_host_buf(header.kv_bytes);
    if (!in.read(reinterpret_cast<char*>(kv_host_buf.data()), header.kv_bytes)) {
        return false;
    }

    const std::size_t single_layer_kv_bytes = header.kv_bytes / kExpectedGqaLayers;
    for (std::size_t j = 0; j < kExpectedGqaLayers; ++j) {
        model.kv_storage(j).upload_raw(
            kv_host_buf.data() + j * single_layer_kv_bytes,
            P,
            stream);
    }

    if (stream != nullptr) {
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
    }

    // Also update model's in-memory reusable_context_
    // The legacy session format has no boundary hidden payload; keep the
    // restored state usable for nonzero suffixes, but not for zero-suffix hits.
    model.reusable_context().save(out_prefix_tokens, model.recurrent_states(), nullptr, stream);

    return true;
}

bool PersistentSession::find_matching_session(
    const std::string& session_dir,
    const std::string& model_id,
    const std::string& quantization,
    std::span<const std::uint32_t> full_prompt,
    std::string& out_session_path,
    std::uint32_t& out_prefix_length) {

    if (!std::filesystem::exists(session_dir) || !std::filesystem::is_directory(session_dir)) {
        return false;
    }

    std::uint32_t best_prefix_len = 0;
    std::string best_session_path;

    for (const auto& entry : std::filesystem::directory_iterator(session_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".miinfer") continue;

        PersistentSessionHeader hdr{};
        std::vector<std::uint32_t> sess_tokens;
        if (!inspect_file(entry.path().string(), hdr, &sess_tokens)) continue;

        if (std::string_view(hdr.model_id) != model_id ||
            std::string_view(hdr.quantization) != quantization) {
            continue;
        }

        if (hdr.prefix_length == 0 || hdr.prefix_length > full_prompt.size()) continue;

        // Check if full_prompt has matching prefix
        bool match = true;
        for (std::size_t i = 0; i < hdr.prefix_length; ++i) {
            if (full_prompt[i] != sess_tokens[i]) {
                match = false;
                break;
            }
        }

        if (match && hdr.prefix_length > best_prefix_len) {
            best_prefix_len = hdr.prefix_length;
            best_session_path = entry.path().string();
        }
    }

    if (best_prefix_len > 0) {
        out_session_path = best_session_path;
        out_prefix_length = best_prefix_len;
        return true;
    }

    return false;
}

} // namespace miinfer::prefill_v2
