#pragma once

#include "miinfer/sha256.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace m31_token_ids {

struct Artifact {
    std::string model;
    std::string model_sha256;
    std::string workload_id;
    std::size_t sequence_index = 0;
    std::string request_sha256;
    std::string token_ids_sha256;
    std::vector<std::uint32_t> ids;
};

inline std::string ids_sha256(const std::vector<std::uint32_t>& ids) {
    std::vector<std::byte> bytes;
    bytes.reserve(ids.size() * sizeof(std::uint32_t));
    for (const auto id : ids) {
        bytes.push_back(static_cast<std::byte>(id & 0xff));
        bytes.push_back(static_cast<std::byte>((id >> 8) & 0xff));
        bytes.push_back(static_cast<std::byte>((id >> 16) & 0xff));
        bytes.push_back(static_cast<std::byte>((id >> 24) & 0xff));
    }
    return miinfer::sha256_bytes(bytes);
}

inline Artifact parse(const nlohmann::json& value) {
    if (!value.is_object() || value.value("schema", std::string{}) != "miinfer-chatml-token-ids-v2"
        || !value.contains("model") || !value["model"].is_string()
        || !value.contains("model_sha256") || !value["model_sha256"].is_string()
        || !value.contains("workload_id") || !value["workload_id"].is_string()
        || !value.contains("sequence_index") || !value["sequence_index"].is_number_integer()
        || !value.contains("request_sha256") || !value["request_sha256"].is_string()
        || !value.contains("token_count") || !value["token_count"].is_number_integer()
        || !value.contains("token_ids_sha256") || !value["token_ids_sha256"].is_string()
        || !value.contains("token_ids") || !value["token_ids"].is_array()) {
        throw std::runtime_error("invalid MIInfer ChatML token-ID artifact schema");
    }
    Artifact result;
    result.model = value["model"].get<std::string>();
    result.model_sha256 = value["model_sha256"].get<std::string>();
    result.workload_id = value["workload_id"].get<std::string>();
    const auto sequence_index = value["sequence_index"].get<std::int64_t>();
    if (sequence_index <= 0) throw std::runtime_error("invalid token-ID workload sequence index");
    result.sequence_index = static_cast<std::size_t>(sequence_index);
    result.request_sha256 = value["request_sha256"].get<std::string>();
    result.token_ids_sha256 = value["token_ids_sha256"].get<std::string>();
    if (result.model.empty() || result.workload_id.empty() || result.model_sha256.size() != 64
        || result.request_sha256.size() != 64) {
        throw std::runtime_error("token-ID artifact is missing model, workload, or request identity");
    }
    for (const auto& item : value["token_ids"]) {
        if (!item.is_number_integer()) throw std::runtime_error("token ID is not an integer");
        const auto id = item.get<std::int64_t>();
        if (id < 0 || static_cast<std::uint64_t>(id) > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("token ID is outside uint32 range");
        }
        result.ids.push_back(static_cast<std::uint32_t>(id));
    }
    if (value["token_count"].get<std::int64_t>() != static_cast<std::int64_t>(result.ids.size())
        || result.ids.empty() || ids_sha256(result.ids) != result.token_ids_sha256) {
        throw std::runtime_error("token-ID artifact count or SHA-256 does not match its contents");
    }
    return result;
}

inline std::size_t adjacent_prefix(const Artifact& first, const Artifact& second,
                                   const std::string& expected_model_sha256) {
    if (first.model != second.model || first.model_sha256 != expected_model_sha256
        || second.model_sha256 != expected_model_sha256 || first.workload_id != second.workload_id
        || second.sequence_index != first.sequence_index + 1) {
        throw std::runtime_error("request artifacts are not adjacent in the same workload and model");
    }
    std::size_t prefix = 0;
    while (prefix < first.ids.size() && prefix < second.ids.size()
           && first.ids[prefix] == second.ids[prefix]) ++prefix;
    if (prefix != first.ids.size() || prefix >= second.ids.size()) {
        throw std::runtime_error("second request does not extend the complete first-request token history");
    }
    return prefix;
}

inline Artifact load(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open token-ID artifact: " + path);
    try {
        return parse(nlohmann::json::parse(input));
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("invalid token-ID artifact JSON: " + std::string(error.what()));
    }
}

inline bool self_check() {
    const std::vector<std::uint32_t> ids{1, 256, 65536};
    const nlohmann::json valid{
        {"schema", "miinfer-chatml-token-ids-v2"}, {"model", "test"},
        {"model_sha256", std::string(64, 'b')}, {"workload_id", "self-check"},
        {"sequence_index", 1},
        {"request_sha256", std::string(64, 'a')}, {"token_count", ids.size()},
        {"token_ids_sha256", ids_sha256(ids)}, {"token_ids", ids},
    };
    const auto parsed = parse(valid);
    if (parsed.ids != ids) return false;
    auto next = valid;
    next["sequence_index"] = 2;
    next["request_sha256"] = std::string(64, 'c');
    next["token_ids"].push_back(7);
    next["token_count"] = ids.size() + 1;
    next["token_ids_sha256"] = ids_sha256({ids[0], ids[1], ids[2], 7});
    if (adjacent_prefix(parsed, parse(next), std::string(64, 'b')) != ids.size()) return false;
    next["sequence_index"] = 3;
    bool rejected = false;
    try { (void)adjacent_prefix(parsed, parse(next), std::string(64, 'b')); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) return false;
    next["sequence_index"] = 2;
    next["model_sha256"] = std::string(64, 'd');
    rejected = false;
    try { (void)adjacent_prefix(parsed, parse(next), std::string(64, 'b')); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) return false;
    next["model_sha256"] = std::string(64, 'b');
    next["sequence_index"] = 2;
    next["token_ids"][1] = 257;
    next["token_ids_sha256"] = ids_sha256({ids[0], 257, ids[2], 7});
    rejected = false;
    try { (void)adjacent_prefix(parsed, parse(next), std::string(64, 'b')); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) return false;
    auto corrupted = valid;
    corrupted["token_ids"][1] = 257;
    try { (void)parse(corrupted); }
    catch (const std::runtime_error&) { return true; }
    return false;
}

}  // namespace m31_token_ids
