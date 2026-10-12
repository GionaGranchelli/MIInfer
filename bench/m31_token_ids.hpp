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
    if (!value.is_object() || value.value("schema", std::string{}) != "miinfer-chatml-token-ids-v1"
        || !value.contains("model") || !value["model"].is_string()
        || !value.contains("request_sha256") || !value["request_sha256"].is_string()
        || !value.contains("token_count") || !value["token_count"].is_number_integer()
        || !value.contains("token_ids_sha256") || !value["token_ids_sha256"].is_string()
        || !value.contains("token_ids") || !value["token_ids"].is_array()) {
        throw std::runtime_error("invalid MIInfer ChatML token-ID artifact schema");
    }
    Artifact result;
    result.model = value["model"].get<std::string>();
    result.request_sha256 = value["request_sha256"].get<std::string>();
    result.token_ids_sha256 = value["token_ids_sha256"].get<std::string>();
    if (result.model.empty() || result.request_sha256.size() != 64) {
        throw std::runtime_error("token-ID artifact is missing model or request identity");
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
        {"schema", "miinfer-chatml-token-ids-v1"}, {"model", "test"},
        {"request_sha256", std::string(64, 'a')}, {"token_count", ids.size()},
        {"token_ids_sha256", ids_sha256(ids)}, {"token_ids", ids},
    };
    if (parse(valid).ids != ids) return false;
    auto corrupted = valid;
    corrupted["token_ids"][1] = 257;
    try { (void)parse(corrupted); }
    catch (const std::runtime_error&) { return true; }
    return false;
}

}  // namespace m31_token_ids
