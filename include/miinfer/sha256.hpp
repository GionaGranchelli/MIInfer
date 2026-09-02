#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace miinfer {

[[nodiscard]] std::string sha256_file(const std::string& path);

[[nodiscard]] std::string sha256_bytes(std::span<const std::byte> bytes);

}  // namespace miinfer
