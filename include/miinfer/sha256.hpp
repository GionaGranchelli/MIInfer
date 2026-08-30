#pragma once

#include <string>

namespace miinfer {

[[nodiscard]] std::string sha256_file(const std::string& path);

}  // namespace miinfer
