#pragma once

#include <stdexcept>
#include <string_view>

namespace miinfer::prefill_v2 {

inline bool mmq_gateup_only_from_env(const char* value) {
    if (value == nullptr || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY must be 0 or 1");
}

} // namespace miinfer::prefill_v2
