#pragma once

#include <cctype>
#include <optional>
#include <string>
#include <string_view>

namespace miinfer::server {

struct DiskSessionRestoreHeader {
    bool valid = true;
    bool required = false;
};

inline std::string_view trim_header_value(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return value;
}

inline bool header_name_equals(std::string_view name, std::string_view expected) {
    if (name.size() != expected.size()) return false;
    for (std::size_t i = 0; i < name.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(name[i]))
            != std::tolower(static_cast<unsigned char>(expected[i]))) return false;
    }
    return true;
}

inline DiskSessionRestoreHeader parse_disk_session_restore_header(std::string_view headers) {
    constexpr std::string_view name = "x-miinfer-require-disk-session-restore";
    std::size_t line_start = 0;
    while (line_start < headers.size()) {
        const std::size_t separator = headers.find("\r\n", line_start);
        const std::size_t line_end = separator == std::string_view::npos ? headers.size() : separator;
        const std::string_view line = headers.substr(line_start, line_end - line_start);
        const std::size_t colon = line.find(':');
        if (colon != std::string_view::npos
            && header_name_equals(trim_header_value(line.substr(0, colon)), name)) {
            const std::string_view value = trim_header_value(line.substr(colon + 1));
            if (value == "0") return {true, false};
            if (value == "1") return {true, true};
            return {false, false};
        }
        if (separator == std::string_view::npos) break;
        line_start = separator + 2;
    }
    return {};
}

}  // namespace miinfer::server
