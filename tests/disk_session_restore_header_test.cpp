#include "disk_session_restore_header.hpp"

#include <iostream>
#include <string>
#include <string_view>

int main() {
    using miinfer::server::parse_disk_session_restore_header;
    const auto absent = parse_disk_session_restore_header("POST /v1/chat/completions HTTP/1.1\r\n");
    if (!absent.valid || absent.required) return 1;

    const auto disabled = parse_disk_session_restore_header(
        "POST / HTTP/1.1\r\nX-MIInfer-Require-Disk-Session-Restore: 0\r\n");
    if (!disabled.valid || disabled.required) return 1;

    const auto enabled = parse_disk_session_restore_header(
        "POST / HTTP/1.1\r\nx-miinfer-require-disk-session-restore:  1 \t\r\n");
    if (!enabled.valid || !enabled.required) return 1;

    for (const std::string_view value : {"true", "2", ""}) {
        const std::string header = "POST / HTTP/1.1\r\nX-MIInfer-Require-Disk-Session-Restore: "
                                 + std::string(value) + "\r\n";
        if (parse_disk_session_restore_header(header).valid) return 1;
    }

    std::cout << "disk session restore request header: PASS\n";
    return 0;
}
