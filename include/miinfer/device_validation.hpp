#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>

namespace miinfer {

struct DeviceInfo {
    int index = -1;
    std::string name;
    std::string architecture;
    std::size_t total_vram_bytes = 0;
};

bool validate_gfx906_device(int requested_index, DeviceInfo& device, std::string& error);
void print_device_info(const DeviceInfo& device, std::ostream& output);

}  // namespace miinfer
