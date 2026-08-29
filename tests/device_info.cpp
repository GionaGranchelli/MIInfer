#include "miinfer/build_info.hpp"
#include "miinfer/device_validation.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    int requested_device = -1;
    if (argc == 2 && std::string(argv[1]) == "--version") {
        miinfer::print_build_info(std::cout);
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--device") {
        requested_device = std::atoi(argv[2]);
    } else if (argc != 1) {
        std::cerr << "usage: miinfer-device-info [--device INDEX] [--version]\n";
        return 2;
    }

    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(requested_device, device, error)) {
        std::cerr << "MIInfer device validation failed: " << error << '\n';
        return 1;
    }
    miinfer::print_build_info(std::cout);
    miinfer::print_device_info(device, std::cout);
    return 0;
}
