#include "miinfer/build_info.hpp"
#include "miinfer/device_validation.hpp"
#include "miinfer/vector_add.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        miinfer::print_build_info(std::cout);
        return 0;
    }
    if (argc != 1) {
        std::cerr << "usage: miinfer-hip-smoke [--version]\n";
        return 2;
    }

    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) {
        std::cerr << "MIInfer HIP smoke test unavailable: " << error << '\n';
        return 1;
    }
    if (!miinfer::run_vector_add(1U << 20, 1.25F, 1.0e-6F)) {
        std::cerr << "MIInfer HIP smoke test correctness check failed\n";
        return 1;
    }
    std::cout << "HIP smoke test passed on " << device.name << " (" << device.architecture
              << ")\n";
    return 0;
}
