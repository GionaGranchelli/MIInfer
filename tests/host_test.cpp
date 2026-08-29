#include "miinfer/build_info.hpp"

#include "miinfer/build_config.hpp"

#include <iostream>
#include <sstream>
#include <string>

int main() {
    if (2 + 2 != 4 || std::string(MIINFER_TARGET_ARCH) != "gfx906") {
        std::cerr << "host-only checks failed\n";
        return 1;
    }

    std::ostringstream build_info;
    miinfer::print_build_info(build_info);
    if (build_info.str().find("MIInfer version:") == std::string::npos) {
        std::cerr << "build information is incomplete\n";
        return 1;
    }
    std::cout << "host-only checks passed\n";
    return 0;
}
