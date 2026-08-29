#include "miinfer/build_info.hpp"

#include "miinfer/build_config.hpp"

#include <ostream>

namespace miinfer {

void print_build_info(std::ostream& output) {
    output << "MIInfer version: " << MIINFER_VERSION << '\n'
           << "Git commit: " << MIINFER_GIT_COMMIT << '\n'
           << "Git dirty: " << MIINFER_GIT_DIRTY << '\n'
           << "Build type: " << MIINFER_BUILD_TYPE << '\n'
           << "Compiler: " << MIINFER_CXX_COMPILER << '\n'
           << "HIP compiler: " << MIINFER_HIP_COMPILER << '\n'
           << "HIP availability: " << MIINFER_HIP_AVAILABLE << '\n'
           << "Target architecture: " << MIINFER_TARGET_ARCH << '\n';
}

}  // namespace miinfer
