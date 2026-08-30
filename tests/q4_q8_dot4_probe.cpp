#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/q4_q8_packed_dot.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <iostream>
#include <string>

int main() {
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) {
        std::cerr << "dot4 probe unavailable: " << error << '\n';
        return 1;
    }
    int* device_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), sizeof(int)));
    // The byte lanes are signed int8. These cases cover centered Q4 values,
    // Q4 extrema, Q8 extrema, zero, and a mixed-sign known dot product.
    struct Case {
        int lhs;
        int rhs;
        int expected;
    };
    const std::array<Case, 5> cases = {{
        {0x00000000, 0x00000000, 0},
        {static_cast<int>(0xF8F8F8F8U), 0x7F7F7F7F, -4064},
        {0x07070707, static_cast<int>(0x81818181U), -3556},
        {static_cast<int>(0xFC03FE01U), 0x08F906FB, -70},
        {0x07070707, 0x01020304, 70},
    }};
    bool passed = true;
    for (const auto& test_case : cases) {
        miinfer::launch_q4_q8_dot4_probe(
            test_case.lhs, test_case.rhs, device_output);
        MIINFER_HIP_CHECK(hipDeviceSynchronize());
        int actual = 0;
        MIINFER_HIP_CHECK(hipMemcpy(&actual, device_output, sizeof(actual), hipMemcpyDeviceToHost));
        std::cout << "dot4_i8 result=" << actual << " expected=" << test_case.expected << '\n';
        passed = actual == test_case.expected && passed;
    }
    MIINFER_HIP_CHECK(hipFree(device_output));
    return passed ? 0 : 1;
}
