#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/model_plan.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) {
        std::cerr << "model GPU test unavailable: " << error << '\n';
        return 1;
    }

    std::vector<std::byte> first(257);
    std::vector<std::byte> second(129);
    for (std::size_t index = 0; index < first.size(); ++index) {
        first[index] = static_cast<std::byte>(index & 0xffU);
    }
    for (std::size_t index = 0; index < second.size(); ++index) {
        second[index] = static_cast<std::byte>((255U - index) & 0xffU);
    }

    miinfer::GpuWeightArena arena(4096);
    arena.upload(0, first.data(), first.size());
    arena.upload(1024, second.data(), second.size());
    std::vector<std::byte> round_trip(first.size());
    std::vector<std::byte> round_trip_second(second.size());
    MIINFER_HIP_CHECK(hipMemcpy(round_trip.data(), arena.data(), round_trip.size(),
                                hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(round_trip_second.data(),
                                static_cast<const std::byte*>(arena.data()) + 1024,
                                round_trip_second.size(), hipMemcpyDeviceToHost));
    if (round_trip != first || round_trip_second != second) {
        std::cerr << "model GPU arena upload verification failed\n";
        return 1;
    }
    std::cout << "model GPU arena test passed on " << device.architecture << '\n';
    return 0;
}
