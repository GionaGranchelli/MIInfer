#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

__global__ void write_pattern(std::uint32_t* ptr, std::size_t words, std::uint32_t value) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < words) ptr[i] = value + static_cast<std::uint32_t>(i);
}

void check(hipError_t status, const char* expression) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(expression) + ": " + hipGetErrorString(status));
    }
}

#define CHECK(expr) check((expr), #expr)

struct Allocation {
    hipMemGenericAllocationHandle_t handle{};
    void* address = nullptr;
    std::size_t bytes = 0;
};

} // namespace

int main() {
    try {
        int device = 0;
        CHECK(hipGetDevice(&device));
        hipDeviceProp_t props{};
        CHECK(hipGetDeviceProperties(&props, device));

        hipMemAllocationProp prop{};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;

        std::size_t granularity = 0;
        CHECK(hipMemGetAllocationGranularity(
            &granularity, &prop, hipMemAllocationGranularityMinimum));
        constexpr std::size_t kReserveBytes = 4ULL * 1024 * 1024 * 1024;
        const std::size_t chunk_bytes = granularity * 16;
        constexpr int kCycles = 50;

        std::cout << "HIP_VMM_FEASIBILITY device=" << props.name
                  << " arch=" << props.gcnArchName
                  << " reserve_bytes=" << kReserveBytes
                  << " granularity_bytes=" << granularity
                  << " chunk_bytes=" << chunk_bytes
                  << " cycles=" << kCycles << '\n';

        double reserve_us = 0.0;
        double map_us = 0.0;
        double unmap_us = 0.0;
        for (int cycle = 0; cycle < kCycles; ++cycle) {
            void* base = nullptr;
            const auto reserve_start = std::chrono::steady_clock::now();
            CHECK(hipMemAddressReserve(&base, kReserveBytes, granularity, nullptr, 0));
            reserve_us += std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - reserve_start).count();

            std::vector<Allocation> allocations(2);
            const auto map_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < allocations.size(); ++i) {
                allocations[i].bytes = chunk_bytes;
                CHECK(hipMemCreate(&allocations[i].handle, chunk_bytes, &prop, 0));
                allocations[i].address = static_cast<std::byte*>(base) + i * chunk_bytes;
                CHECK(hipMemMap(allocations[i].address, chunk_bytes,
                                0, allocations[i].handle, 0));
                hipMemAccessDesc access{};
                access.location = prop.location;
                access.flags = hipMemAccessFlagsProtReadWrite;
                CHECK(hipMemSetAccess(allocations[i].address, chunk_bytes, &access, 1));
            }
            map_us += std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - map_start).count();

            for (std::size_t i = 0; i < allocations.size(); ++i) {
                auto* words = static_cast<std::uint32_t*>(allocations[i].address);
                const auto count = chunk_bytes / sizeof(std::uint32_t);
                hipLaunchKernelGGL(write_pattern, dim3((count + 255) / 256), dim3(256),
                                   0, 0, words, count, 0x1000u + static_cast<std::uint32_t>(i));
                CHECK(hipGetLastError());
            }
            CHECK(hipDeviceSynchronize());
            for (std::size_t i = 0; i < allocations.size(); ++i) {
                const auto* words = static_cast<const std::uint32_t*>(allocations[i].address);
                std::uint32_t first = 0;
                std::uint32_t last = 0;
                CHECK(hipMemcpy(&first, words, sizeof(first), hipMemcpyDeviceToHost));
                CHECK(hipMemcpy(&last, words + chunk_bytes / sizeof(std::uint32_t) - 1,
                                sizeof(last), hipMemcpyDeviceToHost));
                const auto expected_first = 0x1000u + static_cast<std::uint32_t>(i);
                const auto expected_last = expected_first +
                    static_cast<std::uint32_t>(chunk_bytes / sizeof(std::uint32_t) - 1);
                if (first != expected_first || last != expected_last) {
                    throw std::runtime_error("mapped kernel readback mismatch");
                }
            }

            const auto unmap_start = std::chrono::steady_clock::now();
            for (const auto& allocation : allocations) {
                CHECK(hipMemUnmap(allocation.address, allocation.bytes));
                CHECK(hipMemRelease(allocation.handle));
            }
            CHECK(hipMemAddressFree(base, kReserveBytes));
            unmap_us += std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - unmap_start).count();
        }

        std::cout << std::fixed << std::setprecision(2)
                  << "VMM_RESULT HIP_VMM_VIABLE reserve_avg_us=" << reserve_us / kCycles
                  << " map_avg_us=" << map_us / kCycles
                  << " unmap_release_avg_us=" << unmap_us / kCycles << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VMM_RESULT HIP_VMM_NOT_VIABLE error=" << error.what() << '\n';
        return 1;
    }
}
