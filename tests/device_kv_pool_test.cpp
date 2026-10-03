#include "miinfer/device_kv_pool.hpp"

#include "miinfer/device_validation.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void throws(auto&& function) {
    bool caught = false;
    try { function(); } catch (const std::exception&) { caught = true; }
    require(caught, "expected exception");
}
}

int main() {
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(0, device, error)) {
        std::cerr << "DeviceKvPool GPU test unavailable: " << error << '\n';
        return 1;
    }
    require(device.architecture.rfind("gfx906", 0) == 0, "selected device is not gfx906");
    require(device.total_vram_bytes >= 30ULL * 1024 * 1024 * 1024, "selected device is not 32 GiB class");
    miinfer::DeviceKvPool pool(device.index, 4096);
    require(pool.device() == device.index, "device ownership");
    require(pool.capacity_bytes() == 4096 && pool.committed_bytes() == 0
                && pool.available_bytes() == 4096, "initial accounting");
    throws([&] { (void)pool.allocate(0); });
    throws([&] { (void)pool.allocate(4097); });
    const auto first = pool.allocate(1024, 256);
    const auto second = pool.allocate(2048, 256);
    require(first.device == device.index && first.offset % 256 == 0, "first allocation");
    require(pool.committed_bytes() == 3072 && pool.available_bytes() == 1024, "allocation accounting");
    throws([&] { (void)pool.allocate(1025); });
    pool.release(first);
    require(pool.committed_bytes() == 2048 && pool.available_bytes() == 2048, "release accounting");
    throws([&] { pool.release(first); });
    const auto reused = pool.allocate(512, 256);
    require(reused.offset == first.offset, "released range was not reused");
    pool.release(second);
    pool.release(reused);
    require(pool.committed_bytes() == 0 && pool.available_bytes() == pool.capacity_bytes(), "cleanup accounting");
    throws([&] { pool.release({device.index, 0, 1, nullptr, 999}); });
    std::cout << "DeviceKvPool physical test passed on " << device.architecture << "\n";
    return 0;
}
