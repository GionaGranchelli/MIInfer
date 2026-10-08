#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

int main(int argc, char** argv) {
    if (argc != 2 || std::strcmp(argv[1], "--confirm-mi50-guard-test") != 0
        || std::getenv("M31_GPU_RUN_AUTHORIZED") == nullptr
        || std::strcmp(std::getenv("M31_GPU_RUN_AUTHORIZED"), "1") != 0) {
        std::fprintf(stderr,
                     "refusing GPU allocation; pass --confirm-mi50-guard-test and set "
                     "M31_GPU_RUN_AUTHORIZED=1\n");
        return 2;
    }

    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) return 3;
    int gfx906 = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, device) != hipSuccess) return 4;
        if (std::strncmp(properties.gcnArchName, "gfx906", 6) == 0) {
            if (gfx906 != -1) {
                std::fprintf(stderr, "refusing ambiguous device set: multiple gfx906 devices\n");
                return 5;
            }
            gfx906 = device;
        }
    }
    if (gfx906 < 0 || hipSetDevice(gfx906) != hipSuccess) {
        std::fprintf(stderr, "refusing GPU allocation: no selectable gfx906 device\n");
        return 6;
    }

    constexpr std::size_t kBytes = 4U * 1024U * 1024U;
    void* allocation = nullptr;
    if (hipMalloc(&allocation, kBytes) != hipSuccess) return 7;
    if (hipMemset(allocation, 0, kBytes) != hipSuccess
        || hipDeviceSynchronize() != hipSuccess) {
        (void)hipFree(allocation);
        return 8;
    }

    std::printf("M31_GUARD_GPU_WORKER_READY device=%d arch=gfx906 allocated_bytes=%zu\n",
                gfx906, kBytes);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(30));
    return hipFree(allocation) == hipSuccess ? 0 : 9;
}
