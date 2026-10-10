#include <hip/hip_runtime_api.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

extern "C" hipError_t hipMalloc(void** pointer, std::size_t bytes) {
    using Function = hipError_t (*)(void**, std::size_t);
    static auto next = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "hipMalloc"));
    const hipError_t result = next(pointer, bytes);
    const char* path = std::getenv("MIINFER_HIP_ALLOC_TRACE");
    if (path != nullptr) {
        timespec now{};
        clock_gettime(CLOCK_REALTIME, &now);
        char line[192];
        const int length = std::snprintf(line, sizeof(line),
            "event=HIP_ALLOC api=hipMalloc bytes=%zu result=%d epoch_ms=%lld\n",
            bytes, static_cast<int>(result),
            static_cast<long long>(now.tv_sec) * 1000 + now.tv_nsec / 1000000);
        const int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
            (void)write(fd, line, static_cast<std::size_t>(length));
            close(fd);
        }
    }
    return result;
}
