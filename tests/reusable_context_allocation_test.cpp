#include "miinfer/prefill_v2/reusable_context.hpp"

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

std::size_t allocations = 0;
std::size_t copies = 0;

void require(bool condition) {
    if (!condition) std::abort();
}

} // namespace

hipError_t hipMalloc(void** pointer, std::size_t bytes) {
    *pointer = std::malloc(bytes);
    if (*pointer == nullptr) return hipErrorOutOfMemory;
    ++allocations;
    return hipSuccess;
}

hipError_t hipFree(void* pointer) {
    std::free(pointer);
    return hipSuccess;
}

hipError_t hipMemset(void* pointer, int value, std::size_t bytes) {
    std::memset(pointer, value, bytes);
    return hipSuccess;
}

hipError_t hipMemcpy(void* destination, const void* source, std::size_t bytes, hipMemcpyKind) {
    std::memcpy(destination, source, bytes);
    ++copies;
    return hipSuccess;
}

hipError_t hipMemcpyAsync(
    void* destination, const void* source, std::size_t bytes, hipMemcpyKind kind, hipStream_t) {
    return hipMemcpy(destination, source, bytes, kind);
}

hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }

const char* hipGetErrorString(hipError_t) { return "mock HIP error"; }

int main() {
    using namespace miinfer::prefill_v2;

    ReusableContext context;
    require(context.memory_bytes() == 0);
    require(allocations == 1); // Only the small final-hidden buffer is eager.

    std::vector<RecurrentLayerStateStorage> states;
    states.reserve(48);
    for (int i = 0; i < 48; ++i) states.emplace_back();
    const std::size_t allocations_before_save = allocations;

    const std::array<std::uint32_t, 2> prefix{11, 29};
    context.save(prefix, states);

    constexpr std::size_t checkpoint_bytes =
        48 * (RecurrentLayerState::kStateBytes + RecurrentLayerState::kConvHistoryBytes);
    require(allocations == allocations_before_save + 2);
    require(context.memory_bytes() == checkpoint_bytes);
    require(context.has_valid_prefix());
    require(copies == 96); // One state and one history copy for each GDN layer.

    context.clear();
    require(!context.has_valid_prefix());
    require(context.memory_bytes() == checkpoint_bytes); // clear invalidates but retains reusable storage.
}
