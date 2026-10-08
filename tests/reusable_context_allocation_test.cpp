#include "miinfer/prefill_v2/reusable_context.hpp"

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
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

    const std::array<std::uint32_t, 4> extended_prefix{11, 29, 31, 47};
    context.save_extension(2, std::span(extended_prefix).subspan(2), states);
    require(context.cached_prefix_tokens().size() == extended_prefix.size());
    require(context.fingerprint().token_hash == compute_token_sequence_hash(extended_prefix));
    require(copies == 192); // Only checkpoint copies repeat; token hashing is host-only.

    const std::array<std::uint32_t, 6> twice_extended{11, 29, 31, 47, 61, 73};
    context.save_extension(4, std::span(twice_extended).subspan(4), states);
    require(context.fingerprint().token_hash == compute_token_sequence_hash(twice_extended));
    const std::size_t copies_before_invalid_extension = copies;
    bool rejected_invalid_extension = false;
    try {
        context.save_extension(5, std::span(twice_extended).subspan(5), states);
    } catch (const std::runtime_error&) {
        rejected_invalid_extension = true;
    }
    require(rejected_invalid_extension);
    require(copies == copies_before_invalid_extension);

    const std::array<std::uint32_t, 3> divergent_prefix{11, 29, 48};
    context.save(divergent_prefix, states);
    require(context.cached_prefix_tokens().size() == divergent_prefix.size());
    require(context.fingerprint().token_hash == compute_token_sequence_hash(divergent_prefix));

    context.clear();
    require(!context.has_valid_prefix());
    require(context.memory_bytes() == checkpoint_bytes); // clear invalidates but retains reusable storage.
}
