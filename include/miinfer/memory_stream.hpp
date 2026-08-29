#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>

namespace miinfer {

// DIAGNOSTIC ONLY: a simple contiguous device copy for an empirical streaming
// reference. It is not a performance candidate or an HBM peak measurement.
void launch_diagnostic_stream_copy(
    const void* source,
    void* destination,
    std::size_t bytes,
    hipStream_t stream = nullptr);

}  // namespace miinfer
