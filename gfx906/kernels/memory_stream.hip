#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace miinfer {

namespace {

__global__ void diagnostic_stream_copy_kernel(
    const unsigned char* source,
    unsigned char* destination,
    std::size_t bytes) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t offset = index; offset < bytes; offset += stride) {
        destination[offset] = source[offset];
    }
}

}  // namespace

void launch_diagnostic_stream_copy(
    const void* source,
    void* destination,
    std::size_t bytes,
    hipStream_t stream) {
    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>(
        (bytes + threads - 1) / threads > 4096 ? 4096 : (bytes + threads - 1) / threads);
    hipLaunchKernelGGL(
        diagnostic_stream_copy_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        stream,
        static_cast<const unsigned char*>(source),
        static_cast<unsigned char*>(destination),
        bytes);
    MIINFER_HIP_CHECK(hipGetLastError());
}

}  // namespace miinfer
