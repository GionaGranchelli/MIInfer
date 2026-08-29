#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <source_location>

namespace miinfer {

[[noreturn]] inline void hip_check_failed(
    hipError_t error,
    const char* expression,
    const std::source_location location = std::source_location::current()) {
    std::fprintf(
        stderr,
        "MIInfer HIP failure: %s\n  expression: %s\n  location: %s:%u\n",
        hipGetErrorString(error),
        expression,
        location.file_name(),
        location.line());
    std::fflush(stderr);
    std::abort();
}

inline void hip_check(
    hipError_t error,
    const char* expression,
    const std::source_location location = std::source_location::current()) {
    if (error != hipSuccess) {
        hip_check_failed(error, expression, location);
    }
}

}  // namespace miinfer

#define MIINFER_HIP_CHECK(expression) \
    ::miinfer::hip_check((expression), #expression, std::source_location::current())
