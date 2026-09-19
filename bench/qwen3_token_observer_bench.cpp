#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kToken = 42;

hipGraphExec_t capture_observer_graph(miinfer::DeviceDecodeState* state,
                                      std::uint32_t* output, std::uint32_t tokens,
                                      hipStream_t stream) {
    hipGraph_t graph = nullptr;
    MIINFER_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
    miinfer::launch_qwen35_decode_state_advance(state, output, tokens, stream);
    MIINFER_HIP_CHECK(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t executable = nullptr;
    MIINFER_HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    MIINFER_HIP_CHECK(hipGraphDestroy(graph));
    return executable;
}

double observe(hipGraphExec_t graph, miinfer::DeviceDecodeState* device_state,
               const miinfer::DeviceDecodeState& initial_state,
               std::uint32_t* device_ring, std::uint32_t* mapped_ring,
               std::uint32_t* pinned_host_ring, std::uint32_t tokens,
               bool async_copy, hipStream_t stream) {
    MIINFER_HIP_CHECK(hipMemcpyAsync(device_state, &initial_state, sizeof(initial_state),
                                     hipMemcpyHostToDevice, stream));
    MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));
    MIINFER_HIP_CHECK(hipEventRecord(start, stream));
    for (std::uint32_t i = 0; i < tokens; ++i) {
        MIINFER_HIP_CHECK(hipGraphLaunch(graph, stream));
        if (async_copy) {
            MIINFER_HIP_CHECK(hipMemcpyAsync(pinned_host_ring + i, device_ring + i,
                                             sizeof(std::uint32_t), hipMemcpyDeviceToHost,
                                             stream));
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
        const std::uint32_t observed = async_copy ? pinned_host_ring[i] : mapped_ring[i];
        if (observed != kToken) {
            std::cerr << "observer token mismatch at " << i << ": " << observed << '\n';
            return -1.0;
        }
    }
    MIINFER_HIP_CHECK(hipEventRecord(stop, stream));
    MIINFER_HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    MIINFER_HIP_CHECK(hipEventDestroy(stop));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
    miinfer::DeviceDecodeState final_state{};
    MIINFER_HIP_CHECK(hipMemcpy(&final_state, device_state, sizeof(final_state),
                                hipMemcpyDeviceToHost));
    if (final_state.generated != tokens) {
        std::cerr << "state advance mismatch: expected " << tokens
                  << " generated, got " << final_state.generated << '\n';
        return -1.0;
    }
    return elapsed_us;
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const auto middle = samples.size() / 2;
    return samples.size() % 2 == 0
        ? (samples[middle - 1] + samples[middle]) / 2.0 : samples[middle];
}

bool positive_int(const char* text, int& result) {
    try { result = std::stoi(text); } catch (...) { return false; }
    return result > 0;
}

} // namespace

int main(int argc, char** argv) {
    std::uint32_t tokens = 128;
    int iterations = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help") {
            std::cout << "usage: miinfer-qwen3-token-observer-bench [--tokens N] [--iterations N]\n";
            return 0;
        }
        if (i + 1 >= argc) return 2;
        int value = 0;
        if (!positive_int(argv[++i], value)) return 2;
        if (arg == "--tokens" && value <= 4096) tokens = static_cast<std::uint32_t>(value);
        else if (arg == "--iterations") iterations = value;
        else return 2;
    }

    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) {
        std::cerr << "token observer benchmark unavailable: " << error << '\n';
        return 1;
    }
    MIINFER_HIP_CHECK(hipSetDevice(device.index));

    hipStream_t stream = nullptr;
    miinfer::DeviceDecodeState* device_state = nullptr;
    std::uint32_t* device_ring = nullptr;
    std::uint32_t* mapped_host_ring = nullptr;
    std::uint32_t* mapped_device_ring = nullptr;
    std::uint32_t* pinned_host_ring = nullptr;
    MIINFER_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    MIINFER_HIP_CHECK(hipMalloc(&device_state, sizeof(*device_state)));
    MIINFER_HIP_CHECK(hipMalloc(&device_ring, tokens * sizeof(*device_ring)));
    MIINFER_HIP_CHECK(hipHostMalloc(&pinned_host_ring, tokens * sizeof(*pinned_host_ring)));
    MIINFER_HIP_CHECK(hipHostMalloc(&mapped_host_ring, tokens * sizeof(*mapped_host_ring),
                                    hipHostMallocMapped));
    MIINFER_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&mapped_device_ring),
                                              mapped_host_ring, 0));

    const auto device_graph = capture_observer_graph(device_state, device_ring, tokens, stream);
    const auto mapped_graph = capture_observer_graph(device_state, mapped_device_ring, tokens, stream);
    const miinfer::DeviceDecodeState initial_state{.current_token = kToken,
        .position = 0, .generated = 0, .stop = 0, .max_generated = tokens};
    std::vector<double> copy_samples, mapped_samples;
    copy_samples.reserve(static_cast<std::size_t>(iterations));
    mapped_samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < 3; ++i) {
        (void)observe(device_graph, device_state, initial_state, device_ring,
                      mapped_host_ring, pinned_host_ring, tokens, true, stream);
        (void)observe(mapped_graph, device_state, initial_state, device_ring,
                      mapped_host_ring, pinned_host_ring, tokens, false, stream);
    }
    for (int i = 0; i < iterations; ++i) {
        const bool mapped_first = (i & 1) != 0;
        const double first = mapped_first
            ? observe(mapped_graph, device_state, initial_state, device_ring,
                      mapped_host_ring, pinned_host_ring, tokens, false, stream)
            : observe(device_graph, device_state, initial_state, device_ring,
                      mapped_host_ring, pinned_host_ring, tokens, true, stream);
        const double second = mapped_first
            ? observe(device_graph, device_state, initial_state, device_ring,
                      mapped_host_ring, pinned_host_ring, tokens, true, stream)
            : observe(mapped_graph, device_state, initial_state, device_ring,
                      mapped_host_ring, pinned_host_ring, tokens, false, stream);
        if (first < 0.0 || second < 0.0) return 1;
        if (mapped_first) {
            mapped_samples.push_back(first);
            copy_samples.push_back(second);
        } else {
            copy_samples.push_back(first);
            mapped_samples.push_back(second);
        }
    }

    const double copy_median = median(copy_samples);
    const double mapped_median = median(mapped_samples);
    std::cout << std::fixed << std::setprecision(3)
              << "gfx=" << device.architecture << " gpu=\"" << device.name << "\"\n"
              << "tokens=" << tokens << " iterations=" << iterations
              << " observer=async-d2h-pinned median_us=" << copy_median
              << " us_per_token=" << copy_median / tokens << '\n'
              << "tokens=" << tokens << " iterations=" << iterations
              << " observer=mapped-host-visible median_us=" << mapped_median
              << " us_per_token=" << mapped_median / tokens << '\n';

    MIINFER_HIP_CHECK(hipGraphExecDestroy(mapped_graph));
    MIINFER_HIP_CHECK(hipGraphExecDestroy(device_graph));
    MIINFER_HIP_CHECK(hipHostFree(mapped_host_ring));
    MIINFER_HIP_CHECK(hipHostFree(pinned_host_ring));
    MIINFER_HIP_CHECK(hipFree(device_ring));
    MIINFER_HIP_CHECK(hipFree(device_state));
    MIINFER_HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
