#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct DeviceBuffer {
    void* data = nullptr;

    explicit DeviceBuffer(std::size_t bytes) {
        MIINFER_HIP_CHECK(hipMalloc(&data, bytes));
    }

    ~DeviceBuffer() {
        if (data != nullptr) (void)hipFree(data);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

bool all_equal(const std::vector<float>& values, float expected) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i]) || std::fabs(values[i] - expected) > 1.0e-3F) {
            std::cerr << "causal decode KV bound failed at element " << i
                      << ": expected " << expected << ", got " << values[i] << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    constexpr std::uint32_t kQueryHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kHeadDim = 256;
    constexpr std::uint32_t kCapacity = 514;
    constexpr std::uint32_t kSplits = 4;
    constexpr std::size_t kQueryElements = kQueryHeads * kHeadDim;
    constexpr std::size_t kKvElements = kKvHeads * kCapacity * kHeadDim;
    constexpr std::size_t kWorkspaceBytes =
        kSplits * kQueryHeads * (kHeadDim + 2) * sizeof(float);

    std::vector<float> query(kQueryElements, 0.0F);
    std::vector<float> gate(kQueryElements, 20.0F);
    std::vector<__half> key(kKvElements, __float2half(0.0F));
    std::vector<__half> value(kKvElements, __float2half(1.0F));

    DeviceBuffer d_query(query.size() * sizeof(float));
    DeviceBuffer d_gate(gate.size() * sizeof(float));
    DeviceBuffer d_key(key.size() * sizeof(__half));
    DeviceBuffer d_value(value.size() * sizeof(__half));
    DeviceBuffer d_output(kQueryElements * sizeof(float));
    DeviceBuffer d_graph_output(kQueryElements * sizeof(float));
    DeviceBuffer d_workspace(kWorkspaceBytes);
    DeviceBuffer d_decode_state(sizeof(miinfer::DeviceDecodeState));
    MIINFER_HIP_CHECK(hipMemcpy(d_query.data, query.data(), query.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate.data, gate.data(), gate.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_key.data, key.data(), key.size() * sizeof(__half), hipMemcpyHostToDevice));
    hipStream_t stream = nullptr;
    MIINFER_HIP_CHECK(hipStreamCreate(&stream));
    for (const std::uint32_t position : {1U, 511U, 512U}) {
        const std::uint32_t future_position = position + 1;
        for (std::uint32_t head = 0; head < kKvHeads; ++head) {
            const auto base = (static_cast<std::size_t>(head) * kCapacity + future_position) * kHeadDim;
            std::fill(value.begin() + base, value.begin() + base + kHeadDim, __float2half(1000.0F));
        }
        MIINFER_HIP_CHECK(hipMemcpy(d_value.data, value.data(), value.size() * sizeof(__half), hipMemcpyHostToDevice));
        const miinfer::DeviceDecodeState state{0, position, 0, 0, 0};
        MIINFER_HIP_CHECK(hipMemcpy(d_decode_state.data, &state, sizeof(state), hipMemcpyHostToDevice));

        miinfer::launch_qwen35_splitk_suffix_attention_quant(
            static_cast<const float*>(d_query.data),
            static_cast<const __half*>(d_key.data),
            static_cast<const __half*>(d_value.data),
            nullptr, nullptr, nullptr, nullptr,
            static_cast<const float*>(d_gate.data),
            static_cast<float*>(d_output.data),
            static_cast<float*>(d_workspace.data),
            1, position, kCapacity, kQueryHeads, kKvHeads, kHeadDim,
            1.0F / 16.0F, false, false, kSplits, stream);
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<float> eager_output(kQueryElements);
        MIINFER_HIP_CHECK(hipMemcpy(eager_output.data(), d_output.data,
                                    eager_output.size() * sizeof(float), hipMemcpyDeviceToHost));
        if (!all_equal(eager_output, 1.0F)) return 1;

        hipGraph_t graph = nullptr;
        MIINFER_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
        miinfer::launch_qwen35_tiled_online_attention_quant_dynamic(
            static_cast<const float*>(d_query.data),
            static_cast<const __half*>(d_key.data),
            static_cast<const __half*>(d_value.data),
            nullptr, nullptr, nullptr, nullptr,
            static_cast<const miinfer::DeviceDecodeState*>(d_decode_state.data),
            kCapacity, nullptr, static_cast<const float*>(d_gate.data),
            static_cast<float*>(d_graph_output.data),
            kQueryHeads, kKvHeads, kHeadDim, 1.0F / 16.0F, false, false, stream);
        MIINFER_HIP_CHECK(hipStreamEndCapture(stream, &graph));
        hipGraphExec_t executable = nullptr;
        MIINFER_HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        MIINFER_HIP_CHECK(hipGraphDestroy(graph));
        MIINFER_HIP_CHECK(hipGraphLaunch(executable, stream));
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<float> graph_output(kQueryElements);
        MIINFER_HIP_CHECK(hipMemcpy(graph_output.data(), d_graph_output.data,
                                    graph_output.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipGraphExecDestroy(executable));
        if (!all_equal(graph_output, 1.0F)) return 1;
        for (std::size_t i = 0; i < graph_output.size(); ++i) {
            if (std::fabs(graph_output[i] - eager_output[i]) > 1.0e-3F) {
                std::cerr << "graph/eager mismatch at position " << position
                          << ", element " << i << '\n';
                return 1;
            }
        }
        for (std::uint32_t head = 0; head < kKvHeads; ++head) {
            const auto base = (static_cast<std::size_t>(head) * kCapacity + future_position) * kHeadDim;
            std::fill(value.begin() + base, value.begin() + base + kHeadDim, __float2half(1.0F));
        }
        std::cout << "M31-0005 decode graph/eager boundary position=" << position << ": PASS\n";
    }
    MIINFER_HIP_CHECK(hipStreamDestroy(stream));
    std::cout << "M31-0005 causal decode KV bound: PASS\n";
    return 0;
}
