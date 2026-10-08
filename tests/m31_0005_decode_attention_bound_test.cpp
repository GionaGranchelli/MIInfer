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

}  // namespace

int main() {
    constexpr std::uint32_t kQueryHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kHeadDim = 256;
    constexpr std::uint32_t kCapacity = 4;
    constexpr std::uint32_t kPosition = 1;
    constexpr std::uint32_t kSplits = 4;
    constexpr std::size_t kQueryElements = kQueryHeads * kHeadDim;
    constexpr std::size_t kKvElements = kKvHeads * kCapacity * kHeadDim;
    constexpr std::size_t kWorkspaceBytes =
        kSplits * kQueryHeads * (kHeadDim + 2) * sizeof(float);

    std::vector<float> query(kQueryElements, 0.0F);
    std::vector<float> gate(kQueryElements, 0.0F);
    std::vector<__half> key(kKvElements, __float2half(0.0F));
    std::vector<__half> value(kKvElements, __float2half(0.0F));
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
        for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
            const auto base = (static_cast<std::size_t>(head) * kCapacity) * kHeadDim + dim;
            value[base] = __float2half(1.0F);
            value[base + kHeadDim] = __float2half(3.0F);
            value[base + 2 * kHeadDim] = __float2half(1000.0F);
        }
    }

    DeviceBuffer d_query(query.size() * sizeof(float));
    DeviceBuffer d_gate(gate.size() * sizeof(float));
    DeviceBuffer d_key(key.size() * sizeof(__half));
    DeviceBuffer d_value(value.size() * sizeof(__half));
    DeviceBuffer d_output(kQueryElements * sizeof(float));
    DeviceBuffer d_workspace(kWorkspaceBytes);
    MIINFER_HIP_CHECK(hipMemcpy(d_query.data, query.data(), query.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate.data, gate.data(), gate.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_key.data, key.data(), key.size() * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_value.data, value.data(), value.size() * sizeof(__half), hipMemcpyHostToDevice));

    miinfer::launch_qwen35_splitk_suffix_attention_quant(
        static_cast<const float*>(d_query.data),
        static_cast<const __half*>(d_key.data),
        static_cast<const __half*>(d_value.data),
        nullptr, nullptr, nullptr, nullptr,
        static_cast<const float*>(d_gate.data),
        static_cast<float*>(d_output.data),
        static_cast<float*>(d_workspace.data),
        1, kPosition, kCapacity, kQueryHeads, kKvHeads, kHeadDim,
        1.0F / 16.0F, false, false, kSplits);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> output(kQueryElements);
    MIINFER_HIP_CHECK(hipMemcpy(output.data(), d_output.data,
                                output.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i]) || std::fabs(output[i] - 1.0F) > 1.0e-3F) {
            std::cerr << "future KV position affected decode attention at element " << i
                      << ": expected 1.0, got " << output[i] << '\n';
            return 1;
        }
    }

    std::cout << "M31-0005 causal decode KV bound: PASS\n";
    return 0;
}
