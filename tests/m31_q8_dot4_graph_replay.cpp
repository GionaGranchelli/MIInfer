#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

namespace {

template <typename T>
T* allocate_device(std::size_t count) {
    T* ptr = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}

bool compare(const float* actual, const float* expected, std::size_t count,
             double& max_abs, double& cosine) {
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    max_abs = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) return false;
        max_abs = std::max(max_abs, std::abs(static_cast<double>(actual[i]) - expected[i]));
        dot += static_cast<double>(actual[i]) * expected[i];
        actual_norm += static_cast<double>(actual[i]) * actual[i];
        expected_norm += static_cast<double>(expected[i]) * expected[i];
    }
    cosine = dot / (std::sqrt(actual_norm * expected_norm) + 1e-30);
    return max_abs <= 0.01 && cosine >= 0.9999;
}

}  // namespace

int main() {
    constexpr std::uint32_t kQueries = 32;
    constexpr std::uint32_t kHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kDim = 256;
    constexpr std::uint32_t kCapacity = 256;
    constexpr std::uint32_t kSplits = 3;
    constexpr std::size_t q_count = static_cast<std::size_t>(kQueries) * kHeads * kDim;
    constexpr std::size_t kv_count = static_cast<std::size_t>(kKvHeads) * kCapacity * kDim;
    constexpr std::size_t scale_count = static_cast<std::size_t>(kKvHeads) * kCapacity;
    constexpr std::size_t workspace_count =
        static_cast<std::size_t>(kSplits) * kQueries * kHeads * (kDim + 2);

    miinfer::DeviceInfo device{};
    std::string error;
    if (!miinfer::validate_gfx906_device(0, device, error)) {
        std::cerr << "device validation failed: " << error << '\n';
        return 1;
    }

    std::vector<float> host_q(q_count), host_gate(q_count);
    std::vector<std::int8_t> host_k(kv_count), host_v(kv_count);
    std::vector<__half> host_ks(scale_count, __float2half(0.01F));
    std::vector<__half> host_vs(scale_count, __float2half(0.01F));
    for (std::size_t i = 0; i < q_count; ++i) {
        host_q[i] = std::sin(static_cast<float>(i) * 0.003F);
        host_gate[i] = std::cos(static_cast<float>(i) * 0.002F);
    }
    for (std::size_t i = 0; i < kv_count; ++i) {
        host_k[i] = static_cast<std::int8_t>(static_cast<int>(i % 127) - 63);
        host_v[i] = static_cast<std::int8_t>(static_cast<int>((i * 17) % 127) - 63);
    }

    float* d_q = allocate_device<float>(q_count);
    float* d_gate = allocate_device<float>(q_count);
    float* d_graph_output = allocate_device<float>(q_count);
    float* d_direct_output = allocate_device<float>(q_count);
    std::int8_t* d_query_q8 = allocate_device<std::int8_t>(q_count);
    float* d_query_scales = allocate_device<float>(static_cast<std::size_t>(kQueries) * kHeads);
    std::int8_t* d_k = allocate_device<std::int8_t>(kv_count);
    std::int8_t* d_v = allocate_device<std::int8_t>(kv_count);
    __half* d_ks = allocate_device<__half>(scale_count);
    __half* d_vs = allocate_device<__half>(scale_count);
    float* d_workspace = allocate_device<float>(workspace_count);
    auto* d_state = allocate_device<miinfer::DevicePrefillState>(1);

    MIINFER_HIP_CHECK(hipMemcpy(d_q, host_q.data(), q_count * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate, host_gate.data(), q_count * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_k, host_k.data(), kv_count * sizeof(std::int8_t), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_v, host_v.data(), kv_count * sizeof(std::int8_t), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_ks, host_ks.data(), scale_count * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_vs, host_vs.data(), scale_count * sizeof(__half), hipMemcpyHostToDevice));

    const auto stream = hipStreamPerThread;
    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    MIINFER_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
    miinfer::launch_qwen35_kq_fragment_reuse_attention_batch_q8_dot4(
        d_q, d_query_q8, d_query_scales, d_k, d_ks, d_v, d_vs, d_gate,
        d_graph_output, d_workspace, kQueries, 0, kCapacity, kHeads, kKvHeads,
        kDim, 1.0F / 16.0F, kSplits, stream, d_state);
    MIINFER_HIP_CHECK(hipStreamEndCapture(stream, &graph));
    MIINFER_HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    std::vector<float> graph_output(q_count), direct_output(q_count);
    bool passed = true;
    for (const std::uint32_t base_position : {0U, 64U}) {
        const miinfer::DevicePrefillState state{
            base_position, kQueries, base_position + kQueries, kCapacity};
        MIINFER_HIP_CHECK(hipMemcpyAsync(d_state, &state, sizeof(state), hipMemcpyHostToDevice, stream));
        MIINFER_HIP_CHECK(hipGraphLaunch(graph_exec, stream));
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
        miinfer::launch_qwen35_kq_fragment_reuse_attention_batch_q8_dot4(
            d_q, d_query_q8, d_query_scales, d_k, d_ks, d_v, d_vs, d_gate,
            d_direct_output, d_workspace, kQueries, base_position, kCapacity,
            kHeads, kKvHeads, kDim, 1.0F / 16.0F, kSplits, stream, nullptr);
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));
        MIINFER_HIP_CHECK(hipMemcpy(graph_output.data(), d_graph_output,
                                    q_count * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(direct_output.data(), d_direct_output,
                                    q_count * sizeof(float), hipMemcpyDeviceToHost));
        double max_abs = 0.0;
        double cosine = 0.0;
        const bool current_pass = compare(graph_output.data(), direct_output.data(),
                                          q_count, max_abs, cosine);
        std::cout << "graph_replay_base=" << base_position
                  << " max_abs=" << max_abs << " cosine=" << cosine
                  << " pass=" << current_pass << '\n';
        passed = passed && current_pass;
    }

    MIINFER_HIP_CHECK(hipGraphExecDestroy(graph_exec));
    MIINFER_HIP_CHECK(hipGraphDestroy(graph));
    MIINFER_HIP_CHECK(hipFree(d_q));
    MIINFER_HIP_CHECK(hipFree(d_gate));
    MIINFER_HIP_CHECK(hipFree(d_graph_output));
    MIINFER_HIP_CHECK(hipFree(d_direct_output));
    MIINFER_HIP_CHECK(hipFree(d_query_q8));
    MIINFER_HIP_CHECK(hipFree(d_query_scales));
    MIINFER_HIP_CHECK(hipFree(d_k));
    MIINFER_HIP_CHECK(hipFree(d_v));
    MIINFER_HIP_CHECK(hipFree(d_ks));
    MIINFER_HIP_CHECK(hipFree(d_vs));
    MIINFER_HIP_CHECK(hipFree(d_workspace));
    MIINFER_HIP_CHECK(hipFree(d_state));
    return passed ? 0 : 1;
}
