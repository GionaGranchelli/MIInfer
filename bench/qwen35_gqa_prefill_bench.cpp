#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

template<class T> struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(std::size_t count) { hipMalloc(&ptr, count * sizeof(T)); }
    ~DeviceBuffer() { hipFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
};

float elapsed(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0F;
    hipEventElapsedTime(&ms, start, stop);
    return ms;
}

float time_path(int path, const float* q, const __half* key, const __half* value,
                const float* gate, float* output, std::uint32_t tokens,
                std::uint32_t capacity, int iterations) {
    hipEvent_t start{}, stop{};
    hipEventCreate(&start);
    hipEventCreate(&stop);
    for (int i = 0; i < 2; ++i) {
        if (path == 1) {
            miinfer::launch_qwen35_gqa_tiled_online_attention_batch_f16(
                q, key, value, gate, output, tokens, 0, capacity, 24, 4, 256,
                1.0F / std::sqrt(256.0F));
        } else if (path == 2) {
            miinfer::launch_qwen35_query_tiled_online_attention_batch_f16(
                q, key, value, gate, output, tokens, 0, capacity, 24, 4, 256,
                1.0F / std::sqrt(256.0F));
        } else if (path == 3) {
            miinfer::launch_qwen35_spillfree_query_tiled_attention_v2_batch_f16(
                q, key, value, gate, output, tokens, 0, capacity, 24, 4, 256,
                1.0F / std::sqrt(256.0F));
        } else {
            miinfer::launch_qwen35_tiled_online_attention_batch_f16(
                q, key, value, gate, output, tokens, 0, capacity, 24, 4, 256,
                1.0F / std::sqrt(256.0F));
        }
    }
    hipDeviceSynchronize();
    float total = 0.0F;
    for (int i = 0; i < iterations; ++i) {
        hipEventRecord(start);
        if (path == 1) {
            miinfer::launch_qwen35_gqa_tiled_online_attention_batch_f16(
                q, key, value, gate, output, tokens, 0, capacity, 24, 4, 256,
                1.0F / std::sqrt(256.0F));
        } else if (path == 2) {
            miinfer::launch_qwen35_query_tiled_online_attention_batch_f16(
                q, key, value, gate, output, tokens, 0, capacity, 24, 4, 256,
                1.0F / std::sqrt(256.0F));
        } else if (path == 3) {
            miinfer::launch_qwen35_spillfree_query_tiled_attention_v2_batch_f16(
                q, key, value, gate, output, tokens, 0, capacity, 24, 4, 256,
                1.0F / std::sqrt(256.0F));
        } else {
            miinfer::launch_qwen35_tiled_online_attention_batch_f16(
                q, key, value, gate, output, tokens, 0, capacity, 24, 4, 256,
                1.0F / std::sqrt(256.0F));
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        total += elapsed(start, stop);
    }
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return total / static_cast<float>(iterations);
}

} // namespace

int main() {
    constexpr std::uint32_t kHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kDim = 256;
    constexpr std::uint32_t kCapacity = 16384;
    constexpr std::size_t q_elements = static_cast<std::size_t>(kCapacity) * kHeads * kDim;
    constexpr std::size_t kv_elements = static_cast<std::size_t>(kKvHeads) * kCapacity * kDim;
    std::vector<float> q(q_elements), gate(q_elements);
    std::vector<__half> key(kv_elements), value(kv_elements);
    std::mt19937 generator(7);
    std::uniform_real_distribution<float> distribution(-0.05F, 0.05F);
    for (auto& element : q) element = distribution(generator);
    for (auto& element : gate) element = distribution(generator);
    for (auto& element : key) element = __float2half(distribution(generator));
    for (auto& element : value) element = __float2half(distribution(generator));
    DeviceBuffer<float> d_q(q.size()), d_gate(gate.size()), d_control(q.size()),
        d_candidate(q.size()), d_schedule(q.size());
    DeviceBuffer<__half> d_key(key.size()), d_value(value.size());
    hipMemcpy(d_q.ptr, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_gate.ptr, gate.data(), gate.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_key.ptr, key.data(), key.size() * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_value.ptr, value.data(), value.size() * sizeof(__half), hipMemcpyHostToDevice);
    for (const std::uint32_t tokens : {512U, 2048U, 4096U, 8192U}) {
        const float control = time_path(0, d_q.ptr, d_key.ptr, d_value.ptr, d_gate.ptr,
                                        d_control.ptr, tokens, kCapacity, 5);
        const float rejected = time_path(1, d_q.ptr, d_key.ptr, d_value.ptr, d_gate.ptr,
                                         d_candidate.ptr, tokens, kCapacity, 5);
        const float candidate = time_path(2, d_q.ptr, d_key.ptr, d_value.ptr, d_gate.ptr,
                                          d_candidate.ptr, tokens, kCapacity, 5);
        const float schedule = time_path(3, d_q.ptr, d_key.ptr, d_value.ptr, d_gate.ptr,
                                         d_schedule.ptr, tokens, kCapacity, 5);
        std::vector<float> control_output(static_cast<std::size_t>(tokens) * kHeads * kDim);
        std::vector<float> candidate_output(control_output.size());
        std::vector<float> schedule_output(control_output.size());
        hipMemcpy(control_output.data(), d_control.ptr, control_output.size() * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(candidate_output.data(), d_candidate.ptr, candidate_output.size() * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(schedule_output.data(), d_schedule.ptr, schedule_output.size() * sizeof(float), hipMemcpyDeviceToHost);
        float max_error = 0.0F;
        float schedule_error = 0.0F;
        for (std::size_t i = 0; i < control_output.size(); ++i) {
            max_error = std::max(max_error, std::fabs(control_output[i] - candidate_output[i]));
            schedule_error = std::max(schedule_error, std::fabs(control_output[i] - schedule_output[i]));
        }
        std::cout << "tokens=" << tokens << " control_ms=" << control
                  << " rejected_ms=" << rejected << " candidate_ms=" << candidate
                  << " schedule_ms=" << schedule << " speedup=" << control / candidate
                  << " schedule_speedup=" << control / schedule
                  << " max_abs_error=" << max_error
                  << " schedule_max_abs_error=" << schedule_error << '\n';
    }
}
