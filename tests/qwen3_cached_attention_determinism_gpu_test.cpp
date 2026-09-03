#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        MIINFER_HIP_CHECK(hipMalloc(&data_, bytes));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }

    void* get() const noexcept { return data_; }

private:
    void* data_ = nullptr;
};

template <typename T>
void upload(const std::vector<T>& source, DeviceBuffer& destination) {
    MIINFER_HIP_CHECK(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                                hipMemcpyHostToDevice));
}

template <typename T>
std::vector<T> download(const DeviceBuffer& source, std::size_t count) {
    std::vector<T> result(count);
    MIINFER_HIP_CHECK(hipMemcpy(result.data(), source.get(), count * sizeof(T),
                                hipMemcpyDeviceToHost));
    return result;
}

bool same_bytes(const std::vector<float>& left, const std::vector<float>& right) {
    return left.size() == right.size()
        && std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

}  // namespace

int main() {
    try {
        constexpr std::size_t query_heads = 24;
        constexpr std::size_t kv_heads = 4;
        constexpr std::size_t head_dim = 256;
        constexpr std::size_t cache_length = 32;
        constexpr std::size_t cache_capacity = 64;
        const std::size_t query_count = query_heads * head_dim;
        const std::size_t cache_count = kv_heads * cache_capacity * head_dim;
        const std::size_t score_count = query_heads * cache_length;

        std::vector<float> query(query_count);
        std::vector<float> keys(cache_count);
        std::vector<float> values(cache_count);
        for (std::size_t index = 0; index < query.size(); ++index) {
            query[index] = std::sin(static_cast<float>(index) * 0.017F) * 0.25F;
        }
        for (std::size_t index = 0; index < keys.size(); ++index) {
            keys[index] = std::cos(static_cast<float>(index) * 0.013F) * 0.5F;
            values[index] = std::sin(static_cast<float>(index) * 0.011F) * 0.75F;
        }

        DeviceBuffer query_device(query.size() * sizeof(float));
        DeviceBuffer key_device(keys.size() * sizeof(float));
        DeviceBuffer value_device(values.size() * sizeof(float));
        DeviceBuffer output_device(query.size() * sizeof(float));
        DeviceBuffer scores_device(score_count * sizeof(float));
        DeviceBuffer probabilities_device(score_count * sizeof(float));
        upload(query, query_device);
        upload(keys, key_device);
        upload(values, value_device);

        std::vector<float> expected_output;
        std::vector<float> expected_scores;
        std::vector<float> expected_probabilities;
        for (int repeat = 0; repeat < 16; ++repeat) {
            MIINFER_HIP_CHECK(hipMemset(output_device.get(), 0xCD, query.size() * sizeof(float)));
            MIINFER_HIP_CHECK(hipMemset(scores_device.get(), 0xCD, score_count * sizeof(float)));
            MIINFER_HIP_CHECK(hipMemset(probabilities_device.get(), 0xCD,
                                        score_count * sizeof(float)));
            miinfer::launch_qwen3_cached_attention_parallel(
                static_cast<const float*>(query_device.get()),
                static_cast<const float*>(key_device.get()),
                static_cast<const float*>(value_device.get()), cache_length, cache_capacity,
                static_cast<float*>(output_device.get()), static_cast<float*>(scores_device.get()),
                static_cast<float*>(probabilities_device.get()), query_heads, kv_heads,
                head_dim, 1.0F / std::sqrt(static_cast<float>(head_dim)));
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            const auto output = download<float>(output_device, query_count);
            const auto scores = download<float>(scores_device, score_count);
            const auto probabilities = download<float>(probabilities_device, score_count);
            if (repeat == 0) {
                expected_output = output;
                expected_scores = scores;
                expected_probabilities = probabilities;
            } else if (!same_bytes(output, expected_output)
                       || !same_bytes(scores, expected_scores)
                       || !same_bytes(probabilities, expected_probabilities)) {
                throw std::runtime_error("cached attention repeated execution is not byte-identical");
            }
        }
        std::cout << "qwen3 cached attention determinism: PASS (16/16 exact replays)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen3 cached attention determinism: FAIL: " << error.what() << '\n';
        return 1;
    }
}
