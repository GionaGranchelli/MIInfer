#define MIINFER_M6A3_HELPERS_ONLY
#include "m6a3_qwen35_layer.cpp"

#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kQueryElements = kKHeads * kState;
constexpr std::size_t kValueElements = kVHeads * kState;
constexpr std::size_t kHistoryCapacity = 4;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) {
        MIINFER_HIP_CHECK(hipMalloc(&data_, elements * sizeof(T)));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    T* get() const noexcept { return data_; }

private:
    T* data_ = nullptr;
};

void upload(const void* host, void* device, std::size_t bytes) {
    MIINFER_HIP_CHECK(hipMemcpy(device, host, bytes, hipMemcpyHostToDevice));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a19-qwen35-conv-gpu MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const std::filesystem::path fixture = argv[2];
        const auto& conv = tensor(*model.file(), "blk.0.ssm_conv1d.weight");
        if (conv.type != miinfer::GgufTensorType::f32 || conv.byte_size != kChannels * 4 * sizeof(float)) {
            throw std::runtime_error("unexpected qwen35 convolution tensor");
        }

        DeviceBuffer<float> current(kChannels);
        DeviceBuffer<float> weights(kChannels * 4);
        DeviceBuffer<float> history(kHistoryCapacity * kChannels);
        DeviceBuffer<float> query(kQueryElements);
        DeviceBuffer<float> key(kQueryElements);
        DeviceBuffer<float> value(kValueElements);
        DeviceBuffer<float> query_norm(kQueryElements);
        DeviceBuffer<float> key_norm(kQueryElements);
        MIINFER_HIP_CHECK(hipMemset(history.get(), 0,
                                    kHistoryCapacity * kChannels * sizeof(float)));
        upload(conv.data, weights.get(), conv.byte_size);

        for (const std::size_t position : {std::size_t{0}, std::size_t{1}}) {
            const auto raw = read_f32(checkpoint(
                fixture, position, "conv_output_raw-0"), kChannels);
            const auto projected = read_f32(checkpoint(
                fixture, position, "linear_attn_qkv_mixed-0"), kChannels);
            upload(projected.data(), current.get(), projected.size() * sizeof(float));
            miinfer::launch_qwen35_conv_silu_split(
                current.get(), weights.get(), history.get(), query.get(), key.get(), value.get(),
                static_cast<std::uint32_t>(position), kHistoryCapacity,
                kChannels, 4);
            MIINFER_HIP_CHECK(hipDeviceSynchronize());

            std::vector<float> actual(kChannels);
            MIINFER_HIP_CHECK(hipMemcpy(actual.data(), query.get(),
                                        kQueryElements * sizeof(float), hipMemcpyDeviceToHost));
            MIINFER_HIP_CHECK(hipMemcpy(actual.data() + kQueryElements, key.get(),
                                        kQueryElements * sizeof(float), hipMemcpyDeviceToHost));
            MIINFER_HIP_CHECK(hipMemcpy(actual.data() + 2 * kQueryElements, value.get(),
                                        kValueElements * sizeof(float), hipMemcpyDeviceToHost));
            auto expected = raw;
            silu(expected);
            const auto activation_error = compare(actual, expected);
            require_match("GPU convolution/SiLU", activation_error, 2.0e-5F);

            std::vector<float> expected_query(expected.begin(), expected.begin() + kQueryElements);
            std::vector<float> expected_key(expected.begin() + kQueryElements,
                                             expected.begin() + 2 * kQueryElements);
            l2_norm(expected_query, kState);
            l2_norm(expected_key, kState);
            miinfer::launch_qwen35_head_l2_normalize(
                query.get(), query_norm.get(), kKHeads, kState);
            miinfer::launch_qwen35_head_l2_normalize(
                key.get(), key_norm.get(), kKHeads, kState);
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> actual_query(kQueryElements);
            std::vector<float> actual_key(kQueryElements);
            MIINFER_HIP_CHECK(hipMemcpy(actual_query.data(), query_norm.get(),
                                        actual_query.size() * sizeof(float), hipMemcpyDeviceToHost));
            MIINFER_HIP_CHECK(hipMemcpy(actual_key.data(), key_norm.get(),
                                        actual_key.size() * sizeof(float), hipMemcpyDeviceToHost));
            const auto query_error = compare(actual_query, expected_query);
            const auto key_error = compare(actual_key, expected_key);
            require_match("GPU query L2 normalization", query_error, 2.0e-5F);
            require_match("GPU key L2 normalization", key_error, 2.0e-5F);
            std::cout << "position=" << position
                      << " conv_silu_max_abs=" << activation_error.max_abs
                      << " query_l2_max_abs=" << query_error.max_abs
                      << " key_l2_max_abs=" << key_error.max_abs << '\n';
        }
        std::cout << "history_bytes=" << kHistoryCapacity * kChannels * sizeof(float) << '\n'
                  << "M6-A19 qwen35 convolution GPU PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A19 failed: " << error.what() << '\n';
        return 1;
    }
}
