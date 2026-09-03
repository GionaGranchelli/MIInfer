#define MIINFER_M6A3_HELPERS_ONLY
#include "m6a3_qwen35_layer.cpp"

#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kKeyHeads = 16;
constexpr std::size_t kValueHeads = 48;
constexpr std::size_t kStateSize = 128;

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

std::vector<float> recurrent_inputs(const miinfer::GgufFile& model,
                                    const std::filesystem::path& fixture,
                                    std::size_t position,
                                    std::vector<float>& key,
                                    std::vector<float>& value,
                                    std::vector<float>& beta,
                                    std::vector<float>& decay) {
    const auto input = read_f32(checkpoint(fixture, position, "attn_norm-0"), kHidden);
    const auto raw = read_f32(checkpoint(fixture, position, "conv_output_raw-0"), kChannels);
    std::vector<float> activated = raw;
    silu(activated);
    std::vector<float> query(activated.begin(), activated.begin() + kKeyHeads * kStateSize);
    key.assign(activated.begin() + kKeyHeads * kStateSize,
               activated.begin() + 2 * kKeyHeads * kStateSize);
    value.assign(activated.begin() + 2 * kKeyHeads * kStateSize, activated.end());
    l2_norm(query, kStateSize);
    l2_norm(key, kStateSize);

    const auto beta_raw = f32_gemv(tensor(model, "blk.0.ssm_beta.weight"), input,
                                   kValueHeads, kHidden);
    const auto alpha_raw = f32_gemv(tensor(model, "blk.0.ssm_alpha.weight"), input,
                                    kValueHeads, kHidden);
    const auto dt = f32_values(tensor(model, "blk.0.ssm_dt.bias"), kValueHeads);
    const auto a = f32_values(tensor(model, "blk.0.ssm_a"), kValueHeads);
    beta.resize(kValueHeads);
    decay.resize(kValueHeads);
    for (std::size_t head = 0; head < kValueHeads; ++head) {
        beta[head] = 1.0F / (1.0F + std::exp(-beta_raw[head]));
        const float x = alpha_raw[head] + dt[head];
        const float softplus = x <= 20.0F ? std::log1p(std::exp(x)) : x;
        decay[head] = std::exp(softplus * a[head]);
    }
    return query;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a18-qwen35-deltanet-state-gpu MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const std::filesystem::path fixture = argv[2];
        const std::size_t state_elements = kValueHeads * kStateSize * kStateSize;
        const std::size_t output_elements = kValueHeads * kStateSize;
        const auto state_initial = read_f32(
            checkpoint(fixture, 0, "state_predelta-0"), state_elements);

        DeviceBuffer<float> query(kKeyHeads * kStateSize);
        DeviceBuffer<float> key(kKeyHeads * kStateSize);
        DeviceBuffer<float> value(kValueHeads * kStateSize);
        DeviceBuffer<float> beta(kValueHeads);
        DeviceBuffer<float> decay(kValueHeads);
        DeviceBuffer<float> state(state_elements);
        DeviceBuffer<float> output(output_elements);
        upload(state_initial.data(), state.get(), state_initial.size() * sizeof(float));

        for (const std::size_t position : {std::size_t{0}, std::size_t{1}}) {
            std::vector<float> host_key;
            std::vector<float> host_value;
            std::vector<float> host_beta;
            std::vector<float> host_decay;
            const auto host_query = recurrent_inputs(
                *model.file(), fixture, position, host_key, host_value,
                host_beta, host_decay);
            upload(host_query.data(), query.get(), host_query.size() * sizeof(float));
            upload(host_key.data(), key.get(), host_key.size() * sizeof(float));
            upload(host_value.data(), value.get(), host_value.size() * sizeof(float));
            upload(host_beta.data(), beta.get(), host_beta.size() * sizeof(float));
            upload(host_decay.data(), decay.get(), host_decay.size() * sizeof(float));

            miinfer::launch_qwen35_deltanet_state_update(
                query.get(), key.get(), value.get(), beta.get(), decay.get(),
                state.get(), output.get(), kKeyHeads, kValueHeads, kStateSize);
            MIINFER_HIP_CHECK(hipDeviceSynchronize());

            std::vector<float> actual_output(output_elements);
            std::vector<float> actual_state(state_elements);
            MIINFER_HIP_CHECK(hipMemcpy(actual_output.data(), output.get(),
                                        actual_output.size() * sizeof(float),
                                        hipMemcpyDeviceToHost));
            MIINFER_HIP_CHECK(hipMemcpy(actual_state.data(), state.get(),
                                        actual_state.size() * sizeof(float),
                                        hipMemcpyDeviceToHost));
            const auto expected_output = read_f32(
                checkpoint(fixture, position, "attn_output-0"), output_elements);
            const auto expected_state = read_f32(
                checkpoint(fixture, position + 1, "state_predelta-0"), state_elements);
            const auto output_error = compare(actual_output, expected_output);
            const auto state_error = compare(actual_state, expected_state);
            require_match("GPU recurrent output", output_error, 1.0e-3F);
            require_match("GPU recurrent state", state_error, 1.0e-2F);
            std::cout << "position=" << position
                      << " output_max_abs=" << output_error.max_abs
                      << " state_max_abs=" << state_error.max_abs << '\n';
        }
        std::cout << "state_elements=" << state_elements
                  << " persistent_state_bytes=" << state_elements * sizeof(float) << '\n'
                  << "M6-A18 qwen35 DeltaNet GPU state core PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A18 failed: " << error.what() << '\n';
        return 1;
    }
}
