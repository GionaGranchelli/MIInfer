#define MIINFER_M6A3_HELPERS_ONLY
#include "m6a3_qwen35_layer.cpp"

#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
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

std::vector<float> transpose_state(std::span<const float> logical) {
    const std::size_t matrix = kStateSize * kStateSize;
    if (logical.size() % matrix != 0) throw std::runtime_error("invalid state size");
    std::vector<float> transposed(logical.size());
    const std::size_t heads = logical.size() / matrix;
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t row = 0; row < kStateSize; ++row) {
            for (std::size_t column = 0; column < kStateSize; ++column) {
                transposed[head * matrix + column * kStateSize + row] =
                    logical[head * matrix + row * kStateSize + column];
            }
        }
    }
    return transposed;
}

std::vector<float> logical_state(std::span<const float> transposed) {
    const std::size_t matrix = kStateSize * kStateSize;
    if (transposed.size() % matrix != 0) throw std::runtime_error("invalid state size");
    std::vector<float> logical(transposed.size());
    const std::size_t heads = transposed.size() / matrix;
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t row = 0; row < kStateSize; ++row) {
            for (std::size_t column = 0; column < kStateSize; ++column) {
                logical[head * matrix + row * kStateSize + column] =
                    transposed[head * matrix + column * kStateSize + row];
            }
        }
    }
    return logical;
}

float timed_state_update(std::size_t iterations, bool transposed,
                         const float* query, const float* key, const float* value,
                         const float* beta, const float* decay, float* state,
                         float* output) {
    hipEvent_t start = nullptr;
    hipEvent_t end = nullptr;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&end));
    MIINFER_HIP_CHECK(hipEventRecord(start));
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        if (transposed) {
            miinfer::launch_qwen35_deltanet_state_update_transposed(
                query, key, value, beta, decay, state, output,
                kKeyHeads, kValueHeads, kStateSize);
        } else {
            miinfer::launch_qwen35_deltanet_state_update_no_decay_store(
                query, key, value, beta, decay, state, output,
                kKeyHeads, kValueHeads, kStateSize);
        }
    }
    MIINFER_HIP_CHECK(hipEventRecord(end));
    MIINFER_HIP_CHECK(hipEventSynchronize(end));
    float elapsed = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&elapsed, start, end));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(end));
    return elapsed * 1000.0F / static_cast<float>(iterations);
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
        DeviceBuffer<float> transposed_state(state_elements);
        DeviceBuffer<float> output(output_elements);
        DeviceBuffer<float> transposed_output(output_elements);
        upload(state_initial.data(), state.get(), state_initial.size() * sizeof(float));
        const auto state_initial_transposed = transpose_state(state_initial);
        upload(state_initial_transposed.data(), transposed_state.get(),
               state_initial_transposed.size() * sizeof(float));

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
            miinfer::launch_qwen35_deltanet_state_update_transposed(
                query.get(), key.get(), value.get(), beta.get(), decay.get(),
                transposed_state.get(), transposed_output.get(),
                kKeyHeads, kValueHeads, kStateSize);
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
            std::vector<float> actual_transposed_state(state_elements);
            MIINFER_HIP_CHECK(hipMemcpy(actual_transposed_state.data(), transposed_state.get(),
                                        actual_transposed_state.size() * sizeof(float),
                                        hipMemcpyDeviceToHost));
            const auto transposed_state_error = compare(
                logical_state(actual_transposed_state), expected_state);
            std::vector<float> actual_transposed_output(output_elements);
            MIINFER_HIP_CHECK(hipMemcpy(actual_transposed_output.data(), transposed_output.get(),
                                        actual_transposed_output.size() * sizeof(float),
                                        hipMemcpyDeviceToHost));
            const auto transposed_output_error = compare(actual_transposed_output, expected_output);
            require_match("GPU recurrent output", output_error, 1.0e-3F);
            require_match("GPU recurrent state", state_error, 1.0e-2F);
            require_match("transposed recurrent output", transposed_output_error, 1.0e-3F);
            require_match("transposed recurrent state", transposed_state_error, 1.0e-2F);
            std::cout << "position=" << position
                      << " output_max_abs=" << output_error.max_abs
                      << " state_max_abs=" << state_error.max_abs
                      << " transposed_output_max_abs=" << transposed_output_error.max_abs
                      << " transposed_state_max_abs=" << transposed_state_error.max_abs << '\n';
        }
        const std::size_t timing_iterations = 100;
        const auto timing_state = state_initial;
        const auto timing_transposed_state = state_initial_transposed;
        upload(timing_state.data(), state.get(), timing_state.size() * sizeof(float));
        upload(timing_transposed_state.data(), transposed_state.get(),
               timing_transposed_state.size() * sizeof(float));
        const float control_us = timed_state_update(
            timing_iterations, false, query.get(), key.get(), value.get(), beta.get(), decay.get(),
            state.get(), output.get());
        upload(timing_transposed_state.data(), transposed_state.get(),
               timing_transposed_state.size() * sizeof(float));
        const float candidate_us = timed_state_update(
            timing_iterations, true, query.get(), key.get(), value.get(), beta.get(), decay.get(),
            transposed_state.get(), transposed_output.get());
        std::cout << "state_update_control_us=" << control_us
                  << " state_update_transposed_us=" << candidate_us
                  << " change_pct=" << ((control_us - candidate_us) / control_us * 100.0F) << '\n';
        std::cout << "state_elements=" << state_elements
                  << " persistent_state_bytes=" << state_elements * sizeof(float) << '\n'
                  << "M6-A18 qwen35 DeltaNet GPU state core PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A18 failed: " << error.what() << '\n';
        return 1;
    }
}
