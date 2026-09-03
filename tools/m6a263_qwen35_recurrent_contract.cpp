#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kKeyHeads = 16;
constexpr std::size_t kValueHeads = 48;
constexpr std::size_t kState = 128;
constexpr std::size_t kStateElements = kValueHeads * kState * kState;

std::filesystem::path checkpoint(const std::filesystem::path& root, std::size_t position,
                                 std::string_view name) {
    const auto prefix = std::to_string(position) + "-" + std::string(name) + "-";
    for (const auto& entry : std::filesystem::directory_iterator(root / "tensors")) {
        if (entry.path().filename().string().starts_with(prefix)) return entry.path();
    }
    throw std::runtime_error("missing checkpoint: " + prefix);
}

std::vector<float> read_f32(const std::filesystem::path& path, std::size_t elements) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    std::vector<float> values(elements);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(elements * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(elements * sizeof(float))) {
        throw std::runtime_error("short checkpoint: " + path.string());
    }
    return values;
}

struct Metrics {
    float max_abs = 0.0F;
    double squared = 0.0;
};

Metrics compare(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("comparison size mismatch");
    Metrics result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        result.max_abs = std::max(result.max_abs, error);
        result.squared += static_cast<double>(error) * error;
    }
    return result;
}

void apply_external(std::vector<float>& state, std::span<const float> key,
                    std::span<const float> value, std::span<const float> beta,
                    std::span<const float> decay) {
    for (std::size_t head = 0; head < kValueHeads; ++head) {
        const float* key_values = key.data() + (head % kKeyHeads) * kState;
        for (std::size_t row = 0; row < kState; ++row) {
            float* state_row = state.data() + (head * kState + row) * kState;
            float key_dot = 0.0F;
            for (std::size_t column = 0; column < kState; ++column) {
                state_row[column] *= decay[head];
                key_dot += state_row[column] * key_values[column];
            }
            const float delta = (value[head * kState + row] - key_dot) * beta[head];
            for (std::size_t column = 0; column < kState; ++column) {
                state_row[column] += delta * key_values[column];
            }
        }
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) { MIINFER_HIP_CHECK(hipMalloc(&data_, bytes)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    void* get() const noexcept { return data_; }

private:
    void* data_ = nullptr;
};

void upload(const std::vector<float>& host, DeviceBuffer& device) {
    MIINFER_HIP_CHECK(hipMemcpy(device.get(), host.data(), host.size() * sizeof(float),
                                hipMemcpyHostToDevice));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-m6a263-qwen35-recurrent-contract FIXTURE_DIR\n";
        return 2;
    }
    try {
        const std::filesystem::path fixture = argv[1];
        const auto previous = read_f32(checkpoint(fixture, 19, "state_predelta-30"), kStateElements);
        const auto expected = read_f32(checkpoint(fixture, 20, "state_predelta-30"), kStateElements);
        const auto external_q = read_f32(checkpoint(fixture, 19, "q_in-30"), kValueHeads * kState);
        const auto external_k = read_f32(checkpoint(fixture, 19, "k_in-30"), kValueHeads * kState);
        const auto value = read_f32(checkpoint(fixture, 19, "v_in-30"), kValueHeads * kState);
        const auto beta = read_f32(checkpoint(fixture, 19, "b_in-30"), kValueHeads);
        const auto gate = read_f32(checkpoint(fixture, 19, "g_in-30"), kValueHeads);
        std::vector<float> decay(kValueHeads);
        for (std::size_t i = 0; i < kValueHeads; ++i) decay[i] = std::exp(gate[i]);

        auto external_state = previous;
        apply_external(external_state, external_k, value, beta, decay);
        const auto external_error = compare(external_state, expected);

        std::vector<float> query(external_q.begin(), external_q.begin() + kKeyHeads * kState);
        std::vector<float> key(external_k.begin(), external_k.begin() + kKeyHeads * kState);
        DeviceBuffer d_query(query.size() * sizeof(float));
        DeviceBuffer d_key(key.size() * sizeof(float));
        DeviceBuffer d_value(value.size() * sizeof(float));
        DeviceBuffer d_beta(beta.size() * sizeof(float));
        DeviceBuffer d_decay(decay.size() * sizeof(float));
        DeviceBuffer d_state(previous.size() * sizeof(float));
        DeviceBuffer d_output(kValueHeads * kState * sizeof(float));
        upload(query, d_query);
        upload(key, d_key);
        upload(value, d_value);
        upload(beta, d_beta);
        upload(decay, d_decay);
        DeviceBuffer d_previous(previous.size() * sizeof(float));
        upload(previous, d_previous);
        MIINFER_HIP_CHECK(hipMemcpy(d_state.get(), d_previous.get(), previous.size() * sizeof(float),
                                    hipMemcpyDeviceToDevice));
        miinfer::launch_qwen35_deltanet_state_update(
            static_cast<const float*>(d_query.get()), static_cast<const float*>(d_key.get()),
            static_cast<const float*>(d_value.get()), static_cast<const float*>(d_beta.get()),
            static_cast<const float*>(d_decay.get()), static_cast<float*>(d_state.get()),
            static_cast<float*>(d_output.get()), kKeyHeads, kValueHeads, kState);
        MIINFER_HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> gpu_state(kStateElements);
        MIINFER_HIP_CHECK(hipMemcpy(gpu_state.data(), d_state.get(), gpu_state.size() * sizeof(float),
                                    hipMemcpyDeviceToHost));
        const auto gpu_error = compare(gpu_state, expected);
        constexpr std::size_t tracked_index = 86796;
        std::cout << "M6-A26.3 L30 P19->P20 external-input replay\n"
                  << "external_inputs/external_formula max_abs=" << external_error.max_abs
                  << " rms=" << std::sqrt(external_error.squared / kStateElements) << '\n'
                  << "external_inputs/MIInfer_GPU max_abs=" << gpu_error.max_abs
                  << " rms=" << std::sqrt(gpu_error.squared / kStateElements) << '\n'
                  << "tracked_index=" << tracked_index
                  << " expected=" << expected[tracked_index]
                  << " external_formula=" << external_state[tracked_index]
                  << " MIInfer_GPU=" << gpu_state[tracked_index] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M6-A26.3 failed: " << error.what() << '\n';
        return 1;
    }
}
