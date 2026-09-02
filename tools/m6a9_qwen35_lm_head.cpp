#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kHidden = 5120;
constexpr std::size_t kVocab = 248320;

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) { MIINFER_HIP_CHECK(hipMalloc(&data_, bytes)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    [[nodiscard]] void* data() const noexcept { return data_; }

private:
    void* data_ = nullptr;
};

std::vector<float> read_f32(const std::filesystem::path& path, std::size_t elements) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open fixture: " + path.string());
    std::vector<float> result(elements);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(elements * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(elements * sizeof(float))) {
        throw std::runtime_error("short fixture: " + path.string());
    }
    return result;
}

float max_abs_error(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("comparison size mismatch");
    float result = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error("non-finite LM-head result");
        }
        result = std::max(result, std::fabs(actual[i] - expected[i]));
    }
    return result;
}

std::size_t argmax(const std::vector<float>& values) {
    return static_cast<std::size_t>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a9-qwen35-lm-head MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        const auto input = read_f32(fixture / "tensors/0-result_norm-849.f32", kHidden);
        const auto expected = read_f32(fixture / "logits/logits-0.f32", kVocab);
        const auto output_weight = model.tensor("output.weight");
        if (output_weight.type() != miinfer::GgufTensorType::q6_k
            || output_weight.dimensions() != std::vector<std::uint64_t>{kHidden, kVocab}) {
            throw std::runtime_error("unexpected qwen35 Q6_K LM-head shape");
        }
        miinfer::DeviceInfo device;
        std::string error;
        if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);

        DeviceBuffer input_device(kHidden * sizeof(float));
        DeviceBuffer weight_device(output_weight.bytes());
        DeviceBuffer q8_device((kHidden / 256) * sizeof(miinfer::Q8KDeviceBlock));
        DeviceBuffer logits_device(kVocab * sizeof(float));
        MIINFER_HIP_CHECK(hipMemcpy(input_device.data(), input.data(),
                                    input.size() * sizeof(float), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(weight_device.data(), output_weight.data(),
                                    output_weight.bytes(), hipMemcpyHostToDevice));
        miinfer::launch_qwen3_q8_k_quantize(
            static_cast<const float*>(input_device.data()),
            static_cast<miinfer::Q8KDeviceBlock*>(q8_device.data()), kHidden);
        miinfer::launch_qwen3_q6_k_q8_k_gemv(
            static_cast<const miinfer::Q6KDeviceBlock*>(weight_device.data()),
            static_cast<const miinfer::Q8KDeviceBlock*>(q8_device.data()),
            static_cast<float*>(logits_device.data()), kVocab, kHidden);
        MIINFER_HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> actual(kVocab);
        MIINFER_HIP_CHECK(hipMemcpy(actual.data(), logits_device.data(),
                                    actual.size() * sizeof(float), hipMemcpyDeviceToHost));
        const float error_abs = max_abs_error(actual, expected);
        const auto actual_token = argmax(actual);
        const auto expected_token = argmax(expected);

        std::cout << "model=" << model.model_name() << '\n'
                  << "architecture=qwen35\n"
                  << "device=" << device.name << " (" << device.architecture << ")\n"
                  << "operation=Q6_KxQ8_K LM-head\n"
                  << "shape=" << kHidden << "->" << kVocab << '\n'
                  << "max_abs_error=" << error_abs << '\n'
                  << "argmax=" << actual_token << " expected=" << expected_token << '\n';
        if (error_abs > 1.0F || actual_token != expected_token) {
            throw std::runtime_error("qwen35 LM-head fixture mismatch");
        }
        std::cout << "M6-A9 qwen35 LM-head PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A9 failed: " << error.what() << '\n';
        return 1;
    }
}
