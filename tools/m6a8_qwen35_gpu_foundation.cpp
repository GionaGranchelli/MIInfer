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
#include <string>
#include <vector>

namespace {

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
            throw std::runtime_error("non-finite RMSNorm result");
        }
        result = std::max(result, std::fabs(actual[i] - expected[i]));
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a8-qwen35-gpu-foundation MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        const auto input = read_f32(fixture / "tensors/0-model_input_embed-0.f32",
                                    model.config().hidden_size);
        const auto expected = read_f32(fixture / "tensors/0-attn_norm-0-1.f32",
                                       model.config().hidden_size);
        miinfer::DeviceInfo device;
        std::string error;
        if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);

        const auto norm = model.tensor("blk.0.attn_norm.weight");
        DeviceBuffer input_device(input.size() * sizeof(float));
        DeviceBuffer norm_device(norm.bytes());
        DeviceBuffer output_device(expected.size() * sizeof(float));
        MIINFER_HIP_CHECK(hipMemcpy(input_device.data(), input.data(),
                                    input.size() * sizeof(float), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(norm_device.data(), norm.data(), norm.bytes(),
                                    hipMemcpyHostToDevice));
        miinfer::launch_qwen3_rms_norm(
            static_cast<const float*>(input_device.data()),
            static_cast<const float*>(norm_device.data()),
            static_cast<float*>(output_device.data()), model.config().hidden_size,
            model.config().rms_epsilon);
        MIINFER_HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> actual(expected.size());
        MIINFER_HIP_CHECK(hipMemcpy(actual.data(), output_device.data(),
                                    actual.size() * sizeof(float), hipMemcpyDeviceToHost));
        const float error_abs = max_abs_error(actual, expected);

        std::cout << "model=" << model.model_name() << '\n'
                  << "architecture=qwen35\n"
                  << "device=" << device.name << " (" << device.architecture << ")\n"
                  << "hidden=" << model.config().hidden_size << '\n'
                  << "main_layers=" << model.config().main_layer_count << '\n'
                  << "rms_norm=blk.0.attn_norm.weight\n"
                  << "max_abs_error=" << error_abs << '\n';
        if (error_abs > 1.0e-3F) throw std::runtime_error("qwen35 RMSNorm GPU fixture mismatch");
        std::cout << "M6-A8 qwen35 GPU foundation RMSNorm PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A8 failed: " << error.what() << '\n';
        return 1;
    }
}
