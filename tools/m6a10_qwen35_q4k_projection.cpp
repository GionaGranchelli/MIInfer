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

constexpr std::size_t kInput = 5120;
constexpr std::size_t kOutput = 12288;

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
            throw std::runtime_error("non-finite projection result");
        }
        result = std::max(result, std::fabs(actual[i] - expected[i]));
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a10-qwen35-q4k-projection MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        const auto input = read_f32(fixture / "tensors/0-attn_norm-3-31.f32", kInput);
        const auto expected = read_f32(fixture / "tensors/0-Qcur_full-3-32.f32", kOutput);
        const auto projection = model.tensor("blk.3.attn_q.weight");
        if (projection.type() != miinfer::GgufTensorType::q4_k
            || projection.dimensions() != std::vector<std::uint64_t>{kInput, kOutput}) {
            throw std::runtime_error("unexpected qwen35 Q4_K attention-Q shape");
        }
        miinfer::DeviceInfo device;
        std::string error;
        if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);

        DeviceBuffer input_device(kInput * sizeof(float));
        DeviceBuffer weight_device(projection.bytes());
        DeviceBuffer q8_device((kInput / 256) * sizeof(miinfer::Q8KDeviceBlock));
        DeviceBuffer output_device(kOutput * sizeof(float));
        MIINFER_HIP_CHECK(hipMemcpy(input_device.data(), input.data(),
                                    input.size() * sizeof(float), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(weight_device.data(), projection.data(),
                                    projection.bytes(), hipMemcpyHostToDevice));
        miinfer::launch_qwen3_q8_k_quantize(
            static_cast<const float*>(input_device.data()),
            static_cast<miinfer::Q8KDeviceBlock*>(q8_device.data()), kInput);
        miinfer::launch_qwen3_q4_k_q8_k_gemv(
            static_cast<const miinfer::Q4KDeviceBlock*>(weight_device.data()),
            static_cast<const miinfer::Q8KDeviceBlock*>(q8_device.data()),
            static_cast<float*>(output_device.data()), kOutput, kInput);
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> actual(kOutput);
        MIINFER_HIP_CHECK(hipMemcpy(actual.data(), output_device.data(),
                                    actual.size() * sizeof(float), hipMemcpyDeviceToHost));
        const float error_abs = max_abs_error(actual, expected);
        std::cout << "model=" << model.model_name() << '\n'
                  << "architecture=qwen35\n"
                  << "device=" << device.name << " (" << device.architecture << ")\n"
                  << "operation=Q4_KxQ8_K attn_q\n"
                  << "shape=" << kInput << "->" << kOutput << '\n'
                  << "max_abs_error=" << error_abs << '\n';
        if (error_abs > 1.0F) throw std::runtime_error("qwen35 Q4_K projection fixture mismatch");
        std::cout << "M6-A10 qwen35 Q4_K projection PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A10 failed: " << error.what() << '\n';
        return 1;
    }
}
