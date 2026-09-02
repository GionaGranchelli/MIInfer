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
constexpr std::size_t kQOutput = 12288;

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
            throw std::runtime_error("non-finite layer-prefix result");
        }
        result = std::max(result, std::fabs(actual[i] - expected[i]));
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a11-qwen35-attention-prefix MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        const auto layer_input = read_f32(fixture / "tensors/0-l_out-2-30.f32", kHidden);
        const auto expected_norm = read_f32(fixture / "tensors/0-attn_norm-3-31.f32", kHidden);
        const auto expected_q = read_f32(fixture / "tensors/0-Qcur_full-3-32.f32", kQOutput);
        const auto norm = model.tensor("blk.3.attn_norm.weight");
        const auto projection = model.tensor("blk.3.attn_q.weight");
        if (norm.type() != miinfer::GgufTensorType::f32
            || norm.dimensions() != std::vector<std::uint64_t>{kHidden}) {
            throw std::runtime_error("unexpected layer-3 attention norm shape");
        }
        if (projection.type() != miinfer::GgufTensorType::q4_k
            || projection.dimensions() != std::vector<std::uint64_t>{kHidden, kQOutput}) {
            throw std::runtime_error("unexpected layer-3 Q projection shape");
        }
        miinfer::DeviceInfo device;
        std::string error;
        if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);

        DeviceBuffer input_device(kHidden * sizeof(float));
        DeviceBuffer norm_device(norm.bytes());
        DeviceBuffer normalized_device(kHidden * sizeof(float));
        DeviceBuffer weight_device(projection.bytes());
        DeviceBuffer q8_device((kHidden / 256) * sizeof(miinfer::Q8KDeviceBlock));
        DeviceBuffer q_device(kQOutput * sizeof(float));
        MIINFER_HIP_CHECK(hipMemcpy(input_device.data(), layer_input.data(),
                                    layer_input.size() * sizeof(float), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(norm_device.data(), norm.data(), norm.bytes(),
                                    hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(weight_device.data(), projection.data(),
                                    projection.bytes(), hipMemcpyHostToDevice));

        miinfer::launch_qwen3_rms_norm(
            static_cast<const float*>(input_device.data()),
            static_cast<const float*>(norm_device.data()),
            static_cast<float*>(normalized_device.data()), kHidden, model.config().rms_epsilon);
        miinfer::launch_qwen3_q8_k_quantize(
            static_cast<const float*>(normalized_device.data()),
            static_cast<miinfer::Q8KDeviceBlock*>(q8_device.data()), kHidden);
        miinfer::launch_qwen3_q4_k_q8_k_gemv(
            static_cast<const miinfer::Q4KDeviceBlock*>(weight_device.data()),
            static_cast<const miinfer::Q8KDeviceBlock*>(q8_device.data()),
            static_cast<float*>(q_device.data()), kQOutput, kHidden);
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> actual_norm(kHidden);
        std::vector<float> actual_q(kQOutput);
        MIINFER_HIP_CHECK(hipMemcpy(actual_norm.data(), normalized_device.data(),
                                    actual_norm.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(actual_q.data(), q_device.data(),
                                    actual_q.size() * sizeof(float), hipMemcpyDeviceToHost));
        const float norm_error = max_abs_error(actual_norm, expected_norm);
        const float q_error = max_abs_error(actual_q, expected_q);
        std::cout << "model=" << model.model_name() << '\n'
                  << "architecture=qwen35\n"
                  << "device=" << device.name << " (" << device.architecture << ")\n"
                  << "layer=3 full-attention prefix\n"
                  << "operations=RMSNorm->Q8_K->Q4_KxQ8_K attn_q\n"
                  << "norm_max_abs_error=" << norm_error << '\n'
                  << "q_max_abs_error=" << q_error << '\n';
        if (norm_error > 1.0e-3F || q_error > 1.0F) {
            throw std::runtime_error("qwen35 attention prefix fixture mismatch");
        }
        std::cout << "M6-A11 qwen35 attention prefix PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A11 failed: " << error.what() << '\n';
        return 1;
    }
}
