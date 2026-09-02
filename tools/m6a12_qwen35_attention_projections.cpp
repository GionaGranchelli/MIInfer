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

constexpr std::size_t kHidden = 5120;
constexpr std::size_t kQOutput = 12288;
constexpr std::size_t kKvOutput = 1024;

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
        std::cerr << "usage: miinfer-m6a12-qwen35-attention-projections MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        const auto input = read_f32(fixture / "tensors/0-l_out-2-30.f32", kHidden);
        const auto expected_q = read_f32(fixture / "tensors/0-Qcur_full-3-32.f32", kQOutput);
        const auto expected_k = read_f32(fixture / "tensors/0-Kcur_normed-3-39.f32", kKvOutput);
        const auto expected_v = read_f32(fixture / "tensors/0-Vcur-3-35.f32", kKvOutput);
        const auto norm = model.tensor("blk.3.attn_norm.weight");
        const auto q_projection = model.tensor("blk.3.attn_q.weight");
        const auto k_projection = model.tensor("blk.3.attn_k.weight");
        const auto v_projection = model.tensor("blk.3.attn_v.weight");
        if (norm.type() != miinfer::GgufTensorType::f32
            || norm.dimensions() != std::vector<std::uint64_t>{kHidden}) {
            throw std::runtime_error("unexpected layer-3 attention norm shape");
        }
        if (q_projection.type() != miinfer::GgufTensorType::q4_k
            || q_projection.dimensions() != std::vector<std::uint64_t>{kHidden, kQOutput}) {
            throw std::runtime_error("unexpected layer-3 Q projection shape");
        }
        if (k_projection.type() != miinfer::GgufTensorType::q4_k
            || k_projection.dimensions() != std::vector<std::uint64_t>{kHidden, kKvOutput}) {
            throw std::runtime_error("unexpected layer-3 K projection shape");
        }
        if ((v_projection.type() != miinfer::GgufTensorType::q4_k
             && v_projection.type() != miinfer::GgufTensorType::q6_k)
            || v_projection.dimensions() != std::vector<std::uint64_t>{kHidden, kKvOutput}) {
            throw std::runtime_error("unexpected layer-3 V projection shape");
        }
        const auto k_norm = model.tensor("blk.3.attn_k_norm.weight");
        if (k_norm.type() != miinfer::GgufTensorType::f32
            || k_norm.dimensions() != std::vector<std::uint64_t>{256}) {
            throw std::runtime_error("unexpected layer-3 K norm shape");
        }

        miinfer::DeviceInfo device;
        std::string error;
        if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);
        DeviceBuffer input_device(kHidden * sizeof(float));
        DeviceBuffer norm_device(norm.bytes());
        DeviceBuffer normalized_device(kHidden * sizeof(float));
        DeviceBuffer q_weight_device(q_projection.bytes());
        DeviceBuffer k_weight_device(k_projection.bytes());
        DeviceBuffer v_weight_device(v_projection.bytes());
        DeviceBuffer q8_device((kHidden / 256) * sizeof(miinfer::Q8KDeviceBlock));
        DeviceBuffer q_device(kQOutput * sizeof(float));
        DeviceBuffer k_device(kKvOutput * sizeof(float));
        DeviceBuffer v_device(kKvOutput * sizeof(float));
        DeviceBuffer k_norm_weights_device(k_norm.bytes());
        DeviceBuffer k_normalized_device(kKvOutput * sizeof(float));

        MIINFER_HIP_CHECK(hipMemcpy(input_device.data(), input.data(),
                                    input.size() * sizeof(float), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(norm_device.data(), norm.data(), norm.bytes(),
                                    hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(q_weight_device.data(), q_projection.data(),
                                    q_projection.bytes(), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(k_weight_device.data(), k_projection.data(),
                                    k_projection.bytes(), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(v_weight_device.data(), v_projection.data(),
                                    v_projection.bytes(), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(k_norm_weights_device.data(), k_norm.data(), k_norm.bytes(),
                                    hipMemcpyHostToDevice));

        miinfer::launch_qwen3_rms_norm(
            static_cast<const float*>(input_device.data()),
            static_cast<const float*>(norm_device.data()),
            static_cast<float*>(normalized_device.data()), kHidden, model.config().rms_epsilon);
        miinfer::launch_qwen3_q8_k_quantize(
            static_cast<const float*>(normalized_device.data()),
            static_cast<miinfer::Q8KDeviceBlock*>(q8_device.data()), kHidden);
        miinfer::launch_qwen3_q4_k_q8_k_gemv(
            static_cast<const miinfer::Q4KDeviceBlock*>(q_weight_device.data()),
            static_cast<const miinfer::Q8KDeviceBlock*>(q8_device.data()),
            static_cast<float*>(q_device.data()), kQOutput, kHidden);
        miinfer::launch_qwen3_q4_k_q8_k_gemv(
            static_cast<const miinfer::Q4KDeviceBlock*>(k_weight_device.data()),
            static_cast<const miinfer::Q8KDeviceBlock*>(q8_device.data()),
            static_cast<float*>(k_device.data()), kKvOutput, kHidden);
        if (v_projection.type() == miinfer::GgufTensorType::q4_k) {
            miinfer::launch_qwen3_q4_k_q8_k_gemv(
                static_cast<const miinfer::Q4KDeviceBlock*>(v_weight_device.data()),
                static_cast<const miinfer::Q8KDeviceBlock*>(q8_device.data()),
                static_cast<float*>(v_device.data()), kKvOutput, kHidden);
        } else {
            miinfer::launch_qwen3_q6_k_q8_k_gemv(
                static_cast<const miinfer::Q6KDeviceBlock*>(v_weight_device.data()),
                static_cast<const miinfer::Q8KDeviceBlock*>(q8_device.data()),
                static_cast<float*>(v_device.data()), kKvOutput, kHidden);
        }
        for (std::size_t head = 0; head < 4; ++head) {
            miinfer::launch_qwen3_rms_normalize(
                static_cast<const float*>(k_device.data()) + head * 256,
                static_cast<float*>(k_normalized_device.data()) + head * 256,
                256, model.config().rms_epsilon);
        }
        miinfer::launch_qwen3_head_mul(
            static_cast<const float*>(k_normalized_device.data()),
            static_cast<const float*>(k_norm_weights_device.data()),
            static_cast<float*>(k_normalized_device.data()), 4, 256);
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> actual_q(kQOutput);
        std::vector<float> actual_k(kKvOutput);
        std::vector<float> actual_v(kKvOutput);
        std::vector<float> actual_k_norm(kKvOutput);
        MIINFER_HIP_CHECK(hipMemcpy(actual_q.data(), q_device.data(),
                                    actual_q.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(actual_k.data(), k_device.data(),
                                    actual_k.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(actual_v.data(), v_device.data(),
                                    actual_v.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(actual_k_norm.data(), k_normalized_device.data(),
                                    actual_k_norm.size() * sizeof(float), hipMemcpyDeviceToHost));
        const float q_error = max_abs_error(actual_q, expected_q);
        const float k_error = max_abs_error(actual_k_norm, expected_k);
        const float v_error = max_abs_error(actual_v, expected_v);
        std::cout << "model=" << model.model_name() << '\n'
                  << "architecture=qwen35\n"
                  << "device=" << device.name << " (" << device.architecture << ")\n"
                  << "layer=3 full-attention projections\n"
                  << "operations=RMSNorm->Q8_K->{Q,K,V}->KNorm\n"
                  << "v_type=" << miinfer::gguf_tensor_type_name(v_projection.type()) << '\n'
                  << "q_max_abs_error=" << q_error << '\n'
                  << "k_norm_max_abs_error=" << k_error << '\n'
                  << "v_max_abs_error=" << v_error << '\n';
        if (q_error > 1.0F || k_error > 1.0F || v_error > 1.0F) {
            throw std::runtime_error("qwen35 attention projection fixture mismatch");
        }
        std::cout << "M6-A12 qwen35 attention projections PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A12 failed: " << error.what() << '\n';
        return 1;
    }
}
