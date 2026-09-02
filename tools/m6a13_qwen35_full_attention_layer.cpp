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
constexpr std::size_t kQFull = 12288;
constexpr std::size_t kAttention = 6144;
constexpr std::size_t kKv = 1024;
constexpr std::size_t kFfn = 17408;
constexpr std::size_t kHeads = 24;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kCacheCapacity = 16;

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

std::filesystem::path checkpoint(const std::filesystem::path& fixture,
                                 std::size_t position, std::string_view name) {
    const auto prefix = std::to_string(position) + "-" + std::string(name) + "-";
    for (const auto& entry : std::filesystem::directory_iterator(fixture / "tensors")) {
        if (entry.path().filename().string().starts_with(prefix)) return entry.path();
    }
    throw std::runtime_error("missing checkpoint: " + prefix);
}

float max_abs_error(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("comparison size mismatch");
    float result = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error("non-finite layer result");
        }
        result = std::max(result, std::fabs(actual[i] - expected[i]));
    }
    return result;
}

void copy_to_host(const DeviceBuffer& source, std::vector<float>& destination) {
    MIINFER_HIP_CHECK(hipMemcpy(destination.data(), source.data(),
                                destination.size() * sizeof(float), hipMemcpyDeviceToHost));
}

float check(const DeviceBuffer& actual_device, const std::filesystem::path& expected_path,
            std::size_t elements) {
    std::vector<float> actual(elements);
    copy_to_host(actual_device, actual);
    return max_abs_error(actual, read_f32(expected_path, elements));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a13-qwen35-full-attention-layer MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        const auto q_weight = model.tensor("blk.3.attn_q.weight");
        const auto k_weight = model.tensor("blk.3.attn_k.weight");
        const auto v_weight = model.tensor("blk.3.attn_v.weight");
        const auto o_weight = model.tensor("blk.3.attn_output.weight");
        const auto ffn_gate_weight = model.tensor("blk.3.ffn_gate.weight");
        const auto ffn_up_weight = model.tensor("blk.3.ffn_up.weight");
        const auto ffn_down_weight = model.tensor("blk.3.ffn_down.weight");
        const auto attn_norm = model.tensor("blk.3.attn_norm.weight");
        const auto q_norm = model.tensor("blk.3.attn_q_norm.weight");
        const auto k_norm = model.tensor("blk.3.attn_k_norm.weight");
        const auto post_norm = model.tensor("blk.3.post_attention_norm.weight");
        if (q_weight.type() != miinfer::GgufTensorType::q4_k
            || k_weight.type() != miinfer::GgufTensorType::q4_k
            || o_weight.type() != miinfer::GgufTensorType::q4_k
            || ffn_gate_weight.type() != miinfer::GgufTensorType::q4_k
            || ffn_up_weight.type() != miinfer::GgufTensorType::q4_k
            || (ffn_down_weight.type() != miinfer::GgufTensorType::q4_k
                && ffn_down_weight.type() != miinfer::GgufTensorType::q6_k)
            || v_weight.dimensions() != std::vector<std::uint64_t>{kHidden, kKv}) {
            throw std::runtime_error("unsupported layer-3 full-attention tensor contract");
        }
        miinfer::DeviceInfo device;
        std::string error;
        if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);

        DeviceBuffer input(kHidden * sizeof(float));
        DeviceBuffer attn_norm_device(attn_norm.bytes());
        DeviceBuffer normalized(kHidden * sizeof(float));
        DeviceBuffer q_weight_device(q_weight.bytes());
        DeviceBuffer k_weight_device(k_weight.bytes());
        DeviceBuffer v_weight_device(v_weight.bytes());
        DeviceBuffer o_weight_device(o_weight.bytes());
        DeviceBuffer ffn_gate_weight_device(ffn_gate_weight.bytes());
        DeviceBuffer ffn_up_weight_device(ffn_up_weight.bytes());
        DeviceBuffer ffn_down_weight_device(ffn_down_weight.bytes());
        DeviceBuffer q_norm_device(q_norm.bytes());
        DeviceBuffer k_norm_device(k_norm.bytes());
        DeviceBuffer post_norm_device(post_norm.bytes());
        DeviceBuffer q8(kFfn / 256 * sizeof(miinfer::Q8KDeviceBlock));
        DeviceBuffer qfull(kQFull * sizeof(float));
        DeviceBuffer query(kAttention * sizeof(float));
        DeviceBuffer query_norm(kAttention * sizeof(float));
        DeviceBuffer gate(kAttention * sizeof(float));
        DeviceBuffer key(kKv * sizeof(float));
        DeviceBuffer key_norm(kKv * sizeof(float));
        DeviceBuffer value(kKv * sizeof(float));
        DeviceBuffer query_rope(kAttention * sizeof(float));
        DeviceBuffer key_rope(kKv * sizeof(float));
        DeviceBuffer key_cache(kKvHeads * kCacheCapacity * kHeadDim * sizeof(float));
        DeviceBuffer value_cache(kKvHeads * kCacheCapacity * kHeadDim * sizeof(float));
        DeviceBuffer attention(kAttention * sizeof(float));
        DeviceBuffer scores(kHeads * kCacheCapacity * sizeof(float));
        DeviceBuffer probabilities(kHeads * kCacheCapacity * sizeof(float));
        DeviceBuffer gated_attention(kAttention * sizeof(float));
        DeviceBuffer projected(kHidden * sizeof(float));
        DeviceBuffer residual(kHidden * sizeof(float));
        DeviceBuffer post_normalized(kHidden * sizeof(float));
        DeviceBuffer ffn_gate(kFfn * sizeof(float));
        DeviceBuffer ffn_up(kFfn * sizeof(float));
        DeviceBuffer ffn(kFfn * sizeof(float));
        DeviceBuffer output(kHidden * sizeof(float));

        auto upload = [](const auto& tensor, DeviceBuffer& destination) {
            MIINFER_HIP_CHECK(hipMemcpy(destination.data(), tensor.data(), tensor.bytes(),
                                        hipMemcpyHostToDevice));
        };
        upload(attn_norm, attn_norm_device);
        upload(q_weight, q_weight_device);
        upload(k_weight, k_weight_device);
        upload(v_weight, v_weight_device);
        upload(o_weight, o_weight_device);
        upload(ffn_gate_weight, ffn_gate_weight_device);
        upload(ffn_up_weight, ffn_up_weight_device);
        upload(ffn_down_weight, ffn_down_weight_device);
        upload(q_norm, q_norm_device);
        upload(k_norm, k_norm_device);
        upload(post_norm, post_norm_device);
        MIINFER_HIP_CHECK(hipMemset(key_cache.data(), 0,
                                    kKvHeads * kCacheCapacity * kHeadDim * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(value_cache.data(), 0,
                                    kKvHeads * kCacheCapacity * kHeadDim * sizeof(float)));

        float maximum_layer_error = 0.0F;
        float maximum_attention_error = 0.0F;
        float maximum_ffn_error = 0.0F;
        for (std::size_t position = 0; position <= 8; ++position) {
            const auto layer_input = read_f32(checkpoint(fixture, position, "l_out-2"), kHidden);
            MIINFER_HIP_CHECK(hipMemcpy(input.data(), layer_input.data(),
                                        layer_input.size() * sizeof(float), hipMemcpyHostToDevice));
            miinfer::launch_qwen3_rms_norm(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(attn_norm_device.data()),
                static_cast<float*>(normalized.data()), kHidden, model.config().rms_epsilon);
            miinfer::launch_qwen3_q8_k_quantize(
                static_cast<const float*>(normalized.data()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8.data()), kHidden);
            miinfer::launch_qwen3_q4_k_q8_k_gemv(
                static_cast<const miinfer::Q4KDeviceBlock*>(q_weight_device.data()),
                static_cast<const miinfer::Q8KDeviceBlock*>(q8.data()),
                static_cast<float*>(qfull.data()), kQFull, kHidden);
            miinfer::launch_qwen35_split_q_gate(
                static_cast<const float*>(qfull.data()), static_cast<float*>(query.data()),
                static_cast<float*>(gate.data()), kHeads, kHeadDim);
            for (std::size_t head = 0; head < kHeads; ++head) {
                miinfer::launch_qwen3_rms_normalize(
                    static_cast<const float*>(query.data()) + head * kHeadDim,
                    static_cast<float*>(query_norm.data()) + head * kHeadDim,
                    kHeadDim, model.config().rms_epsilon);
            }
            miinfer::launch_qwen3_head_mul(
                static_cast<const float*>(query_norm.data()),
                static_cast<const float*>(q_norm_device.data()),
                static_cast<float*>(query_norm.data()), kHeads, kHeadDim);
            miinfer::launch_qwen3_q4_k_q8_k_gemv(
                static_cast<const miinfer::Q4KDeviceBlock*>(k_weight_device.data()),
                static_cast<const miinfer::Q8KDeviceBlock*>(q8.data()),
                static_cast<float*>(key.data()), kKv, kHidden);
            for (std::size_t head = 0; head < kKvHeads; ++head) {
                miinfer::launch_qwen3_rms_normalize(
                    static_cast<const float*>(key.data()) + head * kHeadDim,
                    static_cast<float*>(key_norm.data()) + head * kHeadDim,
                    kHeadDim, model.config().rms_epsilon);
            }
            miinfer::launch_qwen3_head_mul(
                static_cast<const float*>(key_norm.data()),
                static_cast<const float*>(k_norm_device.data()),
                static_cast<float*>(key_norm.data()), kKvHeads, kHeadDim);
            miinfer::launch_qwen3_q6_k_q8_k_gemv(
                static_cast<const miinfer::Q6KDeviceBlock*>(v_weight_device.data()),
                static_cast<const miinfer::Q8KDeviceBlock*>(q8.data()),
                static_cast<float*>(value.data()), kKv, kHidden);
            miinfer::launch_qwen35_rope_sections(
                static_cast<const float*>(query_norm.data()), static_cast<float*>(query_rope.data()),
                kHeads, kHeadDim, position, model.config().rope_theta);
            miinfer::launch_qwen35_rope_sections(
                static_cast<const float*>(key_norm.data()), static_cast<float*>(key_rope.data()),
                kKvHeads, kHeadDim, position, model.config().rope_theta);
            miinfer::launch_qwen3_kv_cache_store(
                static_cast<const float*>(key_rope.data()), static_cast<const float*>(value.data()),
                static_cast<float*>(key_cache.data()), static_cast<float*>(value_cache.data()),
                position, kCacheCapacity, kKvHeads, kHeadDim);
            miinfer::launch_qwen3_cached_attention_parallel(
                static_cast<const float*>(query_rope.data()),
                static_cast<const float*>(key_cache.data()),
                static_cast<const float*>(value_cache.data()), position + 1, kCacheCapacity,
                static_cast<float*>(attention.data()), static_cast<float*>(scores.data()),
                static_cast<float*>(probabilities.data()), kHeads, kKvHeads, kHeadDim,
                1.0F / std::sqrt(static_cast<float>(kHeadDim)));
            miinfer::launch_qwen35_sigmoid_mul(
                static_cast<const float*>(attention.data()), static_cast<const float*>(gate.data()),
                static_cast<float*>(gated_attention.data()), kAttention);
            miinfer::launch_qwen3_q8_k_quantize(
                static_cast<const float*>(gated_attention.data()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8.data()), kAttention);
            miinfer::launch_qwen3_q4_k_q8_k_gemv(
                static_cast<const miinfer::Q4KDeviceBlock*>(o_weight_device.data()),
                static_cast<const miinfer::Q8KDeviceBlock*>(q8.data()),
                static_cast<float*>(projected.data()), kHidden, kAttention);
            miinfer::launch_qwen3_add(
                static_cast<const float*>(input.data()), static_cast<const float*>(projected.data()),
                static_cast<float*>(residual.data()), kHidden);
            miinfer::launch_qwen3_rms_norm(
                static_cast<const float*>(residual.data()),
                static_cast<const float*>(post_norm_device.data()),
                static_cast<float*>(post_normalized.data()), kHidden, model.config().rms_epsilon);
            miinfer::launch_qwen3_q8_k_quantize(
                static_cast<const float*>(post_normalized.data()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8.data()), kHidden);
            miinfer::launch_qwen3_q4_k_q8_k_gemv(
                static_cast<const miinfer::Q4KDeviceBlock*>(ffn_gate_weight_device.data()),
                static_cast<const miinfer::Q8KDeviceBlock*>(q8.data()),
                static_cast<float*>(ffn_gate.data()), kFfn, kHidden);
            miinfer::launch_qwen3_q4_k_q8_k_gemv(
                static_cast<const miinfer::Q4KDeviceBlock*>(ffn_up_weight_device.data()),
                static_cast<const miinfer::Q8KDeviceBlock*>(q8.data()),
                static_cast<float*>(ffn_up.data()), kFfn, kHidden);
            miinfer::launch_qwen3_silu_mul(
                static_cast<const float*>(ffn_gate.data()), static_cast<const float*>(ffn_up.data()),
                static_cast<float*>(ffn.data()), kFfn);
            miinfer::launch_qwen3_q8_k_quantize(
                static_cast<const float*>(ffn.data()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8.data()), kFfn);
            if (ffn_down_weight.type() == miinfer::GgufTensorType::q4_k) {
                miinfer::launch_qwen3_q4_k_q8_k_gemv(
                    static_cast<const miinfer::Q4KDeviceBlock*>(ffn_down_weight_device.data()),
                    static_cast<const miinfer::Q8KDeviceBlock*>(q8.data()),
                    static_cast<float*>(output.data()), kHidden, kFfn);
            } else {
            if (ffn_down_weight.type() == miinfer::GgufTensorType::q4_k) {
                miinfer::launch_qwen3_q4_k_q8_k_gemv(
                    static_cast<const miinfer::Q4KDeviceBlock*>(ffn_down_weight_device.data()),
                    static_cast<const miinfer::Q8KDeviceBlock*>(q8.data()),
                    static_cast<float*>(output.data()), kHidden, kFfn);
            } else {
                miinfer::launch_qwen3_q6_k_q8_k_gemv(
                    static_cast<const miinfer::Q6KDeviceBlock*>(ffn_down_weight_device.data()),
                    static_cast<const miinfer::Q8KDeviceBlock*>(q8.data()),
                    static_cast<float*>(output.data()), kHidden, kFfn);
            }
            }
            miinfer::launch_qwen3_add(
                static_cast<const float*>(residual.data()), static_cast<const float*>(output.data()),
                static_cast<float*>(output.data()), kHidden);
            MIINFER_HIP_CHECK(hipDeviceSynchronize());

            const auto norm_error = check(normalized, checkpoint(fixture, position, "attn_norm-3"), kHidden);
            const auto q_error = check(qfull, checkpoint(fixture, position, "Qcur_full-3"), kQFull);
            const auto k_error = check(key_norm, checkpoint(fixture, position, "Kcur_normed-3"), kKv);
            const auto v_error = check(value, checkpoint(fixture, position, "Vcur-3"), kKv);
            const auto rope_k_error = check(key_rope, checkpoint(fixture, position, "Kcur-3"), kKv);
            const auto attention_error = check(attention, checkpoint(fixture, position, "attn_pregate-3"), kAttention);
            const auto gated_error = check(gated_attention, checkpoint(fixture, position, "attn_gated-3"), kAttention);
            const auto projected_error = check(projected, checkpoint(fixture, position, "attn_output-3"), kHidden);
            const auto residual_error = check(residual, checkpoint(fixture, position, "attn_residual-3"), kHidden);
            const auto post_error = check(post_normalized, checkpoint(fixture, position, "attn_post_norm-3"), kHidden);
            const auto ffn_error = check(output, checkpoint(fixture, position, "l_out-3"), kHidden);
            maximum_attention_error = std::max({maximum_attention_error, attention_error, gated_error});
            maximum_ffn_error = std::max({maximum_ffn_error, projected_error, residual_error, post_error});
            maximum_layer_error = std::max({maximum_layer_error, norm_error, q_error, k_error, v_error,
                                            rope_k_error, ffn_error});
            std::cout << "position=" << position
                      << " norm=" << norm_error
                      << " q=" << q_error
                      << " k=" << k_error
                      << " v=" << v_error
                      << " rope_k=" << rope_k_error
                      << " attention=" << attention_error
                      << " gated=" << gated_error
                      << " projected=" << projected_error
                      << " residual=" << residual_error
                      << " post_norm=" << post_error
                      << " layer=" << ffn_error << '\n';
            if (std::max({norm_error, q_error, k_error, v_error, rope_k_error,
                         attention_error, gated_error, projected_error, residual_error,
                         post_error, ffn_error}) > 1.0F) {
                throw std::runtime_error("qwen35 layer-3 external checkpoint mismatch");
            }
        }
        std::cout << "model=" << model.model_name() << '\n'
                  << "architecture=qwen35\n"
                  << "device=" << device.name << " (" << device.architecture << ")\n"
                  << "layer=3 complete full-attention vertical slice\n"
                  << "max_attention_error=" << maximum_attention_error << '\n'
                  << "max_ffn_error=" << maximum_ffn_error << '\n'
                  << "max_layer_error=" << maximum_layer_error << '\n'
                  << "M6-A13 qwen35 full-attention layer PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A13 failed: " << error.what() << '\n';
        return 1;
    }
}
