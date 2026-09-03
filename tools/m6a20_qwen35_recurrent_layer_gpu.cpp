#define MIINFER_M6A3_HELPERS_ONLY
#include "m6a3_qwen35_layer.cpp"

#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

void upload(const void* source, void* destination, std::size_t bytes) {
    MIINFER_HIP_CHECK(hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice));
}

Metrics device_error(const float* device, std::size_t elements,
                     const std::filesystem::path& expected_path) {
    std::vector<float> actual(elements);
    MIINFER_HIP_CHECK(hipMemcpy(actual.data(), device, elements * sizeof(float),
                                hipMemcpyDeviceToHost));
    return compare(actual, read_f32(expected_path, elements));
}

void check_device(const char* label, const float* device, std::size_t elements,
                  const std::filesystem::path& expected_path, float tolerance,
                  float& maximum) {
    const auto error = device_error(device, elements, expected_path);
    require_match(label, error, tolerance);
    maximum = std::max(maximum, error.max_abs);
    std::cout << label << " max_abs=" << error.max_abs << " rmse=" << error.rmse << '\n';
}

void check_values(const char* label, const float* device, std::span<const float> expected,
                  float tolerance, float& maximum) {
    std::vector<float> actual(expected.size());
    MIINFER_HIP_CHECK(hipMemcpy(actual.data(), device, actual.size() * sizeof(float),
                                hipMemcpyDeviceToHost));
    const auto error = compare(actual, expected);
    require_match(label, error, tolerance);
    maximum = std::max(maximum, error.max_abs);
    std::cout << label << " max_abs=" << error.max_abs << " rmse=" << error.rmse << '\n';
}

void project(const miinfer::GgufTensor& weight, const void* device_weight,
             const float* input, miinfer::Q8KDeviceBlock* q8, float* output,
             std::uint32_t rows, std::uint32_t columns) {
    miinfer::launch_qwen3_q8_k_quantize(input, q8, columns);
    switch (weight.type) {
    case miinfer::GgufTensorType::q4_k:
        miinfer::launch_qwen3_q4_k_q8_k_gemv(
            static_cast<const miinfer::Q4KDeviceBlock*>(device_weight), q8,
            output, rows, columns);
        return;
    case miinfer::GgufTensorType::q5_k:
        miinfer::launch_qwen3_q5_k_q8_k_gemv(
            static_cast<const miinfer::Q5KDeviceBlock*>(device_weight), q8,
            output, rows, columns);
        return;
    case miinfer::GgufTensorType::q6_k:
        miinfer::launch_qwen3_q6_k_q8_k_gemv(
            static_cast<const miinfer::Q6KDeviceBlock*>(device_weight), q8,
            output, rows, columns);
        return;
    default:
        throw std::runtime_error("unsupported recurrent projection: " + weight.name);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a20-qwen35-recurrent-layer-gpu MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        const auto& file = *model.file();
        const auto& qkv_weight = tensor(file, "blk.0.attn_qkv.weight");
        const auto& gate_weight = tensor(file, "blk.0.attn_gate.weight");
        const auto& attn_norm_weight = tensor(file, "blk.0.attn_norm.weight");
        const auto& beta_weight = tensor(file, "blk.0.ssm_beta.weight");
        const auto& alpha_weight = tensor(file, "blk.0.ssm_alpha.weight");
        const auto& dt_weight = tensor(file, "blk.0.ssm_dt.bias");
        const auto& a_weight = tensor(file, "blk.0.ssm_a");
        const auto& conv_weight = tensor(file, "blk.0.ssm_conv1d.weight");
        const auto& ssm_norm_weight = tensor(file, "blk.0.ssm_norm.weight");
        const auto& ssm_out_weight = tensor(file, "blk.0.ssm_out.weight");
        const auto& post_norm_weight = tensor(file, "blk.0.post_attention_norm.weight");
        const auto& ffn_gate_weight = tensor(file, "blk.0.ffn_gate.weight");
        const auto& ffn_up_weight = tensor(file, "blk.0.ffn_up.weight");
        const auto& ffn_down_weight = tensor(file, "blk.0.ffn_down.weight");

        if ((qkv_weight.type != miinfer::GgufTensorType::q4_k
             && qkv_weight.type != miinfer::GgufTensorType::q6_k)
            || gate_weight.type != miinfer::GgufTensorType::q4_k
            || beta_weight.type != miinfer::GgufTensorType::f32
            || alpha_weight.type != miinfer::GgufTensorType::f32
            || dt_weight.type != miinfer::GgufTensorType::f32
            || a_weight.type != miinfer::GgufTensorType::f32
            || conv_weight.type != miinfer::GgufTensorType::f32
            || ssm_norm_weight.type != miinfer::GgufTensorType::f32
            || ssm_out_weight.type != miinfer::GgufTensorType::q5_k
            || post_norm_weight.type != miinfer::GgufTensorType::f32
            || ffn_gate_weight.type != miinfer::GgufTensorType::q4_k
            || ffn_up_weight.type != miinfer::GgufTensorType::q4_k
            || (ffn_down_weight.type != miinfer::GgufTensorType::q4_k
                && ffn_down_weight.type != miinfer::GgufTensorType::q6_k)) {
            throw std::runtime_error("unexpected layer-0 recurrent tensor contract");
        }

        DeviceBuffer<float> input(kHidden);
        DeviceBuffer<float> normalized(kHidden);
        DeviceBuffer<float> qkv(kChannels);
        DeviceBuffer<float> gate(kInner);
        DeviceBuffer<float> beta_raw(kVHeads);
        DeviceBuffer<float> alpha_raw(kVHeads);
        DeviceBuffer<float> beta(kVHeads);
        DeviceBuffer<float> decay(kVHeads);
        DeviceBuffer<float> conv_history(4 * kChannels);
        DeviceBuffer<float> query(kKHeads * kState);
        DeviceBuffer<float> key(kKHeads * kState);
        DeviceBuffer<float> value(kVHeads * kState);
        DeviceBuffer<float> query_norm(kKHeads * kState);
        DeviceBuffer<float> key_norm(kKHeads * kState);
        DeviceBuffer<float> state(kVHeads * kState * kState);
        DeviceBuffer<float> recurrent_output(kVHeads * kState);
        DeviceBuffer<float> head_norm(kVHeads * kState);
        DeviceBuffer<float> gated_recurrent(kVHeads * kState);
        DeviceBuffer<float> projected(kHidden);
        DeviceBuffer<float> residual(kHidden);
        DeviceBuffer<float> post_normalized(kHidden);
        DeviceBuffer<float> ffn_gate(kFfnInner);
        DeviceBuffer<float> ffn_up(kFfnInner);
        DeviceBuffer<float> ffn_activation(kFfnInner);
        DeviceBuffer<float> output(kHidden);
        DeviceBuffer<miinfer::Q8KDeviceBlock> q8(kFfnInner / 256);

        DeviceBuffer<std::byte> qkv_device(qkv_weight.byte_size);
        DeviceBuffer<std::byte> gate_device(gate_weight.byte_size);
        DeviceBuffer<std::byte> attn_norm_device(attn_norm_weight.byte_size);
        DeviceBuffer<std::byte> beta_device(beta_weight.byte_size);
        DeviceBuffer<std::byte> alpha_device(alpha_weight.byte_size);
        DeviceBuffer<std::byte> conv_device(conv_weight.byte_size);
        DeviceBuffer<std::byte> ssm_norm_device(ssm_norm_weight.byte_size);
        DeviceBuffer<std::byte> ssm_out_device(ssm_out_weight.byte_size);
        DeviceBuffer<std::byte> post_norm_device(post_norm_weight.byte_size);
        DeviceBuffer<std::byte> ffn_gate_device(ffn_gate_weight.byte_size);
        DeviceBuffer<std::byte> ffn_up_device(ffn_up_weight.byte_size);
        DeviceBuffer<std::byte> ffn_down_device(ffn_down_weight.byte_size);
        DeviceBuffer<float> dt_device(kVHeads);
        DeviceBuffer<float> a_device(kVHeads);

        const auto copy_tensor = [](const miinfer::GgufTensor& source, auto& destination) {
            upload(source.data, destination.get(), source.byte_size);
        };
        copy_tensor(qkv_weight, qkv_device);
        copy_tensor(gate_weight, gate_device);
        copy_tensor(attn_norm_weight, attn_norm_device);
        copy_tensor(beta_weight, beta_device);
        copy_tensor(alpha_weight, alpha_device);
        copy_tensor(conv_weight, conv_device);
        copy_tensor(ssm_norm_weight, ssm_norm_device);
        copy_tensor(ssm_out_weight, ssm_out_device);
        copy_tensor(post_norm_weight, post_norm_device);
        copy_tensor(ffn_gate_weight, ffn_gate_device);
        copy_tensor(ffn_up_weight, ffn_up_device);
        copy_tensor(ffn_down_weight, ffn_down_device);
        upload(dt_weight.data, dt_device.get(), dt_weight.byte_size);
        upload(a_weight.data, a_device.get(), a_weight.byte_size);
        MIINFER_HIP_CHECK(hipMemset(conv_history.get(), 0,
                                    4 * kChannels * sizeof(float)));
        const auto initial_state = read_f32(checkpoint(fixture, 0, "state_predelta-0"),
                                            kVHeads * kState * kState);
        upload(initial_state.data(), state.get(),
               kVHeads * kState * kState * sizeof(float));

        float maximum_error = 0.0F;
        for (const std::size_t position : {std::size_t{0}, std::size_t{1}}) {
            const auto host_input = read_f32(
                checkpoint(fixture, position, "model_input_embed"), kHidden);
            upload(host_input.data(), input.get(), host_input.size() * sizeof(float));

            miinfer::launch_qwen3_rms_norm(
                input.get(), reinterpret_cast<const float*>(attn_norm_device.get()),
                normalized.get(), kHidden, model.config().rms_epsilon);
            check_device("attn_norm", normalized.get(), kHidden,
                         checkpoint(fixture, position, "attn_norm-0"), 1.0e-3F,
                         maximum_error);

            project(qkv_weight, qkv_device.get(), normalized.get(), q8.get(), qkv.get(),
                    kChannels, kHidden);
            check_device("qkv", qkv.get(), kChannels,
                         checkpoint(fixture, position, "linear_attn_qkv_mixed-0"),
                         1.0F, maximum_error);
            project(gate_weight, gate_device.get(), normalized.get(), q8.get(), gate.get(),
                    kInner, kHidden);
            const auto host_gate = gemv(
                gate_weight, read_f32(checkpoint(fixture, position, "attn_norm-0"), kHidden),
                kInner, kHidden);
            check_values("gate", gate.get(), host_gate, 1.0F, maximum_error);
            miinfer::launch_qwen35_f32_gemv(
                reinterpret_cast<const float*>(beta_device.get()), normalized.get(),
                beta_raw.get(), kVHeads, kHidden);
            miinfer::launch_qwen35_f32_gemv(
                reinterpret_cast<const float*>(alpha_device.get()), normalized.get(),
                alpha_raw.get(), kVHeads, kHidden);
            miinfer::launch_qwen35_prepare_beta_decay(
                beta_raw.get(), alpha_raw.get(), dt_device.get(), a_device.get(),
                beta.get(), decay.get(), kVHeads);

            miinfer::launch_qwen35_conv_silu_split(
                qkv.get(), reinterpret_cast<const float*>(conv_device.get()),
                conv_history.get(), query.get(), key.get(), value.get(),
                static_cast<std::uint32_t>(position), 4, kChannels, 4);
            miinfer::launch_qwen35_head_l2_normalize(
                query.get(), query_norm.get(), kKHeads, kState);
            miinfer::launch_qwen35_head_l2_normalize(
                key.get(), key_norm.get(), kKHeads, kState);

            auto raw = read_f32(
                checkpoint(fixture, position, "conv_output_raw-0"), kChannels);
            silu(raw);
            std::vector<float> expected_query(raw.begin(), raw.begin() + kKHeads * kState);
            std::vector<float> expected_key(raw.begin() + kKHeads * kState,
                                             raw.begin() + 2 * kKHeads * kState);
            l2_norm(expected_query, kState);
            l2_norm(expected_key, kState);
            const auto query_error = [&] {
                std::vector<float> actual(expected_query.size());
                MIINFER_HIP_CHECK(hipMemcpy(actual.data(), query_norm.get(),
                                            actual.size() * sizeof(float), hipMemcpyDeviceToHost));
                return compare(actual, expected_query);
            }();
            const auto key_error = [&] {
                std::vector<float> actual(expected_key.size());
                MIINFER_HIP_CHECK(hipMemcpy(actual.data(), key_norm.get(),
                                            actual.size() * sizeof(float), hipMemcpyDeviceToHost));
                return compare(actual, expected_key);
            }();
            require_match("query L2", query_error, 2.0e-5F);
            require_match("key L2", key_error, 2.0e-5F);
            maximum_error = std::max({maximum_error, query_error.max_abs, key_error.max_abs});

            miinfer::launch_qwen35_deltanet_state_update(
                query_norm.get(), key_norm.get(), value.get(), beta.get(), decay.get(),
                state.get(), recurrent_output.get(), kKHeads, kVHeads, kState);
            check_device("recurrent_output", recurrent_output.get(), kVHeads * kState,
                         checkpoint(fixture, position, "attn_output-0"), 1.0e-2F,
                         maximum_error);
            check_device("state", state.get(), kVHeads * kState * kState,
                         checkpoint(fixture, position + 1, "state_predelta-0"), 1.0e-2F,
                         maximum_error);

            miinfer::launch_qwen3_head_rms_normalize(
                recurrent_output.get(), head_norm.get(), kVHeads, kState,
                model.config().rms_epsilon);
            miinfer::launch_qwen3_head_mul(
                head_norm.get(), reinterpret_cast<const float*>(ssm_norm_device.get()),
                gated_recurrent.get(), kVHeads, kState);
            const auto host_recurrent = read_f32(
                checkpoint(fixture, position, "attn_output-0"), kVHeads * kState);
            auto host_head_norm = host_recurrent;
            const std::vector<float> unit_weights(kState, 1.0F);
            rms_rows(host_recurrent, unit_weights, host_head_norm, kState);
            check_values("head_norm", head_norm.get(), host_head_norm, 1.0e-3F, maximum_error);
            auto host_head_scaled = host_head_norm;
            const auto host_ssm_norm = f32_values(ssm_norm_weight, kState);
            for (std::size_t i = 0; i < host_head_scaled.size(); ++i) {
                host_head_scaled[i] *= host_ssm_norm[i % kState];
            }
            check_values("head_scaled", gated_recurrent.get(), host_head_scaled,
                         1.0e-3F, maximum_error);
            miinfer::launch_qwen3_silu_mul(
                gate.get(), gated_recurrent.get(), gated_recurrent.get(), kVHeads * kState);
            check_device("gated_recurrent", gated_recurrent.get(), kVHeads * kState,
                         checkpoint(fixture, position, "final_output-0"), 5.0e-2F,
                         maximum_error);

            project(ssm_out_weight, ssm_out_device.get(), gated_recurrent.get(), q8.get(),
                    projected.get(), kHidden, kInner);
            miinfer::launch_qwen3_add(
                input.get(), projected.get(), residual.get(), kHidden);
            check_device("attention_residual", residual.get(), kHidden,
                         checkpoint(fixture, position, "attn_residual-0"), 2.0e-1F,
                         maximum_error);

            miinfer::launch_qwen3_rms_norm(
                residual.get(), reinterpret_cast<const float*>(post_norm_device.get()),
                post_normalized.get(), kHidden, model.config().rms_epsilon);
            check_device("post_attention_norm", post_normalized.get(), kHidden,
                         checkpoint(fixture, position, "attn_post_norm-0"), 2.0e-2F,
                         maximum_error);

            project(ffn_gate_weight, ffn_gate_device.get(), post_normalized.get(), q8.get(),
                    ffn_gate.get(), kFfnInner, kHidden);
            project(ffn_up_weight, ffn_up_device.get(), post_normalized.get(), q8.get(),
                    ffn_up.get(), kFfnInner, kHidden);
            miinfer::launch_qwen3_silu_mul(
                ffn_gate.get(), ffn_up.get(), ffn_activation.get(), kFfnInner);
            project(ffn_down_weight, ffn_down_device.get(), ffn_activation.get(), q8.get(),
                    output.get(), kHidden, kFfnInner);
            check_device("ffn_output", output.get(), kHidden,
                         checkpoint(fixture, position, "ffn_out-0"), 1.0F,
                         maximum_error);
            miinfer::launch_qwen3_add(
                residual.get(), output.get(), output.get(), kHidden);
            check_device("layer_output", output.get(), kHidden,
                         checkpoint(fixture, position, "l_out-0"), 1.0F,
                         maximum_error);

            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            std::cout << "position=" << position << " complete\n";
        }
        std::cout << "max_error=" << maximum_error << '\n'
                  << "M6-A20 qwen35 recurrent GPU layer PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A20 failed: " << error.what() << '\n';
        return 1;
    }
}
