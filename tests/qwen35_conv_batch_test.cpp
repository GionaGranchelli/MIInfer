#include "miinfer/hip_check.hpp"
#include "miinfer/device_validation.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        MIINFER_HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_ != nullptr) (void) hipFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    T* get() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

bool close(float a, float b) {
    return std::fabs(a - b) <= 2.0e-6F * std::fmax(1.0F, std::fabs(b));
}

bool close_mmq(float a, float b) {
    return std::fabs(a - b) <= 1.0e-5F * std::fmax(1.0F, std::fabs(b));
}

bool close_q6_mmq(float a, float b) {
    return std::fabs(a - b) <= 3.0e-5F * std::fmax(1.0F, std::fabs(b));
}

} // namespace

int main() {
    miinfer::DeviceInfo device;
    std::string device_error;
    if (!miinfer::validate_gfx906_device(-1, device, device_error)) {
        std::cerr << "Qwen35 convolution/MMQ GPU test unavailable: " << device_error << '\n';
        return 1;
    }
    constexpr std::uint32_t kChannels = 10;
    constexpr std::uint32_t kTokens = 128;
    constexpr std::uint32_t kHistory = 4;
    constexpr std::uint32_t kQ = kChannels / 5;
    constexpr std::uint32_t kV = 3 * kQ;
    std::vector<float> qkv(kTokens * kChannels), weights(kChannels * kHistory);
    for (std::size_t i = 0; i < qkv.size(); ++i) qkv[i] = static_cast<float>((i % 11) - 5) * 0.125F;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(i % 5 + 1) * 0.05F;

    DeviceBuffer<float> d_qkv(qkv.size()), d_weights(weights.size());
    DeviceBuffer<float> d_history_single(kHistory * kChannels), d_history_batch(kHistory * kChannels);
    DeviceBuffer<float> d_query_single(kTokens * kQ), d_key_single(kTokens * kQ), d_value_single(kTokens * kV);
    DeviceBuffer<float> d_query_batch(kTokens * kQ), d_key_batch(kTokens * kQ), d_value_batch(kTokens * kV);
    DeviceBuffer<float> d_query_single_norm(kTokens * kQ), d_key_single_norm(kTokens * kQ);
    DeviceBuffer<float> d_query_batch_norm(kTokens * kQ), d_key_batch_norm(kTokens * kQ);
    constexpr std::uint32_t kRows = 3;
    std::vector<float> projection_a(kRows * kChannels), projection_b(kRows * kChannels);
    for (std::size_t i = 0; i < projection_a.size(); ++i) {
        projection_a[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.0625F;
        projection_b[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.09375F;
    }
    DeviceBuffer<float> d_projection_a(projection_a.size()), d_projection_b(projection_b.size());
    DeviceBuffer<float> d_projection_a_single(kTokens * kRows), d_projection_b_single(kTokens * kRows);
    DeviceBuffer<float> d_projection_a_batch(kTokens * kRows), d_projection_b_batch(kTokens * kRows);
    constexpr std::uint32_t kNormElements = 1024;
    std::vector<float> norm_input(kTokens * kNormElements), norm_weights(kNormElements, 0.75F);
    for (std::size_t i = 0; i < norm_input.size(); ++i) {
        norm_input[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125F;
    }
    DeviceBuffer<float> d_norm_input(norm_input.size()), d_norm_weights(norm_weights.size());
    DeviceBuffer<float> d_norm_single(norm_input.size()), d_norm_batch(norm_input.size());
    DeviceBuffer<float> d_residual(kTokens * kNormElements), d_projection(kTokens * kNormElements);
    DeviceBuffer<float> d_residual_single(kTokens * kNormElements), d_post_norm_single(kTokens * kNormElements);
    DeviceBuffer<float> d_residual_batch(kTokens * kNormElements), d_post_norm_batch(kTokens * kNormElements);
    DeviceBuffer<miinfer::M23Q8_1MmqBlock> d_mmq_q8(kTokens * (kNormElements / 128));
    constexpr std::uint32_t kMmqRows = 128, kMmqColumns = 1024;
    std::vector<miinfer::Q4KDeviceBlock> mmq_weights(kMmqRows * (kMmqColumns / 256));
    for (std::size_t i = 0; i < mmq_weights.size(); ++i) {
        auto& block = mmq_weights[i];
        block.d = __float2half(0.125F);
        block.dmin = __float2half(0.0F);
        std::fill(std::begin(block.scales), std::end(block.scales), static_cast<std::uint8_t>(1));
        for (std::size_t q = 0; q < std::size(block.qs); ++q) block.qs[q] = static_cast<std::uint8_t>((q + i) & 0xFFU);
    }
    DeviceBuffer<miinfer::Q4KDeviceBlock> d_mmq_weights(mmq_weights.size());
    DeviceBuffer<float> d_mmq_output(kTokens * kMmqRows);
    miinfer::GgufTensor mmq_q4_tensor{};
    mmq_q4_tensor.name = "synthetic_m23_q4";
    mmq_q4_tensor.type = miinfer::GgufTensorType::q4_k;
    mmq_q4_tensor.dimensions = {kMmqColumns, kMmqRows};
    mmq_q4_tensor.byte_size = mmq_weights.size() * sizeof(mmq_weights.front());
    mmq_q4_tensor.data = reinterpret_cast<const std::byte*>(mmq_weights.data());
    const auto mmq_q4_repacked = pack_q4k_mmq_tensor(mmq_q4_tensor);
    DeviceBuffer<Q4KMmqTile> d_mmq_q4_repacked(mmq_q4_repacked.size());
    DeviceBuffer<float> d_mmq_q4_repacked_output(kTokens * kMmqRows);
    std::vector<miinfer::Q6KDeviceBlock> mmq_q6_weights(kMmqRows * (kMmqColumns / 256));
    for (std::size_t i = 0; i < mmq_q6_weights.size(); ++i) {
        auto& block = mmq_q6_weights[i];
        for (std::size_t q = 0; q < std::size(block.ql); ++q) {
            block.ql[q] = static_cast<std::uint8_t>(q * 37U + i * 11U + 3U);
        }
        for (std::size_t q = 0; q < std::size(block.qh); ++q) {
            block.qh[q] = static_cast<std::uint8_t>(q * 19U + i * 7U + 5U);
        }
        std::fill(std::begin(block.scales), std::end(block.scales), static_cast<std::int8_t>(1));
        block.d = __float2half(0.0625F);
    }
    miinfer::GgufTensor mmq_q6_tensor{};
    mmq_q6_tensor.name = "synthetic_m23_q6";
    mmq_q6_tensor.type = miinfer::GgufTensorType::q6_k;
    mmq_q6_tensor.dimensions = {kMmqColumns, kMmqRows};
    mmq_q6_tensor.byte_size = mmq_q6_weights.size() * sizeof(mmq_q6_weights.front());
    mmq_q6_tensor.data = reinterpret_cast<const std::byte*>(mmq_q6_weights.data());
    const auto mmq_q6_repacked = pack_q6k_mmq_tensor(mmq_q6_tensor);
    DeviceBuffer<miinfer::Q6KDeviceBlock> d_mmq_q6_weights(mmq_q6_weights.size());
    DeviceBuffer<float> d_mmq_q6_output(kTokens * kMmqRows), d_mmq_q6_output_canonical(kTokens * kMmqRows);
    DeviceBuffer<Q6KMmqTile> d_mmq_q6_repacked(mmq_q6_repacked.size());
    DeviceBuffer<float> d_mmq_q6_repacked_output(kTokens * kMmqRows);
    DeviceBuffer<miinfer::Q8_1Block> d_mmq_q8_canonical(kTokens * (kMmqColumns / miinfer::kQ8_1BlockSize));
    DeviceBuffer<float> d_mmq_output_canonical(kTokens * kMmqRows);
    MIINFER_HIP_CHECK(hipMemcpy(d_qkv.get(), qkv.data(), d_qkv.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_weights.get(), weights.data(), d_weights.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_projection_a.get(), projection_a.data(), d_projection_a.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_projection_b.get(), projection_b.data(), d_projection_b.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_norm_input.get(), norm_input.data(), d_norm_input.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_norm_weights.get(), norm_weights.data(), d_norm_weights.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_residual.get(), d_norm_input.get(), d_norm_input.bytes(), hipMemcpyDeviceToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_mmq_weights.get(), mmq_weights.data(), d_mmq_weights.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_mmq_q4_repacked.get(), mmq_q4_repacked.data(),
                                d_mmq_q4_repacked.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_mmq_q6_weights.get(), mmq_q6_weights.data(), d_mmq_q6_weights.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_mmq_q6_repacked.get(), mmq_q6_repacked.data(),
                                d_mmq_q6_repacked.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_projection.get(), d_norm_input.get(), d_norm_input.bytes(), hipMemcpyDeviceToDevice));
    MIINFER_HIP_CHECK(hipMemset(d_history_single.get(), 0, d_history_single.bytes()));
    MIINFER_HIP_CHECK(hipMemset(d_history_batch.get(), 0, d_history_batch.bytes()));

    for (std::uint32_t token = 0; token < kTokens; ++token) {
        miinfer::launch_qwen35_conv_silu_split(
            d_qkv.get() + token * kChannels, d_weights.get(), d_history_single.get(),
            d_query_single.get() + token * kQ, d_key_single.get() + token * kQ,
            d_value_single.get() + token * kV, token, kHistory, kChannels, kHistory);
    }
    miinfer::launch_qwen35_conv_silu_split_batch(
        d_qkv.get(), d_weights.get(), d_history_batch.get(), d_query_batch.get(), d_key_batch.get(),
        d_value_batch.get(), 0, kTokens, kHistory, kChannels, kHistory);
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        miinfer::launch_qwen35_dual_head_l2_normalize(
            d_query_single.get() + token * kQ, d_key_single.get() + token * kQ,
            d_query_single_norm.get() + token * kQ, d_key_single_norm.get() + token * kQ,
            1, kQ);
    }
    miinfer::launch_qwen35_dual_head_l2_normalize_batch(
        d_query_batch.get(), d_key_batch.get(), d_query_batch_norm.get(), d_key_batch_norm.get(),
        kTokens, 1, kQ);
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        miinfer::launch_qwen35_f32_dual_gemv(
            d_projection_a.get(), d_projection_b.get(), d_qkv.get() + token * kChannels,
            d_projection_a_single.get() + token * kRows, d_projection_b_single.get() + token * kRows,
            kRows, kChannels);
    }
    miinfer::launch_qwen35_f32_dual_gemm_batch(
        d_projection_a.get(), d_projection_b.get(), d_qkv.get(),
        d_projection_a_batch.get(), d_projection_b_batch.get(), kTokens, kRows, kChannels);
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        miinfer::launch_qwen3_rms_norm(
            d_norm_input.get() + token * kNormElements, d_norm_weights.get(),
            d_norm_single.get() + token * kNormElements, kNormElements, 1.0e-6F);
    }
    miinfer::launch_qwen3_rms_norm_batch(
        d_norm_input.get(), d_norm_weights.get(), d_norm_batch.get(), kTokens, kNormElements, 1.0e-6F);
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        miinfer::launch_qwen3_fused_add_rms_norm(
            d_residual.get() + token * kNormElements, d_projection.get() + token * kNormElements,
            d_norm_weights.get(), d_residual_single.get() + token * kNormElements,
            d_post_norm_single.get() + token * kNormElements, kNormElements, 1.0e-6F);
    }
    miinfer::launch_qwen3_fused_add_rms_norm_batch(
        d_residual.get(), d_projection.get(), d_norm_weights.get(), d_residual_batch.get(),
        d_post_norm_batch.get(), kTokens, kNormElements, 1.0e-6F);
    miinfer::launch_m23_q8_1_mmq_quantize(
        d_norm_input.get(), d_mmq_q8.get(), kTokens, kNormElements);
    miinfer::launch_m23_q4_k_q8_1_mmq(
        d_mmq_weights.get(), d_mmq_q8.get(), d_mmq_output.get(), kMmqRows, kMmqColumns, kTokens);
    launch_m23_q4k_repacked_mmq(
        d_mmq_q4_repacked.get(), d_mmq_q8.get(), d_mmq_q4_repacked_output.get(),
        kMmqRows, kMmqColumns, kTokens);
    miinfer::launch_m23_q6_k_q8_1_mmq(
        d_mmq_q6_weights.get(), d_mmq_q8.get(), d_mmq_q6_output.get(), kMmqRows, kMmqColumns, kTokens);
    launch_m23_q6k_repacked_mmq(
        d_mmq_q6_repacked.get(), d_mmq_q8.get(), d_mmq_q6_repacked_output.get(),
        kMmqRows, kMmqColumns, kTokens);
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        miinfer::launch_q8_1_quantize_f32(
            d_norm_input.get() + token * kMmqColumns,
            d_mmq_q8_canonical.get() + token * (kMmqColumns / miinfer::kQ8_1BlockSize), kMmqColumns);
        miinfer::launch_qwen3_q4_k_q8_1_mmvq(
            d_mmq_weights.get(),
            d_mmq_q8_canonical.get() + token * (kMmqColumns / miinfer::kQ8_1BlockSize),
            d_mmq_output_canonical.get() + token * kMmqRows, kMmqRows, kMmqColumns);
        miinfer::launch_qwen3_q6_k_q8_1_mmvq(
            d_mmq_q6_weights.get(),
            d_mmq_q8_canonical.get() + token * (kMmqColumns / miinfer::kQ8_1BlockSize),
            d_mmq_q6_output_canonical.get() + token * kMmqRows, kMmqRows, kMmqColumns);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> single(kTokens * kChannels), batch(kTokens * kChannels);
    auto copy_result = [&](float* query, float* key, float* value, std::vector<float>& out) {
        std::vector<float> q(kTokens * kQ), k(kTokens * kQ), v(kTokens * kV);
        MIINFER_HIP_CHECK(hipMemcpy(q.data(), query, q.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(k.data(), key, k.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v.data(), value, v.size() * sizeof(float), hipMemcpyDeviceToHost));
        for (std::uint32_t token = 0; token < kTokens; ++token) {
            std::copy_n(q.data() + token * kQ, kQ, out.data() + token * kChannels);
            std::copy_n(k.data() + token * kQ, kQ, out.data() + token * kChannels + kQ);
            std::copy_n(v.data() + token * kV, kV, out.data() + token * kChannels + 2 * kQ);
        }
    };
    copy_result(d_query_single.get(), d_key_single.get(), d_value_single.get(), single);
    copy_result(d_query_batch.get(), d_key_batch.get(), d_value_batch.get(), batch);
    std::vector<float> history_single(kHistory * kChannels), history_batch(kHistory * kChannels);
    MIINFER_HIP_CHECK(hipMemcpy(history_single.data(), d_history_single.get(), d_history_single.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(history_batch.data(), d_history_batch.get(), d_history_batch.bytes(), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < single.size(); ++i) if (!close(single[i], batch[i])) return 1;
    for (std::size_t i = 0; i < history_single.size(); ++i) if (!close(history_single[i], history_batch[i])) return 1;
    std::vector<float> norm_single(kTokens * 2 * kQ), norm_batch(kTokens * 2 * kQ);
    MIINFER_HIP_CHECK(hipMemcpy(norm_single.data(), d_query_single_norm.get(), kTokens * kQ * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(norm_single.data() + kTokens * kQ, d_key_single_norm.get(), kTokens * kQ * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(norm_batch.data(), d_query_batch_norm.get(), kTokens * kQ * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(norm_batch.data() + kTokens * kQ, d_key_batch_norm.get(), kTokens * kQ * sizeof(float), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < norm_single.size(); ++i) if (!close(norm_single[i], norm_batch[i])) return 1;
    std::vector<float> projection_single(kTokens * 2 * kRows), projection_batch(kTokens * 2 * kRows);
    MIINFER_HIP_CHECK(hipMemcpy(projection_single.data(), d_projection_a_single.get(), kTokens * kRows * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(projection_single.data() + kTokens * kRows, d_projection_b_single.get(), kTokens * kRows * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(projection_batch.data(), d_projection_a_batch.get(), kTokens * kRows * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(projection_batch.data() + kTokens * kRows, d_projection_b_batch.get(), kTokens * kRows * sizeof(float), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < projection_single.size(); ++i) if (!close(projection_single[i], projection_batch[i])) return 1;
    std::vector<float> norm_single_rms(norm_input.size()), norm_batch_rms(norm_input.size());
    MIINFER_HIP_CHECK(hipMemcpy(norm_single_rms.data(), d_norm_single.get(), d_norm_single.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(norm_batch_rms.data(), d_norm_batch.get(), d_norm_batch.bytes(), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < norm_single_rms.size(); ++i) if (!close(norm_single_rms[i], norm_batch_rms[i])) return 1;
    std::vector<float> residual_single(norm_input.size()), residual_batch(norm_input.size());
    std::vector<float> post_norm_single(norm_input.size()), post_norm_batch(norm_input.size());
    MIINFER_HIP_CHECK(hipMemcpy(residual_single.data(), d_residual_single.get(), d_residual_single.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(residual_batch.data(), d_residual_batch.get(), d_residual_batch.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(post_norm_single.data(), d_post_norm_single.get(), d_post_norm_single.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(post_norm_batch.data(), d_post_norm_batch.get(), d_post_norm_batch.bytes(), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < residual_single.size(); ++i) {
        if (!close(residual_single[i], residual_batch[i]) || !close(post_norm_single[i], post_norm_batch[i])) return 1;
    }
    std::vector<miinfer::M23Q8_1MmqBlock> mmq_q8(kTokens * (kNormElements / 128));
    MIINFER_HIP_CHECK(hipMemcpy(mmq_q8.data(), d_mmq_q8.get(), d_mmq_q8.bytes(), hipMemcpyDeviceToHost));
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        for (std::uint32_t block = 0; block < kNormElements / 128; ++block) {
            const auto& actual = mmq_q8[block * kTokens + token];
            for (std::uint32_t group = 0; group < 4; ++group) {
                float max_abs = 0.0F, sum = 0.0F;
                int qsum = 0;
                for (std::uint32_t lane = 0; lane < 32; ++lane) {
                    const float value = norm_input[token * kNormElements + block * 128 + group * 32 + lane];
                    max_abs = std::max(max_abs, std::fabs(value));
                    sum += value;
                }
                const float d = max_abs == 0.0F ? 0.0F : max_abs / 127.0F;
                for (std::uint32_t lane = 0; lane < 32; ++lane) {
                    const float value = norm_input[token * kNormElements + block * 128 + group * 32 + lane];
                    const auto q = static_cast<std::int8_t>(d == 0.0F ? 0 : std::rint(value / d));
                    if (actual.qs[group * 32 + lane] != q) return 1;
                    qsum += q;
                }
                const float qsum_scaled = __half2float(__float2half_rn(d)) * static_cast<float>(qsum);
                if (!close(actual.d[group], d) || !close(actual.s[group], sum)
                    || !close(actual.qsum_scaled[group], qsum_scaled)) return 1;
            }
        }
    }
    std::vector<float> mmq_output(kTokens * kMmqRows), mmq_output_canonical(kTokens * kMmqRows);
    MIINFER_HIP_CHECK(hipMemcpy(mmq_output.data(), d_mmq_output.get(), d_mmq_output.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(mmq_output_canonical.data(), d_mmq_output_canonical.get(),
                                d_mmq_output_canonical.bytes(), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < mmq_output.size(); ++i) {
        if (!close_mmq(mmq_output[i], mmq_output_canonical[i])) return 1;
    }
    std::vector<float> mmq_q4_repacked_output(kTokens * kMmqRows);
    MIINFER_HIP_CHECK(hipMemcpy(mmq_q4_repacked_output.data(), d_mmq_q4_repacked_output.get(),
                                d_mmq_q4_repacked_output.bytes(), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < mmq_q4_repacked_output.size(); ++i) {
        if (!close_mmq(mmq_q4_repacked_output[i], mmq_output_canonical[i])) return 1;
    }
    std::vector<float> mmq_q6_output(kTokens * kMmqRows), mmq_q6_output_canonical(kTokens * kMmqRows);
    MIINFER_HIP_CHECK(hipMemcpy(mmq_q6_output.data(), d_mmq_q6_output.get(), d_mmq_q6_output.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(mmq_q6_output_canonical.data(), d_mmq_q6_output_canonical.get(),
                                d_mmq_q6_output_canonical.bytes(), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < mmq_q6_output.size(); ++i) {
        if (!close_q6_mmq(mmq_q6_output[i], mmq_q6_output_canonical[i])) {
            std::cerr << "Q6 MMQ mismatch index=" << i << " actual=" << mmq_q6_output[i]
                      << " expected=" << mmq_q6_output_canonical[i] << '\n';
            return 1;
        }
    }
    std::vector<float> mmq_q6_repacked_output(kTokens * kMmqRows);
    MIINFER_HIP_CHECK(hipMemcpy(mmq_q6_repacked_output.data(), d_mmq_q6_repacked_output.get(),
                                d_mmq_q6_repacked_output.bytes(), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < mmq_q6_repacked_output.size(); ++i) {
        if (!close_q6_mmq(mmq_q6_repacked_output[i], mmq_q6_output_canonical[i])) {
            std::cerr << "Q6 repacked MMQ mismatch index=" << i << " actual="
                      << mmq_q6_repacked_output[i] << " expected=" << mmq_q6_output_canonical[i]
                      << '\n';
            return 1;
        }
    }
    constexpr std::uint32_t kPostTokens = 3;
    constexpr std::uint32_t kPostBase = 1;
    constexpr std::uint32_t kPostCapacity = 8;
    constexpr std::uint32_t kPostQueryHeads = 2;
    constexpr std::uint32_t kPostKvHeads = 1;
    constexpr std::uint32_t kPostHeadDim = 256;
    constexpr std::uint32_t kPostQfullRows =
        (kPostQueryHeads * 2 + kPostKvHeads) * kPostHeadDim;
    std::vector<float> post_qfull(kPostTokens * kPostQfullRows);
    std::vector<float> post_value(kPostTokens * kPostKvHeads * kPostHeadDim);
    std::vector<float> post_q_norm(kPostHeadDim), post_k_norm(kPostHeadDim);
    for (std::size_t i = 0; i < post_qfull.size(); ++i) {
        post_qfull[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.015625F;
    }
    for (std::size_t i = 0; i < post_value.size(); ++i) {
        post_value[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.03125F;
    }
    for (std::size_t i = 0; i < post_q_norm.size(); ++i) {
        post_q_norm[i] = 0.75F + static_cast<float>(i % 7) * 0.03125F;
        post_k_norm[i] = 0.625F + static_cast<float>(i % 5) * 0.046875F;
    }
    DeviceBuffer<float> d_post_qfull(post_qfull.size()), d_post_value(post_value.size());
    DeviceBuffer<float> d_post_q_norm(post_q_norm.size()), d_post_k_norm(post_k_norm.size());
    DeviceBuffer<float> d_post_query(kPostTokens * kPostQueryHeads * kPostHeadDim);
    DeviceBuffer<float> d_post_gate(kPostTokens * kPostQueryHeads * kPostHeadDim);
    DeviceBuffer<float> d_post_keys(kPostCapacity * kPostKvHeads * kPostHeadDim);
    DeviceBuffer<float> d_post_values(kPostCapacity * kPostKvHeads * kPostHeadDim);
    MIINFER_HIP_CHECK(hipMemcpy(d_post_qfull.get(), post_qfull.data(), d_post_qfull.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_post_value.get(), post_value.data(), d_post_value.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_post_q_norm.get(), post_q_norm.data(), d_post_q_norm.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_post_k_norm.get(), post_k_norm.data(), d_post_k_norm.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemset(d_post_keys.get(), 0, d_post_keys.bytes()));
    MIINFER_HIP_CHECK(hipMemset(d_post_values.get(), 0, d_post_values.bytes()));
    miinfer::launch_qwen35_fused_q_split_norm_rope_batch(
        d_post_qfull.get(), d_post_q_norm.get(), d_post_query.get(), d_post_gate.get(),
        kPostTokens, kPostBase, kPostQueryHeads, kPostKvHeads, kPostHeadDim, 1.0F, 1.0e-6F);
    miinfer::launch_qwen35_fused_k_norm_rope_kv_store_batch(
        d_post_qfull.get(), d_post_value.get(), d_post_k_norm.get(), d_post_keys.get(),
        d_post_values.get(), kPostTokens, kPostBase, kPostCapacity, kPostQueryHeads,
        kPostKvHeads, kPostHeadDim, 1.0F, 1.0e-6F);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> post_query(d_post_query.bytes() / sizeof(float));
    std::vector<float> post_gate(post_query.size());
    std::vector<float> post_keys(d_post_keys.bytes() / sizeof(float));
    std::vector<float> post_values(d_post_values.bytes() / sizeof(float));
    MIINFER_HIP_CHECK(hipMemcpy(post_query.data(), d_post_query.get(), d_post_query.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(post_gate.data(), d_post_gate.get(), d_post_gate.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(post_keys.data(), d_post_keys.get(), d_post_keys.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(post_values.data(), d_post_values.get(), d_post_values.bytes(), hipMemcpyDeviceToHost));
    auto post_rope = [](float x0, float x1, std::uint32_t local, std::uint32_t position) {
        if (local >= 32) return std::pair<float, float>{x0, x1};
        constexpr int sections[] = {11, 11, 10, 0};
        const int sector = static_cast<int>(local % 32);
        const float angle = ((sector % 3 == 1 && sector < 3 * sections[1])
                             || (sector % 3 == 2 && sector < 3 * sections[2])
                             || (sector % 3 == 0 && sector < 3 * sections[0]))
            ? static_cast<float>(position) : 0.0F;
        return std::pair<float, float>{x0 * std::cos(angle) - x1 * std::sin(angle),
                                       x0 * std::sin(angle) + x1 * std::cos(angle)};
    };
    for (std::uint32_t token = 0; token < kPostTokens; ++token) {
        const std::size_t full_base = static_cast<std::size_t>(token) * kPostQfullRows;
        for (std::uint32_t head = 0; head < kPostQueryHeads; ++head) {
            const std::size_t source = full_base + head * 2 * kPostHeadDim;
            const std::size_t output =
                (static_cast<std::size_t>(token) * kPostQueryHeads + head) * kPostHeadDim;
            float sum = 0.0F;
            for (std::uint32_t dim = 0; dim < kPostHeadDim; ++dim) {
                sum += post_qfull[source + dim] * post_qfull[source + dim];
            }
            const float inv_rms = 1.0F / std::sqrt(sum / kPostHeadDim + 1.0e-6F);
            for (std::uint32_t local = 0; local < 32; ++local) {
                const auto rotated = post_rope(
                    post_qfull[source + local] * inv_rms * post_q_norm[local],
                    post_qfull[source + local + 32] * inv_rms * post_q_norm[local + 32],
                    local, kPostBase + token);
                if (!close(post_query[output + local], rotated.first)
                    || !close(post_query[output + local + 32], rotated.second)) return 1;
            }
            for (std::uint32_t dim = 64; dim < kPostHeadDim; ++dim) {
                const float expected = post_qfull[source + dim] * inv_rms * post_q_norm[dim];
                if (!close(post_query[output + dim], expected)) return 1;
            }
            for (std::uint32_t dim = 0; dim < kPostHeadDim; ++dim) {
                if (!close(post_gate[output + dim], post_qfull[source + kPostHeadDim + dim])) return 1;
            }
        }
        const std::size_t key_source = full_base + kPostQueryHeads * 2 * kPostHeadDim;
        float key_sum = 0.0F;
        for (std::uint32_t dim = 0; dim < kPostHeadDim; ++dim) {
            key_sum += post_qfull[key_source + dim] * post_qfull[key_source + dim];
        }
        const float key_inv_rms = 1.0F / std::sqrt(key_sum / kPostHeadDim + 1.0e-6F);
        const std::size_t cache_base =
            static_cast<std::size_t>(kPostBase + token) * kPostHeadDim;
        for (std::uint32_t local = 0; local < 32; ++local) {
            const auto rotated = post_rope(
                post_qfull[key_source + local] * key_inv_rms * post_k_norm[local],
                post_qfull[key_source + local + 32] * key_inv_rms * post_k_norm[local + 32],
                local, kPostBase + token);
            if (!close(post_keys[cache_base + local], rotated.first)
                || !close(post_keys[cache_base + local + 32], rotated.second)) return 1;
        }
        for (std::uint32_t dim = 64; dim < kPostHeadDim; ++dim) {
            const float expected = post_qfull[key_source + dim] * key_inv_rms * post_k_norm[dim];
            if (!close(post_keys[cache_base + dim], expected)) return 1;
        }
        for (std::uint32_t dim = 0; dim < kPostHeadDim; ++dim) {
            if (!close(post_values[cache_base + dim], post_value[token * kPostHeadDim + dim])) return 1;
        }
    }
    DeviceBuffer<__half> d_post_keys_f16(kPostCapacity * kPostKvHeads * kPostHeadDim);
    DeviceBuffer<__half> d_post_values_f16(kPostCapacity * kPostKvHeads * kPostHeadDim);
    MIINFER_HIP_CHECK(hipMemset(d_post_keys_f16.get(), 0, d_post_keys_f16.bytes()));
    MIINFER_HIP_CHECK(hipMemset(d_post_values_f16.get(), 0, d_post_values_f16.bytes()));
    miinfer::launch_qwen35_fused_k_norm_rope_kv_store_batch_f16(
        d_post_qfull.get(), d_post_value.get(), d_post_k_norm.get(), d_post_keys_f16.get(),
        d_post_values_f16.get(), kPostTokens, kPostBase, kPostCapacity, kPostQueryHeads,
        kPostKvHeads, kPostHeadDim, 1.0F, 1.0e-6F);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<__half> post_keys_f16(d_post_keys_f16.bytes() / sizeof(__half));
    std::vector<__half> post_values_f16(d_post_values_f16.bytes() / sizeof(__half));
    MIINFER_HIP_CHECK(hipMemcpy(post_keys_f16.data(), d_post_keys_f16.get(),
                                d_post_keys_f16.bytes(), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(post_values_f16.data(), d_post_values_f16.get(),
                                d_post_values_f16.bytes(), hipMemcpyDeviceToHost));
    for (std::uint32_t token = 0; token < kPostTokens; ++token) {
        const std::size_t cache_base =
            static_cast<std::size_t>(kPostBase + token) * kPostHeadDim;
        for (std::uint32_t dim = 0; dim < kPostHeadDim; ++dim) {
            if (__half2float(post_keys_f16[cache_base + dim])
                != __half2float(__float2half(post_keys[cache_base + dim]))) return 1;
            if (__half2float(post_values_f16[cache_base + dim])
                != __half2float(__float2half(post_value[token * kPostHeadDim + dim]))) return 1;
        }
    }
    constexpr std::uint32_t kAttentionTokens = 8;
    constexpr std::uint32_t kAttentionBase = 3;
    constexpr std::uint32_t kAttentionCapacity = 16;
    constexpr std::uint32_t kQueryHeads = 2;
    constexpr std::uint32_t kKvHeads = 1;
    constexpr std::uint32_t kHeadDim = 256;
    std::vector<float> attention_q(kAttentionTokens * kQueryHeads * kHeadDim);
    std::vector<float> attention_gate(attention_q.size());
    std::vector<float> attention_keys(kAttentionCapacity * kKvHeads * kHeadDim);
    std::vector<float> attention_values(attention_keys.size());
    for (std::size_t i = 0; i < attention_q.size(); ++i) {
        attention_q[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.03125F;
        attention_gate[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.0625F;
    }
    for (std::size_t i = 0; i < attention_keys.size(); ++i) {
        attention_keys[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.015625F;
        attention_values[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.0234375F;
    }
    DeviceBuffer<float> d_attention_q(attention_q.size()), d_attention_gate(attention_gate.size());
    DeviceBuffer<float> d_attention_keys(attention_keys.size()), d_attention_values(attention_values.size());
    DeviceBuffer<float> d_attention_output(attention_q.size());
    MIINFER_HIP_CHECK(hipMemcpy(d_attention_q.get(), attention_q.data(), d_attention_q.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_attention_gate.get(), attention_gate.data(), d_attention_gate.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_attention_keys.get(), attention_keys.data(), d_attention_keys.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_attention_values.get(), attention_values.data(), d_attention_values.bytes(), hipMemcpyHostToDevice));
    miinfer::launch_qwen35_tiled_online_attention_batch(
        d_attention_q.get(), d_attention_keys.get(), d_attention_values.get(),
        d_attention_gate.get(), d_attention_output.get(), kAttentionTokens,
        kAttentionBase, kAttentionCapacity, kQueryHeads, kKvHeads, kHeadDim,
        1.0F / std::sqrt(static_cast<float>(kHeadDim)));
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> attention_output(attention_q.size());
    MIINFER_HIP_CHECK(hipMemcpy(attention_output.data(), d_attention_output.get(),
                                d_attention_output.bytes(), hipMemcpyDeviceToHost));
    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> attention_expected(attention_q.size());
    for (std::uint32_t token = 0; token < kAttentionTokens; ++token) {
        const std::uint32_t cache_length = kAttentionBase + token + 1;
        for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
            const std::size_t query_base =
                (static_cast<std::size_t>(token) * kQueryHeads + head) * kHeadDim;
            std::vector<float> scores(cache_length);
            float maximum = -1.0e30F;
            for (std::uint32_t position = 0; position < cache_length; ++position) {
                float score = 0.0F;
                const std::size_t cache_base = static_cast<std::size_t>(position) * kHeadDim;
                for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
                    score += attention_q[query_base + dim] * attention_keys[cache_base + dim];
                }
                scores[position] = score * attention_scale;
                maximum = std::max(maximum, scores[position]);
            }
            float denominator = 0.0F;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
                float expected = 0.0F;
                for (std::uint32_t position = 0; position < cache_length; ++position) {
                    expected += (scores[position] / denominator)
                        * attention_values[static_cast<std::size_t>(position) * kHeadDim + dim];
                }
                const float gate = attention_gate[query_base + dim];
                expected /= 1.0F + std::exp(-gate);
                attention_expected[query_base + dim] = expected;
                const float actual = attention_output[query_base + dim];
                if (std::fabs(actual - expected) > 5.0e-4F * std::fmax(1.0F, std::fabs(expected))) {
                    std::cerr << "wide attention batch mismatch token=" << token
                              << " head=" << head << " dim=" << dim
                              << " actual=" << actual << " expected=" << expected << '\n';
                    return 1;
                }
            }
        }
    }
    std::vector<__half> attention_keys_f16(attention_keys.size()), attention_values_f16(attention_values.size());
    for (std::size_t i = 0; i < attention_keys.size(); ++i) {
        attention_keys_f16[i] = __float2half(attention_keys[i]);
        attention_values_f16[i] = __float2half(attention_values[i]);
    }
    DeviceBuffer<__half> d_attention_keys_f16(attention_keys_f16.size());
    DeviceBuffer<__half> d_attention_values_f16(attention_values_f16.size());
    DeviceBuffer<float> d_attention_output_f16(attention_q.size());
    MIINFER_HIP_CHECK(hipMemcpy(d_attention_keys_f16.get(), attention_keys_f16.data(),
                                d_attention_keys_f16.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_attention_values_f16.get(), attention_values_f16.data(),
                                d_attention_values_f16.bytes(), hipMemcpyHostToDevice));
    miinfer::launch_qwen35_tiled_online_attention_batch_f16(
        d_attention_q.get(), d_attention_keys_f16.get(), d_attention_values_f16.get(),
        d_attention_gate.get(), d_attention_output_f16.get(), kAttentionTokens,
        kAttentionBase, kAttentionCapacity, kQueryHeads, kKvHeads, kHeadDim,
        attention_scale);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> attention_output_f16(attention_q.size());
    MIINFER_HIP_CHECK(hipMemcpy(attention_output_f16.data(), d_attention_output_f16.get(),
                                d_attention_output_f16.bytes(), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < attention_output_f16.size(); ++i) {
        if (std::fabs(attention_output_f16[i] - attention_expected[i])
            > 6.0e-3F * std::fmax(1.0F, std::fabs(attention_expected[i]))) {
            std::cerr << "wide attention FP16 KV mismatch index=" << i
                      << " actual=" << attention_output_f16[i]
                      << " expected=" << attention_expected[i] << '\n';
            return 1;
        }
    }
    std::cout << "qwen35 B128 wide RMS, convolution, Q/K normalization, dual projection, Q8_1 MMQ pack, and causal attention=PASS\n";
}
