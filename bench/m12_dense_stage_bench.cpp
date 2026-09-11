#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/m12_dense_stage.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kRows = 5120;
constexpr int kColumns = 17408;
constexpr int kQ4BlockSize = 256;
constexpr int kBlocksPerRow = kColumns / kQ4BlockSize;
constexpr std::array<int, 4> kBatches{64, 128, 256, 512};

struct Buffer {
    void* pointer = nullptr;

    explicit Buffer(std::size_t bytes) { MIINFER_HIP_CHECK(hipMalloc(&pointer, bytes)); }
    ~Buffer() { if (pointer != nullptr) (void)hipFree(pointer); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    template <typename T>
    T* as() const { return static_cast<T*>(pointer); }
};

struct Event {
    hipEvent_t value = nullptr;
    Event() { MIINFER_HIP_CHECK(hipEventCreate(&value)); }
    ~Event() { if (value != nullptr) (void)hipEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <typename Fn>
double measure(Fn&& fn, int warmup = 2, int iterations = 7) {
    for (int i = 0; i < warmup; ++i) fn();
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    Event start;
    Event stop;
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        MIINFER_HIP_CHECK(hipEventRecord(start.value, hipStreamPerThread));
        fn();
        MIINFER_HIP_CHECK(hipEventRecord(stop.value, hipStreamPerThread));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop.value));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start.value, stop.value));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    return median(std::move(samples));
}

int q4_scale(const miinfer::Q4KDeviceBlock& block, int group) {
    if (group < 4) return block.scales[group] & 63;
    return (block.scales[group + 4] & 0x0f)
           | ((block.scales[group - 4] >> 6) << 4);
}

int q4_minimum(const miinfer::Q4KDeviceBlock& block, int group) {
    if (group < 4) return block.scales[group + 4] & 63;
    return (block.scales[group + 4] >> 4)
           | ((block.scales[group] >> 6) << 4);
}

float q4_value(const miinfer::Q4KDeviceBlock& block, int index) {
    const int group = index / 32;
    const int q_index = (index / 64) * 32 + index % 32;
    const auto packed = block.qs[q_index];
    const int quantized = (group & 1) == 0 ? packed & 0x0f : packed >> 4;
    return __half2float(block.d) * static_cast<float>(q4_scale(block, group) * quantized)
           - __half2float(block.dmin) * static_cast<float>(q4_minimum(block, group));
}

void check_hipblas(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: "
                                 + std::to_string(static_cast<int>(status)));
    }
}

void gemm(hipblasHandle_t handle, const __half* weights, const __half* input,
          __half* output, int batch) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    // Row-major C[B,M] = A[B,K] * W^T[K,M], exposed as column-major.
    check_hipblas(hipblasGemmEx(
        handle, HIPBLAS_OP_N, HIPBLAS_OP_N, batch, kRows, kColumns,
        &alpha, input, HIP_R_16F, batch, weights, HIP_R_16F, kColumns,
        &beta, output, HIP_R_16F, batch, HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT), "hipblasGemmEx");
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error(
        "usage: miinfer-m12-dense-stage-bench MODEL.gguf");

    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) {
        throw std::runtime_error(error);
    }
    const auto model = miinfer::Qwen35Model::load(argv[1]);
    const miinfer::GgufTensor* selected = nullptr;
    for (const auto& tensor : model.tensors()) {
        if (tensor.type == miinfer::GgufTensorType::q4_k
            && tensor.dimensions == std::vector<std::uint64_t>{kColumns, kRows}
            && tensor.name.ends_with(".ffn_down.weight")) {
            selected = &tensor;
            break;
        }
    }
    if (selected == nullptr) throw std::runtime_error("exact Q4_K FFN Down tensor not found");

    const auto tensor = model.tensor(selected->name);
    const auto* source_host = reinterpret_cast<const miinfer::Q4KDeviceBlock*>(tensor.data());
    const std::size_t canonical_bytes = tensor.bytes();
    const std::size_t dense_bytes = static_cast<std::size_t>(kRows) * kColumns * sizeof(__half);
    const std::size_t max_input_bytes = static_cast<std::size_t>(kBatches.back()) * kColumns * sizeof(__half);
    const std::size_t max_output_bytes = static_cast<std::size_t>(kBatches.back()) * kRows * sizeof(__half);
    const std::size_t mmq_blocks = kColumns / 128;

    Buffer source(canonical_bytes);
    Buffer dense(dense_bytes);
    Buffer resident_dense(dense_bytes);
    Buffer input(max_input_bytes);
    Buffer output(max_output_bytes);
    Buffer input_f32(static_cast<std::size_t>(kBatches.back()) * kColumns * sizeof(float));
    Buffer baseline_q8(static_cast<std::size_t>(kBatches.back())
                       * (kColumns / miinfer::kQ8_1BlockSize) * sizeof(miinfer::Q8_1Block));
    Buffer baseline_output(max_output_bytes * 2);
    const auto mmq_packed = pack_q4k_mmq_tensor(*selected);
    Buffer mmq_weights(mmq_packed.size() * sizeof(mmq_packed[0]));
    Buffer mmq_output(max_output_bytes * 2);
    std::array<std::unique_ptr<Buffer>, kBatches.size()> mmq_inputs;
    MIINFER_HIP_CHECK(hipMemcpy(source.pointer, tensor.data(), canonical_bytes,
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(mmq_weights.pointer, mmq_packed.data(),
                                mmq_packed.size() * sizeof(mmq_packed[0]),
                                hipMemcpyHostToDevice));

    std::vector<__half> input_host(static_cast<std::size_t>(kBatches.back()) * kColumns);
    std::vector<float> input_f32_host(input_host.size());
    for (int batch = 0; batch < kBatches.back(); ++batch) {
        for (int column = 0; column < kColumns; ++column) {
            const float value = std::sin(0.0017F * static_cast<float>(column)
                                         + 0.071F * static_cast<float>(batch));
            input_f32_host[static_cast<std::size_t>(batch) * kColumns + column] = value;
            input_host[static_cast<std::size_t>(batch) * kColumns + column] = __float2half_rn(value);
        }
    }
    MIINFER_HIP_CHECK(hipMemcpy(input.pointer, input_host.data(), max_input_bytes,
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(input_f32.pointer, input_f32_host.data(),
                                input_f32_host.size() * sizeof(float), hipMemcpyHostToDevice));

    const auto packed = pack_q4k_wave_down(*selected);
    Buffer packed_device(packed.size() * sizeof(packed[0]));
    MIINFER_HIP_CHECK(hipMemcpy(packed_device.pointer, packed.data(),
                                packed.size() * sizeof(packed[0]), hipMemcpyHostToDevice));

    const std::size_t q8_stride = kColumns / miinfer::kQ8_1BlockSize;
    for (int batch = 0; batch < kBatches.back(); ++batch) {
        miinfer::launch_q8_1_quantize_f32(
            input_f32.as<float>() + static_cast<std::size_t>(batch) * kColumns,
            baseline_q8.as<miinfer::Q8_1Block>() + static_cast<std::size_t>(batch) * q8_stride,
            kColumns, hipStreamPerThread);
    }
    for (std::size_t index = 0; index < kBatches.size(); ++index) {
        const int batch = kBatches[index];
        mmq_inputs[index] = std::make_unique<Buffer>(
            static_cast<std::size_t>(batch) * mmq_blocks * sizeof(miinfer::M23Q8_1MmqBlock));
        miinfer::launch_m23_q8_1_mmq_quantize(
            input_f32.as<float>(), mmq_inputs[index]->as<miinfer::M23Q8_1MmqBlock>(),
            batch, kColumns, hipStreamPerThread);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    const double repack_us = measure([&] {
        miinfer::launch_m12_q4k_to_fp16(
            source.as<miinfer::Q4KDeviceBlock>(), dense.as<__half>(), kRows, kColumns,
            hipStreamPerThread);
    }, 2, 7);
    const double resident_repack_us = measure([&] {
        launch_m24_q4k_mmq_to_fp16(
            mmq_weights.as<Q4KMmqTile>(), resident_dense.as<__half>(), kRows, kColumns,
            hipStreamPerThread);
    }, 2, 7);

    std::vector<__half> dense_rows(static_cast<std::size_t>(kColumns) * 3);
    double max_repack_error = 0.0;
    double max_resident_repack_error = 0.0;
    for (int sample = 0; sample < 3; ++sample) {
        const int row = sample == 0 ? 0 : sample == 1 ? 37 : kRows - 1;
        MIINFER_HIP_CHECK(hipMemcpy(
            dense_rows.data() + static_cast<std::size_t>(sample) * kColumns,
            dense.as<__half>() + static_cast<std::size_t>(row) * kColumns,
            static_cast<std::size_t>(kColumns) * sizeof(__half), hipMemcpyDeviceToHost));
        for (int column = 0; column < kColumns; ++column) {
            const auto& block = source_host[static_cast<std::size_t>(row) * kBlocksPerRow
                                           + column / kQ4BlockSize];
            max_repack_error = std::max(max_repack_error, static_cast<double>(std::abs(
                __half2float(dense_rows[static_cast<std::size_t>(sample) * kColumns + column])
                - q4_value(block, column % kQ4BlockSize))));
        }
        std::vector<__half> resident_row(kColumns);
        MIINFER_HIP_CHECK(hipMemcpy(
            resident_row.data(),
            resident_dense.as<__half>() + static_cast<std::size_t>(row) * kColumns,
            static_cast<std::size_t>(kColumns) * sizeof(__half), hipMemcpyDeviceToHost));
        for (int column = 0; column < kColumns; ++column) {
            const auto& block = source_host[static_cast<std::size_t>(row) * kBlocksPerRow
                                           + column / kQ4BlockSize];
            max_resident_repack_error = std::max(max_resident_repack_error, static_cast<double>(std::abs(
                __half2float(resident_row[column])
                - q4_value(block, column % kQ4BlockSize))));
        }
    }

    hipblasHandle_t handle = nullptr;
    check_hipblas(hipblasCreate(&handle), "hipblasCreate");
    check_hipblas(hipblasSetStream(handle, hipStreamPerThread), "hipblasSetStream");

    std::cout << std::fixed << std::setprecision(6)
              << "{\"tensor\":\"" << tensor.name() << "\",\"rows\":" << kRows
              << ",\"columns\":" << kColumns
              << ",\"dense_bytes\":" << dense_bytes
              << ",\"mmq_packed_bytes\":"
              << mmq_packed.size() * sizeof(mmq_packed[0])
              << ",\"repack_us\":" << repack_us
              << ",\"resident_repack_us\":" << resident_repack_us
              << ",\"max_repack_error\":" << max_repack_error
              << ",\"max_resident_repack_error\":" << max_resident_repack_error
              << ",\"batches\":[";

    bool first = true;
    double max_mmq_vs_gemm_error = 0.0;
    for (std::size_t batch_index = 0; batch_index < kBatches.size(); ++batch_index) {
        const int batch = kBatches[batch_index];
        const double gemm_us = measure([&] {
            gemm(handle, resident_dense.as<__half>(), input.as<__half>(), output.as<__half>(), batch);
        });
        const double mmq_us = measure([&] {
            launch_m23_q4k_repacked_mmq(
                mmq_weights.as<Q4KMmqTile>(),
                mmq_inputs[batch_index]->as<miinfer::M23Q8_1MmqBlock>(),
                mmq_output.as<float>(), kRows, kColumns, batch, hipStreamPerThread);
        });
        const double baseline_us = measure([&] {
            for (int offset = 0; offset < batch; offset += 4) {
                launch_q4k_wave_gemv_batched4(
                    packed_device.as<Q4KWaveTile>(),
                    baseline_q8.as<miinfer::Q8_1Block>() + static_cast<std::size_t>(offset) * q8_stride,
                    baseline_output.as<float>() + static_cast<std::size_t>(offset) * kRows,
                    kRows, kColumns, hipStreamPerThread);
            }
        });
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"batch\":" << batch
                  << ",\"baseline_b4_us\":" << baseline_us
                  << ",\"resident_mmq_us\":" << mmq_us
                  << ",\"gemm_us\":" << gemm_us
                  << ",\"repack_plus_gemm_us\":" << resident_repack_us + gemm_us
                  << ",\"resident_mmq_speedup_vs_repack_plus_gemm\":"
                  << (resident_repack_us + gemm_us) / mmq_us
                  << ",\"speedup_vs_b4\":" << baseline_us / (resident_repack_us + gemm_us)
                  << '}';
    }
    std::cout << "]}\n";

    gemm(handle, resident_dense.as<__half>(), input.as<__half>(), output.as<__half>(), kBatches[0]);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<__half> output_host(static_cast<std::size_t>(kBatches[0]) * kRows);
    MIINFER_HIP_CHECK(hipMemcpy(output_host.data(), output.pointer,
                                output_host.size() * sizeof(__half), hipMemcpyDeviceToHost));
    launch_m23_q4k_repacked_mmq(
        mmq_weights.as<Q4KMmqTile>(), mmq_inputs[0]->as<miinfer::M23Q8_1MmqBlock>(),
        mmq_output.as<float>(), kRows, kColumns, kBatches[0], hipStreamPerThread);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> mmq_output_host(static_cast<std::size_t>(kBatches[0]) * kRows);
    MIINFER_HIP_CHECK(hipMemcpy(mmq_output_host.data(), mmq_output.pointer,
                                mmq_output_host.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < mmq_output_host.size(); ++i) {
        if (!std::isfinite(mmq_output_host[i])) {
            throw std::runtime_error("resident MMQ produced non-finite output");
        }
    }
    for (std::size_t i = 0; i < mmq_output_host.size(); ++i) {
        max_mmq_vs_gemm_error = std::max(max_mmq_vs_gemm_error, static_cast<double>(std::abs(
            mmq_output_host[i] - __half2float(output_host[i]))));
    }
    double max_gemm_error = 0.0;
    double max_gemm_relative_error = 0.0;
    for (int batch = 0; batch < 2; ++batch) {
        for (int row : {0, 37, kRows - 1}) {
            float expected = 0.0F;
            for (int column = 0; column < kColumns; ++column) {
                const auto& block = source_host[static_cast<std::size_t>(row) * kBlocksPerRow
                                               + column / kQ4BlockSize];
                expected += q4_value(block, column % kQ4BlockSize)
                            * __half2float(input_host[static_cast<std::size_t>(batch) * kColumns + column]);
            }
            const double error = std::abs(
                __half2float(output_host[static_cast<std::size_t>(batch) * kRows + row]) - expected);
            max_gemm_error = std::max(max_gemm_error, error);
            max_gemm_relative_error = std::max(
                max_gemm_relative_error, error / std::max(1.0, std::abs(static_cast<double>(expected))));
        }
    }
    check_hipblas(hipblasDestroy(handle), "hipblasDestroy");
    std::cerr << "m12 correctness: max_repack_error=" << max_repack_error
              << " max_resident_repack_error=" << max_resident_repack_error
              << " max_gemm_error=" << max_gemm_error
              << " max_gemm_relative_error=" << max_gemm_relative_error
              << " max_mmq_vs_gemm_error=" << max_mmq_vs_gemm_error << '\n';
    if (max_repack_error > 0.02
        || (max_gemm_error > 2.0 && max_gemm_relative_error > 0.01)) {
        throw std::runtime_error("dense staging correctness check failed");
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
