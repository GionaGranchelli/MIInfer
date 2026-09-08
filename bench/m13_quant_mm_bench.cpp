#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<unsigned, 5> kBatches{64, 128, 256, 512, 2048};

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

template <typename Fn>
double measure(Fn&& fn) {
    for (int i = 0; i < 2; ++i) fn();
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    Event start;
    Event stop;
    std::vector<double> samples;
    for (int i = 0; i < 7; ++i) {
        MIINFER_HIP_CHECK(hipEventRecord(start.value, hipStreamPerThread));
        fn();
        MIINFER_HIP_CHECK(hipEventRecord(stop.value, hipStreamPerThread));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop.value));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start.value, stop.value));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

const miinfer::GgufTensor* find_tensor(
    const miinfer::Qwen35Model& model, miinfer::GgufTensorType type,
    std::uint64_t columns, bool prefer_ffn_down) {
    const miinfer::GgufTensor* fallback = nullptr;
    for (const auto& tensor : model.tensors()) {
        if (tensor.type != type || tensor.dimensions.size() != 2
            || tensor.dimensions[0] != columns || tensor.dimensions[0] % 1024 != 0
            || tensor.dimensions[1] % 4 != 0) {
            continue;
        }
        if (tensor.name.ends_with(".ffn_down.weight")) return &tensor;
        if (type == miinfer::GgufTensorType::q6_k
            && tensor.name.ends_with(".attn_qkv.weight")) return &tensor;
        if (fallback == nullptr && !prefer_ffn_down) fallback = &tensor;
    }
    return fallback;
}

template <typename Tile, typename Pack, typename Launch4, typename LaunchMm>
void run_case(const miinfer::GgufTensor& tensor, Pack pack, Launch4 launch4, LaunchMm launch_mm,
              const char* label, std::vector<std::string>& results) {
    const unsigned columns = static_cast<unsigned>(tensor.dimensions[0]);
    const unsigned rows = static_cast<unsigned>(tensor.dimensions[1]);
    const unsigned input_blocks = columns / 32;
    const auto packed = pack(tensor);
    const std::size_t packed_bytes = packed.size() * sizeof(Tile);
    Buffer weights(packed_bytes);
    Buffer input(static_cast<std::size_t>(kBatches.back()) * input_blocks
                 * sizeof(miinfer::Q8_1Block));
    Buffer input_f32(static_cast<std::size_t>(kBatches.back()) * columns * sizeof(float));
    Buffer baseline(static_cast<std::size_t>(kBatches.back()) * rows * sizeof(float));
    Buffer candidate(static_cast<std::size_t>(kBatches.back()) * rows * sizeof(float));
    MIINFER_HIP_CHECK(hipMemcpy(weights.pointer, packed.data(), packed_bytes,
                                hipMemcpyHostToDevice));

    std::vector<float> input_host(static_cast<std::size_t>(kBatches.back()) * columns);
    for (unsigned batch = 0; batch < kBatches.back(); ++batch) {
        for (unsigned column = 0; column < columns; ++column) {
            input_host[static_cast<std::size_t>(batch) * columns + column] =
                std::sin(0.0017F * static_cast<float>(column)
                         + 0.071F * static_cast<float>(batch));
        }
    }
    MIINFER_HIP_CHECK(hipMemcpy(
        input_f32.pointer, input_host.data(), input_host.size() * sizeof(float),
        hipMemcpyHostToDevice));
    for (unsigned batch = 0; batch < kBatches.back(); ++batch) {
        miinfer::launch_q8_1_quantize_f32(
            input_f32.as<float>() + static_cast<std::size_t>(batch) * columns,
            input.as<miinfer::Q8_1Block>() + static_cast<std::size_t>(batch) * input_blocks,
            columns, hipStreamPerThread);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    std::cout << (results.empty() ? "" : ",")
              << "{\"type\":\"" << label << "\",\"tensor\":\"" << tensor.name
              << "\",\"rows\":" << rows << ",\"columns\":" << columns
              << ",\"packed_bytes\":" << packed_bytes << ",\"batches\":[";
    bool first = true;
    double max_error = 0.0;
    for (const unsigned batch : kBatches) {
        const double baseline_us = measure([&] {
            for (unsigned offset = 0; offset < batch; offset += 4) {
                launch4(weights.as<Tile>(),
                        input.as<miinfer::Q8_1Block>() + static_cast<std::size_t>(offset) * input_blocks,
                        baseline.as<float>() + static_cast<std::size_t>(offset) * rows,
                        rows, columns, hipStreamPerThread);
            }
        });
        const double mm_us = measure([&] {
            launch_mm(weights.as<Tile>(), input.as<miinfer::Q8_1Block>(), candidate.as<float>(),
                      rows, columns, batch, input_blocks, hipStreamPerThread);
        });
        if (batch == kBatches.front()) {
            for (unsigned offset = 0; offset < batch; offset += 4) {
                launch4(weights.as<Tile>(),
                        input.as<miinfer::Q8_1Block>() + static_cast<std::size_t>(offset) * input_blocks,
                        baseline.as<float>() + static_cast<std::size_t>(offset) * rows,
                        rows, columns, hipStreamPerThread);
            }
            launch_mm(weights.as<Tile>(), input.as<miinfer::Q8_1Block>(), candidate.as<float>(),
                      rows, columns, batch, input_blocks, hipStreamPerThread);
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> baseline_host(static_cast<std::size_t>(batch) * rows);
            std::vector<float> candidate_host(baseline_host.size());
            MIINFER_HIP_CHECK(hipMemcpy(baseline_host.data(), baseline.pointer,
                                        baseline_host.size() * sizeof(float), hipMemcpyDeviceToHost));
            MIINFER_HIP_CHECK(hipMemcpy(candidate_host.data(), candidate.pointer,
                                        candidate_host.size() * sizeof(float), hipMemcpyDeviceToHost));
            for (std::size_t i = 0; i < baseline_host.size(); ++i) {
                max_error = std::max(max_error, static_cast<double>(std::abs(
                    baseline_host[i] - candidate_host[i])));
            }
        }
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"batch\":" << batch
                  << ",\"baseline_b4_us\":" << baseline_us
                  << ",\"quant_mm_us\":" << mm_us
                  << ",\"speedup_vs_b4\":" << baseline_us / mm_us << '}';
    }
    std::cout << "],\"max_b4_vs_mm_error\":" << max_error << "}";
    results.emplace_back(label);
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error(
        "usage: miinfer-m13-quant-mm-bench MODEL.gguf");
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);
    const auto model = miinfer::Qwen35Model::load(argv[1]);
    const auto* q4 = find_tensor(model, miinfer::GgufTensorType::q4_k, 17408, true);
    const auto* q6 = find_tensor(model, miinfer::GgufTensorType::q6_k, 5120, false);
    if (q4 == nullptr) throw std::runtime_error("exact Q4_K FFN-down tensor not found");

    std::vector<std::string> results;
    std::cout << std::fixed << std::setprecision(6) << "{\"cases\":[";
    run_case<Q4KWaveTile>(
        *q4,
        [](const miinfer::GgufTensor& tensor) { return pack_q4k_wave_tensor(tensor); },
        [](const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
           unsigned rows, unsigned columns, hipStream_t stream) {
            launch_q4k_wave_gemv_batched4(w, x, y, rows, columns, stream);
        },
        [](const Q4KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
           unsigned rows, unsigned columns, unsigned batch, unsigned stride, hipStream_t stream) {
            launch_q4k_wave_gemv_batched_mm(w, x, y, rows, columns, batch, stride, stream);
        },
        "Q4_K", results);
    if (q6 != nullptr) {
        run_case<Q6KWaveTile>(
            *q6,
            [](const miinfer::GgufTensor& tensor) { return pack_q6k_wave_tensor(tensor); },
            [](const Q6KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
               unsigned rows, unsigned columns, hipStream_t stream) {
                launch_q6k_wave_gemv_batched4(w, x, y, rows, columns, stream);
            },
            [](const Q6KWaveTile* w, const miinfer::Q8_1Block* x, float* y,
               unsigned rows, unsigned columns, unsigned batch, unsigned stride, hipStream_t stream) {
                launch_q6k_wave_gemv_batched_mm(w, x, y, rows, columns, batch, stride, stream);
            },
            "Q6_K", results);
    }
    std::cout << "]}\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
