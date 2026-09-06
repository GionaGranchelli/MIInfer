#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t bytes) { MIINFER_HIP_CHECK(hipMalloc(&p, bytes)); }
    ~Buffer() { if (p) (void)hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
};

struct Event {
    hipEvent_t p = nullptr;
    Event() { MIINFER_HIP_CHECK(hipEventCreate(&p)); }
    ~Event() { if (p) (void)hipEventDestroy(p); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

void run_q6k_benchmark(const miinfer::Qwen35Model& model, const std::string& tensor_name,
                       int warmups, int batches, int iters_per_batch) {
    const auto tensor = model.tensor(tensor_name);
    if (tensor.type() != miinfer::GgufTensorType::q6_k)
        throw std::runtime_error("expected Q6_K tensor: " + tensor_name);
    const auto dims = tensor.dimensions();
    const std::uint32_t columns = static_cast<std::uint32_t>(dims[0]);
    const std::uint32_t rows = static_cast<std::uint32_t>(dims[1]);

    std::cout << "\n=== Benchmarking Q6_K: " << tensor_name << " [" << columns << " x " << rows << "] ===\n";

    // Find the GgufTensor pointer in model.tensors()
    const miinfer::GgufTensor* raw_tensor = nullptr;
    for (const auto& t : model.tensors()) {
        if (t.name == tensor_name) { raw_tensor = &t; break; }
    }
    if (!raw_tensor) throw std::runtime_error("tensor not found in model: " + tensor_name);

    auto native = pack_q6k_wave_tensor(*raw_tensor);
    const std::size_t native_bytes = native.size() * sizeof(native[0]);
    const std::size_t canonical_bytes = tensor.bytes();

    Buffer d_canonical(canonical_bytes);
    Buffer d_native(native_bytes);
    Buffer d_input(columns * sizeof(float));
    Buffer d_q8(columns / 32 * sizeof(miinfer::Q8_1Block));
    Buffer d_out_canon(rows * sizeof(float));
    Buffer d_out_native(rows * sizeof(float));

    MIINFER_HIP_CHECK(hipMemcpy(d_canonical.p, tensor.data(), canonical_bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_native.p, native.data(), native_bytes, hipMemcpyHostToDevice));

    std::vector<float> h_input(columns);
    for (std::size_t i = 0; i < columns; ++i) h_input[i] = std::sin(float(i) * 0.017f);
    MIINFER_HIP_CHECK(hipMemcpy(d_input.p, h_input.data(), columns * sizeof(float), hipMemcpyHostToDevice));

    miinfer::launch_q8_1_quantize_f32(d_input.as<float>(), d_q8.as<miinfer::Q8_1Block>(), columns);

    // Correctness verification
    miinfer::launch_qwen3_q6_k_q8_1_mmvq(
        d_canonical.as<miinfer::Q6KDeviceBlock>(), d_q8.as<miinfer::Q8_1Block>(),
        d_out_canon.as<float>(), rows, columns);
    launch_q6k_wave_gemv(
        d_native.as<Q6KWaveTile>(), d_q8.as<miinfer::Q8_1Block>(),
        d_out_native.as<float>(), rows, columns);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_canon(rows), h_native(rows);
    MIINFER_HIP_CHECK(hipMemcpy(h_canon.data(), d_out_canon.p, rows * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(h_native.data(), d_out_native.p, rows * sizeof(float), hipMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (std::size_t i = 0; i < rows; ++i) {
        if (!std::isfinite(h_canon[i]) || !std::isfinite(h_native[i]))
            throw std::runtime_error("nonfinite value in output at " + std::to_string(i));
        max_err = std::max(max_err, std::fabs(h_canon[i] - h_native[i]));
    }
    std::cout << "Correctness: max_abs_diff = " << max_err << " (" << (max_err < 1e-4f ? "PASS" : "WARN") << ")\n";

    // Timing warmups
    for (int i = 0; i < warmups; ++i) {
        miinfer::launch_qwen3_q6_k_q8_1_mmvq(
            d_canonical.as<miinfer::Q6KDeviceBlock>(), d_q8.as<miinfer::Q8_1Block>(),
            d_out_canon.as<float>(), rows, columns);
        launch_q6k_wave_gemv(
            d_native.as<Q6KWaveTile>(), d_q8.as<miinfer::Q8_1Block>(),
            d_out_native.as<float>(), rows, columns);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    Event start_canon, stop_canon, start_nat, stop_nat;
    std::vector<float> canon_times, nat_times;

    for (int b = 0; b < batches; ++b) {
        MIINFER_HIP_CHECK(hipEventRecord(start_canon.p));
        for (int it = 0; it < iters_per_batch; ++it) {
            miinfer::launch_qwen3_q6_k_q8_1_mmvq(
                d_canonical.as<miinfer::Q6KDeviceBlock>(), d_q8.as<miinfer::Q8_1Block>(),
                d_out_canon.as<float>(), rows, columns);
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop_canon.p));

        MIINFER_HIP_CHECK(hipEventRecord(start_nat.p));
        for (int it = 0; it < iters_per_batch; ++it) {
            launch_q6k_wave_gemv(
                d_native.as<Q6KWaveTile>(), d_q8.as<miinfer::Q8_1Block>(),
                d_out_native.as<float>(), rows, columns);
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop_nat.p));

        MIINFER_HIP_CHECK(hipEventSynchronize(stop_nat.p));

        float ms_canon = 0, ms_nat = 0;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms_canon, start_canon.p, stop_canon.p));
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms_nat, start_nat.p, stop_nat.p));

        canon_times.push_back((ms_canon * 1000.0f) / float(iters_per_batch));
        nat_times.push_back((ms_nat * 1000.0f) / float(iters_per_batch));
    }

    std::sort(canon_times.begin(), canon_times.end());
    std::sort(nat_times.begin(), nat_times.end());

    const float med_canon = canon_times[canon_times.size() / 2];
    const float med_nat = nat_times[nat_times.size() / 2];
    const float speedup = med_canon / med_nat;

    std::cout << "Canonical Q6_K MMVQ median: " << med_canon << " us\n";
    std::cout << "Native Q6_K Wave64 median:  " << med_nat << " us\n";
    std::cout << "Speedup:                    " << speedup << "x (" << (speedup - 1.0f) * 100.0f << "%)\n";
}

void run_q5k_benchmark(const miinfer::Qwen35Model& model, const std::string& tensor_name,
                       int warmups, int batches, int iters_per_batch) {
    const auto tensor = model.tensor(tensor_name);
    if (tensor.type() != miinfer::GgufTensorType::q5_k)
        throw std::runtime_error("expected Q5_K tensor: " + tensor_name);
    const auto dims = tensor.dimensions();
    const std::uint32_t columns = static_cast<std::uint32_t>(dims[0]);
    const std::uint32_t rows = static_cast<std::uint32_t>(dims[1]);

    std::cout << "\n=== Benchmarking Q5_K: " << tensor_name << " [" << columns << " x " << rows << "] ===\n";

    const miinfer::GgufTensor* raw_tensor = nullptr;
    for (const auto& t : model.tensors()) {
        if (t.name == tensor_name) { raw_tensor = &t; break; }
    }
    if (!raw_tensor) throw std::runtime_error("tensor not found in model: " + tensor_name);

    auto native = pack_q5k_wave_tensor(*raw_tensor);
    const std::size_t native_bytes = native.size() * sizeof(native[0]);
    const std::size_t canonical_bytes = tensor.bytes();

    Buffer d_canonical(canonical_bytes);
    Buffer d_native(native_bytes);
    Buffer d_input(columns * sizeof(float));
    Buffer d_q8(columns / 32 * sizeof(miinfer::Q8_1Block));
    Buffer d_out_canon(rows * sizeof(float));
    Buffer d_out_native(rows * sizeof(float));

    MIINFER_HIP_CHECK(hipMemcpy(d_canonical.p, tensor.data(), canonical_bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_native.p, native.data(), native_bytes, hipMemcpyHostToDevice));

    std::vector<float> h_input(columns);
    for (std::size_t i = 0; i < columns; ++i) h_input[i] = std::sin(float(i) * 0.017f);
    MIINFER_HIP_CHECK(hipMemcpy(d_input.p, h_input.data(), columns * sizeof(float), hipMemcpyHostToDevice));

    miinfer::launch_q8_1_quantize_f32(d_input.as<float>(), d_q8.as<miinfer::Q8_1Block>(), columns);

    // Correctness verification
    miinfer::launch_qwen3_q5_k_q8_1_mmvq(
        d_canonical.as<miinfer::Q5KDeviceBlock>(), d_q8.as<miinfer::Q8_1Block>(),
        d_out_canon.as<float>(), rows, columns);
    launch_q5k_wave_gemv(
        d_native.as<Q5KWaveTile>(), d_q8.as<miinfer::Q8_1Block>(),
        d_out_native.as<float>(), rows, columns);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_canon(rows), h_native(rows);
    MIINFER_HIP_CHECK(hipMemcpy(h_canon.data(), d_out_canon.p, rows * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(h_native.data(), d_out_native.p, rows * sizeof(float), hipMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (std::size_t i = 0; i < rows; ++i) {
        if (!std::isfinite(h_canon[i]) || !std::isfinite(h_native[i]))
            throw std::runtime_error("nonfinite value in output at " + std::to_string(i));
        max_err = std::max(max_err, std::fabs(h_canon[i] - h_native[i]));
    }
    std::cout << "Correctness: max_abs_diff = " << max_err << " (" << (max_err < 1e-4f ? "PASS" : "WARN") << ")\n";

    // Timing warmups
    for (int i = 0; i < warmups; ++i) {
        miinfer::launch_qwen3_q5_k_q8_1_mmvq(
            d_canonical.as<miinfer::Q5KDeviceBlock>(), d_q8.as<miinfer::Q8_1Block>(),
            d_out_canon.as<float>(), rows, columns);
        launch_q5k_wave_gemv(
            d_native.as<Q5KWaveTile>(), d_q8.as<miinfer::Q8_1Block>(),
            d_out_native.as<float>(), rows, columns);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    Event start_canon, stop_canon, start_nat, stop_nat;
    std::vector<float> canon_times, nat_times;

    for (int b = 0; b < batches; ++b) {
        MIINFER_HIP_CHECK(hipEventRecord(start_canon.p));
        for (int it = 0; it < iters_per_batch; ++it) {
            miinfer::launch_qwen3_q5_k_q8_1_mmvq(
                d_canonical.as<miinfer::Q5KDeviceBlock>(), d_q8.as<miinfer::Q8_1Block>(),
                d_out_canon.as<float>(), rows, columns);
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop_canon.p));

        MIINFER_HIP_CHECK(hipEventRecord(start_nat.p));
        for (int it = 0; it < iters_per_batch; ++it) {
            launch_q5k_wave_gemv(
                d_native.as<Q5KWaveTile>(), d_q8.as<miinfer::Q8_1Block>(),
                d_out_native.as<float>(), rows, columns);
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop_nat.p));

        MIINFER_HIP_CHECK(hipEventSynchronize(stop_nat.p));

        float ms_canon = 0, ms_nat = 0;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms_canon, start_canon.p, stop_canon.p));
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms_nat, start_nat.p, stop_nat.p));

        canon_times.push_back((ms_canon * 1000.0f) / float(iters_per_batch));
        nat_times.push_back((ms_nat * 1000.0f) / float(iters_per_batch));
    }

    std::sort(canon_times.begin(), canon_times.end());
    std::sort(nat_times.begin(), nat_times.end());

    const float med_canon = canon_times[canon_times.size() / 2];
    const float med_nat = nat_times[nat_times.size() / 2];
    const float speedup = med_canon / med_nat;

    std::cout << "Canonical Q5_K MMVQ median: " << med_canon << " us\n";
    std::cout << "Native Q5_K Wave64 median:  " << med_nat << " us\n";
    std::cout << "Speedup:                    " << speedup << "x (" << (speedup - 1.0f) * 100.0f << "%)\n";
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::cerr << "usage: miinfer-kquant-layout-bench MODEL.gguf [--warmups N] [--batches N] [--iters N]\n";
        return 2;
    }
    const char* model_path = argv[1];
    int warmups = 20;
    int batches = 50;
    int iters = 10;

    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);

    std::cout << "Loading model: " << model_path << '\n';
    const auto model = miinfer::Qwen35Model::load(model_path);

    // Benchmark Q6_K recurrent attn_qkv projection
    run_q6k_benchmark(model, "blk.0.attn_qkv.weight", warmups, batches, iters);

    // Benchmark Q5_K recurrent ssm_out projection
    run_q5k_benchmark(model, "blk.0.ssm_out.weight", warmups, batches, iters);

    // Benchmark Q6_K recurrent ffn_down projection
    run_q6k_benchmark(model, "blk.0.ffn_down.weight", warmups, batches, iters);

    // Benchmark Q6_K full-attn v projection
    run_q6k_benchmark(model, "blk.3.attn_v.weight", warmups, batches, iters);

    std::cout << "\nK-quant layout laboratory benchmark completed successfully.\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}
