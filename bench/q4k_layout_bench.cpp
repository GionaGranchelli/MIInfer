#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/q4k_wave_layout.hpp"
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
constexpr int rows = 5120, columns = 17408, blocks = columns / 256;
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
}

int main(int argc, char** argv) try {
    if (argc != 2 && !(argc == 4 && std::string(argv[2]) == "--json-output"))
        throw std::runtime_error("usage: miinfer-q4k-layout-bench MODEL [--json-output FILE]");
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);
    const auto model = miinfer::Qwen35Model::load(argv[1]);
    std::size_t down_count = 0;
    const miinfer::GgufTensor* selected = nullptr;
    for (const auto& t : model.tensors()) {
        if (t.name.ends_with(".ffn_down.weight") && t.type == miinfer::GgufTensorType::q4_k &&
            t.dimensions == std::vector<std::uint64_t>{columns, rows}) { if (!selected) selected = &t; ++down_count; }
    }
    if (!selected) throw std::runtime_error("model has no exact-shape Q4_K FFN Down");
    const auto tensor = model.tensor(selected->name);
    if (tensor.type() != miinfer::GgufTensorType::q4_k ||
        tensor.dimensions() != std::vector<std::uint64_t>{columns, rows})
        throw std::runtime_error("expected Q4_K FFN Down 5120 x 17408");
    const auto* source = reinterpret_cast<const miinfer::Q4KDeviceBlock*>(tensor.data());
    std::vector<miinfer::Q4KExpandedDeviceBlock> expanded(rows * blocks);
    for (std::size_t b = 0; b < expanded.size(); ++b) {
        auto& dst = expanded[b];
        const auto& src = source[b];
        dst.d = src.d; dst.dmin = src.dmin;
        for (int g = 0; g < 8; ++g) {
            dst.scales[g] = g < 4 ? src.scales[g] & 63 :
                (src.scales[g + 4] & 15) | ((src.scales[g - 4] >> 6) << 4);
            dst.minimums[g] = g < 4 ? src.scales[g + 4] & 63 :
                (src.scales[g + 4] >> 4) | ((src.scales[g] >> 6) << 4);
        }
        for (int i = 0; i < 128; ++i) {
            dst.qs[2*i] = src.qs[i] & 15;
            dst.qs[2*i+1] = src.qs[i] >> 4;
        }
    }
    const std::size_t expanded_bytes = expanded.size() * sizeof(expanded[0]);
    const auto native = pack_q4k_wave_down(*selected);
    const std::size_t native_bytes = native.size() * sizeof(native[0]);
    Buffer canonical(tensor.bytes()), packed(expanded_bytes);
    Buffer wave(native_bytes), out2(rows*sizeof(float));
    MIINFER_HIP_CHECK(hipMemcpy(wave.p, native.data(), native_bytes, hipMemcpyHostToDevice));
    Buffer input(columns * sizeof(float)), q8(columns / 32 * sizeof(miinfer::Q8_1Block));
    Buffer out0(rows * sizeof(float)), out1(rows * sizeof(float));
    std::vector<float> x(columns);
    for (int i = 0; i < columns; ++i) x[i] = std::sin(float(i)*0.017F);
    MIINFER_HIP_CHECK(hipMemcpy(canonical.p, tensor.data(), tensor.bytes(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(packed.p, expanded.data(), expanded_bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(input.p, x.data(), x.size()*sizeof(float), hipMemcpyHostToDevice));
    miinfer::launch_q8_1_quantize_f32(input.as<float>(), q8.as<miinfer::Q8_1Block>(), columns);
    const auto launch = [&](int variant) {
        if (variant == 0) miinfer::launch_qwen3_q4_k_q8_1_mmvq_lds_decoded_metadata(
            canonical.as<miinfer::Q4KDeviceBlock>(), q8.as<miinfer::Q8_1Block>(), out0.as<float>(), rows, columns);
        else if (variant == 1) miinfer::launch_qwen3_q4_k_q8_1_mmvq_expanded(
            packed.as<miinfer::Q4KExpandedDeviceBlock>(), q8.as<miinfer::Q8_1Block>(), out1.as<float>(), rows, columns);
        else launch_q4k_wave_down(wave.as<Q4KWaveTile>(), q8.as<miinfer::Q8_1Block>(), out2.as<float>());
    };
    for (int i = 0; i < 50; ++i) { launch(0); launch(1); launch(2); }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> a(rows), b(rows);
    MIINFER_HIP_CHECK(hipMemcpy(a.data(), out0.p, rows*sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(b.data(), out1.p, rows*sizeof(float), hipMemcpyDeviceToHost));
    float max_error = 0;
    for (int i = 0; i < rows; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite output");
        max_error = std::max(max_error, std::abs(a[i]-b[i]));
    }
    if (max_error > 1e-4F) throw std::runtime_error("baseline arithmetic mismatch");
    MIINFER_HIP_CHECK(hipMemcpy(b.data(), out2.p, rows*sizeof(float), hipMemcpyDeviceToHost));
    float native_error = 0;
    for (int i = 0; i < rows; ++i) {
        if (!std::isfinite(b[i])) throw std::runtime_error("nonfinite native output");
        native_error = std::max(native_error, std::abs(a[i]-b[i]));
    }
    if (native_error > 1e-4F) throw std::runtime_error("native arithmetic mismatch");
    // Check zero, alternating extremes, and a second deterministic input.
    for (int test = 0; test < 3; ++test) {
        for (int i = 0; i < columns; ++i) x[i] = test == 0 ? 0.0F :
            test == 1 ? (i % 2 ? -12.0F : 12.0F) : float((i*73)%257-128)/31.0F;
        MIINFER_HIP_CHECK(hipMemcpy(input.p, x.data(), x.size()*sizeof(float), hipMemcpyHostToDevice));
        miinfer::launch_q8_1_quantize_f32(input.as<float>(), q8.as<miinfer::Q8_1Block>(), columns);
        launch(1); launch(2);
        MIINFER_HIP_CHECK(hipMemcpy(a.data(), out1.p, rows*sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(b.data(), out2.p, rows*sizeof(float), hipMemcpyDeviceToHost));
        for (int i = 0; i < rows; ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite test output");
            native_error = std::max(native_error, std::abs(a[i]-b[i]));
        }
    }
    if (native_error > 1e-4F) throw std::runtime_error("native input sweep mismatch");
    for (int i = 0; i < columns; ++i) x[i] = std::sin(float(i)*0.017F);
    MIINFER_HIP_CHECK(hipMemcpy(input.p, x.data(), x.size()*sizeof(float), hipMemcpyHostToDevice));
    miinfer::launch_q8_1_quantize_f32(input.as<float>(), q8.as<miinfer::Q8_1Block>(), columns);
    Event start, stop;
    std::array<std::vector<float>, 3> samples;
    for (int round = 0; round < 101; ++round) for (int order = 0; order < 3; ++order) {
        const int variant = (round + order) % 3;
        MIINFER_HIP_CHECK(hipEventRecord(start.p));
        for (int i = 0; i < 10; ++i) launch(variant);
        MIINFER_HIP_CHECK(hipEventRecord(stop.p));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop.p));
        float ms = 0;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start.p, stop.p));
        samples[variant].push_back(ms * 100);
    }
    std::ofstream file;
    if (argc == 4) { file.open(argv[3]); if (!file) throw std::runtime_error("cannot open result"); }
    const auto report = [&](std::ostream& out) {
        out << "{\"tensor\":\"" << tensor.name() << "\",\"rows\":" << rows << ",\"columns\":" << columns
            << ",\"q4_down_tensor_count\":" << down_count
            << ",\"max_error\":" << max_error << ",\"native_max_error\":" << native_error
            << ",\"cache_regime\":\"repeated_tensor_larger_than_L2\",\"variants\":[";
        for (int v = 0; v < 3; ++v) {
            if (v) out << ',';
            auto sorted = samples[v]; std::sort(sorted.begin(), sorted.end());
            const auto bytes = v == 2 ? native_bytes : v ? expanded_bytes : tensor.bytes();
            out << "{\"name\":\"" << (v == 2 ? "native_wave_planes" : v ? "production_expanded" : "B41_canonical")
                << "\",\"weight_bytes\":" << bytes << ",\"median_us\":" << sorted[50]
                << ",\"physical_GB_s\":" << bytes / (sorted[50]*1000.0) << ",\"samples_us\":[";
            for (std::size_t i = 0; i < samples[v].size(); ++i) { if (i) out << ','; out << samples[v][i]; }
            out << "]}";
        }
        out << "]}\n";
    };
    report(std::cout);
    if (file.is_open()) { report(file); file.flush(); if (!file) throw std::runtime_error("result write failed"); }
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
