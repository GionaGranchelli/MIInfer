#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "miinfer/gguf.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: gateup-operation-reference MODEL.gguf PCI_BDF\n";
        return 2;
    }

    int device = -1;
    MIINFER_HIP_CHECK(hipDeviceGetByPCIBusId(&device, argv[2]));
    MIINFER_HIP_CHECK(hipSetDevice(device));

    constexpr std::uint32_t columns = 5120;
    constexpr std::uint32_t rows = 17408;
    auto model = miinfer::Qwen35Model::load(argv[1]);
    const auto gate = model.tensor("blk.0.ffn_gate.weight");
    const auto up = model.tensor("blk.0.ffn_up.weight");
    const auto gate_mmq = pack_mx_q4k_repacked_tensor(*gate.source);
    const auto up_mmq = pack_mx_q4k_repacked_tensor(*up.source);
    const auto fused = pack_q4k_wave_swiglu_fused(*gate.source, *up.source);

    std::vector<float> input(columns);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.5f * std::sin(static_cast<float>(i) * 0.013f);
    }
    std::vector<float> mmq_output(rows), fused_output(rows);
    std::uint8_t *d_gate = nullptr, *d_up = nullptr;
    miinfer::MxQ8_1MmqBlock* d_mmq_input = nullptr;
    miinfer::Q8_1Block* d_fused_input = nullptr;
    float *d_input = nullptr, *d_gate_output = nullptr, *d_up_output = nullptr;
    float *d_mmq_output = nullptr, *d_fused_output = nullptr;
    Q4KWaveSwigluFusedTile* d_fused = nullptr;

    MIINFER_HIP_CHECK(hipMalloc(&d_gate, gate_mmq.size()));
    MIINFER_HIP_CHECK(hipMalloc(&d_up, up_mmq.size()));
    MIINFER_HIP_CHECK(hipMalloc(&d_fused, fused.size() * sizeof(fused.front())));
    MIINFER_HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_mmq_input, columns / 32 * sizeof(*d_mmq_input)));
    MIINFER_HIP_CHECK(hipMalloc(&d_fused_input, columns / 32 * sizeof(*d_fused_input)));
    MIINFER_HIP_CHECK(hipMalloc(&d_gate_output, rows * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_up_output, rows * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_mmq_output, rows * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_fused_output, rows * sizeof(float)));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate, gate_mmq.data(), gate_mmq.size(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_up, up_mmq.data(), up_mmq.size(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_fused, fused.data(), fused.size() * sizeof(fused.front()), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice));

    miinfer::launch_mx_q8_1_mmq_quantize(d_input, d_mmq_input, 1, columns, true);
    launch_mx_q4k_repacked_mmq(d_gate, d_mmq_input, d_gate_output, rows, columns, 1);
    launch_mx_q4k_repacked_mmq(d_up, d_mmq_input, d_up_output, rows, columns, 1);
    miinfer::launch_qwen3_silu_mul(d_gate_output, d_up_output, d_mmq_output, rows);
    miinfer::launch_q8_1_quantize_f32(d_input, d_fused_input, columns);
    launch_q4k_wave_fused_gate_up_swiglu_paired(
        d_fused, d_fused_input, d_fused_output, rows, columns);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    MIINFER_HIP_CHECK(hipMemcpy(mmq_output.data(), d_mmq_output, rows * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(fused_output.data(), d_fused_output, rows * sizeof(float), hipMemcpyDeviceToHost));

    double dot = 0.0, mmq_norm = 0.0, fused_norm = 0.0, error_sq = 0.0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < mmq_output.size(); ++i) {
        if (!std::isfinite(mmq_output[i]) || !std::isfinite(fused_output[i])) {
            throw std::runtime_error("non-finite Gate/Up output");
        }
        const double a = mmq_output[i], b = fused_output[i], difference = a - b;
        dot += a * b;
        mmq_norm += a * a;
        fused_norm += b * b;
        error_sq += difference * difference;
        max_abs = std::max(max_abs, std::abs(difference));
    }
    const double cosine = dot / std::sqrt(mmq_norm * fused_norm);
    const double relative_l2 = std::sqrt(error_sq / fused_norm);
    const bool pass = cosine >= 0.995 && relative_l2 <= 0.02;
    std::cout << "gateup_operation_reference=" << (pass ? "PASS" : "FAIL")
              << " bdf=" << argv[2] << " layer=0 rows=" << rows << " columns=" << columns
              << " cosine=" << cosine << " relative_l2=" << relative_l2
              << " max_abs=" << max_abs << " thresholds=cosine:0.995,relative_l2:0.02\n";

    MIINFER_HIP_CHECK(hipFree(d_fused_output));
    MIINFER_HIP_CHECK(hipFree(d_mmq_output));
    MIINFER_HIP_CHECK(hipFree(d_up_output));
    MIINFER_HIP_CHECK(hipFree(d_gate_output));
    MIINFER_HIP_CHECK(hipFree(d_fused_input));
    MIINFER_HIP_CHECK(hipFree(d_mmq_input));
    MIINFER_HIP_CHECK(hipFree(d_input));
    MIINFER_HIP_CHECK(hipFree(d_fused));
    MIINFER_HIP_CHECK(hipFree(d_up));
    MIINFER_HIP_CHECK(hipFree(d_gate));
    return pass ? 0 : 1;
}
