#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <hip/hip_runtime.h>

#include "miinfer/hip_check.hpp"
#include "miinfer/gguf.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

int main() {
    std::string model_path = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";
    std::cout << "[INFO] Loading model to verify Candidate A (Shared Paired FFN)...\n";
    auto model = miinfer::Qwen35Model::load(model_path);

    const auto gate_t = model.tensor("blk.0.ffn_gate.weight");
    const auto up_t = model.tensor("blk.0.ffn_up.weight");

    const unsigned rows = 17408;
    const unsigned columns = 5120;

    // Pack both representations
    const auto gate_mmq_packed = pack_mx_q4k_repacked_tensor(*gate_t.source);
    const auto up_mmq_packed = pack_mx_q4k_repacked_tensor(*up_t.source);
    const auto fused_swiglu = pack_q4k_wave_swiglu_fused(*gate_t.source, *up_t.source);

    std::uint8_t *d_gate_mmq = nullptr, *d_up_mmq = nullptr;
    Q4KWaveSwigluFusedTile *d_fused = nullptr;

    MIINFER_HIP_CHECK(hipMalloc(&d_gate_mmq, gate_mmq_packed.size()));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate_mmq, gate_mmq_packed.data(), gate_mmq_packed.size(), hipMemcpyHostToDevice));

    MIINFER_HIP_CHECK(hipMalloc(&d_up_mmq, up_mmq_packed.size()));
    MIINFER_HIP_CHECK(hipMemcpy(d_up_mmq, up_mmq_packed.data(), up_mmq_packed.size(), hipMemcpyHostToDevice));

    MIINFER_HIP_CHECK(hipMalloc(&d_fused, fused_swiglu.size() * sizeof(Q4KWaveSwigluFusedTile)));
    MIINFER_HIP_CHECK(hipMemcpy(d_fused, fused_swiglu.data(), fused_swiglu.size() * sizeof(Q4KWaveSwigluFusedTile), hipMemcpyHostToDevice));

    for (unsigned token_count : {64, 128, 512}) {
        std::cout << "\n=======================================================\n";
        std::cout << "  Testing N = " << token_count << " tokens\n";
        std::cout << "=======================================================\n";

        // Generate synthetic random float activations
        std::vector<float> h_input(token_count * columns);
        for (std::size_t i = 0; i < h_input.size(); ++i) {
            h_input[i] = static_cast<float>((i % 1000) - 500) / 500.0f;
        }

        float* d_input = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_input, h_input.size() * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), hipMemcpyHostToDevice));

        // Baseline: Quantize to MxQ8_1MmqBlock, run 2 MMQ launches + silu_mul
        miinfer::MxQ8_1MmqBlock* d_mmq_q8 = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_mmq_q8, (columns / 32) * token_count * sizeof(miinfer::MxQ8_1MmqBlock)));
        float *d_gate_out = nullptr, *d_up_out = nullptr, *d_base_act = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_gate_out, token_count * rows * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_up_out, token_count * rows * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_base_act, token_count * rows * sizeof(float)));

        // Candidate: Quantize to Q8_1Block, run launch_q4k_wave_fused_gate_up_swiglu_paired_batch
        miinfer::Q8_1Block* d_q8_1 = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_q8_1, token_count * (columns / 32) * sizeof(miinfer::Q8_1Block)));
        float* d_cand_act = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_cand_act, token_count * rows * sizeof(float)));

        // Run baseline
        miinfer::launch_mx_q8_1_mmq_quantize(d_input, d_mmq_q8, token_count, columns, true);
        launch_mx_q4k_repacked_mmq(d_gate_mmq, d_mmq_q8, d_gate_out, rows, columns, token_count);
        launch_mx_q4k_repacked_mmq(d_up_mmq, d_mmq_q8, d_up_out, rows, columns, token_count);
        miinfer::launch_qwen3_silu_mul(d_gate_out, d_up_out, d_base_act, token_count * rows);

        // Run candidate
        for (unsigned t = 0; t < token_count; ++t) {
            miinfer::launch_q8_1_quantize_f32(d_input + t * columns, d_q8_1 + t * (columns / 32), columns);
        }
        launch_q4k_wave_fused_gate_up_swiglu_paired_batch(d_fused, d_q8_1, d_cand_act, rows, columns, token_count);

        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        // Verify correctness
        std::vector<float> h_base(token_count * rows), h_cand(token_count * rows);
        MIINFER_HIP_CHECK(hipMemcpy(h_base.data(), d_base_act, h_base.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(h_cand.data(), d_cand_act, h_cand.size() * sizeof(float), hipMemcpyDeviceToHost));

        double max_err = 0.0, sum_err = 0.0, dot = 0.0, norm_b = 0.0, norm_c = 0.0;
        for (std::size_t i = 0; i < h_base.size(); ++i) {
            double b = h_base[i];
            double c = h_cand[i];
            double diff = std::abs(b - c);
            if (diff > max_err) max_err = diff;
            sum_err += diff;
            dot += b * c;
            norm_b += b * b;
            norm_c += c * c;
        }
        double cosine = dot / (std::sqrt(norm_b) * std::sqrt(norm_c));
        std::cout << "  Cosine Similarity: " << std::setprecision(7) << cosine << "\n";
        std::cout << "  Max Abs Error:     " << max_err << "\n";
        std::cout << "  Mean Abs Error:    " << sum_err / h_base.size() << "\n";

        // Benchmark timing (50 iterations)
        const int kIters = 50;
        hipEvent_t start, stop;
        MIINFER_HIP_CHECK(hipEventCreate(&start));
        MIINFER_HIP_CHECK(hipEventCreate(&stop));

        // Baseline time
        MIINFER_HIP_CHECK(hipEventRecord(start));
        for (int it = 0; it < kIters; ++it) {
            miinfer::launch_mx_q8_1_mmq_quantize(d_input, d_mmq_q8, token_count, columns, true);
            launch_mx_q4k_repacked_mmq(d_gate_mmq, d_mmq_q8, d_gate_out, rows, columns, token_count);
            launch_mx_q4k_repacked_mmq(d_up_mmq, d_mmq_q8, d_up_out, rows, columns, token_count);
            miinfer::launch_qwen3_silu_mul(d_gate_out, d_up_out, d_base_act, token_count * rows);
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float base_ms = 0.0f;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&base_ms, start, stop));
        base_ms /= kIters;

        // Candidate time
        MIINFER_HIP_CHECK(hipEventRecord(start));
        for (int it = 0; it < kIters; ++it) {
            for (unsigned t = 0; t < token_count; ++t) {
                miinfer::launch_q8_1_quantize_f32(d_input + t * columns, d_q8_1 + t * (columns / 32), columns);
            }
            launch_q4k_wave_fused_gate_up_swiglu_paired_batch(d_fused, d_q8_1, d_cand_act, rows, columns, token_count);
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float cand_ms = 0.0f;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&cand_ms, start, stop));
        cand_ms /= kIters;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Baseline MMQ (Gate+Up+SwiGLU): " << base_ms << " ms\n";
        std::cout << "  Candidate Paired Fused SwiGLU: " << cand_ms << " ms\n";
        std::cout << "  Speedup: " << base_ms / cand_ms << "x\n";

        MIINFER_HIP_CHECK(hipFree(d_input));
        MIINFER_HIP_CHECK(hipFree(d_mmq_q8));
        MIINFER_HIP_CHECK(hipFree(d_gate_out));
        MIINFER_HIP_CHECK(hipFree(d_up_out));
        MIINFER_HIP_CHECK(hipFree(d_base_act));
        MIINFER_HIP_CHECK(hipFree(d_q8_1));
        MIINFER_HIP_CHECK(hipFree(d_cand_act));
    }

    MIINFER_HIP_CHECK(hipFree(d_gate_mmq));
    MIINFER_HIP_CHECK(hipFree(d_up_mmq));
    MIINFER_HIP_CHECK(hipFree(d_fused));

    return 0;
}
