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

struct TimingResult {
    std::string type_str;
    float wave_us;
    float mmq_us;
    float delta_us; // mmq - wave (cost of using MMQ)
    float extra_mib_per_layer;
    float total_mib;
    float total_model_ms; // delta across all layers in model
};

int main() {
    std::string model_path = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";
    std::cout << "===================================================================\n";
    std::cout << "  Candidate B: Decode-Native Wave vs Mx MMQ Performance Economics\n";
    std::cout << "  Hardware: 1 x AMD Instinct MI50 32GB (gfx906, Wave64)\n";
    std::cout << "===================================================================\n\n";

    auto model = miinfer::Qwen35Model::load(model_path);
    const int kIters = 200;
    hipEvent_t start, stop;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));

    auto benchmark_family = [&](const std::string& name,
                                const std::string& tensor_name,
                                unsigned rows,
                                unsigned columns,
                                int layer_count) -> TimingResult {
        const auto t = model.tensor(tensor_name);
        miinfer::GgufTensorType type = t.type();
        std::string type_str = (type == miinfer::GgufTensorType::q4_k) ? "Q4_K" :
                               (type == miinfer::GgufTensorType::q5_k) ? "Q5_K" :
                               (type == miinfer::GgufTensorType::q6_k) ? "Q6_K" : "OTHER";
        
        // Pack Wave and MMQ
        std::vector<std::uint8_t> mmq_packed;
        std::size_t wave_bytes = 0;
        void* d_wave = nullptr;
        std::uint8_t* d_mmq = nullptr;

        if (type == miinfer::GgufTensorType::q4_k) {
            mmq_packed = pack_mx_q4k_repacked_tensor(*t.source);
            auto wave_packed = pack_q4k_wave_tensor(*t.source);
            wave_bytes = wave_packed.size() * sizeof(Q4KWaveTile);
            MIINFER_HIP_CHECK(hipMalloc(&d_wave, wave_bytes));
            MIINFER_HIP_CHECK(hipMemcpy(d_wave, wave_packed.data(), wave_bytes, hipMemcpyHostToDevice));
        } else if (type == miinfer::GgufTensorType::q5_k) {
            mmq_packed = pack_mx_q5k_repacked_tensor(*t.source);
            auto wave_packed = pack_q5k_wave_tensor(*t.source);
            wave_bytes = wave_packed.size() * sizeof(Q5KWaveTile);
            MIINFER_HIP_CHECK(hipMalloc(&d_wave, wave_bytes));
            MIINFER_HIP_CHECK(hipMemcpy(d_wave, wave_packed.data(), wave_bytes, hipMemcpyHostToDevice));
        } else if (type == miinfer::GgufTensorType::q6_k) {
            mmq_packed = pack_mx_q6k_repacked_tensor(*t.source);
            auto wave_packed = pack_q6k_wave_tensor(*t.source);
            wave_bytes = wave_packed.size() * sizeof(Q6KWaveTile);
            MIINFER_HIP_CHECK(hipMalloc(&d_wave, wave_bytes));
            MIINFER_HIP_CHECK(hipMemcpy(d_wave, wave_packed.data(), wave_bytes, hipMemcpyHostToDevice));
        }

        MIINFER_HIP_CHECK(hipMalloc(&d_mmq, mmq_packed.size()));
        MIINFER_HIP_CHECK(hipMemcpy(d_mmq, mmq_packed.data(), mmq_packed.size(), hipMemcpyHostToDevice));

        // Setup inputs and outputs
        std::vector<float> h_input(columns, 1.0f);
        float* d_input = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_input, columns * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemcpy(d_input, h_input.data(), columns * sizeof(float), hipMemcpyHostToDevice));

        miinfer::Q8_1Block* d_q8 = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_q8, (columns / 32) * sizeof(miinfer::Q8_1Block)));
        miinfer::launch_q8_1_quantize_f32(d_input, d_q8, columns);

        miinfer::MxQ8_1MmqBlock* d_mmq_q8 = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_mmq_q8, (columns / 32) * sizeof(miinfer::MxQ8_1MmqBlock)));
        miinfer::launch_mx_q8_1_mmq_quantize(d_input, d_mmq_q8, 1, columns, type != miinfer::GgufTensorType::q6_k);

        float* d_out_wave = nullptr;
        float* d_out_mmq = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_out_wave, rows * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_out_mmq, rows * sizeof(float)));

        // Warm up
        for (int i = 0; i < 5; ++i) {
            if (type == miinfer::GgufTensorType::q4_k) {
                launch_q4k_wave_gemv(static_cast<const Q4KWaveTile*>(d_wave), d_q8, d_out_wave, rows, columns);
                launch_mx_q4k_repacked_mmq(d_mmq, d_mmq_q8, d_out_mmq, rows, columns, 1);
            } else if (type == miinfer::GgufTensorType::q5_k) {
                launch_q5k_wave_gemv(static_cast<const Q5KWaveTile*>(d_wave), d_q8, d_out_wave, rows, columns);
                launch_mx_q5k_repacked_mmq(d_mmq, d_mmq_q8, d_out_mmq, rows, columns, 1);
            } else if (type == miinfer::GgufTensorType::q6_k) {
                launch_q6k_wave_gemv(static_cast<const Q6KWaveTile*>(d_wave), d_q8, d_out_wave, rows, columns);
                launch_mx_q6k_repacked_mmq(d_mmq, d_mmq_q8, d_out_mmq, rows, columns, 1);
            }
        }
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        // Measure Wave
        MIINFER_HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < kIters; ++i) {
            if (type == miinfer::GgufTensorType::q4_k) {
                launch_q4k_wave_gemv(static_cast<const Q4KWaveTile*>(d_wave), d_q8, d_out_wave, rows, columns);
            } else if (type == miinfer::GgufTensorType::q5_k) {
                launch_q5k_wave_gemv(static_cast<const Q5KWaveTile*>(d_wave), d_q8, d_out_wave, rows, columns);
            } else if (type == miinfer::GgufTensorType::q6_k) {
                launch_q6k_wave_gemv(static_cast<const Q6KWaveTile*>(d_wave), d_q8, d_out_wave, rows, columns);
            }
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float wave_ms = 0.0f;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&wave_ms, start, stop));
        float wave_us = (wave_ms / kIters) * 1000.0f;

        // Measure MMQ (N=1)
        MIINFER_HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < kIters; ++i) {
            if (type == miinfer::GgufTensorType::q4_k) {
                launch_mx_q4k_repacked_mmq(d_mmq, d_mmq_q8, d_out_mmq, rows, columns, 1);
            } else if (type == miinfer::GgufTensorType::q5_k) {
                launch_mx_q5k_repacked_mmq(d_mmq, d_mmq_q8, d_out_mmq, rows, columns, 1);
            } else if (type == miinfer::GgufTensorType::q6_k) {
                launch_mx_q6k_repacked_mmq(d_mmq, d_mmq_q8, d_out_mmq, rows, columns, 1);
            }
        }
        MIINFER_HIP_CHECK(hipEventRecord(stop));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float mmq_ms = 0.0f;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&mmq_ms, start, stop));
        float mmq_us = (mmq_ms / kIters) * 1000.0f;

        float extra_mib = static_cast<float>(wave_bytes) / (1024.0f * 1024.0f);
        float total_mib = extra_mib * layer_count;
        float delta_us = mmq_us - wave_us;
        float total_model_ms = (delta_us * layer_count) / 1000.0f;

        MIINFER_HIP_CHECK(hipFree(d_wave));
        MIINFER_HIP_CHECK(hipFree(d_mmq));
        MIINFER_HIP_CHECK(hipFree(d_input));
        MIINFER_HIP_CHECK(hipFree(d_q8));
        MIINFER_HIP_CHECK(hipFree(d_mmq_q8));
        MIINFER_HIP_CHECK(hipFree(d_out_wave));
        MIINFER_HIP_CHECK(hipFree(d_out_mmq));

        return {type_str, wave_us, mmq_us, delta_us, extra_mib, total_mib, total_model_ms};
    };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "| Family | Layers | Shape | Quant | Extra MiB | Wave us | MMQ us | Δ us/layer | Full Model Δ ms | MiB / ms saved |\n";
    std::cout << "|:---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|\n";

    // 1. Recurrent QKV (48 layers, 10240x5120)
    auto r_qkv = benchmark_family("recurrent QKV", "blk.0.attn_qkv.weight", 10240, 5120, 48);
    std::cout << "| recurrent QKV | 48 | [10240, 5120] | " << r_qkv.type_str << " | " << r_qkv.total_mib << " MiB | " << r_qkv.wave_us << " us | " << r_qkv.mmq_us << " us | +" << r_qkv.delta_us << " us | +" << r_qkv.total_model_ms << " ms | " << (r_qkv.total_model_ms > 0 ? r_qkv.total_mib / r_qkv.total_model_ms : 0) << " MiB/ms |\n";

    // 2. Recurrent Gate (48 layers, 6144x5120)
    auto r_gate = benchmark_family("recurrent Gate", "blk.0.attn_gate.weight", 6144, 5120, 48);
    std::cout << "| recurrent Gate | 48 | [6144, 5120] | " << r_gate.type_str << " | " << r_gate.total_mib << " MiB | " << r_gate.wave_us << " us | " << r_gate.mmq_us << " us | +" << r_gate.delta_us << " us | +" << r_gate.total_model_ms << " ms | " << (r_gate.total_model_ms > 0 ? r_gate.total_mib / r_gate.total_model_ms : 0) << " MiB/ms |\n";

    // 3. Recurrent SSM Out (48 layers, 5120x6144)
    auto r_ssm_out = benchmark_family("SSM Out", "blk.0.ssm_out.weight", 5120, 6144, 48);
    std::cout << "| SSM Out | 48 | [5120, 6144] | " << r_ssm_out.type_str << " | " << r_ssm_out.total_mib << " MiB | " << r_ssm_out.wave_us << " us | " << r_ssm_out.mmq_us << " us | +" << r_ssm_out.delta_us << " us | +" << r_ssm_out.total_model_ms << " ms | " << (r_ssm_out.total_model_ms > 0 ? r_ssm_out.total_mib / r_ssm_out.total_model_ms : 0) << " MiB/ms |\n";

    // 4. Attention Q (16 layers, 6144x5120)
    auto a_q = benchmark_family("attention Q", "blk.3.attn_q.weight", 6144, 5120, 16);
    std::cout << "| attention Q | 16 | [6144, 5120] | " << a_q.type_str << " | " << a_q.total_mib << " MiB | " << a_q.wave_us << " us | " << a_q.mmq_us << " us | +" << a_q.delta_us << " us | +" << a_q.total_model_ms << " ms | " << (a_q.total_model_ms > 0 ? a_q.total_mib / a_q.total_model_ms : 0) << " MiB/ms |\n";

    // 5. Attention K (16 layers, 1024x5120)
    auto a_k = benchmark_family("attention K", "blk.3.attn_k.weight", 1024, 5120, 16);
    std::cout << "| attention K | 16 | [1024, 5120] | " << a_k.type_str << " | " << a_k.total_mib << " MiB | " << a_k.wave_us << " us | " << a_k.mmq_us << " us | +" << a_k.delta_us << " us | +" << a_k.total_model_ms << " ms | " << (a_k.total_model_ms > 0 ? a_k.total_mib / a_k.total_model_ms : 0) << " MiB/ms |\n";

    // 6. Attention V (16 layers, 1024x5120)
    auto a_v = benchmark_family("attention V", "blk.3.attn_v.weight", 1024, 5120, 16);
    std::cout << "| attention V | 16 | [1024, 5120] | " << a_v.type_str << " | " << a_v.total_mib << " MiB | " << a_v.wave_us << " us | " << a_v.mmq_us << " us | +" << a_v.delta_us << " us | +" << a_v.total_model_ms << " ms | " << (a_v.total_model_ms > 0 ? a_v.total_mib / a_v.total_model_ms : 0) << " MiB/ms |\n";

    // 7. Attention O (16 layers, 5120x6144)
    auto a_o = benchmark_family("attention O", "blk.3.attn_output.weight", 5120, 6144, 16);
    std::cout << "| attention O | 16 | [5120, 6144] | " << a_o.type_str << " | " << a_o.total_mib << " MiB | " << a_o.wave_us << " us | " << a_o.mmq_us << " us | +" << a_o.delta_us << " us | +" << a_o.total_model_ms << " ms | " << (a_o.total_model_ms > 0 ? a_o.total_mib / a_o.total_model_ms : 0) << " MiB/ms |\n";

    return 0;
}
