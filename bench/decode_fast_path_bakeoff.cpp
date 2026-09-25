#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "miinfer/hip_check.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;

namespace {

const char* kDefaultModelPath = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";

struct TimingResult {
    double min_us = 1e9;
    double mean_us = 0.0;
    double max_us = 0.0;
};

template <typename F>
TimingResult measure(F&& fn, int warmup = 50, int iters = 500, hipStream_t stream = nullptr) {
    for (int i = 0; i < warmup; ++i) {
        fn(stream);
    }
    MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> runs;
    runs.reserve(iters);

    hipEvent_t start, stop;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));

    for (int i = 0; i < iters; ++i) {
        MIINFER_HIP_CHECK(hipEventRecord(start, stream));
        fn(stream);
        MIINFER_HIP_CHECK(hipEventRecord(stop, stream));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        runs.push_back(ms * 1000.0); // us
    }

    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(stop));

    TimingResult res;
    double sum = 0.0;
    for (double us : runs) {
        res.min_us = std::min(res.min_us, us);
        res.max_us = std::max(res.max_us, us);
        sum += us;
    }
    res.mean_us = sum / static_cast<double>(runs.size());
    return res;
}

static const GgufTensor* find_tensor(const GgufFile& file, const std::string& name) {
    for (const auto& tensor : file.tensors()) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0009: Decode Fast Path Micro-Bakeoff\n";
    std::cout << "  Hardware: 1 x AMD Instinct MI50 32GB (gfx906, Wave64)\n";
    std::cout << "===================================================================\n\n";

    const std::string model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    const auto qwen_model = Qwen35Model::load(model_path);
    const auto file = qwen_model.file();

    constexpr std::uint32_t kHidden = 5120;
    constexpr std::uint32_t kFfnInner = 13824;

    // Load Layer 0 FFN Gate, Up, Down tensors
    const auto* t_gate = find_tensor(*file, "blk.0.ffn_gate.weight");
    const auto* t_up = find_tensor(*file, "blk.0.ffn_up.weight");
    const auto* t_down = find_tensor(*file, "blk.0.ffn_down.weight");

    if (!t_gate || !t_up || !t_down) {
        std::cerr << "Error: missing ffn tensors\n";
        return 1;
    }

    std::cout << "[INFO] Repacking FFN weights into representations...\n";

    // 1. Mx repacked representation (Control V2)
    auto mx_gate_host = pack_mx_q4k_repacked_tensor(*t_gate);
    auto mx_up_host = pack_mx_q4k_repacked_tensor(*t_up);
    auto mx_down_host = (t_down->type == GgufTensorType::q6_k)
        ? pack_mx_q6k_repacked_tensor(*t_down)
        : pack_mx_q4k_repacked_tensor(*t_down);

    std::uint8_t *d_mx_gate = nullptr, *d_mx_up = nullptr, *d_mx_down = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&d_mx_gate, mx_gate_host.size()));
    MIINFER_HIP_CHECK(hipMalloc(&d_mx_up, mx_up_host.size()));
    MIINFER_HIP_CHECK(hipMalloc(&d_mx_down, mx_down_host.size()));
    MIINFER_HIP_CHECK(hipMemcpy(d_mx_gate, mx_gate_host.data(), mx_gate_host.size(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_mx_up, mx_up_host.data(), mx_up_host.size(), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_mx_down, mx_down_host.data(), mx_down_host.size(), hipMemcpyHostToDevice));

    // 2. Native Wave fused SwiGLU paired representation (Candidate A)
    auto fused_swiglu_host = pack_q4k_wave_swiglu_fused(*t_gate, *t_up);
    Q4KWaveSwigluFusedTile* d_fused_swiglu = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&d_fused_swiglu, fused_swiglu_host.size() * sizeof(Q4KWaveSwigluFusedTile)));
    MIINFER_HIP_CHECK(hipMemcpy(d_fused_swiglu, fused_swiglu_host.data(),
                                fused_swiglu_host.size() * sizeof(Q4KWaveSwigluFusedTile), hipMemcpyHostToDevice));

    // 3. Native Wave individual representations (Candidate C)
    auto wave_gate_host = pack_q4k_wave_tensor(*t_gate);
    auto wave_up_host = pack_q4k_wave_tensor(*t_up);
    Q4KWaveTile *d_wave_gate = nullptr, *d_wave_up = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&d_wave_gate, wave_gate_host.size() * sizeof(Q4KWaveTile)));
    MIINFER_HIP_CHECK(hipMalloc(&d_wave_up, wave_up_host.size() * sizeof(Q4KWaveTile)));
    MIINFER_HIP_CHECK(hipMemcpy(d_wave_gate, wave_gate_host.data(), wave_gate_host.size() * sizeof(Q4KWaveTile), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_wave_up, wave_up_host.data(), wave_up_host.size() * sizeof(Q4KWaveTile), hipMemcpyHostToDevice));

    // Alloc intermediate activation buffers
    float *d_input = nullptr, *d_ffn_gate = nullptr, *d_ffn_up = nullptr, *d_ffn_act = nullptr, *d_output = nullptr;
    MxQ8_1MmqBlock* d_mmq_q8 = nullptr;
    Q8_1Block* d_q8_1 = nullptr;

    MIINFER_HIP_CHECK(hipMalloc(&d_input, kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_ffn_gate, kFfnInner * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_ffn_up, kFfnInner * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_ffn_act, kFfnInner * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_output, kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_mmq_q8, (kFfnInner / 128) * sizeof(MxQ8_1MmqBlock)));
    MIINFER_HIP_CHECK(hipMalloc(&d_q8_1, (kFfnInner / 32) * sizeof(Q8_1Block)));

    // Fill input with test values
    std::vector<float> host_input(kHidden, 0.05f);
    MIINFER_HIP_CHECK(hipMemcpy(d_input, host_input.data(), kHidden * sizeof(float), hipMemcpyHostToDevice));

    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Bakeoff 1: FFN Gate + Up + SwiGLU (13824 x 5120)\n";
    std::cout << "-------------------------------------------------------------------\n";

    // Route 1: Control V2 (Mx MMQ quantize + 2x Mx MMV + silu_mul)
    auto t_ctrl = measure([&](hipStream_t s) {
        launch_mx_q8_1_mmq_quantize(d_input, d_mmq_q8, 1, kHidden, true, s);
        launch_mx_q4k_repacked_mmq(d_mx_gate, d_mmq_q8, d_ffn_gate, kFfnInner, kHidden, 1, s);
        launch_mx_q4k_repacked_mmq(d_mx_up, d_mmq_q8, d_ffn_up, kFfnInner, kHidden, 1, s);
        launch_qwen3_silu_mul(d_ffn_gate, d_ffn_up, d_ffn_act, kFfnInner, s);
    });

    // Route 2: Candidate A (Canonical Q8_1 quantize + Fused Paired SwiGLU Wave Kernel)
    auto t_cand_a = measure([&](hipStream_t s) {
        launch_q8_1_quantize_f32(d_input, d_q8_1, kHidden, s);
        launch_q4k_wave_fused_gate_up_swiglu_paired(d_fused_swiglu, d_q8_1, d_ffn_act, kFfnInner, kHidden, s);
    });

    // Route 3: Candidate A2 (Canonical Q8_1 quantize + Fused Separate SwiGLU Wave Kernel)
    auto t_cand_a2 = measure([&](hipStream_t s) {
        launch_q8_1_quantize_f32(d_input, d_q8_1, kHidden, s);
        launch_q4k_wave_fused_gate_up_swiglu(d_wave_gate, d_wave_up, d_q8_1, d_ffn_act, kFfnInner, kHidden, s);
    });

    std::cout << "  Control V2 (Mx MMQ 2x + SiLU)   : " << std::fixed << std::setprecision(2)
              << t_ctrl.mean_us << " us (min=" << t_ctrl.min_us << ")\n";
    std::cout << "  Candidate A (Fused Paired Wave)  : " << std::fixed << std::setprecision(2)
              << t_cand_a.mean_us << " us (min=" << t_cand_a.min_us << ") -> SPEEDUP: "
              << (t_ctrl.mean_us / t_cand_a.mean_us) << "x (Saves "
              << (t_ctrl.mean_us - t_cand_a.mean_us) << " us/layer)\n";
    std::cout << "  Candidate A2 (Fused Sep Wave)    : " << std::fixed << std::setprecision(2)
              << t_cand_a2.mean_us << " us (min=" << t_cand_a2.min_us << ") -> SPEEDUP: "
              << (t_ctrl.mean_us / t_cand_a2.mean_us) << "x\n";

    double total_fused_savings_ms = (t_ctrl.mean_us - t_cand_a.mean_us) * 64.0 / 1000.0;
    std::cout << "  >> Projected Full-Model (64 Layers) FFN Savings: "
              << std::fixed << std::setprecision(2) << total_fused_savings_ms << " ms/token!\n";

    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Bakeoff 2: Single Projection (5120 -> 5120)\n";
    std::cout << "-------------------------------------------------------------------\n";

    // Load Layer 0 SSM Out tensor (Q5_K)
    const auto* t_ssm_out = find_tensor(*file, "blk.0.ssm_out.weight");
    if (t_ssm_out) {
        auto mx_ssm_host = pack_mx_q5k_repacked_tensor(*t_ssm_out);
        auto wave_ssm_host = pack_q5k_wave_tensor(*t_ssm_out);
        std::uint8_t* d_mx_ssm = nullptr;
        Q5KWaveTile* d_wave_ssm = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_mx_ssm, mx_ssm_host.size()));
        MIINFER_HIP_CHECK(hipMalloc(&d_wave_ssm, wave_ssm_host.size() * sizeof(Q5KWaveTile)));
        MIINFER_HIP_CHECK(hipMemcpy(d_mx_ssm, mx_ssm_host.data(), mx_ssm_host.size(), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_wave_ssm, wave_ssm_host.data(), wave_ssm_host.size() * sizeof(Q5KWaveTile), hipMemcpyHostToDevice));

        auto t_mx_ssm = measure([&](hipStream_t s) {
            launch_mx_q8_1_mmq_quantize(d_input, d_mmq_q8, 1, kHidden, true, s);
            launch_mx_q5k_repacked_mmq(d_mx_ssm, d_mmq_q8, d_output, kHidden, kHidden, 1, s);
        });

        auto t_wave_ssm = measure([&](hipStream_t s) {
            launch_q8_1_quantize_f32(d_input, d_q8_1, kHidden, s);
            launch_q5k_wave_gemv(d_wave_ssm, d_q8_1, d_output, kHidden, kHidden, s);
        });

        std::cout << "  SSM-Out (Q5_K 5120x5120) Mx MMV    : " << std::fixed << std::setprecision(2)
                  << t_mx_ssm.mean_us << " us\n";
        std::cout << "  SSM-Out (Q5_K 5120x5120) Wave GEMV  : " << std::fixed << std::setprecision(2)
                  << t_wave_ssm.mean_us << " us -> SPEEDUP: "
                  << (t_mx_ssm.mean_us / t_wave_ssm.mean_us) << "x (Saves "
                  << (t_mx_ssm.mean_us - t_wave_ssm.mean_us) << " us/layer)\n";

        double total_ssm_savings_ms = (t_mx_ssm.mean_us - t_wave_ssm.mean_us) * 48.0 / 1000.0;
        std::cout << "  >> Projected 48 Recurrent Layers SSM-Out Savings: "
                  << std::fixed << std::setprecision(2) << total_ssm_savings_ms << " ms/token!\n";

        MIINFER_HIP_CHECK(hipFree(d_mx_ssm));
        MIINFER_HIP_CHECK(hipFree(d_wave_ssm));
    }

    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Bakeoff 3: FFN Down Projection (5120 x 13824)\n";
    std::cout << "-------------------------------------------------------------------\n";

    if (t_down->type == GgufTensorType::q6_k) {
        Q6KDeviceBlock* d_raw_down = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_raw_down, t_down->byte_size));
        MIINFER_HIP_CHECK(hipMemcpy(d_raw_down, t_down->data, t_down->byte_size, hipMemcpyHostToDevice));

        Q8KDeviceBlock* d_q8_k = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_q8_k, (kFfnInner / 256) * sizeof(Q8KDeviceBlock)));

        auto t_mx_down = measure([&](hipStream_t s) {
            launch_mx_q8_1_mmq_quantize(d_ffn_act, d_mmq_q8, 1, kFfnInner, false, s);
            launch_mx_q6k_repacked_mmq(d_mx_down, d_mmq_q8, d_output, kHidden, kFfnInner, 1, s);
        });

        auto t_mmvq_down = measure([&](hipStream_t s) {
            launch_q8_1_quantize_f32(d_ffn_act, d_q8_1, kFfnInner, s);
            launch_qwen3_q6_k_q8_1_mmvq(d_raw_down, d_q8_1, d_output, kHidden, kFfnInner, s);
        });

        auto t_q8k_down = measure([&](hipStream_t s) {
            launch_qwen3_q8_k_quantize(d_ffn_act, d_q8_k, kFfnInner, s);
            launch_qwen3_q6_k_q8_k_gemv(d_raw_down, d_q8_k, d_output, kHidden, kFfnInner, s);
        });

        std::cout << "  FFN Down (Q6_K 5120x13824) Mx MMV     : " << std::fixed << std::setprecision(2)
                  << t_mx_down.mean_us << " us\n";
        std::cout << "  FFN Down (Q6_K 5120x13824) Q8_1 MMVQ  : " << std::fixed << std::setprecision(2)
                  << t_mmvq_down.mean_us << " us -> SPEEDUP: "
                  << (t_mx_down.mean_us / t_mmvq_down.mean_us) << "x (Saves "
                  << (t_mx_down.mean_us - t_mmvq_down.mean_us) << " us/layer)\n";
        std::cout << "  FFN Down (Q6_K 5120x13824) Q8_K GEMV  : " << std::fixed << std::setprecision(2)
                  << t_q8k_down.mean_us << " us -> SPEEDUP: "
                  << (t_mx_down.mean_us / t_q8k_down.mean_us) << "x\n";

        double total_down_savings_ms = (t_mx_down.mean_us - t_mmvq_down.mean_us) * 64.0 / 1000.0;
        std::cout << "  >> Projected Full-Model (64 Layers) FFN-Down Savings: "
                  << std::fixed << std::setprecision(2) << total_down_savings_ms << " ms/token!\n";

        MIINFER_HIP_CHECK(hipFree(d_raw_down));
        MIINFER_HIP_CHECK(hipFree(d_q8_k));
    }

    std::cout << "\n===================================================================\n";
    std::cout << "  Bakeoff Complete.\n";
    std::cout << "===================================================================\n";

    // Clean up
    MIINFER_HIP_CHECK(hipFree(d_mx_gate));
    MIINFER_HIP_CHECK(hipFree(d_mx_up));
    MIINFER_HIP_CHECK(hipFree(d_mx_down));
    MIINFER_HIP_CHECK(hipFree(d_fused_swiglu));
    MIINFER_HIP_CHECK(hipFree(d_wave_gate));
    MIINFER_HIP_CHECK(hipFree(d_wave_up));
    MIINFER_HIP_CHECK(hipFree(d_input));
    MIINFER_HIP_CHECK(hipFree(d_ffn_gate));
    MIINFER_HIP_CHECK(hipFree(d_ffn_up));
    MIINFER_HIP_CHECK(hipFree(d_ffn_act));
    MIINFER_HIP_CHECK(hipFree(d_output));
    MIINFER_HIP_CHECK(hipFree(d_mmq_q8));
    MIINFER_HIP_CHECK(hipFree(d_q8_1));

    return 0;
}
