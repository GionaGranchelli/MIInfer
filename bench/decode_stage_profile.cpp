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
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;
using namespace miinfer::prefill_v2;

namespace {

const char* kDefaultModelPath = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";

} // namespace

int main(int argc, char** argv) {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0010: Decode Layer & Block Timing Breakdown\n";
    std::cout << "  Hardware: 1 x AMD Instinct MI50 32GB (gfx906, Wave64)\n";
    std::cout << "===================================================================\n\n";

    const std::string model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    const auto qwen_model = Qwen35Model::load(model_path);
    std::cout << "[INFO] Initializing PrefillV2Model for diagnostic profiling...\n";
    PrefillV2Model v2_model(qwen_model, 32768, /*load_lm_head=*/true);

    hipStream_t stream = nullptr;
    MIINFER_HIP_CHECK(hipStreamCreate(&stream));

    // Warmup single decode step
    v2_model.reset_state();
    std::vector<std::uint32_t> prompt(64, 15);
    GenerateOptions opts{};
    opts.max_new_tokens = 1;
    opts.use_hip_graph = false;
    auto stats = v2_model.generate(prompt, opts, stream);
    MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

    std::cout << "[INFO] Prefill complete. Profiling decode stages across 50 iterations...\n\n";

    constexpr int kIters = 50;

    auto& ws = v2_model.workspace();
    float* d_ping = nullptr;
    float* d_pong = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&d_ping, 5120 * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_pong, 5120 * sizeof(float)));

    hipEvent_t ev_start, ev_gdn0, ev_gdn1, ev_gdn2, ev_gqa3, ev_stop;
    MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_gdn0));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_gdn1));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_gdn2));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_gqa3));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_stop));

    // Measure Block 0 Layer Breakdown
    double t_gdn0 = 0.0, t_gdn1 = 0.0, t_gdn2 = 0.0, t_gqa3 = 0.0;
    auto& blk0 = v2_model.block(0);
    auto st0 = v2_model.recurrent_storage(0).view();
    auto st1 = v2_model.recurrent_storage(1).view();
    auto st2 = v2_model.recurrent_storage(2).view();
    auto kv0 = v2_model.kv_storage(0).view();

    for (int i = 0; i < kIters; ++i) {
        MIINFER_HIP_CHECK(hipEventRecord(ev_start, stream));
        blk0.gdn0().decode(d_ping, d_pong, st0, ws, nullptr, stream);
        MIINFER_HIP_CHECK(hipEventRecord(ev_gdn0, stream));
        blk0.gdn1().decode(d_pong, d_ping, st1, ws, nullptr, stream);
        MIINFER_HIP_CHECK(hipEventRecord(ev_gdn1, stream));
        blk0.gdn2().decode(d_ping, d_pong, st2, ws, nullptr, stream);
        MIINFER_HIP_CHECK(hipEventRecord(ev_gdn2, stream));
        blk0.gqa3().decode(d_pong, d_ping, kv0, ws, 64, nullptr, stream);
        MIINFER_HIP_CHECK(hipEventRecord(ev_gqa3, stream));
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

        float ms = 0.0f;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_gdn0)); t_gdn0 += ms * 1000.0;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_gdn0, ev_gdn1)); t_gdn1 += ms * 1000.0;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_gdn1, ev_gdn2)); t_gdn2 += ms * 1000.0;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_gdn2, ev_gqa3)); t_gqa3 += ms * 1000.0;
    }

    t_gdn0 /= kIters; t_gdn1 /= kIters; t_gdn2 /= kIters; t_gqa3 /= kIters;
    double t_gdn_avg = (t_gdn0 + t_gdn1 + t_gdn2) / 3.0;

    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Block 0 Layer Decode Latencies (50-run mean)\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Layer 0 (GDN SSM 0):  " << t_gdn0 << " us\n";
    std::cout << "  Layer 1 (GDN SSM 1):  " << t_gdn1 << " us\n";
    std::cout << "  Layer 2 (GDN SSM 2):  " << t_gdn2 << " us\n";
    std::cout << "  Layer 3 (GQA Attn 3): " << t_gqa3 << " us\n";
    std::cout << "  Mean GDN SSM Layer:   " << t_gdn_avg << " us\n";
    std::cout << "  Mean GQA Attn Layer:  " << t_gqa3 << " us\n";
    std::cout << "  Block 0 Total:        " << (t_gdn0 + t_gdn1 + t_gdn2 + t_gqa3) << " us\n\n";

    // Measure Whole Single Decode Step
    double t_step_direct = 0.0;
    for (int i = 0; i < kIters; ++i) {
        MIINFER_HIP_CHECK(hipEventRecord(ev_start, stream));
        (void)v2_model.decode_step(15, 64 + i, nullptr, stream);
        MIINFER_HIP_CHECK(hipEventRecord(ev_stop, stream));
        MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

        float ms = 0.0f;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        t_step_direct += ms * 1000.0;
    }
    t_step_direct /= kIters;

    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Full Single Decode Step (Direct Dispatch, 50-run mean)\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Total Step Latency:   " << (t_step_direct / 1000.0) << " ms/step ("
              << (1000000.0 / t_step_direct) << " tok/s)\n\n";

    // Modeled Totals
    double modeled_recurrent_ms = 48.0 * t_gdn_avg / 1000.0;
    double modeled_attention_ms = 16.0 * t_gqa3 / 1000.0;
    double modeled_layers_ms = modeled_recurrent_ms + modeled_attention_ms;
    double lm_head_and_other_ms = (t_step_direct / 1000.0) - modeled_layers_ms;

    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Component Breakdown of Whole Step\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  48 Recurrent Layers:  " << modeled_recurrent_ms << " ms ("
              << (modeled_recurrent_ms / (t_step_direct / 1000.0) * 100.0) << " %)\n";
    std::cout << "  16 Attention Layers:  " << modeled_attention_ms << " ms ("
              << (modeled_attention_ms / (t_step_direct / 1000.0) * 100.0) << " %)\n";
    std::cout << "  LM Head + Overhead:   " << lm_head_and_other_ms << " ms ("
              << (lm_head_and_other_ms / (t_step_direct / 1000.0) * 100.0) << " %)\n";
    std::cout << "===================================================================\n";

    MIINFER_HIP_CHECK(hipFree(d_ping));
    MIINFER_HIP_CHECK(hipFree(d_pong));
    MIINFER_HIP_CHECK(hipEventDestroy(ev_start));
    MIINFER_HIP_CHECK(hipEventDestroy(ev_gdn0));
    MIINFER_HIP_CHECK(hipEventDestroy(ev_gdn1));
    MIINFER_HIP_CHECK(hipEventDestroy(ev_gdn2));
    MIINFER_HIP_CHECK(hipEventDestroy(ev_gqa3));
    MIINFER_HIP_CHECK(hipEventDestroy(ev_stop));
    MIINFER_HIP_CHECK(hipStreamDestroy(stream));

    return 0;
}
