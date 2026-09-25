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
    std::cout << "  MIInfer V2-0010: Detailed Decode Phase Attribution (50 iters)\n";
    std::cout << "  Hardware: 1 x AMD Instinct MI50 32GB (gfx906, Wave64)\n";
    std::cout << "===================================================================\n\n";

    const std::string model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    const auto qwen_model = Qwen35Model::load(model_path);
    PrefillV2Model v2_model(qwen_model, 32768, /*load_lm_head=*/true);

    hipStream_t stream = nullptr;
    MIINFER_HIP_CHECK(hipStreamCreate(&stream));

    v2_model.reset_state();
    std::vector<std::uint32_t> prompt(64, 15);
    GenerateOptions opts{};
    opts.max_new_tokens = 1;
    opts.use_hip_graph = false;
    v2_model.generate(prompt, opts, stream);
    MIINFER_HIP_CHECK(hipStreamSynchronize(stream));

    constexpr int kIters = 50;

    auto& ws = v2_model.workspace();
    float* d_input = nullptr;
    float* d_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&d_input, 5120 * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_output, 5120 * sizeof(float)));

    auto& gdn0 = v2_model.block(0).gdn0();
    auto st0 = v2_model.recurrent_storage(0).view();

    RecurrentLayerDecodePhaseTimings gdn_sum{};
    for (int i = 0; i < kIters; ++i) {
        RecurrentLayerDecodePhaseTimings t{};
        gdn0.decode_profiled(d_input, d_output, st0, ws, nullptr, t, stream);
        gdn_sum.norm_beta_alpha_ms += t.norm_beta_alpha_ms;
        gdn_sum.qkv_gate_proj_ms += t.qkv_gate_proj_ms;
        gdn_sum.conv_l2_norm_ms += t.conv_l2_norm_ms;
        gdn_sum.gdn_step_ms += t.gdn_step_ms;
        gdn_sum.ssm_post_out_ms += t.ssm_post_out_ms;
        gdn_sum.residual_norm_ms += t.residual_norm_ms;
        gdn_sum.ffn_gate_up_swiglu_ms += t.ffn_gate_up_swiglu_ms;
        gdn_sum.ffn_down_residual_ms += t.ffn_down_residual_ms;
        gdn_sum.total_layer_ms += t.total_layer_ms;
    }

    auto to_us = [&](double ms_sum) { return (ms_sum / kIters) * 1000.0; };

    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Recurrent Layer 0 Phase Attribution (GDN SSM, 50-run mean)\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  1. Input RMSNorm + Beta/Alpha GEMM:  " << std::setw(8) << to_us(gdn_sum.norm_beta_alpha_ms) << " us\n";
    std::cout << "  2. Q8_1 Quant + QKV + Gate Wave:     " << std::setw(8) << to_us(gdn_sum.qkv_gate_proj_ms) << " us\n";
    std::cout << "  3. Conv1D + SiLU + Dual Head L2 Norm:" << std::setw(8) << to_us(gdn_sum.conv_l2_norm_ms) << " us\n";
    std::cout << "  4. GDN DeltaNet Single Step Update:  " << std::setw(8) << to_us(gdn_sum.gdn_step_ms) << " us\n";
    std::cout << "  5. SSM Post (Norm+Gate) + SSM-Out:   " << std::setw(8) << to_us(gdn_sum.ssm_post_out_ms) << " us\n";
    std::cout << "  6. Residual Add + Post RMSNorm:      " << std::setw(8) << to_us(gdn_sum.residual_norm_ms) << " us\n";
    std::cout << "  7. Q8_1 Quant + FFN SwiGLU Paired:   " << std::setw(8) << to_us(gdn_sum.ffn_gate_up_swiglu_ms) << " us\n";
    std::cout << "  8. Q8 MMQ Quant + FFN Down + Res Add:" << std::setw(8) << to_us(gdn_sum.ffn_down_residual_ms) << " us\n";
    std::cout << " -----------------------------------------------------------------\n";
    std::cout << "  Total Per Recurrent Layer:           " << std::setw(8) << to_us(gdn_sum.total_layer_ms) << " us\n";
    std::cout << "  Projected 48 Recurrent Layers:       " << std::setw(8) << (to_us(gdn_sum.total_layer_ms) * 48.0 / 1000.0) << " ms\n\n";

    // Attention Layer 3 Breakdown
    auto& gqa3 = v2_model.block(0).gqa3();
    auto kv0 = v2_model.kv_storage(0).view();

    AttentionLayerDecodePhaseTimings gqa_sum{};
    for (int i = 0; i < kIters; ++i) {
        AttentionLayerDecodePhaseTimings t{};
        gqa3.decode_profiled(d_input, d_output, kv0, ws, 64, nullptr, t, stream);
        gqa_sum.norm_ms += t.norm_ms;
        gqa_sum.qkv_proj_ms += t.qkv_proj_ms;
        gqa_sum.qk_rope_kv_store_ms += t.qk_rope_kv_store_ms;
        gqa_sum.splitk_attention_ms += t.splitk_attention_ms;
        gqa_sum.o_proj_ms += t.o_proj_ms;
        gqa_sum.residual_norm_ms += t.residual_norm_ms;
        gqa_sum.ffn_gate_up_swiglu_ms += t.ffn_gate_up_swiglu_ms;
        gqa_sum.ffn_down_residual_ms += t.ffn_down_residual_ms;
        gqa_sum.total_layer_ms += t.total_layer_ms;
    }

    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Attention Layer 3 Phase Attribution (GQA Attn, 50-run mean)\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  1. Input RMSNorm:                    " << std::setw(8) << to_us(gqa_sum.norm_ms) << " us\n";
    std::cout << "  2. Q8_1 Quant + Q, K, V Wave GEMVs:  " << std::setw(8) << to_us(gqa_sum.qkv_proj_ms) << " us\n";
    std::cout << "  3. Fused Q/K RoPE + KV Cache Store:  " << std::setw(8) << to_us(gqa_sum.qk_rope_kv_store_ms) << " us\n";
    std::cout << "  4. Split-K Dynamic Attention:        " << std::setw(8) << to_us(gqa_sum.splitk_attention_ms) << " us\n";
    std::cout << "  5. Q8_1 Quant + O Wave GEMV:         " << std::setw(8) << to_us(gqa_sum.o_proj_ms) << " us\n";
    std::cout << "  6. Residual Add + Post RMSNorm:      " << std::setw(8) << to_us(gqa_sum.residual_norm_ms) << " us\n";
    std::cout << "  7. Q8_1 Quant + FFN SwiGLU Paired:   " << std::setw(8) << to_us(gqa_sum.ffn_gate_up_swiglu_ms) << " us\n";
    std::cout << "  8. Q8 MMQ Quant + FFN Down + Res Add:" << std::setw(8) << to_us(gqa_sum.ffn_down_residual_ms) << " us\n";
    std::cout << " -----------------------------------------------------------------\n";
    std::cout << "  Total Per Attention Layer:           " << std::setw(8) << to_us(gqa_sum.total_layer_ms) << " us\n";
    std::cout << "  Projected 16 Attention Layers:       " << std::setw(8) << (to_us(gqa_sum.total_layer_ms) * 16.0 / 1000.0) << " ms\n\n";

    double grand_total_ms = (to_us(gdn_sum.total_layer_ms) * 48.0 / 1000.0) + (to_us(gqa_sum.total_layer_ms) * 16.0 / 1000.0) + 0.54;
    std::cout << "===================================================================\n";
    std::cout << "  Grand Total Modeled Step Latency:    " << grand_total_ms << " ms/token ("
              << (1000.0 / grand_total_ms) << " tok/s)\n";
    std::cout << "===================================================================\n";

    MIINFER_HIP_CHECK(hipFree(d_input));
    MIINFER_HIP_CHECK(hipFree(d_output));
    MIINFER_HIP_CHECK(hipStreamDestroy(stream));

    return 0;
}
