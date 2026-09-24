#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/recurrent_layer.hpp"
#include "miinfer/prefill_v2/state.hpp"
#include "miinfer/prefill_v2/workspace.hpp"
#include "miinfer/qwen35_model.hpp"

#define MIINFER_M6A3_HELPERS_ONLY
#include "../tools/qwen35_gpu_pipeline.hpp"

#include <hip/hip_runtime.h>

namespace {

using namespace miinfer::prefill_v2;

struct NumericalMetrics {
    float max_abs_err = 0.0F;
    float mean_abs_err = 0.0F;
    float rmse = 0.0F;
    float rel_rms = 0.0F;
    float cosine_similarity = 0.0F;
    bool all_finite = true;
    std::size_t worst_index = 0;
};

NumericalMetrics compute_metrics(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::runtime_error("compute_metrics: mismatched or empty spans");
    }

    double max_err = 0.0;
    double sum_err = 0.0;
    double sum_sq_err = 0.0;
    double sum_sq_exp = 0.0;
    double dot_prod = 0.0;
    double sum_sq_act = 0.0;
    std::size_t worst_idx = 0;
    bool finite = true;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = actual[i];
        const float e = expected[i];
        if (!std::isfinite(a) || !std::isfinite(e)) {
            finite = false;
        }
        const double diff = std::fabs(static_cast<double>(a) - static_cast<double>(e));
        if (diff > max_err) {
            max_err = diff;
            worst_idx = i;
        }
        sum_err += diff;
        sum_sq_err += diff * diff;
        sum_sq_exp += static_cast<double>(e) * static_cast<double>(e);
        sum_sq_act += static_cast<double>(a) * static_cast<double>(a);
        dot_prod += static_cast<double>(a) * static_cast<double>(e);
    }

    const double n = static_cast<double>(actual.size());
    const double mae = sum_err / n;
    const double rms = std::sqrt(sum_sq_err / n);
    const double exp_rms = std::sqrt(sum_sq_exp / n);
    const double rel_rms = exp_rms > 1.0e-12 ? (rms / exp_rms) : rms;
    const double norm_product = std::sqrt(sum_sq_act) * std::sqrt(sum_sq_exp);
    const double cosine = norm_product > 1.0e-12 ? (dot_prod / norm_product) : 1.0;

    return NumericalMetrics{
        static_cast<float>(max_err),
        static_cast<float>(mae),
        static_cast<float>(rms),
        static_cast<float>(rel_rms),
        static_cast<float>(cosine),
        finite,
        worst_idx
    };
}

std::vector<float> generate_realistic_input(std::size_t tokens, std::size_t hidden) {
    std::vector<float> input(tokens * hidden);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < hidden; ++h) {
            // Sinusoidal multi-frequency pattern to stress dynamic ranges and activation paths
            const float ft = static_cast<float>(t);
            const float fh = static_cast<float>(h);
            input[t * hidden + h] = 0.05F * std::sin(0.0017F * fh + 0.071F * ft)
                                  + 0.02F * std::cos(0.0031F * fh + 0.013F * ft);
        }
    }
    return input;
}

std::vector<float> generate_initial_state() {
    std::vector<float> state(RecurrentLayerState::kStateElements);
    for (std::size_t i = 0; i < state.size(); ++i) {
        const float fi = static_cast<float>(i);
        state[i] = 0.001F * std::sin(0.0005F * fi + 0.1F);
    }
    return state;
}

std::vector<float> generate_initial_history() {
    std::vector<float> history(RecurrentLayerState::kConvHistoryElements);
    for (std::size_t i = 0; i < history.size(); ++i) {
        const float fi = static_cast<float>(i);
        history[i] = 0.002F * std::cos(0.001F * fi + 0.2F);
    }
    return history;
}

struct LayerEvaluationResult {
    std::uint32_t tokens = 0;
    std::uint32_t internal_chunks = 0;

    // Correctness
    NumericalMetrics output_metrics{};
    NumericalMetrics state_metrics{};
    NumericalMetrics history_metrics{};

    // Performance (ms)
    double v1_token_oracle_ms = 0.0;
    double v2_median_ms = 0.0;
    double v2_min_ms = 0.0;
    double v2_max_ms = 0.0;
    double v2_tok_s = 0.0;
    double speedup_vs_token_oracle = 0.0;

    RecurrentLayerPhaseTimings v2_breakdown{};
};

LayerEvaluationResult evaluate_token_count(
    const miinfer::Qwen35Model& model,
    std::size_t layer_index,
    std::uint32_t token_count,
    RecurrentLayer& v1_layer,
    PrefillV2RecurrentLayer& v2_layer,
    RecurrentLayerWorkspaceManager& ws_manager) {

    LayerEvaluationResult res;
    res.tokens = token_count;
    res.internal_chunks = token_count / kGdnChunkSize;

    const auto host_input = generate_realistic_input(token_count, kHidden);
    const auto initial_state = generate_initial_state();
    const auto initial_history = generate_initial_history();

    // -------------------------------------------------------------
    // 1. Run V1 Canonical Token-by-Token Oracle
    // -------------------------------------------------------------
    v1_layer.reset();
    v1_layer.upload_state(initial_state);
    upload(initial_history.data(), v1_layer.history->get(), initial_history.size() * sizeof(float));

    float* d_v1_in = nullptr;
    float* d_v1_out = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_v1_in), kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_v1_out), token_count * kHidden * sizeof(float)));

    const auto v1_start = std::chrono::steady_clock::now();
    for (std::uint32_t t = 0; t < token_count; ++t) {
        upload(host_input.data() + t * kHidden, d_v1_in, kHidden * sizeof(float));
        v1_layer.run(d_v1_in, t, d_v1_out + t * kHidden);
    }
    MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    const auto v1_end = std::chrono::steady_clock::now();
    res.v1_token_oracle_ms = std::chrono::duration<double, std::milli>(v1_end - v1_start).count();

    const auto oracle_output = download(d_v1_out, token_count * kHidden);
    const auto oracle_state = v1_layer.logical_state();
    const auto oracle_history = download(v1_layer.history->get(), RecurrentLayerState::kConvHistoryElements);

    (void)hipFree(d_v1_in);
    (void)hipFree(d_v1_out);

    // -------------------------------------------------------------
    // 2. Run Prefill V2 Single-Layer Forward
    // -------------------------------------------------------------
    float* d_v2_input = nullptr;
    float* d_v2_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_v2_input), token_count * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_v2_output), token_count * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMemcpy(d_v2_input, host_input.data(),
                                token_count * kHidden * sizeof(float), hipMemcpyHostToDevice));

    RecurrentLayerStateStorage v2_state_storage;
    v2_state_storage.upload(initial_state, initial_history, 0);
    auto v2_state_view = v2_state_storage.view();

    auto ws = ws_manager.workspace();

    // Verification run
    v2_layer.forward(
        d_v2_input, d_v2_output, v2_state_view, v2_state_view, ws, token_count, hipStreamPerThread);
    MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    std::vector<float> v2_output(token_count * kHidden);
    MIINFER_HIP_CHECK(hipMemcpy(v2_output.data(), d_v2_output,
                                token_count * kHidden * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<float> v2_state_downloaded, v2_history_downloaded;
    v2_state_storage.download(v2_state_downloaded, v2_history_downloaded);

    // -------------------------------------------------------------
    // 3. Compare Numerical Metrics
    // -------------------------------------------------------------
    res.output_metrics = compute_metrics(v2_output, oracle_output);
    res.state_metrics = compute_metrics(v2_state_downloaded, oracle_state);
    res.history_metrics = compute_metrics(v2_history_downloaded, oracle_history);

    // Stage-by-stage comparison for N=64
    if (token_count == 64) {
        std::vector<float> v2_norm(token_count * kHidden);
        std::vector<float> v2_qkv(token_count * kChannels);
        std::vector<float> v2_gate(token_count * kInner);
        std::vector<float> v2_beta(token_count * kVHeads);
        std::vector<float> v2_decay(token_count * kVHeads);
        std::vector<float> v2_gdn_out(token_count * kVHeads * kState);
        std::vector<float> v2_gated(token_count * kVHeads * kState);
        std::vector<float> v2_ssm_out(token_count * kHidden);
        std::vector<float> v2_residual(token_count * kHidden);
        std::vector<float> v2_post_norm(token_count * kHidden);
        std::vector<float> v2_ffn_gate(token_count * kFfnInner);
        std::vector<float> v2_ffn_up(token_count * kFfnInner);
        std::vector<float> v2_ffn_down(token_count * kHidden);

        MIINFER_HIP_CHECK(hipMemcpy(v2_norm.data(), ws.normalized, v2_norm.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_qkv.data(), ws.qkv, v2_qkv.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_gate.data(), ws.gate, v2_gate.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_beta.data(), ws.beta, v2_beta.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_decay.data(), ws.decay, v2_decay.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_gdn_out.data(), ws.gdn_raw_output, v2_gdn_out.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_gated.data(), ws.gated_output, v2_gated.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_ssm_out.data(), ws.ssm_output, v2_ssm_out.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_residual.data(), ws.residual, v2_residual.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_post_norm.data(), ws.post_normalized, v2_post_norm.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_ffn_gate.data(), ws.ffn_gate, v2_ffn_gate.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_ffn_up.data(), ws.ffn_up, v2_ffn_up.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(v2_ffn_down.data(), ws.ffn_down, v2_ffn_down.size() * sizeof(float), hipMemcpyDeviceToHost));

        // Let's run V1 single token 0 to inspect stage errors at token 0:
        v1_layer.reset();
        v1_layer.upload_state(initial_state);
        upload(initial_history.data(), v1_layer.history->get(), initial_history.size() * sizeof(float));
        float* d_in0 = nullptr; float* d_out0 = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_in0), kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out0), kHidden * sizeof(float)));
        upload(host_input.data(), d_in0, kHidden * sizeof(float));
        v1_layer.run(d_in0, 0, d_out0);

        const auto v1_norm0 = download(v1_layer.normalized->get(), kHidden);
        const auto v1_qkv0 = download(v1_layer.qkv->get(), kChannels);
        const auto v1_gate0 = download(v1_layer.gate->get(), kInner);
        const auto v1_beta0 = download(v1_layer.beta->get(), kVHeads);
        const auto v1_decay0 = download(v1_layer.decay->get(), kVHeads);
        const auto v1_gdn_out0 = download(v1_layer.recurrent_output->get(), kVHeads * kState);
        const auto v1_gated0 = download(v1_layer.gated->get(), kVHeads * kState);
        const auto v1_ssm_out0 = download(v1_layer.projected->get(), kHidden);
        const auto v1_residual0 = download(v1_layer.residual->get(), kHidden);
        const auto v1_post_norm0 = download(v1_layer.post_normalized->get(), kHidden);
        const auto v1_ffn_gate0 = download(v1_layer.ffn_gate->get(), kFfnInner);
        const auto v1_ffn_up0 = download(v1_layer.ffn_up->get(), kFfnInner);
        const auto v1_ffn_down0 = download(v1_layer.layer_output->get(), kHidden);

        std::cout << "\n    --- Stage Attribution Probe at Token 0 (N=64) ---\n";
        std::cout << "      attn_norm max_err:    " << compute_metrics(std::span<const float>(v2_norm.data(), kHidden), v1_norm0).max_abs_err << "\n";
        std::cout << "      beta max_err:         " << compute_metrics(std::span<const float>(v2_beta.data(), kVHeads), v1_beta0).max_abs_err << "\n";
        std::cout << "      decay max_err:        " << compute_metrics(std::span<const float>(v2_decay.data(), kVHeads), v1_decay0).max_abs_err << "\n";
        std::cout << "      qkv proj max_err:     " << compute_metrics(std::span<const float>(v2_qkv.data(), kChannels), v1_qkv0).max_abs_err << "\n";
        std::cout << "      gate proj max_err:    " << compute_metrics(std::span<const float>(v2_gate.data(), kInner), v1_gate0).max_abs_err << "\n";
        std::cout << "      gdn raw out max_err:  " << compute_metrics(std::span<const float>(v2_gdn_out.data(), kVHeads * kState), v1_gdn_out0).max_abs_err << "\n";
        std::cout << "      gated ssm max_err:    " << compute_metrics(std::span<const float>(v2_gated.data(), kVHeads * kState), v1_gated0).max_abs_err << "\n";
        std::cout << "      ssm out proj max_err: " << compute_metrics(std::span<const float>(v2_ssm_out.data(), kHidden), v1_ssm_out0).max_abs_err << "\n";
        std::cout << "      residual max_err:     " << compute_metrics(std::span<const float>(v2_residual.data(), kHidden), v1_residual0).max_abs_err << "\n";
        std::cout << "      post_norm max_err:    " << compute_metrics(std::span<const float>(v2_post_norm.data(), kHidden), v1_post_norm0).max_abs_err << "\n";
        std::cout << "      ffn gate max_err:     " << compute_metrics(std::span<const float>(v2_ffn_gate.data(), kFfnInner), v1_ffn_gate0).max_abs_err << "\n";
        std::cout << "      ffn up max_err:       " << compute_metrics(std::span<const float>(v2_ffn_up.data(), kFfnInner), v1_ffn_up0).max_abs_err << "\n";
        std::cout << "      ffn down max_err:     " << compute_metrics(std::span<const float>(v2_ffn_down.data(), kHidden), v1_ffn_down0).max_abs_err << "\n";
        std::cout << "    -------------------------------------------------\n\n";

        (void)hipFree(d_in0);
        (void)hipFree(d_out0);
    }

    // -------------------------------------------------------------
    // 4. Performance Benchmarking (HIP Events)
    // -------------------------------------------------------------
    constexpr int kWarmup = 3;
    constexpr int kMeasured = 10;

    for (int w = 0; w < kWarmup; ++w) {
        v2_state_storage.upload(initial_state, initial_history, 0);
        v2_layer.forward(
            d_v2_input, d_v2_output, v2_state_view, v2_state_view, ws, token_count, hipStreamPerThread);
    }
    MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    hipEvent_t ev_start, ev_stop;
    MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_stop));

    std::vector<double> v2_samples(kMeasured);
    for (int iter = 0; iter < kMeasured; ++iter) {
        v2_state_storage.upload(initial_state, initial_history, 0);
        MIINFER_HIP_CHECK(hipEventRecord(ev_start, hipStreamPerThread));
        v2_layer.forward(
            d_v2_input, d_v2_output, v2_state_view, v2_state_view, ws, token_count, hipStreamPerThread);
        MIINFER_HIP_CHECK(hipEventRecord(ev_stop, hipStreamPerThread));
        MIINFER_HIP_CHECK(hipEventSynchronize(ev_stop));

        float ms = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        v2_samples[iter] = static_cast<double>(ms);
    }
    (void)hipEventDestroy(ev_start);
    (void)hipEventDestroy(ev_stop);

    std::sort(v2_samples.begin(), v2_samples.end());
    res.v2_min_ms = v2_samples.front();
    res.v2_max_ms = v2_samples.back();
    res.v2_median_ms = v2_samples[kMeasured / 2];
    res.v2_tok_s = (static_cast<double>(token_count) / res.v2_median_ms) * 1000.0;
    res.speedup_vs_token_oracle = res.v1_token_oracle_ms / res.v2_median_ms;

    // Profiled phase attribution run
    v2_state_storage.upload(initial_state, initial_history, 0);
    v2_layer.forward_profiled(
        d_v2_input, d_v2_output, v2_state_view, v2_state_view, ws, token_count, res.v2_breakdown, hipStreamPerThread);

    (void)hipFree(d_v2_input);
    (void)hipFree(d_v2_output);

    return res;
}

} // namespace

int main(int argc, char** argv) try {
    std::string model_path = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";
    std::size_t layer_index = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--layer" && i + 1 < argc) {
            layer_index = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg[0] != '-') {
            model_path = arg;
        }
    }

    std::cout << "========================================================================\n";
    std::cout << "  MIInfer Prefill V2 Recurrent Layer Vertical Slice Qualification\n";
    std::cout << "========================================================================\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Layer: " << layer_index << " (Recurrent Gated DeltaNet)\n";

    miinfer::DeviceInfo device;
    std::string device_err;
    if (!miinfer::validate_gfx906_device(-1, device, device_err)) {
        throw std::runtime_error("Device validation failed: " + device_err);
    }
    std::cout << "Device: " << device.name << " (" << device.architecture << ", 60 CUs, Wave64)\n\n";

    const auto model = miinfer::Qwen35Model::load(model_path);

    std::cout << "Loading V1 Oracle layer and Prefill V2 layer...\n";
    RecurrentLayer v1_layer(model, layer_index, {});
    PrefillV2RecurrentLayer v2_layer(model, layer_index);
    RecurrentLayerWorkspaceManager ws_manager(kMaxPrefillBatch);

    std::cout << "Prefill V2 Persistent Weights: "
              << (v2_layer.persistent_weight_bytes() / (1024 * 1024)) << " MiB\n";
    std::cout << "Prefill V2 Scratch Workspace:   "
              << (ws_manager.total_workspace_bytes() / (1024 * 1024)) << " MiB\n\n";

    std::vector<std::uint32_t> test_tokens = {64, 128, 512};
    std::vector<LayerEvaluationResult> results;

    for (const auto tokens : test_tokens) {
        std::cout << ">>> Running Evaluation for N = " << tokens
                  << " tokens (" << (tokens / kGdnChunkSize) << " x C64 internal chunks) ...\n";
        auto res = evaluate_token_count(model, layer_index, tokens, v1_layer, v2_layer, ws_manager);
        results.push_back(res);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "    [Correctness]\n";
        std::cout << "      Output:  max_abs=" << res.output_metrics.max_abs_err
                  << "  rmse=" << res.output_metrics.rmse
                  << "  rel_rms=" << res.output_metrics.rel_rms
                  << "  cosine=" << res.output_metrics.cosine_similarity
                  << "  finite=" << (res.output_metrics.all_finite ? "YES" : "NO") << "\n";
        std::cout << "      State:   max_abs=" << res.state_metrics.max_abs_err
                  << "  rmse=" << res.state_metrics.rmse
                  << "  rel_rms=" << res.state_metrics.rel_rms << "\n";
        std::cout << "      History: max_abs=" << res.history_metrics.max_abs_err
                  << "  rmse=" << res.history_metrics.rmse << "\n";

        std::cout << std::setprecision(3);
        std::cout << "    [Performance]\n";
        std::cout << "      V1 Token Oracle: " << res.v1_token_oracle_ms << " ms\n";
        std::cout << "      Prefill V2 (med): " << res.v2_median_ms << " ms (min="
                  << res.v2_min_ms << " ms, max=" << res.v2_max_ms << " ms)\n";
        std::cout << "      Speedup vs V1:   " << res.speedup_vs_token_oracle << "x\n";
        std::cout << "      V2 Throughput:   " << res.v2_tok_s << " tok/s\n";
        std::cout << "      V2 ms / token:   " << (res.v2_median_ms / tokens) << " ms/tok\n";
        std::cout << "    [Phase Breakdown]\n";
        std::cout << "      Norm + Beta/Alpha: " << res.v2_breakdown.norm_beta_alpha_ms << " ms\n";
        std::cout << "      QKV + Gate Proj:   " << res.v2_breakdown.qkv_gate_proj_ms << " ms\n";
        std::cout << "      Conv + L2 Norm:    " << res.v2_breakdown.conv_l2_norm_ms << " ms\n";
        std::cout << "      GDN Chunkwise C64: " << res.v2_breakdown.gdn_chunkwise_ms << " ms\n";
        std::cout << "      SSM Post + Out:    " << res.v2_breakdown.ssm_post_out_ms << " ms\n";
        std::cout << "      Residual + PostNorm:" << res.v2_breakdown.residual_norm_ms << " ms\n";
        std::cout << "      FFN Gate + Up:     " << res.v2_breakdown.ffn_gate_up_ms << " ms\n";
        std::cout << "      SwiGLU + Down+Add: " << res.v2_breakdown.swiglu_down_residual_ms << " ms\n\n";
    }

    // Print summary table
    std::cout << "=========================================================================================\n";
    std::cout << "                                  SUMMARY RESULTS TABLE\n";
    std::cout << "=========================================================================================\n";
    std::cout << std::setw(8) << "Tokens"
              << std::setw(10) << "Chunks"
              << std::setw(16) << "Out MaxAbsErr"
              << std::setw(14) << "Out Cosine"
              << std::setw(14) << "V1 Oracle ms"
              << std::setw(14) << "Prefill V2 ms"
              << std::setw(10) << "Speedup"
              << std::setw(14) << "V2 Tok/s"
              << "\n";
    std::cout << "-----------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::cout << std::setw(8) << r.tokens
                  << std::setw(10) << r.internal_chunks
                  << std::setw(16) << std::scientific << std::setprecision(3) << r.output_metrics.max_abs_err
                  << std::setw(14) << std::fixed << std::setprecision(6) << r.output_metrics.cosine_similarity
                  << std::setw(14) << std::setprecision(3) << r.v1_token_oracle_ms
                  << std::setw(14) << r.v2_median_ms
                  << std::setw(9) << std::setprecision(2) << r.speedup_vs_token_oracle << "x"
                  << std::setw(14) << std::setprecision(1) << r.v2_tok_s
                  << "\n";
    }
    std::cout << "=========================================================================================\n";

    return 0;
} catch (const std::exception& e) {
    std::cerr << "FATAL ERROR: " << e.what() << "\n";
    return 1;
}
