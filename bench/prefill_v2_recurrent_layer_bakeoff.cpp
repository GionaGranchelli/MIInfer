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

namespace miinfer::prefill_v2 {

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
            const float ft = static_cast<float>(t);
            const float fh = static_cast<float>(h);
            input[t * hidden + h] = 0.05F * std::sin(0.0017F * fh + 0.071F * ft)
                                 + 0.02F * std::cos(0.0053F * fh * (ft + 1.0F));
        }
    }
    return input;
}

std::vector<float> generate_initial_state() {
    constexpr std::size_t kTotalStateElements = kVHeads * kState * kState; // 48 * 128 * 128 = 786,432
    std::vector<float> state(kTotalStateElements);
    for (std::size_t i = 0; i < kTotalStateElements; ++i) {
        const float fi = static_cast<float>(i);
        state[i] = 0.001F * std::sin(0.0003F * fi + 0.1F);
    }
    return state;
}

std::vector<float> generate_initial_history() {
    constexpr std::size_t kTotalHistoryElements = 4 * kChannels; // 4 * 10240 = 40,960
    std::vector<float> history(kTotalHistoryElements);
    for (std::size_t i = 0; i < kTotalHistoryElements; ++i) {
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
    double v1_fast_median_ms = 0.0;
    double v1_fast_min_ms = 0.0;
    double v1_fast_max_ms = 0.0;

    double v2_median_ms = 0.0;
    double v2_min_ms = 0.0;
    double v2_max_ms = 0.0;
    double v2_tok_s = 0.0;

    double speedup_vs_token_oracle = 0.0;
    double speedup_vs_fast_v1 = 0.0;

    RecurrentLayerPhaseTimings v2_breakdown{};
};

LayerEvaluationResult evaluate_token_count(
    const miinfer::Qwen35Model& model,
    std::size_t layer_index,
    std::uint32_t token_count,
    RecurrentLayer& v1_oracle_layer,
    RecurrentLayer& v1_fast_layer,
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
    v1_oracle_layer.reset();
    v1_oracle_layer.upload_state(initial_state);
    upload(initial_history.data(), v1_oracle_layer.history->get(), initial_history.size() * sizeof(float));

    float* d_v1_in = nullptr;
    float* d_v1_out = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_v1_in), kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_v1_out), token_count * kHidden * sizeof(float)));

    const auto v1_start = std::chrono::steady_clock::now();
    for (std::uint32_t t = 0; t < token_count; ++t) {
        upload(host_input.data() + t * kHidden, d_v1_in, kHidden * sizeof(float));
        v1_oracle_layer.run(d_v1_in, t, d_v1_out + t * kHidden);
    }
    MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    const auto v1_end = std::chrono::steady_clock::now();
    res.v1_token_oracle_ms = std::chrono::duration<double, std::milli>(v1_end - v1_start).count();

    std::vector<float> oracle_output(token_count * kHidden);
    MIINFER_HIP_CHECK(hipMemcpy(oracle_output.data(), d_v1_out,
                                token_count * kHidden * sizeof(float), hipMemcpyDeviceToHost));
    const auto oracle_state = download(v1_oracle_layer.state->get(), kVHeads * kState * kState);
    const auto oracle_history = download(v1_oracle_layer.history->get(), 4 * kChannels);

    (void)hipFree(d_v1_in);
    (void)hipFree(d_v1_out);

    constexpr int kWarmup = 3;
    constexpr int kMeasured = 10;

    // -------------------------------------------------------------
    // 2. Run FAST_V1 (B512 Mx Repacked Prefill Path)
    // -------------------------------------------------------------
    if (token_count == 512) {
        float* d_fast_in = nullptr;
        float* d_fast_out = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_fast_in), token_count * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_fast_out), token_count * kHidden * sizeof(float)));
        upload(host_input.data(), d_fast_in, token_count * kHidden * sizeof(float));

        for (int w = 0; w < kWarmup; ++w) {
            v1_fast_layer.upload_state(initial_state);
            upload(initial_history.data(), v1_fast_layer.history->get(), initial_history.size() * sizeof(float));
            v1_fast_layer.prefill_wide(d_fast_in, d_fast_out, 0, token_count);
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

        hipEvent_t ev_start, ev_stop;
        MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
        MIINFER_HIP_CHECK(hipEventCreate(&ev_stop));

        std::vector<double> fast_samples(kMeasured);
        for (int iter = 0; iter < kMeasured; ++iter) {
            v1_fast_layer.upload_state(initial_state);
            upload(initial_history.data(), v1_fast_layer.history->get(), initial_history.size() * sizeof(float));
            MIINFER_HIP_CHECK(hipEventRecord(ev_start, hipStreamPerThread));
            v1_fast_layer.prefill_wide(d_fast_in, d_fast_out, 0, token_count);
            MIINFER_HIP_CHECK(hipEventRecord(ev_stop, hipStreamPerThread));
            MIINFER_HIP_CHECK(hipEventSynchronize(ev_stop));

            float ms = 0.0F;
            MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
            fast_samples[iter] = static_cast<double>(ms);
        }
        (void)hipEventDestroy(ev_start);
        (void)hipEventDestroy(ev_stop);

        std::sort(fast_samples.begin(), fast_samples.end());
        res.v1_fast_min_ms = fast_samples.front();
        res.v1_fast_max_ms = fast_samples.back();
        res.v1_fast_median_ms = fast_samples[kMeasured / 2];

        (void)hipFree(d_fast_in);
        (void)hipFree(d_fast_out);
    } else {
        res.v1_fast_min_ms = 0.0;
        res.v1_fast_max_ms = 0.0;
        res.v1_fast_median_ms = 0.0;
    }

    // -------------------------------------------------------------
    // 3. Run Prefill V2 (with Mx Compact MMQ)
    // -------------------------------------------------------------
    float* d_v2_input = nullptr;
    float* d_v2_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_v2_input), token_count * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_v2_output), token_count * kHidden * sizeof(float)));
    upload(host_input.data(), d_v2_input, token_count * kHidden * sizeof(float));

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
    // 4. Compare Numerical Metrics
    // -------------------------------------------------------------
    res.output_metrics = compute_metrics(v2_output, oracle_output);
    res.state_metrics = compute_metrics(v2_state_downloaded, oracle_state);
    res.history_metrics = compute_metrics(v2_history_downloaded, oracle_history);

    // -------------------------------------------------------------
    // 5. Performance Benchmarking (HIP Events)
    // -------------------------------------------------------------
    for (int w = 0; w < kWarmup; ++w) {
        v2_state_storage.upload(initial_state, initial_history, 0);
        v2_layer.forward(
            d_v2_input, d_v2_output, v2_state_view, v2_state_view, ws, token_count, hipStreamPerThread);
    }
    MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    hipEvent_t ev_v2_start, ev_v2_stop;
    MIINFER_HIP_CHECK(hipEventCreate(&ev_v2_start));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_v2_stop));

    std::vector<double> v2_samples(kMeasured);
    for (int iter = 0; iter < kMeasured; ++iter) {
        v2_state_storage.upload(initial_state, initial_history, 0);
        MIINFER_HIP_CHECK(hipEventRecord(ev_v2_start, hipStreamPerThread));
        v2_layer.forward(
            d_v2_input, d_v2_output, v2_state_view, v2_state_view, ws, token_count, hipStreamPerThread);
        MIINFER_HIP_CHECK(hipEventRecord(ev_v2_stop, hipStreamPerThread));
        MIINFER_HIP_CHECK(hipEventSynchronize(ev_v2_stop));

        float ms = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_v2_start, ev_v2_stop));
        v2_samples[iter] = static_cast<double>(ms);
    }
    (void)hipEventDestroy(ev_v2_start);
    (void)hipEventDestroy(ev_v2_stop);

    std::sort(v2_samples.begin(), v2_samples.end());
    res.v2_min_ms = v2_samples.front();
    res.v2_max_ms = v2_samples.back();
    res.v2_median_ms = v2_samples[kMeasured / 2];
    res.v2_tok_s = (static_cast<double>(token_count) / res.v2_median_ms) * 1000.0;
    res.speedup_vs_token_oracle = res.v1_token_oracle_ms / res.v2_median_ms;
    res.speedup_vs_fast_v1 = res.v1_fast_median_ms / res.v2_median_ms;

    // Profiled phase attribution run
    v2_state_storage.upload(initial_state, initial_history, 0);
    v2_layer.forward_profiled(
        d_v2_input, d_v2_output, v2_state_view, v2_state_view, ws, token_count, res.v2_breakdown, hipStreamPerThread);

    (void)hipFree(d_v2_input);
    (void)hipFree(d_v2_output);

    return res;
}

void print_result(const LayerEvaluationResult& r, const char* label) {
    std::cout << "\n=======================================================\n";
    std::cout << "  Evaluation: " << label << " | Tokens: " << r.tokens
              << " (" << r.internal_chunks << " x C64 chunks)\n";
    std::cout << "=======================================================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Numerical Metrics vs Canonical Oracle:\n";
    std::cout << "    Output:     Cosine = " << r.output_metrics.cosine_similarity
              << " | MaxErr = " << r.output_metrics.max_abs_err
              << " | MAE = " << r.output_metrics.mean_abs_err
              << " | RMS = " << r.output_metrics.rmse
              << " | RelRMS = " << r.output_metrics.rel_rms
              << " | Finite = " << (r.output_metrics.all_finite ? "YES" : "NO") << "\n";
    std::cout << "    State:      Cosine = " << r.state_metrics.cosine_similarity
              << " | MaxErr = " << r.state_metrics.max_abs_err
              << " | RelRMS = " << r.state_metrics.rel_rms << "\n";
    std::cout << "    Conv Hist:  Cosine = " << r.history_metrics.cosine_similarity
              << " | MaxErr = " << r.history_metrics.max_abs_err << "\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n  Performance Comparison (N=" << r.tokens << "):\n";
    std::cout << "    V1 Token Oracle:    " << std::setw(8) << r.v1_token_oracle_ms << " ms\n";
    std::cout << "    FAST_V1 (B512 Mx):  " << std::setw(8) << r.v1_fast_median_ms
              << " ms (min=" << r.v1_fast_min_ms << ", max=" << r.v1_fast_max_ms << ")\n";
    std::cout << "    Prefill V2 (Mx):    " << std::setw(8) << r.v2_median_ms
              << " ms (min=" << r.v2_min_ms << ", max=" << r.v2_max_ms << ")\n";
    std::cout << "    Speedup vs Oracle:  " << std::setw(8) << r.speedup_vs_token_oracle << "x\n";
    std::cout << "    Speedup vs FAST_V1: " << std::setw(8) << r.speedup_vs_fast_v1 << "x\n";
    std::cout << "    Throughput:         " << std::setw(8) << r.v2_tok_s << " tok/s ("
              << (r.v2_median_ms / static_cast<double>(r.tokens)) << " ms/tok)\n";

    if (r.tokens == 512) {
        std::cout << "\n  V2 Execution Phase Breakdown (N=512, Total=" << r.v2_breakdown.total_layer_ms << " ms):\n";
        const double tot = r.v2_breakdown.total_layer_ms;
        const auto print_ph = [&](const char* name, double ms) {
            std::cout << "    " << std::left << std::setw(32) << name << ": "
                      << std::right << std::setw(6) << ms << " ms ("
                      << std::setw(5) << (ms / tot * 100.0) << "%)\n";
        };
        print_ph("Phase 1: Norm & Beta/Alpha Prep", r.v2_breakdown.norm_beta_alpha_ms);
        print_ph("Phase 2: QKV & Gate Proj (Mx MMQ)", r.v2_breakdown.qkv_gate_proj_ms);
        print_ph("Phase 3: Conv1D & L2 Norm", r.v2_breakdown.conv_l2_norm_ms);
        print_ph("Phase 4: Chunkwise GDN Core (C64)", r.v2_breakdown.gdn_chunkwise_ms);
        print_ph("Phase 5: SSM Post & Out (Q5_K)", r.v2_breakdown.ssm_post_out_ms);
        print_ph("Phase 6: Residual & Post Norm", r.v2_breakdown.residual_norm_ms);
        print_ph("Phase 7: FFN Gate/Up (Q4_K MMQ)", r.v2_breakdown.ffn_gate_up_ms);
        print_ph("Phase 8: SwiGLU, Down & Residual", r.v2_breakdown.swiglu_down_residual_ms);
    }
}

int run_bakeoff_main(int argc, char** argv) {
    std::string model_path = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    DeviceInfo dev_info{};
    std::string dev_err;
    if (!validate_gfx906_device(0, dev_info, dev_err)) {
        std::cerr << "Device validation failed: " << dev_err << "\n";
        return 1;
    }
    print_device_info(dev_info, std::cout);

    // 0. Set environment variables to enable FAST_V1 Mx configuration for V1
    setenv("MIINFER_PREFILL_LAYER_MAJOR", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_CHUNK", "1", 1);
    setenv("MIINFER_PREFILL_CHUNK", "512", 1);
    setenv("MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_MMQ_FFN", "1", 1);
    setenv("MIINFER_PREFILL_REPACKED_RESIDENT_ALL", "1", 1);
    setenv("MIINFER_PREFILL_REPACKED_RESIDENT_FFN", "1", 1);

    const auto model = miinfer::Qwen35Model::load(model_path);
    std::cout << "Model loaded: layers=" << model.config().block_count
              << ", hidden=" << model.config().hidden_size
              << ", ffn=" << model.config().intermediate_size << "\n\n";

    RecurrentLayerWorkspaceManager ws_manager(kMaxPrefillBatch);
    std::cout << "V2 Shared Workspace: " << (ws_manager.total_workspace_bytes() / (1024.0 * 1024.0)) << " MiB allocated\n";

    miinfer::RocblasGemmHandle rocblas_handle;
    __half* d_dense_weights = nullptr;
    __half* d_dense_input = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_dense_weights),
                                std::max(kChannels, kFfnInner) * kHidden * sizeof(__half)));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_dense_input),
                                kMaxPrefillBatch * kHidden * sizeof(__half)));

    const std::size_t gdn_bytes = kVHeads * 64 * kState * sizeof(float);
    float *d_gdn_new = nullptr, *d_gdn_decayed = nullptr, *d_gdn_solved_values = nullptr;
    float *d_gdn_solved_keys = nullptr, *d_gdn_corrected = nullptr, *d_gdn_raw_output = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_gdn_new), gdn_bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_gdn_decayed), gdn_bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_gdn_solved_values), gdn_bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_gdn_solved_keys), gdn_bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_gdn_corrected), gdn_bytes));
    MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_gdn_raw_output),
                                kMaxPrefillBatch * kVHeads * kState * sizeof(float)));
    const miinfer::M12GdnChunkWorkspace gdn_ws{
        d_gdn_new, d_gdn_decayed, d_gdn_solved_values, d_gdn_solved_keys, d_gdn_corrected
    };

    auto setup_v1_fast = [&](RecurrentLayer& layer) {
        layer.set_m12_dense_workspace(d_dense_weights, d_dense_input, &rocblas_handle);
        layer.set_m12_gdn_workspace(gdn_ws, d_gdn_raw_output);
        layer.ensure_mx_repacked();
    };

    // -------------------------------------------------------------
    // Part A: Signature Audit & Representative Layer Evaluation
    // -------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Part A: Representative Layer Signatures\n";
    std::cout << "-------------------------------------------------------------------\n";

    // Representative Layer 0 (Signature A: Q6_K QKV / Q6_K FFN-down)
    std::cout << "\nLoading Layer 0 (Representative of Signature A: Q6_K QKV, Q4_K Gate, Q5_K SSM-out, Q4_K FFN-g/u, Q6_K FFN-down)...\n";
    RecurrentLayer v1_oracle_l0(model, 0, {});
    RecurrentLayer v1_fast_l0(model, 0, {});
    setup_v1_fast(v1_fast_l0);
    PrefillV2RecurrentLayer v2_l0(model, 0);
    std::cout << "  V2 Layer 0 Persistent Weights: " << (v2_l0.persistent_weight_bytes() / (1024.0 * 1024.0)) << " MiB\n";

    const auto r0_64 = evaluate_token_count(model, 0, 64, v1_oracle_l0, v1_fast_l0, v2_l0, ws_manager);
    print_result(r0_64, "Layer 0 (Signature A)");

    const auto r0_128 = evaluate_token_count(model, 0, 128, v1_oracle_l0, v1_fast_l0, v2_l0, ws_manager);
    print_result(r0_128, "Layer 0 (Signature A)");

    const auto r0_512 = evaluate_token_count(model, 0, 512, v1_oracle_l0, v1_fast_l0, v2_l0, ws_manager);
    print_result(r0_512, "Layer 0 (Signature A)");

    // Representative Layer 8 (Signature B: Q4_K QKV / Q4_K FFN-down)
    std::cout << "\nLoading Layer 8 (Representative of Signature B: Q4_K QKV, Q4_K Gate, Q5_K SSM-out, Q4_K FFN-g/u, Q4_K FFN-down)...\n";
    RecurrentLayer v1_oracle_l8(model, 8, {});
    RecurrentLayer v1_fast_l8(model, 8, {});
    setup_v1_fast(v1_fast_l8);
    PrefillV2RecurrentLayer v2_l8(model, 8);
    std::cout << "  V2 Layer 8 Persistent Weights: " << (v2_l8.persistent_weight_bytes() / (1024.0 * 1024.0)) << " MiB\n";

    const auto r8_64 = evaluate_token_count(model, 8, 64, v1_oracle_l8, v1_fast_l8, v2_l8, ws_manager);
    print_result(r8_64, "Layer 8 (Signature B)");

    const auto r8_128 = evaluate_token_count(model, 8, 128, v1_oracle_l8, v1_fast_l8, v2_l8, ws_manager);
    print_result(r8_128, "Layer 8 (Signature B)");

    const auto r8_512 = evaluate_token_count(model, 8, 512, v1_oracle_l8, v1_fast_l8, v2_l8, ws_manager);
    print_result(r8_512, "Layer 8 (Signature B)");

    // -------------------------------------------------------------
    // Part B: Mandatory Summary Table (Section 6)
    // -------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Mandatory N=512 Baseline Comparison Table\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "| Layer / Signature | Oracle (ms) | FAST_V1 (ms) | V2-0001 (ms) | V2-Mx (ms) | V2 / FAST_V1 | V2 vs V2-0001 |\n";
    std::cout << "|:---|---:|---:|---:|---:|---:|---:|\n";
    std::cout << "| Layer 0 (Sig A: Q6/Q6) | " << std::setw(9) << r0_512.v1_token_oracle_ms
              << " | " << std::setw(10) << r0_512.v1_fast_median_ms
              << " | " << std::setw(10) << 95.86
              << " | " << std::setw(8) << r0_512.v2_median_ms
              << " | " << std::setw(10) << (r0_512.v2_median_ms / r0_512.v1_fast_median_ms) << "x"
              << " | " << std::setw(11) << (95.86 / r0_512.v2_median_ms) << "x |\n";
    std::cout << "| Layer 8 (Sig B: Q4/Q4) | " << std::setw(9) << r8_512.v1_token_oracle_ms
              << " | " << std::setw(10) << r8_512.v1_fast_median_ms
              << " | " << std::setw(10) << 89.40
              << " | " << std::setw(8) << r8_512.v2_median_ms
              << " | " << std::setw(10) << (r8_512.v2_median_ms / r8_512.v1_fast_median_ms) << "x"
              << " | " << std::setw(11) << (89.40 / r8_512.v2_median_ms) << "x |\n";

    // -------------------------------------------------------------
    // Part C: 3-Recurrent-Layer Chain Semantic Stress Test (L0 -> L1 -> L2)
    // -------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Part C: Three-Recurrent-Layer Chain (L0 -> L1 -> L2) Semantic Stress Test\n";
    std::cout << "-------------------------------------------------------------------\n";

    RecurrentLayer v1_l0(model, 0, {});
    RecurrentLayer v1_l1(model, 1, {});
    RecurrentLayer v1_l2(model, 2, {});
    setup_v1_fast(v1_l0);
    setup_v1_fast(v1_l1);
    setup_v1_fast(v1_l2);

    PrefillV2RecurrentLayer v2_l0_chain(model, 0);
    PrefillV2RecurrentLayer v2_l1_chain(model, 1);
    PrefillV2RecurrentLayer v2_l2_chain(model, 2);

    RecurrentLayerStateStorage v2_state_l0, v2_state_l1, v2_state_l2;

    const auto test_3layer_chain = [&](std::uint32_t N) {
        const auto host_in = generate_realistic_input(N, kHidden);
        const auto s0 = generate_initial_state();
        const auto h0 = generate_initial_history();

        // 1. Run Canonical V1 Chain
        v1_l0.reset(); v1_l0.upload_state(s0); upload(h0.data(), v1_l0.history->get(), h0.size() * sizeof(float));
        v1_l1.reset(); v1_l1.upload_state(s0); upload(h0.data(), v1_l1.history->get(), h0.size() * sizeof(float));
        v1_l2.reset(); v1_l2.upload_state(s0); upload(h0.data(), v1_l2.history->get(), h0.size() * sizeof(float));

        float* d_v1_in = nullptr; float* d_v1_mid1 = nullptr; float* d_v1_mid2 = nullptr; float* d_v1_out = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_v1_in, kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_v1_mid1, kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_v1_mid2, kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_v1_out, N * kHidden * sizeof(float)));

        for (std::uint32_t t = 0; t < N; ++t) {
            upload(host_in.data() + t * kHidden, d_v1_in, kHidden * sizeof(float));
            v1_l0.run(d_v1_in, t, d_v1_mid1);
            v1_l1.run(d_v1_mid1, t, d_v1_mid2);
            v1_l2.run(d_v1_mid2, t, d_v1_out + t * kHidden);
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

        std::vector<float> v1_chain_out(N * kHidden);
        MIINFER_HIP_CHECK(hipMemcpy(v1_chain_out.data(), d_v1_out, N * kHidden * sizeof(float), hipMemcpyDeviceToHost));
        const auto v1_s0 = download(v1_l0.state->get(), kVHeads * kState * kState);
        const auto v1_s1 = download(v1_l1.state->get(), kVHeads * kState * kState);
        const auto v1_s2 = download(v1_l2.state->get(), kVHeads * kState * kState);

        (void)hipFree(d_v1_in); (void)hipFree(d_v1_mid1); (void)hipFree(d_v1_mid2); (void)hipFree(d_v1_out);

        // 2. Run V2 Chain with shared ping/pong activations and shared workspace
        float* d_ping = nullptr; float* d_pong = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_ping, N * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_pong, N * kHidden * sizeof(float)));
        upload(host_in.data(), d_ping, N * kHidden * sizeof(float));

        v2_state_l0.upload(s0, h0, 0); auto view0 = v2_state_l0.view();
        v2_state_l1.upload(s0, h0, 0); auto view1 = v2_state_l1.view();
        v2_state_l2.upload(s0, h0, 0); auto view2 = v2_state_l2.view();

        auto ws = ws_manager.workspace();

        // L0: d_ping -> d_pong
        v2_l0_chain.forward(d_ping, d_pong, view0, view0, ws, N, hipStreamPerThread);
        // L1: d_pong -> d_ping
        v2_l1_chain.forward(d_pong, d_ping, view1, view1, ws, N, hipStreamPerThread);
        // L2: d_ping -> d_pong (final output)
        v2_l2_chain.forward(d_ping, d_pong, view2, view2, ws, N, hipStreamPerThread);
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

        std::vector<float> v2_chain_out(N * kHidden);
        MIINFER_HIP_CHECK(hipMemcpy(v2_chain_out.data(), d_pong, N * kHidden * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<float> v2_s0, v2_h0, v2_s1, v2_h1, v2_s2, v2_h2;
        v2_state_l0.download(v2_s0, v2_h0);
        v2_state_l1.download(v2_s1, v2_h1);
        v2_state_l2.download(v2_s2, v2_h2);

        (void)hipFree(d_ping); (void)hipFree(d_pong);

        const auto out_m = compute_metrics(v2_chain_out, v1_chain_out);
        const auto s0_m = compute_metrics(v2_s0, v1_s0);
        const auto s1_m = compute_metrics(v2_s1, v1_s1);
        const auto s2_m = compute_metrics(v2_s2, v1_s2);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n  3-Layer Chain (L0->L1->L2) at N=" << N << ":\n";
        std::cout << "    Final L2 Output: Cosine = " << out_m.cosine_similarity
                  << " | MaxErr = " << out_m.max_abs_err
                  << " | RelRMS = " << out_m.rel_rms
                  << " | Finite = " << (out_m.all_finite ? "YES" : "NO") << "\n";
        std::cout << "    L0 State:        Cosine = " << s0_m.cosine_similarity << " | RelRMS = " << s0_m.rel_rms << "\n";
        std::cout << "    L1 State:        Cosine = " << s1_m.cosine_similarity << " | RelRMS = " << s1_m.rel_rms << "\n";
        std::cout << "    L2 State:        Cosine = " << s2_m.cosine_similarity << " | RelRMS = " << s2_m.rel_rms << "\n";
    };

    test_3layer_chain(64);
    test_3layer_chain(128);
    test_3layer_chain(512);

    // -------------------------------------------------------------
    // Part D: Stateful Split-Call Invariant Test (512 vs 256+256) & Continuation (512 + 128)
    // -------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Part D: Stateful Split-Call Invariant (512 vs 256+256) & Continuation\n";
    std::cout << "-------------------------------------------------------------------\n";

    const auto host_in512 = generate_realistic_input(512, kHidden);
    const auto s0 = generate_initial_state();
    const auto h0 = generate_initial_history();

    // 1. One-shot 512
    float* d_in512 = nullptr; float* d_out512_oneshot = nullptr; float* d_mid512 = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&d_in512, 512 * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_out512_oneshot, 512 * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_mid512, 512 * kHidden * sizeof(float)));
    upload(host_in512.data(), d_in512, 512 * kHidden * sizeof(float));

    v2_state_l0.upload(s0, h0, 0); auto v0 = v2_state_l0.view();
    v2_state_l1.upload(s0, h0, 0); auto v1 = v2_state_l1.view();
    v2_state_l2.upload(s0, h0, 0); auto v2 = v2_state_l2.view();
    auto ws = ws_manager.workspace();

    v2_l0_chain.forward(d_in512, d_mid512, v0, v0, ws, 512, hipStreamPerThread);
    v2_l1_chain.forward(d_mid512, d_in512, v1, v1, ws, 512, hipStreamPerThread);
    v2_l2_chain.forward(d_in512, d_out512_oneshot, v2, v2, ws, 512, hipStreamPerThread);
    MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    std::vector<float> oneshot_out(512 * kHidden);
    MIINFER_HIP_CHECK(hipMemcpy(oneshot_out.data(), d_out512_oneshot, 512 * kHidden * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<float> oneshot_s0, oneshot_h0, oneshot_s1, oneshot_h1, oneshot_s2, oneshot_h2;
    v2_state_l0.download(oneshot_s0, oneshot_h0);
    v2_state_l1.download(oneshot_s1, oneshot_h1);
    v2_state_l2.download(oneshot_s2, oneshot_h2);

    // 2. Split-call 256 + 256
    float* d_out512_split = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&d_out512_split, 512 * kHidden * sizeof(float)));

    v2_state_l0.upload(s0, h0, 0); v0 = v2_state_l0.view();
    v2_state_l1.upload(s0, h0, 0); v1 = v2_state_l1.view();
    v2_state_l2.upload(s0, h0, 0); v2 = v2_state_l2.view();

    // Chunk 1: tokens 0..255
    upload(host_in512.data(), d_in512, 256 * kHidden * sizeof(float));
    v2_l0_chain.forward(d_in512, d_mid512, v0, v0, ws, 256, hipStreamPerThread);
    v2_l1_chain.forward(d_mid512, d_in512, v1, v1, ws, 256, hipStreamPerThread);
    v2_l2_chain.forward(d_in512, d_out512_split, v2, v2, ws, 256, hipStreamPerThread);

    // Chunk 2: tokens 256..511
    upload(host_in512.data() + 256 * kHidden, d_in512, 256 * kHidden * sizeof(float));
    v2_l0_chain.forward(d_in512, d_mid512, v0, v0, ws, 256, hipStreamPerThread);
    v2_l1_chain.forward(d_mid512, d_in512, v1, v1, ws, 256, hipStreamPerThread);
    v2_l2_chain.forward(d_in512, d_out512_split + 256 * kHidden, v2, v2, ws, 256, hipStreamPerThread);
    MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    std::vector<float> split_out(512 * kHidden);
    MIINFER_HIP_CHECK(hipMemcpy(split_out.data(), d_out512_split, 512 * kHidden * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<float> split_s0, split_h0, split_s1, split_h1, split_s2, split_h2;
    v2_state_l0.download(split_s0, split_h0);
    v2_state_l1.download(split_s1, split_h1);
    v2_state_l2.download(split_s2, split_h2);

    (void)hipFree(d_in512); (void)hipFree(d_out512_oneshot); (void)hipFree(d_mid512); (void)hipFree(d_out512_split);

    const auto split_out_m = compute_metrics(split_out, oneshot_out);
    const auto split_s0_m = compute_metrics(split_s0, oneshot_s0);
    const auto split_s1_m = compute_metrics(split_s1, oneshot_s1);
    const auto split_s2_m = compute_metrics(split_s2, oneshot_s2);
    const auto split_h0_m = compute_metrics(split_h0, oneshot_h0);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Split-call Equivalence: One-Shot 512 vs Split (256 + 256):\n";
    std::cout << "    Output Cosine:      " << split_out_m.cosine_similarity
              << " | MaxErr: " << split_out_m.max_abs_err << " | RelRMS: " << split_out_m.rel_rms << "\n";
    std::cout << "    L0 State Cosine:    " << split_s0_m.cosine_similarity << " | MaxErr: " << split_s0_m.max_abs_err << "\n";
    std::cout << "    L1 State Cosine:    " << split_s1_m.cosine_similarity << " | MaxErr: " << split_s1_m.max_abs_err << "\n";
    std::cout << "    L2 State Cosine:    " << split_s2_m.cosine_similarity << " | MaxErr: " << split_s2_m.max_abs_err << "\n";
    std::cout << "    L0 ConvHist Cosine: " << split_h0_m.cosine_similarity << " | MaxErr: " << split_h0_m.max_abs_err << "\n";

    // -------------------------------------------------------------
    // Part E: 3-Layer Chain Performance Benchmark (FAST_V1 vs V2)
    // -------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Part E: Three-Layer Chain Performance (N=512 Contiguous Timed Interval)\n";
    std::cout << "-------------------------------------------------------------------\n";

    float* d_chain_in = nullptr; float* d_chain_out = nullptr; float* d_chain_mid = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&d_chain_in, 512 * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_chain_out, 512 * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_chain_mid, 512 * kHidden * sizeof(float)));
    upload(host_in512.data(), d_chain_in, 512 * kHidden * sizeof(float));

    hipEvent_t ev_chain_start, ev_chain_stop;
    MIINFER_HIP_CHECK(hipEventCreate(&ev_chain_start));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_chain_stop));

    // 1. Benchmark FAST_V1 3-Layer Chain
    std::vector<double> fast_chain_samples(10);
    for (int iter = 0; iter < 10; ++iter) {
        v1_l0.upload_state(s0); upload(h0.data(), v1_l0.history->get(), h0.size() * sizeof(float));
        v1_l1.upload_state(s0); upload(h0.data(), v1_l1.history->get(), h0.size() * sizeof(float));
        v1_l2.upload_state(s0); upload(h0.data(), v1_l2.history->get(), h0.size() * sizeof(float));

        MIINFER_HIP_CHECK(hipEventRecord(ev_chain_start, hipStreamPerThread));
        v1_l0.prefill_wide(d_chain_in, d_chain_mid, 0, 512);
        v1_l1.prefill_wide(d_chain_mid, d_chain_in, 0, 512);
        v1_l2.prefill_wide(d_chain_in, d_chain_out, 0, 512);
        MIINFER_HIP_CHECK(hipEventRecord(ev_chain_stop, hipStreamPerThread));
        MIINFER_HIP_CHECK(hipEventSynchronize(ev_chain_stop));

        float ms = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_chain_start, ev_chain_stop));
        fast_chain_samples[iter] = static_cast<double>(ms);
    }
    std::sort(fast_chain_samples.begin(), fast_chain_samples.end());
    const double fast_chain_median = fast_chain_samples[5];

    // 2. Benchmark V2 3-Layer Chain
    std::vector<double> v2_chain_samples(10);
    for (int iter = 0; iter < 10; ++iter) {
        v2_state_l0.upload(s0, h0, 0); v0 = v2_state_l0.view();
        v2_state_l1.upload(s0, h0, 0); v1 = v2_state_l1.view();
        v2_state_l2.upload(s0, h0, 0); v2 = v2_state_l2.view();

        MIINFER_HIP_CHECK(hipEventRecord(ev_chain_start, hipStreamPerThread));
        v2_l0_chain.forward(d_chain_in, d_chain_mid, v0, v0, ws, 512, hipStreamPerThread);
        v2_l1_chain.forward(d_chain_mid, d_chain_in, v1, v1, ws, 512, hipStreamPerThread);
        v2_l2_chain.forward(d_chain_in, d_chain_out, v2, v2, ws, 512, hipStreamPerThread);
        MIINFER_HIP_CHECK(hipEventRecord(ev_chain_stop, hipStreamPerThread));
        MIINFER_HIP_CHECK(hipEventSynchronize(ev_chain_stop));

        float ms = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_chain_start, ev_chain_stop));
        v2_chain_samples[iter] = static_cast<double>(ms);
    }
    std::sort(v2_chain_samples.begin(), v2_chain_samples.end());
    const double v2_chain_median = v2_chain_samples[5];

    (void)hipEventDestroy(ev_chain_start);
    (void)hipEventDestroy(ev_chain_stop);
    (void)hipFree(d_chain_in); (void)hipFree(d_chain_out); (void)hipFree(d_chain_mid);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  3-Layer Chain Performance (N=512):\n";
    std::cout << "    FAST_V1 3-Layer Chain: " << fast_chain_median << " ms (min=" << fast_chain_samples.front() << ", max=" << fast_chain_samples.back() << ")\n";
    std::cout << "    Prefill V2 3-Layer Chain: " << v2_chain_median << " ms (min=" << v2_chain_samples.front() << ", max=" << v2_chain_samples.back() << ")\n";
    std::cout << "    Speedup vs FAST_V1:       " << (fast_chain_median / v2_chain_median) << "x\n";

    // -------------------------------------------------------------
    // Part F: Full-Model Memory & Economic Budget (Section 28 & 29)
    // -------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Part F: Full-Model Memory & Economic Budget Projection\n";
    std::cout << "-------------------------------------------------------------------\n";

    const double bytes_sigA = static_cast<double>(v2_l0.persistent_weight_bytes());
    const double bytes_sigB = static_cast<double>(v2_l8.persistent_weight_bytes());
    const double total_recurrent_weights_gb = (24.0 * bytes_sigA + 24.0 * bytes_sigB) / (1024.0 * 1024.0 * 1024.0);
    const double state_per_layer_mb = (RecurrentLayerState::kStateBytes + RecurrentLayerState::kConvHistoryBytes) / (1024.0 * 1024.0);
    const double total_recurrent_states_mb = 48.0 * state_per_layer_mb;
    const double shared_ws_mb = ws_manager.total_workspace_bytes() / (1024.0 * 1024.0);

    std::cout << "  Persistent Weights per Layer:\n";
    std::cout << "    Signature A (Q6/Q6): " << (bytes_sigA / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "    Signature B (Q4/Q4): " << (bytes_sigB / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "  Projected 48 Recurrent Layers Weights: " << total_recurrent_weights_gb << " GiB\n";
    std::cout << "  Projected 48 Recurrent Layer States:  " << total_recurrent_states_mb << " MiB (" << state_per_layer_mb << " MiB/layer)\n";
    std::cout << "  Shared Recurrent Scratch Workspace:   " << shared_ws_mb << " MiB (allocated once)\n";
    std::cout << "  Total Recurrent Static VRAM Footprint: " << (total_recurrent_weights_gb + (total_recurrent_states_mb + shared_ws_mb) / 1024.0) << " GiB / 32 GiB\n";

    // Recurrent-only P512 floor: 24 * Layer0 + 24 * Layer8
    const double recurrent_floor_ms = 24.0 * r0_512.v2_median_ms + 24.0 * r8_512.v2_median_ms;
    std::cout << "\n  Recurrent-Only Full-Model P512 Budget:\n";
    std::cout << "    24 x Signature A (" << r0_512.v2_median_ms << " ms) + 24 x Signature B (" << r8_512.v2_median_ms << " ms)\n";
    std::cout << "    = " << recurrent_floor_ms << " ms (" << (recurrent_floor_ms / 1000.0) << " seconds)\n";
    std::cout << "    Target mx-llama.cpp P512: ~2,310 ms (~2.31 s)\n";
    if (recurrent_floor_ms < 2310.0) {
        std::cout << "    Status: Recurrent floor (" << (recurrent_floor_ms / 1000.0) << " s) leaves ~"
                  << ((2310.0 - recurrent_floor_ms) / 1000.0) << " s for 16 GQA layers to beat mx-llama.cpp!\n";
    } else {
        std::cout << "    Status: Recurrent floor exceeds 2.31 s target.\n";
    }

    std::cout << "\n===================================================================\n";
    std::cout << "  Bakeoff Complete.\n";
    std::cout << "===================================================================\n";

    (void)hipFree(d_dense_weights);
    (void)hipFree(d_dense_input);
    (void)hipFree(d_gdn_new);
    (void)hipFree(d_gdn_decayed);
    (void)hipFree(d_gdn_solved_values);
    (void)hipFree(d_gdn_solved_keys);
    (void)hipFree(d_gdn_corrected);
    (void)hipFree(d_gdn_raw_output);

    return 0;
}

} // namespace miinfer::prefill_v2

int main(int argc, char** argv) {
    return miinfer::prefill_v2::run_bakeoff_main(argc, argv);
}
