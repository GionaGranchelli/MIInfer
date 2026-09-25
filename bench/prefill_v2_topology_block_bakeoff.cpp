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
#include "miinfer/prefill_v2/attention_layer.hpp"
#include "miinfer/prefill_v2/kv_cache.hpp"
#include "miinfer/prefill_v2/recurrent_layer.hpp"
#include "miinfer/prefill_v2/state.hpp"
#include "miinfer/prefill_v2/topology_block.hpp"
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
    std::vector<float> state(kVHeads * kState * kState);
    for (std::size_t i = 0; i < state.size(); ++i) {
        state[i] = 0.001F * std::sin(0.00031F * static_cast<float>(i));
    }
    return state;
}

std::vector<float> generate_initial_history() {
    std::vector<float> history(4 * kChannels);
    for (std::size_t i = 0; i < history.size(); ++i) {
        history[i] = 0.002F * std::cos(0.00077F * static_cast<float>(i));
    }
    return history;
}

struct BlockEvaluationResult {
    std::uint32_t tokens = 0;
    double v1_token_oracle_ms = 0.0;
    double v1_fast_min_ms = 0.0;
    double v1_fast_max_ms = 0.0;
    double v1_fast_median_ms = 0.0;

    double v2_min_ms = 0.0;
    double v2_max_ms = 0.0;
    double v2_median_ms = 0.0;
    double v2_tok_s = 0.0;

    double speedup_vs_token_oracle = 0.0;
    double speedup_vs_fast_v1 = 0.0;

    NumericalMetrics output_metrics{};
    NumericalMetrics s0_metrics{};
    NumericalMetrics s1_metrics{};
    NumericalMetrics s2_metrics{};
    NumericalMetrics k3_metrics{};
    NumericalMetrics v3_metrics{};

    TopologyBlockProfileBreakdown v2_breakdown{};
};

int run_bakeoff_main(int argc, char** argv) {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer Prefill V2: Slice 2 (4-Layer Topology Block Bakeoff)\n";
    std::cout << "  Target: 1 x AMD Instinct MI50 (gfx906) | Qwen3.8-27B-Q4_K_M\n";
    std::cout << "  Topology Block 0: 3 x GDN (L0, L1, L2) + 1 x GQA Attention (L3)\n";
    std::cout << "===================================================================\n\n";

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

    // Set environment variables to enable TRUE FASTEST_V1 Mx configuration for V1 baseline
    setenv("MIINFER_PREFILL_LAYER_MAJOR", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_CHUNK", "1", 1);
    setenv("MIINFER_PREFILL_CHUNK", "512", 1);
    setenv("MIINFER_PREFILL_WIDE_ATTN", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_REPACKED_MMQ", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_MMQ_FFN", "1", 1);
    setenv("MIINFER_PREFILL_REPACKED_RESIDENT_ALL", "1", 1);
    setenv("MIINFER_PREFILL_REPACKED_RESIDENT_FFN", "1", 1);
    setenv("MIINFER_PREFILL_MX_GDN", "1", 1);

    const auto model = miinfer::Qwen35Model::load(model_path);
    std::cout << "Model loaded: layers=" << model.config().block_count
              << ", hidden=" << model.config().hidden_size
              << ", ffn=" << model.config().intermediate_size << "\n\n";

    PrefillV2WorkspaceManager ws_manager(kMaxPrefillBatch);
    std::cout << "V2 Monolithic Shared Workspace: "
              << (ws_manager.total_workspace_bytes() / (1024.0 * 1024.0)) << " MiB allocated\n";

    // Allocate FAST_V1 helper buffers
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

    auto setup_v1_recurrent_fast = [&](RecurrentLayer& layer) {
        layer.set_m12_dense_workspace(d_dense_weights, d_dense_input, &rocblas_handle);
        layer.set_m12_gdn_workspace(gdn_ws, d_gdn_raw_output);
        layer.ensure_mx_repacked();
    };

    auto setup_v1_attn_fast = [&](FullAttentionLayer& layer) {
        layer.set_m12_dense_workspace(d_dense_weights, d_dense_input, &rocblas_handle);
        layer.ensure_m23_repacked();
    };

    // -------------------------------------------------------------
    // Load V1 Layers (Oracle & Fast V1) for Block 0
    // -------------------------------------------------------------
    std::cout << "\nLoading V1 Layers for Block 0 (L0, L1, L2 Recurrent + L3 Attention)...\n";
    RecurrentLayer v1_oracle_l0(model, 0, {});
    RecurrentLayer v1_oracle_l1(model, 1, {});
    RecurrentLayer v1_oracle_l2(model, 2, {});
    FullAttentionLayer v1_oracle_l3(model, 3);

    RecurrentLayer v1_fast_l0(model, 0, {});
    RecurrentLayer v1_fast_l1(model, 1, {});
    RecurrentLayer v1_fast_l2(model, 2, {});
    FullAttentionLayer v1_fast_l3(model, 3);
    setup_v1_recurrent_fast(v1_fast_l0);
    setup_v1_recurrent_fast(v1_fast_l1);
    setup_v1_recurrent_fast(v1_fast_l2);
    setup_v1_attn_fast(v1_fast_l3);

    // -------------------------------------------------------------
    // Load Prefill V2 Block 0 (L0, L1, L2 Recurrent + L3 Attention)
    // -------------------------------------------------------------
    std::cout << "Loading Prefill V2 Topology Block 0...\n";
    PrefillV2TopologyBlock v2_block0(model, 0);
    std::cout << "  V2 Block 0 Persistent Weights: "
              << (v2_block0.persistent_weight_bytes() / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "    L0 (GDN): " << (v2_block0.gdn0().persistent_weight_bytes() / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "    L1 (GDN): " << (v2_block0.gdn1().persistent_weight_bytes() / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "    L2 (GDN): " << (v2_block0.gdn2().persistent_weight_bytes() / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "    L3 (GQA): " << (v2_block0.gqa3().persistent_weight_bytes() / (1024.0 * 1024.0)) << " MiB\n";

    // Recurrent State Storages and Attention KV Cache
    RecurrentLayerStateStorage v2_s0_storage, v2_s1_storage, v2_s2_storage;
    AttentionLayerKvCacheStorage v2_kv3_storage(kDefaultCacheCapacity);
    std::cout << "  L3 KV Cache Allocated: " << (v2_kv3_storage.total_bytes() / (1024.0 * 1024.0))
              << " MiB (capacity=" << v2_kv3_storage.capacity() << " tokens)\n";

    // -------------------------------------------------------------
    // Helper to evaluate Block 0 across N
    // -------------------------------------------------------------
    const auto evaluate_block = [&](std::uint32_t N) -> BlockEvaluationResult {
        BlockEvaluationResult res;
        res.tokens = N;

        const auto host_in = generate_realistic_input(N, kHidden);
        const auto s_init = generate_initial_state();
        const auto h_init = generate_initial_history();

        // 1. Run V1 Canonical Sequential Token Oracle across 4 layers
        v1_oracle_l0.reset(); v1_oracle_l0.upload_state(s_init);
        upload(h_init.data(), v1_oracle_l0.history->get(), h_init.size() * sizeof(float));

        v1_oracle_l1.reset(); v1_oracle_l1.upload_state(s_init);
        upload(h_init.data(), v1_oracle_l1.history->get(), h_init.size() * sizeof(float));

        v1_oracle_l2.reset(); v1_oracle_l2.upload_state(s_init);
        upload(h_init.data(), v1_oracle_l2.history->get(), h_init.size() * sizeof(float));

        v1_oracle_l3.reset();

        float *d_v1_in = nullptr, *d_v1_mid1 = nullptr, *d_v1_mid2 = nullptr, *d_v1_mid3 = nullptr, *d_v1_out = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_v1_in, kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_v1_mid1, N * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_v1_mid2, N * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_v1_mid3, N * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_v1_out, N * kHidden * sizeof(float)));

        const auto v1_start = std::chrono::steady_clock::now();
        // L0 Token Loop
        for (std::uint32_t t = 0; t < N; ++t) {
            upload(host_in.data() + t * kHidden, d_v1_in, kHidden * sizeof(float));
            v1_oracle_l0.run(d_v1_in, t, d_v1_mid1 + t * kHidden);
        }
        // L1 Token Loop
        for (std::uint32_t t = 0; t < N; ++t) {
            v1_oracle_l1.run(d_v1_mid1 + t * kHidden, t, d_v1_mid2 + t * kHidden);
        }
        // L2 Token Loop
        for (std::uint32_t t = 0; t < N; ++t) {
            v1_oracle_l2.run(d_v1_mid2 + t * kHidden, t, d_v1_mid3 + t * kHidden);
        }
        // L3 Token Loop (Attention)
        for (std::uint32_t t = 0; t < N; ++t) {
            v1_oracle_l3.run(d_v1_mid3 + t * kHidden, t, d_v1_out + t * kHidden);
        }
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        const auto v1_end = std::chrono::steady_clock::now();
        res.v1_token_oracle_ms = std::chrono::duration<double, std::milli>(v1_end - v1_start).count();

        std::vector<float> oracle_output(N * kHidden);
        MIINFER_HIP_CHECK(hipMemcpy(oracle_output.data(), d_v1_out, N * kHidden * sizeof(float), hipMemcpyDeviceToHost));
        const auto oracle_s0 = download(v1_oracle_l0.state->get(), kVHeads * kState * kState);
        const auto oracle_s1 = download(v1_oracle_l1.state->get(), kVHeads * kState * kState);
        const auto oracle_s2 = download(v1_oracle_l2.state->get(), kVHeads * kState * kState);

        (void)hipFree(d_v1_in); (void)hipFree(d_v1_mid1); (void)hipFree(d_v1_mid2); (void)hipFree(d_v1_mid3); (void)hipFree(d_v1_out);

        // 2. Run FAST_V1 (B512 Mx Repacked Path across Block 0) if N == 512
        if (N == 512) {
            float *d_fast_in = nullptr, *d_fast_mid1 = nullptr, *d_fast_mid2 = nullptr, *d_fast_mid3 = nullptr, *d_fast_out = nullptr;
            MIINFER_HIP_CHECK(hipMalloc(&d_fast_in, 512 * kHidden * sizeof(float)));
            MIINFER_HIP_CHECK(hipMalloc(&d_fast_mid1, 512 * kHidden * sizeof(float)));
            MIINFER_HIP_CHECK(hipMalloc(&d_fast_mid2, 512 * kHidden * sizeof(float)));
            MIINFER_HIP_CHECK(hipMalloc(&d_fast_mid3, 512 * kHidden * sizeof(float)));
            MIINFER_HIP_CHECK(hipMalloc(&d_fast_out, 512 * kHidden * sizeof(float)));
            upload(host_in.data(), d_fast_in, 512 * kHidden * sizeof(float));

            hipEvent_t ev_fstart, ev_fstop;
            MIINFER_HIP_CHECK(hipEventCreate(&ev_fstart));
            MIINFER_HIP_CHECK(hipEventCreate(&ev_fstop));

            std::vector<double> fast_samples(10);
            for (int iter = 0; iter < 10; ++iter) {
                v1_fast_l0.upload_state(s_init); upload(h_init.data(), v1_fast_l0.history->get(), h_init.size() * sizeof(float));
                v1_fast_l1.upload_state(s_init); upload(h_init.data(), v1_fast_l1.history->get(), h_init.size() * sizeof(float));
                v1_fast_l2.upload_state(s_init); upload(h_init.data(), v1_fast_l2.history->get(), h_init.size() * sizeof(float));
                v1_fast_l3.reset();

                MIINFER_HIP_CHECK(hipEventRecord(ev_fstart, hipStreamPerThread));
                v1_fast_l0.prefill_wide(d_fast_in, d_fast_mid1, 0, 512);
                v1_fast_l1.prefill_wide(d_fast_mid1, d_fast_mid2, 0, 512);
                v1_fast_l2.prefill_wide(d_fast_mid2, d_fast_mid3, 0, 512);
                v1_fast_l3.prepare_prefill_batch(d_fast_mid3, 512, false);
                v1_fast_l3.finish_prefill_attention(0, 512);
                v1_fast_l3.finish_prefill_wide(d_fast_mid3, d_fast_out, 512, nullptr, nullptr);
                MIINFER_HIP_CHECK(hipEventRecord(ev_fstop, hipStreamPerThread));
                MIINFER_HIP_CHECK(hipEventSynchronize(ev_fstop));

                float ms = 0.0F;
                MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_fstart, ev_fstop));
                fast_samples[iter] = static_cast<double>(ms);
            }
            (void)hipEventDestroy(ev_fstart);
            (void)hipEventDestroy(ev_fstop);

            std::sort(fast_samples.begin(), fast_samples.end());
            res.v1_fast_min_ms = fast_samples.front();
            res.v1_fast_max_ms = fast_samples.back();
            res.v1_fast_median_ms = fast_samples[5];

            (void)hipFree(d_fast_in); (void)hipFree(d_fast_mid1); (void)hipFree(d_fast_mid2); (void)hipFree(d_fast_mid3); (void)hipFree(d_fast_out);
        }

        // 3. Run Prefill V2 Block 0 with Ping-Pong Activations & Monolithic Workspace
        float *d_ping = nullptr, *d_pong = nullptr, *d_v2_final_out = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(&d_ping, N * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_pong, N * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(&d_v2_final_out, N * kHidden * sizeof(float)));
        upload(host_in.data(), d_ping, N * kHidden * sizeof(float));

        v2_s0_storage.upload(s_init, h_init, 0); auto view0 = v2_s0_storage.view();
        v2_s1_storage.upload(s_init, h_init, 0); auto view1 = v2_s1_storage.view();
        v2_s2_storage.upload(s_init, h_init, 0); auto view2 = v2_s2_storage.view();
        v2_kv3_storage.reset(hipStreamPerThread); auto view3 = v2_kv3_storage.view();

        auto ws = ws_manager.workspace();

        // Warmup & Verification
        v2_block0.forward(d_ping, d_pong, d_v2_final_out, view0, view0, view1, view1, view2, view2, view3, ws, 0, N, hipStreamPerThread);
        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

        std::vector<float> v2_out(N * kHidden);
        MIINFER_HIP_CHECK(hipMemcpy(v2_out.data(), d_v2_final_out, N * kHidden * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<float> v2_s0, v2_h0, v2_s1, v2_h1, v2_s2, v2_h2;
        v2_s0_storage.download(v2_s0, v2_h0);
        v2_s1_storage.download(v2_s1, v2_h1);
        v2_s2_storage.download(v2_s2, v2_h2);

        res.output_metrics = compute_metrics(v2_out, oracle_output);
        res.s0_metrics = compute_metrics(v2_s0, oracle_s0);
        res.s1_metrics = compute_metrics(v2_s1, oracle_s1);
        res.s2_metrics = compute_metrics(v2_s2, oracle_s2);

        // Performance Benchmark (HIP Events)
        hipEvent_t ev_start, ev_stop;
        MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
        MIINFER_HIP_CHECK(hipEventCreate(&ev_stop));

        std::vector<double> v2_samples(10);
        for (int iter = 0; iter < 10; ++iter) {
            upload(host_in.data(), d_ping, N * kHidden * sizeof(float));
            v2_s0_storage.upload(s_init, h_init, 0);
            v2_s1_storage.upload(s_init, h_init, 0);
            v2_s2_storage.upload(s_init, h_init, 0);
            v2_kv3_storage.reset(hipStreamPerThread);

            MIINFER_HIP_CHECK(hipEventRecord(ev_start, hipStreamPerThread));
            v2_block0.forward(d_ping, d_pong, d_v2_final_out, view0, view0, view1, view1, view2, view2, view3, ws, 0, N, hipStreamPerThread);
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
        res.v2_median_ms = v2_samples[5];
        res.v2_tok_s = (static_cast<double>(N) / res.v2_median_ms) * 1000.0;
        res.speedup_vs_token_oracle = res.v1_token_oracle_ms / res.v2_median_ms;
        res.speedup_vs_fast_v1 = res.v1_fast_median_ms > 0.0 ? (res.v1_fast_median_ms / res.v2_median_ms) : 0.0;

        // Profiled run
        upload(host_in.data(), d_ping, N * kHidden * sizeof(float));
        v2_s0_storage.upload(s_init, h_init, 0);
        v2_s1_storage.upload(s_init, h_init, 0);
        v2_s2_storage.upload(s_init, h_init, 0);
        v2_kv3_storage.reset(hipStreamPerThread);

        v2_block0.forward_profiled(d_ping, d_pong, d_v2_final_out, view0, view0, view1, view1, view2, view2, view3, ws, 0, N, res.v2_breakdown, hipStreamPerThread);

        (void)hipFree(d_ping); (void)hipFree(d_pong); (void)hipFree(d_v2_final_out);

        return res;
    };

    const auto print_block_result = [](const BlockEvaluationResult& r) {
        std::cout << "\n=======================================================\n";
        std::cout << "  Evaluation: Topology Block 0 (4 Layers) | Tokens: " << r.tokens << "\n";
        std::cout << "=======================================================\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Numerical Metrics vs Canonical Oracle:\n";
        std::cout << "    Block Output: Cosine = " << r.output_metrics.cosine_similarity
                  << " | MaxErr = " << r.output_metrics.max_abs_err
                  << " | RelRMS = " << r.output_metrics.rel_rms
                  << " | Finite = " << (r.output_metrics.all_finite ? "YES" : "NO") << "\n";
        std::cout << "    L0 State:     Cosine = " << r.s0_metrics.cosine_similarity << " | RelRMS = " << r.s0_metrics.rel_rms << "\n";
        std::cout << "    L1 State:     Cosine = " << r.s1_metrics.cosine_similarity << " | RelRMS = " << r.s1_metrics.rel_rms << "\n";
        std::cout << "    L2 State:     Cosine = " << r.s2_metrics.cosine_similarity << " | RelRMS = " << r.s2_metrics.rel_rms << "\n";

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n  Performance Comparison (N=" << r.tokens << "):\n";
        std::cout << "    V1 Token Oracle:    " << std::setw(8) << r.v1_token_oracle_ms << " ms\n";
        if (r.v1_fast_median_ms > 0.0) {
            std::cout << "    FAST_V1 (B512 Mx):  " << std::setw(8) << r.v1_fast_median_ms
                      << " ms (min=" << r.v1_fast_min_ms << ", max=" << r.v1_fast_max_ms << ")\n";
        } else {
            std::cout << "    FAST_V1 (B512 Mx):       N/A (V1 unsupported for N < 512)\n";
        }
        std::cout << "    Prefill V2 (Mx):    " << std::setw(8) << r.v2_median_ms
                  << " ms (min=" << r.v2_min_ms << ", max=" << r.v2_max_ms << ")\n";
        std::cout << "    Speedup vs Oracle:  " << std::setw(8) << r.speedup_vs_token_oracle << "x\n";
        if (r.v1_fast_median_ms > 0.0) {
            std::cout << "    Speedup vs FAST_V1: " << std::setw(8) << r.speedup_vs_fast_v1 << "x\n";
        }
        std::cout << "    Throughput:         " << std::setw(8) << r.v2_tok_s << " tok/s ("
                  << (r.v2_median_ms / static_cast<double>(r.tokens)) << " ms/tok)\n";

        if (r.tokens == 512) {
            std::cout << "\n  V2 Block 0 Layer Breakdown (N=512, Total=" << r.v2_breakdown.total_block_ms << " ms):\n";
            const double tot = r.v2_breakdown.total_block_ms;
            const auto print_l = [&](const char* name, double ms) {
                std::cout << "    " << std::left << std::setw(28) << name << ": "
                          << std::right << std::setw(6) << ms << " ms ("
                          << std::setw(5) << (ms / tot * 100.0) << "%)\n";
            };
            print_l("Layer 0 (GDN Recurrent)", r.v2_breakdown.gdn0_ms);
            print_l("Layer 1 (GDN Recurrent)", r.v2_breakdown.gdn1_ms);
            print_l("Layer 2 (GDN Recurrent)", r.v2_breakdown.gdn2_ms);
            print_l("Layer 3 (GQA Attention)", r.v2_breakdown.gqa3_ms);
        }
    };

    // -------------------------------------------------------------
    // Part A & B: Evaluate Block 0 across N = 64, 128, 512
    // -------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Part A: Multi-Length Topology Block 0 Evaluation\n";
    std::cout << "-------------------------------------------------------------------\n";

    const auto res_64 = evaluate_block(64);
    print_block_result(res_64);

    const auto res_128 = evaluate_block(128);
    print_block_result(res_128);

    const auto res_512 = evaluate_block(512);
    print_block_result(res_512);

    // -------------------------------------------------------------
    // Part C: Mandatory N=512 Summary Table
    // -------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Mandatory N=512 Topology Block Summary Table\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "| Unit / Topology | Oracle (ms) | FAST_V1 (ms) | Prefill V2 (ms) | V2 / FAST_V1 | Speedup vs Oracle |\n";
    std::cout << "|:---|---:|---:|---:|---:|---:|\n";
    std::cout << "| Block 0 (3xGDN + 1xGQA) | " << std::setw(10) << res_512.v1_token_oracle_ms
              << " | " << std::setw(11) << res_512.v1_fast_median_ms
              << " | " << std::setw(14) << res_512.v2_median_ms
              << " | " << std::setw(11) << (res_512.v1_fast_median_ms / res_512.v2_median_ms) << "x"
              << " | " << std::setw(16) << res_512.speedup_vs_token_oracle << "x |\n";

    // -------------------------------------------------------------
    // Part D: Stateful Split-Call Equivalence Test (512 vs 256 + 256)
    // -------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Part D: Stateful Split-Call Invariant Test (512 vs 256 + 256)\n";
    std::cout << "-------------------------------------------------------------------\n";

    const auto host_in512 = generate_realistic_input(512, kHidden);
    const auto s_init = generate_initial_state();
    const auto h_init = generate_initial_history();

    // 1. One-Shot 512
    float *d_in512 = nullptr, *d_pong512 = nullptr, *d_out512_oneshot = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&d_in512, 512 * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_pong512, 512 * kHidden * sizeof(float)));
    MIINFER_HIP_CHECK(hipMalloc(&d_out512_oneshot, 512 * kHidden * sizeof(float)));
    upload(host_in512.data(), d_in512, 512 * kHidden * sizeof(float));

    v2_s0_storage.upload(s_init, h_init, 0); auto v0 = v2_s0_storage.view();
    v2_s1_storage.upload(s_init, h_init, 0); auto v1 = v2_s1_storage.view();
    v2_s2_storage.upload(s_init, h_init, 0); auto v2 = v2_s2_storage.view();
    v2_kv3_storage.reset(hipStreamPerThread); auto v3 = v2_kv3_storage.view();
    auto ws = ws_manager.workspace();

    v2_block0.forward(d_in512, d_pong512, d_out512_oneshot, v0, v0, v1, v1, v2, v2, v3, ws, 0, 512, hipStreamPerThread);
    MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    std::vector<float> oneshot_out(512 * kHidden);
    MIINFER_HIP_CHECK(hipMemcpy(oneshot_out.data(), d_out512_oneshot, 512 * kHidden * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<float> oneshot_s0, oneshot_h0, oneshot_s1, oneshot_h1, oneshot_s2, oneshot_h2;
    v2_s0_storage.download(oneshot_s0, oneshot_h0);
    v2_s1_storage.download(oneshot_s1, oneshot_h1);
    v2_s2_storage.download(oneshot_s2, oneshot_h2);
    std::vector<float> oneshot_k3, oneshot_v3;
    v2_kv3_storage.download_key(oneshot_k3, 512);
    v2_kv3_storage.download_value(oneshot_v3, 512);

    // 2. Split-call 256 + 256
    float *d_out512_split = nullptr;
    MIINFER_HIP_CHECK(hipMalloc(&d_out512_split, 512 * kHidden * sizeof(float)));
    upload(host_in512.data(), d_in512, 512 * kHidden * sizeof(float));

    v2_s0_storage.upload(s_init, h_init, 0); v0 = v2_s0_storage.view();
    v2_s1_storage.upload(s_init, h_init, 0); v1 = v2_s1_storage.view();
    v2_s2_storage.upload(s_init, h_init, 0); v2 = v2_s2_storage.view();
    v2_kv3_storage.reset(hipStreamPerThread); v3 = v2_kv3_storage.view();

    // First 256 chunk
    v2_block0.forward(d_in512, d_pong512, d_out512_split, v0, v0, v1, v1, v2, v2, v3, ws, 0, 256, hipStreamPerThread);
    // Second 256 chunk (base_position = 256)
    v2_block0.forward(d_in512 + 256 * kHidden, d_pong512, d_out512_split + 256 * kHidden, v0, v0, v1, v1, v2, v2, v3, ws, 256, 256, hipStreamPerThread);
    MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    std::vector<float> split_out(512 * kHidden);
    MIINFER_HIP_CHECK(hipMemcpy(split_out.data(), d_out512_split, 512 * kHidden * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<float> split_s0, split_h0, split_s1, split_h1, split_s2, split_h2;
    v2_s0_storage.download(split_s0, split_h0);
    v2_s1_storage.download(split_s1, split_h1);
    v2_s2_storage.download(split_s2, split_h2);
    std::vector<float> split_k3, split_v3;
    v2_kv3_storage.download_key(split_k3, 512);
    v2_kv3_storage.download_value(split_v3, 512);

    (void)hipFree(d_in512); (void)hipFree(d_pong512); (void)hipFree(d_out512_oneshot); (void)hipFree(d_out512_split);

    const auto split_out_m = compute_metrics(split_out, oneshot_out);
    const auto split_s0_m = compute_metrics(split_s0, oneshot_s0);
    const auto split_s1_m = compute_metrics(split_s1, oneshot_s1);
    const auto split_s2_m = compute_metrics(split_s2, oneshot_s2);
    const auto split_k3_m = compute_metrics(split_k3, oneshot_k3);
    const auto split_v3_m = compute_metrics(split_v3, oneshot_v3);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Split-call Equivalence: One-Shot 512 vs Split (256 + 256):\n";
    std::cout << "    Output Cosine:   " << split_out_m.cosine_similarity
              << " | MaxErr: " << split_out_m.max_abs_err << " | RelRMS: " << split_out_m.rel_rms << "\n";
    std::cout << "    L0 State Cosine: " << split_s0_m.cosine_similarity << " | MaxErr: " << split_s0_m.max_abs_err << "\n";
    std::cout << "    L1 State Cosine: " << split_s1_m.cosine_similarity << " | MaxErr: " << split_s1_m.max_abs_err << "\n";
    std::cout << "    L2 State Cosine: " << split_s2_m.cosine_similarity << " | MaxErr: " << split_s2_m.max_abs_err << "\n";
    std::cout << "    L3 Key Cache:    " << split_k3_m.cosine_similarity << " | MaxErr: " << split_k3_m.max_abs_err << "\n";
    std::cout << "    L3 Value Cache:  " << split_v3_m.cosine_similarity << " | MaxErr: " << split_v3_m.max_abs_err << "\n";

    // -------------------------------------------------------------
    // Part E: Full-Model Latency & VRAM Budget Projections
    // -------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Part E: Full 64-Layer Model Budget & Latency Projection\n";
    std::cout << "-------------------------------------------------------------------\n";

    const double bytes_block0 = static_cast<double>(v2_block0.persistent_weight_bytes());
    const double total_64layers_weights_gb = (16.0 * bytes_block0) / (1024.0 * 1024.0 * 1024.0);
    const double total_recurrent_states_mb = 48.0 * (RecurrentLayerState::kStateBytes + RecurrentLayerState::kConvHistoryBytes) / (1024.0 * 1024.0);
    const double total_kv_cache_gb = (16.0 * v2_kv3_storage.total_bytes()) / (1024.0 * 1024.0 * 1024.0);
    const double shared_ws_mb = ws_manager.total_workspace_bytes() / (1024.0 * 1024.0);
    const double total_vram_gb = total_64layers_weights_gb + total_kv_cache_gb + (total_recurrent_states_mb + shared_ws_mb) / 1024.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Persistent Weights per 4-Layer Block: " << (bytes_block0 / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "  Projected 64 Layers Weights (16 Blocks): " << total_64layers_weights_gb << " GiB\n";
    std::cout << "  Projected 48 Recurrent Layer States:    " << total_recurrent_states_mb << " MiB\n";
    std::cout << "  Projected 16 Attention KV Caches:       " << total_kv_cache_gb << " GiB (32K capacity)\n";
    std::cout << "  Monolithic Shared Workspace:            " << shared_ws_mb << " MiB\n";
    std::cout << "  Total Static VRAM Footprint:            " << total_vram_gb << " GiB / 32 GiB\n";

    const double full_model_p512_ms = 16.0 * res_512.v2_median_ms;
    std::cout << "\n  Projected Full 64-Layer Model P512 Latency:\n";
    std::cout << "    16 x Block 0 (" << res_512.v2_median_ms << " ms) = "
              << full_model_p512_ms << " ms (" << (full_model_p512_ms / 1000.0) << " seconds)\n";
    std::cout << "    Target mx-llama.cpp P512: ~2,310 ms (~2.31 s)\n";
    if (full_model_p512_ms < 2310.0) {
        std::cout << "    Status: BEATS mx-llama.cpp! (Speedup: " << (2310.0 / full_model_p512_ms) << "x)\n";
    } else {
        std::cout << "    Status: " << (full_model_p512_ms / 2310.0) << "x of mx-llama.cpp.\n";
    }

    std::cout << "\n===================================================================\n";
    std::cout << "  Slice 2 Topology Block Bakeoff Complete.\n";
    std::cout << "===================================================================\n";

    // Free FAST_V1 buffers
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
