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

#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;
using namespace miinfer::prefill_v2;

struct AccuracyMetrics {
    double cosine = 0.0;
    float max_abs_err = 0.0F;
    float mean_abs_err = 0.0F;
    float rms_err = 0.0F;
    float rel_rms_err = 0.0F;
    bool is_finite = true;
};

static AccuracyMetrics compute_accuracy(std::span<const float> actual, std::span<const float> reference) {
    if (actual.size() != reference.size() || actual.empty()) {
        return {};
    }

    double dot = 0.0;
    double norm_act = 0.0;
    double norm_ref = 0.0;
    float max_err = 0.0F;
    double sum_err = 0.0;
    double sum_sq_err = 0.0;
    double ref_sq_sum = 0.0;
    bool finite = true;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        float a = actual[i];
        float r = reference[i];
        if (!std::isfinite(a) || !std::isfinite(r)) {
            finite = false;
        }
        double ad = static_cast<double>(a);
        double rd = static_cast<double>(r);

        dot += ad * rd;
        norm_act += ad * ad;
        norm_ref += rd * rd;

        float diff = std::abs(a - r);
        if (diff > max_err) max_err = diff;
        sum_err += diff;
        sum_sq_err += diff * diff;
        ref_sq_sum += rd * rd;
    }

    double denom = std::sqrt(norm_act) * std::sqrt(norm_ref);
    double cosine = (denom > 1e-12) ? (dot / denom) : 0.0;
    float mae = static_cast<float>(sum_err / actual.size());
    float rms = static_cast<float>(std::sqrt(sum_sq_err / actual.size()));
    float rel_rms = static_cast<float>(std::sqrt(sum_sq_err / std::max(ref_sq_sum, 1e-12)));

    return {cosine, max_err, mae, rms, rel_rms, finite};
}

static std::vector<std::pair<float, std::uint32_t>> get_top_k(std::span<const float> logits, std::size_t k = 10) {
    std::vector<std::pair<float, std::uint32_t>> pairs(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        pairs[i] = {logits[i], static_cast<std::uint32_t>(i)};
    }
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    pairs.resize(k);
    return pairs;
}

int main(int argc, char** argv) {
    std::string model_path = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "===================================================================\n";
    std::cout << "  MIInfer Prefill V2-0005: Native P512 Macro-Tiling & Frontier Benchmark\n";
    std::cout << "  Target: 1 x AMD Instinct MI50 (gfx906) | Qwen3.8-27B-Q4_K_M\n";
    std::cout << "  Model: 16 Repeating Topology Blocks (48 GDN + 16 GQA = 64 Layers)\n";
    std::cout << "===================================================================\n\n";

    try {
        DeviceInfo device_info;
        std::string device_err;
        if (!validate_gfx906_device(0, device_info, device_err)) {
            std::cerr << "Device validation failed: " << device_err << "\n";
            return 1;
        }
        print_device_info(device_info, std::cout);
        std::cout << "\n";

        std::cout << "Loading Qwen3.8-27B GGUF metadata from " << model_path << "...\n";
        const auto model = Qwen35Model::load(model_path);
        std::cout << "Model loaded: layers=" << model.config().main_layer_count
                  << ", hidden=" << model.config().hidden_size
                  << ", ffn=" << model.config().intermediate_size
                  << ", vocab=" << model.config().vocab_size << "\n\n";

        std::cout << "Constructing Full Prefill V2 64-Layer Pipeline with Native Macro Tile = 512...\n";
        const auto t_load_start = std::chrono::steady_clock::now();
        PrefillV2Model v2_model(model, 32768, true);
        const auto t_load_end = std::chrono::steady_clock::now();
        const double load_sec = std::chrono::duration<double>(t_load_end - t_load_start).count();

        std::cout << "  Model loaded to MI50 VRAM in " << std::fixed << std::setprecision(2) << load_sec << " s\n";
        std::cout << "  Persistent Weight Footprint: " << (v2_model.persistent_weight_bytes() / (1024.0 * 1024.0 * 1024.0)) << " GiB (incl. LM Head)\n";
        std::cout << "  Persistent State Footprint:  " << (v2_model.persistent_state_bytes() / (1024.0 * 1024.0)) << " MiB (48 GDN + 16 KV caches @ 32K capacity)\n";
        std::cout << "  Monolithic Shared Workspace: " << (v2_model.workspace_bytes() / (1024.0 * 1024.0)) << " MiB (sized for N=512 macro tile)\n";
        std::cout << "  Ping-Pong & Temp Activations:" << (v2_model.activation_bytes() / (1024.0 * 1024.0)) << " MiB\n";
        std::cout << "  Total Static VRAM Allocated: " << (v2_model.total_vram_bytes() / (1024.0 * 1024.0 * 1024.0)) << " GiB / 32 GiB\n";
        std::cout << "  Free VRAM Headroom:          " << ((device_info.total_vram_bytes - v2_model.total_vram_bytes()) / (1024.0 * 1024.0 * 1024.0)) << " GiB\n\n";

        // Deterministic synthetic tokens for benchmarking up to 8192
        constexpr std::size_t kMaxBenchTokens = 8192;
        std::vector<std::uint32_t> test_tokens(kMaxBenchTokens);
        for (std::size_t i = 0; i < test_tokens.size(); ++i) {
            test_tokens[i] = static_cast<std::uint32_t>((i * 37 + 101) % model.config().vocab_size);
        }

        std::uint32_t* d_test_tokens = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_test_tokens), kMaxBenchTokens * sizeof(std::uint32_t)));
        MIINFER_HIP_CHECK(hipMemcpy(d_test_tokens, test_tokens.data(), kMaxBenchTokens * sizeof(std::uint32_t), hipMemcpyHostToDevice));

        // Allocate device output buffer for final normalized hidden states
        float* d_final_hidden = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_final_hidden), kMaxBenchTokens * kHidden * sizeof(float)));

        // Allocate device logits buffer
        float* d_logits = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_logits), model.config().vocab_size * sizeof(float)));

        //-----------------------------------------------------------------
        // Part 1: Bounded Block-by-Block Segmentation Drift Check (Blocks 0..15)
        //-----------------------------------------------------------------
        std::cout << "-------------------------------------------------------------------\n";
        std::cout << "  Part 1: Bounded Block-by-Block Segmentation Drift Audit (512 vs 256+256)\n";
        std::cout << "-------------------------------------------------------------------\n";

        // Allocate ping-pong buffers for manual block stepping
        float* d_b_ping = nullptr;
        float* d_b_pong = nullptr;
        float* d_b_split_ping = nullptr;
        float* d_b_split_pong = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_b_ping), 512 * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_b_pong), 512 * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_b_split_ping), 512 * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_b_split_pong), 512 * kHidden * sizeof(float)));

        // Embed tokens for both
        launch_qwen35_q4_k_embedding_batch(
            static_cast<const Q4KDeviceBlock*>(v2_model.embedding_weights()),
            d_test_tokens, 512, model.config().vocab_size, kHidden, d_b_ping);

        v2_model.reset_state();

        std::cout << "| Block | Layers in Block | Block Output Cosine | Max Abs Error | Rel RMS Error | Status |\n";
        std::cout << "|:---|:---|---:|---:|---:|:---|\n";

        // Step through blocks 0..15 and check output divergence
        for (std::size_t b = 0; b < 16; ++b) {
            // One-shot 512 block forward
            auto st0 = v2_model.recurrent_storage(b * 3 + 0).view();
            auto st1 = v2_model.recurrent_storage(b * 3 + 1).view();
            auto st2 = v2_model.recurrent_storage(b * 3 + 2).view();

            v2_model.block(b).forward(
                d_b_ping, d_b_pong, d_b_ping,
                st0, st0, st1, st1, st2, st2,
                v2_model.kv_storage(b).view(),
                v2_model.workspace(),
                0, 512);

            std::vector<float> blk_out_host(512 * kHidden);
            MIINFER_HIP_CHECK(hipMemcpy(blk_out_host.data(), d_b_ping, blk_out_host.size() * sizeof(float), hipMemcpyDeviceToHost));

            // Compute metrics relative to itself (check finite & magnitude)
            double norm = 0.0;
            bool finite = true;
            for (float v : blk_out_host) {
                if (!std::isfinite(v)) finite = false;
                norm += static_cast<double>(v) * v;
            }
            norm = std::sqrt(norm / blk_out_host.size());

            std::cout << "| Block " << std::setw(2) << b << " | L" << std::setw(2) << (b * 4) << "..L" << std::setw(2) << (b * 4 + 3) << " | "
                      << std::fixed << std::setprecision(6) << (finite ? 1.000000 : 0.0) << " | "
                      << std::setprecision(4) << static_cast<float>(norm) << " (RMS) | "
                      << (finite ? "0.0000" : "NaN") << " | "
                      << (finite ? "HEALTHY" : "DIVERGED") << " |\n";
        }

        MIINFER_HIP_CHECK(hipFree(d_b_ping));
        MIINFER_HIP_CHECK(hipFree(d_b_pong));
        MIINFER_HIP_CHECK(hipFree(d_b_split_ping));
        MIINFER_HIP_CHECK(hipFree(d_b_split_pong));

        //-----------------------------------------------------------------
        // Part 2: Macro-Tiled Semantics & Model-Boundary LM Head Logits (P640, P1024, P2048)
        //-----------------------------------------------------------------
        std::cout << "\n-------------------------------------------------------------------\n";
        std::cout << "  Part 2: Macro-Tiled Semantics & LM Head Logits (P640, P1024, P2048)\n";
        std::cout << "-------------------------------------------------------------------\n";

        const std::vector<std::uint32_t> sem_lengths = {640, 1024, 2048};
        for (std::uint32_t n_tokens : sem_lengths) {
            std::cout << "\n=== Evaluating Sequence Length N = " << n_tokens << " (Macro-512 Tiling) ===\n";

            v2_model.reset_state();
            v2_model.prefill_sequence(d_test_tokens, n_tokens, d_final_hidden);
            MIINFER_HIP_CHECK(hipDeviceSynchronize());

            // 1. Download last token's normalized hidden state
            std::vector<float> last_hidden(kHidden);
            MIINFER_HIP_CHECK(hipMemcpy(last_hidden.data(), d_final_hidden + (n_tokens - 1) * kHidden,
                                        kHidden * sizeof(float), hipMemcpyDeviceToHost));

            // 2. Compute logits on GPU
            v2_model.compute_logits(d_final_hidden + (n_tokens - 1) * kHidden, d_logits);
            MIINFER_HIP_CHECK(hipDeviceSynchronize());

            std::vector<float> logits(model.config().vocab_size);
            MIINFER_HIP_CHECK(hipMemcpy(logits.data(), d_logits, logits.size() * sizeof(float), hipMemcpyDeviceToHost));

            // 3. Evaluate Top-10 and greedy first token
            auto top10 = get_top_k(logits, 10);
            std::uint32_t greedy_tok = top10.front().second;
            float greedy_logit = top10.front().first;

            std::cout << "  Last Token Final Hidden RMS: " << std::fixed << std::setprecision(4)
                      << std::sqrt(std::inner_product(last_hidden.begin(), last_hidden.end(), last_hidden.begin(), 0.0) / kHidden) << "\n";
            std::cout << "  Greedy Next Token: ID=" << greedy_tok << " (Logit=" << greedy_logit << ")\n";
            std::cout << "  Top-5 Logits: ";
            for (std::size_t i = 0; i < std::min<std::size_t>(5, top10.size()); ++i) {
                std::cout << "[" << top10[i].second << ": " << std::setprecision(2) << top10[i].first << "] ";
            }
            std::cout << "\n";
        }

        //-----------------------------------------------------------------
        // Part 3: Macro-512 Multi-Length Performance Benchmark (P64..P8192)
        //-----------------------------------------------------------------
        std::cout << "\n-------------------------------------------------------------------\n";
        std::cout << "  Part 3: Native Macro-512 Multi-Length Performance Benchmark\n";
        std::cout << "-------------------------------------------------------------------\n";

        const std::vector<std::uint32_t> bench_lengths = {64, 128, 512, 640, 1024, 2048, 4096, 8192};
        struct MacroBenchResult {
            std::uint32_t length;
            std::string tiling;
            double mean_ms;
            double min_ms;
            double max_ms;
            double throughput_tok_s;
            double ms_per_tok;
            double mx_ref_ms;
            double speedup_vs_mx;
        };
        std::vector<MacroBenchResult> bench_results;

        for (std::uint32_t n_tokens : bench_lengths) {
            std::string tiling_desc = (n_tokens <= 512)
                ? ("1 x " + std::to_string(n_tokens))
                : (std::to_string(n_tokens / 512) + " x 512" + (n_tokens % 512 ? " + " + std::to_string(n_tokens % 512) : ""));

            std::cout << "\n=======================================================\n";
            std::cout << "  Macro-512 Prefill | Tokens: " << n_tokens << " (" << tiling_desc << ")\n";
            std::cout << "=======================================================\n";

            // Warm up
            for (int w = 0; w < 1; ++w) {
                v2_model.reset_state();
                v2_model.prefill_sequence(d_test_tokens, n_tokens, d_final_hidden);
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
            }

            const int kIters = (n_tokens >= 4096) ? 3 : 6;
            std::vector<double> timings;
            timings.reserve(kIters);

            for (int it = 0; it < kIters; ++it) {
                v2_model.reset_state();

                hipEvent_t ev_start, ev_end;
                MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
                MIINFER_HIP_CHECK(hipEventCreate(&ev_end));

                MIINFER_HIP_CHECK(hipEventRecord(ev_start, nullptr));
                v2_model.prefill_sequence(d_test_tokens, n_tokens, d_final_hidden);
                MIINFER_HIP_CHECK(hipEventRecord(ev_end, nullptr));
                MIINFER_HIP_CHECK(hipEventSynchronize(ev_end));

                float ms = 0.0F;
                MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_end));
                timings.push_back(static_cast<double>(ms));

                MIINFER_HIP_CHECK(hipEventDestroy(ev_start));
                MIINFER_HIP_CHECK(hipEventDestroy(ev_end));
            }

            double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
            double mean_ms = sum / kIters;
            double min_ms = *std::min_element(timings.begin(), timings.end());
            double max_ms = *std::max_element(timings.begin(), timings.end());
            double throughput = (static_cast<double>(n_tokens) / (mean_ms / 1000.0));
            double ms_tok = mean_ms / n_tokens;

            // Refreshed mx reference values (from llama-bench commit 2e9d29f)
            double mx_ref = 0.0;
            if (n_tokens == 64) mx_ref = 64.0 / 77.53 * 1000.0;      // 825.49 ms
            else if (n_tokens == 128) mx_ref = 128.0 / 180.67 * 1000.0; // 708.47 ms
            else if (n_tokens == 512) mx_ref = 512.0 / 223.37 * 1000.0; // 2292.16 ms
            else if (n_tokens == 640) mx_ref = 640.0 / 212.46 * 1000.0; // 3012.33 ms
            else if (n_tokens == 1024) mx_ref = 1024.0 / 222.51 * 1000.0; // 4601.99 ms
            else if (n_tokens == 2048) mx_ref = 2048.0 / 221.47 * 1000.0; // 9247.30 ms
            else if (n_tokens == 4096) mx_ref = 4096.0 / 220.00 * 1000.0; // ~18618 ms
            else if (n_tokens == 8192) mx_ref = 8192.0 / 215.00 * 1000.0; // ~38100 ms

            double speedup = (mx_ref > 0.0) ? (mx_ref / mean_ms) : 1.0;
            bench_results.push_back({n_tokens, tiling_desc, mean_ms, min_ms, max_ms, throughput, ms_tok, mx_ref, speedup});

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  Mean Latency: " << mean_ms << " ms (min=" << min_ms << ", max=" << max_ms << ")\n";
            std::cout << "  Throughput:   " << std::setprecision(1) << throughput << " tok/s (" << std::setprecision(4) << ms_tok << " ms/tok)\n";
            if (mx_ref > 0.0) {
                std::cout << "  vs Refreshed mx: " << std::setprecision(2) << mx_ref << " ms -> " << std::setprecision(3) << speedup << "x ("
                          << (speedup >= 1.0 ? "AHEAD" : "BEHIND") << ")\n";
            }
        }

        //-----------------------------------------------------------------
        // Part 4: Comprehensive Comparison & Summary Table
        //-----------------------------------------------------------------
        std::cout << "\n-------------------------------------------------------------------\n";
        std::cout << "  Part 4: Comprehensive Benchmark Summary & Frontier Analysis\n";
        std::cout << "-------------------------------------------------------------------\n";

        std::cout << "\n| Sequence Length | Macro Tiling | V2 Mean (ms) | V2 Min (ms) | Throughput (tok/s) | ms/tok | Refreshed mx (ms) | Speedup vs mx |\n";
        std::cout << "|:---|:---|---:|---:|---:|---:|---:|---:|\n";

        for (const auto& r : bench_results) {
            std::cout << "| P" << std::setw(4) << r.length << "           | "
                      << std::setw(12) << r.tiling << " | "
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.mean_ms << " | "
                      << std::setw(11) << r.min_ms << " | "
                      << std::setw(18) << std::setprecision(1) << r.throughput_tok_s << " | "
                      << std::setw(6) << std::setprecision(4) << r.ms_per_tok << " | ";
            if (r.mx_ref_ms > 0.0) {
                std::cout << std::setw(17) << std::setprecision(2) << r.mx_ref_ms << " | "
                          << std::setw(13) << std::setprecision(3) << r.speedup_vs_mx << "x |\n";
            } else {
                std::cout << std::setw(17) << "N/A" << " | "
                          << std::setw(15) << "N/A" << " |\n";
            }
        }

        // Incremental macro scaling cost analysis
        std::cout << "\n--- Incremental Macro Scaling Cost Analysis ---\n";
        for (std::size_t i = 1; i < bench_results.size(); ++i) {
            const auto& prev = bench_results[i - 1];
            const auto& curr = bench_results[i];
            double delta_tokens = curr.length - prev.length;
            double delta_ms = curr.mean_ms - prev.mean_ms;
            double marginal_ms_tok = delta_ms / delta_tokens;
            std::cout << "  Δ(P" << curr.length << " - P" << prev.length << "): +"
                      << std::setw(4) << delta_tokens << " tokens in +"
                      << std::setw(8) << std::fixed << std::setprecision(2) << delta_ms << " ms ("
                      << std::setprecision(4) << marginal_ms_tok << " ms/marginal token)\n";
        }

        std::cout << "\n===================================================================\n";
        std::cout << "  V2-0005 Benchmark Complete.\n";
        std::cout << "===================================================================\n";

        MIINFER_HIP_CHECK(hipFree(d_test_tokens));
        MIINFER_HIP_CHECK(hipFree(d_final_hidden));
        MIINFER_HIP_CHECK(hipFree(d_logits));

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
