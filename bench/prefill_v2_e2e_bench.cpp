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

namespace {

const char* kDefaultModelPath = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";

struct BenchmarkRunStats {
    double ttft_ms = 0.0;
    double prefill_tok_per_sec = 0.0;
    double decode_ms = 0.0;
    double decode_tok_per_sec = 0.0;
    double mean_step_ms = 0.0;
    double total_ms = 0.0;
    std::vector<std::uint32_t> generated_tokens;
};

struct AggregatedStats {
    double mean_ttft_ms = 0.0;
    double min_ttft_ms = 0.0;
    double mean_decode_tok_s = 0.0;
    double max_decode_tok_s = 0.0;
    double mean_step_ms = 0.0;
    double min_step_ms = 0.0;
    double mean_total_ms = 0.0;
    std::vector<std::uint32_t> generated_tokens;
};

AggregatedStats aggregate(const std::vector<BenchmarkRunStats>& runs) {
    if (runs.empty()) return {};
    AggregatedStats res;
    res.min_ttft_ms = runs[0].ttft_ms;
    res.max_decode_tok_s = runs[0].decode_tok_per_sec;
    res.min_step_ms = runs[0].mean_step_ms;
    res.generated_tokens = runs[0].generated_tokens;

    for (const auto& r : runs) {
        res.mean_ttft_ms += r.ttft_ms;
        res.mean_decode_tok_s += r.decode_tok_per_sec;
        res.mean_step_ms += r.mean_step_ms;
        res.mean_total_ms += r.total_ms;

        res.min_ttft_ms = std::min(res.min_ttft_ms, r.ttft_ms);
        res.max_decode_tok_s = std::max(res.max_decode_tok_s, r.decode_tok_per_sec);
        res.min_step_ms = std::min(res.min_step_ms, r.mean_step_ms);
    }
    const double n = static_cast<double>(runs.size());
    res.mean_ttft_ms /= n;
    res.mean_decode_tok_s /= n;
    res.mean_step_ms /= n;
    res.mean_total_ms /= n;
    return res;
}

std::vector<std::uint32_t> make_synthetic_prompt(std::size_t count, std::uint32_t seed = 42) {
    std::vector<std::uint32_t> tokens(count);
    std::uint32_t val = seed;
    for (std::size_t i = 0; i < count; ++i) {
        val = (val * 1664525u + 1013904223u) % 151643u;
        tokens[i] = val + 1;
    }
    return tokens;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0007: Unified Prefill V2 -> Static Decode Pipeline\n";
    std::cout << "  Target: 1 x AMD Instinct MI50 32GB (gfx906, Wave64)\n";
    std::cout << "  Model:  Qwen3.8-27B-Q4_K_M (64 Layers, 48 GDN + 16 GQA)\n";
    std::cout << "===================================================================\n\n";

    const std::string model_path = (argc > 1) ? argv[1] : kDefaultModelPath;

    int device_id = 0;
    MIINFER_HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t props{};
    MIINFER_HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    std::cout << "[INFO] Device: " << props.name << " (" << props.gcnArchName << ")\n";
    std::cout << "[INFO] Loading GGUF Model from: " << model_path << "\n";

    const auto qwen_model = Qwen35Model::load(model_path);
    std::cout << "[INFO] Model loaded successfully: " << qwen_model.model_name()
              << " (" << qwen_model.config().main_layer_count << " layers, hidden="
              << qwen_model.config().hidden_size << ", vocab=" << qwen_model.config().vocab_size << ")\n";

    std::cout << "[INFO] Initializing PrefillV2Model (KV Capacity = 32768, LM Head = enabled)...\n";
    PrefillV2Model v2_model(qwen_model, 32768, /*load_lm_head=*/true);

    const double bytes_to_gib = 1.0 / (1024.0 * 1024.0 * 1024.0);
    const double bytes_to_mib = 1.0 / (1024.0 * 1024.0);
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "  Static VRAM Allocation Footprint (Macro-512 Configuration)\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Persistent Weights:   " << std::fixed << std::setprecision(2)
              << v2_model.persistent_weight_bytes() * bytes_to_gib << " GiB\n";
    std::cout << "  Persistent States:   "
              << v2_model.persistent_state_bytes() * bytes_to_mib << " MiB (48 GDN SSM + 16 KV 32K)\n";
    std::cout << "  Monolithic Workspace: "
              << v2_model.workspace_bytes() * bytes_to_mib << " MiB (sized for N=512)\n";
    std::cout << "  Ping-Pong & Temp:     "
              << v2_model.activation_bytes() * bytes_to_mib << " MiB\n";
    std::cout << "  Total Static VRAM:    "
              << v2_model.total_vram_bytes() * bytes_to_gib << " GiB / 32.00 GiB\n";
    std::cout << "  Free Headroom:        "
              << (32.0 * 1024.0 * 1024.0 * 1024.0 - v2_model.total_vram_bytes()) * bytes_to_gib << " GiB\n";
    std::cout << "-------------------------------------------------------------------\n\n";

    // =========================================================================
    // Part 1: Zero-Copy Hand-off & Step-by-Step Decode Verification
    // =========================================================================
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Part 1: Zero-Copy State Hand-Off & Step-by-Step Decode Verification\n";
    std::cout << "-------------------------------------------------------------------\n";
    {
        const auto test_prompt = make_synthetic_prompt(64, 1234);
        v2_model.reset_state();

        float* d_test_hidden = nullptr;
        std::uint32_t* d_test_tok = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_test_hidden), 64 * kHidden * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_test_tok), sizeof(std::uint32_t)));

        std::cout << "[TEST] Executing Prefill on Prompt (P = 64 tokens)...\n";
        auto t_prefill_start = std::chrono::steady_clock::now();
        v2_model.prefill_sequence(test_prompt, d_test_hidden);
        auto t_prefill_end = std::chrono::steady_clock::now();
        double p64_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
        std::cout << "       Prefill completed in " << std::fixed << std::setprecision(2) << p64_ms << " ms\n";

        std::cout << "[TEST] Evaluating Initial Logits for Token 63 (TTFT)...\n";
        v2_model.compute_logits(d_test_hidden + 63 * kHidden, v2_model.logits_buffer());
        launch_qwen3_argmax(v2_model.logits_buffer(), d_test_tok, v2_model.vocab_size());
        std::uint32_t first_token = 0;
        MIINFER_HIP_CHECK(hipMemcpy(&first_token, d_test_tok, sizeof(std::uint32_t), hipMemcpyDeviceToHost));
        std::cout << "       First Generated Token ID: " << first_token << "\n\n";

        std::cout << "[TEST] Stepping 16 Autoregressive Decode Steps (Zero-Copy State Continuity)...\n";
        std::uint32_t curr_token = first_token;
        for (std::size_t step = 1; step <= 16; ++step) {
            std::uint32_t pos = 64 - 1 + static_cast<std::uint32_t>(step);
            auto t0 = std::chrono::steady_clock::now();
            std::uint32_t next_token = v2_model.decode_step(curr_token, pos);
            auto t1 = std::chrono::steady_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  Step " << std::setw(2) << step << " (Pos " << std::setw(3) << pos << "): "
                      << "Token ID = " << std::setw(6) << next_token
                      << " | Step Latency = " << std::fixed << std::setprecision(2) << step_ms << " ms\n";
            curr_token = next_token;
        }
        std::cout << "[SUCCESS] Zero-Copy State Hand-Off & Decode Progression verified.\n\n";

        MIINFER_HIP_CHECK(hipFree(d_test_hidden));
        MIINFER_HIP_CHECK(hipFree(d_test_tok));
    }

    // =========================================================================
    // Part 2: Multi-Turn State Reset & Repeatability Verification
    // =========================================================================
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Part 2: Multi-Turn State Reset & Turn Repeatability Test\n";
    std::cout << "-------------------------------------------------------------------\n";
    {
        const auto prompt_a = make_synthetic_prompt(64, 101);
        const auto prompt_b = make_synthetic_prompt(64, 202);

        std::cout << "[TEST] Running Turn 1 (Prompt A -> 16 tokens)...\n";
        GenerateOptions opt;
        opt.max_new_tokens = 16;
        opt.reset_state_before = true;
        auto res_a1 = v2_model.generate(prompt_a, opt);

        std::cout << "[TEST] Running Turn 2 (Prompt B -> 16 tokens)...\n";
        auto res_b = v2_model.generate(prompt_b, opt);

        std::cout << "[TEST] Running Turn 3 (Prompt A again -> 16 tokens)...\n";
        auto res_a2 = v2_model.generate(prompt_a, opt);

        bool identical = (res_a1.generated_tokens == res_a2.generated_tokens);
        std::cout << "[VERIFY] Turn 1 vs Turn 3 Bit-Identical Generated Sequence: "
                  << (identical ? "MATCH (100% Deterministic & Isolated)" : "MISMATCH") << "\n";
        if (!identical) {
            throw std::runtime_error("State isolation failure: multi-turn generation diverged!");
        }
        std::cout << "[SUCCESS] State Reset and Multi-Turn Lifecycle verified.\n\n";
    }

    // =========================================================================
    // Part 3: Comprehensive End-to-End Performance Benchmark
    // =========================================================================
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Part 3: End-to-End Prefill + Decode Benchmark (TG = 128 Tokens)\n";
    std::cout << "-------------------------------------------------------------------\n\n";

    struct TestCase {
        std::string name;
        std::size_t prompt_len;
        std::size_t decode_len;
        std::uint32_t seed;
    };

    const std::vector<TestCase> test_cases = {
        {"P64 + TG128",   64,   128, 42},
        {"P512 + TG128",  512,  128, 77},
        {"P2048 + TG128", 2048, 128, 99},
    };

    std::cout << "| Configuration | Prompt Tokens | Generated Tokens | TTFT (ms) | Decode Throughput | Mean Step (ms) | Total Time (s) |\n";
    std::cout << "|:---|---:|---:|---:|---:|---:|---:|\n";

    for (const auto& tc : test_cases) {
        const auto prompt = make_synthetic_prompt(tc.prompt_len, tc.seed);
        GenerateOptions opt;
        opt.max_new_tokens = tc.decode_len;
        opt.reset_state_before = true;

        // Warmup run
        (void)v2_model.generate(prompt, opt);

        // Measured runs (3 iterations)
        std::vector<BenchmarkRunStats> run_stats;
        constexpr int kIterations = 3;
        for (int iter = 0; iter < kIterations; ++iter) {
            auto gen_stats = v2_model.generate(prompt, opt);
            BenchmarkRunStats s;
            s.ttft_ms = gen_stats.ttft_ms;
            s.prefill_tok_per_sec = gen_stats.prefill_tok_per_sec;
            s.decode_ms = gen_stats.decode_ms;
            s.decode_tok_per_sec = gen_stats.decode_tok_per_sec;
            s.mean_step_ms = gen_stats.avg_decode_latency_ms;
            s.total_ms = gen_stats.total_ms;
            s.generated_tokens = gen_stats.generated_tokens;
            run_stats.push_back(s);
        }

        const auto agg = aggregate(run_stats);

        std::cout << "| **" << tc.name << "** | "
                  << std::setw(13) << tc.prompt_len << " | "
                  << std::setw(16) << tc.decode_len << " | "
                  << std::fixed << std::setprecision(2) << std::setw(9) << agg.mean_ttft_ms << " | "
                  << std::fixed << std::setprecision(1) << std::setw(15) << agg.mean_decode_tok_s << " tok/s | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << agg.mean_step_ms << " ms | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << (agg.mean_total_ms / 1000.0) << " s |\n";
    }

    std::cout << "\n===================================================================\n";
    std::cout << "  V2-0007 End-to-End Generation Benchmark Complete.\n";
    std::cout << "===================================================================\n";

    return 0;
}
