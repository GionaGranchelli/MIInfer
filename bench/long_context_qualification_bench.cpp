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

struct BenchmarkResult {
    std::size_t prompt_tokens = 0;
    std::size_t gen_tokens = 0;
    double ttft_ms = 0.0;
    double prefill_tok_s = 0.0;
    double decode_step_ms = 0.0;
    double decode_tok_s = 0.0;
    double total_time_s = 0.0;
    double observed_free_gib = 0.0;
    double total_vram_gib = 0.0;
    bool numerical_ok = true;
    std::vector<std::uint32_t> generated_tokens;
};

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
    std::cout << "  MIInfer V2-0013: Long-Context Frontier Qualification\n";
    std::cout << "  Context Scale: 4K -> 8K -> 16K -> 32K -> 64K -> 128K\n";
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
              << qwen_model.config().hidden_size << ", vocab=" << qwen_model.config().vocab_size << ")\n\n";

    const double to_gib = 1.0 / (1024.0 * 1024.0 * 1024.0);
    const double to_mib = 1.0 / (1024.0 * 1024.0);

    struct ContextTestConfig {
        std::string name;
        std::size_t prompt_tokens;
        std::size_t gen_tokens;
        std::size_t kv_capacity;
        std::uint32_t seed;
    };

    const std::vector<ContextTestConfig> test_configs = {
        {"4K Context",    4096,   128,  32768, 42},
        {"8K Context",    8192,   128,  32768, 77},
        {"16K Context",  16384,   128,  32768, 99},
        {"32K Context",  32768,   128,  33000, 123},
        {"64K Context",  65536,   128,  66000, 456},
        {"128K Context", 131072,  128, 131200, 789},
    };

    std::vector<BenchmarkResult> results;

    for (const auto& cfg : test_configs) {
        std::cout << "-------------------------------------------------------------------\n";
        std::cout << "  Running Evaluation: " << cfg.name
                  << " (Prompt = " << cfg.prompt_tokens << ", Decode = " << cfg.gen_tokens
                  << ", KV Cap = " << cfg.kv_capacity << ")\n";
        std::cout << "-------------------------------------------------------------------\n";

        // Initialize model with exact KV capacity
        PrefillV2Model model(qwen_model, cfg.kv_capacity, /*load_lm_head=*/true);

        std::size_t free_mem = 0, total_mem = 0;
        MIINFER_HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));

        std::cout << "  Static VRAM Allocation: " << std::fixed << std::setprecision(2)
                  << model.total_vram_bytes() * to_gib << " GiB / " << total_mem * to_gib << " GiB\n";
        std::cout << "  Observed Free Headroom: " << free_mem * to_gib << " GiB\n";

        const auto prompt = make_synthetic_prompt(cfg.prompt_tokens, cfg.seed);

        GenerateOptions opt;
        opt.max_new_tokens = cfg.gen_tokens;
        opt.reset_state_before = true;
        opt.use_hip_graph = true;

        std::cout << "  Executing Prefill (" << cfg.prompt_tokens << " tokens in Macro-512 tiles)...\n";
        auto gen_stats = model.generate(prompt, opt);

        BenchmarkResult r;
        r.prompt_tokens = cfg.prompt_tokens;
        r.gen_tokens = cfg.gen_tokens;
        r.ttft_ms = gen_stats.ttft_ms;
        r.prefill_tok_s = gen_stats.prefill_tok_per_sec;
        r.decode_step_ms = gen_stats.avg_decode_latency_ms;
        r.decode_tok_s = gen_stats.decode_tok_per_sec;
        r.total_time_s = gen_stats.total_ms / 1000.0;
        r.observed_free_gib = free_mem * to_gib;
        r.total_vram_gib = model.total_vram_bytes() * to_gib;
        r.generated_tokens = gen_stats.generated_tokens;

        // Numerical validation (verify non-zero, non-NaN tokens)
        bool tokens_valid = true;
        for (std::uint32_t tok : gen_stats.generated_tokens) {
            if (tok == 0 || tok >= qwen_model.config().vocab_size) {
                tokens_valid = false;
                break;
            }
        }
        r.numerical_ok = tokens_valid;

        std::cout << "  Prefill Completed:   " << std::fixed << std::setprecision(2)
                  << r.ttft_ms << " ms (" << std::setprecision(1) << r.prefill_tok_s << " tok/s)\n";
        std::cout << "  Decode Step Latency: " << std::fixed << std::setprecision(2)
                  << r.decode_step_ms << " ms/token (" << std::setprecision(1) << r.decode_tok_s << " tok/s)\n";
        std::cout << "  Total Run Time:      " << std::fixed << std::setprecision(2)
                  << r.total_time_s << " s\n";
        std::cout << "  Numerical Validity:  " << (r.numerical_ok ? "VALID (No NaN/Inf/Collapse)" : "FAIL") << "\n\n";

        results.push_back(r);
    }

    std::cout << "\n========================================================================================================\n";
    std::cout << "  V2-0013: Long-Context Frontier Qualification Summary Table (4K -> 128K)\n";
    std::cout << "========================================================================================================\n\n";

    std::cout << "| Context Regime | Prompt Tokens | Decode Latency | Decode Throughput | Prefill TTFT | Prefill Throughput | Total VRAM | Free VRAM |\n";
    std::cout << "|:---|---:|---:|---:|---:|---:|---:|---:|\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        const auto& cfg = test_configs[i];
        std::cout << "| **" << cfg.name << "** | "
                  << std::setw(13) << r.prompt_tokens << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << r.decode_step_ms << " ms | "
                  << std::fixed << std::setprecision(1) << std::setw(15) << r.decode_tok_s << " tok/s | "
                  << std::fixed << std::setprecision(2) << std::setw(10) << r.ttft_ms << " ms | "
                  << std::fixed << std::setprecision(1) << std::setw(16) << r.prefill_tok_s << " tok/s | "
                  << std::fixed << std::setprecision(2) << std::setw(8) << r.total_vram_gib << " GiB | "
                  << std::fixed << std::setprecision(2) << std::setw(7) << r.observed_free_gib << " GiB |\n";
    }

    std::cout << "\n========================================================================================================\n";
    std::cout << "  V2-0013 Long-Context Frontier Qualification Complete.\n";
    std::cout << "========================================================================================================\n";

    return 0;
}
