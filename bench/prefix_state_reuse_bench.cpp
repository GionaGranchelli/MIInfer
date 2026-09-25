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
#include "miinfer/prefill_v2/reusable_context.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;
using namespace miinfer::prefill_v2;

namespace {

const char* kDefaultModelPath = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";

struct MatrixScenarioResult {
    std::string name;
    std::size_t prefix_len = 0;
    std::size_t suffix_len = 0;
    std::size_t gen_tokens = 0;
    std::size_t kv_capacity = 0;

    // Cold metrics
    double cold_ttft_ms = 0.0;
    double cold_prefill_tok_s = 0.0;
    double cold_decode_step_ms = 0.0;
    double cold_decode_tok_s = 0.0;
    std::vector<std::uint32_t> cold_generated_tokens;

    // Reuse metrics
    bool reuse_hit = false;
    std::uint32_t prefix_tokens_reused = 0;
    std::uint32_t suffix_tokens_dispatched = 0;
    double restore_ms = 0.0;
    double suffix_prefill_ms = 0.0;
    double reuse_ttft_ms = 0.0;
    double suffix_tok_s = 0.0;
    double reuse_decode_step_ms = 0.0;
    double reuse_decode_tok_s = 0.0;
    std::vector<std::uint32_t> reuse_generated_tokens;

    // Derived
    double speedup_ratio = 0.0;
    double decode_delta_pct = 0.0;
    bool parity_pass = false;
    bool numerical_ok = true;
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

void test_failure_cases() {
    std::cout << "\n===================================================================\n";
    std::cout << "  Running ReusableContext Failure Case & Validation Suite\n";
    std::cout << "===================================================================\n";

    ReusableContext ctx("Qwen3.8-27B", "Q4_K_M");
    std::vector<RecurrentLayerStateStorage> dummy_states(48);

    auto prefix_tokens = make_synthetic_prompt(512, 100);
    ctx.save(prefix_tokens, dummy_states);

    // 1. Exact match test
    auto prompt_exact = prefix_tokens;
    prompt_exact.push_back(1234);
    auto res1 = ctx.check_match(prompt_exact, "Qwen3.8-27B", "Q4_K_M");
    std::cout << "  [1] Exact Prefix Match:        " << (res1 == ReusableContext::MatchResult::ExactMatch ? "PASS (ExactMatch)" : "FAIL") << "\n";

    // 2. Model ID mismatch
    auto res2 = ctx.check_match(prompt_exact, "Qwen3.8-32B", "Q4_K_M");
    std::cout << "  [2] Model ID Mismatch:         " << (res2 == ReusableContext::MatchResult::ModelMismatch ? "PASS (ModelMismatch)" : "FAIL") << "\n";

    // 3. Quantization mismatch
    auto res3 = ctx.check_match(prompt_exact, "Qwen3.8-27B", "Q8_0");
    std::cout << "  [3] Quantization Mismatch:     " << (res3 == ReusableContext::MatchResult::QuantizationMismatch ? "PASS (QuantizationMismatch)" : "FAIL") << "\n";

    // 4. Token mismatch at index 50
    auto prompt_corrupt = prompt_exact;
    prompt_corrupt[50] ^= 0x5555;
    auto res4 = ctx.check_match(prompt_corrupt, "Qwen3.8-27B", "Q4_K_M");
    std::cout << "  [4] Token Content Mismatch:    " << (res4 == ReusableContext::MatchResult::PrefixMismatch ? "PASS (PrefixMismatch)" : "FAIL") << "\n";

    // 5. Prompt shorter than prefix
    auto prompt_short = make_synthetic_prompt(256, 100);
    auto res5 = ctx.check_match(prompt_short, "Qwen3.8-27B", "Q4_K_M");
    std::cout << "  [5] Prompt Shorter than Prefix:" << (res5 == ReusableContext::MatchResult::PromptShorterThanPrefix ? "PASS (PromptShorterThanPrefix)" : "FAIL") << "\n";

    // 6. Clear / Empty Cache
    ctx.clear();
    auto res6 = ctx.check_match(prompt_exact, "Qwen3.8-27B", "Q4_K_M");
    std::cout << "  [6] Empty Cache Check:         " << (res6 == ReusableContext::MatchResult::EmptyCache ? "PASS (EmptyCache)" : "FAIL") << "\n";
    std::cout << "-------------------------------------------------------------------\n";
}

} // namespace

int main(int argc, char** argv) {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0014: Prefix & State Reuse with Suffix-Only Prefill\n";
    std::cout << "  Target: 1 x AMD Instinct MI50 32GB (gfx906, Wave64)\n";
    std::cout << "  Model:  Qwen3.8-27B-Q4_K_M (64 Layers: 48 GDN + 16 GQA)\n";
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

    const double to_gib = 1.0 / (1024.0 * 1024.0 * 1024.0);
    const double to_mib = 1.0 / (1024.0 * 1024.0);

    // Run failure case & unit validation
    test_failure_cases();

    struct MatrixConfig {
        std::string name;
        std::size_t prefix_len;
        std::size_t suffix_len;
        std::size_t gen_tokens;
        std::size_t kv_capacity;
        std::uint32_t prefix_seed;
        std::uint32_t suffix_seed;
    };

    const std::vector<MatrixConfig> matrix_configs = {
        {"Scenario 1: 4K Control",         4096,   512,  128,  32768, 42, 1042},
        {"Scenario 2: 32K Agent History",   32768,   512,  128,  34000, 77, 1077},
        {"Scenario 3: 64K Large History",   65536,   512,  128,  67000, 99, 1099},
        {"Scenario 4: 64K Large Suffix",    65536,  4096,  128,  70000, 99, 2099},
        {"Scenario 5: 128K Frontier",      128000,   512,  128, 131200, 123, 3123},
    };

    std::vector<MatrixScenarioResult> results;

    for (const auto& cfg : matrix_configs) {
        std::cout << "\n===================================================================\n";
        std::cout << "  Executing " << cfg.name << "\n";
        std::cout << "  Prefix = " << cfg.prefix_len << " tokens | Suffix = " << cfg.suffix_len
                  << " tokens | Decode = " << cfg.gen_tokens << " | KV Cap = " << cfg.kv_capacity << "\n";
        std::cout << "===================================================================\n";

        // Create model with exact KV capacity
        PrefillV2Model model(qwen_model, cfg.kv_capacity, /*load_lm_head=*/true);

        std::size_t free_mem = 0, total_mem = 0;
        MIINFER_HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
        std::cout << "  Static VRAM Allocation: " << std::fixed << std::setprecision(2)
                  << model.total_vram_bytes() * to_gib << " GiB / " << total_mem * to_gib << " GiB\n";
        std::cout << "  GDN Checkpoint Buffer:  " << model.cached_state_bytes() * to_mib << " MiB\n";
        std::cout << "  Observed Free Headroom: " << free_mem * to_gib << " GiB\n\n";

        // Generate synthetic prefix and suffix
        const auto prefix_tokens = make_synthetic_prompt(cfg.prefix_len, cfg.prefix_seed);
        const auto suffix_tokens = make_synthetic_prompt(cfg.suffix_len, cfg.suffix_seed);

        std::vector<std::uint32_t> full_prompt;
        full_prompt.reserve(cfg.prefix_len + cfg.suffix_len);
        full_prompt.insert(full_prompt.end(), prefix_tokens.begin(), prefix_tokens.end());
        full_prompt.insert(full_prompt.end(), suffix_tokens.begin(), suffix_tokens.end());

        MatrixScenarioResult r;
        r.name = cfg.name;
        r.prefix_len = cfg.prefix_len;
        r.suffix_len = cfg.suffix_len;
        r.gen_tokens = cfg.gen_tokens;
        r.kv_capacity = cfg.kv_capacity;

        // -----------------------------------------------------------------
        // [A] COLD BASELINE RUN: Prefill full_prompt (Prefix + Suffix)
        // -----------------------------------------------------------------
        std::cout << "  [Cold Run] Prefilling full prompt (" << full_prompt.size() << " tokens)...\n";
        GenerateOptions cold_opt;
        cold_opt.max_new_tokens = cfg.gen_tokens;
        cold_opt.reset_state_before = true;
        cold_opt.enable_prefix_reuse = false;
        cold_opt.cache_prefix_after = false;
        cold_opt.use_hip_graph = true;

        auto cold_stats = model.generate(full_prompt, cold_opt);
        r.cold_ttft_ms = cold_stats.ttft_ms;
        r.cold_prefill_tok_s = cold_stats.prefill_tok_per_sec;
        r.cold_decode_step_ms = cold_stats.avg_decode_latency_ms;
        r.cold_decode_tok_s = cold_stats.decode_tok_per_sec;
        r.cold_generated_tokens = cold_stats.generated_tokens;

        std::cout << "    Cold TTFT:           " << std::fixed << std::setprecision(2)
                  << r.cold_ttft_ms << " ms (" << std::setprecision(1) << r.cold_prefill_tok_s << " tok/s)\n";
        std::cout << "    Cold Decode Latency: " << std::fixed << std::setprecision(2)
                  << r.cold_decode_step_ms << " ms/token (" << std::setprecision(1) << r.cold_decode_tok_s << " tok/s)\n\n";

        // -----------------------------------------------------------------
        // [B] TURN 1: Prefill Prefix & Cache Reusable State
        // -----------------------------------------------------------------
        std::cout << "  [Turn 1] Prefilling Prefix (" << prefix_tokens.size() << " tokens) and caching state...\n";
        GenerateOptions turn1_opt;
        turn1_opt.max_new_tokens = cfg.gen_tokens;
        turn1_opt.reset_state_before = true;
        turn1_opt.enable_prefix_reuse = false;
        turn1_opt.cache_prefix_after = true;
        turn1_opt.cache_prefix_len = cfg.prefix_len;
        turn1_opt.use_hip_graph = true;

        auto turn1_stats = model.generate(prefix_tokens, turn1_opt);
        std::cout << "    Turn 1 TTFT:         " << std::fixed << std::setprecision(2)
                  << turn1_stats.ttft_ms << " ms\n";
        std::cout << "    Turn 1 Decode Done:  Generated " << turn1_stats.generated_tokens.size() << " tokens.\n";
        std::cout << "    ReusableContext has: " << model.reusable_context().prefix_length()
                  << " prefix tokens cached (" << (model.reusable_context().has_valid_prefix() ? "VALID" : "INVALID") << ")\n\n";

        // -----------------------------------------------------------------
        // [C] TURN 2: Reuse Prefix & Prefill ONLY Suffix
        // -----------------------------------------------------------------
        std::cout << "  [Turn 2 (Reuse)] Suffix-only prefill on Full Prompt (" << full_prompt.size() << " tokens)...\n";
        GenerateOptions turn2_opt;
        turn2_opt.max_new_tokens = cfg.gen_tokens;
        turn2_opt.reset_state_before = false; // Must preserve KV cache!
        turn2_opt.enable_prefix_reuse = true;
        turn2_opt.cache_prefix_after = false;
        turn2_opt.use_hip_graph = true;

        auto reuse_stats = model.generate(full_prompt, turn2_opt);
        r.reuse_hit = reuse_stats.reuse_hit;
        r.prefix_tokens_reused = reuse_stats.prefix_tokens_reused;
        r.suffix_tokens_dispatched = reuse_stats.suffix_tokens_dispatched;
        r.restore_ms = reuse_stats.restore_ms;
        r.suffix_prefill_ms = reuse_stats.suffix_prefill_ms;
        r.reuse_ttft_ms = reuse_stats.ttft_ms;
        r.suffix_tok_s = (r.reuse_ttft_ms > 0.0) ? (cfg.suffix_len * 1000.0 / r.reuse_ttft_ms) : 0.0;
        r.reuse_decode_step_ms = reuse_stats.avg_decode_latency_ms;
        r.reuse_decode_tok_s = reuse_stats.decode_tok_per_sec;
        r.reuse_generated_tokens = reuse_stats.generated_tokens;

        r.speedup_ratio = (r.reuse_ttft_ms > 0.0) ? (r.cold_ttft_ms / r.reuse_ttft_ms) : 1.0;
        r.decode_delta_pct = ((r.reuse_decode_step_ms - r.cold_decode_step_ms) / r.cold_decode_step_ms) * 100.0;

        // Verify greedy parity
        r.parity_pass = (r.cold_generated_tokens == r.reuse_generated_tokens);

        std::cout << "    Reuse Hit:           " << (r.reuse_hit ? "YES" : "NO") << "\n";
        std::cout << "    Prefix Tokens Reused:" << r.prefix_tokens_reused << "\n";
        std::cout << "    Suffix Dispatched:   " << r.suffix_tokens_dispatched << "\n";
        std::cout << "    GDN Restore Latency: " << std::fixed << std::setprecision(3) << r.restore_ms << " ms\n";
        std::cout << "    Suffix Prefill Time: " << std::fixed << std::setprecision(2) << r.suffix_prefill_ms << " ms\n";
        std::cout << "    Reuse Total TTFT:    " << std::fixed << std::setprecision(2) << r.reuse_ttft_ms << " ms\n";
        std::cout << "    TTFT Speedup:        " << std::fixed << std::setprecision(2) << r.speedup_ratio << "x\n";
        std::cout << "    Reuse Decode Latency:" << std::fixed << std::setprecision(2) << r.reuse_decode_step_ms
                  << " ms/token (Delta: " << std::showpos << std::setprecision(1) << r.decode_delta_pct << "%)\n" << std::noshowpos;
        std::cout << "    Greedy Parity:       " << (r.parity_pass ? "EXACT MATCH (PASS)" : "MISMATCH (FAIL)") << "\n";

        results.push_back(r);
    }

    std::cout << "\n====================================================================================================================================\n";
    std::cout << "  V2-0014: Prefix / State Reuse vs Cold Execution Summary Table\n";
    std::cout << "====================================================================================================================================\n\n";

    std::cout << "| Scenario | Prefix / Suffix | Cold TTFT | Reuse TTFT | TTFT Speedup | Suffix Dispatched | Restore Latency | Decode Latency | Parity |\n";
    std::cout << "|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|\n";

    for (const auto& r : results) {
        std::string p_s = std::to_string(r.prefix_len / 1024) + "K + " + std::to_string(r.suffix_len);
        std::cout << "| **" << r.name << "** | "
                  << std::setw(11) << p_s << " | "
                  << std::fixed << std::setprecision(2) << std::setw(9) << r.cold_ttft_ms << " ms | "
                  << std::fixed << std::setprecision(2) << std::setw(10) << r.reuse_ttft_ms << " ms | "
                  << std::fixed << std::setprecision(1) << std::setw(10) << r.speedup_ratio << "x | "
                  << std::setw(17) << r.suffix_tokens_dispatched << " | "
                  << std::fixed << std::setprecision(3) << std::setw(13) << r.restore_ms << " ms | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << r.reuse_decode_step_ms << " ms | "
                  << (r.parity_pass ? "PASS" : "FAIL") << " |\n";
    }

    std::cout << "\n====================================================================================================================================\n";
    std::cout << "  V2-0014 Qualification Campaign Complete.\n";
    std::cout << "====================================================================================================================================\n";

    return 0;
}
