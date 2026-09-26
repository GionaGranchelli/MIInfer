#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;
using namespace miinfer::prefill_v2;

namespace {

const char* kDefaultModelPath = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";

struct GpuBuffer {
    void* ptr = nullptr;
    explicit GpuBuffer(std::size_t bytes) {
        MIINFER_HIP_CHECK(hipMalloc(&ptr, bytes));
    }
    ~GpuBuffer() {
        if (ptr != nullptr) hipFree(ptr);
    }
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
};

float elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
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

void compute_accuracy(
    const std::vector<float>& ref,
    const std::vector<float>& cand,
    double& max_abs,
    double& mae,
    double& rmse,
    double& cosine_sim) {
    max_abs = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_cand = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i];
        const double c = cand[i];
        const double diff = std::fabs(r - c);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
        sum_sq += diff * diff;
        dot += r * c;
        norm_ref += r * r;
        norm_cand += c * c;
    }
    mae = sum_abs / ref.size();
    rmse = std::sqrt(sum_sq / ref.size());
    cosine_sim = dot / (std::sqrt(norm_ref) * std::sqrt(norm_cand) + 1e-12);
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0019: Quantized KV Cache Qualification Benchmark\n";
    std::cout << "  Target: 1 x AMD Instinct MI50 32GB (gfx906, Wave64, 1606/1000 MHz)\n";
    std::cout << "  Workload: P = 65,536 cached prefix tokens, S = 512 new suffix tokens\n";
    std::cout << "===================================================================\n\n";

    int device_id = 0;
    MIINFER_HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t props{};
    MIINFER_HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    std::cout << "[INFO] Device: " << props.name << " (" << props.gcnArchName << "), CUs="
              << props.multiProcessorCount << ", Clock=" << props.clockRate / 1000 << " MHz\n\n";

    // -------------------------------------------------------------------------
    // 1. Isolated Attention Kernel Bake-Off (16 GQA Layers Equivalent)
    // -------------------------------------------------------------------------
    constexpr std::uint32_t kQueryHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kHeadDim = 256;
    constexpr std::uint32_t kGqaLayers = 16;
    constexpr float kScale = 1.0f / 16.0f;

    constexpr std::uint32_t kPrefixLen = 65536;
    constexpr std::uint32_t kSuffixLen = 512;
    constexpr std::uint32_t kCacheCap = kPrefixLen + kSuffixLen + 1024;

    const std::size_t q_elements = static_cast<std::size_t>(kSuffixLen) * kQueryHeads * kHeadDim;
    const std::size_t kv_elements = static_cast<std::size_t>(kKvHeads) * kCacheCap * kHeadDim;
    const std::size_t scale_elements = static_cast<std::size_t>(kKvHeads) * kCacheCap;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.4f);

    std::vector<float> h_q(q_elements);
    std::vector<float> h_gate(q_elements);
    std::vector<__half> h_key_f16(kv_elements);
    std::vector<__half> h_val_f16(kv_elements);
    std::vector<int8_t> h_key_q8(kv_elements);
    std::vector<__half> h_key_scales(scale_elements);
    std::vector<int8_t> h_val_q8(kv_elements);
    std::vector<__half> h_val_scales(scale_elements);

    for (auto& v : h_q) v = dist(rng);
    for (auto& v : h_gate) v = dist(rng);

    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
        for (std::uint32_t pos = 0; pos < kPrefixLen + kSuffixLen; ++pos) {
            const std::size_t base = (static_cast<std::size_t>(h) * kCacheCap + pos) * kHeadDim;
            const std::size_t sidx = static_cast<std::size_t>(h) * kCacheCap + pos;

            float max_k = 0.0f;
            std::vector<float> k_tmp(kHeadDim);
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                k_tmp[d] = dist(rng);
                max_k = std::max(max_k, std::fabs(k_tmp[d]));
            }
            const float ks = max_k / 127.0f;
            const float inv_ks = ks > 0.0f ? 1.0f / ks : 0.0f;
            h_key_scales[sidx] = __float2half(ks);
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                h_key_f16[base + d] = __float2half(k_tmp[d]);
                h_key_q8[base + d] = static_cast<int8_t>(std::clamp(std::round(k_tmp[d] * inv_ks), -127.0f, 127.0f));
            }

            float max_v = 0.0f;
            std::vector<float> v_tmp(kHeadDim);
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                v_tmp[d] = dist(rng);
                max_v = std::max(max_v, std::fabs(v_tmp[d]));
            }
            const float vs = max_v / 127.0f;
            const float inv_vs = vs > 0.0f ? 1.0f / vs : 0.0f;
            h_val_scales[sidx] = __float2half(vs);
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                h_val_f16[base + d] = __float2half(v_tmp[d]);
                h_val_q8[base + d] = static_cast<int8_t>(std::clamp(std::round(v_tmp[d] * inv_vs), -127.0f, 127.0f));
            }
        }
    }

    GpuBuffer d_q(q_elements * sizeof(float));
    GpuBuffer d_gate(q_elements * sizeof(float));
    GpuBuffer d_out_a(q_elements * sizeof(float));
    GpuBuffer d_out_b(q_elements * sizeof(float));
    GpuBuffer d_out_c(q_elements * sizeof(float));
    GpuBuffer d_out_d(q_elements * sizeof(float));

    GpuBuffer d_key_f16(kv_elements * sizeof(__half));
    GpuBuffer d_val_f16(kv_elements * sizeof(__half));
    GpuBuffer d_key_q8(kv_elements * sizeof(int8_t));
    GpuBuffer d_key_scales(scale_elements * sizeof(__half));
    GpuBuffer d_val_q8(kv_elements * sizeof(int8_t));
    GpuBuffer d_val_scales(scale_elements * sizeof(__half));

    const std::size_t split_ws_bytes = static_cast<std::size_t>(64) * kSuffixLen * kQueryHeads * (2 + kHeadDim) * sizeof(float);
    GpuBuffer d_split_ws(split_ws_bytes);

    MIINFER_HIP_CHECK(hipMemcpy(d_q.ptr, h_q.data(), q_elements * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate.ptr, h_gate.data(), q_elements * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_key_f16.ptr, h_key_f16.data(), kv_elements * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_val_f16.ptr, h_val_f16.data(), kv_elements * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_key_q8.ptr, h_key_q8.data(), kv_elements * sizeof(int8_t), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_key_scales.ptr, h_key_scales.data(), scale_elements * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_val_q8.ptr, h_val_q8.data(), kv_elements * sizeof(int8_t), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_val_scales.ptr, h_val_scales.data(), scale_elements * sizeof(__half), hipMemcpyHostToDevice));

    hipEvent_t start_ev, stop_ev;
    MIINFER_HIP_CHECK(hipEventCreate(&start_ev));
    MIINFER_HIP_CHECK(hipEventCreate(&stop_ev));

    auto bench_candidate = [&](const std::string& name, bool is_k_q8, bool is_v_q8, void* out_ptr) {
        (void)name;
        // Warmup
        for (int w = 0; w < 3; ++w) {
            miinfer::launch_qwen35_splitk_suffix_attention_quant(
                static_cast<const float*>(d_q.ptr),
                static_cast<const __half*>(d_key_f16.ptr),
                static_cast<const __half*>(d_val_f16.ptr),
                static_cast<const int8_t*>(d_key_q8.ptr),
                static_cast<const __half*>(d_key_scales.ptr),
                static_cast<const int8_t*>(d_val_q8.ptr),
                static_cast<const __half*>(d_val_scales.ptr),
                static_cast<const float*>(d_gate.ptr),
                static_cast<float*>(out_ptr),
                static_cast<float*>(d_split_ws.ptr),
                kSuffixLen, kPrefixLen, kCacheCap,
                kQueryHeads, kKvHeads, kHeadDim, kScale,
                is_k_q8, is_v_q8, 32);
        }
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        constexpr int kRounds = 15;
        std::vector<float> times;
        times.reserve(kRounds);

        for (int r = 0; r < kRounds; ++r) {
            MIINFER_HIP_CHECK(hipEventRecord(start_ev));
            // Launch 16 layers worth of attention
            for (std::uint32_t l = 0; l < kGqaLayers; ++l) {
                miinfer::launch_qwen35_splitk_suffix_attention_quant(
                    static_cast<const float*>(d_q.ptr),
                    static_cast<const __half*>(d_key_f16.ptr),
                    static_cast<const __half*>(d_val_f16.ptr),
                    static_cast<const int8_t*>(d_key_q8.ptr),
                    static_cast<const __half*>(d_key_scales.ptr),
                    static_cast<const int8_t*>(d_val_q8.ptr),
                    static_cast<const __half*>(d_val_scales.ptr),
                    static_cast<const float*>(d_gate.ptr),
                    static_cast<float*>(out_ptr),
                    static_cast<float*>(d_split_ws.ptr),
                    kSuffixLen, kPrefixLen, kCacheCap,
                    kQueryHeads, kKvHeads, kHeadDim, kScale,
                    is_k_q8, is_v_q8, 32);
            }
            MIINFER_HIP_CHECK(hipEventRecord(stop_ev));
            MIINFER_HIP_CHECK(hipEventSynchronize(stop_ev));
            times.push_back(elapsed_ms(start_ev, stop_ev));
        }
        std::sort(times.begin(), times.end());
        const float median_ms = times[times.size() / 2];
        const float min_ms = times.front();
        const float max_ms = times.back();
        const float mean_ms = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();

        return std::make_tuple(median_ms, mean_ms, min_ms, max_ms);
    };

    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  PHASE 1: 16-Layer GQA Suffix Attention Kernel Bake-Off (64K + 512)\n";
    std::cout << "-------------------------------------------------------------------\n";

    const auto [med_a, mean_a, min_a, max_a] = bench_candidate("Candidate A (FP16/FP16 Control)", false, false, d_out_a.ptr);
    const auto [med_b, mean_b, min_b, max_b] = bench_candidate("Candidate B (Q8-K / FP16-V)", true, false, d_out_b.ptr);
    const auto [med_c, mean_c, min_c, max_c] = bench_candidate("Candidate C (FP16-K / Q8-V)", false, true, d_out_c.ptr);
    const auto [med_d, mean_d, min_d, max_d] = bench_candidate("Candidate D (Q8-K  / Q8-V)", true, true, d_out_d.ptr);

    std::vector<float> h_out_a(q_elements), h_out_b(q_elements), h_out_c(q_elements), h_out_d(q_elements);
    MIINFER_HIP_CHECK(hipMemcpy(h_out_a.data(), d_out_a.ptr, q_elements * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(h_out_b.data(), d_out_b.ptr, q_elements * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(h_out_c.data(), d_out_c.ptr, q_elements * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(h_out_d.data(), d_out_d.ptr, q_elements * sizeof(float), hipMemcpyDeviceToHost));

    double diff_max_b = 0, mae_b = 0, rmse_b = 0, cos_b = 0;
    double diff_max_c = 0, mae_c = 0, rmse_c = 0, cos_c = 0;
    double diff_max_d = 0, mae_d = 0, rmse_d = 0, cos_d = 0;
    compute_accuracy(h_out_a, h_out_b, diff_max_b, mae_b, rmse_b, cos_b);
    compute_accuracy(h_out_a, h_out_c, diff_max_c, mae_c, rmse_c, cos_c);
    compute_accuracy(h_out_a, h_out_d, diff_max_d, mae_d, rmse_d, cos_d);

    // Calculate effective bandwidth & modeled traffic
    // Traffic per GQA layer: S query tokens * (QueryHeads/2 waves) * (Prefix + Suffix/2) * (K_bytes + V_bytes)
    // For V2-0016 2-head sharing wave: 12 waves per token.
    const double avg_tokens = static_cast<double>(kPrefixLen) + static_cast<double>(kSuffixLen) / 2.0;
    const double waves = 12.0; // 24 heads / 2 heads per wave
    const double kv_fetches = static_cast<double>(kGqaLayers) * kSuffixLen * waves * avg_tokens;
    const double bytes_a = kv_fetches * (2.0 * kHeadDim + 2.0 * kHeadDim); // 1024 bytes per token fetch
    const double bytes_b = kv_fetches * (1.0 * kHeadDim + 2.0 + 2.0 * kHeadDim); // 770 bytes
    const double bytes_c = kv_fetches * (2.0 * kHeadDim + 1.0 * kHeadDim + 2.0); // 770 bytes
    const double bytes_d = kv_fetches * (1.0 * kHeadDim + 2.0 + 1.0 * kHeadDim + 2.0); // 516 bytes

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Candidate A (FP16 / FP16 Control): " << med_a << " ms (mean " << mean_a
              << " ms) | Traffic: " << bytes_a / 1e12 << " TB | Eff. BW: "
              << (bytes_a / (med_a * 1e-3)) / 1e9 << " GB/s | Baseline\n";

    std::cout << "  Candidate B (Q8-K  / FP16-V)      : " << med_b << " ms (mean " << mean_b
              << " ms) | Traffic: " << bytes_b / 1e12 << " TB | Eff. BW: "
              << (bytes_b / (med_b * 1e-3)) / 1e9 << " GB/s | Speedup: "
              << med_a / med_b << "x | CosSim: " << std::setprecision(6) << cos_b << "\n";

    std::cout << "  Candidate C (FP16-K / Q8-V)       : " << med_c << " ms (mean " << mean_c
              << " ms) | Traffic: " << bytes_c / 1e12 << " TB | Eff. BW: "
              << (bytes_c / (med_c * 1e-3)) / 1e9 << " GB/s | Speedup: "
              << med_a / med_c << "x | CosSim: " << std::setprecision(6) << cos_c << "\n";

    std::cout << "  Candidate D (Q8-K  / Q8-V)        : " << med_d << " ms (mean " << mean_d
              << " ms) | Traffic: " << bytes_d / 1e12 << " TB | Eff. BW: "
              << (bytes_d / (med_d * 1e-3)) / 1e9 << " GB/s | Speedup: "
              << med_a / med_d << "x | CosSim: " << std::setprecision(6) << cos_d << "\n\n";

    // -------------------------------------------------------------------------
    // 2. Context Ladder Scaling (4K -> 32K -> 64K)
    // -------------------------------------------------------------------------
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  PHASE 2: Context Scaling Ladder (Suffix S = 512, 16 GQA Layers)\n";
    std::cout << "-------------------------------------------------------------------\n";

    for (const std::uint32_t plen : {4096u, 32768u, 65536u}) {
        const std::uint32_t cap = plen + 512 + 512;
        auto run_ladder = [&](bool is_k_q8, bool is_v_q8) {
            std::vector<float> t;
            for (int r = 0; r < 5; ++r) {
                MIINFER_HIP_CHECK(hipEventRecord(start_ev));
                for (std::uint32_t l = 0; l < kGqaLayers; ++l) {
                    miinfer::launch_qwen35_splitk_suffix_attention_quant(
                        static_cast<const float*>(d_q.ptr),
                        static_cast<const __half*>(d_key_f16.ptr),
                        static_cast<const __half*>(d_val_f16.ptr),
                        static_cast<const int8_t*>(d_key_q8.ptr),
                        static_cast<const __half*>(d_key_scales.ptr),
                        static_cast<const int8_t*>(d_val_q8.ptr),
                        static_cast<const __half*>(d_val_scales.ptr),
                        static_cast<const float*>(d_gate.ptr),
                        static_cast<float*>(d_out_d.ptr),
                        static_cast<float*>(d_split_ws.ptr),
                        kSuffixLen, plen, cap,
                        kQueryHeads, kKvHeads, kHeadDim, kScale,
                        is_k_q8, is_v_q8, 32);
                }
                MIINFER_HIP_CHECK(hipEventRecord(stop_ev));
                MIINFER_HIP_CHECK(hipEventSynchronize(stop_ev));
                t.push_back(elapsed_ms(start_ev, stop_ev));
            }
            std::sort(t.begin(), t.end());
            return t[t.size() / 2];
        };

        const float t_a = run_ladder(false, false);
        const float t_d = run_ladder(true, true);
        std::cout << "  Prefix P = " << std::setw(5) << plen << " : Candidate A (FP16) = "
                  << std::setw(8) << std::fixed << std::setprecision(2) << t_a << " ms | Candidate D (Q8) = "
                  << std::setw(8) << t_d << " ms | Speedup = " << std::setprecision(2) << t_a / t_d << "x\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // 3. Single-Token Decode Latency Check (Regressions <= 5%)
    // -------------------------------------------------------------------------
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  PHASE 3: Single-Token Decode Latency Check (16 GQA Layers)\n";
    std::cout << "-------------------------------------------------------------------\n";

    DeviceDecodeState h_dec_state{65535, 0, 0, 0};
    GpuBuffer d_dec_state(sizeof(DeviceDecodeState));
    MIINFER_HIP_CHECK(hipMemcpy(d_dec_state.ptr, &h_dec_state, sizeof(DeviceDecodeState), hipMemcpyHostToDevice));

    auto bench_decode = [&](bool is_k_q8, bool is_v_q8) {
        std::vector<float> t;
        for (int r = 0; r < 20; ++r) {
            MIINFER_HIP_CHECK(hipEventRecord(start_ev));
            for (std::uint32_t l = 0; l < kGqaLayers; ++l) {
                miinfer::launch_qwen35_tiled_online_attention_quant_dynamic(
                    static_cast<const float*>(d_q.ptr),
                    static_cast<const __half*>(d_key_f16.ptr),
                    static_cast<const __half*>(d_val_f16.ptr),
                    static_cast<const int8_t*>(d_key_q8.ptr),
                    static_cast<const __half*>(d_key_scales.ptr),
                    static_cast<const int8_t*>(d_val_q8.ptr),
                    static_cast<const __half*>(d_val_scales.ptr),
                    static_cast<const DeviceDecodeState*>(d_dec_state.ptr),
                    kCacheCap,
                    static_cast<float*>(d_out_d.ptr),
                    static_cast<const float*>(d_gate.ptr),
                    static_cast<float*>(d_out_d.ptr),
                    kQueryHeads, kKvHeads, kHeadDim, kScale,
                    is_k_q8, is_v_q8);
            }
            MIINFER_HIP_CHECK(hipEventRecord(stop_ev));
            MIINFER_HIP_CHECK(hipEventSynchronize(stop_ev));
            t.push_back(elapsed_ms(start_ev, stop_ev));
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };

    const float dec_t_a = bench_decode(false, false);
    const float dec_t_d = bench_decode(true, true);
    std::cout << "  Decode Step @ 64K Context: FP16 = " << dec_t_a << " ms | Q8 = " << dec_t_d
              << " ms | Delta = " << std::showpos << (dec_t_d - dec_t_a) / dec_t_a * 100.0f
              << "% " << std::noshowpos << (dec_t_d <= dec_t_a * 1.05f ? "[PASS: Within 5%]" : "[FAIL]") << "\n\n";

    // -------------------------------------------------------------------------
    // 4. End-to-End Full 64-Layer Model Qualification (Qwen3.8-27B-Q4_K_M)
    // -------------------------------------------------------------------------
    const std::string model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  PHASE 4: Full 64-Layer Model Suffix Prefill Qualification\n";
    std::cout << "  Model: " << model_path << "\n";
    std::cout << "-------------------------------------------------------------------\n";

    const auto qwen_model = Qwen35Model::load(model_path);
    std::cout << "[INFO] Model loaded: " << qwen_model.model_name() << "\n";

    const auto prompt_64k = make_synthetic_prompt(65536, 42);
    const auto suffix_512 = make_synthetic_prompt(512, 1001);

    auto test_full_model = [&](const std::string& label, KvCacheQuantMode quant_mode) {
        std::cout << "\n>>> Instantiating Full Model for " << label << "...\n";
        PrefillV2Model model(qwen_model, 66048, /*load_lm_head=*/true, quant_mode);

        std::size_t free_b = 0, total_b = 0;
        MIINFER_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
        std::cout << "  VRAM Allocated: " << std::fixed << std::setprecision(2)
                  << model.total_vram_bytes() / (1024.0 * 1024.0 * 1024.0) << " GiB | Free VRAM: "
                  << free_b / (1024.0 * 1024.0 * 1024.0) << " GiB\n";

        // Step 1: Prime prefix (64K) and cache prefix state
        std::cout << "  [1/3] Priming and Caching 64K Prefix State...\n";
        GenerateOptions opt_prefix;
        opt_prefix.max_new_tokens = 1;
        opt_prefix.reset_state_before = true;
        opt_prefix.cache_prefix_after = true;
        opt_prefix.cache_prefix_len = 65536;
        const auto res_prefix = model.generate(std::span<const std::uint32_t>(prompt_64k), opt_prefix);
        std::cout << "        Prefix primed in " << res_prefix.prefill_ms / 1000.0 << " s (TTFT = "
                  << res_prefix.ttft_ms << " ms)\n";

        // Step 2: Benchmark Suffix Prefill (512 tokens) with warmup and 3 runs
        std::cout << "  [2/3] Benchmarking 512 Suffix Prefill over 64K Context...\n";
        GenerateOptions opt_suffix;
        opt_suffix.max_new_tokens = 1;
        opt_suffix.reset_state_before = false;
        opt_suffix.enable_prefix_reuse = true;

        std::vector<double> suffix_times;
        for (int r = 0; r < 3; ++r) {
            const auto res_suffix = model.generate(std::span<const std::uint32_t>(suffix_512), opt_suffix);
            suffix_times.push_back(res_suffix.ttft_ms);
        }
        std::sort(suffix_times.begin(), suffix_times.end());
        const double med_suffix_ms = suffix_times[suffix_times.size() / 2];

        std::cout << "        512 Suffix TTFT: " << med_suffix_ms << " ms (min " << suffix_times.front()
                  << " ms, max " << suffix_times.back() << " ms)\n";

        // Step 3: Multi-turn 20-step sequential decode qualification
        std::cout << "  [3/3] Running 20-Step Sequential Multi-Turn Generation...\n";
        GenerateOptions opt_gen;
        opt_gen.max_new_tokens = 20;
        opt_gen.reset_state_before = false;
        opt_gen.enable_prefix_reuse = true;
        const auto res_gen = model.generate(std::span<const std::uint32_t>(suffix_512), opt_gen);

        std::cout << "        Generated Tokens (first 10): ";
        for (std::size_t i = 0; i < std::min<std::size_t>(10, res_gen.generated_tokens.size()); ++i) {
            std::cout << res_gen.generated_tokens[i] << " ";
        }
        std::cout << "\n        Decode Step Mean Latency: " << res_gen.avg_decode_latency_ms << " ms/tok\n";

        return std::make_pair(med_suffix_ms, res_gen.generated_tokens);
    };

    const auto [suffix_ttft_a, tokens_a] = test_full_model("Candidate A (FP16 KV Cache)", KvCacheQuantMode::kFp16Fp16);
    const auto [suffix_ttft_d, tokens_d] = test_full_model("Candidate D (Q8 KV Cache)", KvCacheQuantMode::kQ8Q8);

    std::cout << "\n===================================================================\n";
    std::cout << "  FINAL V2-0019 QUALIFICATION SUMMARY\n";
    std::cout << "===================================================================\n";
    std::cout << "  FP16 KV Suffix TTFT : " << std::fixed << std::setprecision(2) << suffix_ttft_a << " ms\n";
    std::cout << "  Q8 KV Suffix TTFT   : " << std::fixed << std::setprecision(2) << suffix_ttft_d << " ms\n";
    std::cout << "  Full Suffix Speedup : " << std::fixed << std::setprecision(2) << suffix_ttft_a / suffix_ttft_d << "x\n";

    bool token_match = (tokens_a.size() == tokens_d.size());
    std::size_t matching_tokens = 0;
    for (std::size_t i = 0; i < std::min(tokens_a.size(), tokens_d.size()); ++i) {
        if (tokens_a[i] == tokens_d[i]) ++matching_tokens;
    }
    std::cout << "  Multi-turn 20-token Agreement: " << matching_tokens << " / " << tokens_a.size() << "\n";
    std::cout << "===================================================================\n";

    MIINFER_HIP_CHECK(hipEventDestroy(start_ev));
    MIINFER_HIP_CHECK(hipEventDestroy(stop_ev));

    return 0;
}
