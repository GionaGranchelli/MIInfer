#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t count = 0;

    explicit DeviceBuffer(std::size_t n) : count(n) {
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr) (void)hipFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct AccuracyMetrics {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double rms_error = 0.0;
    double rel_rms_error = 0.0;
    double cosine = 0.0;
    bool is_finite = true;
};

AccuracyMetrics compute_accuracy(const std::vector<float>& act, const std::vector<float>& ref) {
    AccuracyMetrics m;
    if (act.size() != ref.size() || act.empty()) return m;

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    double act_sq = 0.0;
    double dot = 0.0;

    for (std::size_t i = 0; i < act.size(); ++i) {
        float a = act[i];
        float r = ref[i];

        if (!std::isfinite(a) || !std::isfinite(r)) {
            m.is_finite = false;
        }

        double diff = std::abs(static_cast<double>(a) - static_cast<double>(r));
        if (diff > m.max_abs_error) m.max_abs_error = diff;
        sum_abs += diff;
        sum_sq += diff * diff;
        ref_sq += static_cast<double>(r) * r;
        act_sq += static_cast<double>(a) * a;
        dot += static_cast<double>(a) * r;
    }

    m.mean_abs_error = sum_abs / act.size();
    m.rms_error = std::sqrt(sum_sq / act.size());
    double ref_rms = std::sqrt(ref_sq / act.size());
    m.rel_rms_error = (ref_rms > 0.0) ? (m.rms_error / ref_rms) : 0.0;

    double denom = std::sqrt(act_sq) * std::sqrt(ref_sq);
    m.cosine = (denom > 0.0) ? (dot / denom) : 0.0;

    return m;
}

} // namespace

int main() {
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer Prefill V2-0006: Standalone Attention Bakeoff\n";
    std::cout << "  Target: 1 x AMD Instinct MI50 (gfx906, Wave64)\n";
    std::cout << "  Qwen3.8 GQA: 24 Q Heads, 4 KV Heads, Head Dim 256\n";
    std::cout << "  Control:   launch_qwen35_tiled_online_attention_batch_f16 (1Q x 1H Wave64)\n";
    std::cout << "  Candidate: launch_prefill_v2_query_tiled_attention_f16 (8 Tokens x 2 Q Heads)\n";
    std::cout << "===================================================================\n\n";

    constexpr std::uint32_t kQueryHeads = 24;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint32_t kHeadDim = 256;
    constexpr std::uint32_t kMaxCapacity = 16384;
    constexpr float kScale = 1.0F / 16.0F; // 1.0 / sqrt(256)

    // Allocate synthetic test inputs
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.05F, 0.05F);

    std::vector<float> h_q(kMaxCapacity * kQueryHeads * kHeadDim);
    std::vector<float> h_gate(kMaxCapacity * kQueryHeads * kHeadDim);
    std::vector<__half> h_k(static_cast<std::size_t>(kKvHeads) * kMaxCapacity * kHeadDim);
    std::vector<__half> h_v(static_cast<std::size_t>(kKvHeads) * kMaxCapacity * kHeadDim);

    for (auto& v : h_q) v = dist(rng);
    for (auto& v : h_gate) v = dist(rng);
    for (auto& v : h_k) v = __float2half(dist(rng));
    for (auto& v : h_v) v = __float2half(dist(rng));

    DeviceBuffer<float> d_q(h_q.size());
    DeviceBuffer<float> d_gate(h_gate.size());
    DeviceBuffer<__half> d_k(h_k.size());
    DeviceBuffer<__half> d_v(h_v.size());

    DeviceBuffer<float> d_out_control(h_q.size());
    DeviceBuffer<float> d_out_candidate_a(h_q.size());
    DeviceBuffer<float> d_out_candidate_b2(h_q.size());
    DeviceBuffer<float> d_out_candidate_b4(h_q.size());

    // Split workspace for max 8 splits: (max_cap * query_heads) * 8 * (1 max + 1 sum + 256 acc)
    constexpr std::size_t kMaxSplitWorkspaceFloats = static_cast<std::size_t>(kMaxCapacity) * kQueryHeads * 8 * 258;
    DeviceBuffer<float> d_split_workspace(kMaxSplitWorkspaceFloats);

    MIINFER_HIP_CHECK(hipMemcpy(d_q.ptr, h_q.data(), h_q.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate.ptr, h_gate.data(), h_gate.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_k.ptr, h_k.data(), h_k.size() * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_v.ptr, h_v.data(), h_v.size() * sizeof(__half), hipMemcpyHostToDevice));

    //-----------------------------------------------------------------
    // Part 1: Numerical Correctness Gate across Sequence Lengths
    //-----------------------------------------------------------------
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Part 1: Numerical Correctness Gate (Control vs Candidates)\n";
    std::cout << "-------------------------------------------------------------------\n";

    const std::vector<std::pair<std::uint32_t, std::uint32_t>> test_cases = {
        {64, 0},
        {128, 0},
        {512, 0},
        {1024, 0},
        {2048, 0},
        {4096, 0},
        {8192, 0},
        {512, 4096}, // Continuation test with non-zero base_position
    };

    std::cout << "| Token Count (N) | Base Pos | Candidate | Max Abs Error | Mean Abs Error | Rel RMS Error | Cosine | Status |\n";
    std::cout << "|---:|---:|:---|---:|---:|---:|---:|:---|\n";

    bool all_passed = true;

    for (const auto& [n_tok, base_pos] : test_cases) {
        MIINFER_HIP_CHECK(hipMemset(d_out_control.ptr, 0, n_tok * kQueryHeads * kHeadDim * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(d_out_candidate_a.ptr, 0, n_tok * kQueryHeads * kHeadDim * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(d_out_candidate_b2.ptr, 0, n_tok * kQueryHeads * kHeadDim * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(d_out_candidate_b4.ptr, 0, n_tok * kQueryHeads * kHeadDim * sizeof(float)));

        // Run Control
        miinfer::launch_qwen35_tiled_online_attention_batch_f16(
            d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_control.ptr,
            n_tok, base_pos, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale);

        // Run Candidate A (Query-Tiled)
        miinfer::launch_prefill_v2_query_tiled_attention_f16(
            d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_candidate_a.ptr,
            n_tok, base_pos, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale);

        // Run Candidate B2 (Split-KV S=2)
        miinfer::launch_prefill_v2_split_kv_attention_f16(
            d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_candidate_b2.ptr, d_split_workspace.ptr,
            n_tok, base_pos, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale, 2);

        // Run Candidate B4 (Split-KV S=4)
        miinfer::launch_prefill_v2_split_kv_attention_f16(
            d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_candidate_b4.ptr, d_split_workspace.ptr,
            n_tok, base_pos, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale, 4);

        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> host_control(n_tok * kQueryHeads * kHeadDim);
        std::vector<float> host_cand_a(n_tok * kQueryHeads * kHeadDim);
        std::vector<float> host_cand_b2(n_tok * kQueryHeads * kHeadDim);
        std::vector<float> host_cand_b4(n_tok * kQueryHeads * kHeadDim);

        MIINFER_HIP_CHECK(hipMemcpy(host_control.data(), d_out_control.ptr, host_control.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(host_cand_a.data(), d_out_candidate_a.ptr, host_cand_a.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(host_cand_b2.data(), d_out_candidate_b2.ptr, host_cand_b2.size() * sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(host_cand_b4.data(), d_out_candidate_b4.ptr, host_cand_b4.size() * sizeof(float), hipMemcpyDeviceToHost));

        auto acc_a = compute_accuracy(host_cand_a, host_control);
        auto acc_b2 = compute_accuracy(host_cand_b2, host_control);
        auto acc_b4 = compute_accuracy(host_cand_b4, host_control);

        bool pass_a = acc_a.is_finite && (acc_a.cosine > 0.999) && (acc_a.rel_rms_error < 0.05);
        bool pass_b2 = acc_b2.is_finite && (acc_b2.cosine > 0.999) && (acc_b2.rel_rms_error < 0.05);
        bool pass_b4 = acc_b4.is_finite && (acc_b4.cosine > 0.999) && (acc_b4.rel_rms_error < 0.05);

        if (!pass_a || !pass_b2 || !pass_b4) all_passed = false;

        std::cout << "| " << std::setw(15) << n_tok << " | "
                  << std::setw(8) << base_pos << " | "
                  << "Cand A (Q-Tile)  | "
                  << std::setw(13) << std::scientific << std::setprecision(4) << acc_a.max_abs_error << " | "
                  << std::setw(14) << acc_a.mean_abs_error << " | "
                  << std::setw(13) << acc_a.rel_rms_error << " | "
                  << std::fixed << std::setprecision(6) << acc_a.cosine << " | "
                  << (pass_a ? "PASS" : "FAIL") << " |\n";

        std::cout << "| " << std::setw(15) << n_tok << " | "
                  << std::setw(8) << base_pos << " | "
                  << "Cand B2 (Split2) | "
                  << std::setw(13) << std::scientific << std::setprecision(4) << acc_b2.max_abs_error << " | "
                  << std::setw(14) << acc_b2.mean_abs_error << " | "
                  << std::setw(13) << acc_b2.rel_rms_error << " | "
                  << std::fixed << std::setprecision(6) << acc_b2.cosine << " | "
                  << (pass_b2 ? "PASS" : "FAIL") << " |\n";

        std::cout << "| " << std::setw(15) << n_tok << " | "
                  << std::setw(8) << base_pos << " | "
                  << "Cand B4 (Split4) | "
                  << std::setw(13) << std::scientific << std::setprecision(4) << acc_b4.max_abs_error << " | "
                  << std::setw(14) << acc_b4.mean_abs_error << " | "
                  << std::setw(13) << acc_b4.rel_rms_error << " | "
                  << std::fixed << std::setprecision(6) << acc_b4.cosine << " | "
                  << (pass_b4 ? "PASS" : "FAIL") << " |\n";
    }

    if (!all_passed) {
        std::cerr << "\n[FATAL] Correctness verification failed!\n";
        return 1;
    }
    std::cout << "\n[SUCCESS] Correctness Gate PASSED across all test configurations.\n\n";

    //-----------------------------------------------------------------
    // Part 2: Standalone Attention Performance Bakeoff
    //-----------------------------------------------------------------
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Part 2: Standalone Attention Performance Benchmark (1 Layer)\n";
    std::cout << "-------------------------------------------------------------------\n";

    const std::vector<std::uint32_t> bench_lengths = {512, 1024, 2048, 4096, 8192};

    std::cout << "\n| Sequence Length (N) | Control (ms) | Cand A (ms) | Cand B2 (ms) | Cand B4 (ms) | Best vs Control |\n";
    std::cout << "|---:|---:|---:|---:|---:|---:|\n";

    for (std::uint32_t n_tok : bench_lengths) {
        // Warm up
        for (int w = 0; w < 2; ++w) {
            miinfer::launch_qwen35_tiled_online_attention_batch_f16(
                d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_control.ptr,
                n_tok, 0, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale);
            miinfer::launch_prefill_v2_query_tiled_attention_f16(
                d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_candidate_a.ptr,
                n_tok, 0, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale);
            miinfer::launch_prefill_v2_split_kv_attention_f16(
                d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_candidate_b2.ptr, d_split_workspace.ptr,
                n_tok, 0, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale, 2);
            miinfer::launch_prefill_v2_split_kv_attention_f16(
                d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_candidate_b4.ptr, d_split_workspace.ptr,
                n_tok, 0, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale, 4);
        }
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        const int kIters = 10;
        hipEvent_t start, stop;
        MIINFER_HIP_CHECK(hipEventCreate(&start));
        MIINFER_HIP_CHECK(hipEventCreate(&stop));

        const auto measure_path = [&](auto fn) {
            std::vector<float> times(kIters);
            for (int i = 0; i < kIters; ++i) {
                MIINFER_HIP_CHECK(hipEventRecord(start, nullptr));
                fn();
                MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
                MIINFER_HIP_CHECK(hipEventSynchronize(stop));
                MIINFER_HIP_CHECK(hipEventElapsedTime(&times[i], start, stop));
            }
            std::sort(times.begin(), times.end());
            return times[kIters / 2];
        };

        float ctrl_ms = measure_path([&] {
            miinfer::launch_qwen35_tiled_online_attention_batch_f16(
                d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_control.ptr,
                n_tok, 0, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale);
        });

        float cand_a_ms = measure_path([&] {
            miinfer::launch_prefill_v2_query_tiled_attention_f16(
                d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_candidate_a.ptr,
                n_tok, 0, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale);
        });

        float cand_b2_ms = measure_path([&] {
            miinfer::launch_prefill_v2_split_kv_attention_f16(
                d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_candidate_b2.ptr, d_split_workspace.ptr,
                n_tok, 0, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale, 2);
        });

        float cand_b4_ms = measure_path([&] {
            miinfer::launch_prefill_v2_split_kv_attention_f16(
                d_q.ptr, d_k.ptr, d_v.ptr, d_gate.ptr, d_out_candidate_b4.ptr, d_split_workspace.ptr,
                n_tok, 0, kMaxCapacity, kQueryHeads, kKvHeads, kHeadDim, kScale, 4);
        });

        MIINFER_HIP_CHECK(hipEventDestroy(start));
        MIINFER_HIP_CHECK(hipEventDestroy(stop));

        float best_cand = std::min({cand_a_ms, cand_b2_ms, cand_b4_ms});
        float best_speedup = ctrl_ms / best_cand;

        std::cout << "| " << std::setw(19) << n_tok << " | "
                  << std::setw(12) << std::fixed << std::setprecision(3) << ctrl_ms << " | "
                  << std::setw(11) << cand_a_ms << " | "
                  << std::setw(12) << cand_b2_ms << " | "
                  << std::setw(12) << cand_b4_ms << " | "
                  << std::setw(14) << std::setprecision(3) << best_speedup << "x |\n";
    }

    std::cout << "\n===================================================================\n";
    std::cout << "  Attention Bakeoff Complete.\n";
    std::cout << "===================================================================\n";

    return 0;
}
