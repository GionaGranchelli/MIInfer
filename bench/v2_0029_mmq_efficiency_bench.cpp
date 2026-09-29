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
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;
using namespace miinfer::prefill_v2;

namespace {

const char* kDefaultModelPath = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() % 2 == 1) return v[v.size() / 2];
    return 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

const GgufTensor* find_tensor(const GgufFile& file, const std::string& name) {
    for (const auto& t : file.tensors()) {
        if (t.name == name) return &t;
    }
    return nullptr;
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

struct ShapeBenchmark {
    std::string name;
    std::string type;
    std::uint32_t k;
    std::uint32_t n;
    std::uint32_t m;
    std::size_t weight_bytes;
    double median_ms;
    double useful_tops;
    double effective_gb_s;
    std::uint32_t layer_occurrences;
};

} // namespace

int main(int argc, char** argv) {
    std::string model_path = kDefaultModelPath;
    int runs = 10;
    std::uint32_t test_m = 512;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::stoi(argv[++i]);
        } else if (arg == "--tokens" && i + 1 < argc) {
            test_m = std::stoul(argv[++i]);
        }
    }

    std::cout << "================================================================================\n";
    std::cout << " MIInfer V2-0029: MMQ Useful-Compute Efficiency & Instruction Analysis Bench    \n";
    std::cout << " Target: AMD Instinct MI50 (gfx906, 60 CUs, Wave64, 32GB HBM2)                  \n";
    std::cout << " Model:  " << model_path << "\n";
    std::cout << " Tokens: M = " << test_m << " | Warmup = 5 | Timed Trials = " << runs << "\n";
    std::cout << "================================================================================\n\n";

    try {
        DeviceInfo device_info;
        std::string device_err;
        if (!validate_gfx906_device(0, device_info, device_err)) {
            std::cerr << "Device validation failed: " << device_err << "\n";
            return 1;
        }

        std::cout << "Loading model file...\n";
        const auto model = Qwen35Model::load(model_path);
        const auto& file = *model.file();

        // 1. Locate representative projection tensors from blk.0 and blk.3 (GDN and GQA layers)
        const auto* t_qkv = find_tensor(file, "blk.0.attn_qkv.weight");
        const auto* t_gate = find_tensor(file, "blk.0.attn_gate.weight");
        const auto* t_ssm_out = find_tensor(file, "blk.0.ssm_out.weight");
        const auto* t_ffn_gate = find_tensor(file, "blk.0.ffn_gate.weight");
        const auto* t_ffn_down = find_tensor(file, "blk.0.ffn_down.weight");
        const auto* t_gqa_k = find_tensor(file, "blk.3.attn_k.weight");
        const auto* t_gqa_o = find_tensor(file, "blk.3.attn_output.weight");

        if (!t_qkv || !t_gate || !t_ssm_out || !t_ffn_gate || !t_ffn_down || !t_gqa_k || !t_gqa_o) {
            throw std::runtime_error("Could not find all required projection tensors in model file");
        }

        std::cout << "Packing weights into native gfx906 repacked format...\n";
        auto packed_qkv = (t_qkv->type == GgufTensorType::q4_k) ? pack_mx_q4k_repacked_tensor(*t_qkv) : pack_mx_q6k_repacked_tensor(*t_qkv);
        auto packed_gate = pack_mx_q4k_repacked_tensor(*t_gate);
        auto packed_ssm_out = pack_mx_q5k_repacked_tensor(*t_ssm_out);
        auto packed_ffn_gate = pack_mx_q4k_repacked_tensor(*t_ffn_gate);
        auto packed_ffn_down = (t_ffn_down->type == GgufTensorType::q4_k) ? pack_mx_q4k_repacked_tensor(*t_ffn_down) : pack_mx_q6k_repacked_tensor(*t_ffn_down);
        auto packed_gqa_k = pack_mx_q4k_repacked_tensor(*t_gqa_k);
        auto packed_gqa_o = pack_mx_q4k_repacked_tensor(*t_gqa_o);

        // Allocate device memory for weights
        std::uint8_t *d_qkv = nullptr, *d_gate = nullptr, *d_ssm_out = nullptr;
        std::uint8_t *d_ffn_gate = nullptr, *d_ffn_down = nullptr, *d_gqa_k = nullptr, *d_gqa_o = nullptr;

        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_qkv), packed_qkv.size()));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_gate), packed_gate.size()));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ssm_out), packed_ssm_out.size()));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ffn_gate), packed_ffn_gate.size()));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ffn_down), packed_ffn_down.size()));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_gqa_k), packed_gqa_k.size()));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_gqa_o), packed_gqa_o.size()));

        MIINFER_HIP_CHECK(hipMemcpy(d_qkv, packed_qkv.data(), packed_qkv.size(), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_gate, packed_gate.data(), packed_gate.size(), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_ssm_out, packed_ssm_out.data(), packed_ssm_out.size(), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_ffn_gate, packed_ffn_gate.data(), packed_ffn_gate.size(), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_ffn_down, packed_ffn_down.data(), packed_ffn_down.size(), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_gqa_k, packed_gqa_k.data(), packed_gqa_k.size(), hipMemcpyHostToDevice));
        MIINFER_HIP_CHECK(hipMemcpy(d_gqa_o, packed_gqa_o.data(), packed_gqa_o.size(), hipMemcpyHostToDevice));

        // Workspace for activation quantization and outputs
        constexpr std::uint32_t kMaxDim = 17408;
        float* d_input_f32 = nullptr;
        float* d_output_f32 = nullptr;
        MxQ8_1MmqBlock* d_input_q8_affine = nullptr;
        MxQ8_1MmqBlock* d_input_q8_nonaffine = nullptr;

        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input_f32), test_m * kMaxDim * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output_f32), test_m * kMaxDim * sizeof(float)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input_q8_affine), test_m * (kMaxDim / 32) * sizeof(MxQ8_1MmqBlock)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input_q8_nonaffine), test_m * (kMaxDim / 32) * sizeof(MxQ8_1MmqBlock)));

        MIINFER_HIP_CHECK(hipMemset(d_input_f32, 0x3c, test_m * kMaxDim * sizeof(float)));

        // Quantize synthetic input for testing
        launch_mx_q8_1_mmq_quantize(d_input_f32, d_input_q8_affine, test_m, kMaxDim, true, nullptr);
        launch_mx_q8_1_mmq_quantize(d_input_f32, d_input_q8_nonaffine, test_m, kMaxDim, false, nullptr);
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t ev_start, ev_stop;
        MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
        MIINFER_HIP_CHECK(hipEventCreate(&ev_stop));

        auto measure_kernel = [&](auto fn) -> double {
            for (int i = 0; i < 5; ++i) fn();
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            std::vector<double> times;
            for (int i = 0; i < runs; ++i) {
                MIINFER_HIP_CHECK(hipEventRecord(ev_start, nullptr));
                fn();
                MIINFER_HIP_CHECK(hipEventRecord(ev_stop, nullptr));
                MIINFER_HIP_CHECK(hipEventSynchronize(ev_stop));
                float ms = 0.0f;
                MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
                times.push_back(ms);
            }
            return median(times);
        };

        std::vector<ShapeBenchmark> shapes;

        // 1. FFN Gate / Up: [5120 -> 17408] (Q4_K)
        {
            double ms = measure_kernel([&]() {
                launch_mx_q4k_repacked_mmq(d_ffn_gate, d_input_q8_affine, d_output_f32, 17408, 5120, test_m, nullptr);
            });
            double gops = 2.0 * test_m * 17408.0 * 5120.0 / 1e9;
            double tops = gops / ms;
            double bw = (packed_ffn_gate.size() / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
            shapes.push_back({"FFN Gate/Up", "Q4_K", 5120, 17408, test_m, packed_ffn_gate.size(), ms, tops, bw, 128});
        }

        // 2. FFN Down: [17408 -> 5120] (Q6_K)
        {
            double ms = measure_kernel([&]() {
                if (t_ffn_down->type == GgufTensorType::q6_k) {
                    launch_mx_q6k_repacked_mmq(d_ffn_down, d_input_q8_nonaffine, d_output_f32, 5120, 17408, test_m, nullptr);
                } else {
                    launch_mx_q4k_repacked_mmq(d_ffn_down, d_input_q8_affine, d_output_f32, 5120, 17408, test_m, nullptr);
                }
            });
            double gops = 2.0 * test_m * 5120.0 * 17408.0 / 1e9;
            double tops = gops / ms;
            double bw = (packed_ffn_down.size() / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
            shapes.push_back({"FFN Down", (t_ffn_down->type == GgufTensorType::q6_k ? "Q6_K" : "Q4_K"), 17408, 5120, test_m, packed_ffn_down.size(), ms, tops, bw, 64});
        }

        // 3. GDN QKV: [5120 -> 10240] (Q4_K / Q6_K)
        {
            double ms = measure_kernel([&]() {
                if (t_qkv->type == GgufTensorType::q4_k) {
                    launch_mx_q4k_repacked_mmq(d_qkv, d_input_q8_affine, d_output_f32, 10240, 5120, test_m, nullptr);
                } else {
                    launch_mx_q6k_repacked_mmq(d_qkv, d_input_q8_nonaffine, d_output_f32, 10240, 5120, test_m, nullptr);
                }
            });
            double gops = 2.0 * test_m * 10240.0 * 5120.0 / 1e9;
            double tops = gops / ms;
            double bw = (packed_qkv.size() / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
            shapes.push_back({"GDN QKV", (t_qkv->type == GgufTensorType::q4_k ? "Q4_K" : "Q6_K"), 5120, 10240, test_m, packed_qkv.size(), ms, tops, bw, 48});
        }

        // 4. GDN Gate: [5120 -> 6144] (Q4_K)
        {
            double ms = measure_kernel([&]() {
                launch_mx_q4k_repacked_mmq(d_gate, d_input_q8_affine, d_output_f32, 6144, 5120, test_m, nullptr);
            });
            double gops = 2.0 * test_m * 6144.0 * 5120.0 / 1e9;
            double tops = gops / ms;
            double bw = (packed_gate.size() / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
            shapes.push_back({"GDN Gate", "Q4_K", 5120, 6144, test_m, packed_gate.size(), ms, tops, bw, 48});
        }

        // 5. GDN SSM Out: [6144 -> 5120] (Q5_K)
        {
            double ms = measure_kernel([&]() {
                launch_mx_q5k_repacked_mmq(d_ssm_out, d_input_q8_affine, d_output_f32, 5120, 6144, test_m, nullptr);
            });
            double gops = 2.0 * test_m * 5120.0 * 6144.0 / 1e9;
            double tops = gops / ms;
            double bw = (packed_ssm_out.size() / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
            shapes.push_back({"GDN SSM Out", "Q5_K", 6144, 5120, test_m, packed_ssm_out.size(), ms, tops, bw, 48});
        }

        // 6. GQA O: [6144 -> 5120] (Q4_K)
        {
            double ms = measure_kernel([&]() {
                launch_mx_q4k_repacked_mmq(d_gqa_o, d_input_q8_affine, d_output_f32, 5120, 6144, test_m, nullptr);
            });
            double gops = 2.0 * test_m * 5120.0 * 6144.0 / 1e9;
            double tops = gops / ms;
            double bw = (packed_gqa_o.size() / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
            shapes.push_back({"GQA Output", "Q4_K", 6144, 5120, test_m, packed_gqa_o.size(), ms, tops, bw, 16});
        }

        // 7. GQA K/V: [5120 -> 1024] (Q4_K, skinny shape)
        {
            double ms = measure_kernel([&]() {
                launch_mx_q4k_repacked_mmq(d_gqa_k, d_input_q8_affine, d_output_f32, 1024, 5120, test_m, nullptr);
            });
            double gops = 2.0 * test_m * 1024.0 * 5120.0 / 1e9;
            double tops = gops / ms;
            double bw = (packed_gqa_k.size() / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
            shapes.push_back({"GQA K/V (Skinny)", "Q4_K", 5120, 1024, test_m, packed_gqa_k.size(), ms, tops, bw, 32});
        }

        // Print shape breakdown table
        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << " Isolated Projection Kernel Performance (M=" << test_m << " tokens)\n";
        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << "| Projection Family | Type | Shape [K -> N] | Kernel ms | Useful TOPS | % Peak DP4A | Full 64L ms | Full 64L TOPs |\n";
        std::cout << "|:---|:---:|:---:|---:|---:|---:|---:|---:|\n";

        double total_extrapolated_ms = 0.0;
        double total_extrapolated_tops_work = 0.0;

        for (const auto& s : shapes) {
            double efficiency = (s.useful_tops / 49.33) * 100.0;
            double layer_total_ms = s.median_ms * s.layer_occurrences;
            double layer_work_tops = (2.0 * s.m * s.k * s.n / 1e12) * s.layer_occurrences;
            total_extrapolated_ms += layer_total_ms;
            total_extrapolated_tops_work += layer_work_tops;

            std::cout << "| " << std::left << std::setw(17) << s.name
                      << " | " << std::setw(4) << s.type
                      << " | [" << s.k << " -> " << std::setw(5) << s.n << "]"
                      << " | " << std::right << std::setw(8) << std::fixed << std::setprecision(3) << s.median_ms
                      << " ms | " << std::setw(6) << std::setprecision(1) << s.useful_tops
                      << " TOPS | " << std::setw(6) << std::setprecision(1) << efficiency
                      << "% | " << std::setw(8) << std::setprecision(1) << layer_total_ms
                      << " ms | " << std::setw(6) << std::setprecision(2) << layer_work_tops << " TOP |\n";
        }

        double model_useful_tops = (total_extrapolated_tops_work / (total_extrapolated_ms / 1000.0));
        std::cout << "\n>>> Total Extrapolated MMQ Projection Time across 64 layers: "
                  << std::fixed << std::setprecision(2) << total_extrapolated_ms << " ms\n";
        std::cout << ">>> Total Useful Arithmetic Work: " << total_extrapolated_tops_work << " TOP\n";
        std::cout << ">>> Aggregate Useful MMQ Throughput: " << std::setprecision(2) << model_useful_tops << " TOPS ("
                  << (model_useful_tops / 49.33 * 100.0) << "% of 49.33 TOPS DP4A Peak)\n\n";

        // Free device test buffers
        hipFree(d_qkv);
        hipFree(d_gate);
        hipFree(d_ssm_out);
        hipFree(d_ffn_gate);
        hipFree(d_ffn_down);
        hipFree(d_gqa_k);
        hipFree(d_gqa_o);
        hipFree(d_input_f32);
        hipFree(d_output_f32);
        hipFree(d_input_q8_affine);
        hipFree(d_input_q8_nonaffine);
        hipEventDestroy(ev_start);
        hipEventDestroy(ev_stop);

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
