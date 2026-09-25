#include <iostream>
#include <iomanip>
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/hip_check.hpp"
#include <hip/hip_runtime.h>

using namespace miinfer;
using namespace miinfer::prefill_v2;

int main() {
    const std::string model_path = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";
    std::cout << "===================================================================\n";
    std::cout << "  MIInfer V2-0012: Live 64K & 128K Context Envelope Allocation Test\n";
    std::cout << "  Target: 1 x AMD Instinct MI50 32GB (gfx906, Wave64)\n";
    std::cout << "===================================================================\n\n";

    std::cout << "[INFO] Loading GGUF model: " << model_path << "\n";
    const auto qwen_model = Qwen35Model::load(model_path);

    std::size_t free_bytes = 0, total_bytes = 0;
    const double to_gib = 1.0 / (1024.0 * 1024.0 * 1024.0);

    // Test 1: 32K KV
    std::cout << "\n[TEST 1] Testing 32K KV (capacity = 32768)...\n";
    {
        PrefillV2Model model_32k(qwen_model, 32768, /*load_lm_head=*/true);
        MIINFER_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
        std::cout << "  32K Static VRAM:   " << std::fixed << std::setprecision(2)
                  << model_32k.total_vram_bytes() * to_gib << " GiB\n";
        std::cout << "  Observed Free:     " << free_bytes * to_gib << " GiB / " << total_bytes * to_gib << " GiB\n";
        std::cout << "  [32K STATUS: PASS]\n";
    }

    // Test 2: 64K KV
    std::cout << "\n[TEST 2] Testing 64K KV (capacity = 65536)...\n";
    {
        PrefillV2Model model_64k(qwen_model, 65536, /*load_lm_head=*/true);
        MIINFER_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
        std::cout << "  64K Static VRAM:   " << std::fixed << std::setprecision(2)
                  << model_64k.total_vram_bytes() * to_gib << " GiB\n";
        std::cout << "  Observed Free:     " << free_bytes * to_gib << " GiB / " << total_bytes * to_gib << " GiB\n";
        std::cout << "  [64K STATUS: PASS]\n";
    }

    // Test 3: 128K KV
    std::cout << "\n[TEST 3] Testing 128K KV (capacity = 131072)...\n";
    {
        PrefillV2Model model_128k(qwen_model, 131072, /*load_lm_head=*/true);
        MIINFER_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
        std::cout << "  128K Static VRAM:  " << std::fixed << std::setprecision(2)
                  << model_128k.total_vram_bytes() * to_gib << " GiB\n";
        std::cout << "  Observed Free:     " << free_bytes * to_gib << " GiB / " << total_bytes * to_gib << " GiB\n";
        std::cout << "  [128K STATUS: PASS]\n";
    }

    std::cout << "\n===================================================================\n";
    std::cout << "  V2_128K_MEMORY_ENVELOPE_QUALIFIED\n";
    std::cout << "===================================================================\n";
    return 0;
}
