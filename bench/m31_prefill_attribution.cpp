#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;
using namespace miinfer::prefill_v2;

int main(int argc, char** argv) {
    try {
        const std::string model_path = argc > 1 ? argv[1] : "/models/Qwen3.8-27B-Q4_K_M.gguf";
        const std::uint32_t context = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 8192;
        if (context < 1024 || context % kMaxPrefillBatch != 0) {
            throw std::runtime_error("context must be a multiple of 512 and at least 1024");
        }

        DeviceInfo device{};
        std::string error;
        if (!validate_gfx906_device(0, device, error)) {
            throw std::runtime_error("device validation failed: " + error);
        }
        print_device_info(device, std::cout);

        const auto source = Qwen35Model::load(model_path);
        PrefillV2Model model(source, context, false, KvCacheQuantMode::kQ8Q8);
        std::vector<std::uint32_t> host_tokens(context);
        for (std::uint32_t i = 0; i < context; ++i) {
            host_tokens[i] = (i * 37u + 101u) % source.config().vocab_size;
        }

        std::uint32_t* d_tokens = nullptr;
        float* d_hidden = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_tokens), kMaxPrefillBatch * sizeof(*d_tokens)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_hidden), kMaxPrefillBatch * kHidden * sizeof(*d_hidden)));

        model.reset_state();
        for (std::uint32_t pos = 0; pos + kMaxPrefillBatch < context; pos += kMaxPrefillBatch) {
            MIINFER_HIP_CHECK(hipMemcpy(
                d_tokens, host_tokens.data() + pos, kMaxPrefillBatch * sizeof(*d_tokens), hipMemcpyHostToDevice));
            model.forward(d_tokens, pos, kMaxPrefillBatch, d_hidden);
        }
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        const std::uint32_t profile_pos = context - kMaxPrefillBatch;
        MIINFER_HIP_CHECK(hipMemcpy(
            d_tokens, host_tokens.data() + profile_pos, kMaxPrefillBatch * sizeof(*d_tokens), hipMemcpyHostToDevice));
        ModelProfileBreakdown profile;
        model.forward_profiled(d_tokens, profile_pos, kMaxPrefillBatch, d_hidden, profile);

        std::vector<float> last_hidden(kHidden);
        MIINFER_HIP_CHECK(hipMemcpy(
            last_hidden.data(), d_hidden + (kMaxPrefillBatch - 1) * kHidden,
            kHidden * sizeof(float), hipMemcpyDeviceToHost));
        if (!std::all_of(last_hidden.begin(), last_hidden.end(), [](float x) { return std::isfinite(x); })) {
            throw std::runtime_error("profiled final hidden state contains non-finite values");
        }

        double gqa_total = 0.0;
        double attention_total = 0.0;
        for (const auto& block : profile.block_breakdowns) {
            gqa_total += block.gqa3_ms;
            attention_total += block.gqa3_detail.causal_attn_ms;
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "profile_context=" << context << " tile=" << kMaxPrefillBatch
                  << " profile_base=" << profile_pos << " kv=Q8/Q8\n"
                  << "profile_tile_gpu_ms=" << profile.total_model_ms
                  << " embedding_ms=" << profile.embedding_ms
                  << " final_norm_ms=" << profile.final_norm_ms
                  << " gqa_blocks_ms=" << gqa_total
                  << " q8_attention_ms=" << attention_total
                  << " q8_attention_pct=" << (profile.total_model_ms > 0.0 ? 100.0 * attention_total / profile.total_model_ms : 0.0)
                  << " finite_final_hidden=yes\n";
        for (std::size_t i = 0; i < profile.block_breakdowns.size(); ++i) {
            const auto& b = profile.block_breakdowns[i];
            std::cout << "block=" << i << " gdn_ms=" << (b.gdn0_ms + b.gdn1_ms + b.gdn2_ms)
                      << " gqa_ms=" << b.gqa3_ms
                      << " q8_attention_ms=" << b.gqa3_detail.causal_attn_ms << '\n';
        }

        MIINFER_HIP_CHECK(hipFree(d_tokens));
        MIINFER_HIP_CHECK(hipFree(d_hidden));
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
