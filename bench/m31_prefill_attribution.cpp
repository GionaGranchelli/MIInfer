#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/prefill_v2/persistent_session.hpp"
#include "miinfer/qwen3_tokenizer.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;
using namespace miinfer::prefill_v2;

int main(int argc, char** argv) {
    try {
        const std::string model_path = argc > 1 ? argv[1] : "/models/Qwen3.8-27B-Q4_K_M.gguf";
        std::uint32_t context = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 8192;
        bool fp16 = false;
        std::string prompt_file;
        std::string session_path;
        for (int i = 3; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--fp16") fp16 = true;
            else if (arg == "--prompt-file" && i + 1 < argc) prompt_file = argv[++i];
            else if (arg == "--session" && i + 1 < argc) session_path = argv[++i];
            else throw std::runtime_error("usage: miinfer-m31-prefill-attribution MODEL.gguf CONTEXT [--fp16] [--prompt-file PATH] [--session PATH]");
        }
        std::uint32_t cached_prefix = 0;
        if (!session_path.empty()) {
            if (fp16 || !prompt_file.empty()) {
                throw std::runtime_error("--session cannot be combined with --fp16 or --prompt-file");
            }
            PersistentSessionHeader header{};
            if (!PersistentSession::inspect_file(session_path, header) ||
                header.kv_quant_mode != static_cast<std::uint8_t>(KvCacheQuantMode::kQ8Q8)) {
                throw std::runtime_error("session must be a valid Q8/Q8 MIInfer checkpoint");
            }
            cached_prefix = header.prefix_length;
            context = cached_prefix + kMaxPrefillBatch;
        }
        if (session_path.empty() && (context < 1024 || context % kMaxPrefillBatch != 0)) {
            throw std::runtime_error("context must be a multiple of 512 and at least 1024");
        }

        DeviceInfo device{};
        std::string error;
        if (!validate_gfx906_device(0, device, error)) {
            throw std::runtime_error("device validation failed: " + error);
        }
        print_device_info(device, std::cout);
        char pci_bus_id[32]{};
        MIINFER_HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
        std::cout << "profile_device_bdf=" << pci_bus_id << '\n';

        const auto source = Qwen35Model::load(model_path);
        if (fp16 && ::setenv("MIINFER_V2_0043_GQA_ATTENTION", "1", 1) != 0) {
            throw std::runtime_error("failed to enable the existing tiled attention path");
        }
        const auto kv_mode = fp16 ? KvCacheQuantMode::kFp16Fp16 : KvCacheQuantMode::kQ8Q8;
        PrefillV2Model model(source, context + 4096, false, kv_mode);
        std::vector<std::uint32_t> host_tokens(context);
        if (prompt_file.empty()) {
            for (std::uint32_t i = 0; i < context; ++i) {
                host_tokens[i] = (i * 37u + 101u) % source.config().vocab_size;
            }
        } else {
            std::ifstream input(prompt_file, std::ios::binary);
            if (!input) throw std::runtime_error("failed to open prompt file: " + prompt_file);
            const std::string text(std::istreambuf_iterator<char>(input), {});
            const auto encoded = Qwen3Tokenizer::load(*source.file()).encode(text);
            if (encoded.size() < context) {
                throw std::runtime_error("prompt file has fewer tokens than the requested context");
            }
            std::copy_n(encoded.begin(), context, host_tokens.begin());
            std::cout << "prompt_text_bytes=" << text.size()
                      << " prompt_tokens_available=" << encoded.size()
                      << " prompt_tokens_processed=" << context << '\n';
        }

        std::uint32_t* d_tokens = nullptr;
        float* d_hidden = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_tokens), kMaxPrefillBatch * sizeof(*d_tokens)));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_hidden), kMaxPrefillBatch * kHidden * sizeof(*d_hidden)));

        model.reset_state();
        double session_restore_ms = 0.0;
        if (!session_path.empty()) {
            std::vector<std::uint32_t> restored_tokens;
            const auto restore_start = std::chrono::steady_clock::now();
            if (!model.load_session(session_path, restored_tokens) || restored_tokens.size() != cached_prefix) {
                throw std::runtime_error("failed to restore the expected persistent prefix session");
            }
            session_restore_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - restore_start).count();
        } else {
            for (std::uint32_t pos = 0; pos + kMaxPrefillBatch < context; pos += kMaxPrefillBatch) {
                MIINFER_HIP_CHECK(hipMemcpy(
                    d_tokens, host_tokens.data() + pos, kMaxPrefillBatch * sizeof(*d_tokens), hipMemcpyHostToDevice));
                model.forward(d_tokens, pos, kMaxPrefillBatch, d_hidden);
            }
        }
        MIINFER_HIP_CHECK(hipDeviceSynchronize());

        const std::uint32_t profile_pos = session_path.empty() ? context - kMaxPrefillBatch : cached_prefix;
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
                  << " cached_prefix=" << cached_prefix << " profile_base=" << profile_pos
                  << " session_restore_ms=" << session_restore_ms
                  << " kv=" << (fp16 ? "FP16/FP16" : "Q8/Q8") << '\n'
                  << "profile_tile_gpu_ms=" << profile.total_model_ms
                  << " embedding_ms=" << profile.embedding_ms
                  << " final_norm_ms=" << profile.final_norm_ms
                  << " gqa_blocks_ms=" << gqa_total
                  << " causal_attention_ms=" << attention_total
                  << " causal_attention_pct=" << (profile.total_model_ms > 0.0 ? 100.0 * attention_total / profile.total_model_ms : 0.0)
                  << " finite_final_hidden=yes\n";
        for (std::size_t i = 0; i < profile.block_breakdowns.size(); ++i) {
            const auto& b = profile.block_breakdowns[i];
            std::cout << "block=" << i << " gdn_ms=" << (b.gdn0_ms + b.gdn1_ms + b.gdn2_ms)
                      << " gqa_ms=" << b.gqa3_ms
                      << " causal_attention_ms=" << b.gqa3_detail.causal_attn_ms
                      << " gdn0_mmq_qkv_gate_ms=" << b.gdn0_detail.qkv_gate_proj_ms
                      << " gdn0_mmq_ffn_gate_up_ms=" << b.gdn0_detail.ffn_gate_up_ms
                      << " gdn0_mmq_down_ms=" << b.gdn0_detail.swiglu_down_residual_ms
                      << " gdn1_mmq_qkv_gate_ms=" << b.gdn1_detail.qkv_gate_proj_ms
                      << " gdn1_mmq_ffn_gate_up_ms=" << b.gdn1_detail.ffn_gate_up_ms
                      << " gdn1_mmq_down_ms=" << b.gdn1_detail.swiglu_down_residual_ms
                      << " gdn2_mmq_qkv_gate_ms=" << b.gdn2_detail.qkv_gate_proj_ms
                      << " gdn2_mmq_ffn_gate_up_ms=" << b.gdn2_detail.ffn_gate_up_ms
                      << " gdn2_mmq_down_ms=" << b.gdn2_detail.swiglu_down_residual_ms
                      << " gqa_qkv_proj_ms=" << b.gqa3_detail.qkv_proj_ms
                      << " gqa_o_proj_ms=" << b.gqa3_detail.o_proj_ms
                      << " gqa_ffn_gate_up_ms=" << b.gqa3_detail.ffn_gate_up_ms
                      << " gqa_ffn_down_ms=" << b.gqa3_detail.swiglu_down_res_ms << '\n';
        }

        MIINFER_HIP_CHECK(hipFree(d_tokens));
        MIINFER_HIP_CHECK(hipFree(d_hidden));
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
