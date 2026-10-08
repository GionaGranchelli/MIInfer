#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-m31-0006-production-graph-decode-smoke MODEL.gguf\n";
        return 2;
    }
    try {
        const auto weights = miinfer::Qwen35Model::load(argv[1]);
        miinfer::prefill_v2::PrefillV2Model model(weights, 64, true);
        const std::vector<std::uint32_t> prompt{14990, 8, 271, 912, 42, 73, 101, 7};
        miinfer::prefill_v2::GenerateOptions options;
        options.max_new_tokens = 4;
        options.reset_state_before = true;
        options.use_hip_graph = true;
        options.temperature = 0.0F;
        options.top_p = 1.0F;
        options.top_k = 1;
        options.stop_token_ids.clear();
        options.seed = 7;

        const auto stats = model.generate(prompt, options);
        if (!stats.used_hip_graph || stats.generated_tokens.size() != options.max_new_tokens
            || !std::isfinite(stats.prefill_ms) || !std::isfinite(stats.decode_ms)
            || stats.decode_ms <= 0.0) {
            throw std::runtime_error("short production graph decode did not complete as requested");
        }
        std::cout << "M31-0006 production graph decode: PASS"
                  << " prompt_tokens=" << prompt.size()
                  << " generated_tokens=" << stats.generated_tokens.size()
                  << " used_hip_graph=" << std::boolalpha << stats.used_hip_graph
                  << " prefill_ms=" << stats.prefill_ms
                  << " decode_ms=" << stats.decode_ms
                  << " decode_forward_tokens=" << stats.generated_tokens.size() - 1
                  << " decode_tok_per_sec=" << stats.decode_tok_per_sec
                  << " avg_decode_latency_ms=" << stats.avg_decode_latency_ms
                  << " tokens=";
        for (std::size_t i = 0; i < stats.generated_tokens.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << stats.generated_tokens[i];
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M31-0006 production graph decode: FAIL: " << error.what() << '\n';
        return 1;
    }
}
