#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using miinfer::prefill_v2::GenerateOptions;
using miinfer::prefill_v2::GenerateStats;

GenerateOptions options(bool reuse, bool cache) {
    GenerateOptions result;
    result.max_new_tokens = 1;
    result.reset_state_before = true;
    result.use_hip_graph = false;
    result.enable_prefix_reuse = reuse;
    result.cache_prefix_after = cache;
    result.temperature = 0.7F;
    result.top_p = 1.0F;
    result.top_k = 1;
    result.seed = 7;
    result.stop_token_ids.clear();
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_same_tokens(const GenerateStats& lhs, const GenerateStats& rhs, const char* message) {
    require(lhs.generated_tokens == rhs.generated_tokens, message);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-prefill-v2-exact-prefix-reuse-test MODEL.gguf\n";
        return 2;
    }

    try {
        auto model = miinfer::Qwen35Model::load(argv[1]);
        const std::vector<std::uint32_t> prompt = {14990, 8, 271, 912, 42, 73, 101, 7};
        auto extended = prompt;
        extended.push_back(314);

        std::vector<std::uint32_t> cold_tokens;
        {
            miinfer::prefill_v2::PrefillV2Model engine(model, 1024, true);
            const auto cold = engine.generate(prompt, options(true, true));
            const auto zero_one = engine.generate(prompt, options(true, true));
            const auto zero_two = engine.generate(prompt, options(true, true));
            const auto suffix = engine.generate(extended, options(true, true));
            const auto clean_extended = engine.generate(extended, options(false, false));

            auto mismatch = extended;
            mismatch.back() ^= 1U;
            const auto rejected = engine.generate(mismatch, options(true, false));

            require(!cold.reuse_hit && cold.prefix_tokens_replayed == prompt.size(), "cold path reused or did not replay prompt");
            require(zero_one.reuse_hit && zero_one.prefix_tokens_reused == prompt.size()
                        && zero_one.suffix_tokens_executed == 0 && zero_one.prefix_tokens_replayed == 0
                        && zero_one.checkpoint_position == prompt.size(), "first zero-suffix hit failed");
            require(zero_two.reuse_hit && zero_two.suffix_tokens_executed == 0, "repeated zero-suffix hit failed");
            require(suffix.reuse_hit && suffix.prefix_tokens_reused == prompt.size()
                        && suffix.suffix_tokens_executed == 1, "nonzero suffix hit failed");
            require_same_tokens(cold, zero_one, "zero-suffix output differs from clean prefix output");
            require_same_tokens(zero_one, zero_two, "repeated zero-suffix output differs");
            require_same_tokens(suffix, clean_extended, "suffix output differs from clean extended output");
            require(!rejected.reuse_hit && rejected.prefix_tokens_replayed == mismatch.size(), "mismatch did not fall back");
            cold_tokens = cold.generated_tokens;
        }

        miinfer::prefill_v2::PrefillV2Model fresh_engine(model, 1024, true);
        const auto fresh = fresh_engine.generate(prompt, options(true, true));

        require(!fresh.reuse_hit, "fresh session unexpectedly reused state");
        require(fresh.generated_tokens == cold_tokens, "fresh-session output differs from clean output");

        {
            miinfer::prefill_v2::PrefillV2Model boundary_engine(model, 2048, true);
            for (const std::size_t length : {511U, 512U, 513U, 1024U, 1025U}) {
                std::vector<std::uint32_t> boundary_prompt(length);
                for (std::size_t i = 0; i < length; ++i) {
                    boundary_prompt[i] = static_cast<std::uint32_t>((i * 31 + length) % 1000 + 1);
                }
                const auto boundary_cold = boundary_engine.generate(boundary_prompt, options(true, true));
                const auto boundary_hit = boundary_engine.generate(boundary_prompt, options(true, true));
                require(!boundary_cold.reuse_hit, "boundary cold run unexpectedly reused state");
                require(boundary_hit.reuse_hit && boundary_hit.prefix_tokens_reused == length
                            && boundary_hit.suffix_tokens_executed == 0,
                        "boundary exact-prefix hit failed");
                require_same_tokens(boundary_cold, boundary_hit, "boundary reuse output differs from cold output");
            }
        }

        std::cout << "M30-0001 exact-prefix reuse: PASS"
                  << " zero_suffix_twice=PASS"
                  << " suffix=PASS"
                  << " boundary_prefixes=PASS"
                  << " mismatch_fallback=PASS"
                  << " fresh_session=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M30-0001 exact-prefix reuse: FAIL: " << error.what() << '\n';
        return 1;
    }
}
