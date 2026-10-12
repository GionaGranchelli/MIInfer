#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/logit_validation.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using miinfer::prefill_v2::GenerateOptions;
using miinfer::prefill_v2::PrefillV2Model;

constexpr std::size_t kTurns = 8;
constexpr std::size_t kContextTokens = 65536;
constexpr std::size_t kTurnAppendTokens = 16;
constexpr std::size_t kRestoreTurn = 4;
constexpr std::uint32_t kKvCapacity = 65792;
constexpr std::size_t kMinimumFreeVramBytes = 1ull << 30;

std::size_t generation_budget(std::size_t turn) {
    return turn + 1 == kRestoreTurn ? 1 : 2;
}

std::vector<std::uint32_t> event_tokens(std::size_t count, std::size_t turn,
                                        std::uint32_t vocab_size) {
    std::vector<std::uint32_t> tokens;
    tokens.reserve(count);
    std::uint32_t state = static_cast<std::uint32_t>(0x9e3779b9u ^ (turn * 7919u));
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        tokens.push_back(state % (vocab_size - 1) + 1);
    }
    return tokens;
}

void require(bool condition, const std::string& message) {
    assert(condition);
    if (!condition) throw std::runtime_error(message);
}

void require_finite_logits(PrefillV2Model& model) {
    std::vector<float> logits(model.vocab_size());
    MIINFER_HIP_CHECK(hipMemcpy(logits.data(), model.logits_buffer(),
                                logits.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (const float value : logits) miinfer::prefill_v2::detail::require_finite_logit(value);
}

void require_positions(const PrefillV2Model& model, std::uint32_t expected) {
    require(model.recurrent_states().size() == 48, "expected 48 recurrent layers");
    for (std::size_t i = 0; i < model.recurrent_states().size(); ++i) {
        require(model.recurrent_states()[i].position() == expected,
                "recurrent position mismatch at layer " + std::to_string(i));
    }
}

std::uint64_t state_fingerprint(PrefillV2Model& model, std::size_t kv_tokens) {
    std::uint64_t hash = 14695981039346656037ull;
    const auto add = [&hash](const void* data, std::size_t bytes) {
        const auto* input = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < bytes; ++i) {
            hash = (hash ^ input[i]) * 1099511628211ull;
        }
    };
    for (const auto& layer : model.recurrent_states()) {
        std::vector<float> state;
        std::vector<float> history;
        layer.download(state, history);
        add(state.data(), state.size() * sizeof(float));
        add(history.data(), history.size() * sizeof(float));
    }
    const auto kv_bytes = model.kv_storage(0).raw_tokens_bytes(kv_tokens);
    std::vector<std::uint8_t> kv(kv_bytes);
    for (const auto& layer : model.kv_caches()) {
        layer.download_raw(kv.data(), kv_tokens);
        add(kv.data(), kv.size());
    }
    return hash;
}

bool observe_token(std::uint32_t token, std::uint32_t& previous, std::size_t& run) {
    run = (run == 0 || token != previous) ? 1 : run + 1;
    previous = token;
    return run >= 4;
}

void self_check() {
    miinfer::prefill_v2::detail::require_finite_logit(0.0f);
    for (const float value : {NAN, INFINITY, -INFINITY}) {
        bool rejected = false;
        try {
            miinfer::prefill_v2::detail::require_finite_logit(value);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "non-finite logit self-check unexpectedly passed");
    }
    static_assert(kTurns >= 8);
    static_assert(kKvCapacity > kContextTokens + (kTurns - 1) * kTurnAppendTokens);
    std::size_t prompt_size = kContextTokens;
    require(prompt_size == 65536, "initial prompt must start at exactly 65536 tokens");
    std::size_t generated = 0;
    for (std::size_t turn = 0; turn < kTurns; ++turn) {
        const auto budget = generation_budget(turn);
        if (turn != 0) prompt_size += generated + (kTurnAppendTokens - generated);
        require(prompt_size == kContextTokens + turn * kTurnAppendTokens,
                "turn does not append exactly 16 prompt tokens");
        generated = budget;
    }
    require(prompt_size == kContextTokens + (kTurns - 1) * kTurnAppendTokens,
            "unexpected final prompt length");
    require(generation_budget(kRestoreTurn - 1) == 1,
            "session-save turn must not decode past its checkpoint");
    require(kKvCapacity > kContextTokens, "KV capacity must leave room for bounded generation");

    constexpr std::array<std::uint32_t, 5> repeated{7, 7, 7, 7, 7};
    std::uint32_t previous = 0;
    std::size_t run = 0;
    bool detected = false;
    for (const auto token : repeated) detected |= observe_token(token, previous, run);
    require(detected, "repetition-loop self-check did not detect a repeated-token run");
}

GenerateOptions options(std::size_t max_new_tokens, bool reuse, bool reset) {
    GenerateOptions value;
    value.max_new_tokens = max_new_tokens;
    value.reset_state_before = reset;
    value.use_hip_graph = true;
    value.enable_prefix_reuse = reuse;
    value.cache_prefix_after = true;
    value.temperature = 0.0f;
    value.top_p = 1.0f;
    value.top_k = 1;
    value.stop_token_ids.clear();
    value.repetition_penalty = 1.0f;
    value.presence_penalty = 0.0f;
    value.frequency_penalty = 0.0f;
    value.seed = 7;
    return value;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-check") {
            self_check();
            std::cout << "64K workload self-check: PASS (8 turns, 65536 prompt tokens)\n";
            return 0;
        }
        if (argc != 3) {
            std::cerr << "usage: miinfer-m31-accel-0001-agent64k MODEL.gguf SESSION_FILE\n"
                         "       miinfer-m31-accel-0001-agent64k --self-check\n";
            return 2;
        }
        self_check();

        const auto qwen = miinfer::Qwen35Model::load(argv[1]);
        PrefillV2Model model(qwen, kKvCapacity, true);
        std::size_t free_vram_bytes = 0;
        std::size_t total_vram_bytes = 0;
        MIINFER_HIP_CHECK(hipMemGetInfo(&free_vram_bytes, &total_vram_bytes));
        require(free_vram_bytes >= kMinimumFreeVramBytes,
                "less than 1 GiB free VRAM after model allocation");
        std::size_t peak_sampled_vram_bytes = total_vram_bytes - free_vram_bytes;
        std::cout << "post_allocation_free_vram_bytes=" << free_vram_bytes
                  << " total_vram_bytes=" << total_vram_bytes << std::endl;
        std::vector<std::uint32_t> prompt;
        prompt.reserve(kContextTokens + (kTurns - 1) * kTurnAppendTokens);
        std::vector<std::uint32_t> previous_output;
        std::size_t repeated_run = 0;
        std::uint32_t previous_token = 0;

        for (std::size_t turn = 0; turn < kTurns; ++turn) {
            const auto previous_prompt_size = prompt.size();
            if (turn == 0) {
                prompt = event_tokens(kContextTokens, turn, model.vocab_size());
            } else {
                const auto added = event_tokens(kTurnAppendTokens - previous_output.size(), turn,
                                                model.vocab_size());
                prompt.insert(prompt.end(), previous_output.begin(), previous_output.end());
                prompt.insert(prompt.end(), added.begin(), added.end());
            }
            require(prompt.size() == kContextTokens + turn * kTurnAppendTokens,
                    "unexpected prompt length at turn " + std::to_string(turn + 1));

            const auto stats = model.generate(prompt,
                options(generation_budget(turn), turn != 0, turn == 0));
            require(stats.reuse_hit == (turn != 0), "unexpected prefix-reuse result");
            if (turn != 0) {
                require(stats.prefix_tokens_reused == previous_prompt_size,
                        "prefix reuse did not start at the previous turn boundary");
                require(stats.suffix_tokens_dispatched == kTurnAppendTokens,
                        "turn suffix was not exactly 16 tokens");
            }
            require(stats.generated_tokens.size() == generation_budget(turn),
                    "bounded generation returned an unexpected token count");
            for (const auto token : stats.generated_tokens) {
                require(token < model.vocab_size(), "generated token ID is outside the vocabulary");
                require(!observe_token(token, previous_token, repeated_run),
                        "four-token repetition loop detected");
            }
            const auto expected_position = static_cast<std::uint32_t>(
                prompt.size() + stats.generated_tokens.size() - 1);
            require_positions(model, expected_position);
            require_finite_logits(model);
            MIINFER_HIP_CHECK(hipMemGetInfo(&free_vram_bytes, &total_vram_bytes));
            const auto sampled_vram_bytes = total_vram_bytes - free_vram_bytes;
            peak_sampled_vram_bytes = std::max(peak_sampled_vram_bytes, sampled_vram_bytes);
            std::cout << "turn=" << (turn + 1)
                      << " prompt_tokens=" << prompt.size()
                      << " reuse_hit=" << (stats.reuse_hit ? "true" : "false")
                      << " reused_tokens=" << stats.prefix_tokens_reused
                      << " suffix_tokens=" << stats.suffix_tokens_dispatched
                      << " generated_tokens=" << stats.generated_tokens.size()
                      << " gdn_position=" << expected_position
                      << " ttft_ms=" << stats.ttft_ms
                      << " total_ms=" << stats.total_ms
                      << " decode_tok_per_sec=" << stats.decode_tok_per_sec
                      << " restore_ms=" << stats.restore_ms
                      << " suffix_prefill_ms=" << stats.suffix_prefill_ms
                      << " sampled_vram_used_bytes=" << sampled_vram_bytes
                      << " sampled_vram_free_bytes=" << free_vram_bytes << std::endl;

            if (turn + 1 == kRestoreTurn) {
                const auto state_before_restore = state_fingerprint(model, prompt.size());
                model.save_session(argv[2], prompt);
                model.reset_state();
                std::vector<std::uint32_t> restored;
                require(model.load_session(argv[2], restored), "persistent session restore failed");
                require(restored == prompt, "restored token history differs from saved history");
                require_positions(model, static_cast<std::uint32_t>(prompt.size()));
                require(state_fingerprint(model, restored.size()) == state_before_restore,
                        "GDN/KV state changed during persistent-session restore");
                std::cout << "session_restore=PASS prefix_tokens=" << restored.size()
                          << " gdn_kv_state=BYTE_IDENTICAL" << std::endl;
            }
            previous_output = stats.generated_tokens;
        }

        require(prompt.size() == kContextTokens + (kTurns - 1) * kTurnAppendTokens,
                "final prompt length mismatch");
        std::cout << "M31-ACCEL-0001 64K persistent multi-turn workload: PASS\n"
                  << "turns=" << kTurns << " initial_context_tokens=" << kContextTokens
                  << " final_context_tokens=" << prompt.size()
                  << " kv_capacity=" << kKvCapacity
                  << " cached_state_bytes=" << model.cached_state_bytes()
                  << " model_accounted_vram_bytes=" << model.total_vram_bytes()
                  << " peak_sampled_hip_vram_used_bytes=" << peak_sampled_vram_bytes << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "M31-ACCEL-0001 64K persistent multi-turn workload: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}
