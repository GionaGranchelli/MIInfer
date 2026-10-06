#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Model = miinfer::prefill_v2::PrefillV2Model;
using Options = miinfer::prefill_v2::GenerateOptions;

namespace {

struct Turn {
    const char* kind;
    std::vector<std::uint32_t> tokens;
};

std::vector<std::uint32_t> event_tokens(std::size_t turn, std::uint32_t salt) {
    std::vector<std::uint32_t> tokens;
    tokens.reserve(256);
    std::uint32_t value = 1000u + salt + static_cast<std::uint32_t>(turn * 17);
    for (std::size_t i = 0; i < 256; ++i) {
        value = value * 1664525u + 1013904223u;
        tokens.push_back(value % 151000u + 1u);
    }
    return tokens;
}

std::vector<Turn> workload() {
    const char* kinds[] = {
        "conversation", "code_generation", "tool_invocation", "tool_result",
        "continuation", "branch", "failed_attempt", "rollback", "alternate_branch",
        "long_tool_result", "continuation", "snapshot", "another_branch",
        "tool_invocation", "tool_result", "continuation", "nested_branch",
        "rollback", "canonical_resume", "final_answer"
    };
    std::vector<Turn> result;
    for (std::size_t i = 0; i < 20; ++i) result.push_back({kinds[i], event_tokens(i + 1, 77)});
    return result;
}

Options options(std::size_t generated, bool reuse, bool reset, bool cache) {
    Options result;
    result.max_new_tokens = generated;
    result.reset_state_before = reset;
    result.enable_prefix_reuse = reuse;
    result.cache_prefix_after = cache;
    result.cache_prefix_len = 0;
    result.use_hip_graph = true;
    result.temperature = 0.0F;
    result.top_p = 1.0F;
    result.top_k = 1;
    result.stop_token_ids.clear();
    result.seed = 7;
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-m31-0003-canonical-agent-workload MODEL.gguf\n";
        return 2;
    }

    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto turns = workload();
        std::vector<std::vector<std::uint32_t>> prompts;
        std::vector<std::vector<std::uint32_t>> cold_outputs;
        std::vector<std::uint32_t> prompt;
        prompt.reserve(6000);
        prompt.push_back(151644); // deterministic conversation marker
        for (const auto& turn : turns) {
            prompt.insert(prompt.end(), turn.tokens.begin(), turn.tokens.end());
            prompts.push_back(prompt);
        }

        Model engine(model, 16384, true);
        double cold_ms = 0.0;
        std::size_t cold_replayed = 0;
        std::vector<std::uint32_t> cold_branch_output;
        for (const auto& current : prompts) {
            const auto start = std::chrono::steady_clock::now();
            const auto stats = engine.generate(current, options(32, false, true, false));
            cold_ms += elapsed_ms(start);
            cold_replayed += current.size();
            cold_outputs.push_back(stats.generated_tokens);
            if (cold_outputs.size() == 6) {
                auto alternate = current;
                alternate.push_back(2718);
                const auto branch = engine.generate(alternate, options(32, false, true, false));
                cold_branch_output = branch.generated_tokens;
            }
        }

        engine.reset_state();
        double persistent_ms = 0.0;
        std::size_t persistent_replayed = 0;
        std::size_t avoided = 0;
        bool parity = true;
        std::vector<Model::SnapshotId> snapshots;
        std::size_t branch_events = 0;

        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const auto& current = prompts[i];
            auto run_options = options(32, i != 0, i == 0, true);
            const auto start = std::chrono::steady_clock::now();
            const auto stats = engine.generate(current, run_options);
            persistent_ms += elapsed_ms(start);
            persistent_replayed += stats.prefix_tokens_replayed;
            avoided += stats.prefix_tokens_reused;
            parity = parity && stats.generated_tokens == cold_outputs[i];

            const auto snapshot = engine.snapshot(current);
            require(snapshot != Model::kInvalidSnapshotId, "canonical snapshot failed");
            if (i == 5 || i == 11 || i == 16) {
                snapshots.push_back(snapshot);
            } else {
                require(engine.release_snapshot(snapshot), "short-lived snapshot release failed");
            }

            if (i == 5) {
                const auto sibling = engine.fork(snapshot);
                require(sibling != Model::kInvalidSnapshotId, "agent sibling fork failed");
                require(engine.restore_snapshot(sibling), "agent sibling restore failed");
                auto alternate = current;
                alternate.push_back(2718);
                const auto branch = engine.generate(alternate, options(32, true, false, false));
                parity = parity && branch.generated_tokens == cold_branch_output;
                ++branch_events;
                require(engine.rollback_snapshot(snapshot), "agent rollback failed");
                require(engine.release_snapshot(sibling), "agent sibling release failed");
            }
        }

        const auto telemetry = engine.snapshot_telemetry();
        const auto checkpoint_bytes = engine.snapshot_bytes();
        const auto peak_vram = engine.total_vram_bytes();
        for (const auto id : snapshots) require(engine.release_snapshot(id), "agent snapshot release failed");
        require(engine.snapshot_count() == 0, "agent snapshot leak");

        const auto replay_baseline = cold_replayed == 0 ? 1.0 : static_cast<double>(cold_replayed);
        const double speedup = persistent_ms > 0.0 ? cold_ms / persistent_ms : 0.0;
        const double replay_avoided = 1.0 - static_cast<double>(persistent_replayed) / replay_baseline;
        std::cout << std::fixed << std::setprecision(3)
                  << "M31-0003 canonical-agent workload: " << (parity ? "PASS" : "FAIL") << '\n'
                  << "turns=" << turns.size()
                  << " cold_replay_wall_ms=" << cold_ms
                  << " persistent_runtime_wall_ms=" << persistent_ms
                  << " agent_runtime_speedup=" << speedup
                  << " baseline_replayed_tokens=" << cold_replayed
                  << " replayed_tokens=" << persistent_replayed
                  << " replay_avoided=" << replay_avoided
                  << " branch_events=" << branch_events
                  << " avoided_tokens=" << avoided
                  << " checkpoint_bytes=" << checkpoint_bytes
                  << " peak_vram_bytes=" << peak_vram
                  << " cow_shared_bytes=" << telemetry.physical_shared_bytes
                  << " cow_private_bytes=" << telemetry.private_branch_bytes
                  << " cow_events=" << telemetry.cow_events
                  << " cow_bytes_copied=" << telemetry.cow_bytes_copied
                  << " output_parity=" << (parity ? "PASS" : "FAIL") << '\n';
        return parity ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "M31-0003 canonical-agent workload: FAIL: " << error.what() << '\n';
        return 1;
    }
}
