#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Model = miinfer::prefill_v2::PrefillV2Model;
using Options = miinfer::prefill_v2::GenerateOptions;

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<std::uint32_t> prompt(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint32_t> result(count);
    std::uint32_t value = seed;
    for (auto& token : result) {
        value = value * 1664525u + 1013904223u;
        token = value % 151643u + 1;
    }
    return result;
}

Options options(std::size_t generated, bool reuse, bool reset) {
    Options result;
    result.max_new_tokens = generated;
    result.reset_state_before = reset;
    result.use_hip_graph = true;
    result.enable_prefix_reuse = reuse;
    result.temperature = 0.0f;
    result.top_p = 1.0f;
    result.top_k = 1;
    result.stop_token_ids.clear();
    result.seed = 7;
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void report_mismatch(const std::vector<std::uint32_t>& expected,
                     const std::vector<std::uint32_t>& actual) {
    std::size_t index = 0;
    while (index < expected.size() && index < actual.size() && expected[index] == actual[index]) {
        ++index;
    }
    std::cerr << "prefix_diagnostic expected_size=" << expected.size()
              << " actual_size=" << actual.size() << " first_mismatch=" << index;
    if (index < expected.size()) std::cerr << " expected=" << expected[index];
    if (index < actual.size()) std::cerr << " actual=" << actual[index];
    std::cerr << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-m31-0002-persistent-context-qualification MODEL.gguf\n";
        return 2;
    }

    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto prefix = prompt(8192, 77);
        const auto suffix = prompt(512, 177);
        auto full = prefix;
        full.insert(full.end(), suffix.begin(), suffix.end());
        auto branch_a = prefix;
        branch_a.push_back(314);
        auto branch_b = prefix;
        branch_b.push_back(2718);
        auto nested = branch_a;
        nested.push_back(99);

        Model engine(model, 16384, true);
        const auto cold = engine.generate(full, options(32, false, true));
        const auto warm = engine.generate(full, options(32, false, true));
        require(cold.generated_tokens == warm.generated_tokens, "cold/warm output mismatch");

        const auto cold_branch_a = engine.generate(branch_a, options(32, false, true));
        const auto cold_nested = engine.generate(nested, options(32, false, true));
        const auto cold_branch_b = engine.generate(branch_b, options(32, false, true));

        const auto prefix_stats = engine.generate(prefix, [&] {
            auto result = options(0, false, true);
            result.cache_prefix_after = true;
            result.cache_prefix_len = prefix.size();
            return result;
        }());
        (void)prefix_stats;
        const auto reuse_start = now_ms();
        const auto reuse = engine.generate(full, options(32, true, false));
        const double reuse_wall_ms = now_ms() - reuse_start;
        require(reuse.reuse_hit && reuse.prefix_tokens_reused == prefix.size(),
                "prefix reuse did not hit the canonical prefix");
        if (reuse.generated_tokens != cold.generated_tokens) {
            report_mismatch(cold.generated_tokens, reuse.generated_tokens);
            require(false, "prefix output mismatch");
        }

        engine.reset_state();
        require(!engine.restore_snapshot(Model::kInvalidSnapshotId),
                "invalid snapshot was accepted");
        const auto base = engine.generate(prefix, options(0, false, true));
        (void)base;
        const auto base_snapshot = engine.snapshot(prefix);
        require(base_snapshot != Model::kInvalidSnapshotId, "base snapshot failed");

        const double fork_start = now_ms();
        const auto sibling = engine.fork(base_snapshot);
        const double fork_ms = now_ms() - fork_start;
        require(sibling != Model::kInvalidSnapshotId, "sibling fork failed");

        const double restore_start = now_ms();
        require(engine.restore_snapshot(sibling), "sibling restore failed");
        const double restore_ms = now_ms() - restore_start;
        const auto branch_stats = engine.generate(branch_a, options(0, true, false));
        require(branch_stats.reuse_hit && branch_stats.prefix_tokens_reused == prefix.size(),
                "sibling continuation failed");
        const auto branch_snapshot = engine.snapshot(branch_a);
        require(branch_snapshot != Model::kInvalidSnapshotId, "branch snapshot failed");
        const auto nested_fork = engine.fork(branch_snapshot);
        require(nested_fork != Model::kInvalidSnapshotId, "nested fork failed");
        require(engine.restore_snapshot(nested_fork), "nested restore failed");
        const auto nested_stats = engine.generate(nested, options(32, true, false));
        require(nested_stats.reuse_hit && nested_stats.generated_tokens == cold_nested.generated_tokens,
                "nested COW continuation or output parity failed");
        const auto telemetry = engine.snapshot_telemetry();
        require(telemetry.cow_events > 0 && telemetry.private_branch_bytes > 0,
                "COW mutation was not observed");

        require(engine.restore_snapshot(branch_snapshot), "branch restore failed");
        const auto branch_resume = engine.generate(branch_a, options(32, true, false));
        require(branch_resume.reuse_hit && branch_resume.generated_tokens == cold_branch_a.generated_tokens,
                "sibling output parity failed");

        require(engine.rollback_snapshot(base_snapshot), "canonical rollback failed");
        const auto resumed = engine.generate(branch_b, options(32, true, false));
        require(resumed.reuse_hit && resumed.generated_tokens == cold_branch_b.generated_tokens,
                "canonical branch resume or output parity failed");

        const auto checkpoint_bytes = engine.snapshot_bytes();
        const auto peak_vram = engine.total_vram_bytes();
        const auto replayed_tokens = full.size();
        const auto avoided_tokens = reuse.prefix_tokens_reused;
        require(engine.release_snapshot(nested_fork), "nested release failed");
        require(engine.release_snapshot(branch_snapshot), "branch release failed");
        require(engine.release_snapshot(sibling), "sibling release failed");
        require(engine.release_snapshot(base_snapshot), "base release failed");
        require(!engine.restore_snapshot(base_snapshot), "released snapshot was accepted");
        require(!engine.release_snapshot(base_snapshot), "released snapshot was released twice");
        require(engine.snapshot_count() == 0, "snapshot cleanup failed");

        std::cout << std::fixed << std::setprecision(3)
                  << "M31-0002 persistent-context qualification: PASS\n"
                  << "cold_ttft_ms=" << cold.ttft_ms
                  << " warm_ttft_ms=" << warm.ttft_ms
                  << " reuse_ttft_ms=" << reuse.ttft_ms
                  << " reuse_wall_ms=" << reuse_wall_ms
                  << " restore_ms=" << restore_ms
                  << " fork_ms=" << fork_ms
                  << " replayed_tokens=" << replayed_tokens
                  << " avoided_tokens=" << avoided_tokens
                  << " checkpoint_bytes=" << checkpoint_bytes
                  << " peak_vram_bytes=" << peak_vram
                  << " shared_bytes=" << telemetry.physical_shared_bytes
                  << " private_bytes=" << telemetry.private_branch_bytes
                  << " references=" << telemetry.reference_count
                  << " cow_events=" << telemetry.cow_events
                  << " cow_bytes_copied=" << telemetry.cow_bytes_copied
                  << " output_parity=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M31-0002 persistent-context qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
