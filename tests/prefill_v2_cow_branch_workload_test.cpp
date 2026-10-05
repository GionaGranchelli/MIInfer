#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <time.h>
#include <vector>

namespace {

using Model = miinfer::prefill_v2::PrefillV2Model;

miinfer::prefill_v2::GenerateOptions options(bool reuse, std::size_t max_new_tokens) {
    miinfer::prefill_v2::GenerateOptions result;
    result.max_new_tokens = max_new_tokens;
    result.reset_state_before = true;
    result.use_hip_graph = false;
    result.enable_prefix_reuse = reuse;
    result.temperature = 0.0F;
    result.top_p = 1.0F;
    result.top_k = 1;
    result.seed = 7;
    result.stop_token_ids.clear();
    return result;
}

double now_ms() {
    timespec value{};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<double>(value.tv_sec) * 1000.0
        + static_cast<double>(value.tv_nsec) / 1000000.0;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct RunResult {
    std::vector<std::vector<std::uint32_t>> outputs;
    std::size_t snapshot_count = 0;
    std::size_t checkpoint_bytes = 0;
    std::size_t peak_vram = 0;
    std::size_t physical_tokens = 0;
    double wall_ms = 0.0;
    double restore_ms = 0.0;
    std::size_t shared_bytes = 0;
    std::size_t private_bytes = 0;
    std::size_t references = 0;
    std::size_t cow_events = 0;
    std::size_t cow_bytes = 0;
};

RunResult run_snapshot_workload(const miinfer::Qwen35Model& model) {
    const std::vector<std::uint32_t> base = {14990, 8, 271, 912, 42, 73, 101, 7};
    const std::vector<std::vector<std::uint32_t>> branches = {
        {14990, 8, 271, 912, 42, 73, 101, 7, 314},
        {14990, 8, 271, 912, 42, 73, 101, 7, 2718},
        {14990, 8, 271, 912, 42, 73, 101, 7, 99},
        {14990, 8, 271, 912, 42, 73, 101, 7, 314, 2718},
    };
    const auto branch_a2 = std::vector<std::uint32_t>{
        14990, 8, 271, 912, 42, 73, 101, 7, 314, 99};
    const auto branch_a3 = std::vector<std::uint32_t>{
        14990, 8, 271, 912, 42, 73, 101, 7, 314, 99, 123};
    Model engine(model, 1024, true);
    const auto prefill_only = options(false, 0);
    (void)engine.generate(base, prefill_only);
    const auto base_id = engine.snapshot(base);
    require(base_id != Model::kInvalidSnapshotId, "base snapshot failed");
    const auto continue_options = options(true, 0);

    std::vector<Model::SnapshotId> fork_ids;
    for (int i = 0; i < 4; ++i) {
        const auto id = engine.fork(base_id);
        require(id != Model::kInvalidSnapshotId, "sibling fork failed");
        fork_ids.push_back(id);
    }

    RunResult result;
    for (std::size_t i = 0; i < branches.size(); ++i) {
        const auto restore_start = now_ms();
        require(engine.restore_snapshot(fork_ids[i]), "sibling restore failed");
        result.restore_ms += now_ms() - restore_start;
        const auto start = now_ms();
        const auto stats = engine.generate(branches[i], options(true, 1));
        result.wall_ms += now_ms() - start;
        result.physical_tokens += stats.suffix_tokens_executed;
        result.outputs.push_back(stats.generated_tokens);
        require(stats.reuse_hit && stats.prefix_tokens_reused == base.size(),
                "sibling did not reuse shared prefix");
    }

    const auto branch_restore_start = now_ms();
    require(engine.restore_snapshot(fork_ids[0]), "branch A restore failed");
    result.restore_ms += now_ms() - branch_restore_start;
    (void)engine.generate(branches[0], continue_options);
    const auto branch_a_id = engine.snapshot(branches[0]);
    require(branch_a_id != Model::kInvalidSnapshotId, "branch A snapshot failed");
    const auto nested_id = engine.fork(branch_a_id);
    require(nested_id != Model::kInvalidSnapshotId, "nested fork failed");
    const auto nested_restore_start = now_ms();
    require(engine.restore_snapshot(nested_id), "nested restore failed");
    result.restore_ms += now_ms() - nested_restore_start;
    (void)engine.generate(branch_a2, continue_options);
    const auto nested_state_id = engine.snapshot(branch_a2);
    require(nested_state_id != Model::kInvalidSnapshotId, "nested state snapshot failed");
    const auto nested_state_restore_start = now_ms();
    require(engine.restore_snapshot(nested_state_id), "nested state restore failed");
    result.restore_ms += now_ms() - nested_state_restore_start;
    const auto nested_start = now_ms();
    const auto nested = engine.generate(branch_a3, options(true, 1));
    result.wall_ms += now_ms() - nested_start;
    result.physical_tokens += nested.suffix_tokens_executed;
    result.outputs.push_back(nested.generated_tokens);
    require(nested.reuse_hit && nested.prefix_tokens_reused == branch_a2.size(),
            "nested branch did not use COW suffix");

    result.snapshot_count = engine.snapshot_count();
    result.checkpoint_bytes = engine.snapshot_bytes();
    result.peak_vram = engine.total_vram_bytes();
#if MIINFER_M30_0005_COW
    const auto telemetry = engine.snapshot_telemetry();
    result.shared_bytes = telemetry.physical_shared_bytes;
    result.private_bytes = telemetry.private_branch_bytes;
    result.references = telemetry.reference_count;
    result.cow_events = telemetry.cow_events;
    result.cow_bytes = telemetry.cow_bytes_copied;
    if (!(result.snapshot_count == 8 && result.shared_bytes > 0
                && result.private_bytes > 0 && result.references >= 8
                && result.cow_events >= 2)) {
        std::cerr << "COW accounting snapshot_count=" << result.snapshot_count
                  << " shared_bytes=" << result.shared_bytes
                  << " private_bytes=" << result.private_bytes
                  << " references=" << result.references
                  << " cow_events=" << result.cow_events << '\n';
        throw std::runtime_error("COW accounting did not reflect sibling and nested sharing");
    }
#else
    result.private_bytes = result.checkpoint_bytes;
#endif

    for (auto it = fork_ids.rbegin(); it != fork_ids.rend(); ++it) {
        require(engine.release_snapshot(*it), "sibling release failed");
    }
    require(engine.release_snapshot(nested_state_id), "nested state release failed");
    require(engine.release_snapshot(nested_id), "nested release failed");
    require(engine.release_snapshot(branch_a_id), "branch A release failed");
    require(engine.release_snapshot(base_id), "base release failed");
    require(engine.snapshot_count() == 0 && engine.snapshot_bytes() == 0,
            "COW cleanup left physical backing");
#if MIINFER_M30_0005_COW
    require(engine.snapshot_telemetry().reference_count == 0,
            "COW cleanup left backing references");
#endif
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-prefill-v2-cow-branch-workload-test MODEL.gguf\n";
        return 2;
    }
    try {
        auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto candidate = run_snapshot_workload(model);

        const std::vector<std::vector<std::uint32_t>> branches = {
            {14990, 8, 271, 912, 42, 73, 101, 7, 314},
            {14990, 8, 271, 912, 42, 73, 101, 7, 2718},
            {14990, 8, 271, 912, 42, 73, 101, 7, 99},
            {14990, 8, 271, 912, 42, 73, 101, 7, 314, 2718},
            {14990, 8, 271, 912, 42, 73, 101, 7, 314, 99, 123},
        };
        std::vector<std::vector<std::uint32_t>> cold_outputs;
        std::size_t cold_tokens = 0;
        double cold_wall_ms = 0.0;
        {
            Model cold(model, 1024, true);
            for (const auto& branch : branches) {
                const auto start = now_ms();
                const auto stats = cold.generate(branch, options(false, 1));
                cold_wall_ms += now_ms() - start;
                cold_tokens += branch.size();
                cold_outputs.push_back(stats.generated_tokens);
            }
        }
        require(candidate.outputs.size() == cold_outputs.size(), "output count mismatch");
        for (std::size_t i = 0; i < candidate.outputs.size(); ++i) {
            require(candidate.outputs[i] == cold_outputs[i], "COW/cold output mismatch");
        }

#if MIINFER_M30_0005_COW
        std::cout << "M30-0005 COW workload: PASS"
                  << " mode=COW operations=8"
                  << " snapshot_count=" << candidate.snapshot_count
                  << " checkpoint_bytes=" << candidate.checkpoint_bytes
                  << " shared_bytes=" << candidate.shared_bytes
                  << " private_bytes=" << candidate.private_bytes
                  << " references=" << candidate.references
                  << " cow_events=" << candidate.cow_events
                  << " cow_bytes_copied=" << candidate.cow_bytes
                  << " peak_vram=" << candidate.peak_vram
                  << " restore_ms=" << candidate.restore_ms
                  << " branch_wall_ms=" << candidate.wall_ms
                  << " cold_wall_ms=" << cold_wall_ms
                  << " physical_tokens=" << candidate.physical_tokens
                  << " cold_tokens=" << cold_tokens << '\n';
#else
        std::cout << "M30-0005 COW workload: PASS"
                  << " mode=FULL_COPY operations=8"
                  << " snapshot_count=" << candidate.snapshot_count
                  << " checkpoint_bytes=" << candidate.checkpoint_bytes
                  << " private_bytes=" << candidate.checkpoint_bytes
                  << " peak_vram=" << candidate.peak_vram
                  << " restore_ms=" << candidate.restore_ms
                  << " branch_wall_ms=" << candidate.wall_ms
                  << " cold_wall_ms=" << cold_wall_ms
                  << " physical_tokens=" << candidate.physical_tokens
                  << " cold_tokens=" << cold_tokens << '\n';
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M30-0005 COW workload: FAIL: " << error.what() << '\n';
        return 1;
    }
}
