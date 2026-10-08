#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen35_model.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using Model = miinfer::prefill_v2::PrefillV2Model;

miinfer::prefill_v2::GenerateOptions options() {
    miinfer::prefill_v2::GenerateOptions result;
    result.max_new_tokens = 1;
    result.reset_state_before = true;
    result.use_hip_graph = false;
    result.enable_prefix_reuse = true;
    result.cache_prefix_after = false;
    result.temperature = 0.0F;
    result.top_p = 1.0F;
    result.top_k = 1;
    result.seed = 7;
    result.stop_token_ids.clear();
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-prefill-v2-snapshot-fork-rollback-test MODEL.gguf\n";
        return 2;
    }

    try {
        auto model = miinfer::Qwen35Model::load(argv[1]);
        const std::vector<std::uint32_t> prompt = {14990, 8, 271, 912, 42, 73, 101, 7};
        auto branch_a = prompt;
        branch_a.push_back(314);
        auto branch_b = prompt;
        branch_b.push_back(2718);
        auto branch_a2 = branch_a;
        branch_a2.push_back(99);
        std::vector<std::uint32_t> branch_b_tokens;

        {
            Model engine(model, 1024, true);
            auto prefill_only = options();
            prefill_only.max_new_tokens = 0;
            (void)engine.generate(prompt, prefill_only);
            const auto s0 = engine.snapshot(prompt);
            require(s0 != Model::kInvalidSnapshotId, "snapshot S0 was not created");

            require(engine.restore_snapshot(s0), "snapshot S0 restore failed");
            const auto saved_position = engine.recurrent_storage(0).position();
            engine.recurrent_storage(0).reset();
            require(engine.restore_snapshot(s0), "snapshot restore after recurrent-state mutation failed");
            require(engine.recurrent_storage(0).position() == saved_position,
                    "snapshot restore did not recover recurrent position after mutation");
            require(engine.rollback_snapshot(s0), "snapshot S0 rollback failed");

            const auto fork_b = engine.fork(s0);
            require(fork_b != Model::kInvalidSnapshotId, "fork B was not created");
            const auto fork_c = engine.fork(s0);
            require(fork_c != Model::kInvalidSnapshotId, "fork C was not created");
            const auto shared = engine.snapshot_telemetry();
            require(shared.logical_snapshot_count == 3
                        && shared.physical_shared_bytes > 0
                        && shared.reference_count >= 3,
                    "forks did not share physical snapshot backing");

            require(engine.restore_snapshot(s0), "restore S0 before branch A failed");
            const auto a = engine.generate(branch_a, prefill_only);
            require(a.reuse_hit && a.prefix_tokens_reused == prompt.size()
                        && a.suffix_tokens_executed == 1,
                    "branch A did not continue from S0");

            const auto s1 = engine.snapshot(branch_a);
            require(s1 != Model::kInvalidSnapshotId, "nested snapshot S1 was not created");
            const auto cow = engine.snapshot_telemetry();
            require(cow.cow_events >= 1 && cow.cow_bytes_copied > 0
                        && cow.private_branch_bytes > 0,
                    "divergent snapshot did not create private COW backing");
            require(engine.generate(branch_a2, prefill_only).reuse_hit, "nested continuation failed");
            require(engine.rollback_snapshot(s1), "rollback to nested S1 failed");
            require(engine.generate(branch_a2, prefill_only).reuse_hit,
                    "continuation after nested rollback failed");

            require(engine.restore_snapshot(fork_b), "restore fork B failed");
            const auto b = engine.generate(branch_b, options());
            require(b.reuse_hit && b.prefix_tokens_reused == prompt.size()
                        && b.suffix_tokens_executed == 1,
                    "branch B did not continue independently from S0");
            branch_b_tokens = b.generated_tokens;

            require(engine.rollback_snapshot(s0), "repeated rollback to S0 failed");
            const auto b_again = engine.generate(branch_b, options());
            require(b_again.generated_tokens == branch_b_tokens,
                    "repeated rollback changed branch B output");

            require(!engine.restore_snapshot(999999), "invalid snapshot ID was accepted");
            require(engine.release_snapshot(fork_b), "fork B release failed");
            require(!engine.restore_snapshot(fork_b), "released fork B remained valid");
            require(engine.release_snapshot(fork_c), "fork C release failed");
            require(engine.release_snapshot(s1), "nested snapshot S1 release failed");
            require(engine.release_snapshot(s0), "snapshot S0 release failed");
            require(!engine.restore_snapshot(s0), "released S0 remained valid");

            (void)engine.generate(branch_b, prefill_only);
            std::vector<Model::SnapshotId> eviction_ids;
            for (int i = 0; i < 9; ++i) {
                const auto id = engine.snapshot(branch_b);
                require(id != Model::kInvalidSnapshotId, "snapshot for eviction test failed");
                eviction_ids.push_back(id);
            }
            require(!engine.restore_snapshot(eviction_ids.front()), "old snapshot was not evicted");
            require(engine.restore_snapshot(eviction_ids.back()), "newest snapshot was evicted");
            engine.reset_state();
            require(!engine.restore_snapshot(eviction_ids.back()), "reset retained a snapshot");
        }

        Model fresh_engine(model, 1024, true);
        const auto cold = fresh_engine.generate(branch_b, options());
        require(cold.generated_tokens == branch_b_tokens,
                "snapshot branch output differs from cold replay");
        fresh_engine.clear_snapshots();
        require(!fresh_engine.restore_snapshot(1), "cleared session retained a snapshot");

        std::cout << "M30-0004 snapshot/fork/rollback: PASS"
                  << " snapshot=PASS fork=PASS rollback=PASS nested=PASS"
                  << " invalid=PASS release=PASS shared=PASS cow=PASS"
                  << " evict=PASS reset=PASS cold_parity=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M30-0004 snapshot/fork/rollback: FAIL: " << error.what() << '\n';
        return 1;
    }
}
