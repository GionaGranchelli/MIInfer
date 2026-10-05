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

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double now_ms() {
    timespec value{};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<double>(value.tv_sec) * 1000.0
        + static_cast<double>(value.tv_nsec) / 1000000.0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: miinfer-prefill-v2-agent-branch-workload-test MODEL.gguf\n";
        return 2;
    }

    try {
        auto model = miinfer::Qwen35Model::load(argv[1]);
        const std::vector<std::uint32_t> base = {14990, 8, 271, 912, 42, 73, 101, 7};
        std::vector<std::vector<std::uint32_t>> branches;
        const std::vector<std::uint32_t> branch_seeds = {314, 2718, 99, 314, 2718,
                                                          99, 314, 2718, 99, 314};
        for (std::uint32_t operation = 0; operation < 10; ++operation) {
            auto branch = base;
            for (std::uint32_t suffix = 0; suffix <= operation % 4; ++suffix) {
                branch.push_back(branch_seeds[operation] + suffix);
            }
            branches.push_back(std::move(branch));
        }

        std::vector<std::vector<std::uint32_t>> candidate_tokens;
        double candidate_wall_ms = 0.0;
        std::size_t candidate_physical_tokens = 0;
        std::size_t snapshot_bytes = 0;
        std::size_t total_vram = 0;
        {
            Model candidate(model, 1024, true);
            auto prefill_only = options(false, 0);
            (void)candidate.generate(base, prefill_only);
            const auto snapshot = candidate.snapshot(base);
            require(snapshot != Model::kInvalidSnapshotId, "base snapshot failed");
            snapshot_bytes = candidate.snapshot_bytes();
            total_vram = candidate.total_vram_bytes();

            for (const auto& branch : branches) {
                require(candidate.restore_snapshot(snapshot), "branch restore failed");
                const auto start = now_ms();
                const auto stats = candidate.generate(branch, options(true, 1));
                candidate_wall_ms += now_ms() - start;
                require(stats.reuse_hit && stats.prefix_tokens_reused == base.size(),
                        "candidate did not reuse base prefix");
                candidate_physical_tokens += stats.suffix_tokens_executed;
                candidate_tokens.push_back(stats.generated_tokens);
            }
        }

        std::vector<std::vector<std::uint32_t>> cold_tokens;
        double cold_wall_ms = 0.0;
        std::size_t cold_physical_tokens = 0;
        {
            Model cold(model, 1024, true);
            for (const auto& branch : branches) {
                const auto start = now_ms();
                const auto stats = cold.generate(branch, options(false, 1));
                cold_wall_ms += now_ms() - start;
                cold_physical_tokens += branch.size();
                cold_tokens.push_back(stats.generated_tokens);
            }
        }

        for (std::size_t i = 0; i < candidate_tokens.size(); ++i) {
            if (candidate_tokens[i] != cold_tokens[i]) {
                std::cerr << "candidate/cold mismatch operation=" << i
                          << " candidate_tokens=" << candidate_tokens[i].size()
                          << " cold_tokens=" << cold_tokens[i].size();
                if (!candidate_tokens[i].empty() && !cold_tokens[i].empty()) {
                    std::cerr << " candidate_first=" << candidate_tokens[i].front()
                              << " cold_first=" << cold_tokens[i].front();
                }
                std::cerr << '\n';
                throw std::runtime_error("candidate/cold branch outputs differ");
            }
        }
        const auto avoided = cold_physical_tokens - candidate_physical_tokens;
        std::cout << "M30-0004 agent branch workload: PASS operations=" << branches.size()
                  << " replay_physical_tokens=" << cold_physical_tokens
                  << " rollback_physical_tokens=" << candidate_physical_tokens
                  << " tokens_avoided=" << avoided
                  << " candidate_wall_ms=" << candidate_wall_ms
                  << " cold_wall_ms=" << cold_wall_ms
                  << " checkpoint_bytes=" << snapshot_bytes
                  << " total_vram_bytes=" << total_vram << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M30-0004 agent branch workload: FAIL: " << error.what() << '\n';
        return 1;
    }
}
