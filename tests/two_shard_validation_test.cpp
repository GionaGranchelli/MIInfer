#include "miinfer/device_kv_pool.hpp"
#include "miinfer/device_kv_shard.hpp"
#include "miinfer/device_validation.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void throws(auto&& function) {
    bool caught = false;
    try { function(); } catch (const std::exception&) { caught = true; }
    require(caught, "expected exception");
}
}

int main() {
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(0, device, error)) {
        std::cerr << "two-shard validation unavailable: " << error << '\n';
        return 1;
    }
    require(device.architecture.rfind("gfx906", 0) == 0, "selected device is not gfx906");
    require(device.total_vram_bytes >= 30ULL * 1024 * 1024 * 1024, "selected device is not 32 GiB class");
    std::size_t free_before = 0;
    std::size_t total = 0;
    MIINFER_HIP_CHECK(hipMemGetInfo(&free_before, &total));

    constexpr std::size_t context_tokens = 131072;
    constexpr std::size_t page_tokens = 4096;
    constexpr std::size_t page_count = context_tokens / page_tokens;
    miinfer::ContextSpace context(context_tokens, page_tokens);
    context.append(context_tokens);
    std::vector<miinfer::LogicalPageId> ids;
    ids.reserve(page_count);
    for (std::size_t page = 0; page < page_count; ++page) ids.push_back(context.page_id_at(page * page_tokens));
    miinfer::DeviceKvPool pool(device.index, 2 * page_count * 8192);

    miinfer::PlacementPlan one(context);
    std::vector<miinfer::DeviceKvBlock> one_blocks;
    one_blocks.reserve(page_count);
    for (std::size_t page = 0; page < page_count; ++page) {
        const auto block = pool.allocate(8192, 256);
        one_blocks.push_back(block);
        one.place({device.index, ids[page], {page * page_tokens, page_tokens},
                   {block.data, 0, block.bytes}, {0, 4}});
    }
    require(one.resolve({0, context_tokens}, {0, 4}).size() == page_count,
            "one-shard 128K coverage");
    const auto one_committed = pool.committed_bytes();
    require(one_committed == page_count * 8192, "one-shard accounting");
    for (std::size_t page = 0; page < page_count; ++page) {
        require(one.clear_page(ids[page]) == 1, "clear one-shard page");
        pool.release(one_blocks[page]);
    }
    require(pool.committed_bytes() == 0, "one-shard release");

    miinfer::PlacementPlan two(context);
    std::vector<miinfer::DeviceKvBlock> two_blocks;
    two_blocks.reserve(page_count * 2);
    for (std::size_t page = 0; page < page_count; ++page) {
        const auto a = pool.allocate(4096, 256);
        const auto b = pool.allocate(4096, 256);
        two_blocks.push_back(a);
        two_blocks.push_back(b);
        const miinfer::LogicalRange logical{page * page_tokens, page_tokens};
        two.place({device.index, ids[page], logical, {a.data, 0, a.bytes}, {0, 2}});
        two.place({device.index, ids[page], logical, {b.data, 0, b.bytes}, {2, 2}});
        require(a.offset != b.offset, "same-device physical ranges overlap");
    }
    const auto resolved = two.resolve({0, context_tokens}, {0, 4});
    require(resolved.size() == page_count * 2, "two-shard 128K coverage");
    require(pool.committed_bytes() == one_committed, "two-shard payload duplicated");
    std::size_t free_after = 0;
    MIINFER_HIP_CHECK(hipMemGetInfo(&free_after, &total));
    require(free_after > 32ULL * 1024 * 1024, "two-shard safety reserve");
    for (std::size_t page = 0; page < page_count; ++page) {
        require(resolved[page * 2].logical_page() == ids[page]
                    && resolved[page * 2 + 1].logical_page() == ids[page],
                "logical identity changed");
    }

    const auto duplicate = pool.allocate(4096, 256);
    throws([&] { two.place({device.index, ids[0], {0, page_tokens},
                            {duplicate.data, 0, duplicate.bytes}, {1, 1}}); });
    pool.release(duplicate);
    miinfer::PlacementPlan incomplete(context);
    incomplete.place({device.index, ids[0], {0, page_tokens},
                      {two_blocks[0].data, 0, two_blocks[0].bytes}, {0, 2}});
    throws([&] { (void)incomplete.resolve({0, page_tokens}, {0, 4}); });
    const auto stale = pool.allocate(4096, 256);
    pool.release(stale);
    require(!pool.owns(stale), "released allocation still owned");
    throws([&] { pool.release(stale); });
    throws([&] { miinfer::DeviceKvShard(-1, ids[0], {0, page_tokens},
                                         {two_blocks[0].data, 0, two_blocks[0].bytes}, {0, 2}); });

    for (std::size_t page = 0; page < page_count; ++page) {
        require(two.clear_page(ids[page]) == 2, "clear two-shard page");
        pool.release(two_blocks[page * 2]);
        pool.release(two_blocks[page * 2 + 1]);
    }
    require(pool.committed_bytes() == 0 && pool.available_bytes() == pool.capacity_bytes(),
            "two-shard deterministic cleanup");
    std::cout << "Two-shard validation passed on " << device.architecture
              << "; total_vram=" << total << " free_before=" << free_before
              << " free_after=" << free_after << " committed_payload=" << one_committed << "\n";
    return 0;
}
