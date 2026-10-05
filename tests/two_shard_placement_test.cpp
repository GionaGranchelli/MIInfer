#include "miinfer/device_kv_pool.hpp"
#include "miinfer/device_kv_shard.hpp"
#include "miinfer/device_validation.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

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
        std::cerr << "two-shard GPU test unavailable: " << error << '\n';
        return 1;
    }
    require(device.architecture.rfind("gfx906", 0) == 0, "two-shard device is not gfx906");
    require(device.total_vram_bytes >= 30ULL * 1024 * 1024 * 1024, "two-shard device is not 32 GiB class");

    miinfer::ContextSpace context(4096, 4096);
    const auto logical = context.append(4096);
    const auto logical_id = context.page_id_at(0);
    miinfer::DeviceKvPool pool(device.index, 65536);
    miinfer::PlacementPlan plan(context);

    const auto one = pool.allocate(8192, 256);
    plan.place({device.index, logical_id, logical, {one.data, 0, one.bytes}, {0, 4}});
    require(plan.resolve(logical, {0, 4}).size() == 1, "one-shard placement");
    require(plan.resolve(logical)[0].logical_page() == logical_id, "one-shard identity");
    require(pool.committed_bytes() == 8192, "one-shard accounting");

    require(plan.clear_page(logical_id) == 1, "clear one-shard placement");
    pool.release(one);
    const auto shard_a = pool.allocate(4096, 256);
    const auto shard_b = pool.allocate(4096, 256);
    require(shard_a.offset != shard_b.offset, "two physical ranges must be disjoint");
    plan.place({device.index, logical_id, logical, {shard_a.data, 0, shard_a.bytes}, {0, 2}});
    plan.place({device.index, logical_id, logical, {shard_b.data, 0, shard_b.bytes}, {2, 2}});
    const auto two = plan.resolve(logical, {0, 4});
    require(two.size() == 2, "two-shard placement");
    require(two[0].owning_device() == device.index && two[1].owning_device() == device.index,
            "same-device ownership");
    require(two[0].logical_page() == logical_id && two[1].logical_page() == logical_id,
            "two-shard identity");
    require(pool.committed_bytes() == 8192 && pool.available_bytes() == 57344,
            "two-shard accounting without payload duplication");

    const auto bad = pool.allocate(4096, 256);
    throws([&] { plan.place({device.index, logical_id, logical, {bad.data, 0, bad.bytes}, {1, 1}}); });
    pool.release(bad);
    miinfer::PlacementPlan incomplete(context);
    incomplete.place({device.index, logical_id, logical, {shard_a.data, 0, shard_a.bytes}, {0, 2}});
    require(incomplete.resolve(logical).size() == 1, "partial metadata resolution");
    throws([&] { (void)incomplete.resolve(logical, {0, 4}); });

    require(plan.clear_page(logical_id) == 2, "clear two-shard placement");
    pool.release(shard_a);
    pool.release(shard_b);
    const auto one_again = pool.allocate(8192, 256);
    plan.place({device.index, logical_id, logical, {one_again.data, 0, one_again.bytes}, {0, 4}});
    require(context.page_id_at(0) == logical_id && plan.resolve(logical, {0, 4}).size() == 1,
            "one-shard identity after round trip");
    require(pool.committed_bytes() == 8192, "reallocation accounting");
    require(plan.clear_page(logical_id) == 1, "clear final placement");
    pool.release(one_again);
    require(pool.committed_bytes() == 0 && pool.available_bytes() == pool.capacity_bytes(),
            "deterministic cleanup");

    std::cout << "Two-shard placement passed on " << device.architecture
              << "; shard_bytes=4096+4096 total=8192 metadata_bytes="
              << 2 * sizeof(miinfer::DeviceKvBlock) << "\n";
    return 0;
}
