#include "miinfer/device_kv_shard.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
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
    miinfer::ContextSpace context(8, 4);
    context.append(8);
    const auto first_id = context.page_id_at(0);
    const auto second_id = context.page_id_at(4);
    std::uint32_t first_storage[4] = {};
    std::uint32_t second_storage[4] = {};
    miinfer::PlacementPlan plan(context);
    plan.place({0, first_id, {0, 4}, {first_storage, 0, sizeof(first_storage)}});
    plan.place({0, second_id, {4, 4}, {second_storage, 0, sizeof(second_storage)}});
    require(plan.size() == 2, "placement count");
    const auto resolved = plan.resolve({1, 6});
    require(resolved.size() == 2, "N=1 range resolution");
    require(resolved[0].owning_device() == 0 && resolved[1].owning_device() == 0, "single owner");
    require(resolved[0].logical_page() == first_id && resolved[1].logical_page() == second_id, "logical identity");
    const auto replacement_id = context.page_id_at(0);
    std::uint32_t replacement_storage[4] = {};
    plan.replace({1, replacement_id, {0, 4}, {replacement_storage, 0, sizeof(replacement_storage)}});
    require(context.page_id_at(0) == first_id && plan.resolve({0, 4})[0].owning_device() == 1,
            "replacement preserves logical identity");
    throws([&] { plan.place({0, first_id, {0, 4}, {first_storage, 0, sizeof(first_storage)}}); });
    throws([&] { plan.place({0, second_id, {3, 4}, {second_storage, 0, sizeof(second_storage)}}); });
    throws([&] { plan.place({0, second_id, {4, 4}, {replacement_storage, 0, sizeof(replacement_storage)}}); });
    throws([&] { (void)plan.resolve({0, 9}); });
    throws([&] { miinfer::DeviceKvShard(-1, first_id, {0, 4}, {first_storage, 0, sizeof(first_storage)}); });
    throws([&] { miinfer::DeviceKvShard(0, first_id, {0, 4}, {nullptr, 0, 4}); });
    std::cout << "DeviceKvShard placement tests passed\n";
    return 0;
}
