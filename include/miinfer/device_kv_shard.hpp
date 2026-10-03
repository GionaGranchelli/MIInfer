#pragma once

#include "miinfer/context_space.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace miinfer {

struct PhysicalKvRange {
    const void* data = nullptr;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    [[nodiscard]] std::size_t end() const noexcept { return offset + bytes; }
    [[nodiscard]] PhysicalPageView view() const noexcept {
        return {static_cast<const std::byte*>(data) + offset, bytes};
    }
};

struct KvHeadRange {
    std::size_t begin = 0;
    std::size_t count = 0;
    [[nodiscard]] std::size_t end() const noexcept { return begin + count; }
};

class DeviceKvShard {
public:
    DeviceKvShard(int owning_device,
                  LogicalPageId logical_page,
                  LogicalRange logical_range,
                  PhysicalKvRange physical,
                  KvHeadRange heads = {});

    [[nodiscard]] int owning_device() const noexcept { return device_; }
    [[nodiscard]] LogicalPageId logical_page() const noexcept { return page_; }
    [[nodiscard]] LogicalRange logical_range() const noexcept { return logical_; }
    [[nodiscard]] PhysicalKvRange physical_range() const noexcept { return physical_; }
    [[nodiscard]] KvHeadRange kv_heads() const noexcept { return heads_; }

private:
    int device_;
    LogicalPageId page_;
    LogicalRange logical_;
    PhysicalKvRange physical_;
    KvHeadRange heads_;
};

class PlacementPlan {
public:
    explicit PlacementPlan(const ContextSpace& context) : context_(&context) {}

    void place(DeviceKvShard shard);
    void replace(DeviceKvShard shard);
    [[nodiscard]] std::size_t clear_page(LogicalPageId logical_page);
    [[nodiscard]] std::vector<DeviceKvShard> resolve(LogicalRange range) const;
    [[nodiscard]] std::size_t size() const noexcept { return shards_.size(); }

private:
    void validate(const DeviceKvShard& shard) const;
    [[nodiscard]] bool overlaps(LogicalRange left, LogicalRange right) const noexcept;

    const ContextSpace* context_;
    std::vector<DeviceKvShard> shards_;
};

}  // namespace miinfer
