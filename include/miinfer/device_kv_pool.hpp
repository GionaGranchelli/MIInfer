#pragma once

#include "miinfer/context_space.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace miinfer {

struct DeviceKvBlock {
    int device = -1;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    void* data = nullptr;
    std::uint64_t id = 0;
    [[nodiscard]] PhysicalPageView physical_view() const noexcept { return {data, bytes}; }
};

class DeviceKvPool {
public:
    DeviceKvPool(int device, std::size_t capacity_bytes);
    ~DeviceKvPool();

    DeviceKvPool(const DeviceKvPool&) = delete;
    DeviceKvPool& operator=(const DeviceKvPool&) = delete;
    DeviceKvPool(DeviceKvPool&&) = delete;
    DeviceKvPool& operator=(DeviceKvPool&&) = delete;

    [[nodiscard]] int device() const noexcept { return device_; }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t committed_bytes() const noexcept { return committed_; }
    [[nodiscard]] std::size_t available_bytes() const noexcept { return capacity_ - committed_; }

    [[nodiscard]] DeviceKvBlock allocate(std::size_t bytes, std::size_t alignment = 1);
    void release(const DeviceKvBlock& block);
    [[nodiscard]] bool owns(const DeviceKvBlock& block) const noexcept;

private:
    struct FreeRange {
        std::size_t offset;
        std::size_t bytes;
    };

    void set_device() const;
    void coalesce();

    int device_;
    std::size_t capacity_;
    std::size_t committed_ = 0;
    void* backing_ = nullptr;
    std::uint64_t next_id_ = 1;
    std::vector<FreeRange> free_;
    std::vector<DeviceKvBlock> active_;
};

}  // namespace miinfer
