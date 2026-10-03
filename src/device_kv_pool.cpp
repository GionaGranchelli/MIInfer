#include "miinfer/device_kv_pool.hpp"

#include "miinfer/device_validation.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace miinfer {

namespace {

void hip_require(hipError_t result, const char* operation) {
    if (result != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result));
    }
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    const std::size_t remainder = value % alignment;
    if (remainder == 0) return value;
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - remainder)) {
        throw std::overflow_error("DeviceKvPool allocation alignment overflow");
    }
    return value + alignment - remainder;
}

}  // namespace

DeviceKvPool::DeviceKvPool(int device, std::size_t capacity_bytes)
    : device_(device), capacity_(capacity_bytes) {
    if (capacity_ == 0) throw std::invalid_argument("DeviceKvPool capacity must be > 0");
    DeviceInfo info;
    std::string error;
    if (!validate_gfx906_device(device_, info, error)) {
        throw std::invalid_argument("DeviceKvPool requires a selected gfx906 device: " + error);
    }
    set_device();
    hip_require(hipMalloc(&backing_, capacity_), "hipMalloc(DeviceKvPool)");
    free_.push_back({0, capacity_});
}

DeviceKvPool::~DeviceKvPool() {
    if (backing_ != nullptr) {
        (void)hipSetDevice(device_);
        (void)hipFree(backing_);
    }
}

void DeviceKvPool::set_device() const {
    hip_require(hipSetDevice(device_), "hipSetDevice(DeviceKvPool)");
}

DeviceKvBlock DeviceKvPool::allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) throw std::invalid_argument("DeviceKvPool allocation size must be > 0");
    if (alignment == 0) throw std::invalid_argument("DeviceKvPool alignment must be > 0");
    set_device();
    for (std::size_t index = 0; index < free_.size(); ++index) {
        const std::size_t offset = align_up(free_[index].offset, alignment);
        if (offset < free_[index].offset || offset - free_[index].offset > free_[index].bytes
            || bytes > free_[index].bytes - (offset - free_[index].offset)) {
            continue;
        }
        const std::size_t prefix = offset - free_[index].offset;
        const std::size_t suffix_offset = offset + bytes;
        const std::size_t suffix = free_[index].bytes - prefix - bytes;
        std::vector<FreeRange> replacement;
        if (prefix != 0) replacement.push_back({free_[index].offset, prefix});
        if (suffix != 0) replacement.push_back({suffix_offset, suffix});
        free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(index));
        free_.insert(free_.begin() + static_cast<std::ptrdiff_t>(index), replacement.begin(), replacement.end());
        if (next_id_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("DeviceKvPool allocation identity exhausted");
        }
        DeviceKvBlock block{device_, offset, bytes,
                            static_cast<std::byte*>(backing_) + offset, next_id_++};
        active_.push_back(block);
        committed_ += bytes;
        return block;
    }
    throw std::bad_alloc();
}

bool DeviceKvPool::owns(const DeviceKvBlock& block) const noexcept {
    return block.device == device_ && block.id != 0 && block.data != nullptr
        && block.offset <= capacity_ && block.bytes <= capacity_ - block.offset
        && block.data == static_cast<const std::byte*>(backing_) + block.offset;
}

void DeviceKvPool::release(const DeviceKvBlock& block) {
    if (!owns(block)) throw std::invalid_argument("DeviceKvPool block belongs to another pool");
    const auto it = std::find_if(active_.begin(), active_.end(),
                                 [&](const DeviceKvBlock& current) { return current.id == block.id; });
    if (it == active_.end()) throw std::invalid_argument("DeviceKvPool block was already released");
    committed_ -= it->bytes;
    free_.push_back({it->offset, it->bytes});
    active_.erase(it);
    coalesce();
}

void DeviceKvPool::coalesce() {
    std::sort(free_.begin(), free_.end(), [](const FreeRange& left, const FreeRange& right) {
        return left.offset < right.offset;
    });
    std::vector<FreeRange> merged;
    for (const FreeRange range : free_) {
        if (!merged.empty() && merged.back().offset + merged.back().bytes == range.offset) {
            merged.back().bytes += range.bytes;
        } else {
            merged.push_back(range);
        }
    }
    free_ = std::move(merged);
}

}  // namespace miinfer
