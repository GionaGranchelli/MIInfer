#include "miinfer/device_kv_shard.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace miinfer {

namespace {
void check_end(LogicalRange range, const char* name) {
    if (range.length == 0 || range.begin > std::numeric_limits<std::size_t>::max() - range.length) {
        throw std::invalid_argument(std::string(name) + " is empty or overflows");
    }
}
}

DeviceKvShard::DeviceKvShard(int owning_device,
                             LogicalPageId logical_page,
                             LogicalRange logical_range,
                             PhysicalKvRange physical,
                             KvHeadRange heads)
    : device_(owning_device), page_(logical_page), logical_(logical_range), physical_(physical), heads_(heads) {
    if (device_ < 0) throw std::invalid_argument("DeviceKvShard owner must be a device index");
    if (logical_page == 0) throw std::invalid_argument("DeviceKvShard logical page must be non-zero");
    check_end(logical_, "logical range");
    if (physical_.data == nullptr || physical_.bytes == 0
        || physical_.offset > std::numeric_limits<std::size_t>::max() - physical_.bytes) {
        throw std::invalid_argument("DeviceKvShard physical range is invalid");
    }
    if (heads_.begin > std::numeric_limits<std::size_t>::max() - heads_.count) {
        throw std::invalid_argument("DeviceKvShard KV-head range overflows");
    }
}

void PlacementPlan::validate(const DeviceKvShard& shard) const {
    const LogicalRange logical = shard.logical_range();
    check_end(logical, "logical range");
    if (logical.begin >= context_->committed_extent() || logical.end() > context_->committed_extent()) {
        throw std::out_of_range("DeviceKvShard logical range is not committed");
    }
    if (logical.begin % context_->logical_page_size() != 0) {
        throw std::invalid_argument("DeviceKvShard logical range must begin at a page boundary");
    }
    const std::size_t expected = std::min(context_->logical_page_size(), context_->committed_extent() - logical.begin);
    if (logical.length != expected || context_->page_id_at(logical.begin) != shard.logical_page()) {
        throw std::invalid_argument("DeviceKvShard does not describe the ContextSpace page");
    }
    for (const auto& existing : shards_) {
        if (overlaps(existing.logical_range(), logical)) {
            const auto existing_logical = existing.logical_range();
            const auto existing_heads = existing.kv_heads();
            const auto new_heads = shard.kv_heads();
            const bool same_page = existing.logical_page() == shard.logical_page()
                && existing_logical.begin == logical.begin && existing_logical.end() == logical.end();
            const bool disjoint_heads = existing_heads.count != 0 && new_heads.count != 0
                && (existing_heads.end() <= new_heads.begin || new_heads.end() <= existing_heads.begin);
            if (!same_page || !disjoint_heads) {
                throw std::invalid_argument("PlacementPlan has overlapping logical placement");
            }
        }
        if (existing.owning_device() == shard.owning_device()) {
            const auto left = existing.physical_range();
            const auto right = shard.physical_range();
            if (left.data == right.data && left.offset < right.end() && right.offset < left.end()) {
                throw std::invalid_argument("PlacementPlan has overlapping physical placement");
            }
        }
    }
}

bool PlacementPlan::overlaps(LogicalRange left, LogicalRange right) const noexcept {
    return left.begin < right.end() && right.begin < left.end();
}

void PlacementPlan::place(DeviceKvShard shard) {
    validate(shard);
    shards_.push_back(std::move(shard));
    std::sort(shards_.begin(), shards_.end(), [](const auto& left, const auto& right) {
        if (left.logical_range().begin != right.logical_range().begin) {
            return left.logical_range().begin < right.logical_range().begin;
        }
        return left.kv_heads().begin < right.kv_heads().begin;
    });
}

std::size_t PlacementPlan::clear_page(LogicalPageId logical_page) {
    const auto old_size = shards_.size();
    shards_.erase(std::remove_if(shards_.begin(), shards_.end(), [&](const auto& shard) {
        return shard.logical_page() == logical_page;
    }), shards_.end());
    return old_size - shards_.size();
}

void PlacementPlan::replace(DeviceKvShard shard) {
    const auto it = std::find_if(shards_.begin(), shards_.end(), [&](const auto& current) {
        return current.logical_page() == shard.logical_page();
    });
    if (it == shards_.end()) throw std::out_of_range("PlacementPlan has no shard to replace");
    const DeviceKvShard previous = *it;
    shards_.erase(it);
    try {
        validate(shard);
    } catch (...) {
        shards_.push_back(previous);
        std::sort(shards_.begin(), shards_.end(), [](const auto& left, const auto& right) {
            return left.logical_range().begin < right.logical_range().begin;
        });
        throw;
    }
    shards_.push_back(std::move(shard));
    std::sort(shards_.begin(), shards_.end(), [](const auto& left, const auto& right) {
        return left.logical_range().begin < right.logical_range().begin;
    });
}

std::vector<DeviceKvShard> PlacementPlan::resolve(LogicalRange range) const {
    if (range.length == 0) {
        if (range.begin > context_->committed_extent()) {
            throw std::out_of_range("PlacementPlan resolve range is not committed");
        }
        return {};
    }
    check_end(range, "resolve range");
    if (range.begin > context_->committed_extent() || range.end() > context_->committed_extent()) {
        throw std::out_of_range("PlacementPlan resolve range is not committed");
    }
    std::vector<DeviceKvShard> result;
    std::size_t covered = range.begin;
    for (const auto& shard : shards_) {
        const auto logical = shard.logical_range();
        if (logical.end() <= range.begin || logical.begin >= range.end()) continue;
        const bool same_logical_coverage = !result.empty()
            && logical.begin == result.back().logical_range().begin
            && logical.end() == result.back().logical_range().end();
        if ((!same_logical_coverage && result.empty() && logical.begin > covered)
            || (!same_logical_coverage && !result.empty() && logical.begin != covered)) {
            throw std::logic_error("PlacementPlan has a gap in resolved placement");
        }
        result.push_back(shard);
        covered = std::max(covered, logical.end());
    }
    if (covered < range.end()) throw std::logic_error("PlacementPlan has no complete physical placement");
    return result;
}

std::vector<DeviceKvShard> PlacementPlan::resolve(LogicalRange range, KvHeadRange heads) const {
    if (heads.count == 0 || heads.begin > std::numeric_limits<std::size_t>::max() - heads.count) {
        throw std::invalid_argument("PlacementPlan expected KV-head range is invalid");
    }
    const auto result = resolve(range);
    LogicalPageId current_page = 0;
    std::size_t covered = heads.begin;
    for (const auto& shard : result) {
        if (shard.logical_page() != current_page) {
            if (current_page != 0 && covered != heads.end()) {
                throw std::logic_error("PlacementPlan has incomplete KV-head coverage");
            }
            current_page = shard.logical_page();
            covered = heads.begin;
        }
        const auto shard_heads = shard.kv_heads();
        if (shard_heads.begin != covered || shard_heads.end() > heads.end()) {
            throw std::logic_error("PlacementPlan has overlapping or incomplete KV-head coverage");
        }
        covered = shard_heads.end();
    }
    if (current_page == 0 || covered != heads.end()) {
        throw std::logic_error("PlacementPlan has incomplete KV-head coverage");
    }
    return result;
}

}  // namespace miinfer
