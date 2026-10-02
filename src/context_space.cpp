#include "miinfer/context_space.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace miinfer {

ContextSpace::ContextSpace(std::size_t capacity, std::size_t logical_page_size)
    : capacity_(capacity), page_size_(logical_page_size) {
    if (capacity_ == 0) throw std::invalid_argument("ContextSpace capacity must be > 0");
    if (page_size_ == 0) throw std::invalid_argument("ContextSpace page size must be > 0");
    pages_.reserve((capacity_ + page_size_ - 1) / page_size_);
}

LogicalRange ContextSpace::append(std::size_t length) {
    if (length > capacity_ - committed_) {
        throw std::out_of_range("ContextSpace append exceeds capacity");
    }
    const LogicalRange result{committed_, length};
    const std::size_t old_pages = (committed_ + page_size_ - 1) / page_size_;
    committed_ += length;
    const std::size_t new_pages = (committed_ + page_size_ - 1) / page_size_;
    for (std::size_t index = old_pages; index < new_pages; ++index) {
        if (next_id_ == std::numeric_limits<LogicalPageId>::max()) {
            throw std::overflow_error("ContextSpace logical page identity exhausted");
        }
        pages_.push_back({next_id_++, {}});
    }
    return result;
}

LogicalPageId ContextSpace::page_id_at(std::size_t position) const {
    if (position >= committed_) throw std::out_of_range("ContextSpace position is not committed");
    return pages_[position / page_size_].id;
}

void ContextSpace::remap(LogicalPageId logical_id, PhysicalPageView physical) {
    if (physical.data == nullptr || physical.bytes == 0) {
        throw std::invalid_argument("ContextSpace physical view must be non-empty");
    }
    for (auto& page : pages_) {
        if (page.id == logical_id) {
            page.physical = physical;
            return;
        }
    }
    throw std::out_of_range("ContextSpace logical page identity is unknown");
}

std::vector<ResolvedPageView> ContextSpace::resolve(LogicalRange range) const {
    check_range(range);
    if (range.length == 0) return {};

    const std::size_t first = range.begin / page_size_;
    const std::size_t last = (range.end() - 1) / page_size_;
    std::vector<ResolvedPageView> result;
    result.reserve(last - first + 1);
    for (std::size_t index = first; index <= last; ++index) {
        const auto& page = pages_[index];
        if (page.physical.data == nullptr || page.physical.bytes == 0) {
            throw std::logic_error("ContextSpace page has no physical view");
        }
        const std::size_t page_begin = index * page_size_;
        const std::size_t page_end = std::min(page_begin + page_size_, committed_);
        const std::size_t begin = std::max(page_begin, range.begin);
        const std::size_t end = std::min(page_end, range.end());
        result.push_back({page.id, {begin, end - begin}, page.physical});
    }
    return result;
}

const ContextSpace::Page& ContextSpace::page_for_id(LogicalPageId logical_id) const {
    for (const auto& page : pages_) {
        if (page.id == logical_id) return page;
    }
    throw std::out_of_range("ContextSpace logical page identity is unknown");
}

void ContextSpace::check_range(LogicalRange range) const {
    if (range.begin > committed_ || range.length > committed_ - range.begin) {
        throw std::out_of_range("ContextSpace range is outside committed extent");
    }
}

}  // namespace miinfer
