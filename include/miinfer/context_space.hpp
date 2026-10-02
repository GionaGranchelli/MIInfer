#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace miinfer {

using LogicalPageId = std::uint64_t;

struct LogicalRange {
    std::size_t begin = 0;
    std::size_t length = 0;

    [[nodiscard]] std::size_t end() const noexcept { return begin + length; }
};

struct PhysicalPageView {
    const void* data = nullptr;
    std::size_t bytes = 0;
};

struct ResolvedPageView {
    LogicalPageId logical_id = 0;
    LogicalRange logical_range;
    PhysicalPageView physical;
};

class ContextSpace {
public:
    ContextSpace(std::size_t capacity, std::size_t logical_page_size);

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t logical_page_size() const noexcept { return page_size_; }
    [[nodiscard]] std::size_t committed_extent() const noexcept { return committed_; }

    LogicalRange append(std::size_t length);
    [[nodiscard]] LogicalPageId page_id_at(std::size_t position) const;
    void remap(LogicalPageId logical_id, PhysicalPageView physical);
    [[nodiscard]] std::vector<ResolvedPageView> resolve(LogicalRange range) const;

private:
    struct Page {
        LogicalPageId id;
        PhysicalPageView physical;
    };

    [[nodiscard]] const Page& page_for_id(LogicalPageId logical_id) const;
    void check_range(LogicalRange range) const;

    std::size_t capacity_;
    std::size_t page_size_;
    std::size_t committed_ = 0;
    LogicalPageId next_id_ = 1;
    std::vector<Page> pages_;
};

}  // namespace miinfer
