#include "miinfer/context_space.hpp"

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
    try {
        function();
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, "expected exception");
}

}  // namespace

int main() {
    using miinfer::ContextSpace;
    using miinfer::LogicalRange;

    throws([] { ContextSpace(0, 4); });
    throws([] { ContextSpace(8, 0); });

    ContextSpace space(10, 4);
    require(space.capacity() == 10, "capacity");
    require(space.committed_extent() == 0, "empty extent");
    require(space.resolve({0, 0}).empty(), "empty resolve");

    const auto first = space.append(1);
    require(first.begin == 0 && first.length == 1, "first append");
    require(space.committed_extent() == 1, "first extent");
    const auto first_id = space.page_id_at(0);
    require(first_id != 0, "first identity");

    std::uint32_t first_storage = 1;
    space.remap(first_id, {&first_storage, sizeof(first_storage)});
    const auto first_view = space.resolve(first);
    require(first_view.size() == 1, "first view count");
    require(first_view[0].logical_id == first_id, "first view identity");
    require(first_view[0].physical.data == &first_storage, "first view backing");

    const auto growth = space.append(7);
    require(growth.begin == 1 && growth.length == 7, "growth append");
    require(space.committed_extent() == 8, "growth extent");
    const auto second_id = space.page_id_at(4);
    require(second_id != first_id, "multiple identities");
    require(space.page_id_at(0) == first_id, "stable first identity after growth");
    require(space.page_id_at(7) == second_id, "page boundary identity");
    throws([&] { (void)space.resolve({3, 3}); });
    throws([&] { (void)space.resolve({8, 1}); });
    throws([&] { (void)space.page_id_at(8); });
    throws([&] { space.append(3); });
    throws([&] { space.remap(999, {&first_storage, sizeof(first_storage)}); });
    throws([&] { space.remap(second_id, {nullptr, sizeof(first_storage)}); });
    throws([&] { (void)space.resolve({4, 1}); });

    std::uint32_t second_storage = 2;
    space.remap(second_id, {&second_storage, sizeof(second_storage)});
    require(space.resolve({1, 7}).size() == 2, "multiple page resolution");
    require(space.resolve({3, 3})[0].logical_range.begin == 3, "range boundary first");
    require(space.resolve({3, 3})[1].logical_range.begin == 4, "range boundary second");
    const auto remapped = space.resolve({4, 1});
    require(remapped.size() == 1, "remapped view count");
    require(remapped[0].logical_id == second_id, "remapped identity");
    require(remapped[0].physical.data == &second_storage, "remapped backing");
    space.remap(second_id, {&first_storage, sizeof(first_storage)});
    require(space.page_id_at(4) == second_id, "stable identity after remap");

    ContextSpace empty_tail(8, 4);
    empty_tail.append(4);
    throws([&] { (void)empty_tail.resolve({0, 4}); });
    std::cout << "ContextSpace host tests passed\n";
    return 0;
}
