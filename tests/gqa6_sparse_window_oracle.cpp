#include "miinfer/gqa6_sparse_window.hpp"

#include <cassert>
#include <cstdint>
#include <initializer_list>

int main() {
    for (const std::uint32_t total : {1U, 2047U, 2048U}) {
        assert(miinfer::gqa6_sparse_key_count(total) == total);
        for (std::uint32_t i = 0; i < total; ++i) {
            assert(miinfer::gqa6_sparse_key_at(total, i) == i);
        }
    }
    for (const std::uint32_t total : {2049U, 2560U, 2561U, 65535U, 65536U, 65537U, 73099U}) {
        const std::uint32_t count = miinfer::gqa6_sparse_key_count(total);
        const std::uint32_t start = total - miinfer::kGqa6SparseWindow;
        const std::uint32_t anchors = (start + 511U) / 512U;
        assert(count == 2048U + anchors);
        std::uint32_t previous = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::uint32_t key = miinfer::gqa6_sparse_key_at(total, i);
            assert(key < total);
            if (i != 0) assert(previous < key);
            previous = key;
        }
        for (std::uint32_t key = start; key < total; ++key) {
            const std::uint32_t ordinal = anchors + key - start;
            assert(miinfer::gqa6_sparse_key_at(total, ordinal) == key);
        }
        for (std::uint32_t i = 0; i < anchors; ++i) {
            assert(miinfer::gqa6_sparse_key_at(total, i) == i * 512U);
            assert(i * 512U < start);
        }

        constexpr std::uint32_t splits = 32;
        std::uint32_t covered = 0;
        for (std::uint32_t split = 0; split < splits; ++split) {
            const auto first = miinfer::gqa6_sparse_split_start(count, split, splits);
            const auto last = miinfer::gqa6_sparse_split_end(count, split, splits);
            assert(first == covered);
            assert(last >= first && last <= count);
            covered = last;
        }
        assert(covered == count);
    }
    std::uint32_t covered = 0;
    for (std::uint32_t split = 0; split < 32; ++split) {
        const auto first = miinfer::gqa6_sparse_split_start(3, split, 32);
        const auto last = miinfer::gqa6_sparse_split_end(3, split, 32);
        if (first >= 3) {
            assert(last == 3);
        } else {
            assert(first == covered);
            assert(last == first + 1);
            covered = last;
        }
    }
    assert(covered == 3);
}
