#include "miinfer/gqa6_query_block_sparse.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    for (const std::uint32_t total : {
             1U, 2047U, 2048U, 2049U, 2559U, 2560U, 2561U,
             8191U, 8192U, 8193U, 65535U, 65536U, 65537U,
             73099U, 98304U, 131071U, 131072U}) {
        const auto local_start = miinfer::gqa6_query_sparse_local_start(total);
        const auto block_count = local_start / miinfer::kGqa6QuerySparseBlock;
        const auto selected_count = std::min(block_count, miinfer::kGqa6QuerySparseTopBlocks);
        std::vector<std::uint32_t> selected(selected_count);
        for (std::uint32_t i = 0; i < selected_count; ++i) {
            selected[i] = i * block_count / selected_count;
        }
        const auto key_count = miinfer::gqa6_query_sparse_key_count(total, selected_count);
        std::vector<std::uint32_t> mapped;
        mapped.reserve(key_count);
        for (std::uint32_t ordinal = 0; ordinal < key_count; ++ordinal) {
            const auto key = miinfer::gqa6_query_sparse_key_at(
                total, selected.data(), selected_count, ordinal);
            assert(key < total);
            mapped.push_back(key);
        }
        assert(std::is_sorted(mapped.begin(), mapped.end()));
        assert(std::adjacent_find(mapped.begin(), mapped.end()) == mapped.end());
        if (local_start == 0) {
            assert(mapped.size() == total);
            for (std::uint32_t key = 0; key < total; ++key) assert(mapped[key] == key);
        } else {
            assert(mapped.size() == selected_count * miinfer::kGqa6QuerySparseBlock
                + total - local_start);
            for (std::uint32_t key = local_start; key < total; ++key) {
                assert(std::binary_search(mapped.begin(), mapped.end(), key));
            }
        }
    }
}
