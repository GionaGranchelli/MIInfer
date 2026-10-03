#include "miinfer/prefill_v2/kv_cache.hpp"

#include <iostream>

int main() {
    miinfer::prefill_v2::AttentionLayerKvCacheStorage storage(8);
    const auto view = storage.view();

    if (!view.is_single_shard() || view.shard_count != 1
        || view.shards[0].key_cache != view.key_cache
        || view.shards[0].value_cache != view.value_cache
        || view.shards[0].head_begin != 0
        || view.shards[0].head_count != view.head_count_kv) {
        std::cerr << "PhysicalKvView single-shard resolution failed\n";
        return 1;
    }

    std::cout << "PhysicalKvView single-shard resolution passed\n";
    return 0;
}
