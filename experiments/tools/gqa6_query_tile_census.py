#!/usr/bin/env python3
"""Count Split-K KV rows requested by GQA6 query tiles, ignoring cache effects."""

import argparse
import math


def split_range(length: int, split: int, split_count: int) -> tuple[int, int]:
    chunk = (length + split_count - 1) // split_count
    start = split * chunk
    return start, min(start + chunk, length)


def census(base_position: int, query_count: int, split_count: int, tile: int) -> tuple[int, int, int]:
    per_query_rows = 0
    tiled_union_rows = 0
    cta_count = 0
    for first in range(0, query_count, tile):
        queries = range(first, min(first + tile, query_count))
        cta_count += split_count
        for split in range(split_count):
            ranges = [split_range(base_position + query + 1, split, split_count) for query in queries]
            per_query_rows += sum(end - start for start, end in ranges)
            tiled_union_rows += max(end for _, end in ranges) - min(start for start, _ in ranges)
    return per_query_rows, tiled_union_rows, cta_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-position", type=int, default=73099)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--splits", type=int, default=32)
    parser.add_argument("--tiles", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    if min(args.base_position, args.queries, args.splits, *args.tiles) <= 0:
        parser.error("positions, queries, splits, and tile sizes must be positive")

    baseline_rows = None
    print(f"base={args.base_position} queries={args.queries} splits={args.splits}")
    print("tile ctas_per_kv_head requested_rows staged_union_rows ideal_reuse traffic_vs_tile1")
    for tile in args.tiles:
        requested, staged, ctas = census(args.base_position, args.queries, args.splits, tile)
        baseline_rows = requested if baseline_rows is None else baseline_rows
        assert requested == baseline_rows
        print(f"{tile:4d} {ctas:16d} {requested:13d} {staged:16d} "
              f"{requested / staged:11.3f}x {staged / requested:14.3%}")

    one_query = census(args.base_position, args.queries, args.splits, 1)
    assert one_query[0] == one_query[1]


if __name__ == "__main__":
    main()
