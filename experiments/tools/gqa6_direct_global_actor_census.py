#!/usr/bin/env python3
"""Count direct-global GQA6 Stage1 actors/loads and summarize rocprof counters."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def split_interval(length: int, split: int, splits: int) -> tuple[int, int]:
    chunk = (length + splits - 1) // splits
    start = split * chunk
    return start, min(start + chunk, length)


def stage1_counts(base: int, queries: int, splits: int) -> tuple[int, int]:
    active_ctas = 0
    key_visits = 0
    for token in range(queries):
        length = base + token + 1
        for split in range(splits):
            start, end = split_interval(length, split, splits)
            count = max(0, end - start)
            if count:
                active_ctas += 1
                key_visits += count
    return active_ctas, key_visits


def pair_union_visits(base: int, queries: int, splits: int) -> int:
    union_visits = 0
    for first in range(0, queries, 2):
        lengths = [base + token + 1 for token in range(first, min(first + 2, queries))]
        for split in range(splits):
            intervals = [split_interval(length, split, splits) for length in lengths]
            active = [(start, end) for start, end in intervals if start < end]
            if active:
                union_visits += max(end for _, end in active) - min(start for start, _ in active)
    return union_visits


def active_pair_ctas(base: int, queries: int, splits: int) -> int:
    count = 0
    for first in range(0, queries, 2):
        lengths = [base + token + 1 for token in range(first, min(first + 2, queries))]
        for split in range(splits):
            if any(start < end for start, end in
                   (split_interval(length, split, splits) for length in lengths)):
                count += 1
    return count


def summarize_profile(paths: list[Path]) -> None:
    totals: dict[str, list[float]] = defaultdict(list)
    dispatches: set[str] = set()
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                name = row["Kernel_Name"]
                if "qwen35_splitk_suffix_attn_stage1_quant_kernel" not in name:
                    continue
                if "true, true, 4, true, false, false, true>" not in name:
                    continue
                totals[row["Counter_Name"]].append(float(row["Counter_Value"]))
                dispatches.add(row["Dispatch_Id"])

    if not totals:
        raise SystemExit("profile CSVs contain no direct-global GQA6 Stage1 rows")

    print(f"profile_direct_global_dispatches={len(dispatches)}")
    for name, values in sorted(totals.items()):
        print(f"counter_{name}_mean={sum(values) / len(values):.6g} samples={len(values)}")
    hit = sum(totals.get("TCC_HIT", []))
    miss = sum(totals.get("TCC_MISS", []))
    if hit + miss:
        print(f"l2_hit_fraction={hit / (hit + miss):.6%}")
    ea = sum(totals.get("TCC_EA_RDREQ_32B", []))
    if ea:
        print(f"hbm_read_requests_32B={ea:.0f} hbm_read_bytes_estimate={ea * 32:.0f}")
    else:
        print("hbm_read_requests=unavailable (TCC_EA_RDREQ_32B absent or zero)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-position", type=int, default=73099)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--splits", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--query-heads-per-kv", type=int, default=6)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--profile-csv", type=Path, action="append", default=[])
    args = parser.parse_args()
    if min(args.base_position, args.queries, args.splits, args.kv_heads,
           args.query_heads_per_kv, args.head_dim) <= 0:
        parser.error("all dimensions and counts must be positive")

    # Each query head issues 32 x 8-byte loads for both K and V per key.
    active, visits = stage1_counts(args.base_position, args.queries, args.splits)
    ctas = active * args.kv_heads
    kv_bytes_per_head = 2 * args.head_dim
    pair_visits = pair_union_visits(args.base_position, args.queries, args.splits)
    pair_ctas = active_pair_ctas(args.base_position, args.queries, args.splits) * args.kv_heads
    current_kv_bytes = visits * args.kv_heads * args.query_heads_per_kv * kv_bytes_per_head
    pair_kv_bytes = pair_visits * args.kv_heads * args.query_heads_per_kv * kv_bytes_per_head
    one_load_per_kv_head_bytes = visits * args.kv_heads * kv_bytes_per_head
    scale_bytes_per_key = args.query_heads_per_kv * 2 * 2
    print(f"base={args.base_position} queries={args.queries} splits={args.splits}")
    print(f"active_ctas={ctas} q_heads_per_cta={args.query_heads_per_kv} kv_heads={args.kv_heads}")
    print(f"key_visits_per_kv_head={visits} adjacent_pair_union_visits_per_kv_head={pair_visits}")
    print(f"active_pair_ctas={pair_ctas} rows_per_pair_cta=2")
    print(f"current_direct_global_q8_kv_load_payload_bytes={current_kv_bytes}")
    print(f"pair_row_reuse_q8_kv_load_payload_bytes={pair_kv_bytes}")
    print(f"pair_reuse_payload_fraction={pair_kv_bytes / current_kv_bytes:.6%}")
    print(f"current_payload_bytes_per_cta={current_kv_bytes / ctas:.1f}")
    print(f"pair_payload_bytes_per_pair_cta={pair_kv_bytes / pair_ctas:.1f}")
    print(f"current_payload_bytes_per_query_row={current_kv_bytes / args.queries:.1f}")
    print(f"pair_payload_bytes_per_query_row={pair_kv_bytes / args.queries:.1f}")
    print(f"one_q8_kv_load_per_kv_head_payload_bytes={one_load_per_kv_head_bytes}")
    print(f"current_scale_load_payload_bytes={visits * args.kv_heads * scale_bytes_per_key}")
    print("load payload is logical lane load traffic, not HBM bytes")
    if args.profile_csv:
        summarize_profile(args.profile_csv)


if __name__ == "__main__":
    assert stage1_counts(0, 1, 32) == (1, 1)
    assert stage1_counts(1, 1, 32) == (2, 2)
    assert pair_union_visits(0, 0, 32) == 0  # no active intervals
    assert pair_union_visits(0, 1, 32) == 1  # one active row
    assert pair_union_visits(0, 2, 32) == 2  # union of key positions [0, 2)
    assert active_pair_ctas(0, 1, 32) == 1  # one active pair CTA
    main()
