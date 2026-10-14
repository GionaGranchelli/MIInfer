#!/usr/bin/env python3
"""Count GQA6 Split-K Stage1 work assigned to each split for fixed workloads."""

from statistics import mean


UNROLL = 4
SPLIT_COUNTS = (32, 64)


def report(name: str, query_lengths: range, num_splits: int) -> None:
    split_work = [0] * num_splits
    cta_lengths = []
    for total_length in query_lengths:
        chunk = (total_length + num_splits - 1) // num_splits
        for split in range(num_splits):
            start = split * chunk
            count = max(0, min(start + chunk, total_length) - start)
            if count:
                split_work[split] += count
                cta_lengths.append(count)

    loop_groups = [(count + UNROLL - 1) // UNROLL for count in cta_lengths]
    average_split_work = mean(split_work)
    skew = (max(split_work) / average_split_work - 1) * 100
    print(
        f"{name} splits={num_splits} active_ctas_per_kv_head={len(cta_lengths):,} "
        f"key_visits_per_kv_head={sum(split_work):,} "
        f"keys_per_cta={mean(cta_lengths):.1f} [{min(cta_lengths)}, {max(cta_lengths)}] "
        f"unroll4_groups_per_cta={mean(loop_groups):.1f} "
        f"split_key_visits=[{min(split_work):,}, {max(split_work):,}] "
        f"max_split_skew={skew:.4f}%"
    )


def main() -> None:
    workloads = (
        ("restored_73099_plus_1024", range(73100, 74124)),
        ("cold_64k", range(1, 65537)),
    )
    for name, query_lengths in workloads:
        for splits in SPLIT_COUNTS:
            report(name, query_lengths, splits)


if __name__ == "__main__":
    main()
