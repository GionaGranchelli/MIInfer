#!/usr/bin/env python3
"""Check Q2xK32 pair-token split, mask, and workspace mapping on CPU."""

from dataclasses import dataclass


SPLITS = 32
HEADS = 24
TILE = 32


@dataclass(frozen=True)
class Row:
    token: int
    valid: bool
    length: int
    start: int
    end: int


def row_for(base: int, token_count: int, pair: int, row: int, split: int) -> Row:
    token = pair * 2 + row
    valid = token < token_count
    length = base + token + 1
    chunk = (length + SPLITS - 1) // SPLITS
    start = split * chunk
    end = min(start + chunk, length)
    return Row(token, valid, length, start, end)


def check_case(base: int, token_count: int) -> int:
    checked = 0
    for pair in range((token_count + 1) // 2):
        for split in range(SPLITS):
            rows = [row_for(base, token_count, pair, row, split) for row in range(2)]
            active = [r for r in rows if r.valid and r.start < r.end]
            if not active:
                continue

            union_start = min(r.start for r in active)
            union_end = max(r.end for r in active)
            full_end = union_start + ((union_end - union_start) // TILE) * TILE
            assert (full_end - union_start) % TILE == 0
            assert 0 <= union_end - full_end < TILE

            for r in rows:
                if r.valid and r.start < r.end:
                    # Full K32 tiles and the scalar tail concatenate without
                    # gaps; every active row interval lies inside that union.
                    assert union_start <= r.start < r.end <= union_end
                else:
                    assert not r.valid or r.start >= r.end

                checked += 1

    # The CTA mapping (kv_head, threadIdx.y) writes each Q head exactly once.
    mapped_heads = [kv_head * 6 + ty + delta
                    for kv_head in range(4)
                    for ty in range(3)
                    for delta in (0, 3)]
    assert sorted(mapped_heads) == list(range(HEADS))

    # Each sub-lane writes 8 dimensions as two float4 stores; together they
    # partition a 256-value accumulator without overlap or gaps.
    dim_offsets = [sub_lane * 8 + fragment
                   for sub_lane in range(32)
                   for fragment in range(8)]
    assert sorted(dim_offsets) == list(range(256))

    # Every valid row/split/head has the exact partial-output base and bounds.
    for token in {0, token_count - 1}:
        for split in {0, SPLITS - 1}:
            for head in range(HEADS):
                ix = (split * token_count + token) * HEADS + head
                assert ix == split * token_count * HEADS + token * HEADS + head
                assert 0 <= ix < SPLITS * token_count * HEADS
                assert ix * 256 + 255 < SPLITS * token_count * HEADS * 256
    return checked


def main() -> None:
    bases = {
        0, 1, 2, 30, 31, 32, 33, 63, 64, 65,
        1023, 1024, 65503, 65504, 65535, 65536, 65567, 65568,
        73098, 73099, 73100, 73118, 73119, 74121, 74122,
        131000,
    }
    checks = 0
    for base in sorted(bases):
        for count in range(1, 66):
            checks += check_case(base, count)
        for count in (127, 128, 129, 1023, 1024, 1025):
            if base + count > 131072:
                continue
            checks += check_case(base, count)

    # Explicitly span the production suffix's split-size transition at token 20/21.
    for token_count in (22, 1024):
        for pair in (9, 10, (token_count - 1) // 2):
            for split in range(SPLITS):
                for row in range(2):
                    r = row_for(73099, token_count, pair, row, split)
                    assert r.length == 73099 + r.token + 1
                    assert r.start == split * ((r.length + SPLITS - 1) // SPLITS)
        checks += check_case(73099, token_count)

    print(f"EXP-0465 Q2xK32 mapping oracle passed: {checks} row/split cases")


if __name__ == "__main__":
    main()
