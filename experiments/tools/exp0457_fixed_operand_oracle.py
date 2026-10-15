#!/usr/bin/env python3
"""Small mapping and FP32 online-softmax oracle for EXP-0457."""
from __future__ import annotations

import math
import numpy as np


def split_bounds(length: int, splits: int, split: int) -> tuple[int, int]:
    chunk = (length + splits - 1) // splits
    start = split * chunk
    return start, min(start + chunk, length)


def sequential(q: np.ndarray, keys: np.ndarray, key_scale: np.ndarray,
               values: np.ndarray, value_scale: np.ndarray, start: int,
               end: int, attn_scale: float) -> tuple[float, float, np.ndarray]:
    m, s = -math.inf, 0.0
    acc = np.zeros(values.shape[1], dtype=np.float32)
    for pos in range(start, end):
        score = np.float32(0.0)
        for d in range(q.size):
            score = np.float32(score + np.float32(q[d] * np.float32(keys[pos, d])))
        score = np.float32(score * np.float32(key_scale[pos] * attn_scale))
        nm = max(m, float(score))
        a = math.exp(m - nm) if s else 0.0
        b = math.exp(float(score) - nm)
        s = s * a + b
        acc *= np.float32(a)
        acc += np.float32(b * value_scale[pos]) * values[pos].astype(np.float32)
        m = nm
    return float(m), float(s), acc


def tiled(q: np.ndarray, keys: np.ndarray, key_scale: np.ndarray,
          values: np.ndarray, value_scale: np.ndarray, start: int,
          end: int, attn_scale: float, tile: int = 32
          ) -> tuple[float, float, np.ndarray]:
    m, s = -math.inf, 0.0
    acc = np.zeros(values.shape[1], dtype=np.float32)
    for tile_start in range(start, end, tile):
        tile_end = min(tile_start + tile, end)
        scores = []
        for pos in range(tile_start, tile_end):
            score = np.float32(0.0)
            for d in range(q.size):
                score = np.float32(score + np.float32(q[d] * np.float32(keys[pos, d])))
            scores.append(np.float32(score * np.float32(key_scale[pos] * attn_scale)))
        tm = max(float(x) for x in scores)
        weights = [math.exp(float(x) - tm) for x in scores]
        ts = sum(weights)
        nm = max(m, tm)
        a = math.exp(m - nm) if s else 0.0
        b = math.exp(tm - nm)
        acc *= np.float32(a)
        for offset, weight in enumerate(weights):
            pos = tile_start + offset
            acc += np.float32(weight * b * value_scale[pos]) * values[pos].astype(np.float32)
        s = s * a + ts * b
        m = nm
    return float(m), float(s), acc


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    a = np.asarray(actual, dtype=np.float64).ravel()
    e = np.asarray(expected, dtype=np.float64).ravel()
    max_abs = float(np.max(np.abs(a - e))) if a.size else 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(e))
    cosine = float(np.dot(a, e) / denom) if denom else 1.0
    assert max_abs <= 1e-4 and cosine >= 0.99999, (name, max_abs, cosine)


def main() -> None:
    # Explicit GQA row and workspace layout assertions for a CTA.
    kv_head, head_pair, pair, token_count = 2, 1, 10, 40
    rows = [(pair * 2 + row // 2, kv_head * 6 + head_pair * 2 + row % 2)
            for row in range(4)]
    assert rows == [(20, 14), (20, 15), (21, 14), (21, 15)]
    split, splits = 31, 32
    bounds = [split_bounds(73_099 + token + 1, splits, split)
              for token in (20, 21)]
    assert bounds == [(70_835, 73_120), (70_866, 73_121)], bounds
    # The CTA stages the union, but each row consumes only its own interval.
    union = (min(x[0] for x in bounds), max(x[1] for x in bounds))
    assert union == (70_835, 73_121)
    assert [p for p in range(*union) if bounds[0][0] <= p < bounds[0][1]][-1] == 73_119
    assert [p for p in range(*union) if bounds[1][0] <= p < bounds[1][1]][0] == 70_866

    # Exercise cold split-boundary and empty-split contracts, including partial BK.
    for length in (30, 31, 32, 33, 65):
        for sidx in (0, 31):
            start, end = split_bounds(length, 32, sidx)
            assert end <= length and (start <= end or sidx == 31)
    assert split_bounds(1, 32, 31)[0] > split_bounds(1, 32, 31)[1]

    rng = np.random.default_rng(457)
    dim, length = 32, 97
    q = rng.normal(0, 0.08, size=(4, dim)).astype(np.float32)
    keys = rng.integers(-127, 128, size=(length, dim), dtype=np.int8)
    values = rng.integers(-127, 128, size=(length, dim), dtype=np.int8)
    ks = rng.uniform(0.001, 0.01, size=length).astype(np.float32)
    vs = rng.uniform(0.001, 0.01, size=length).astype(np.float32)
    # Four logical rows use their own bounds across a split-size transition;
    # an artificial transition at token 20/21 exercises union staging/masking.
    row_bounds = [(0, 70), (0, 70), (31, 97), (32, 97)]
    for row, (start, end) in enumerate(row_bounds):
        scalar = sequential(q[row], keys, ks, values, vs, start, end, 1 / math.sqrt(dim))
        block = tiled(q[row], keys, ks, values, vs, start, end, 1 / math.sqrt(dim))
        assert_close(f"row{row}.state", np.array(block[:2]), np.array(scalar[:2]))
        assert_close(f"row{row}.acc", block[2], scalar[2])
        assert_close(f"row{row}.output", block[2] / block[1], scalar[2] / scalar[1])
    print("EXP-0457 fixed-operand oracle PASS; GQA rows, split bounds, masks, online state, partial tiles")


if __name__ == "__main__":
    main()
