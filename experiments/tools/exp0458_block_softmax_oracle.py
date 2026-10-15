#!/usr/bin/env python3
"""Check the four-key online-softmax/PV combine against sequential updates."""

import math
import random


def sequential(scores, values, scales, state):
    maximum, total, accumulator = state
    accumulator = accumulator[:]
    for score, value, scale in zip(scores, values, scales):
        next_maximum = max(maximum, score)
        alpha = math.exp(maximum - next_maximum)
        weight = math.exp(score - next_maximum)
        total = total * alpha + weight
        maximum = next_maximum
        accumulator = [a * alpha + weight * scale * v
                       for a, v in zip(accumulator, value)]
    return maximum, total, accumulator


def block4(scores, values, scales, state):
    old_maximum, old_total, old_accumulator = state
    tile_maximum = max(scores)
    new_maximum = max(old_maximum, tile_maximum)
    alpha = math.exp(old_maximum - new_maximum)
    weights = [math.exp(score - new_maximum) for score in scores]
    new_total = old_total * alpha + sum(weights)
    new_accumulator = [
        old_accumulator[d] * alpha
        + sum(weights[i] * scales[i] * values[i][d] for i in range(4))
        for d in range(len(old_accumulator))
    ]
    return new_maximum, new_total, new_accumulator


def check(name, scores, values, scales, state):
    expected = sequential(scores, values, scales, state)
    actual = block4(scores, values, scales, state)
    for lhs, rhs in zip(expected[:2], actual[:2]):
        assert math.isclose(lhs, rhs, rel_tol=2e-14, abs_tol=2e-14), (name, lhs, rhs)
    for lhs, rhs in zip(expected[2], actual[2]):
        assert math.isclose(lhs, rhs, rel_tol=2e-14, abs_tol=2e-14), (name, lhs, rhs)


def main():
    rng = random.Random(458)
    values = [[rng.uniform(-127, 127) for _ in range(8)] for _ in range(4)]
    scales = [rng.uniform(0.0001, 0.5) for _ in range(4)]
    cases = [
        ("first-tile", [-2.0, 1.0, 0.0, 3.0], (-math.inf, 0.0, [0.0] * 8)),
        ("old-max-wins", [-3.0, -2.0, -4.0, -1.0], (8.0, 1.25, [0.5] * 8)),
        ("new-max-wins", [2.0, 9.0, 3.0, 1.0], (0.0, 2.0, [-0.25] * 8)),
        ("ties", [4.0, 4.0, 4.0, 4.0], (4.0, 3.0, [1.0] * 8)),
    ]
    for name, scores, state in cases:
        check(name, scores, values, scales, state)

    sequential_state = (-math.inf, 0.0, [0.0] * 8)
    block_state = (-math.inf, 0.0, [0.0] * 8)
    for tile in range(128):
        scores = [rng.uniform(-20.0, 20.0) for _ in range(4)]
        values = [[rng.uniform(-127, 127) for _ in range(8)] for _ in range(4)]
        scales = [rng.uniform(0.0001, 0.5) for _ in range(4)]
        sequential_state = sequential(scores, values, scales, sequential_state)
        block_state = block4(scores, values, scales, block_state)
        for lhs, rhs in zip(sequential_state[:2], block_state[:2]):
            assert math.isclose(lhs, rhs, rel_tol=2e-12, abs_tol=2e-12), (tile, lhs, rhs)
        for lhs, rhs in zip(sequential_state[2], block_state[2]):
            assert math.isclose(lhs, rhs, rel_tol=2e-12, abs_tol=2e-12), (tile, lhs, rhs)
    print("exp0458_block_softmax_oracle=PASS deterministic_cases=4 random_tiles=128")


if __name__ == "__main__":
    main()
