#!/usr/bin/env python3
"""Compare M26-C route logits with a CPU oracle and each other."""

import math
import struct
import sys


def read_logits(path):
    data = open(path, "rb").read()
    if not data or len(data) % 4:
        raise ValueError(f"{path}: expected nonempty little-endian f32 logits")
    values = list(struct.unpack("<" + "f" * (len(data) // 4), data))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: logits contain NaN or Inf")
    winner = max(range(len(values)), key=values.__getitem__)
    return values, winner


def metrics(actual, expected):
    if len(actual) != len(expected):
        raise ValueError("logit vector lengths differ")
    dot = aa = bb = error = maximum = 0.0
    for a, b in zip(actual, expected):
        dot += a * b
        aa += a * a
        bb += b * b
        delta = a - b
        error += delta * delta
        maximum = max(maximum, abs(delta))
    return dot / math.sqrt(aa * bb), maximum, math.sqrt(error / len(actual))


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: compare-m26c-logits.py CPU.f32 NAME=ROUTE.f32 [NAME=ROUTE.f32 ...]")
    try:
        reference, ref_winner = read_logits(sys.argv[1])
        routes = []
        for specification in sys.argv[2:]:
            name, separator, path = specification.partition("=")
            if not separator or not name or not path:
                raise ValueError(f"invalid route argument: {specification!r}")
            values, winner = read_logits(path)
            if len(reference) != len(values):
                raise ValueError(f"CPU/{name} vocabulary sizes differ")
            routes.append((name, values, winner))
        for name, values, winner in routes:
            cosine, max_abs, rms = metrics(values, reference)
            floor = "meets" if cosine >= 0.9995 else "below"
            print(f"{name}: argmax={winner} reference_argmax={ref_winner} "
                  f"match={winner == ref_winner} cosine={cosine:.9g} "
                  f"max_abs={max_abs:.9g} rms={rms:.9g} cosine_0.9995_floor={floor} "
                  "(position-512 diagnostic; formal M9 gate not run)")
        for i, (left_name, left, left_winner) in enumerate(routes):
            for right_name, right, right_winner in routes[i + 1:]:
                cosine, max_abs, rms = metrics(left, right)
                print(f"{left_name}_vs_{right_name}: argmax_equal={left_winner == right_winner} "
                      f"cosine={cosine:.9g} max_abs={max_abs:.9g} rms={rms:.9g}")
    except (OSError, ValueError, OverflowError) as error:
        raise SystemExit(f"logit comparison failed: {error}")


if __name__ == "__main__":
    main()
