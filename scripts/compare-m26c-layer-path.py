#!/usr/bin/env python3
"""Compare M26-C one-token layer-path captures."""

import itertools
import math
import os
import struct
import sys


FIELDS = (
    "input", "normalized", "qkv", "recurrent_output", "gated",
    "attention_residual", "post_normalized", "ffn_output", "layer_output",
    "gate.normalized", "gate.projection", "gate.recurrent_output",
    "gate.head_norm", "gate.head_scaled", "gate.gated",
)


def read(prefix, layer, field):
    path = f"{prefix}.layer{layer}.{field}.f32"
    data = open(path, "rb").read()
    if not data or len(data) % 4:
        raise ValueError(f"{path}: expected nonempty f32 data")
    values = struct.unpack("<" + "f" * (len(data) // 4), data)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: contains NaN or Inf")
    return values


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: compare-m26c-layer-path.py NAME=PREFIX NAME=PREFIX [...]")
    routes = []
    for argument in sys.argv[1:]:
        name, separator, prefix = argument.partition("=")
        if not separator or not name or not prefix:
            raise SystemExit(f"invalid route argument: {argument!r}")
        routes.append((name, prefix))
    try:
        for (left_name, left_prefix), (right_name, right_prefix) in itertools.combinations(routes, 2):
            for layer in (0, 1):
                for field in FIELDS:
                    left_path = f"{left_prefix}.layer{layer}.{field}.f32"
                    right_path = f"{right_prefix}.layer{layer}.{field}.f32"
                    if not os.path.exists(left_path) and not os.path.exists(right_path):
                        print(f"{left_name}_vs_{right_name} layer={layer} field={field} unavailable=both")
                        continue
                    left = read(left_prefix, layer, field)
                    right = read(right_prefix, layer, field)
                    if len(left) != len(right):
                        raise ValueError(f"layer {layer} {field}: element counts differ")
                    maximum = 0.0
                    square_error = 0.0
                    dot = left_norm = right_norm = 0.0
                    exact = True
                    for a, b in zip(left, right):
                        delta = a - b
                        exact &= a == b
                        maximum = max(maximum, abs(delta))
                        square_error += delta * delta
                        dot += a * b
                        left_norm += a * a
                        right_norm += b * b
                    cosine = dot / math.sqrt(left_norm * right_norm) if left_norm and right_norm else float("nan")
                    rms = math.sqrt(square_error / len(left))
                    print(f"{left_name}_vs_{right_name} layer={layer} field={field} "
                          f"elements={len(left)} exact={exact} max_abs={maximum:.9g} "
                          f"rms={rms:.9g} cosine={cosine:.9g}")
    except (OSError, ValueError, OverflowError) as error:
        raise SystemExit(f"layer-path comparison failed: {error}")


if __name__ == "__main__":
    main()
