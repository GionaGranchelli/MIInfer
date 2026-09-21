#!/usr/bin/env python3
"""Compare every recurrent-layer boundary captured at one teacher-forced position."""

import glob
import math
import struct
import sys

POSITIONS = (512, 519, 527, 543, 559, 564, 565, 566, 567, 568, 569, 575)


def read(path):
    data = open(path, "rb").read()
    if not data or len(data) % 4:
        raise ValueError(f"{path}: malformed F32 vector")
    values = struct.unpack("<" + "f" * (len(data) // 4), data)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: non-finite value")
    return values


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare-m26cq-layer-path.py LEGACY_PREFIX INTERACTIVE_PREFIX POSITION")
    left_prefix, right_prefix, position_arg = sys.argv[1:]
    positions = POSITIONS if position_arg == "all" else (int(position_arg),)
    for position in positions:
        left_glob = f"{left_prefix}.pos-{position}.layer*.f32"
        left_paths = sorted(glob.glob(left_glob))
        if not left_paths:
            raise ValueError(f"no captures match {left_glob}")
        for left_path in left_paths:
            suffix = left_path[len(left_prefix):]
            right_path = right_prefix + suffix
            left, right = read(left_path), read(right_path)
            if len(left) != len(right):
                raise ValueError(f"shape mismatch: {suffix}")
            error = sum((a - b) ** 2 for a, b in zip(left, right))
            norm = sum(a * a for a in left)
            maximum = max(abs(a - b) for a, b in zip(left, right))
            exact = left == right
            relative = math.sqrt(error / norm) if norm else float("nan")
            layer, field = suffix.split(".layer", 1)[1].split(".", 1)
            print(f"position={position} layer={layer} field={field.removesuffix('.f32')} "
                  f"elements={len(left)} max_abs={maximum:.9g} rms={math.sqrt(error / len(left)):.9g} "
                  f"relative_l2={relative:.9g} finite=True exact={exact}")


if __name__ == "__main__":
    main()
