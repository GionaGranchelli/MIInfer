#!/usr/bin/env python3
"""Compare teacher-forced route logits, greedy sensitivity, and top-five margins."""

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
        raise ValueError(f"{path}: non-finite logits")
    return values


def top(values, count=5):
    return sorted(range(len(values)), key=values.__getitem__, reverse=True)[:count]


def error_metrics(left, right):
    error = sum((a - b) ** 2 for a, b in zip(left, right))
    left_sq = sum(a * a for a in left)
    right_sq = sum(b * b for b in right)
    dot = sum(a * b for a, b in zip(left, right))
    return (max(abs(a - b) for a, b in zip(left, right)),
            math.sqrt(error / len(left)), math.sqrt(error / left_sq),
            dot / math.sqrt(left_sq * right_sq))


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-m26cq-logits.py LEGACY_PREFIX INTERACTIVE_PREFIX")
    left_prefix, right_prefix = sys.argv[1:]
    for position in POSITIONS:
        left = read(f"{left_prefix}.pos-{position}.f32")
        right = read(f"{right_prefix}.pos-{position}.f32")
        if len(left) != len(right):
            raise ValueError("route vocabularies differ")
        ltop, rtop = top(left), top(right)
        maximum, rms, relative_l2, cosine = error_metrics(left, right)
        left_margin = left[ltop[0]] - left[ltop[1]]
        right_margin = right[rtop[0]] - right[rtop[1]]
        print(f"position={position} legacy_top5={ltop} interactive_top5={rtop} "
              f"legacy_logits={left[ltop[0]]:.9g},{left[ltop[1]]:.9g} "
              f"interactive_logits={right[rtop[0]]:.9g},{right[rtop[1]]:.9g} "
              f"legacy_margin={left_margin:.9g} interactive_margin={right_margin:.9g} "
              f"max_abs={maximum:.9g} rms={rms:.9g} "
              f"relative_l2={relative_l2:.9g} cosine={cosine:.9g} "
              f"top1_equal={ltop[0] == rtop[0]}")
        left_norm = read(f"{left_prefix}.pos-{position}.final-norm.f32")
        right_norm = read(f"{right_prefix}.pos-{position}.final-norm.f32")
        if len(left_norm) != len(right_norm):
            raise ValueError("final normalized hidden sizes differ")
        maximum, rms, relative_l2, cosine = error_metrics(left_norm, right_norm)
        print(f"position={position} final_norm_max_abs={maximum:.9g} "
              f"final_norm_rms={rms:.9g} final_norm_relative_l2={relative_l2:.9g} "
              f"final_norm_cosine={cosine:.9g}")


if __name__ == "__main__":
    main()
