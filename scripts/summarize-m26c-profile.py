#!/usr/bin/env python3
"""Summarize selected-token layer events from two M26-C direct-route logs."""

import argparse
import re


PROFILE = re.compile(
    r"layer=(\d+) kind=(recurrent|attention) family=([^ ]+).*?recorded=1 gpu_ms=([0-9.eE+-]+)"
)
TOTAL = re.compile(r"accounted_sample_ms=([0-9.eE+-]+).*?wall_ms=([0-9.eE+-]+)")


def parse(path):
    families = {}
    totals = None
    with open(path, encoding="utf-8") as log:
        for line in log:
            match = PROFILE.search(line)
            if match:
                layer, kind, family, elapsed = match.groups()
                key = (kind, family)
                values = families.setdefault(key, [0.0, 0])
                values[0] += float(elapsed)
                values[1] += 1
            match = TOTAL.search(line)
            if match:
                totals = tuple(map(float, match.groups()))
    if totals is None or not families:
        raise ValueError(f"no selected-token profile found in {path}")
    return families, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_log")
    parser.add_argument("interactive_log")
    args = parser.parse_args()
    legacy, legacy_totals = parse(args.legacy_log)
    interactive, interactive_totals = parse(args.interactive_log)
    print("family,legacy_gpu_ms,interactive_gpu_ms,delta_ms,legacy_layers,interactive_layers")
    for key in sorted(set(legacy) | set(interactive)):
        a, ac = legacy.get(key, (0.0, 0))
        b, bc = interactive.get(key, (0.0, 0))
        print(f"{key[0]}.{key[1]},{a:.6f},{b:.6f},{b-a:.6f},{ac},{bc}")
    print(f"accounted_total_ms,{legacy_totals[0]:.6f},{interactive_totals[0]:.6f},"
          f"{interactive_totals[0]-legacy_totals[0]:.6f}")
    print(f"outer_wall_ms,{legacy_totals[1]:.6f},{interactive_totals[1]:.6f},"
          f"{interactive_totals[1]-legacy_totals[1]:.6f}")


if __name__ == "__main__":
    main()
