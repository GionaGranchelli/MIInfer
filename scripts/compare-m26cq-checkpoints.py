#!/usr/bin/env python3
"""Run the established M26-C semantic snapshot comparator at CQ checkpoints."""

import argparse
import subprocess
import sys
from pathlib import Path


POSITIONS = (513, 520, 528, 544, 560, 565, 566, 567, 568, 569, 570, 576)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_dir")
    parser.add_argument("interactive_dir")
    parser.add_argument("output")
    args = parser.parse_args()
    comparator = Path(__file__).with_name("compare-m26c-state.py")
    with open(args.output, "w", encoding="utf-8") as log:
        for position in POSITIONS:
            log.write(f"\n=== processed position {position} (input position {position - 1}) ===\n")
            result = subprocess.run((sys.executable, str(comparator),
                str(Path(args.legacy_dir) / f"position-{position}.state"),
                str(Path(args.interactive_dir) / f"position-{position}.state")),
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(result.stdout)
            if result.returncode not in (0, 1):
                raise SystemExit(result.returncode)
    print(f"checkpoint_comparison_log={args.output} positions={len(POSITIONS)}")


if __name__ == "__main__":
    main()
