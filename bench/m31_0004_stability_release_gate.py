#!/usr/bin/env python3
"""Aggregate fixed M31 repetitions into an explicit N=1 release decision."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path


def stats(values):
    if not values:
        return {}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * .95) - 1)]
    mean = statistics.fmean(values)
    return {"median": statistics.median(values), "p95": p95, "min": min(values),
            "max": max(values), "coefficient_of_variation": statistics.pstdev(values) / mean if mean else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSON array or object with cold/warm/agent runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    runs = payload.get("runs", payload) if isinstance(payload, dict) else payload
    categories = {name: list(runs.get(name, [])) if isinstance(runs, dict) else []
                  for name in ("cold", "warm", "agent")}
    measurements = {name: stats([float(item["wall_ms"]) for item in values
                                 if isinstance(item, dict) and "wall_ms" in item])
                    for name, values in categories.items()}
    failures = [item for values in categories.values() for item in values
                if isinstance(item, dict) and item.get("status") not in (None, "PASS")]
    gate = all(len(categories[name]) >= 10 for name in categories) and not failures
    result = {"benchmark": "M31-0004", "required_repetitions": {name: 10 for name in categories},
              "runs": categories, "statistics": measurements,
              "vram_drift": payload.get("vram_drift"), "correctness_failures": failures,
              "gpu_errors": payload.get("gpu_errors", []), "oom": payload.get("oom", []),
              "state_leaks": payload.get("state_leaks", []),
              "result": "SINGLE_MI50_QUALIFIED" if gate else "SINGLE_MI50_NOT_QUALIFIED"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["result"])
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
