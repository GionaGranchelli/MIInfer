#!/usr/bin/env python3
"""Report LC-0009 matched-logit, greedy-token, and repeat metrics without thresholds."""

import argparse
import array
import hashlib
import heapq
import json
import math
import sys
from pathlib import Path


VOCAB = 248320
DECISIONS = 5
GREEDY_TOKENS = 64


def read_logits(path: Path) -> list[array.array]:
    data = path.read_bytes()
    expected = VOCAB * DECISIONS * 4
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {len(data)}")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{path}: non-finite logits")
    return [values[i * VOCAB:(i + 1) * VOCAB] for i in range(DECISIONS)]


def read_ids(path: Path) -> list[int]:
    ids = [int(value) for value in path.read_text().split()]
    if any(value < 0 or value >= VOCAB for value in ids):
        raise ValueError(f"{path}: token outside model vocabulary")
    return ids


def rank(values: array.array, token: int) -> int:
    value = values[token]
    return 1 + sum(other > value or (other == value and i < token)
                   for i, other in enumerate(values))


def decision_metrics(actual: array.array, reference: array.array) -> dict:
    if len(actual) != len(reference):
        raise ValueError("logit vector lengths differ")
    squared_error = squared_reference = 0.0
    max_abs = 0.0
    for a, r in zip(actual, reference):
        delta = float(a) - float(r)
        squared_error += delta * delta
        squared_reference += float(r) * float(r)
        max_abs = max(max_abs, abs(delta))
    ref_top = heapq.nlargest(10, range(VOCAB), key=reference.__getitem__)
    actual_top = heapq.nlargest(10, range(VOCAB), key=actual.__getitem__)
    ref_winner, actual_winner = ref_top[0], actual_top[0]
    return {
        "relative_l2": math.sqrt(squared_error / squared_reference) if squared_reference else None,
        "rmse": math.sqrt(squared_error / VOCAB),
        "max_abs_error": max_abs,
        "reference_top1_margin": reference[ref_top[0]] - reference[ref_top[1]],
        "route_top1_margin": actual[actual_top[0]] - actual[actual_top[1]],
        "top5_overlap": len(set(ref_top[:5]) & set(actual_top[:5])),
        "top10_overlap": len(set(ref_top) & set(actual_top)),
        "reference_argmax": ref_winner,
        "route_argmax": actual_winner,
        "reference_winner_rank_in_route": rank(actual, ref_winner),
        "route_winner_rank_in_reference": rank(reference, actual_winner),
    }


def named_paths(values: list[str], field: str) -> dict[str, Path]:
    result = {}
    for value in values:
        name, sep, raw_path = value.partition("=")
        if not sep or not name or not raw_path or name in result:
            raise ValueError(f"invalid or duplicate {field}: {value!r}")
        result[name] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_logits", type=Path, help="five-decision CPU teacher-forced F32 capture")
    parser.add_argument("--route", action="append", default=[], metavar="NAME=FILE",
                        help="route teacher-forced capture; repeat for fused/MMQ and modes")
    parser.add_argument("--greedy", action="append", default=[], metavar="NAME=FILE",
                        help="64-token greedy ID file; include cpu=FILE and each route")
    parser.add_argument("--repeat", action="append", default=[], metavar="NAME=FILE",
                        help="additional teacher-forced repeat file; repeat three times per route")
    parser.add_argument("--output", type=Path, help="write JSON report; refuses to overwrite")
    args = parser.parse_args()

    cpu = read_logits(args.cpu_logits)
    routes = named_paths(args.route, "route")
    if not routes:
        raise ValueError("at least one --route NAME=FILE is required")
    greedy_paths = named_paths(args.greedy, "greedy capture")
    repeat_paths = {}
    for value in args.repeat:
        name, sep, raw_path = value.partition("=")
        if not sep or not name or not raw_path:
            raise ValueError(f"invalid repeat capture: {value!r}")
        repeat_paths.setdefault(name, []).append(Path(raw_path))
    result = {
        "schema": "m31-lc-0009-capture-comparison-v1",
        "cpu_capture_sha256": hashlib.sha256(args.cpu_logits.read_bytes()).hexdigest(),
        "decisions": DECISIONS,
        "vocab_size": VOCAB,
        "comparisons": {},
        "greedy_mismatches": {},
        "repeat_bitwise_equal": {},
    }
    for name, path in routes.items():
        capture = read_logits(path)
        result["comparisons"][name] = {
            "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "decisions": [decision_metrics(capture[i], cpu[i]) for i in range(DECISIONS)],
        }
    for name, paths in repeat_paths.items():
        if name not in result["comparisons"]:
            raise ValueError(f"repeat capture has no matching --route {name}=FILE")
        baseline_hash = result["comparisons"][name]["capture_sha256"]
        repeat_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        result["repeat_bitwise_equal"][name] = {
            "repeat_count": 1 + len(paths),
            "each_repeat_matches_first": all(digest == baseline_hash for digest in repeat_hashes),
            "repeat_sha256": repeat_hashes,
        }
    if greedy_paths:
        if "cpu" not in greedy_paths:
            raise ValueError("--greedy cpu=FILE is required when greedy captures are supplied")
        cpu_ids = read_ids(greedy_paths["cpu"])
        if len(cpu_ids) != GREEDY_TOKENS:
            raise ValueError(f"CPU greedy capture must contain {GREEDY_TOKENS} IDs")
        for name, path in greedy_paths.items():
            if name == "cpu":
                continue
            ids = read_ids(path)
            limit = max(len(cpu_ids), len(ids))
            result["greedy_mismatches"][name] = {
                "positions_1based": [i + 1 for i in range(limit)
                    if i >= len(cpu_ids) or i >= len(ids) or cpu_ids[i] != ids[i]],
                "cpu_tokens": len(cpu_ids),
                "route_tokens": len(ids),
            }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite {args.output}")
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, OverflowError) as error:
        raise SystemExit(f"capture comparison failed: {error}")
