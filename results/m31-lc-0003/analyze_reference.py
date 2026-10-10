#!/usr/bin/env python3
"""Compare the pinned CPU oracle with the two frozen M31-LC-0002 GPU routes."""

import array
import hashlib
import heapq
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "results/m31-lc-0002/capture"
REFERENCE = ROOT / "results/m31-lc-0003/reference"
VOCAB = 248320
DECISIONS = 5
FLOOR = 1e-3


def read_logits(path):
    expected = VOCAB * DECISIONS * 4
    data = path.read_bytes()
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {len(data)}")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(x) for x in values):
        raise ValueError(f"{path}: non-finite logits")
    return [values[i * VOCAB:(i + 1) * VOCAB] for i in range(DECISIONS)]


def rank_of(values, token):
    winner = values[token]
    return 1 + sum(x > winner or (x == winner and i < token)
                   for i, x in enumerate(values))


def compare(route, reference):
    errors = [float(a) - float(b) for a, b in zip(route, reference)]
    ref_norm_sq = sum(float(x) ** 2 for x in reference)
    included = [abs(err / float(ref)) for err, ref in zip(errors, reference)
                if abs(ref) >= FLOOR]
    ref_top = heapq.nlargest(10, range(VOCAB), key=reference.__getitem__)
    route_top = heapq.nlargest(10, range(VOCAB), key=route.__getitem__)
    ref_winner = max(range(VOCAB), key=reference.__getitem__)
    route_winner = max(range(VOCAB), key=route.__getitem__)
    return {
        "rmse": math.sqrt(sum(x * x for x in errors) / VOCAB),
        "max_abs_error": max(map(abs, errors)),
        "relative_l2": math.sqrt(sum(x * x for x in errors) / ref_norm_sq),
        "mean_elementwise_relative_abs_error_ge_1e-3": sum(included) / len(included),
        "max_elementwise_relative_abs_error_ge_1e-3": max(included),
        "elementwise_relative_included": len(included),
        "elementwise_relative_excluded_fraction": 1 - len(included) / VOCAB,
        "top10_overlap": len(set(ref_top) & set(route_top)),
        "reference_argmax": ref_winner,
        "route_argmax": route_winner,
        "reference_winner_rank_in_route": rank_of(route, ref_winner),
        "reference_top10": ref_top,
        "route_top10": route_top,
        "vectors_bitwise_equal": route.tobytes() == reference.tobytes(),
    }


def main():
    prompt_path = CAPTURE / "prompt.ids"
    prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    expected_prompt_sha = "5acd37f00d9e670f4b4e06d46fa12349e1dee06e356e6e73a186d74672f5bb47"
    if prompt_sha != expected_prompt_sha:
        raise ValueError("frozen M31-LC-0002 prompt ID hash mismatch")
    forced = [int(x) for x in (ROOT / "results/m31-lc-0003/forced-history.ids")
              .read_text().split()]
    if forced != [220, 248046, 198, 248045]:
        raise ValueError("forced token history differs from frozen M31-LC-0002 prefix")
    metadata = json.loads((REFERENCE / "llama-cpu.metadata.json").read_text())
    if (metadata["revision"] != "91c631b21d6e5d09e9c6659efdf6baeef5a44ddb"
            or metadata["prompt_tokens"] != 2048 or metadata["decisions"] != DECISIONS
            or metadata["vocab_size"] != VOCAB or metadata["layers"] != 64
            or metadata["embedding_size"] != 5120 or metadata["context"] != 2304
            or metadata["prompt_tile"] != 512 or metadata["threads"] != 24
            or metadata["gpu_layers"] != 0 or metadata["kv_type_k"] != "f16"
            or metadata["kv_type_v"] != "f16"
            or metadata["forced_history"] != [220, 248046, 198, 248045]
            or metadata["bitwise_reproducible"] is not True):
        raise ValueError("CPU oracle metadata does not match the frozen replay protocol")

    oracle_paths = [REFERENCE / f"llama-cpu.run{i}.logits.f32" for i in (1, 2)]
    oracle_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in oracle_paths]
    if oracle_hashes[0] != oracle_hashes[1]:
        raise ValueError("fresh CPU oracle captures are not bitwise reproducible")
    cpu = read_logits(oracle_paths[0])

    benchmark = json.loads((CAPTURE / "benchmark.json").read_text())
    runs = {run["route"]: run for run in benchmark["runs"]}
    if set(runs) != {"control", "mmq_only"}:
        raise ValueError("expected frozen control and mmq_only captures")
    common_prefix = [220, 248046, 198, 248045]
    for name, run in runs.items():
        if (run["result"].get("tokens_valid") != "PASS"
                or run["result"].get("used_hip_graph") is not True
                or run["raw_logits_capture"].get("valid") is not True
                or run["generated_token_ids"][:4] != common_prefix
                or run["result"].get("raw_logits_vocab_size") != VOCAB):
            raise ValueError(f"{name}: GPU capture failed frozen-input validation")

    comparisons = {}
    for name, run in runs.items():
        path = CAPTURE / f"{name}-pair-1.logits.f32"
        expected_sha = run["raw_logits_capture"]["sha256"]
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha != actual_sha:
            raise ValueError(f"{name}: frozen GPU capture hash mismatch")
        gpu = read_logits(path)
        comparisons[name] = {
            "capture_sha256": actual_sha,
            "decisions": [compare(gpu[i], cpu[i]) for i in range(DECISIONS)],
        }

    first_mismatch = {
        name: next((i + 1 for i, d in enumerate(data["decisions"])
                    if not d["vectors_bitwise_equal"]), None)
        for name, data in comparisons.items()
    }
    result = {
        "schema": "m31-lc-0003-cpu-oracle-comparison-v1",
        "cpu_revision": metadata["revision"],
        "cpu_capture_sha256": oracle_hashes[0],
        "model_sha256": "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
        "prompt_ids_sha256": prompt_sha,
        "forced_history": forced,
        "elementwise_relative_floor": FLOOR,
        "relative_error_policy": "absolute reference logits below 1e-3 excluded; excluded fraction reported",
        "cpu_argmax_tokens": metadata["run1_top1"],
        "first_bitwise_logit_mismatch_decision": first_mismatch,
        "comparisons": comparisons,
    }
    out = REFERENCE / "comparison.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cpu_argmax_tokens": result["cpu_argmax_tokens"],
                      "first_bitwise_logit_mismatch_decision": first_mismatch,
                      "routes": {name: [{k: d[k] for k in (
                          "rmse", "max_abs_error", "relative_l2", "top10_overlap",
                          "reference_argmax", "route_argmax",
                          "reference_winner_rank_in_route")}
                                       for d in route["decisions"]]
                                 for name, route in comparisons.items()}}, indent=2))


if __name__ == "__main__":
    main()
