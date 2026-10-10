#!/usr/bin/env python3
"""Compare the first five pre-sampling F32 logit vectors from an M31 pair."""

import argparse
import array
import hashlib
import heapq
import json
import math
import sys
from pathlib import Path


def read_capture(path: Path, vocab_size: int) -> list[array.array]:
    expected = 5 * vocab_size * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {path.stat().st_size}")
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, 5 * vocab_size)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{path}: capture contains a non-finite logit")
    return [values[i * vocab_size:(i + 1) * vocab_size] for i in range(5)]


def softmax(logits: array.array) -> array.array:
    maximum = max(logits)
    weights = array.array("d", (math.exp(value - maximum) for value in logits))
    total = sum(weights)
    for i in range(len(weights)):
        weights[i] /= total
    return weights


def top(logits: array.array, probabilities: array.array, count: int = 10) -> list[dict]:
    return [{"token": token, "logit": value, "probability": probabilities[token]}
            for value, token in heapq.nlargest(count, ((value, i) for i, value in enumerate(logits)))]


def cosine(left, right) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def compare(fused: array.array, mmq: array.array) -> dict:
    fused_mean = sum(fused) / len(fused)
    mmq_mean = sum(mmq) / len(mmq)
    centered_fused = array.array("d", (v - fused_mean for v in fused))
    centered_mmq = array.array("d", (v - mmq_mean for v in mmq))
    errors = [b - a for a, b in zip(fused, mmq)]
    centered_errors = [b - a for a, b in zip(centered_fused, centered_mmq)]
    fused_prob = softmax(fused)
    mmq_prob = softmax(mmq)
    fused_top = top(fused, fused_prob)
    mmq_top = top(mmq, mmq_prob)
    fused_ids = {item["token"] for item in fused_top}
    mmq_ids = {item["token"] for item in mmq_top}
    js = 0.0
    total_variation = 0.0
    for p, q in zip(fused_prob, mmq_prob):
        midpoint = (p + q) / 2
        if p:
            js += 0.5 * p * math.log(p / midpoint)
        if q:
            js += 0.5 * q * math.log(q / midpoint)
        total_variation += abs(p - q)
    top1_fused, top2_fused = fused_top[:2]
    top1_mmq, top2_mmq = mmq_top[:2]
    disputed = (271, 74455)
    return {
        "fused_top1": top1_fused,
        "fused_top2_margin": top1_fused["logit"] - top2_fused["logit"],
        "mmq_top1": top1_mmq,
        "mmq_top2_margin": top1_mmq["logit"] - top2_mmq["logit"],
        "top10_fused": fused_top,
        "top10_mmq": mmq_top,
        "top10_overlap_count": len(fused_ids & mmq_ids),
        "top10_probability_mass_fused": sum(fused_prob[i] for i in fused_ids),
        "top10_probability_mass_mmq": sum(mmq_prob[i] for i in mmq_ids),
        "disputed_tokens": {
            str(token): {
                "fused_logit": fused[token], "mmq_logit": mmq[token],
                "fused_probability": fused_prob[token], "mmq_probability": mmq_prob[token],
            } for token in disputed
        },
        "ordinary_logit_cosine": cosine(fused, mmq),
        "centered_logit_cosine": cosine(centered_fused, centered_mmq),
        "mean_signed_logit_error_mmq_minus_fused": sum(errors) / len(errors),
        "mean_absolute_logit_error": sum(abs(x) for x in errors) / len(errors),
        "rms_logit_error": math.sqrt(sum(x * x for x in errors) / len(errors)),
        "max_absolute_logit_error": max(abs(x) for x in errors),
        "mean_absolute_centered_logit_error": sum(abs(x) for x in centered_errors) / len(centered_errors),
        "rms_centered_logit_error": math.sqrt(sum(x * x for x in centered_errors) / len(centered_errors)),
        "max_absolute_centered_logit_error": max(abs(x) for x in centered_errors),
        "stable_softmax_js_nats": js,
        "stable_softmax_total_variation": total_variation / 2,
        "raw_vectors_bitwise_equal": fused.tobytes() == mmq.tobytes(),
        "rank_material_divergence": top1_fused["token"] != top1_mmq["token"] or fused_ids != mmq_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
    if sys.byteorder != "little":
        raise SystemExit("capture format is native little-endian F32; analyze on a little-endian host")
    capture_dir = args.capture_dir
    benchmark = json.loads((capture_dir / "benchmark.json").read_text())
    runs = {run["route"]: run for run in benchmark["runs"]}
    if set(runs) != {"control", "mmq_only"}:
        raise ValueError("expected one control and one mmq_only capture")
    for route, run in runs.items():
        if (run["result"].get("tokens_valid") != "PASS"
                or run["result"].get("used_hip_graph") is not True
                or run["raw_logits_capture"].get("valid") is not True):
            raise ValueError(f"{route}: run or raw-logit capture failed validation")
    control_tokens = runs["control"]["generated_token_ids"]
    mmq_tokens = runs["mmq_only"]["generated_token_ids"]
    if control_tokens[:4] != [220, 248046, 198, 248045] or mmq_tokens[:4] != control_tokens[:4]:
        raise ValueError("captured pair does not have the established common four-token prefix")
    if (control_tokens[4], mmq_tokens[4]) != (271, 74455):
        raise ValueError("captured fifth-token mismatch differs from the established divergence")
    vocab_size = int(runs["control"]["result"]["raw_logits_vocab_size"])
    if int(runs["mmq_only"]["result"]["raw_logits_vocab_size"]) != vocab_size:
        raise ValueError("route vocabulary sizes differ")
    routes = {}
    for route in ("control", "mmq_only"):
        path = capture_dir / f"{route}-pair-1.logits.f32"
        records = read_capture(path, vocab_size)
        routes[route] = {"file": path.name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "decisions": records}
    analysis = []
    for i, (fused, mmq) in enumerate(zip(routes["control"]["decisions"], routes["mmq_only"]["decisions"]), 1):
        analysis.append({"decision": i, **compare(fused, mmq)})
    raw_first = next((item["decision"] for item in analysis if not item["raw_vectors_bitwise_equal"]), None)
    rank_first = next((item["decision"] for item in analysis if item["rank_material_divergence"]), None)
    result = {
        "schema": "m31-lc-0002-logit-analysis-v1",
        "source_sha": benchmark["source_identity"]["source_sha"],
        "binary_sha256": benchmark["source_identity"]["binary_sha256"],
        "vocab_size": vocab_size,
        "top10_materiality_definition": "first decision where route top-10 token sets or winners differ",
        "capture_validation": "PASS: same benchmark binary, HIP Graph active, valid five-decision files, reproduced four-token prefix and fifth-token split",
        "first_bitwise_raw_logit_difference_decision": raw_first,
        "first_rank_material_divergence_decision": rank_first,
        "decision_analysis": analysis,
    }
    (capture_dir / "logit-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "decision_analysis"}, indent=2))
    for item in analysis:
        print(json.dumps({key: value for key, value in item.items()
                          if key not in ("top10_fused", "top10_mmq", "disputed_tokens")}, sort_keys=True))


if __name__ == "__main__":
    main()
