#!/usr/bin/env python3
"""Locate the first CPU/GPU hidden-state mismatch at matching layer inputs."""

import array
import hashlib
import json
import math
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "reference"
HIDDEN = 5120
LAYERS = 64
PROMPT_TOKENS = 2048


def read_vectors(path, count):
    data = path.read_bytes()
    expected = count * HIDDEN * 4
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {len(data)}")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(x) for x in values):
        raise ValueError(f"{path}: non-finite hidden values")
    return [values[i * HIDDEN:(i + 1) * HIDDEN] for i in range(count)]


def compare(reference, actual):
    errors = [float(a) - float(b) for a, b in zip(actual, reference)]
    ref_norm = math.sqrt(sum(float(x) ** 2 for x in reference))
    actual_norm = math.sqrt(sum(float(x) ** 2 for x in actual))
    return {
        "bitwise_equal": reference.tobytes() == actual.tobytes(),
        "rmse": math.sqrt(sum(x * x for x in errors) / HIDDEN),
        "max_abs_error": max(map(abs, errors)),
        "relative_l2": math.sqrt(sum(x * x for x in errors)) / ref_norm,
        "reference_l2": ref_norm,
        "actual_l2": actual_norm,
    }


def compare_rows(cpu_path, gpu_path):
    cpu = read_vectors(cpu_path, PROMPT_TOKENS)
    gpu = read_vectors(gpu_path, PROMPT_TOKENS)
    rows = [compare(cpu[i], gpu[i]) for i in range(PROMPT_TOKENS)]
    first = next((i for i, row in enumerate(rows) if not row["bitwise_equal"]), None)
    return {
        "cpu_sha256": hashlib.sha256(cpu_path.read_bytes()).hexdigest(),
        "gpu_sha256": hashlib.sha256(gpu_path.read_bytes()).hexdigest(),
        "first_mismatching_position": first,
        "mismatching_rows": sum(not row["bitwise_equal"] for row in rows),
        "first_mismatch_metrics": rows[first] if first is not None else None,
        "last_position_metrics": rows[-1],
    }


def main():
    cpu_logits = REFERENCE / "llama-cpu-layertrace.run1.logits.f32"
    frozen_cpu_logits = REFERENCE / "llama-cpu.run1.logits.f32"
    if hashlib.sha256(cpu_logits.read_bytes()).digest() != hashlib.sha256(frozen_cpu_logits.read_bytes()).digest():
        raise ValueError("trace-enabled CPU replay changed the frozen oracle logits")
    cpu = read_vectors(REFERENCE / "llama-cpu-layertrace.layers.f32", LAYERS)
    gpu_all = read_vectors(REFERENCE / "miinfer-fused-rowtrace.layers.f32", LAYERS + 2)

    boundaries = []
    for layer in range(LAYERS):
        label = "embedding output" if layer == 0 else f"output of layer {layer - 1}"
        boundaries.append({"next_layer_input": layer, "boundary": label,
                           **compare(cpu[layer], gpu_all[layer])})
    first = next((item for item in boundaries if not item["bitwise_equal"]), None)
    row_traces = {
        "embedding_output": compare_rows(
            REFERENCE / "llama-cpu-rowtrace.layers.f32.embedding-rows.f32",
            REFERENCE / "miinfer-fused-rowtrace.embedding-rows.f32"),
        "layer0_output": compare_rows(
            REFERENCE / "llama-cpu-rowtrace.layers.f32.layer0-output-rows.f32",
            REFERENCE / "miinfer-fused-rowtrace.layer0-output-rows.f32"),
    }
    result = {
        "schema": "m31-lc-0003-layer-trace-comparison-v1",
        "capture": "last prompt token at position 2047 after each matching layer input",
        "cpu_trace_sha256": hashlib.sha256((REFERENCE / "llama-cpu-layertrace.layers.f32").read_bytes()).hexdigest(),
        "gpu_trace_sha256": hashlib.sha256((REFERENCE / "miinfer-fused-rowtrace.layers.f32").read_bytes()).hexdigest(),
        "gpu_trace_layout": "embedding; 64 layer outputs; final RMS-normalized hidden",
        "cpu_trace_layout": "64 layer inputs",
        "first_bitwise_mismatch": first,
        "layer0_rowwise_comparison": row_traces,
        "layer_boundaries": boundaries,
        "unpaired_gpu_tail": {
            "layer63_output_l2": math.sqrt(sum(float(x) ** 2 for x in gpu_all[64])),
            "final_norm_hidden_l2": math.sqrt(sum(float(x) ** 2 for x in gpu_all[65])),
        },
    }
    (REFERENCE / "layer-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"first_bitwise_mismatch": first,
                      "layer0_rowwise_comparison": row_traces,
                      "first_eight_boundaries": boundaries[:8],
                      "last_boundary": boundaries[-1]}, indent=2))


if __name__ == "__main__":
    main()
