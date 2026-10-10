#!/usr/bin/env python3
"""Compare captured first-token layer-0 operations against llama.cpp CPU."""

import array
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "reference"
CPU = ROOT / "llama-cpu-optrace"
GPU = ROOT / "miinfer-optrace4"
STAGES = [
    ("initial recurrent state", "state_predelta", "state_predelta", 128 * 128 * 48),
    ("attention RMS norm", "attn_norm", "attn_norm", 5120),
    ("beta projection", "beta", "beta", 48),
    ("alpha projection", "alpha", "alpha", 48),
    ("QKV projection", "linear_attn_qkv_mixed", "linear_attn_qkv_mixed", 10240),
    ("z projection", "z", "z", 6144),
    ("beta sigmoid", "beta_sigmoid", "beta_sigmoid", 48),
    ("Q convolution output", "q_conv", "q_conv", 2048),
    ("K convolution output", "k_conv", "k_conv", 2048),
    ("V convolution output", "v_conv_predelta", "v_conv", 6144),
    ("Q head normalization", "q_conv_predelta", "q_conv_predelta", 2048),
    ("K head normalization", "k_conv_predelta", "k_conv_predelta", 2048),
    ("GDN output after postprocess", "final_output", "final_output", 6144),
    ("attention output projection", "linear_attn_out", "linear_attn_out", 5120),
    ("attention residual", "attn_residual", "attn_residual", 5120),
    ("post-attention RMS norm", "attn_post_norm", "attn_post_norm", 5120),
    ("FFN down projection", "ffn_out", "ffn_out", 5120),
    ("layer output", "l_out", "post_ffn", 5120),
]


def load(path, count):
    raw = path.read_bytes()
    if len(raw) != count * 4:
        raise ValueError(f"{path}: expected {count * 4} bytes, got {len(raw)}")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(x) for x in values):
        raise ValueError(f"{path}: non-finite values")
    return raw, values


def compare(label, cpu_name, gpu_name, count):
    cpu_path = CPU.with_name(f"llama-cpu-optrace.cpu.{cpu_name}-0.f32")
    gpu_path = GPU.with_name(f"{GPU.name}.gpu.{gpu_name}.f32")
    if not cpu_path.is_file() or not gpu_path.is_file():
        return {"stage": label, "available": False,
                "missing": [str(p.name) for p in (cpu_path, gpu_path) if not p.is_file()]}
    cpu_raw, cpu = load(cpu_path, count)
    gpu_raw, gpu = load(gpu_path, count)
    errors = [float(b) - float(a) for a, b in zip(cpu, gpu)]
    norm = math.sqrt(sum(float(x) ** 2 for x in cpu))
    return {"stage": label, "available": True,
            "cpu_sha256": hashlib.sha256(cpu_raw).hexdigest(),
            "gpu_sha256": hashlib.sha256(gpu_raw).hexdigest(),
            "bitwise_equal": cpu_raw == gpu_raw,
            "rmse": math.sqrt(sum(x * x for x in errors) / count),
            "max_abs_error": max(map(abs, errors)),
            "relative_l2": math.sqrt(sum(x * x for x in errors)) / norm if norm else None}


def compare_to_scalar(name, count):
    cpu_path = CPU.with_name(f"llama-cpu-optrace.cpu.{name}-0.f32")
    gpu_path = GPU.with_name(f"{GPU.name}.gpu.{name}.f32")
    scalar_path = ROOT / f"llama-cpu-scalar4-optrace.cpu.scalar-{name}-0.f32"
    cpu_raw, cpu = load(cpu_path, count)
    gpu_raw, gpu = load(gpu_path, count)
    scalar_raw, scalar = load(scalar_path, count)

    def errors(actual):
        delta = [float(x) - float(ref) for ref, x in zip(scalar, actual)]
        norm = math.sqrt(sum(float(x) ** 2 for x in scalar))
        return {"rmse": math.sqrt(sum(x * x for x in delta) / count),
                "max_abs_error": max(map(abs, delta)),
                "relative_l2": math.sqrt(sum(x * x for x in delta)) / norm if norm else None}

    return {"projection": name, "scalar_reference": "GGUF weights dequantized to F32; FP64 dot; rounded to F32",
            "scalar_sha256": hashlib.sha256(scalar_raw).hexdigest(),
            "cpu_sha256": hashlib.sha256(cpu_raw).hexdigest(),
            "gpu_sha256": hashlib.sha256(gpu_raw).hexdigest(),
            "cpu_vs_scalar": errors(cpu), "gpu_vs_scalar": errors(gpu)}


def first_row_matches(path, values):
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if len(raw) != 2048 * 5120 * 4:
        raise ValueError(f"{path}: unexpected row-trace length")
    row = array.array("f")
    row.frombytes(raw[:5120 * 4])
    if sys.byteorder != "little":
        row.byteswap()
    return row.tobytes() == values.tobytes()


def main():
    cpu_logits = ROOT / "llama-cpu-optrace.run1.logits.f32"
    gpu_logits = GPU.with_name(f"{GPU.name}.logits.f32")
    frozen_cpu = ROOT / "llama-cpu.run1.logits.f32"
    frozen_fused = ROOT / "fused-rowtrace.logits.f32"
    if hashlib.sha256(cpu_logits.read_bytes()).digest() != hashlib.sha256(
            frozen_cpu.read_bytes()).digest():
        raise ValueError("CPU operation trace replay changed the frozen CPU logits")
    scalar_logits = ROOT / "llama-cpu-scalar4-optrace.run1.logits.f32"
    if scalar_logits.is_file() and hashlib.sha256(scalar_logits.read_bytes()).digest() != hashlib.sha256(
            frozen_cpu.read_bytes()).digest():
        raise ValueError("CPU scalar replay changed the frozen CPU logits")
    gpu_logit_sha = hashlib.sha256(gpu_logits.read_bytes()).hexdigest() if gpu_logits.exists() else None
    if gpu_logit_sha and frozen_fused.exists() and gpu_logit_sha != hashlib.sha256(
            frozen_fused.read_bytes()).hexdigest():
        raise ValueError("GPU operation trace replay changed the frozen fused logits")
    _, cpu_norm = load(CPU.with_name("llama-cpu-optrace.cpu.attn_norm-0.f32"), 5120)
    _, scalar_input = load(ROOT / "llama-cpu-scalar4-optrace.cpu.scalar-input.f32", 5120)
    if cpu_norm.tobytes() != scalar_input.tobytes():
        raise ValueError("scalar replay input differs from the CPU operation capture")
    results = [compare(*stage) for stage in STAGES]
    first = next((r for r in results if r.get("available") and not r["bitwise_equal"]), None)
    if any(not r.get("available") for r in results):
        missing = [r for r in results if not r.get("available")]
        raise ValueError("operation capture is incomplete: " + json.dumps(missing))
    _, gpu_layer_out = load(GPU.with_name(f"{GPU.name}.gpu.post_ffn.f32"), 5120)
    _, cpu_layer_out = load(CPU.with_name("llama-cpu-optrace.cpu.l_out-0.f32"), 5120)
    gpu_row_match = first_row_matches(ROOT / "miinfer-fused-rowtrace.layer0-output-rows.f32", gpu_layer_out)
    cpu_row_match = first_row_matches(ROOT / "llama-cpu-rowtrace.layers.f32.layer0-output-rows.f32", cpu_layer_out)
    report = {"schema": "m31-lc-0003-gdn-operation-comparison-v1",
              "capture": "layer 0, first 512-token prefill tile, position 0",
              "cpu_logits_sha256": hashlib.sha256(cpu_logits.read_bytes()).hexdigest(),
              "gpu_logits_sha256": gpu_logit_sha,
              "scalar_input_sha256": hashlib.sha256(scalar_input.tobytes()).hexdigest(),
              "gpu_post_ffn_matches_rowtrace_position0": gpu_row_match,
              "cpu_layer_output_matches_rowtrace_position0": cpu_row_match,
              "scalar_input_matches_cpu_norm": True,
              "scalar_projection_comparisons": [
                  compare_to_scalar("beta", 48), compare_to_scalar("alpha", 48),
                  compare_to_scalar("linear_attn_qkv_mixed", 10240)],
              "first_bitwise_mismatch": first, "stages": results}
    out = ROOT / "gdn-operation-comparison.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"first_bitwise_mismatch": first, "stages": results}, indent=2))


if __name__ == "__main__":
    main()
