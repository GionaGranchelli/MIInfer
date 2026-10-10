#!/usr/bin/env python3
"""Compare matched fused/MMQ captures and one-at-a-time GDN substitutions."""

import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_projections import read_f32, sha, stats


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "results/m31-lc-0003/reference"
RUNS = ROOT / "results/m31-lc-0004/z840-causal-replay"
BASELINE = ROOT / "results/m31-lc-0004/z840-selinux-retry"
OUT = ROOT / "results/m31-lc-0004/replay-analysis.json"
VOCAB = 248320
DECISIONS = 5


def digest(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def delta_summary(reference, actual):
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(actual, dtype=np.float64)
    delta = got - ref
    order_ref = np.argsort(-np.abs(ref))[:10]
    order_got = np.argsort(-np.abs(got))[:10]
    ref_norm = float(np.linalg.norm(ref))
    return {
        "bitwise_equal": np.array_equal(np.asarray(reference, dtype="<f4"),
                                        np.asarray(actual, dtype="<f4")),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs_error": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / ref_norm) if ref_norm else None,
        "top_abs_overlap": int(len(set(order_ref.tolist()) & set(order_got.tolist()))),
    }


def load_op(prefix, name, root=RUNS):
    return np.fromfile(root / f"{prefix}-optrace.gpu.{name}.f32", dtype="<f4")


def load_logits(prefix, root=RUNS):
    values = read_f32(root / f"{prefix}.logits.f32", DECISIONS * VOCAB)
    return values.reshape(DECISIONS, VOCAB)


def read_tokens(prefix, root=RUNS):
    text = (root / f"{prefix}.metrics").read_text()
    row = next(line[len("result="):] for line in text.splitlines()
               if line.startswith("result="))
    return json.loads(row)["token_ids"]


def logits_summary(reference, actual):
    rows = []
    for index, (ref, got) in enumerate(zip(reference, actual)):
        order_ref = np.argsort(-ref)
        order_got = np.argsort(-got)
        delta = got.astype(np.float64) - ref.astype(np.float64)
        rows.append({
            "decision": index + 1,
            "reference_top1": int(order_ref[0]),
            "actual_top1": int(order_got[0]),
            "reference_top1_margin": float(ref[order_ref[0]] - ref[order_ref[1]]),
            "actual_top1_margin": float(got[order_got[0]] - got[order_got[1]]),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "max_abs_error": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(ref)),
        })
    return rows


def main():
    cpu_input_path = REFERENCE / "llama-cpu-lc4-optrace.cpu.scalar-input.f32"
    cpu_input = read_f32(cpu_input_path, 5120)
    inputs = {}
    for route in ("fused", "mmq"):
        path = BASELINE / f"{route}-optrace.gpu.attn_norm.f32"
        values = read_f32(path, 5120)
        if values.tobytes() != cpu_input.tobytes():
            raise ValueError(f"{route} layer-0 input differs from the frozen CPU input")
        inputs[route] = {"sha256": sha(path), "bitwise_equal_to_cpu": True}
    if inputs["fused"]["sha256"] != inputs["mmq"]["sha256"]:
        raise ValueError("fused and MMQ layer-0 inputs differ")

    stage_order = (
        "state_predelta", "attn_norm", "beta", "alpha", "beta_sigmoid", "gate",
        "linear_attn_qkv_mixed", "z", "q_conv", "k_conv", "v_conv",
        "q_conv_predelta", "k_conv_predelta", "gdn_raw_output", "final_output",
        "linear_attn_out", "attn_residual", "attn_post_norm", "ffn_gate", "ffn_up",
        "ffn_activation", "ffn_out", "post_ffn",
    )
    route_stages = {}
    for stage in stage_order:
        fused = load_op("fused", stage, BASELINE)
        mmq = load_op("mmq", stage, BASELINE)
        if fused.size != mmq.size or fused.size == 0:
            raise ValueError(f"missing or shape-mismatched route capture for {stage}")
        route_stages[stage] = delta_summary(fused, mmq)
    first_route_difference = next(
        (stage for stage in stage_order if not route_stages[stage]["bitwise_equal"]), None)
    fused_route_logits = load_logits("fused", BASELINE)
    mmq_route_logits = load_logits("mmq", BASELINE)
    route_tokens = {route: read_tokens(route, BASELINE) for route in ("fused", "mmq")}
    for route, logits in (("fused", fused_route_logits), ("mmq", mmq_route_logits)):
        if np.argmax(logits, axis=1).astype(int).tolist() != route_tokens[route]:
            raise ValueError(f"raw-logit argmax does not match generated IDs for {route}")

    projections = {}
    for name, count, ref_name in (
        ("beta", 48, "m31-lc4-beta.reference.f32"),
        ("linear_attn_qkv_mixed", 10240, "m31-lc4-qkv.reference.f32"),
    ):
        reference = read_f32(REFERENCE / ref_name, count)
        fused = read_f32(BASELINE / f"fused-optrace.gpu.{name}.f32", count)
        mmq = read_f32(BASELINE / f"mmq-optrace.gpu.{name}.f32", count)
        if fused.tobytes() != mmq.tobytes():
            raise ValueError(f"fused and MMQ {name} projections differ")
        projections[name] = {
            "reference_sha256": sha(REFERENCE / ref_name),
            "fused_vs_reference": stats(reference, fused),
            "mmq_vs_reference": stats(reference, mmq),
            "fused_vs_mmq": delta_summary(fused, mmq),
        }

    replacements = {
        "beta": ("beta", "beta"),
        "linear_attn_qkv_mixed": ("qkv", "linear_attn_qkv_mixed"),
    }
    downstream = {}
    for projection, (suffix, target_op) in replacements.items():
        ref_name = ("m31-lc4-beta.reference.f32" if projection == "beta"
                    else "m31-lc4-qkv.reference.f32")
        expected = read_f32(REFERENCE / ref_name, 48 if projection == "beta" else 10240)
        stages = ("beta_sigmoid", "gate", "linear_attn_qkv_mixed", "q_conv", "k_conv",
                  "v_conv", "gdn_raw_output", "final_output", "linear_attn_out",
                  "attn_residual", "attn_post_norm", "ffn_gate", "ffn_up",
                  "ffn_activation", "ffn_out", "post_ffn")
        first = stages.index("beta_sigmoid") if projection == "beta" else stages.index("q_conv")
        for route in ("fused", "mmq"):
            prefix = f"{route}-{suffix}"
            captured = read_f32(RUNS / f"{prefix}-optrace.gpu.{projection}.f32", expected.size)
            if captured.tobytes() != expected.tobytes():
                raise ValueError(f"{prefix} did not capture the exact reference vector")
            baseline_prefix = f"{route}-hookbase"
            baseline_input = load_op(baseline_prefix, "attn_norm")
            substituted_input = load_op(prefix, "attn_norm")
            if baseline_input.tobytes() != substituted_input.tobytes():
                raise ValueError(f"{prefix} changed its layer-0 input")
            downstream[prefix] = {
                "substitution": target_op,
                "substituted_output_sha256": digest(captured),
                "stages_vs_route_baseline": {},
            }
            for stage in stages[first:]:
                baseline = load_op(baseline_prefix, stage)
                actual = load_op(prefix, stage)
                if baseline.size != actual.size or baseline.size == 0:
                    raise ValueError(f"missing or shape-mismatched {stage} trace for {prefix}")
                downstream[prefix]["stages_vs_route_baseline"][stage] = delta_summary(baseline, actual)
            baseline_logits = load_logits(baseline_prefix)
            actual_logits = load_logits(prefix)
            tokens = read_tokens(prefix)
            argmax = np.argmax(actual_logits, axis=1).astype(int).tolist()
            if argmax != tokens:
                raise ValueError(f"raw-logit argmax does not match generated IDs for {prefix}")
            downstream[prefix]["baseline_token_ids"] = read_tokens(baseline_prefix)
            downstream[prefix]["generated_token_ids"] = tokens
            downstream[prefix]["logits_vs_route_baseline"] = logits_summary(baseline_logits, actual_logits)

    baseline_tokens = read_tokens("fused-hookbase")
    result = {
        "schema": "m31-lc-0004-causal-replay-v1",
        "case": {"model_sha256": "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
                 "prompt_tokens": 2048, "context_capacity_tokens": 2304,
                 "prompt_fingerprint_fnv1a64": "a998afbe8b675816",
                 "prompt_ids_sha256": "5acd37f00d9e670f4b4e06d46fa12349e1dee06e356e6e73a186d74672f5bb47",
                 "forced_history": [220, 248046, 198, 248045],
                 "gpu_bdf": "0000:06:00.0", "image": "miinfer-dev:rocm-7.2.1"},
        "layer0_input": inputs,
        "fused_vs_mmq_stage_comparison": {
            "scope": "position-0 vectors from the first 512-token layer-0 prefill tile",
            "first_non_bitwise_stage": first_route_difference,
            "stages": route_stages,
        },
        "fused_vs_mmq_logits": logits_summary(fused_route_logits, mmq_route_logits),
        "projections": projections,
        "fused_baseline_token_ids": baseline_tokens,
        "fused_vs_mmq_token_ids": {
            "fused": route_tokens["fused"], "mmq": route_tokens["mmq"],
        },
        "one_at_a_time_substitutions": downstream,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "inputs_identical_to_cpu": all(item["bitwise_equal_to_cpu"] for item in inputs.values()),
        "projections": {name: {"fused_vs_reference": {
            key: value for key, value in data["fused_vs_reference"].items()
            if key in ("bitwise_equal", "rmse", "max_abs_error", "relative_l2", "top_abs_overlap")},
            "fused_vs_mmq": data["fused_vs_mmq"]} for name, data in projections.items()},
        "token_ids": result["fused_vs_mmq_token_ids"],
        "first_non_bitwise_route_stage": first_route_difference,
        "substitutions": {name: {
            "token_ids": data["generated_token_ids"],
            "baseline_token_ids": data["baseline_token_ids"],
            "stage_differences": {stage: values for stage, values in data["stages_vs_route_baseline"].items()
                                  if not values["bitwise_equal"]},
            "logits": data["logits_vs_route_baseline"],
        } for name, data in downstream.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
