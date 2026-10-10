#!/usr/bin/env python3
"""Independent layer-0 beta and Q6_K QKV projection references."""

import array
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results/m31-lc-0003/reference"
PREFIX = "llama-cpu-lc4-optrace"


def read_f32(path, count):
    raw = path.read_bytes()
    if len(raw) != count * 4:
        raise ValueError(f"{path}: expected {count * 4} bytes, got {len(raw)}")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite values")
    return values


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q6k_rows(raw, rows, columns):
    """Decode canonical GGUF Q6_K blocks to row-major F32, without ggml."""
    if columns % 256:
        raise ValueError("Q6_K row length must be divisible by 256")
    blocks_per_row = columns // 256
    expected = rows * blocks_per_row * 210
    if len(raw) != expected:
        raise ValueError(f"Q6_K byte count mismatch: expected {expected}, got {len(raw)}")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(rows, blocks_per_row, 210)
    ql = blocks[:, :, :128]
    qh = blocks[:, :, 128:192]
    scales = blocks[:, :, 192:208].view(np.int8).astype(np.float32)
    d = blocks[:, :, 208:210].copy().view("<f2").reshape(rows, blocks_per_row).astype(np.float32)
    out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
    for half in range(2):
        lo = half * 64
        hi = half * 32
        scale_base = half * 8
        for lane in range(32):
            h = qh[:, :, hi + lane]
            a = ql[:, :, lo + lane]
            b = ql[:, :, lo + lane + 32]
            quantized = (
                (a & 15) | ((h & 3) << 4),
                (b & 15) | (((h >> 2) & 3) << 4),
                (a >> 4) | (((h >> 4) & 3) << 4),
                (b >> 4) | (((h >> 6) & 3) << 4),
            )
            indices = (half * 128 + lane, half * 128 + lane + 32,
                       half * 128 + lane + 64, half * 128 + lane + 96)
            for group, (index, q) in enumerate(zip(indices, quantized)):
                scale_index = scale_base + group * 2 + lane // 16
                factor = d * scales[:, :, scale_index]
                out[:, :, index] = factor * (q.astype(np.int16) - 32).astype(np.float32)
    return out.reshape(rows, columns)


def stats(reference, actual):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    delta = actual - reference
    norm = np.linalg.norm(reference)
    ref_order = np.argsort(-np.abs(reference))
    actual_order = np.argsort(-np.abs(actual))
    error_order = np.argsort(-np.abs(delta))
    top = min(10, len(reference))
    ref_top = ref_order[:top]
    actual_top = actual_order[:top]
    return {
        "bitwise_equal": np.array_equal(reference.astype("<f4"), actual.astype("<f4")),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs_error": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / norm) if norm else None,
        "worst_error_channels": [
            {"index": int(i), "reference": float(reference[i]), "actual": float(actual[i]),
             "absolute_error": float(abs(delta[i]))}
            for i in error_order[:top]
        ],
        "top_abs_indices_reference": ref_top.tolist(),
        "top_abs_indices_actual": actual_top.tolist(),
        "top_abs_overlap": int(len(set(ref_top.tolist()) & set(actual_top.tolist()))),
        "reference_top_abs_rank_in_actual": [int(np.where(actual_order == i)[0][0]) + 1 for i in ref_top],
    }


def q6k_self_check():
    block = bytearray(210)
    block[192:208] = bytes([1] * 16)
    block[208:210] = np.float16(1).tobytes()
    block[0] = 0x21
    block[32] = 0x43
    values = q6k_rows(bytes(block), 1, 256)[0]
    assert values[0] == -31 and values[32] == -29
    assert values[64] == -30 and values[96] == -28
    assert np.isfinite(values).all()


def main():
    q6k_self_check()
    cpu_input_path = ROOT / f"{PREFIX}.cpu.scalar-input.f32"
    gpu_input_path = ROOT / "miinfer-optrace4.gpu.attn_norm.f32"
    cpu_input = read_f32(cpu_input_path, 5120)
    gpu_input = read_f32(gpu_input_path, 5120)
    if cpu_input.tobytes() != gpu_input.tobytes():
        raise ValueError("CPU and fused GPU layer-0 projection inputs differ")

    beta_weight_path = ROOT / f"{PREFIX}.cpu.weight-beta-0.bin"
    beta_meta_path = ROOT / f"{PREFIX}.cpu.weight-beta-0.json"
    beta_meta = json.loads(beta_meta_path.read_text())
    if beta_meta["name"] != "blk.0.ssm_beta.weight" or beta_meta["type"] != 0:
        raise ValueError(f"unexpected beta weight metadata: {beta_meta}")
    if beta_meta["shape"] != [5120, 48] or beta_meta["strides"] != [4, 20480]:
        raise ValueError(f"unexpected beta layout: {beta_meta}")
    beta_weights = np.frombuffer(beta_weight_path.read_bytes(), dtype="<f4").reshape(48, 5120)
    beta_ref = (beta_weights.astype(np.float64) @ cpu_input.astype(np.float64)).astype(np.float32)

    qkv_weight_path = ROOT / f"{PREFIX}.cpu.weight-linear_attn_qkv_mixed-0.bin"
    qkv_meta_path = ROOT / f"{PREFIX}.cpu.weight-linear_attn_qkv_mixed-0.json"
    qkv_meta = json.loads(qkv_meta_path.read_text())
    if qkv_meta["name"] != "blk.0.attn_qkv.weight" or qkv_meta["type"] != 14:
        raise ValueError(f"expected Q6_K layer-0 QKV weights, got {qkv_meta}")
    if qkv_meta["shape"] != [5120, 10240] or qkv_meta["strides"] != [210, 4200]:
        raise ValueError(f"unexpected QKV layout: {qkv_meta}")
    qkv_raw = qkv_weight_path.read_bytes()
    qkv_ref = np.empty(10240, dtype=np.float32)
    x64 = cpu_input.astype(np.float64)
    rows_per_batch = 128
    for start in range(0, 10240, rows_per_batch):
        count = min(rows_per_batch, 10240 - start)
        row_bytes = 5120 // 256 * 210
        raw_rows = qkv_raw[start * row_bytes:(start + count) * row_bytes]
        dequant = q6k_rows(raw_rows, count, 5120)
        qkv_ref[start:start + count] = (dequant.astype(np.float64) @ x64).astype(np.float32)

    comparisons = {}
    for name, reference, outputs in (
        ("beta", beta_ref, [
            ("cpu_ggml", read_f32(ROOT / f"{PREFIX}.cpu.beta-0.f32", 48)),
            ("cpu_ggml_scalar", read_f32(ROOT / f"{PREFIX}.cpu.scalar-beta-0.f32", 48)),
            ("gpu_fused", read_f32(ROOT / "miinfer-optrace4.gpu.beta.f32", 48)),
        ]),
        ("qkv", qkv_ref, [
            ("cpu_ggml", read_f32(ROOT / f"{PREFIX}.cpu.linear_attn_qkv_mixed-0.f32", 10240)),
            ("cpu_ggml_scalar", read_f32(ROOT / f"{PREFIX}.cpu.scalar-linear_attn_qkv_mixed-0.f32", 10240)),
            ("gpu_fused", read_f32(ROOT / "miinfer-optrace4.gpu.linear_attn_qkv_mixed.f32", 10240)),
        ]),
    ):
        ref_path = ROOT / f"m31-lc4-{name}.reference.f32"
        ref_path.write_bytes(np.asarray(reference, dtype="<f4").tobytes())
        comparisons[name] = {
            "reference_sha256": sha(ref_path),
            "weight_sha256": sha(beta_weight_path if name == "beta" else qkv_weight_path),
            "input_sha256": sha(cpu_input_path),
            "vs": {label: stats(reference, values) for label, values in outputs},
        }

    report = {
        "schema": "m31-lc-0004-projection-reference-v1",
        "capture": "GDN layer 0, prompt position 0, frozen 2K input",
        "input_shape": [5120],
        "input_cpu_gpu_bitwise_equal": True,
        "reference": "independent canonical GGUF decode; F32 weights/dequantized weights; FP64 dot; F32 result",
        "qkv_type": "Q6_K (GGUF type 14), shape [5120,10240], row stride 4200 bytes",
        "beta_type": "F32 (GGUF type 0), shape [5120,48], row stride 20480 bytes",
        "comparisons": comparisons,
        "q6k_synthetic_self_check": "PASS",
    }
    (ROOT / "m31-lc4-projection-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
