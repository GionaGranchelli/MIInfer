#!/usr/bin/env python3
"""Split M31 Q6_K QKV error into production-input quantization and MMQ error."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_f32(path, count):
    raw = Path(path).read_bytes()
    if len(raw) != count * 4:
        raise ValueError(f"{path}: expected {count * 4} bytes, got {len(raw)}")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite values")
    return values


def q6k_rows(raw, rows, columns):
    if columns % 256 or len(raw) != rows * (columns // 256) * 210:
        raise ValueError("invalid Q6_K matrix size")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(rows, columns // 256, 210)
    ql, qh = blocks[:, :, :128], blocks[:, :, 128:192]
    scales = blocks[:, :, 192:208].view(np.int8).astype(np.float32)
    d = blocks[:, :, 208:210].copy().view("<f2").reshape(rows, -1).astype(np.float32)
    out = np.empty((rows, columns // 256, 256), dtype=np.float32)
    for half in range(2):
        for lane in range(32):
            h = qh[:, :, half * 32 + lane]
            a, b = ql[:, :, half * 64 + lane], ql[:, :, half * 64 + lane + 32]
            qs = ((a & 15) | ((h & 3) << 4),
                  (b & 15) | (((h >> 2) & 3) << 4),
                  (a >> 4) | (((h >> 4) & 3) << 4),
                  (b >> 4) | (((h >> 6) & 3) << 4))
            indices = (half * 128 + lane, half * 128 + lane + 32,
                       half * 128 + lane + 64, half * 128 + lane + 96)
            for group, (index, q) in enumerate(zip(indices, qs)):
                scale = d * scales[:, :, half * 8 + group * 2 + lane // 16]
                out[:, :, index] = scale * (q.astype(np.int16) - 32).astype(np.float32)
    return out.reshape(rows, columns)


def mx_q8_row(raw, token_count, token=0):
    blocks = 5120 // 128
    expected = blocks * token_count * 144
    if len(raw) != expected or not 0 <= token < token_count:
        raise ValueError(f"invalid Mx Q8 capture: expected {expected} bytes")
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, token_count, 144)[:, token]
    scales = packed[:, :16].copy().view("<f4").reshape(blocks, 4)
    qs = packed[:, 16:].copy().view(np.int8).reshape(blocks, 4, 32)
    return (qs.astype(np.float32) * scales[:, :, None]).reshape(-1)


def quantize_mx_q8_row(values):
    """Reproduce the non-affine 128-column production Mx quantizer."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (5120,):
        raise ValueError("Mx Q8 input must contain 5120 F32 values")
    packed = np.zeros((40, 144), dtype=np.uint8)
    for block in range(40):
        x = values[block * 128:(block + 1) * 128].reshape(4, 32)
        for group in range(4):
            maximum = np.max(np.abs(x[group])).astype(np.float32)
            scale = np.float32(maximum / np.float32(127.0)) if maximum else np.float32(0)
            packed[block, group * 4:group * 4 + 4] = np.frombuffer(scale.tobytes(), dtype=np.uint8)
            if scale:
                ratio = x[group] / scale
                q = np.copysign(np.floor(np.abs(ratio) + np.float32(0.5)), ratio)
                packed[block, 16 + group * 32:16 + (group + 1) * 32] = np.clip(q, -127, 127).astype(np.int8).view(np.uint8)
    return packed.tobytes()


def summary(reference, actual):
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    norm = np.linalg.norm(reference.astype(np.float64))
    return {
        "relative_l2": float(np.linalg.norm(delta) / norm),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def self_check():
    values = np.linspace(-1.0, 1.0, 5120, dtype=np.float32)
    packed = quantize_mx_q8_row(values)
    decoded = mx_q8_row(packed, token_count=1)
    assert len(packed) == 40 * 144
    assert decoded.shape == values.shape and np.isfinite(decoded).all()
    assert np.array_equal(packed, quantize_mx_q8_row(values))


def main():
    self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-f32", type=Path, required=True)
    parser.add_argument("--quantized-input", type=Path, required=True,
                        help="block-major capture: [40 blocks, token_count, 144 bytes]")
    parser.add_argument("--token-count", type=int, required=True)
    parser.add_argument("--gpu-output", type=Path, required=True)
    parser.add_argument("--q6k-weights", type=Path, required=True,
                        help="canonical row-major Q6_K GGUF tensor bytes")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_f32 = read_f32(args.input_f32, 5120)
    quantized_raw = args.quantized_input.read_bytes()
    block_bytes = args.token_count * 144
    first_token = b"".join(quantized_raw[b * block_bytes:b * block_bytes + 144] for b in range(40))
    if quantize_mx_q8_row(input_f32) != first_token:
        raise ValueError("captured position-0 activation does not match the production Mx Q8 quantizer")

    activation_b = mx_q8_row(quantized_raw, args.token_count)
    if not np.array_equal(activation_b, mx_q8_row(quantized_raw, args.token_count)):
        raise ValueError("captured production Q8 activation decode is not deterministic")
    qkv_raw = args.q6k_weights.read_bytes()
    weights = q6k_rows(qkv_raw, 10240, 5120)
    ref_a = (weights.astype(np.float64) @ input_f32.astype(np.float64)).astype(np.float32)
    ref_b = (weights.astype(np.float64) @ activation_b.astype(np.float64)).astype(np.float32)
    gpu = read_f32(args.gpu_output, 10240)
    result = {
        "schema": "m31-lc-0005-qkv-quantization-decomposition-v1",
        "input_sha256": hashlib.sha256(args.input_f32.read_bytes()).hexdigest(),
        "quantized_input_sha256": hashlib.sha256(quantized_raw).hexdigest(),
        "q6k_weight_sha256": hashlib.sha256(qkv_raw).hexdigest(),
        "gpu_output_sha256": hashlib.sha256(args.gpu_output.read_bytes()).hexdigest(),
        "reference": "canonical Q6_K dequantization, FP64 dot, F32 output",
        "activation_b": "dequantized captured MxQ8_1MmqBlock; production bytes verified against quantizer",
        "activation_quantization_error": summary(ref_a, ref_b),
        "gpu_kernel_error_vs_quantized_reference": summary(ref_b, gpu),
        "gpu_total_error_vs_full_precision_input": summary(ref_a, gpu),
        "synthetic_quantizer_self_check": "PASS",
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
