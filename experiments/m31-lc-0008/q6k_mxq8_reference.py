#!/usr/bin/env python3
"""Compare CPU/GPU QKV rows to a Q6_K by captured MxQ8 reference."""
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data = root / "results/m31-lc-0008/z840-operations"
weights = (data / "cpudirect.cpu.weight-linear_attn_qkv_mixed-0.bin").read_bytes()
q8 = (data / "gpu5.gpu.qkv_input_mx_q8.bin").read_bytes()
assert len(weights) == 10240 * 20 * 210
assert len(q8) == 40 * 144

matrix = np.empty((10240, 5120), dtype=np.float32)
for out in range(10240):
    row = memoryview(weights)[out * 4200:(out + 1) * 4200]
    for block in range(20):
        offset = block * 210
        ql = np.frombuffer(row[offset:offset + 128], dtype=np.uint8)
        qh = np.frombuffer(row[offset + 128:offset + 192], dtype=np.uint8)
        scales = np.frombuffer(row[offset + 192:offset + 208], dtype=np.int8)
        d = float(np.frombuffer(row[offset + 208:offset + 210], dtype="<f2")[0])
        for half in range(2):
            ql_offset, qh_offset = half * 64, half * 32
            dest = block * 256 + half * 128
            lo = ql[ql_offset:ql_offset + 32].astype(np.int16)
            hi = ql[ql_offset + 32:ql_offset + 64].astype(np.int16)
            packed = qh[qh_offset:qh_offset + 32]
            qs = (
                (lo & 15) | ((packed & 3).astype(np.int16) << 4),
                (hi & 15) | (((packed >> 2) & 3).astype(np.int16) << 4),
                (lo >> 4) | (((packed >> 4) & 3).astype(np.int16) << 4),
                (hi >> 4) | (((packed >> 6) & 3).astype(np.int16) << 4),
            )
            for group, values in enumerate(qs):
                scale = scales[half * 8 + group * 2:half * 8 + group * 2 + 2]
                scale = scale.astype(np.float32).repeat(16)
                start = dest + group * 32
                matrix[out, start:start + 32] = (values - 32).astype(np.float32) * scale * d

activation = np.empty(5120, dtype=np.float32)
for block in range(40):
    offset = block * 144
    scales = np.frombuffer(q8[offset:offset + 16], dtype="<f4")
    values = np.frombuffer(q8[offset + 16:offset + 144], dtype=np.int8)
    activation[block * 128:(block + 1) * 128] = values.astype(np.float32) * np.repeat(scales, 32)

reference = matrix @ activation
reference.tofile(data / "qkv.q6k-mxq8-reference.f32")
cpu = np.fromfile(data / "cpudirect.cpu.linear_attn_qkv_mixed-0.f32", dtype=np.float32)
gpu = np.fromfile(data / "gpu4.gpu.linear_attn_qkv_mixed.f32", dtype=np.float32)
assert cpu.shape == gpu.shape == reference.shape == (10240,)

def report(label, actual, expected):
    delta = actual - expected
    print(f"{label}: rel_l2={np.linalg.norm(delta) / np.linalg.norm(expected):.9g} "
          f"rmse={np.sqrt(np.mean(delta * delta)):.9g} max_abs={np.max(np.abs(delta)):.9g}")

report("CPU_vs_Q6K_MxQ8_reference", cpu, reference)
report("GPU_vs_Q6K_MxQ8_reference", gpu, reference)
report("GPU_vs_CPU", gpu, cpu)
