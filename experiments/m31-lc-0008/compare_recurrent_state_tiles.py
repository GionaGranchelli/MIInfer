#!/usr/bin/env python3
"""Compare CPU and MIInfer layer-0 recurrent state at each prompt tile start."""
import numpy as np
from pathlib import Path

data = Path(__file__).resolve().parents[2] / "results/m31-lc-0008/z840-operations"
for token in (0, 512, 1024, 1536):
    cpu = np.fromfile(data / f"cpustates.cpu.state-before-token-{token}.f32", dtype=np.float32)
    gpu = np.fromfile(data / f"gpu6.gpu.state-before-token-{token}.f32", dtype=np.float32)
    assert cpu.size == gpu.size == 48 * 128 * 128
    # GGML stores axis 0 fastest; MIInfer stores [head][key][value].
    cpu = cpu.reshape(48, 128, 128).transpose(0, 2, 1).reshape(-1)
    delta = gpu - cpu
    denom = np.linalg.norm(cpu)
    rel = np.linalg.norm(delta) / denom if denom else 0.0
    print(f"token={token} cpu_norm={denom:.9g} gpu_norm={np.linalg.norm(gpu):.9g} "
          f"rel_l2={rel:.9g} rmse={np.sqrt(np.mean(delta * delta)):.9g} "
          f"max_abs={np.max(np.abs(delta)):.9g}")

cpu_after_tile0 = np.fromfile(
    data / "cpustates.cpu.state-before-token-512.f32", dtype=np.float32
).reshape(48, 128, 128).transpose(0, 2, 1).reshape(-1)
gpu_same_inputs = np.fromfile(data / "gdn.gpu-from-cpu-tile0.state-token-512.f32", dtype=np.float32)
delta = gpu_same_inputs - cpu_after_tile0
print(f"same_scan_inputs rel_l2={np.linalg.norm(delta) / np.linalg.norm(cpu_after_tile0):.9g} "
      f"rmse={np.sqrt(np.mean(delta * delta)):.9g} max_abs={np.max(np.abs(delta)):.9g}")
