# EXP-0271 — MIInfer runtime-only prefill/decode curve

## Benchmark

The harness bypasses HTTP, JSON, ChatML, and Hermes and invokes `miinfer run`
directly. It records prompt token count, prefill, one-token decode, TTFT, total
latency, wall time, and raw stdout/stderr.

```bash
python3 scripts/bench-m18-runtime.py \
  build/mi50-release/miinfer \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompts 8,128,512
```

## Environment

MI50/gfx906, ROCm 7.1.52802-9999, Clang 20.0.0, model SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

## Results

| path | prompt tokens | prefill ms | PP tok/s | decode ms | TG tok/s |
|---|---:|---:|---:|---:|---:|
| default | 9 | 247.24 | 32.36 | 35.61 | 28.09 |
| default | 129 | 3860.88 | 33.15 | 35.50 | 28.17 |
| default | 513 | 15781.30 | 32.44 | 36.68 | 27.26 |
| M12 experimental | 9 | 279.34 | 32.22 | 1.63 | 613.65 |
| M12 experimental | 129 | 2735.77 | 47.15 | 1.63 | 613.31 |
| M12 experimental | 513 | 11167.91 | 45.94 | 1.57 | 635.51 |

Raw results: `results/m18-runtime/20260909-105118-3881308/summary.json`.

## Decision

The M12 path is materially faster on these exact MI50 runs, especially for
decode, and is retained as an experimental path. A competitor ratio cannot be
computed because EXP-0269 could not load the same model at the required pin.
