# M31-0000 — Canonical Single-MI50 Baseline Freeze

## Scope

M31-0000 merges the completed M30 runtime into the canonical integration base,
records the exact build and hardware provenance, and runs the clean correctness
gate. It contains no performance work and does not rerun the M30 86-request
workload or reopen COW optimization.

## Required record

```text
M31_BASELINE_SHA=<filled after final evidence commit>
M31_MERGE_SHA=dbcbdca801623ea7e4a6d42c1dcefef16e947685
M31_GATE_SHA=99840d10eabe5e64fe2564d34d3d3fcd42b43521
M31_MODEL_PATH=/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf
M31_MODEL_SHA256=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
M31_CONTAINER_IMAGE=localhost/miinfer-dev:rocm-7.2.1
M31_CONTAINER_DIGEST=sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a
M31_CONTAINER_BASE_DIGEST=sha256:fb0de294a2919ff6d503b4845e055c7fb80db5861954694ee5228e48a14411ec
M31_ROCM_VERSION=7.1.52802-9999 (container); driver=7.2.7-200.fc44.x86_64
M31_HOST=machinist; kernel=7.2.7-200.fc44.x86_64; PCIe=0000:85:00.0
M31_GPU=AMD MI50/gfx906, card1, 32 GiB HBM2
M31_GPU_CLOCKS=fclk=1166MHz, mclk=1000MHz, sclk=1000MHz, socclk=971MHz
M31_POWER_POLICY=power_dpm_force_performance_level=auto; rocm-smi Performance Level=auto
```

## Gate

The clean build and host/GPU correctness smoke must pass on the frozen source.
The complete stdout/stderr and environment capture are retained in
`results/m31-0000-machinist-evidence.txt`. Performance matrices belong to
M31-0001 and later.

## N=1 invariant

The frozen single-MI50 result, including weaknesses, is the regression
invariant for V3. Dual-MI50 work must preserve N=1 correctness and must not
materially regress the frozen single-MI50 path.
