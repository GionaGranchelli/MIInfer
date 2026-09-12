# EXP-0348 — M25 current oracle requalification

**Status:** RETEST; measurement-only, no optimization
**Milestone:** M25-L
**Date:** 2026-09-12
**MIInfer source:** `c0ae2fd`
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Question

What is the current apples-to-apples stretch gap after the recent M25-L
candidate screens, using one clean six-pair session?

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- fixed SCLK/MCLK `1606/1000 MHz`
- fresh process for every sample, six interleaved oracle/MIInfer pairs
- no profiler
- MIInfer used `MIINFER_PRESET=m25_hi_qualified`, `MIINFER_MX_Q8_BATCH=0`,
  and `--max-tokens 0 --no-stream`
- MIInfer used the exact 512-token prompt made by repeating
  `The quick brown fox jumps over the lazy dog. ` 51 times and appending
  `The quick`
- the pinned `llama-bench` path used `-p 512 -n 0 -r 1`; it does not accept
  the exact text prompt and therefore uses its synthetic P512 benchmark tokens

Commands:

```text
env -i PATH=/usr/bin:/bin HSA_OVERRIDE_GFX_VERSION=9.0.6 HIP_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/fedora-workstation/Development/mx-llama-build/bin:/usr/lib64/rocm/lib:/usr/lib64 \
  /home/fedora-workstation/Development/mx-llama-build/bin/llama-bench \
  -m /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv f16 -b 2048 -ub 2048 -p 512 -n 0 -r 1 -o json

env -i PATH=/usr/bin:/bin HOME=/tmp HSA_OVERRIDE_GFX_VERSION=9.0.6 HIP_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/fedora-workstation/Development/mi50/build/mi50-release:/usr/lib64/rocm/lib:/usr/lib64 \
  MIINFER_PRESET=m25_hi_qualified \
  /home/fedora-workstation/Development/mi50/build/mi50-release/miinfer run \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt "$(seq 1 51 | awk '{printf \"The quick brown fox jumps over the lazy dog. \"}')The quick" \
  --max-tokens 0 --no-stream
```

## Correctness and resource checks

All twelve fresh processes exited successfully, processed exactly 512 prompt
tokens on the MIInfer side, and produced finite output. The same-process
continuation/repeat-P512 gate, H/I matrix, CTest, and long-generation gates
remain covered by EXP-0322; this zero-generation screen did not replace those
checks.

MIInfer reported the same setup allocation in every sample:

```text
device_allocated_bytes=21993242964
device_vram_free_bytes=11484004352
```

Continuous 250 ms telemetry recorded 1,405 samples. Every sample reported
SCLK/MCLK `1606/1000 MHz`; junction temperature ranged from `38 C` to `65 C`,
and sampled VRAM use peaked at `22,945,542,144 B`.

## Results

| pair | mx-llama tok/s | MIInfer P512 ms | MIInfer tok/s |
|---:|---:|---:|---:|
| 1 | 221.403796 | 2418.69 | 211.68 |
| 2 | 221.715776 | 2841.29 | 180.20 |
| 3 | 221.372802 | 2431.74 | 210.55 |
| 4 | 221.353909 | 2473.60 | 206.99 |
| 5 | 221.431872 | 2525.09 | 202.76 |
| 6 | 221.546491 | 2700.75 | 189.58 |

Sorted medians are:

```text
mx-llama.cpp: 221.417834 tok/s = 2312.370 ms/P512
MIInfer H/I:  2499.345 ms/P512 = 204.853672 tok/s
gap:          186.975 ms/P512 = 8.086% slower than mx latency
```

The `2841.29 ms` and `2700.75 ms` MIInfer samples are retained. No sample was
discarded to improve the median.

## Interpretation

The stretch target remains open. Relative to EXP-0340's earlier six-pair
screen, the current oracle is `4.736 ms` slower by median while MIInfer is
`24.735 ms` slower; the measured gap consequently moves from `166.976 ms` to
`186.975 ms`. This is material run-to-run variation at the remaining stretch
scale, not evidence for a new kernel target.

The result reinforces the M25-L decision to measure the complete recurrent
execution contract before implementing another candidate. The external
benchmark's synthetic-token limitation remains a comparison caveat.

## Decision

**RETEST.** Keep `m25_hi_qualified` unchanged. Do not promote the attention
decode-reuse selector, pinned QKV selector, or persistent external-state
selector based on this screen. The next performance work requires a
like-for-like recurrent QKV/GDN stage differential with matching boundaries;
generic launch orchestration and isolated kernel swaps remain unproven as the
source of the remaining gap.

## Follow-up

Build the measurement-only M25-L contract table for one real recurrent layer
at B512 with one outer event pair for the whole stage and a separate
decomposition trace. Only implement a new port if the measured differential
survives that boundary-matched comparison.
