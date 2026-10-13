# EXP-0402 — M31 Q8/Q8 Split-K at 64K

**Status:** MEASURED — NOT PROMOTED

**Baseline:** `88fd63eb0cf8b765f37e5f52168c80783df62e7b`

## Objective

At the final 512-token tile of a 64K context (`P=65024, S=512`), test whether
64 Split-K partitions shorten the long per-CTA scan enough to outperform the
current 32-way setting. EXP-0401 found 3-way splitting slower at `P=8192`; this
is a distinct long-context point and doubles scratch to about 800 MB.

## Gates

- One MI50 selected and verified by BDF from inside the process.
- Compare Q8/Q8 outputs (`max_abs <= 0.01`, cosine `>= 0.9999`).
- Five interleaved 32/64 timings, same Q, Q8 KV and `P=65024, S=512`.
- Keep 64 only if it is at least 10% faster and fits without memory pressure;
  then validate one full-model late-context tile before changing dispatch.

## Result

The first run was invalid: scratch was allocated for 32 splits while launching
64, causing an out-of-bounds GPU memory access. No timing or correctness claim
uses that run. After fixing the allocation, the outputs matched exactly on
Machinist GPU2 (`0000:88:00.0`, gfx906; `max_abs=0`, cosine `1.0`).

Interleaved GPU-event samples:

- 32 splits: `1586.494, 1587.699, 1580.696, 1623.573, 1646.058 ms`; median
  `1587.699 ms`.
- 64 splits: `1493.361, 1487.237, 1504.627, 1487.546, 1565.247 ms`; median
  `1493.361 ms`, a 6.3% speedup.

Independent junction polling peaked at 56 C. The 6.3% gain misses the 10%
promotion gate and requires doubling split workspace to about 800 MB per
attention layer tile, so production remains at 32 splits.
