# EXP-0402 — M31 Q8/Q8 Split-K at 64K

**Status:** IN PROGRESS

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

Pending.
