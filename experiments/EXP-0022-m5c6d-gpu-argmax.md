# EXP-0022 — M5-C6d GPU-side greedy argmax

**Status:** KEEP (structural; small measured gain)  
**Milestone:** M5  
**Date:** 2026-09-01  
**Baseline:** `a202a946a8ac` (M5-C6c)  
**Candidate:** `7a1ed5c`

## Hypothesis

Greedy decode only needs the winning vocabulary index, but the fast path
copies the complete approximately 608 KiB logits vector to the host for
`std::max_element`. A deterministic GPU reduction can leave logits resident
and copy one `uint32_t` token ID instead.

## Candidate

The gfx906 path adds a standalone one-workgroup argmax primitive with explicit
first-index tie breaking, matching `std::max_element` for finite logits. The
selected ID is written to a persistent decode-workspace word and copied to the
host. The existing full-logit API remains available for diagnostics and
comparisons.

The A/B harness supports `--mode argmax`: `full-logits` is the control and
`gpu-argmax` is the candidate. The position audit supports `--gpu-argmax` for
copy/dispatch accounting.

## Environment and workload

* Model: Qwen3-8B Q4_0, SHA256
  `458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`
* GPU: AMD Instinct MI50 / gfx906
* Build: Release, HIP 7.1.52802-9999
* Workload: prompt ID `14990`, 8 warmup tokens, 64 measured growing-context
  decode forwards, 3 balanced A/B pairs
* A/B order: `full-logits,gpu-argmax,gpu-argmax,full-logits,full-logits,gpu-argmax`
* Hardware state: observed approximately 930 MHz SCLK / 350 MHz MCLK; rates
  are optimization evidence, not canonical-clock results.

## Correctness

Release CTest passed **19/19**, including the GPU primitive tie fixture
(`{1, 5, 5, -2, 4}` selects index `1`). The control and candidate produced
finite, deterministic, identical generated IDs through all 64 measured
forwards. Persistent KV/cache and reset behavior remain unchanged.

## Structural result

The clean GPU-argmax position audit is retained under
`bench/results/20260901T093211Z-381976/` and reports, per audited token:

| Metric | Full-logit path | GPU-argmax path |
|---|---:|---:|
| Final result copied | 607,744 bytes | **4 bytes** |
| Total copied bytes | 1,197,568 | **589,828** |
| Dispatches | 1,624 | **1,625** |
| Synchronization sites | 38 | 38 |
| Temporary allocations | 0 | 0 |

The remaining 589,824 bytes are the 36 layer-input D2D copies. The extra
dispatch is the standalone argmax reduction; one final token-ID D2H copy and
the profile finalization synchronization remain.

## Performance

Raw clean result: `bench/results/20260901T093127Z-381071/`.

| Policy | Mean decode | Throughput |
|---|---:|---:|
| Full-logit control | 1168.04 ms / 64 | 54.7928 tok/s |
| GPU-argmax candidate | 1163.75 ms / 64 | 54.9945 tok/s |

The candidate measured `1.00368x`, approximately **0.37% faster**, in the
balanced interleaved run. The gain is intentionally reported as small; the
large transfer is only one operation per token and the workload is dominated
by remaining compute and dispatch topology.

## Reproducibility rerun

After refreshing the Release configure/build metadata, a clean-tree rerun was
recorded under `bench/results/20260901T093604Z-383152/` with embedded commit
`ba5d0d3d2680`. It measured 54.7099 tok/s for full-logit control and 54.9812
tok/s for GPU-argmax, or `1.00496x`, with identical IDs. The corresponding
metadata-clean position audit is under
`bench/results/20260901T093647Z-384099/`; it repeats 589,828 copied bytes,
1,625 dispatches, 38 synchronization sites, and zero temporary allocations
at positions 1, 8, 16, 32, and 64.

## Decision

```text
KEEP — full logits D2H reduced from 607,744 to 4 bytes with exact tie
semantics, unchanged token behavior, and a small measured end-to-end gain
```

## Follow-up

The broad copy/materialization cleanup phase is complete. Run a fresh
post-C6d profile before selecting dispatch fusion, HIP graphs, FFN,
quantization, or LM-head kernel work.
