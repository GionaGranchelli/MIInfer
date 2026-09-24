# EXP-0388 — B128 shape-collapse attribution

## Question

Requalify the current aligned P512 baseline before attributing the B128
shape-collapse. If the baseline is valid, split B512 versus B128 by recurrent
and attention family. No optimization was implemented.

## Why the EXP-0387 residual conclusion is deferred

EXP-0387 showed P896 `42.465 s` with `B512 + 3×B128` and no scalar residual,
but its P512 sample was `3450.08 ms`, materially below the prior qualified
~2.48 s class. EXP-0388 therefore stops at baseline integrity instead of
profiling B128 on a potentially regressed or unstable baseline. B64 remains
rejected and B4 remains blocked.

## Baseline integrity gate

| Item | Value |
|---|---|
| Commit | `cae6953` |
| Preset | `m25_hi_qualified` |
| Model | Qwen3.8-27B-Q4_K_M.gguf; SHA `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| GPU/clocks | MI50-class gfx906; SCLK 1606 MHz, MCLK 1000 MHz |
| Policy/fan | 225 W policy; external physical fan full speed |
| HIP/compiler | HIP 7.1.52802-9999; clang 20.0.0.rocm |
| Build | `build/mi50-release/miinfer` |
| Environment | hermetic `env -i`, `HSA_OVERRIDE_GFX_VERSION=9.0.6`, `HIP_VISIBLE_DEVICES=0`, fixed context 1024 |
| Prompt | established repeated-fox P512 vector |
| Allocation | 2685 allocations; 22,999,937,428 bytes, unchanged across samples |

Five clean hermetic P512 samples after one warm-up were:

| Sample | ms | tok/s |
|---:|---:|---:|
| 1 | 2516.79 | 203.43 |
| 2 | 4897.95 | 104.53 |
| 3 | 2558.30 | 200.13 |
| 4 | 2812.08 | 182.07 |
| 5 | 2802.39 | 182.70 |

Median: **2802.39 ms / 182.70 tok/s**. Previous qualified current-generation
P512 class: approximately `2454–2498 ms`, `~200–207 tok/s` (EXP-0349 and
related M25 qualification). The current median is materially slower and the
4897.95 ms sample demonstrates severe instability. The baseline gate is
**FAIL**.

## P512/P640/P896 clean matrix

Not run after Gate A failed. The required stop rule prohibits B128 attribution
until current P512 regression/stability is explained. EXP-0387's earlier
single-sample values remain historical, not requalified EXP-0388 evidence:
P512 `3450.08 ms`, P640 `16031.40 ms`, P896 `42464.99 ms`.

## mx sanity

Not run. The baseline gate failed before the reference-shape comparison was
needed; no mx internal timing is inferred. EXP-0387's current reference values
are not substituted for this stopped gate.

## B512 versus B128 family attribution

Not run by design. No recurrent/attention/other GPU-family table is claimed,
and no B128 collapse is requalified on top of the failed P512 baseline.

## Initial regression observations

The current process uses the same qualified preset, hermetic environment,
fixed context capacity, model, target, allocation size, and nominal 1606/1000
MHz clocks as the historical qualification. The samples nevertheless range
from 2.517 s to 4.898 s. This establishes a current-head baseline regression
and/or runtime-state instability, but does not identify whether the cause is
clock residency, host scheduling, resource selection, synchronization, or
another environment/runtime factor. No speculative cause is promoted.

## Decision

**Disposition: BASELINE_REGRESSION.** The B128 shape-collapse attribution is
paused because the aligned P512 baseline is not currently valid and stable.

Exact next PRIMARY:

> Attribute the current-head aligned P512 regression and high variance under
> the hermetic qualified environment; only after P512 requalifies should clean
> P640/P896 and B128 family attribution resume.

B64: **REJECTED — CURRENT M28 FRONTIER**.

B4: **BLOCKED**.

Explicitly not authorized: new residual architecture, B64, B4, padding-to-512,
replay tricks, new precision strategy, new scheduler family, or speculative
GPU kernel.

No optimization candidate or new residual architecture was implemented. The
next PRIMARY was selected from the measured P512 baseline integrity failure.

## Provenance

Experiment commit and graph SHA are added after graph refresh. Final working
tree status is recorded in the provenance commit.
