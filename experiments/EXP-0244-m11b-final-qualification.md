# EXP-0244 — M11-B final qualification and measured ceiling

## Result

The production layer-major/chunked prefill path is correct and bounded, but
does not reach the current M11-B gate of 100 P512 prompt tokens/s. The best
qualified repeated result is 46.22 tok/s at P513 from EXP-0235.

## Current production evidence

The following single current-binary checks used
`MIINFER_PREFILL_LAYER_MAJOR=1`, `MIINFER_HIP_GRAPH=0`, generation disabled
unless noted, and the qualification model
`Qwen3.8-27B-Q4_K_M.gguf`:

| Workload | Prompt time | PP tok/s | Decode |
| --- | ---: | ---: | ---: |
| P129 / TG64 | 2696.71 ms | 47.84 | 31.53 tok/s |
| P513 | 11006.38 ms | 46.61 | disabled |
| P2049 | 47307.72 ms | 43.31 | disabled |

The repeated production qualification remains P513 at 46.22 tok/s. The
P129/TG64 decode check remains within the 31.5 tok/s M11-A gate. Full CTest
passes 21/21; the working tree is clean at the commit containing this record.
The layer-major path's incremental prefill workspace is approximately 32.1 MiB
(about 12.1 MiB recurrent staging plus about 20 MiB attention-tail staging),
allocated once and not used by decode. The existing CLI profile does not emit
dispatch counts, so no unsupported dispatch number is claimed here.

## P512 Amdahl profile

EXP-0237 measured approximately 11.35 s for P512, with 11.29 s inside layer
regions:

| Component | Time | Share of profiled wall | Causal constraint |
| --- | ---: | ---: | --- |
| Recurrent prepare/projections | 1.525 s | 13.4% | token-local/batchable |
| Recurrent ordered core | 0.815 s | 7.2% | sequential state |
| Recurrent deferred tail | 6.074 s | 53.5% | mostly batchable after state |
| Attention prepare | 0.462 s | 4.1% | batchable |
| Attention ordered/KV work | 0.732 s | 6.5% | causal/KV ordered |
| Attention deferred tail | 1.682 s | 14.8% | mostly batchable |
| Embedding and remaining wall | ~0.079 s | ~0.7% | — |

The dominant remaining work is the recurrent deferred tail, not the
sequential recurrent state update. Removing the measured ordered recurrent
region entirely would still leave roughly 10.5 s, far above the 5.12 s
required for 100 tok/s.

## Ceiling and blocker

The current qualified P513 rate requires `100 / 46.22 = 2.16x` end-to-end
speedup. EXP-0230 measured a generous raw-int8 GEMM ceiling of 1.387x over
native Q4_K B=4 projection work, before Q4_K scale/minimum handling and
conversion costs. Applied to the full current path, that optimistic ceiling
is only about 64.1 tok/s; the earlier 45.56 tok/s qualified point gives the
documented 63.19 tok/s bound in EXP-0233. The production-shaped repacked MMQ
candidate reached only 1.125x at a full 64-token tile and was much slower at
the causal B=4 unit.

Therefore another local B=4/B=8 tile, quantization-launch fusion, or isolated
normalization optimization cannot close the gate. The remaining blocker is a
new causal chunk dataflow that exposes a large token tile to a low-register,
Q4_K/Q5_K grouped projection implementation while preserving recurrent state
ordering and bounded workspace.

## Required next experiment

If M11-B work resumes, the next defensible experiment is one end-to-end
causal 64-token schedule: prepare all token-local recurrent inputs, advance
the recurrent state in order, then run grouped Q4_K/Q5_K projections over the
prepared tile. It must be measured end-to-end at P512; a standalone grouped
projection win is insufficient. The existing 1.125x MMQ result is not a
qualification and must not be treated as one.

## Decision

**QUALIFIED BEST EFFORT — GATE UNMET.** The production architecture, bounded
workspace, correctness checks, decode path, and negative evidence are
complete. The 100 tok/s target is not silently redefined; reaching it requires
the new causal grouped-dataflow experiment above.

## Subsequent re-evaluation

EXP-0245 tested a native Q4_K four-row × 16-token split-4 mapping and measured
`0.594x` against four existing B=4 launches. EXP-0246 widened the same idea
to four rows × 64 tokens, staging each weight tile once; it was numerically
correct but measured `0.660x` against sixteen B=4 launches. Both candidates
were removed. These results further narrow the blocker to a genuinely
different causal grouped-dataflow schedule rather than another native
Q4_K remapping.

EXP-0247 tested a two-thread-per-output-cell, four-row × 64-token Q4_K
decomposition. It was correct within `3.57628e-6` but reached only `0.116x`
the B=4 control. EXP-0248 tested an operator-major production tail ordering;
repeated P546 controls measured `45.45 tok/s` and candidates `45.15 tok/s`
(`0.993x`), with identical one-token continuation checks. EXP-0249 tested a
native 64-row × 64-token LDS-staged MMQ and measured `0.527x`. EXP-0250
tested expanded decoded-metadata grouping; it was correct within `6.4373e-6`
but measured `0.098x`. All four candidates were removed. The local grouped
projection search therefore remains negative: no tested variant closes the
end-to-end gap or justifies changing the production schedule.

The final external intake check found only the already-tested three-plane
repack/MMQ family and generic gfx906 MMQ warp-count tuning. Published Q4_K
discussion also reports that larger warp counts can regress Q4_K even when
they help Q8, so it does not supply a defensible untested candidate for this
path. The next implementation would need a new causal grouped-dataflow design,
not another tile or warp-count sweep.

EXP-0252 retested that three-plane repack/MMQ family against the exact current
FFN-down shape with strict numerical validation. Exact int8 activation sums
were required for correctness; the resulting candidate measured `0.978x` at
B64 and was substantially slower at B4/B16. It was removed, so the required
next implementation remains an end-to-end causal grouped-dataflow schedule.

EXP-0253 precomputed those exact sums into a bounded int16 side buffer to
remove the candidate's inner-loop reductions. It remained numerically correct
but fell to `0.649x` at B64 and was much slower at B4/B16, so the side-buffer
variant was removed as well.

EXP-0254 kept the exact sum in the candidate activation footprint, improving
the grouped tile to `0.894x` at B64 while remaining slower than native B4. It
was removed, closing the repacked-MMQ/token-reuse projection family for this
M11-B path.
