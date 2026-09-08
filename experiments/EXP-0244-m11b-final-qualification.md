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
