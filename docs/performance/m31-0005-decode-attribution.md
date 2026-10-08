# M31-0005 — Decode attribution

Status: source-attributed; GPU component timings pending.

## Production path and measured frontier

The M31-0001 benchmark enables `GenerateOptions::use_hip_graph`. The default
decode path captures one HIP Graph containing the embedding, 16 topology blocks
(48 recurrent layers and 16 GQA attention layers), final norm, and logits.
Each generated step launches that graph, copies the vocabulary logits to the
host, synchronizes the stream, samples on CPU, and uploads the next decode
state. This is the path represented by the frontier results; eager decode is a
different path and must not be used to explain those timings.

Measured mean decode latency is approximately 38.2, 47.4, 62.1, and 90.1
ms/token at 8K, 32K, 64K, and 128K. The end-to-end increase from 8K to 128K is
about 51.9 ms/token. A straight-line endpoint slope is about 0.42 ms per
additional 1K context tokens; it is only a descriptive aggregate, not an
attention timing or causal attribution.

## Source-derived work model

The model has 24 query heads, 4 KV heads, head dimension 256, and 16 GQA
attention layers. Each full-attention layer performs QK and probability-V work
of `2 * 24 * 256 * L` MACs per decode token. Across 16 layers that is
`196,608 * L` MACs: approximately 1.61B MACs at 8K and 25.77B MACs (51.54
GFLOP using 2 FLOP/MAC) at 128K. This is required exact-attention work, not
measured device instructions.

FP16 KV contains 16 * 4 * 256 * 2(K,V) * 2 bytes = 65,536 bytes per token
across the attention layers. The logical unique K+V payload traversed per
decode step is therefore 0.5, 2, 4, and 8 GiB at 8K, 32K, 64K, and 128K.
Each of six query heads mapped to one KV head logically requests the same
head's K/V, so a no-cache/no-reuse upper bound is six times those amounts
(3/12/24/48 GiB). Actual HBM traffic and cache reuse are not measured; these
are logical request bounds, not DRAM counters.

Context-independent work includes the graph's projections, recurrent state
updates, normalization, logits, sampling, and fixed launches. The current
artifacts do not provide a clean, unprofiled per-component decomposition.
Earlier 8K profiled percentages and the eager suffix microbenchmark are not
comparable to the graph frontier. Consequently, the extra ~51.9 ms/token is
not assigned to GQA, GDN, dispatch, or memory bandwidth.

## Main deductions

- Exact full attention is linear in context length; the growth is expected, but
  its coefficient and realized memory traffic remain to be measured.
- Graph stage 1 launches a fixed 64-split grid for each query head. The kernel
  derives active split count from the current decode position and returns from
  excess split CTAs. At 8K, only 16 splits are active; at 128K, all 64 are
  active. This makes inactive-CTA overhead a measurable candidate, not a
  proven bottleneck.
- The 128K llama.cpp/MIInfer comparison is cross-host. The same-host 32K
  directional result and historic 128K result motivate attribution but do not
  prove an implementation cause.
- No broad GPU profile or new performance run was conducted. The real-workload
  stop guard is not yet validated for timely termination.

## Candidate ranking

| Candidate | Evidence | Expected impact | VRAM impact | Correctness risk | Cost |
|---|---|---|---|---|---|
| Graph attention split schedule | Fixed 64-grid; active splits scale 16→64 | Potentially high at long context; unmeasured | Lower active scratch possible, current global scratch fixed | Medium | Medium |
| KV head reuse across six query heads | Logical request multiplicity is 6; actual cache behavior unknown | Potentially high if DRAM traffic dominates | Neutral | High | High |
| FP16 KV layout/ownership | Direct pool view; no serving-path copy found | Low-to-medium unless a layout issue is measured | No identified removable duplicate | Medium | High |
| Host dispatch and sync | Per-token logits D2H + stream sync + CPU sampling are explicit | Mostly context-independent; semantics constrain removal | Neutral | High | Medium |
| Eager Split-K scratch | 32→3 split workspace differs, but not used by graph frontier | No established graph-path impact | 350.6 MiB difference in eager workspace reservation | Low | Low |
| Q8 KV | Saves substantial memory, but historic 64K latency regressed | Not supported as a speed win | Large savings | Medium | High |

The first worthwhile GPU experiment, once stop safety is established, is a
single guarded graph decode comparing default split policy with one controlled
split override at 8K and 32K, exact output parity, and a cooled starting point.
Do not extrapolate the eager 32-vs-3 Split-K microbenchmark to graph decode.
