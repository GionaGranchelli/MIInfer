# EXP-0170 — M6-B75 fused Gate/Up/SwiGLU prototype

## Hypothesis

A single exact-shape Q4_K×Q8_1 kernel can compute Gate and Up, apply SwiGLU,
and write only the Down input fast enough to remove the separate Gate/Up
materialization and activation pass.

## Candidate

Opt-in `MIINFER_Q4K_FUSED_GATE_UP_SWIGLU=1`. The 256-thread kernel computes two
output rows per workgroup, accumulates both Q4_K projections, applies SwiGLU
after reduction, and writes the FP32 Down-input vector. The production B47
shared-input path remains the control.

## Correctness

Native 64-token replay passed. Decode allocations remained zero.

## Results

| Workload | Control | Candidate | Delta |
|---|---:|---:|---:|
| TG64 | 14.280 tok/s | 13.0846 tok/s | -8.37% |

Candidate wall samples were `4880.23, 4889.88, 4891.26, 4896.46, 4897.83 ms`.

## Revised implementation

One allowed revision matched the pinned GCN mapping: two Wave64 groups for one
output row, with both groups accumulating Gate and Up. Replay still passed, but
the five-sample TG64 median was `4861.66 ms` (`13.1642 tok/s`), or `-7.8%`
versus the `14.280 tok/s` production baseline.

## Interpretation

The candidate removes an activation launch and intermediate Gate/Up stores, but
each thread still performs two independent Q4_K metadata/nibble streams. The
extra accumulation and register/LDS pressure costs more than the removed
boundary work. This is not the llama.cpp fusion mechanism's benefit because
the current kernel does not share the Q4 decode or activation tile at the
required granularity.

## Decision

**REJECT.** Both the MIInfer two-row mapping and the reference-mapped two-wave
mapping lose materially. Remove the prototype; B47's separate production path
remains. Revisit only with a new Q4_K weight/activation representation, not
another fusion or geometry variant.
