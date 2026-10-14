# EXP-0436 — GQA6 Stage1 hardware-counter readout

**Status:** LEARN — stage-level limiter remains unresolved

## Question

Does the exact packed GQA6 Stage1 on the restored long-context suffix show an
obvious HBM, L2, or LDS-bank-conflict limit that should determine the next
attention schedule?

## Setup

- Source: `c7ff28ff15da500d2051bc77377edfe077580dbd` (`perf/m31-gqa6-longctx`)
- Image: `localhost/miinfer-m31-rocm721-libdw:20261014T094012Z`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Restored session: 73,099-token prefix; profile tile: 1,024 tokens
- GPU: Machinist GPU2, PCI BDF `0000:88:00.0` (gfx906); GPU0 was not used
- Route: `MIINFER_M31_Q8_GQA6_PROD_ARITH=1`, packed LDS and LDS128 enabled
- Five successful profiled runs; each collected all 16 attention-layer
  Stage1 dispatches. This is an instrumented profiling workload, not a timing
  comparison. It does not measure cold 64K prefill.
- Thermal guard: no warning or stop; sampled junction temperatures were
  51–57°C. All containers exited 0 and the saved session completed its profile
  tile.

## Results

Per Stage1 dispatch, across 16 dispatch samples per counter group:

| Counter | Median | Range | Interpretation |
|---|---:|---:|---|
| `SQ_INSTS_VALU` | 65.448B | 65.416–65.475B | Very large VALU instruction count; aggregate per-SE counter |
| `SQ_ACTIVE_INST_VALU` | 68.409B | 68.371–68.441B | Per-SIMD active VALU-issue cycles, aggregate; not normalized to elapsed cycles |
| `SQ_WAIT_INST_LDS` | 16.061B | 15.991–16.104B | Per-SIMD wave-cycles waiting for LDS issue, in 4-cycle units; not normalized by waves |
| `GRBM_GUI_ACTIVE` | 1.650B | 1.649–1.652B | GPU-active cycles during the measured dispatch |
| `TCC_HIT` | 2.440B | 2.440–2.440B | L2 cache hits |
| `TCC_MISS` | 28.131M | 27.893–28.368M | L2 cache misses; hits/(hits+misses) is 98.86% for these counters |
| `LDSBankConflict` | 0 | 0 | No LDS bank-conflict stalls reported |

`TCC_EA_RDREQ_32B_sum` returned zero for all 16 dispatches despite a successful
process exit. Treat this metric as invalid or unsupported in this collection;
it is not evidence of zero external-memory traffic. A follow-up run added
`SQ_WAVES` to the LDS-wait/VALU group; it reported 393,216 waves per dispatch.
Dividing the aggregate counters by waves gives about 40,868 LDS-wait counter
units per wave and 174,000 VALU-active issue cycles per wave. Converting the
wait counter's four-cycle units gives an aggregate LDS-wait/VALU-active ratio
of about 94%. This is a ratio of overlapping per-SIMD counters, **not** a
claim that 94% of kernel wall time is stalled.

The rocprof code-object
metadata reports 192 threads, 72 VGPR, 48 SGPR, and zero scratch bytes. The
reported LDS allocation is zero; the kernel may use dynamically supplied LDS,
so this field does not establish that the kernel uses no LDS.

Counter definitions and units are architecture-specific. AMD documents
`SQ_WAIT_INST_LDS` as per-SIMD wave-cycles in 4-cycle units and `SQ_ACTIVE_INST_VALU`
as a per-SIMD issue-cycle counter; raw totals cannot be divided directly to
claim an LDS-stall percentage. `SQ_WAVES` and a valid normalization are still
needed. See [AMD ROCprof counter definitions](https://rocm.docs.amd.com/projects/rocprofiler/en/docs-5.6.0/rocprof.html).

## Decision

The high L2 hit ratio and zero reported bank conflicts make an obvious
L2-miss or bank-conflict explanation less likely. The normalized LDS-wait
counter is large relative to VALU-active issue cycles, so reducing LDS issue
pressure is a strong candidate direction. The metric does not measure the
fraction of wall time lost, and LDS bank conflicts remain zero; do not equate
the ratio with a measured speedup ceiling. The zero EA request metric cannot
support an HBM conclusion.

Do not tune HBM prefetch or LDS padding from these counters. Continue with the
key-parallel attention rewrite, and measure whether it reduces LDS-wait per
VALU-active cycle while preserving the exact GQA6 path as control. Require
full-model numerical parity before performance claims. If the rewrite is
rejected, test whether direct register loads can avoid LDS traffic before
reopening tile-size or prefetch sweeps.

## Reproduction artifacts

Raw CSVs and telemetry are on Machinist at
`/home/machinist/m31-gqa6-longctx-results/EXP-0436-gqa6-hw-counters/`.
The local pinned container and source snapshot are recorded in the setup above.
