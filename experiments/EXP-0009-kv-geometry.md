# EXP-0009 — K/V Workgroup and Wave64 Reduction Geometry

**Status:** KEEP  
**Milestone:** M2  
**Author:** MIInfer project  
**Date:** 2026-08-30  
**Baseline commit:** `4842c9dd77ec` (EXP-0007 zero-point dot4)  
**Candidate commit:** `1e423c771d86`  

---

# 1. Question

Does correcting MIInfer's Q4_0 × Q8_1 workgroup geometry and reduction
strategy close the K/V gap against the pinned gfx906 llama.cpp MMVQ path?

---

# 2. Hypothesis

The EXP-0007 K/V kernel launches 256 threads for 128 Q4 blocks per row. Half
of the threads therefore perform no dot work, while all 256 threads enter a
generic LDS reduction. A 128-thread, two-Wave64 geometry should remove that
waste. A Wave64-native reduction may reduce the remaining synchronization
cost.

Split-K is deliberately not tested here: the variable is workgroup geometry
and reduction, not additional workgroups or a second reduction kernel.

---

# 3. Prior evidence

EXP-0008 directly measured the pinned reference at approximately 13.12 µs for
both K and V. Its batch-1 GCN MMVQ configuration uses 128 threads (two
Wave64 waves), 256 B of shared partials, and one cross-wave barrier.

The EXP-0007 MIInfer control used 256 threads, 1 KiB of LDS, and a nine-stage
generic reduction. For K=4096 there are 128 Q4_0 blocks per row, so threads
128–255 were idle during the dot-product pass.

---

# 4. Candidates

The Q4/Q8 zero-point-corrected dot4 arithmetic is unchanged.

| Candidate | Geometry | Reduction |
|---|---|---|
| EXP-0007 control | 256 threads / row | 256-entry LDS tree, 9 barriers |
| A | 128 threads / row | 128-entry LDS tree, 8 barriers |
| B | 128 threads / row | Wave64 shuffle reduction, 1 barrier |

Candidate A assigns one Q4 block to each thread for K=4096. Its strided block
loop also preserves correctness for D, where a row has 384 blocks. Candidate B
reduces each Wave64 with `__shfl_down`/`ds_bpermute`, stores two wave partials,
and combines them after one workgroup barrier.

No K-split, DPP, swizzle, custom layout, prefetch, or arithmetic change was
introduced.

---

# 5. Environment and method

All runs used the M0-approved state: gfx802 isolated, MI50/gfx906 visible,
no competing MI50 workload, and clean candidate commit. The benchmark used
the streaming rotate-3 weight regime, 10 warmups, 20,000 HIP-event samples,
and five independent repetitions per candidate/shape. Environment and
telemetry files are retained under:

```text
bench/results/EXP-0009-kv-geometry/
```

The result files retain individual timing samples, correctness metrics, GPU
identity, commit, and dirty state. The current object build observed active
gfx906 execution with no scratch/private segment. Telemetry included the
expected 1725 MHz SCLK / 1000 MHz MCLK performance state during active runs;
startup and inter-run samples also recorded lower clock states.

---

# 6. Correctness

Candidate A and B passed the accepted quantized CPU-oracle comparison on all
seven real shapes. Representative maximum absolute errors were:

| Shapes | Maximum absolute error | Cosine similarity | NaN/Inf |
|---|---:|---:|---|
| K/V | 0.002668 | 0.999999843 | none |
| Q/O | 0.002610 | approximately 1.0 | none |
| G/U | 0.002754 | approximately 1.0 | none |
| D | 0.004122 | approximately 1.0 | none |

The Debug and Release CTest suites both passed all five tests, including the
real-shape Q4/Q8 correctness test and the dot4 ISA probe.

---

# 7. Results

Values are the mean of five per-run medians in microseconds. The 256-thread
K/V/Q/G values were fresh controls collected in this experiment. O/U/D
256-thread controls are the fresh matched-session controls retained by
EXP-0008; A and B were measured in the EXP-0009 session.

| Shape | M × K | 256T control | A: 128T LDS | B: 128T Wave64 | MMVQ reference |
|---|---:|---:|---:|---:|---:|
| Q | 4096 × 4096 | 25.600 | 22.560 | 22.240 | 26.176 |
| K | 1024 × 4096 | 52.832 | 11.840 | 11.520 | 13.120 |
| V | 1024 × 4096 | 52.864 | 11.744 | 11.520 | 13.120 |
| O | 4096 × 4096 | 25.728 | 22.560 | 22.272 | 26.272 |
| G | 12288 × 4096 | 60.000 | 51.776 | 51.872 | 61.600 |
| U | 12288 × 4096 | 60.352 | 51.744 | 51.904 | 61.632 |
| D | 4096 × 12288 | 57.600 | 73.184 | 72.960 | 64.928 |

Candidate A reduces K/V latency by 77.6–77.8% versus the matched 256-thread
control. Candidate B reduces it by 78.2%. Candidate B's additional reduction
over A is only about 2.7% for K/V, about 1.4% for Q/O, and is slightly slower
than A on G/U.

The geometry is not universal: D has 384 Q4 blocks per row, so 128 threads
must process three blocks each. Both candidates regress there. The correct
selection is therefore shape-specific, with the 128-thread path applicable to
the 128-block-per-row Q/K/V/O/G/U family and the accepted EXP-0007 geometry
retained for D pending a separate experiment.

Logical effective bandwidth for Candidate B was approximately 205 GB/s for
K/V, 424 GB/s for Q/O, 546 GB/s for G/U, and 388 GB/s for D. These values use
the quantized logical-byte definition and are not HBM traffic measurements.

---

# 8. Attribution

The dominant K/V gain comes from Candidate A, which removes the 128 idle
dot-product threads and halves the reduction input. The Wave64-native
reduction contributes a small additional gain after geometry is corrected.
This result rejects the earlier assumption that K/V fundamentally requires
split-K to expose enough grid parallelism: the reference reaches 13.12 µs
with one row per workgroup, and MIInfer reaches 11.52–11.84 µs with the same
basic row distribution after fixing per-workgroup utilization.

---

# 9. ISA and resources

The extracted gfx906 MIInfer object showed the following static observations.
The dot4 counts are static instruction counts, not dynamic execution counts.

| Property | 256T control | A: 128T LDS | B: 128T Wave64 |
|---|---:|---:|---:|
| `v_dot4_i32_i8` | 8 | 8 | 8 |
| barriers | 9 | 8 | 1 |
| VGPR | 23 | 20 | 20 |
| numbered SGPR | 13 | 12 | 12 |
| dynamic LDS launch | 1024 B | 512 B | 0 B |
| static shared partials | none | none | 8 B |
| scratch/spills | none | none | none |
| static body | 0x3bc | 0x360 | 0x308 |

Candidate B contains six `ds_bpermute_b32` operations corresponding to the
Wave64 reduction lowering. Both candidates retain the intended
`v_dot4_i32_i8` arithmetic and do not expand Q4 weights in global memory.

---

# 10. Comparison with MMVQ

Against the pinned EXP-0008 MMVQ measurements, Candidate B is faster by about
12.2% on K/V, 15.0–15.2% on Q/O, and 15.8% on G/U. It is slower by about
12.4% on D. Candidate A shows the same qualitative result and is simpler;
Candidate B is not selected as the universal default because its Wave64
reduction adds little and slightly regresses G/U.

The result establishes a shape-specialized MIInfer kernel family that is
competitive with or faster than the strongest measured reference path across
all seven shapes: A/B cover K/V/Q/O/G/U, while the accepted EXP-0007 D path
already beat MMVQ on D in EXP-0008.

---

# 11. Decision

```text
EXP-0009 KEEP
```

Accept Candidate A's 128-thread geometry as the simple K/V-oriented geometry
result. Candidate B remains a measured Wave64-reduction control and may inform
future work, but it does not replace A as the broad selection.

```text
M2 GO
```

The M2 gate is satisfied by a correctness-valid, repeated isolated-kernel
comparison against the pinned gfx906 implementation. This is not an
end-to-end inference claim; a complete MIInfer runtime remains a later
milestone.

---

# 12. Next experiment

Recommend exactly one next milestone:

```text
M3 — minimal Qwen3-8B runtime scaffold
```

The next task should begin model metadata/tensor loading, explicit supported
configuration validation, GPU weight allocation, and a static execution-plan
scaffold. It should not introduce another kernel optimization in EXP-0009.

