# EXP-0465 — Q2×K32 paired-token Q8 attention

## Candidate

This candidate uses the existing paired-token CTA mapping: one CTA covers two
adjacent suffix tokens, one KV head, and one split. Its three Wave64 groups
cover all six query heads for that KV head. It loads the active pair's union
into shared memory in K32 tiles, then applies the existing per-key FP32 online
softmax and PV update in ascending key order. The per-row causal and split
interval checks, partial-output indices, stage two, Q8 cache representation,
persistent state, and graph route are unchanged.

The change targets repeated Q8 K/V staging and barriers while sharing each
loaded tile across the two tokens and six query heads. It differs from the
rejected Q×K geometry sweeps by retaining the established paired-token CTA
mapping and recurrence order; only its data staging cadence changes. Dynamic
LDS is 16,512 bytes: `2 * 32 * 256` bytes for K and V plus `2 * 32` half
scales. Pair-head-reuse remains at its original four-key allocation.

## CPU interval and layout oracle

`experiments/tools/exp0465_q2k32_mapping_oracle.py` passed 4,374,882
row/split cases. It covered short contexts, odd and even suffix counts,
empty splits, tile tails, split transitions, positions around 65,536 and the
73,099 + 1,024 restored workload, and positions near cache capacity. The
oracle checked contiguous K32-plus-tail coverage, exact per-row split masks,
all 24 Q-head mappings, partial-output bounds, and dimension offsets.

## Compiler and resource gate

The full `miinfer-m31-fp16-full-model` target compiled with HIP Clang 22.0
inside `localhost/miinfer-dev:rocm-7.2.1`, image digest
`sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.
The built gfx906 device object was:

```text
/tmp/m31-gqa6-stage1-exp0465/build/mi50-release/CMakeFiles/miinfer_hip.dir/gfx906/kernels/qwen3_primitives.hip.o.0.hipv4-amdgcn-amd-amdhsa--gfx906
sha256 3fcc28218f20f7a19fbdee154e81ae46ea8f38e10e5c677357decbbb661255d1
```

Exact paired-token kernel metadata:

| Resource | Candidate |
| --- | ---: |
| VGPR | 82 |
| SGPR | 55 |
| VGPR / SGPR spills | 0 / 0 |
| Private segment | 0 B |
| Fixed LDS | 0 B |
| Dynamic LDS requested by launch | 16,512 B |
| Workgroup / wave | 192 threads / 64 lanes |

The predeclared resource limit is 72 VGPR. At 82 VGPR, this candidate fails
the compile/resource gate and is rejected before GPU use. It offers no basis
for spending the larger shared-memory footprint on a runtime measurement.

The generated gfx906 ISA was extracted from the HIP object and inspected. The
kernel stages vector K/V data to LDS at offsets 0 and 8,192 bytes, and scale
data at offsets 16,384 and 16,448 bytes. The disassembly shows the expected
VMEM completion waits before LDS writes, LDS reads for per-key consumption,
and barriers between tile staging and consumption. This confirms the compiled
K32 dataflow and its LDS layout; the 82-VGPR resource result is the decisive
gate.

## Result and disposition

**REJECT — compiler resource gate.** No GPU command was run, no suffix timing
or model numerical result was collected, and no cold 64K run was attempted.
GPU0 remains excluded. Reopening this design requires evidence that the same
K32 paired-token dataflow can compile at or below 72 VGPR without adding spills
or changing the exact recurrence and interval behavior.

The source edit is limited to
`gfx906/kernels/qwen3_primitives.hip`; the CPU oracle is the only added helper.
