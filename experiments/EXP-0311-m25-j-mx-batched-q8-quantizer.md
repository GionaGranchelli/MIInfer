# EXP-0311 — M25-J Mx batched Q8 quantizer

## Hypothesis

Porting the pinned mx MMQ Q8 quantizer's four-block workgroup layout will
reduce activation-quantization overhead for the existing Mx repacked
projections.

## Motivation

The pinned mx implementation at commit
`2e9d29fe736969160f17476ec6f0a6298cee6966` uses 128 threads, `float4` loads,
shuffle reductions over each 32-value group, and a grid of
`[token][ceil(columns / 512)]`. MIInfer's prior path launched one workgroup
per 128-value block and reduced through shared memory. The port preserves the
existing 144-byte MI50 MxQ8 block and its block-major output contract.

## Baseline

The existing MIInfer `mx_q8_1_mmq_quantize_kernel`, selected with
`MIINFER_MX_Q8_BATCH=0`.

## Candidate

`mx_q8_1_mmq_quantize_batch_kernel<AFFINE>` is selected by the opt-in
`MIINFER_MX_Q8_BATCH=1`. The default path is unchanged.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- pinned mx commit `2e9d29fe736969160f17476ec6f0a6298cee6966`
- Release build; `MIINFER_MX_PIPELINE=1`

The Q5 and Q6 bakeoff telemetry observed `1606/1000 MHz` throughout. Some
short Q4 captures also sampled startup SCLK values of `1386` or `1485 MHz`,
so the Q4 absolute timings are retained as relative projection evidence, not
as a fully clock-qualified headline result.

## Benchmark

The existing `miinfer-m24-projection-bakeoff` was extended to report the Q8
quantizer event separately from the Mx MMQ event. Three interleaved
candidate/control process pairs were run for each layer-3 Q4 Gate and Q6
Down projection and layer-0 Q5 SSM-out projection, at B512. The benchmark
command was:

```sh
MIINFER_MX_Q8_BATCH={0|1} MIINFER_MX_PIPELINE=1 \
build/mi50-release/miinfer-m24-projection-bakeoff \
/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
{gate q4 3|ssm_out q5 0|down q6 3}
```

## Correctness

The existing model-sized Mx projection contract harness passed with the
candidate enabled:

- Q4 Gate max absolute output error: `0.95e-6`
- Q5 SSM-out max absolute output error: `1.67e-6`
- Q6 Down max absolute output error: `0.83e-6`
- outputs remained finite

No full-generation correctness claim is made for this experiment. A fresh
wide full-layer model smoke did not complete and was terminated after the
GPU remained busy beyond the bounded diagnostic timeout; it produced no
throughput result. The established M25-H/I full-model result remains the
qualified generation result.

## Results

Median B512 timings across the three pairs were:

| projection | Q8 quantizer control → candidate | total control → candidate | decision signal |
| --- | ---: | ---: | --- |
| Q4 Gate, layer 3 | `64.80 → 35.36 us` (`-45.4%`) | `17815.34 → 17980.78 us` (`+0.9%`) | neutral |
| Q5 SSM-out, layer 0 | `70.56 → 55.36 us` (`-21.5%`) | `6873.43 → 6859.83 us` (`-0.2%`) | neutral |
| Q6 Down, layer 3 | `191.20 → 82.56 us` (`-56.8%`) | `18433.41 → 18353.88 us` (`-0.4%`) | neutral |

The Mx quantizer is materially faster, but it is a small fraction of each
projection. The candidate adds no allocation or persistent VRAM; the
standalone bakeoff buffers and 144-byte block layout are unchanged.

## Profiling

The result is consistent with the source-level hypothesis: shared-memory
reduction overhead is removed, while the repacked MMQ projection remains the
dominant event. No full-model profile was accepted because the diagnostic
smoke did not complete.

## Interpretation

This is a valid kernel-level port and a useful future optimization, but it
does not yet move end-to-end projection time enough to explain a P512 gain.
The full wide path needs a bounded, valid B512 qualification before this
branch can be considered production-safe.

## Decision

**RETEST as an opt-in path.** Keep the default quantizer unchanged.

## Follow-up

Repeat exact P512 candidate/control generation after the wide B512 path is
restored to a known-good device state. If the full-model result remains
neutral, remove the opt-in branch and retain only the benchmark evidence.
