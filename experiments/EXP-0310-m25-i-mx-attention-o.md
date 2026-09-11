# EXP-0310 — M25-I Mx attention O projection

## Hypothesis

After the attention FFN port, the remaining measured attention tail includes a
5.26 ms B512 O projection. The existing Mx Q4 repacked MMQ kernel should lower
that cost while leaving the resident M23 O copy available for decode.

## Baseline

EXP-0309's Mx attention FFN candidate, with M23 attention O for wide prefill
and decode.

## Candidate

`MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_O=1` packs the Q4 attention O weight in
the existing Mx layout and uses the shared Mx Q8_1 activation scratch during
wide prefill. The flag is independent and opt-in. The original M23 O packing
remains resident for scalar/decode execution.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- SCLK/MCLK `1606/1000 MHz`
- context capacity 1024; exact 512-token repeated fox prompt
- Mx recurrent, Mx GDN, and EXP-0309 Mx attention FFN paths enabled
- pinned Mx reference commit `2e9d29fe736969160f17476ec6f0a6298cee6966`

## Benchmark

The standalone attention layer bakeoff used layer 3 at B512. The full-model
qualification used three interleaved candidate/control pairs, each with a
concurrent 250 ms `rocm-smi --json` sampler.

## Correctness

- Combined layer output was finite; max absolute error versus the scalar
  oracle was `0.751`, RMSE `0.017`, within the existing `1.0` max-error
  tolerance.
- `qwen35-conv-batch-gpu` Release CTest passed after the final source build.
- The exact P512 candidate completed and its one-token continuation was
  `brown`.
- Candidate allocation was `18,472,649,044 B` versus control
  `19,108,282,708 B`.

## Results

Attention layer 3, B512:

| path | total GPU us | O projection ms | post-attention FFN ms |
| --- | ---: | ---: | ---: |
| EXP-0309 Mx FFN | 42354.038 | 5.263 | 27.674 |
| Mx FFN + O candidate | 39850.208 | 2.743 | 25.224 |

The O projection itself improved by `47.9%`; total layer time improved by
`5.9%`.

Full-model P512 raw latency samples were:

| path | latency ms | tok/s |
| --- | --- | --- |
| Mx FFN + O candidate | 2722.32, 2421.07, 2404.52 | 188.08, 211.48, 212.93 |
| M23 O control | 3228.44, 3242.44, 2824.63 | 158.59, 157.91, 181.26 |

Candidate median latency was `2421.07 ms`, or `211.48 tok/s`. The additional
unsampled final smoke was `2383.95 ms`, or `214.77 tok/s`. Every one of the six
telemetry captures observed only `1606/1000 MHz`; capture counts were 119,
138, 117, 136, 119, and 136 lines respectively.

## Interpretation

The O port is a measured incremental win on top of EXP-0309 and moves the
qualified median above the 200 tok/s project gate. It reduces VRAM versus the
M23 control despite retaining M23 O storage for decode.

## Decision

**KEEP as an opt-in path together with EXP-0309.**

Use both flags for the qualified candidate:

```text
MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_FFN=1
MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_O=1
```

## Follow-up

Retest longer generation and context boundaries before making either flag a
default. The remaining largest attention stage is QK projection.
