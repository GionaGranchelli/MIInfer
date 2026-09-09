# EXP-0284 — M22 recurrent B4 tail-family profile

## Hypothesis

The deferred recurrent tail must be split into dispatch families before
choosing the next prefill candidate.

## Baseline and candidate

Both runs used the release MIInfer binary, the exact Qwen3.8-27B-Q4_K_M model,
`MIINFER_PREFILL_LAYER_MAJOR=1`, and a P512 PP-only request. The profiler
selected token 511 from the final B4 group. The candidate additionally set
`MIINFER_PREFILL_DENSE_PROJECTIONS=1`.

Raw artifacts:

* baseline: `results/m22-prefill-profile/20260909-215225-2022240/`
* FFN candidate: `results/m22-prefill-profile/20260909-215608-2064107/`
* an earlier candidate capture aborted in the profiler because a deferred
  family had a start event without an end event; it is retained at
  `results/m22-prefill-profile/20260909-213802-1979355/`.

## Environment

* Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* GPU: AMD Instinct MI50/gfx906, SCLK 1606 MHz, MCLK 1000 MHz
* Reference: mx-llama commit `2e9d29fe736969160f17476ec6f0a6298cee6966`

## Results

| path | prefill ms | PP tok/s | whole deferred tail ms | peak VRAM bytes |
|---|---:|---:|---:|---:|
| B4 baseline | 11,161.86 | 45.87 | 966.63 | 29,973,676,372 |
| FFN gate/up dense candidate | 9,188.69 | 55.72 | 717.35 | 29,973,676,372 |

Selected B4 recurrent-family GPU sums across the 48 recurrent layers:

| family | baseline ms | FFN candidate ms |
|---|---:|---:|
| GDN core | 7.87 | 7.84 |
| SSM output and residual | 5.97 | 5.96 |
| FFN gate/up and SwiGLU | 16.87 | deferred to the dense batch path |
| FFN down and residual | 16.64 | deferred to the dense batch path |

The selected-family timings are dispatch-local attribution for one B4 group;
the whole-tail timing is the qualification-wide selected chunk. They are not
added together as a wall-time decomposition.

## Interpretation

Dense FFN gate/up removes the largest quantized B4 tail portion visible in the
selected group and improves whole P512 PP by 21.5% versus this capture. The
remaining whole-tail bucket is still 717 ms in the sampled chunk, while the
batch preparation bucket remains 245.63 ms. The next candidate must therefore
target the remaining recurrent/SSM or preparation dataflow, not repeat the
already accepted FFN conversion.

## Decision

KEEP the profiler and the FFN candidate as opt-in research paths. This
measurement does not qualify either path for production or establish parity.

## Follow-up

Use the family breakdown together with the existing Q5 SSM and M13 matrix
rejection records to evaluate a new causal chunkwise projection design.
