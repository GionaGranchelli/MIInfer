# EXP-0257 — M12 gfx906 Gated DeltaNet chunk prototype

## Hypothesis

The exact chunkwise/WY formulation can run on gfx906 with one workgroup per
value head and expose reusable 64-token state transitions without changing the
decode path.

## Candidate

Implemented `launch_m12_gdn_chunk` as a fixed-shape M12 lab primitive:

- 16 key heads
- 32 value heads
- state size 128
- 128 tokens in two sequential 64-token chunks
- one workgroup per value head
- reusable global workspace for the WY matrices and solved values
- host-ordered chunk launches preserve the recurrent state dependency

The candidate is intentionally simple and fixed-shape. It is a kernel
feasibility probe, not a production prefill scheduler.

## Baseline

The control is the existing
`launch_qwen35_deltanet_state_update_transposed_no_decay_store` kernel launched
once per token. Both paths use the same synthetic inputs, nonzero initial state,
and FP32 state/output buffers.

## Environment

- GPU: AMD Instinct MI50 / gfx906
- Build: `mi50-release`, gfx906, Release
- ROCm: 6.4.0 / LLVM 20
- Geometry: 16/32 heads, state 128, 128 tokens, chunk 64

## Correctness

The GPU candidate agrees with the independent host token recurrence:

```json
{"tokens":128,"chunk":64,"key_heads":16,"value_heads":32,"state_size":128,
 "max_output_error":0.000000013,"max_state_error":0.000000149}
```

## Results

Median GPU timings over five measured runs after one warmup:

| Path | Time |
| --- | ---: |
| Existing token recurrence, 128 launches | 11100.63 us |
| Chunkwise prototype, 2 launches | 9987.03 us |
| Chunkwise speedup | 1.112× |

## Interpretation

The chunkwise algebra and state layout are correct on gfx906, and the two
chunk launches remove most per-token dispatches. The current kernel is only a
1.112× core speedup, so it does not by itself justify production integration or
the approximately 2× M12 dense-prefill gate. Its global workspace and serial
host chunk schedule also leave substantial optimization work before it can
expose the B128+ matrix path from EXP-0255.

## Decision

**KEEP AS A GPU CORRECTNESS PROTOTYPE; DO NOT PROMOTE TO PRODUCTION.** The
decode path remains unchanged.

## Follow-up

Profile the prototype before adding more machinery. The next viable test is to
reduce workspace traffic and fuse the chunk-local preparation/solve with the
head computation. If that cannot expose a materially larger batch window, stop
the dense-prefill branch rather than building a second runtime.
