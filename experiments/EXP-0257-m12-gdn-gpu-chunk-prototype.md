# EXP-0257 — M12 gfx906 Gated DeltaNet chunk prototype

> **Superseded geometry note:** early measurements used an incorrect
> 32-value-head description. The qualified geometry is 16 key heads / 48 value
> heads / state 128; see EXP-0258 and the later M12 re-evaluation.

## Hypothesis

The exact chunkwise/WY formulation can run on gfx906 with one workgroup per
value head and expose reusable 64-token state transitions without changing the
decode path.

## Candidate

Implemented `launch_m12_gdn_chunk` as a fixed-shape M12 lab primitive:

- 16 key heads
- 48 value heads
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

The corrected 48-head kernel is 3.17× faster than the token recurrence in the
isolated core benchmark, but it uses a fixed 64-token chunk and a serial host
launch schedule. That result is sufficient to test runtime composition, not to
claim a production backend or the full M12 gate.

## Decision

**KEEP AS A GPU CORRECTNESS PROTOTYPE; DO NOT PROMOTE TO PRODUCTION.** The
decode path remains unchanged.

## Re-evaluation — 2026-09-08

The runtime integration uses one shared five-buffer workspace and an isolated
raw-output buffer across recurrent layers. At P512 with HIP graphs disabled,
the corrected chunkwise path measured 47.79 tok/s / 10714.43 ms against a
matched 46.50 tok/s / 11011.19 ms control. A greedy P64 one-token check
produced identical output. The runtime path remains opt-in via
MIINFER_PREFILL_GDN_CHUNKWISE=1; decode is unchanged.

An opt-in 128-token staging capacity was added to expose the dense experiment.
At an exact 512-token prompt, 128-token scheduling measured 47.64 tok/s
versus 47.74 tok/s at 64, so 128 is retained for isolated experiments but is
not promoted. Non-128 prompt lengths automatically use 64-token chunks to
avoid a large partial-chunk fallback.

## Follow-up

Profile the prototype before adding more machinery. The next viable test is to
reduce workspace traffic and fuse the chunk-local preparation/solve with the
head computation. If that cannot expose a materially larger batch window, stop
the dense-prefill branch rather than building a second runtime.

## Corrected isolated result — 2026-09-08

The original 32-value-head result above is historical. With the actual
16-key-head/48-value-head Qwen3.8 geometry, the standalone benchmark reports
chunkwise 3826 us versus token recurrence 12141 us, or 3.17x, with maximum
output error 1.4e-8 and final-state error 1.5e-7.
