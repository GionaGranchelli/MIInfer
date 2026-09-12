# EXP-0330 — Mx recurrent FFN Gate/Up pair kernel

## Hypothesis

Computing the recurrent Q4_K Mx Gate and Up projections in one workgroup will
reuse the staged Q8 activation and reduce P512 prefill time.

## Motivation

The current profile identifies recurrent FFN Gate/Up execution as the largest
remaining family. The candidate was an isolated MIInfer-native experiment; the
pinned mx-llama source has no equivalent P512 fused Gate/Up MMQ contract.

## Baseline

The committed separate Mx Q4_K MMQ launches for Gate and Up, with the
qualified interleaved DP4A order.

## Candidate

A temporary opt-in `MIINFER_PREFILL_WIDE_MX_FFN_PAIR=1` kernel staged both Q4_K
weight tiles and accumulated Gate and Up in one launch. The default path was
unchanged and the candidate was removed after screening.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, exact repeated-fox P512 prompt
- Release build, `MIINFER_MX_Q8_BATCH=0`, `MIINFER_MX_PIPELINE` unset
- Explicit M25-H/I runtime vector, clean `env -i` process
- Device allocation: `21,993,242,964 B` live; `11,484,004,352 B` free of
  `34,342,961,152 B`

## Benchmark

One matched fresh process per side, `--max-tokens 0 --no-stream`, with the
same prompt and explicit environment. This was a screening comparison, not a
qualification series.

## Correctness

The candidate passed the real same-process check:

```text
same_process_p512_check=PASS first_token=13477(brown)
continuation_token=37550 first_prefill_ms=2762.22
continuation_ms=517.824 repeat_prefill_ms=2743.23
```

## Results

| path | P512 ms | tok/s |
| --- | ---: | ---: |
| separate Gate/Up | 2540.09 | 201.57 |
| paired Gate/Up | 2747.92 | 186.32 |

The paired kernel was `8.18%` slower at unchanged allocation and VRAM use.

## Profiling

No profiler run was justified after the clear wall-time regression. The
candidate doubled per-thread accumulators and LDS weight storage, making
register/LDS pressure the likely mechanism; that is an interpretation, not a
measured attribution.

## Decision

**REJECT.** Remove the pair kernel and its selector. Keep the committed
separate Mx Gate/Up launches.

## Follow-up

Do not revisit fused P512 Gate/Up without a lower-pressure design or a new
measured external contract. The stretch remains open; the qualified H/I path
and recurrent state contract are unchanged.
