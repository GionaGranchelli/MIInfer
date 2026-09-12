# EXP-0345 — M25 parallel recurrent input branches

**Status:** REJECT; removed after measurement  
**Milestone:** M25 stretch investigation  
**Date:** 2026-09-12  
**Baseline:** `a87258e`  
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Hypothesis

The pinned Qwen3.5 graph constructs recurrent QKV/Z and beta/alpha from the
same normalized input as independent branches. MIInfer serialized beta/alpha
before QKV/Z. Running the existing beta/alpha preparation on a nonblocking
auxiliary stream while the existing Mx QKV/Z projections run on the main
stream might recover useful GPU overlap.

## Candidate

An opt-in `MIINFER_PREFILL_MX_PARALLEL_BETA_ALPHA=1` branch normalized once on
the normal stream, recorded an event, launched the existing beta/alpha GEMM
and preparation on an auxiliary stream, ran the unchanged Mx QKV/Z path on
the normal stream, and joined before GDN. No kernel arithmetic, weight layout,
default preset, or decode path changed.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- exact 512-token repeated-fox prompt, context capacity `1024`
- clean `env -i` processes with `MIINFER_PRESET=m25_hi_qualified`
- three interleaved fresh-process pairs, control first in each pair
- `--max-tokens 0 --no-stream` for the timing screen
- continuous 250 ms telemetry: `910` samples; all SCLK/MCLK samples were
  `1606/1000 MHz`; maximum junction temperature `63 C`
- both paths allocated `21,993,242,964 B` and reported
  `11,484,004,352 B` free after setup

## Correctness

The candidate passed the same-process state gate:

```text
same_process_p512_check=PASS first_token=13477(brown)
continuation_token=37550 first_prefill_ms=2459.87
continuation_ms=504.678 repeat_prefill_ms=2506.55
```

The timing samples also processed exactly 512 prompt tokens and produced
finite output.

## Results

```text
control:   2482.22, 2500.10, 2504.44 ms  (median 2500.10 ms)
candidate: 2504.20, 2486.88, 2857.58 ms  (median 2504.20 ms)
```

The candidate was `+4.10 ms`, or `+0.164%`, slower by median:

```text
control:   204.79 tok/s
candidate: 204.46 tok/s
```

The `2857.58 ms` candidate sample is retained as a raw outlier; it was not
discarded to improve the conclusion.

## Interpretation

The existing beta/alpha work is too small or otherwise not schedulable with
the QKV/Z projection in a way that helps this MI50 workload. The branch adds
stream/event synchronization and resource competition without measurable
benefit. This is execution-contract evidence against this particular
oracle-inspired overlap, not evidence that the pinned graph has no useful
composition differences.

## Decision

**REJECT.** Remove the selector and auxiliary stream. Keep the serialized
MIInfer recurrent input path and the qualified preset unchanged.

## Follow-up

Do not retry this overlap with the same branches. Remaining stretch work needs
a positive exact-shape differential or a more complete pinned graph contract,
not another generic stream experiment.
