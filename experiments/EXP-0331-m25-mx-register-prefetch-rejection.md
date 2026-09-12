# EXP-0331 — Pinned Mx MMQ register weight prefetch

## Hypothesis

The pinned mx-llama P512 MMQ contract prefetches the next row's weight words
and metadata into registers before the current DP4A chain. Porting that
schedule into MIInfer's legacy Mx MMQ kernel will reduce LDS wait time.

## Motivation

The recurrent FFN Gate/Up and Down families remain the largest measured
recurrent cost. This was the smallest remaining source-level differential in
the pinned Q4_K Mx MMQ inner loop after the interleaved DP4A order was already
ported and kept.

## Baseline

The committed staged Mx MMQ kernel with direct per-row LDS loads and the
interleaved Q4/Q5 DP4A order.

## Candidate

Temporarily load the next row's `lo`, `hi`, and scale/offset slot into
registers before computing the current row in the legacy Mx MMQ kernel. The
candidate had no runtime selector and was removed after screening.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Release binaries built from the same working tree, exact repeated-fox P512
  prompt
- `MIINFER_MX_Q8_BATCH=0`; `MIINFER_MX_PIPELINE` unset
- Explicit M25-H/I vector, clean `env -i` process
- Identical allocation: `21,993,242,964 B` live and
  `11,484,004,352 B` free of `34,342,961,152 B`

## Benchmark

Three interleaved fresh-process pairs in order `C A C A C A`, with
`--max-tokens 0 --no-stream`. This was a screening series, not a promotion
qualification.

## Correctness

The candidate passed the real same-process continuation/repeat check:

```text
same_process_p512_check=PASS first_token=13477(brown)
continuation_token=37550 first_prefill_ms=2385.64
continuation_ms=500.87 repeat_prefill_ms=2523.99
```

## Results

| order | path | P512 ms | tok/s |
| ---: | --- | ---: | ---: |
| 1 | control | 2504.55 | 204.43 |
| 2 | candidate | 2427.09 | 210.95 |
| 3 | control | 2435.91 | 210.19 |
| 4 | candidate | 2465.33 | 207.68 |
| 5 | control | 2437.66 | 210.04 |
| 6 | candidate | 2392.10 | 214.04 |

The control median was `2437.66 ms` / `210.04 tok/s`; the candidate median was
`2446.21 ms` / `209.30 tok/s`, `0.35%` slower by robust latency. Pair deltas
were `-77.46`, `+29.42`, and `-45.56 ms`, so the apparent lower candidate
mean was not a stable effect.

## Profiling

No profiler run was justified after the interleaved median failed to improve.
The candidate adds live register state and may trade LDS latency for register
pressure; this remains an interpretation, not a measured attribution.

## Decision

**REJECT.** Restore the direct LDS-load schedule. Keep the committed Mx MMQ
kernel unchanged.

## Follow-up

Do not revisit this schedule without a lower-variance five-plus-sample result
or a profiler showing the expected LDS bottleneck. The P512 stretch remains
open.
