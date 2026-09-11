# EXP-0314 — M25 P512 source-delta retest after stall recovery

**Status:** KEEP; stall not reproduced
**Milestone:** M25
**Date:** 2026-09-12
**Baseline commit:** `3fbe0f1`
**Candidate commit:** `08dc691`

## Question

Does the retained M25-J source delta explain the P512 stall reported by
EXP-0313, and does current main still qualify the M25-H/I path?

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; model SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- exact 512-token repeated fox prompt, context capacity 1024
- clean environment (`env -i` with only the benchmark vector and runtime paths)
- `MIINFER_MX_Q8_BATCH=0` in every run
- fixed observed clocks: SCLK/MCLK `1606/1000 MHz`
- one concurrent 250 ms `sample-gpu.sh` telemetry capture per cross-commit
  candidate run; each contained 121 samples at `1606/1000 MHz`

The qualified vector was:

```text
MIINFER_CONTEXT_CAPACITY=1024
MIINFER_PREFILL_LAYER_MAJOR=1
MIINFER_PREFILL_WIDE_CHUNK=1
MIINFER_PREFILL_FULL_LAYER_MAJOR=1
MIINFER_PREFILL_WIDE_ATTN=1
MIINFER_PREFILL_WIDE_REPACKED_MMQ=1
MIINFER_PREFILL_WIDE_MMQ_QKV=1
MIINFER_PREFILL_WIDE_MMQ_SSM_OUT=1
MIINFER_PREFILL_WIDE_MMQ_FFN=1
MIINFER_M23_REPACKED_ROW128=1
MIINFER_PREFILL_REPACKED_RESIDENT_ALL=1
MIINFER_PREFILL_CHUNK=512
MIINFER_PREFILL_MX_GDN=1
MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ=1
MIINFER_MX_Q8_BATCH=0
```

H/I added these two selectors:

```text
MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_FFN=1
MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_O=1
```

The control omitted both selectors.

## Results

| build | path | P512 latency | P512 tok/s | device bytes |
| --- | --- | ---: | ---: | ---: |
| `3fbe0f1` | H/I | `2476.43 ms` | `206.75` | `18,472,649,044` |
| `08dc691` | H/I | `2527.48 ms` | `202.57` | `18,472,649,044` |

The two H/I runs were both complete, finite, and clock-qualified. The
cross-commit difference is within the observed run variation and does not
support attributing the earlier stall to M25-J.

Current-main interleaved P512 smoke runs, with the same clean vector, were:

| path | tok/s samples | median |
| --- | --- | ---: |
| H/I | `214.89, 204.19, 212.86` | `212.86` |
| control | `177.04, 177.30, 180.05` | `177.30` |

H/I used `18,472,649,044` device bytes; control used `19,108,282,708`.

## Correctness

Current main completed an exact P512 plus one-token continuation with
`MIINFER_DUMP_TOKENS=1`: token ID `13477` (`brown`). The decode step took
`2.76 ms`.

The Release build and 24/24 Release CTest result from EXP-0313 remain valid.

## Diagnostic note

An exploratory run that added `MIINFER_MX_PIPELINE=1` completed at
`85.24 tok/s`. That flag selects the previously rejected M25-C register
pipeline, not the qualified H/I path, and is excluded from this experiment.
This confirms that the benchmark vector must explicitly leave that selector
unset.

## Interpretation

The P512 stall was transient device/runtime or harness state, not a
reproducible regression from M25-J. After the GPU recovered, both the pre-J
baseline and current main completed the same H/I workload above 200 tok/s.

## Decision

**KEEP** M25-H/I as an opt-in qualified path. **RETEST** M25-J separately;
this experiment intentionally keeps it disabled and makes no J performance
claim. Do not promote H/I to default until the longer-generation and
same-process repeat-P512 checks are added.

## Follow-up

Add the same-process `P512 → one-token continuation → P512` test and run the
H/I `00/10/01/11` matrix at P512 before default promotion. Continue source
differential work only after those gates pass.

## Re-evaluation — same-process repeat check — 2026-09-12

The CLI now exposes the minimal specialized check:

```text
--repeat-p512-check
```

On current main it passed with:

```text
same_process_p512_check=PASS
first_token=13477(brown)
first_prefill_ms=2467.63
continuation_ms=2.80049
repeat_prefill_ms=2554.99
```

The check uses the existing engine reset path between the first generation
and repeat prefill, and keeps the same process and allocation pool alive.
The P512 H/I `00/10/01/11` matrix remains a P512 qualification follow-up.
