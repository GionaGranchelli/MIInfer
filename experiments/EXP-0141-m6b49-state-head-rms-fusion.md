# EXP-0141 — M6-B49 recurrent state-update/head-RMS fusion

## Question

Can the production transposed, no-decay-store DeltaNet state update write
head-RMS-normalized output directly, removing the intermediate recurrent
output materialization and one subsequent normalization dispatch?

## Hypothesis

The state update already computes one recurrent output element per thread and
the next kernel immediately reduces those elements per head. Keeping the
recurrent value in the state-update kernel could reduce global traffic and
dispatch overhead without changing the recurrence arithmetic.

## Candidate

An opt-in `MIINFER_DELTA_STATE_HEAD_RMS_FUSED=1` kernel retained the existing
transposed state layout, no-decay-store recurrence, 128-thread head mapping,
and FP32 operations. It wrote head-RMS-normalized output directly and left
diagnostic capture paths on the existing implementation.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Baseline:  B41 production path
```

## Correctness

Native 16-token generation passed exact replay with first token `11`, last
token `585`, zero decode allocations, and unchanged tracked device usage of
`17,019,965,780` bytes. The diagnostic-capture path remained unchanged.

## Results

Fresh same-build serial medians:

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 13.8880 | 13.9261 | -0.27% |
| TG128 | 13.6582 | 13.6801 | +0.16% |

The differences are within run dispersion and do not show a repeatable
whole-token improvement. Candidate replay passed in both benchmark runs.

## Decision

**REJECT.** The fused state-update/head-RMS kernel does not improve end-to-end
decode. The candidate was removed; the B41/B32 transposed no-decay-store path
remains production-selected.

## Follow-up

Do not pursue this fusion boundary again without evidence of a materially
different kernel mapping. The recurrent state-update family remains a real
cost, but this simple materialization-elimination candidate is not a win.
Select the next experiment from a fresh profile and avoid blind fusion.
