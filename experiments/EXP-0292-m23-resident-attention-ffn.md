# EXP-0292 — Resident attention FFN MMQ weights

## Hypothesis

The remaining attention FFN upload bucket can be removed by keeping each
attention layer's packed Gate/Up/Down tiles resident, without changing the
causal attention schedule.

## Candidate

Set `MIINFER_PREFILL_REPACKED_RESIDENT_ATTN_FFN=1` with the established
wide/repacked M23 switches. Recurrent FFN residency is disabled so the
allocation and upload tradeoff is isolated.

## Environment

- AMD Instinct MI50 / gfx906, ROCm 6.4.0
- Qwen3.8-27B-Q4_K_M
- exact P512 prompt, context capacity 1024
- full layer-major, projection width B512, row-128 repacked MMQ

## Correctness

The one-token continuation completed and produced `brown`. No NaN or invalid
output was observed.

## Results

| run | prefill | allocation |
| --- | ---: | ---: |
| P512 smoke | 70.56 tok/s | 22,915,518,804 B |
| P512 continuation | 63.56 tok/s | 22,915,518,804 B |
| profiled P512 | 69.89 tok/s | 22,915,518,804 B |

The profiled run recorded 16,504,586,240 bytes of transient repacked uploads;
attention FFN uploads fell to zero, while recurrent FFN uploads remained
3,476,029,440 B (Gate), 3,476,029,440 B (Up), and 3,609,722,880 B (Down).
The candidate adds about 3.52 GB of
resident packed attention FFN weights versus the non-resident control.

## Decision

**RETEST.** The isolated candidate is memory-safe at context 1024 and removes
the intended upload bucket, but one interleaved pair is not enough to claim a
performance win; it is also slower than the recurrent-FFN-resident candidate.
Do not combine both residency modes: their measured allocations would exceed
the MI50 budget.

## Follow-up

Only rerun with 3–5 interleaved pairs if attention residency becomes relevant
after the projection/causal-width refactor. No micro-tuning is justified yet.
