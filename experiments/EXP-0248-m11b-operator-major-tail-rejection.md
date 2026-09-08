# EXP-0248 — M11-B operator-major recurrent tail rejection

## Hypothesis

The existing causal layer-major schedule interleaves sixteen B=4 recurrent
core/output-projection groups with their FFN tails. Running all causal core
groups first and all FFN tails second might improve locality by keeping each
large projection weight stream contiguous, without changing the qualified
kernels or causal state order.

## Candidate

An opt-in `MIINFER_PREFILL_OPERATOR_MAJOR=1` schedule for recurrent layers:

1. execute all B=4 recurrent core and Q5 output-projection groups;
2. execute all B=4 residual/FFN/down-projection tails.

The default schedule and all attention layers were unchanged.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`
- Prompt: repeated hardware sentence, 546 tokens
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- `MIINFER_HIP_GRAPH=0`
- 2 candidate/control runs in reversed order

## Correctness

The candidate and control produced the same one-token continuation for
`hello` (`,`). The schedule preserves causal recurrent ordering; only
dependency-free deferred tails were reordered.

## Results

| Path | Runs | Median PP | Relative |
| --- | --- | ---: | ---: |
| Existing interleaved B=4 schedule | 45.45, 45.45 tok/s | 45.45 tok/s | 1.000x |
| Operator-major candidate | 45.01, 45.29 tok/s | 45.15 tok/s | 0.993x |

The candidate was within run-to-run noise but did not produce a measurable
gain.

## Decision

**REJECT.** The temporary schedule switch and phase-splitting code were
removed. Production execution is unchanged.

## Follow-up

Operator ordering alone does not expose enough additional reuse. A remaining
M11-B attempt needs a genuinely different grouped projection decomposition and
an end-to-end causal schedule, not another ordering of B=4 launches.
