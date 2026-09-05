# EXP-0124 — M6-B32 transposed recurrent no-decay store

## Question

Can the production-selected transposed DeltaNet state layout also avoid the
intermediate decayed-state global store without changing the recurrent
contract?

## Hypothesis

The B30 transposed kernel still writes the decayed state during its first pass
and reads it again during the final update. Reusing the original state value
in the first pass should remove that global store while retaining the same
transposed layout and recurrence.

## Baseline and candidate

The control is the B30 transposed state-update kernel, selected with
`MIINFER_DELTA_TRANSPOSED_NO_DECAY_STORE=0`. The candidate is the same
transposed state layout with the decayed-state intermediate kept in registers
and not written globally; it is the default path.

No quantization, state layout, model weights, or output representation changed.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Baseline:  f3cf919
```

## Commands

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --generate16
MIINFER_DELTA_TRANSPOSED_NO_DECAY_STORE=0 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --generate16
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --prefix64-observable-contract
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --bench64
MIINFER_DELTA_TRANSPOSED_NO_DECAY_STORE=0 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --bench64
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --bench128
MIINFER_DELTA_TRANSPOSED_NO_DECAY_STORE=0 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --bench128
```

## Correctness

Both native generation paths produced 16 tokens with first token `11`, last
token `585`, and deterministic replay. Decode allocations were `0` and
tracked/peak device bytes were unchanged at `17,019,965,780`.

The complete P64 external observable run passed under the established
margin-aware contract. All teacher-forced positions in the run matched;
final P64 logits had cosine `0.999603`, top-5 overlap `4/5`, and matching
argmax `8719`. The known late recurrent-state envelope diagnostics remained
bounded and non-fatal. Release CTest passed `20/20`.

## Benchmark

Five-sample medians per process at stable_peak:

| Workload | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 ms | 5396.01 | 5305.16 | -1.68% |
| TG64 tok/s | 11.8606 | 12.0637 | +1.71% |
| TG128 ms | 10933.3 | 10757.8 | -1.61% |
| TG128 tok/s | 11.7073 | 11.8983 | +1.63% |

The 16-token A/B also favored the candidate: second decode throughput was
`12.2263` versus `12.0085` tok/s (`+1.81%`).

## Interpretation

The candidate removes a real intermediate global state store from the
transposed recurrent update. It preserves native replay and the accepted
external observable behavior while producing a repeatable roughly 1.6–1.7%
whole-token gain at both tested generation lengths. Device usage is unchanged.

## Decision

**KEEP; production-selected.** The no-decay-store transposed kernel is the
default. `MIINFER_DELTA_TRANSPOSED_NO_DECAY_STORE=0` retains the B30 control
for future A/B comparisons.

## Follow-up

Refresh the post-B32 profile and select the next measured optimization. Do not
return to rejected FFN geometry or split-K experiments without new evidence.
