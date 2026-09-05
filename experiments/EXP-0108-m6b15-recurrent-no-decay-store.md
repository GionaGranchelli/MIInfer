# EXP-0108 — M6-B15 recurrent state-update no-decay-store candidate

## Question

Can the DeltaNet state update avoid materializing the decayed state between
its dependency-ordered passes without changing externally observable Qwen3.8
behavior?

## Baseline and candidate

Baseline commit: `185e1c1` (M6-B14 MMVQ-style LM-head production path).
Candidate commit: `f15e5c4`.

The candidate keeps the recurrence's two passes and arithmetic inputs, but
computes the decayed value in a register during the key-dot pass instead of
writing it to global state and rereading it for the update pass. The candidate
is selected by default; `MIINFER_DELTA_NO_DECAY_STORE=0` retains the baseline
kernel for comparison.

## Environment and workload

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Compiler: Clang 20.0.0.rocm
Fixture: /tmp/m6a273-reference-p12
```

Commands:

```bash
MIINFER_DELTA_NO_DECAY_STORE=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --generate16
MIINFER_DELTA_NO_DECAY_STORE=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --bench64
MIINFER_DELTA_NO_DECAY_STORE=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --bench128
MIINFER_DELTA_NO_DECAY_STORE=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --prefix64-observable-contract
```

## Correctness

* Full external observable run completed through P64 with no observable
  contract failure; teacher-forced positions 0–63 were all `PASS`.
* P2 final logits: cosine `0.999789`, top-5 `5/5`, reference argmax `1318`
  and GPU argmax `1318`.
* P64 final logits: cosine `0.999588`, top-5 `5/5`, matching argmax `8719`.
* Native 16-token generation: first token `11`, last token `46194`, replay
  `PASS`, zero decode allocations.
* Release CTest: `20/20 PASS`.
* No NaN/Inf or host state round-trip was observed.

The strict internal state-envelope diagnostics continue to report known
bounded warnings in late recurrent layers. They are diagnostic only under the
accepted A27 external observable contract; no tolerance was changed.

## Profile

At P64 position 63:

| Metric | Control | Candidate |
| --- | ---: | ---: |
| total GPU event | 116.299 ms | 113.561 ms |
| layer-sum GPU event | 113.081 ms | 110.322 ms |
| L0 state-update stage | 0.303039 ms | 0.211680 ms |
| allocations | 0 | 0 |

The representative state-update stage fell by approximately 30%; the whole
token profile fell by approximately 2.4%.

## Benchmark

Five-sample serial medians at the same stable-peak policy:

| Workload | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| TG64 ms | 7372.48 | 7174.50 | -2.68% |
| TG64 tok/s | 8.68093 | 8.92048 | +2.76% |
| TG128 ms | 14875.40 | 14497.00 | -2.54% |
| TG128 tok/s | 8.60479 | 8.82940 | +2.61% |

Both candidate runs reported replay `PASS`, zero decode allocations, and
`17018712404` tracked/peak device bytes.

## Decision

**KEEP; production-selected.** The candidate removes a real state-memory
round-trip and improves both required generation workloads without violating
the accepted external correctness contract. `MIINFER_DELTA_NO_DECAY_STORE=0`
remains the explicit control.

## Follow-up

Refresh the production profile after selection and rank the remaining
recurrent-layer costs before choosing the next bounded optimization.
