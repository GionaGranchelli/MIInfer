# EXP-0041 — M5-C15 optimization closure and parity decision gate

**Status:** CLOSED — M5 optimization campaign complete; architectural choice pending  
**Milestone:** M5  
**Date:** 2026-09-02  
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Question

What has M5 established about MIInfer's production performance, which causes
of the llama.cpp differential have been experimentally eliminated, and should
the next phase preserve MIInfer's current execution contract or explore a new
reference-correct contract?

## 2. Authoritative production result

The accepted MIInfer path runs at stable peak on the MI50:

```text
SCLK: 1725 MHz
MCLK: 1000 MHz
```

The latest retained stable-peak result is:

| Workload | Result |
|---|---:|
| Short/PP1-style decode | 55.419 tok/s |
| P64 clean wall | 19.579 ms/token |
| P64 attention | 4.928 ms/token |
| P1024 attention | 74.071 ms/token |
| P1024 clean wall | 88.678 ms/token |
| Peak VRAM | 4.636 GiB |
| Dispatches | 1,553/token, flat |
| Synchronizations | 38/token, flat |
| Allocations | 0/token |

The context shape is now predictable: attention grows approximately linearly
with history, while the non-attention floor remains approximately 14.6–14.7
ms/token. The retained llama.cpp control is approximately 90.4–90.8 tok/s at
the same stable-peak clock state, subject to workload-harness differences.

## 3. M5 optimization record

### Accepted

| Change | Evidence | Decision |
|---|---|---|
| Cooperative cached attention | Removed the serial context collapse; preserved trajectory | KEEP |
| Persistent decode workspace | 1,086 allocations/token → 0; up to +32.3% short decode | KEEP |
| Resident norm weights | Copy traffic −37.2%, syncs −18.2%, +9.2% P64 | KEEP |
| Direct layer-output handoff | Layer copies 72 → 36; neutral performance, cleaner ownership | KEEP |
| Coarsened KV stores | 576 tiny KV memcpys → 0; +5.5% P64 | KEEP |
| GPU greedy argmax | Full logits D2H → 4-byte token result; approximately neutral/+0.5% | KEEP |
| Gate/Up Q8_1 reuse | 180/180 byte-identical checks, 64-token identity, +1.14% | KEEP; production-selected |

### Rejected or cleared

| Candidate | Result |
|---|---|
| Down four-Wave64 and split-K geometry | Correctness-valid but 27.7% / 46–48% slower; REJECT |
| Standalone FFN GEMV replacement | Gate/Up/Down tied or beat the retained MMVQ comparison; CLEAR |
| SwiGLU→Down Q8 fusion | +33.75% micro result but 64-token divergence at P38; REJECT |
| FFN norm→F16→Q8 fusion | Exact checks passed but clean decode regressed 5.217%; REJECT |
| History-partition attention candidate | First token changed from 8 to 8673; REJECT |
| LM-head Q6_K differential | External path is Q6_K×Q8_1, MIInfer is Q6_K×Q8_K; INVALID |
| Broad fixed-floor fusion/contract comparisons | No exact external contract established; NOT SELECTED |

## 4. Experimentally eliminated explanations

M5 established that the remaining throughput differential is not explained by:

* MI50 DPM state; stable-peak control changed MIInfer by only about 0.5%.
* The serial cached-attention defect; cooperative attention removed the
  catastrophic P64 collapse and showed no second scaling collapse through
  P1024.
* Growing dispatch, synchronization, allocation, or copy counts; these remain
  flat in the audited context range.
* Q4_0 × Q8_1 Gate, Up, Down, Q, K, V, or O GEMV quality; direct comparisons
  cleared the relevant shapes, with EXP-0009 addressing the historical K/V
  geometry issue.
* Obvious decode scaffolding; workspace, resident weights, layer handoff, KV
  stores, and full-logit transfer were separately addressed.
* Dispatch count alone; C10c removed 108 dispatches and still regressed 5.217%.

The remaining llama.cpp differential is real at the whole-token level but not
attributable to one proven same-contract kernel or to a single unmeasured
orchestration defect.

## 5. M5 result

M5 achieved a locally optimized, production-correct MI50 path under its
current execution contract. It did not demonstrate whole-runtime parity with
the strongest gfx906 llama.cpp control. The remaining comparison is limited by
representation and execution-contract differences, especially Q6_K×Q8_K versus
Q6_K×Q8_1 at the LM head and backend-specific norm, conversion, attention, and
materialization paths.

No further same-contract optimization is currently evidence-backed. Continuing
to select targets from broad MIInfer-only category timings would turn the next
phase into speculation rather than a falsifiable comparison.

## 6. Architectural decision gate

### Path A — preserve the exact current trajectory

Keep the current MIInfer contract:

```text
same internal precision boundaries
same quantized representations
same deterministic token trajectory
```

Under this path, M5 is complete and the approximately 55 tok/s implementation
is locally optimized enough for the current campaign. The next work should
prioritize longer-context capability, robustness, usability, model support,
benchmarking, and release engineering rather than indefinite llama.cpp parity
chasing.

### Path B — make llama.cpp-class performance a primary objective

Start a new phase, **M6-A — reference-correct execution-contract exploration**.
The acceptance contract changes from reproducing current MIInfer bytes and
trajectory to satisfying a pinned external Qwen3 correctness envelope.

This permits bounded experiments involving Q6_K × Q8_1 LM-head input,
llama.cpp-like activation representations, alternate precision/materialization
boundaries, reference-validated fusion, alternate attention reduction
structure, and graph-level representation planning.

Path B does not make llama.cpp arithmetic a golden tensor reference. It uses
model semantics and behavioral correctness as the authority, with bounded
numerical equivalence and performance measured separately.

## 7. Decision

```text
M5-C15: CLOSED
M5 optimization campaign: CLOSED
Production path: unchanged
Architectural fork: Path A or Path B requires explicit selection
```

No C15b or implementation experiment is authorized by this record alone.
