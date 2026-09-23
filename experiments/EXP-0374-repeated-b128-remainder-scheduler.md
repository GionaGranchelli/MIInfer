# EXP-0374 — Qualify repeated-B128 remainder scheduler

## Question

Can the qualified B128 partial route be repeated for arbitrary remainders,
leaving only a final residual below 128 on the existing fallback path?

## Scheduler

The opt-in selector is `MIINFER_EXP0374_REMAINDER_SCHED=1`. After complete
B512 work, the scheduler repeatedly selects B128 while the remainder is at
least 128, then leaves the final `<128` tokens to the existing path. The
complete-B512 branch and decode path are unchanged. Cheap host counters report
the decomposition and scalar residual work; they add no synchronization to
ordinary timing.

## Structural results

| Point | Actual tokens | Decomposition | Scalar residual layer work |
|---|---:|---|---:|
| P640 basis | 640 | `1×B512 + 1×B128` | 0; qualified by EXP-0373 |
| P768 | 768 | `1×B512 + 2×B128` | 0 |
| P896 nearest | 898 | `1×B512 + 3×B128 + 2` | 64 calls / 128 token-layer calls |
| P1022 | 1022 | `1×B512 + 3×B128 + 126` | 64 calls / 8064 token-layer calls |
| P1024 | 1024 | `2×B512` | 0 |
| Later base | 1788 | `3×B512 + 1×B128 + 124` | 64 calls / 7936 token-layer calls |

The counters prove that B128-covered tokens do not enter per-token layer work.
The nearest later-base case proves base advancement across complete chunks and
a subsequent B128 at a later absolute position. The state export was finite;
no stale final slot, skipped token, or duplicated token was observed.

## Semantic qualification

At exact P768, control and scheduler final hidden states were byte-identical
and finite. Both ordinary direct-decode continuations produced the same 16
token IDs (`37550` repeated in this state-import probe).

The existing EXP-0373 P640 semantic qualification remains the B128 basis:
matching argmax/top-10 behavior and identical 16-token continuation. The
nearest P898 and P1788 direct-generation checks produced matching decoded
continuation text for all 16 requested tokens; the hermetic preset stripped the
optional token-dump selector, so those two checks are recorded as decoded-text
parity rather than fresh token-ID lists. No material final-model divergence
was observed. No KV prefix mutation or non-finite state was observed.

## Timing (pre-correction; repeated-B128 route not qualified)

Clean non-profiled scheduler/control timing was collected for the bounded
structural points:

| Point | Control prefill | Scheduler prefill | Result |
|---|---:|---:|---|
| P768 | `29,613.46 ms` | `29,469.05 ms` | `0.49%` faster |
| P1022 | `83,345.80 ms` | `109,056.57 ms` | residual-dominated regression |
| P1024 | `5,068.23 ms` | `5,114.12 ms` | `0.91%` slower |
| P1788 scheduler | — | `85,288.18 ms` | structural only |

Peak tracked allocation was unchanged at `22,060,355,988 B`. The P1022
comparison is not conclusive: source review found that the EXP-0374 dispatcher
did not route `count == 128` through `prefill_full_layer_major_chunk()` because
its divisibility condition also required divisibility by 512. These timings do
not qualify repeated-B128 performance or prove `<128` residual dominance.
EXP-0374 semantic and structural results remain valid; dispatch and performance
are requalified by EXP-0375.

Hardware remained at the qualified 1606 MHz SCLK / 1000 MHz MCLK policy; the
external fan is physically fixed at full speed and ROCm fan telemetry was not
used.

## Decision

**SEMANTICS_QUALIFIED_ONLY** — scheduler decomposition and exact P768 semantics
passed, but repeated-B128 performance was not qualified because the authored
B128 chunks were dispatched through the wrong route. It is not production-
default pending EXP-0375.

## Next PRIMARY

EXP-0375: correct and remeasure repeated-B128 dispatch. Do not begin B64/B4
residual work until corrected repeated-B128 timing and P1022 attribution exist.

No new GPU math kernel was implemented.
