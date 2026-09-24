# EXP-0394 — Physical-B512 recurrent projections for logical B128

**Status:** `REVERTED`
**Disposition:** `REJECT`
**Date:** 2026-09-24
**Base commit:** `dcfa069`

## Hypothesis

The qualified Mx repacked MMQ path may have a severe shape-efficiency collapse
at physical `M=128`. Executing stateless recurrent projections at physical
`M=512`, while keeping the logical B128 state evolution unchanged, might reduce
the pathological B128 wall time.

## Candidate

An opt-in candidate was implemented behind `MIINFER_EXP0394_B128_PROJ512=1`.
It changed only the existing Mx MMQ projection row count for logical B128:

- logical B128 rows: `128`
- candidate physical projection rows: `512`
- stateful recurrent width: `128`
- attention: unchanged
- GDN, convolution/history, beta/decay, normalization, residual, SwiGLU,
  position advancement, and KV writes: unchanged

The candidate zero-padded Q8 rows `128..511`, reused the existing capacity-sized
Q8/output workspaces, consumed only logical rows, and added per-family route
counters. No GPU kernel was added.

The implementation was reverted after the correctness gate failed. The
temporary source/harness changes are not present in the final tree.

## Workspace and route proof

The qualified preset configured `MIINFER_PREFILL_CHUNK=512`; existing per-layer
Q8 and projection output workspaces were capacity-sized for 512 rows. The
candidate route counter observed, for each candidate B128 invocation:

```text
qkv=48 gate=48 ssm_out=48 ffn_gate=48 ffn_up=48 ffn_down=48
```

This matches one candidate call per family for each of the 48 recurrent layers.
The control side reported zero candidate calls. The candidate predicate was
restricted to logical `token_count == 128`, so P512 could not enter it.

## Correctness gate

Exact P640 was run as `B512 + B128` on one engine, alternating control and
candidate with `reuse_session=false`. Each side processed exactly 640 prompt
tokens, completed 16 generated tokens, and remained finite.

| Pair | Order | Control first token | Candidate first token | Control prefill ms | Candidate prefill ms |
|---:|:---:|---:|---:|---:|---:|
| 1 | C→A | 561 | 48948 | 15999.3 | 16973.3 |
| 2 | A→C | 561 | 48948 | 15586.8 | 16633.3 |
| 3 | C→A | 561 | 48948 | 15631.1 | 17081.8 |
| 4 | A→C | 561 | 48948 | 15863.5 | 16660.1 |

The first token changed on all four candidate calls. This is a hard semantic
failure. Final-hidden/logit/top-10 numerical comparison was not run because the
required first-token gate had already failed; no continuation could be treated
as an accepted B128 trajectory.

## P640 end-to-end timing observed during the correctness matrix

These calls included the required 16-token continuation and are diagnostic
evidence, not a clean max-new-tokens-zero performance qualification.

| Pair | Order | Control total ms | Candidate total ms | Delta ms | Candidate/control |
|---:|:---:|---:|---:|---:|---:|
| 1 | C→A | 23777.1 | 24335.8 | +558.7 | 1.0235x |
| 2 | A→C | 23239.0 | 24303.6 | +1064.6 | 1.0458x |
| 3 | C→A | 23365.5 | 24784.4 | +1418.9 | 1.0607x |
| 4 | A→C | 23527.0 | 24327.8 | +800.8 | 1.0340x |

Control median was `23446.25 ms`; candidate median was `24331.80 ms`, a
`+885.55 ms` (`+3.78%`) candidate slowdown. Candidate was slower in all four
pairs. The 15%/2-second harvest gate therefore failed independently of the
semantic failure.

P896 was not run because P640 correctness failed. P512 was not rerun because
the candidate was restricted to logical B128 and was reverted immediately.

## Decision

`REVERT`. The physical-B512 recurrent-projection hypothesis is rejected for
this implementation. The result does not authorize further QKV/SSM/FFN
projection-width decomposition or another attribution chain.

- B128: candidate not viable; original B128 route remains pathological.
- B64: **REJECTED — CURRENT M28 FRONTIER**.
- B4: **BLOCKED**.

**ONE next PRIMARY:** choose a materially different, bounded B128 runtime
intervention; do not reopen physical projection-width archaeology without new
evidence.

The physical-B512 recurrent-projection candidate was reverted. No further
projection-width archaeology is authorized without new evidence.
