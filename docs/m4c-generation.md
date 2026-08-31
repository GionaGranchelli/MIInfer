# M4-C1 — Deterministic first generated token

Status: `COMPLETE`

M4-C1 crosses the boundary from a validated single-token forward pass to
stateful model-level decode. It remains intentionally explicit-token and
greedy; tokenizer, sampling, serving, batching, and performance work are not
part of this slice.

## Decode contract

`Qwen3DecodeCache` and `Qwen3GpuDecodeCache` own one layer-scoped cache for
each of the 36 Qwen3 layers. Each cache preserves the M4-A layout and
semantics:

```text
[kv_head][position][head_dim]
```

Keys are post-RoPE, values are unmodified, and a token at position `t` is
accepted only when every layer cache currently has length `t`. Reset clears
all layer caches. The new decode entry points reuse the existing explicit
layer executor and final norm/Q6_K logits path; no generic graph or scheduler
was introduced.

## Deterministic fixture

The physical fixture uses the pinned token `14990`. Its accepted first greedy
token is `8`, established by the M4-B canonical MI50 logits gate. M4-C1 then
feeds that generated token back at position `1` using the same persistent
36-layer cache state.

## Acceptance

Both MI50 Debug and Release physical runs pass:

```text
prompt token:             14990
first generated token:    8
position-0 cache length:  1 across all 36 layers
position-1 cache length:  2 across all 36 layers
next host argmax:         341
next MI50 argmax:         341
reset determinism:        PASS
invalid position guard:   PASS
```

The position-zero incremental logits are bitwise identical to the existing
single-token GPU forward wrapper. Both repeated two-position executions are
bitwise deterministic and finite. The position-1 call demonstrates that the
generated token is consumed through persistent KV state rather than merely
re-running a stateless position-zero forward.

The non-vacuous physical command is:

```bash
scripts/run-m4c2-acceptance.sh \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

The artifact-free CTest entry remains a smoke/skip entry when no model path
is supplied; the physical command is the acceptance gate for the real model.
The full Debug and Release suites pass `18/18`, including the existing M4-A
four-position KV-cache tests.

## M4-C1 decision

**M4-C1 CLOSED**

## M4-C2 implementation

The short explicit-ID fixture and its independent reference provenance are
documented in [`tests/reference/qwen3/m4c2-greedy/README.md`](../tests/reference/qwen3/m4c2-greedy/README.md).
The sequence test feeds each actually selected token into the next position,
checks host/GPU agreement and all 36 cache lengths, and has a deterministic
GPU replay path.

The physical GPU-only matrix currently splits by build configuration. Debug
matches the reference through positions 0–2 but diverges at position 3: the
reference and host select `470`, while Debug MI50 selects `419`. The fixed
prefix diagnostic reports reference logits `470=22.85797`, `419=22.52974`,
versus Debug MI50 `470=22.4933`, `419=22.5219`. Release MI50 selects `470`
and passes all eight reference tokens plus replay determinism. C2 remains
open because the required Debug/Release behavior is not yet stable.

The fixed-prefix state dump shows position 0 is bitwise identical between
Debug and Release. Position 1 is the first build-sensitive step: outputs are
identical through layer 19, first differ at layer 20, and layer-21 K/V cache
entries are the first materially different cached state. The position-3
difference then grows gradually through layer 35. Serialized Debug produces
the same result, while RelWithDebInfo follows Release, so the current evidence
favors unoptimized HIP arithmetic/code generation over a synchronization race.

The physical command is:

```bash
scripts/run-m4c2-acceptance.sh \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

## M4-C2 decision

**M4-C2 OPEN — MI50 token divergence at position 3 (`470` vs `419`)**

Do not add tokenization, sampling, or performance optimization until the
sequence contract is resolved.
