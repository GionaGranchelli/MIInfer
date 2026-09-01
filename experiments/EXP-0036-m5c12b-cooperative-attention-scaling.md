# EXP-0036 — M5-C12b cooperative attention scaling

**Status:** CLOSED — history-partition candidate REJECTED; production unchanged  
**Milestone:** M5  
**Date:** 2026-09-01  
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Question

How does the accepted cooperative cached-attention path scale from P64 to
P1024, and can one four-Wave64 history-partitioned candidate reduce that slope
without violating the deterministic inference contract?

## 2. Baseline and method

The production path used the accepted M5-C2 cooperative kernel, shared Gate/Up
Q8 reuse, GPU argmax, and stable-peak MI50 clocks:

```text
SCLK 1725 MHz
MCLK 1000 MHz
```

The current production scaling audit used prompt token `14990`, positions
`64,128,256,512,1024`, and the current Release build. The raw artifact is
retained at `bench/results/20260901T-c12b-scaling/position-audit.json`.

The candidate was an opt-in `MIINFER_ATTENTION_KERNEL=history` kernel. It used
four Wave64s to partition cache positions and local max, softmax-denominator,
and value reductions while retaining the existing KV layout. It was not made
the default.

## 3. Production scaling results

| Position | Wall ms | Whole-token GPU ms | Attention ms | Deferred GPU ms |
|---:|---:|---:|---:|---:|
| 64 | 19.665 | 19.533 | 4.932 | 27.392 |
| 128 | 24.109 | 24.241 | 9.491 | 31.737 |
| 256 | 33.100 | 33.175 | 18.607 | 40.747 |
| 512 | 51.161 | 51.476 | 36.775 | 58.912 |
| 1024 | 88.678 | 88.799 | 74.071 | 96.491 |

Structural counters remained constant at every position:

```text
1553 dispatches/token
38 synchronization sites/token
0 temporary allocations/token
589828 residual copy bytes/token
```

Attention increments were approximately linear across every interval:

```text
P64→P128:    +4.559 ms
P128→P256:   +9.116 ms
P256→P512:  +18.168 ms
P512→P1024: +37.296 ms
```

The accepted cooperative implementation therefore has no new pathological
collapse through P1024. Its remaining cost is a predictable linear history
scan.

## 4. Candidate correctness result

The history-partitioned candidate compiled successfully and left the default
production path unchanged. The full decode gate rejected it immediately:

```text
control first generated token:   8
history candidate first token: 8673
```

The short candidate audit produced:

```text
8673, 57497, 79299, 101209, 73541, 115703, 67893, 17515, 2416
```

against the production prefix:

```text
8, 341, 286, 470, 330, 9707, 11, 330, 488
```

The candidate changes the dot-product, softmax, and value accumulation order.
That is not acceptable for the established autoregressive production contract.
No candidate performance result is used for promotion because it fails the
mandatory correctness gate before a valid end-to-end comparison exists.

The unchanged default path passes the complete Release CTest suite:

```text
19/19
```

## 5. Interpretation

The C12b candidate disproves the assumption that history partitioning can be
introduced as a mechanically equivalent four-wave reduction. The numerical
contract is stricter than matching tensor shapes and implementing a stable
softmax formula: changing reduction order changes the selected token.

The C12b scaling audit still provides a useful production result. The current
cooperative attention kernel is the only demonstrated context-growing cost,
but it is no longer pathological. A future improvement must either preserve
the required reduction/materialization semantics or establish a new validated
numerical contract before considering performance.

## 6. Decision

```text
C12b scaling characterization: KEEP as evidence
C12b history-partition candidate: REJECT
Production attention selection: unchanged
```

No C12c implementation is preselected. Any follow-up must begin with a new
numerically faithful attention mechanism, not another arbitrary wave or
split-history geometry.
