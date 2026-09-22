# EXP-0366 — Partial-tail prefill execution contract

## Hypothesis

After a complete B512 full-layer-major prefill, a remainder can be decomposed
into existing aligned batches plus a minimal scalar residue without fake
padding or a new math kernel. For `R=510`, the bounded contract is
`384 + 64 + 62`.

## Scope and baseline

This is an opt-in execution-contract experiment only. The qualified/default
path is unchanged. EXP-0365 measured fixed-capacity `P1022 = 80,973.89 ms`;
the B512 control at this build measured `2,514.36 ms`.

## Source support matrix

| Stage | Existing count contract | EXP-0366 status |
|---|---|---|
| Embedding/copy | Per-token path | Preserved |
| Recurrent `prepare_prefill_batch` | `1..capacity`, divisible by 4 | Existing fallback |
| Recurrent wide QKV, beta/decay, GDN/SSM, FFN | At least 128, divisible by 64 | Reused for aligned chunks |
| Full-attention `prepare_prefill_batch` | Default exactly 64 or 512 | Opt-in widened to aligned `128..512` |
| Full-attention finish | Any prepared positive count within capacity | Reused after preparation |
| `finish_prefill_batch` | Positive count divisible by 4 | Reused |
| Layer-major dispatcher | Complete B512 fast path; smaller counts may call `layer.run` | Opt-in decomposition |

The relevant source contracts are `prefill_layer_major`,
`prepare_prefill_batch`, `prefill_wide`, `finish_prefill_attention`, and
`finish_prefill_batch` in `tools/miinfer_cli.cpp` and
`tools/qwen35_gpu_pipeline.hpp`. Recurrent wide support is not evidence that
full attention has an equivalent partial-count schedule.

## Candidate

`MIINFER_EXP0366_PARTIAL_TAIL=1` preserves complete B512 chunks, decomposes
the post-B512 remainder into aligned chunks, and leaves the smallest residue
on the existing scalar route. The full-attention preparation guard is widened
only under this selector. No padding, new allocation, or new math kernel is
used; the default and qualified selectors do not enable it.

## Resource and correctness gate

The MIinfer release build completed successfully. P1022 allocation remained
`22,463,033,748` bytes; no new workspace was added.

P513 candidate and oracle continuation runs both completed, but the current
CLI has no comparable recurrent/KV state dump for an arbitrary partial prompt.
Generated token identity and intermediate state were therefore not
independently compared. The correctness gate is incomplete, so this candidate
is not qualified for promotion.

## Measurements

| Case | Result |
|---|---:|
| P512 control, default | 2,514.36 ms |
| P513 candidate, opt-in | 3,020.31 ms |
| P513 oracle continuation, default | 2,900.13 ms |
| P1022 candidate, opt-in | 82,260.50 ms |
| P1022 EXP-0365 baseline | 80,973.89 ms |

The P1022 profile recorded `2` aligned partial chunks and `62` scalar tokens.
The selector changed the route but did not recover end-to-end performance; the
clean P1022 result regressed by about 1.29 seconds. Profiled timing is not an
A/B claim because instrumentation changes synchronization and timing behavior.

## Interpretation

The first missing contract was real: by default, aligned partial counts above
64 were rejected by full-attention preparation, so attention layers could fall
back to per-token `layer.run` while recurrent layers had a wide path. Widening
that guard and decomposing the tail is not sufficient. The remaining
pathological stage/dataflow has not been isolated with non-double-counted
timing, and the available continuation check is not a state-level oracle.

This is not evidence to tune another tail geometry. The next action is to
measure the exact stage that remains serial or pathological for an aligned
partial tail and add the missing state/continuation oracle before another
candidate is written.

## Decision

**REJECT** — primary repository disposition. The candidate fails correctness
and performance gates.

**LEARN** — narrative result. The aligned route removes much of the obvious
scalar remainder but does not qualify the contract.

## Next frontier

Refresh exact stage attribution for an aligned partial tail, starting with
full-attention partial preparation/finish stages, using exclusive timing and a
state/continuation oracle. Do not tune the `384 + 64 + 62` decomposition or
claim partial-tail promotion until that evidence exists.
