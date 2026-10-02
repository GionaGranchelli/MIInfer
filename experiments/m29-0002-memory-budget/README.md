# M29-0002 Workstream B — 128K memory budget

## Decision

`128K_FITS_WITH_LIFETIME_CHANGE`

The measured 128K failure is temporary-workspace peak pressure, not a model
weight or KV allocation failure. The smallest evidence-backed path is to avoid
at least `393,687,552` bytes (about `375.45 MiB`) of simultaneous peak
residency: `371,195,904` bytes is the exact failing-workspace deficit, and
`21,966,848` bytes is the already-measured activation allocation that follows
the workspace, excluding small decode/state allocations. This is a lower bound
before adding a production safety reserve.

## Measured allocation families

The successful 64K run reported:

```text
weights          24068487168
recurrent_state    158859264
kv_cache         4325376000
workspace          727711744
activation          21966848
cached_state       158859264
total            29461260288
```

The fixed 512-token workspace is subdivided as:

```text
shared_activation       41943040
recurrent_prefill       87949312
attention_prefill      460324864
quantization            20054016
ffn                    117440512
total                  727711744
```

For 128K (`kv_capacity=131200`), the FP16 KV formula is
`16 caches * capacity * 4 KV heads * 256 head dimension * 2 bytes * 2 (K/V)`:
`8598323200` bytes. If all measured families were resident, the category sum
would be `33734207488` bytes against `34342961152` device bytes. The observed
pre-workspace free bytes were lower because allocator/runtime overhead also
consumes space; this is why the exact failure-site measurement is authoritative.

## Exact failure

```text
requested workspace       727711744 bytes
free before workspace     356515840 bytes
allocation deficit        371195904 bytes
failing operation         hipMalloc(&d_buffer_, total_bytes_)
```

The 64K and 128K runs used the same model and MI50 gfx906 device. 4K, 8K,
16K, 32K, and 64K completed with valid numerics. 128K failed while constructing
the fixed prefill workspace, before prefill execution.

This record does not claim that a 375.45 MiB lifetime reduction is already
implemented or that it provides a safe operating reserve. It identifies the
minimum measured peak change to prove next; no allocator rewrite or KV
representation change was made.
