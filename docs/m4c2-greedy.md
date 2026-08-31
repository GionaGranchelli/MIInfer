# M4-C2 — Short deterministic greedy decode sequence

Status: `CLOSED`

M4-C2 validates repeated autoregressive decode on the pinned Qwen3-8B model.
It uses explicit token IDs and the persistent per-layer KV caches established
by M4-C1. Tokenization, sampling, text output, and performance are excluded.

## Pinned sequence

The independent reference fixture is recorded in
[`tests/reference/qwen3/m4c2-greedy/README.md`](../tests/reference/qwen3/m4c2-greedy/README.md).
The prompt token is `14990`; the eight greedy generated IDs are:

```text
8, 341, 286, 470, 330, 9707, 11, 330
```

The sequence was captured from the pinned `milpster/gfx906-llama-cpp`
reference at `6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`, using the pinned
Qwen3-8B Q4_0 artifact.

## Acceptance

For each position, the test feeds the selected token through the 36-layer
decoder, checks MI50 greedy selection against the pinned next ID, checks
finite outputs, and checks every layer cache length. The GPU sequence is then
replayed from a fresh cache and must produce bitwise-identical logits at every
position. Release is the exact-token production gate. Unoptimized Debug runs
the same fixed prefix as a structural/numerical diagnostic: it requires finite
outputs, valid cache lengths, and deterministic replay while reporting any
token divergence. This keeps the physical gate aligned with the optimized
MI50 execution contract.

The non-vacuous physical gate is:

```bash
scripts/run-m4c2-acceptance.sh \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

## Decision

**M4-C2 CLOSED — Release MI50 reproduces all eight pinned greedy IDs**

The fixed-prefix Debug/Release state dump shows that position 0 is bitwise
identical between builds. During position 1, layer outputs remain identical
through layer 19 and first differ at layer 20 (`0.0117188` max abs); the
corresponding layer-21 K/V cache entries are the first materially different
cached state. The difference then grows through the remaining layers. At
position 3, layers 0–20 are identical, layer 21 differs by `0.0219116`, and
layer 35 by `1.117`; final-norm and logits differ by `0.346703` and `0.405018`.

Serialized Debug (`AMD_SERIALIZE_KERNEL=3`, `AMD_SERIALIZE_COPY=3`) still
selects `419`. A `RelWithDebInfo` build (`-O2 -g -DNDEBUG`) selects `470`,
matching Release (`-O3 -DNDEBUG`). The evidence therefore points to
unoptimized HIP kernel code generation as the current Debug-only cause,
rather than a position-3 cache write race. No production precision or model
semantics were changed.

The optimized Debug-HIP experiment (`Debug` host configuration with HIP
kernels compiled using `-O2 -g`) also reproduces all eight IDs. The canonical
unoptimized Debug configuration remains a diagnostic build and selects `419`
at position 3; this is a build-code-generation difference, not a production
inference failure.

## Next slice

M4-C3 can add tokenizer/detokenizer integration and a minimal text-facing
greedy decode path. Sampling, serving, batching, and performance optimization
remain deferred.
