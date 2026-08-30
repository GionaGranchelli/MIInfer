# M4-B single-token full-model reference trace

This compact fixture is an independent full-depth trace for the pinned
Qwen3-8B Q4_0 artifact. It contains the embedding, every `l_out-0` through
`l_out-35` hidden state, the final normalized hidden state, and the complete
Q6_K vocabulary logits as F32 values.

Metadata:

```text
token: 14990
position: 0
layers: 36
hidden: 4096
vocabulary: 151936
model SHA256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
reference: milpster/gfx906-llama-cpp
reference commit: 6e4ef6c1a553b8f61ad77bba18e9ca05aa677295
reference execution: llama-eval-callback, -ngl 0, -t 24 -tb 24, one explicit token 14990
```

The trace was captured through the pinned reference callback, independently
of the MIInfer host and GPU executors. It is intentionally a correctness
fixture, not a model artifact or performance result. This is the canonical
M4-B fixture: two independent captures from the same pinned build and
settings were byte-identical for all 39 F32 files. The previous fixture is
preserved in `../m4b-single-token-legacy/`; it was captured under different
conditions and is historical evidence only, not an acceptance oracle.
