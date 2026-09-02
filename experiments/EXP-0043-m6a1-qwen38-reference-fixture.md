# EXP-0043 — M6-A1 Qwen3.8-27B external reference fixture

## Question

Can MIInfer consume a reproducible external tensor/state reference for the
selected Qwen3.8-27B model before any DeltaNet or production-kernel work?

## Reference contract

The reference is local upstream llama.cpp commit
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`, run through its public C API on
the CPU backend. This is a correctness fixture, not an MI50 performance
measurement. The opaque state files use the public `llama_state_seq` format
version 2 and are intentionally not decoded by MIInfer yet.

## Model selected

`/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`

* size: 17,106,775,008 bytes (15.93 GiB); SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`;
* GGUF V3, file type Q4_K Medium, 5.01 BPW;
* `general.architecture=qwen35`, 27.32B parameters;
* hidden/input/output embedding: 5120;
* main layers: 64; total GGUF blocks including NextN: 65;
* FFN length: 17408; vocabulary: 248,320;
* attention: 24 heads, 4 KV heads, 256 key/value head length;
* RoPE: 64 dimensions, base 10,000,000, sections `[11,11,10,0]`;
* RMS epsilon: `1e-6`;
* recurrent attention: conv 4, state 128, groups 16, inner 6144, dt rank 48;
* full-attention interval: 4;
* tokenizer: GPT-2 BPE with Qwen35 pretokenizer, BOS 248044, EOS 248046,
  padding 248055.

The main layer pattern is 48 Gated DeltaNet/recurrent layers at
`0,1,2,4,5,6,...,60,61,62`, interleaved with 16 full-attention layers at
`3,7,11,...,59,63`. GGUF block 64 is the dense NextN/MTP block and is not part
of the ordinary 64-layer main forward.

## Fixture contents

The committed exporter is `tools/m6a1_reference_fixture.cpp`; the reproducible
driver is `scripts/run-m6a1-reference-fixture.sh`; validation is performed by
the dependency-free `scripts/check-m6a1-fixture.py`.

The generated bundle is kept outside Git, for example:

```bash
scripts/run-m6a1-reference-fixture.sh \
  /home/fedora-workstation/llama.cpp \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference hello 64
```

The completed run produced:

* prompt token `[14556]`;
* 64 greedy generated tokens;
* detokenized text in `generated_text.txt`;
* logits at positions `0,1,2,4,8,16,32,64`, each 248,320 F32 values;
* opaque state snapshots at the same positions;
* 4,240 selected F32 checkpoints at those positions, including embedding,
  attention norms/residuals, recurrent state/convolution outputs, full-
  attention outputs, FFN outputs, post-FFN outputs, and final norm.

The callback captures only selected positions and selected tensor names. It
does not pretend that the public API exposes the internal recurrent/KV layout.

## Correctness/checks

* exporter syntax check: passed with the pinned llama.cpp include tree;
* exporter link check: passed against the pinned `libllama` and `libggml-base`;
* real Q4_K_M model load: passed;
* 64-token stateful CPU decode: passed;
* fixture validator: passed;
* all captured logits and F32 checkpoints: finite;
* source tree `git diff --check`: passed.

## Decision

**KEEP / M6-A1 complete.** The external reference bundle is reproducible and
contains real hybrid-model state evolution, selected tensor checkpoints, and
final logits without modifying MIInfer production execution.

## M6-A2 next task

Build a read-only projection compatibility table from the GGUF tensor inventory
and current MIInfer kernels. For every recurrent, full-attention, FFN, and
LM-head projection, record GGUF type, dimensions, required activation
representation, available MIInfer primitive, and `REUSE`, `ADAPT`, or `NEW
KERNEL REQUIRED`. Do not implement Qwen3.8 execution or change production
dispatch selection in A2.
