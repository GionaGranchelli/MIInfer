# M4-C3 — Text-facing greedy generation

Status: `CLOSED`

M4-C3 adds the smallest user-facing layer around the accepted M4-C2
incremental MI50 decoder. It uses the tokenizer vocabulary and BPE merges
embedded in the pinned Qwen3 GGUF; no external tokenizer dependency or generic
execution graph is introduced.

## Tokenizer contract

The supported artifact declares:

```text
tokenizer.ggml.model = gpt2
tokenizer.ggml.pre   = qwen2
```

`Qwen3Tokenizer` implements that pinned byte-level BPE contract, including the
Qwen2 pre-tokenizer, embedded merge ranks, BOS/EOS metadata, and GPT-2
byte-to-Unicode detokenization. Other tokenizer families fail explicitly.

## CLI

Build the MI50 Release preset and run:

```bash
build/mi50-release/miinfer-qwen3-generate \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  --prompt hello --max-tokens 8
```

The command tokenizes the prompt, processes prompt tokens through the
persistent 36-layer cache, greedily selects each continuation, and prints both
IDs and generated text. EOS stops generation. Sampling, chat templates,
streaming, batching, and server behavior are intentionally excluded.

## Pinned physical acceptance

The real-model, non-vacuous gate is:

```bash
scripts/run-m4c3-acceptance.sh \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

It requires the MI50 Release executable, uses prompt `hello`, and requires:

```text
prompt IDs:    14990
generated IDs: 8,341,286,470,330,9707,11,330
generated text: ) {\n        return "Hello, "
```

The artifact-free CTest tokenizer entry is a smoke/skip check when no model
path is supplied. The physical command checks both IDs and the generated text
for the real artifact, as with earlier M4 milestones.

## Decision

**M4-C3 CLOSED — pinned text prompt produces the expected greedy continuation**
