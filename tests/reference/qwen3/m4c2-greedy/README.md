# M4-C2 short greedy decode fixture

This explicit-ID fixture is captured from the pinned independent gfx906
reference at commit `6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`, using the
pinned Qwen3-8B Q4_0 artifact with SHA256:

```text
458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
```

The reference command was:

```text
llama-simple -m Qwen3-8B-q4_0-b968826d.gguf -n 8 -ngl 99 hello
```

The reference tokenizer maps the explicit prompt `hello` to token `14990`.
The pinned greedy generated sequence is:

```text
prompt:    14990
generated: 8, 341, 286, 470, 330, 9707, 11, 330
```

MIInfer does not tokenize this prompt. M4-C2 feeds these IDs directly through
the persistent 36-layer GPU KV cache. At position `t`, the selected token is
fed into position `t + 1`; the final generated ID is therefore checked at
position 7 without requiring an additional forward pass.

The C2 gate checks every step for:

* exact expected greedy token ID;
* host/GPU token agreement;
* finite logits and intermediate outputs;
* cache length across all 36 layers;
* bitwise-identical GPU replay of the full sequence.

The physical command is:

```bash
scripts/run-m4c2-acceptance.sh \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

Tokenizer, sampling, text output, batching, and performance work remain
outside M4-C2.
