# M4-C3 text/tokenizer fixture

This fixture is tied to the pinned Qwen3-8B Q4_0 artifact:

```text
SHA256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
reference: milpster/gfx906-llama-cpp
reference commit: 6e4ef6c1a553b8f61ad77bba18e9ca05aa677295
tokenizer metadata: gpt2 / qwen2
```

The pinned tokenizer checks are:

```text
"hello"         -> 14990
"hello world"   -> 14990, 1879
"Hello, world!" -> 9707, 11, 1879, 0
```

The physical greedy-generation check uses prompt `hello` and expects:

```text
generated IDs:   8, 341, 286, 470, 330, 9707, 11, 330
generated text:  ) {\n        return "Hello, "
```

The IDs and continuation were established by the closed M4-C2 explicit-ID
fixture. C3 adds the independent tokenizer/detokenizer checks and exercises
those IDs through the text-facing Release CLI.
