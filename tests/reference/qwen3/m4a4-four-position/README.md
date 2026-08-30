# M4-A5 four-position reference trace

This fixture is captured from the pinned `milpster/gfx906-llama-cpp`
reference at commit `6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`, using the
pinned Qwen3-8B Q4_0 artifact:

```text
SHA256 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
```

The exact explicit token sequence is:

```text
[14990, 42, 31415, 2718]
```

`pos-P-C.f32` is a little-endian F32 tensor for position `P` and canonical
layer-0 checkpoint `C`.  The checkpoint order is:

```text
0 embedding       1 attn_rms       2 attn_norm      3 q_projection
4 q_reshape       5 q_rms          6 q_normed       7 q_rope
8 v_projection    9 v_reshape     10 k_projection  11 k_reshape
12 k_rms         13 k_normed      14 k_rope        15 k_view
16 v_view         17 q_view        18 q_permuted     19 attention_output
20 ffn_input     21 ffn_rms       22 ffn_norm       23 gate
24 up            25 swiglu        26 ffn_output     27 layer_output
```

`pos-P-cache-k.f32` and `pos-P-cache-v.f32` contain the reference cache-write
vectors for the newly appended entry.  The K vector is post-RoPE; V is
unmodified.  This directly pins the cache representation used by the
stateful MI50 test.

The original callback capture, including the reference tensor manifest, is
retained outside the repository at:

```text
/home/fedora-workstation/Development/mi50-artifacts/m4a-trace-4pos/
```

The tracked binary fixture digest (content and sorted filenames; excluding
this README) is:

```text
5fa34b6d48a8fff8e6602e1a2f8ff9002bce3b0bb0f47cc6cfbdbf060f0e1f6c
```

The reference checkout was restored to the pinned commit after capture.
