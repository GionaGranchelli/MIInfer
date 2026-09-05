# EXP-0156 — M6-B64 post-B62 stage profile

## Question

Where does the current production Qwen3.8-27B decode step spend GPU time after
the B62 rejection?

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clocks:    profile_peak (1725 MHz SCLK / 1000 MHz MCLK policy)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Position:  63
```

## Results

```text
total GPU event: 72.4071 ms/token
layer sum:       69.1570 ms/token
final norm:       0.02512 ms
final Q8:         0.00864 ms
LM head:          2.48624 ms
argmax:           0.469599 ms
allocations:      0
```

Representative layer stages:

```text
recurrent FFN Gate/Up:  0.26064–0.277279 ms
recurrent FFN Down:     0.413919–0.4552 ms
recurrent QKV:          0.15632–0.18432 ms
recurrent state update: 0.09104–0.09472 ms
full attention:         0.998239–1.396 ms/layer
```

There are 48 recurrent layers and 16 full-attention layers. Repeated FFN Down
remains the largest single layer family. The topology remains allocation-free;
the native harness reports dispatches as unknown, so this profile does not
attribute launch gaps.

## Interpretation

The profile confirms that the production path is doing real GPU work rather
than waiting on host orchestration. FFN Down has already been tested with
multiple Q4_K geometries and expanded-weight representation, while the latest
standalone recurrent geometry candidate failed correctness. No new blind
geometry experiment is justified by this profile alone.

## Decision

**MEASUREMENT-ONLY.** Keep the current production path.

## Follow-up

The next candidate must be a demonstrated representation or work differential
for the repeated FFN Down path, or a differential comparison against the
strongest pinned llama.cpp implementation. Do not revisit B62 row-wave mapping
without a new independently validated state-layout contract.
