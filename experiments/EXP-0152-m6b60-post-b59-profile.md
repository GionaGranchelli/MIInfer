# EXP-0152 — M6-B60 post-B59 production profile

## Question

After selecting expanded Q4_K FFN Down weights, what remains dominant in the
production Qwen3.8-27B decode path, and does the stage profile identify a new
high-confidence optimization target?

## Baselines

The selected path uses `MIINFER_Q4K_EXPANDED_DOWN=1` by default. The packed
control remains available with `MIINFER_Q4K_EXPANDED_DOWN=0`.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Clock:     profile_peak; 1725 MHz SCLK / 1000 MHz MCLK observed
Fixture:   /tmp/m6a273-reference-p12
Position:  63
```

## Results

| Profile | Expanded | Packed control |
| --- | ---: | ---: |
| Total GPU event (ms/token) | 72.1585 | 72.9130 |
| Layer sum (ms/token) | 68.8474 | 69.6649 |
| Final norm (ms) | 0.0232 | 0.02416 |
| Final Q8 (ms) | 0.0080 | 0.00848 |
| LM head (ms) | 2.52304 | 2.4880 |
| Argmax (ms) | 0.498239 | 0.469119 |
| Decode allocations | 0 | 0 |

Both profiles retain the existing flat dispatch/synchronization topology.
The representative expanded-path stages remain approximately:

```text
FFN Gate/Up:       0.261–0.278 ms
FFN activation:    ~0.007 ms
FFN Down:          0.418–0.448 ms
Recurrent QKV:     ~0.174–0.184 ms
Full attention:    ~1.0–1.45 ms/layer
LM head:           ~2.52 ms
```

## Interpretation

B59 did not change the dominant operation families. FFN Down remains the
largest repeated stage, but its standalone implementation has already been
compared extensively and B59 only improves whole-token throughput modestly.
The profile does not justify another blind FFN geometry or fusion experiment.
The next implementation must come from a demonstrated work or representation
differential, preferably with whole-token leverage larger than the remaining
B59 gain.

## Decision

**MEASUREMENT-ONLY.** This is the authoritative post-B59 profile and does not
select a new kernel.

## Follow-up

Compare the remaining recurrent/full-attention and LM-head contracts against
the pinned llama.cpp path, then select one bounded experiment. Preserve the
expanded Down default while that differential is measured.
