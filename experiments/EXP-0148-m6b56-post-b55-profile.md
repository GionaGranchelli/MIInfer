# EXP-0148 — M6-B56 post-B55 production profile

## Question

Where does the accepted path spend time after DeltaNet query/key LDS staging?

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Clock:     stable_peak; 1725 MHz SCLK / 1000 MHz MCLK observed
Fixture:   /tmp/m6a273-reference-p12
```

## Results

Position-63 profile:

```text
total GPU event: 72.9092 ms/token
layer sum:       69.6170 ms/token
final norm:       0.023039 ms
final Q8:         0.008320 ms
final LM head:    2.503360 ms
final argmax:     0.498400 ms
allocations:      0
```

Representative production stages (ms/token):

| Stage | Layer 0 | Layer 1 | Layer 2 | Attention layer 3 |
| --- | ---: | ---: | ---: | ---: |
| QKV / Q projection | 0.173759 | 0.184320 | 0.173600 | 0.199039 |
| FFN Gate/Up | 0.278080 | 0.277280 | 0.259999 | 0.277919 |
| FFN Down | 0.445599 | 0.443839 | 0.418560 | 0.446080 |
| State update / cached attention | 0.091520 | 0.095679 | 0.090720 | 0.138720 cached attention |

The accepted B55 path therefore lowers the sampled state-update stage from
the approximately `0.114 ms` post-B53 control range to approximately
`0.091–0.096 ms`, while retaining the same state layout and recurrence.
FFN Down remains the largest repeated projection family; no standalone Down
geometry or Q8_K representation change is selected.

## Interpretation

The result confirms B55's local win is visible in whole-token GPU attribution,
but the optimization did not change the dominant family. The next candidate
must be a higher-level FFN Down dataflow or a different major recurrent
family, not another generic LDS staging or already-rejected Down geometry.

## Decision

**MEASUREMENT-ONLY / BASELINE.** This is the authoritative profile after the
B55 production selection.

## Follow-up

Audit the complete FFN Down input-to-residual path and quantify removable
materialization or duplicated work before implementing B57.
