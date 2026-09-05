# EXP-0120 — M6-B28 post-B27 production profile

## Question

After selecting batched full-attention head RMS normalization, where does the
native Qwen3.8-27B GPU token actually spend time, and what single family is
the best next measurement target?

## Scope

Measurement only. Production defaults were used; no kernel or execution-path
changes were made.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak
Observed clocks: 1725 MHz SCLK / 1000 MHz MCLK
Temperature after run: 53 C edge / 67 C junction / 52 C memory
Power after run: 111 W
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Fixture: /tmp/m6a273-reference
Command: miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --profile64
```

## Results

Three serial instrumented profiles reported total GPU times of:

```text
88.9892 ms
89.0098 ms
88.8735 ms
```

Median: **88.9892 ms/token**. The profile ends at position 63 and reports
zero allocations during profiling. The native default generation path also
passes 16-token replay with zero decode allocations.

The production B27 throughput baseline remains:

| Workload | B27 median |
| --- | ---: |
| TG64 | 11.4420 tok/s |
| TG128 | 11.2970 tok/s |

Per-layer totals from the third profile:

| Layer kind | Count | Aggregate GPU time | Average | Range |
| --- | ---: | ---: | ---: | ---: |
| Recurrent | 48 | 64.54208 ms | 1.34463 ms | 1.21136–1.59776 ms |
| Full attention | 16 | 21.01088 ms | 1.31318 ms | 1.16544–1.55328 ms |

Representative detailed stages remain available for recurrent layer 0 and
full-attention layer 3. Recurrent layer 0 includes approximately 0.156 ms
QKV projection, 0.200 ms state update, 0.447 ms FFN Gate/Up, and 0.448 ms
FFN Down. Full-attention layer 3 includes approximately 0.199 ms Q
projection, 0.139 ms cached attention, 0.425 ms FFN Gate/Up, and 0.418 ms
FFN Down. Final LM head time is approximately 2.50 ms.

The native profile harness does not expose a dispatch counter (`unknown_in_native_harness`),
so no dispatch reduction is inferred from this measurement. Allocation
count is zero.

## Ranking

| Candidate family | Evidence | Priority |
| --- | --- | ---: |
| Recurrent layer aggregate | 48 layers / 64.54 ms; repeated support and projection work | 1 |
| Full-attention layer aggregate | 16 layers / 21.01 ms; no new scaling pathology | 2 |
| LM head | ~2.50 ms once per token | 3 |
| Final norm/Q8/argmax | ~0.53 ms combined | 4 |

## Interpretation

B27 is stable and the remaining token cost is dominated by the repeated
recurrent-layer path, not by full-attention scaling or final output handling.
The next experiment should measure one concrete recurrent support/projection
family across representative recurrent layers before changing production
code. Existing QKV and output-projection optimizations remain selected; this
profile does not justify reopening them.

## Decision

**MEASUREMENT-ONLY; B27 remains KEEP.** The next milestone should be a
narrow recurrent-layer stage differential, with whole-token A/B required
before production selection.

## Follow-up

Profile representative recurrent layers beyond layer 0, especially the
state-update and recurrent FFN support stages, and select exactly one bounded
candidate from that evidence. Do not fuse recurrent stages solely to reduce
launch count.
