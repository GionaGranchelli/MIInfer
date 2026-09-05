# EXP-0144 — M6-B52 Q4_K×Q8_1 one-Wave64 row mapping

## Question

Can one Wave64 reduce each Q4_K output row, with two rows sharing the existing
LDS input and decoded-metadata staging, faster than the B41 two-wave-per-row
mapping?

## Candidate

An opt-in kernel used 128 threads per workgroup: two independent Wave64s, one
per output row. Each wave retained the B41 Q4_K×Q8_1 dot helper, decoded
metadata, Q8_1 LDS tile, and FP32 reduction; only the per-row reduction width
changed from 128 to 64 threads.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
```

## Results

Native replay passed with zero decode allocations and unchanged tracked device
usage (`17,019,965,780` bytes). Same-build TG64 medians were:

| Path | Median ms | Throughput |
| --- | ---: | ---: |
| B41 control | 4615.04 | 13.8677 tok/s |
| one-Wave64 candidate | 4850.10 | 13.1956 tok/s |

The candidate is approximately **4.85% slower**.

## Decision

**REJECT.** One Wave64 per output row does not improve the Q4_K×Q8_1 path on
gfx906. The B41 two-wave-per-row mapping remains production-selected. The
candidate code and toggle were removed.

## Follow-up

Do not repeat this row-width family or the already-rejected split-K mapping
without new profiling evidence. The next performance work must target a
different execution mechanism with enough leverage to address the remaining
~1.6× whole-runtime gap.
