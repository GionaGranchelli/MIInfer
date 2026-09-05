# EXP-0147 — M6-B55 DeltaNet LDS input reuse

## Question

Can the transposed no-decay-store DeltaNet update avoid repeatedly rereading
the same per-head query and key vectors by staging them once in LDS?

## Candidate

An opt-in kernel stages each 128-element query/key vector once per value-head
workgroup in 1 KiB of LDS, then executes the existing transposed
no-decay-store recurrence. The default now selects it;
`MIINFER_DELTA_TRANSPOSED_LDS_INPUTS=0` selects the prior control.

No state layout, quantization, projection, accumulation, or output contract
changed.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Clock:     stable_peak; 1725 MHz SCLK / 1000 MHz MCLK observed
Fixture:   /tmp/m6a273-reference-p12
ROCm:      7.1.52802-9999
Compiler:  Clang 20.0.0.rocm
```

## Results

Native 64-token replay passed with zero decode allocations and unchanged
tracked/peak device bytes (`17,019,965,780`). The full P64 external
observable run passed, including poisoned reset/replay; final argmax remained
`8719`. Release CTest passed **20/20**.

Same-build five-sample medians:

| Workload | Control tok/s | LDS-input tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 13.8564 | **14.0953** | **+1.72%** |
| TG128 | 13.6513 | **13.8572** | **+1.51%** |

Control and candidate benchmark replays passed. Candidate TG64 samples were
4532.79, 4539.56, 4540.52, 4547.03, and 4553.26 ms; candidate TG128 samples
were 9225.94, 9233.84, 9237.06, 9238.56, and 9240.25 ms.

## Decision

**KEEP / production-selected.** LDS staging removes redundant per-head
query/key global reads and produces a repeatable whole-token improvement above
the useful threshold with unchanged correctness and memory footprint.

## Follow-up

Refresh the production profile after B55. Do not generalize this result to
other LDS staging without a new measured redundancy.
