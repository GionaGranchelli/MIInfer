# EXP-0167 — post-A28 native generation baseline

## Question

What is the current MIInfer TG64 baseline, and how does it compare with the
pinned llama.cpp build under stable-peak MI50 conditions?

## Environment

```text
GPU:      AMD Instinct MI50 / gfx906
Clocks:   stable_peak (1725 MHz SCLK / 1000 MHz MCLK at setup)
Model:    /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:    build/mi50-release
Fixture:  /tmp/m6a273-reference
MIInfer:  current production path (B66 dual beta/alpha projection)
llama:    c0bc859
```

## Commands

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --bench64
scripts/run-m6b0-llama-baseline.sh \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
```

## Results

MIInfer TG64 (five samples): `4473.01, 4479.28, 4481.80, 4484.77,
4485.03 ms`; median `4481.80 ms`, **14.280 tok/s**. Replay passed, decode
allocations were zero, and tracked device usage was `20,094,914,900` bytes.

llama.cpp medians:

| Workload | tok/s |
| --- | ---: |
| PP512 | 196.415 |
| TG64 | 22.331 |
| TG128 | 22.535 |
| TG256 | 22.411 |
| P1+TG64 | 22.1469 |
| P1+TG256 | 22.0409 |
| P1+TG1024 | 22.0437 |

The direct TG64 comparison is `14.280 / 22.331 = 0.640`, so MIInfer is
approximately **36.0% slower**. The M6 target of `>=1.05x` is not met.

## Decision

**MEASUREMENT-ONLY / B1 BASELINE.** Functional and resource gates pass, but
performance remains below llama.cpp. The final llama benchmark artifact is
`/home/fedora-workstation/Development/mi50-artifacts/m6b0-results/20260905T132214Z-217396`.

## Follow-up

Run a fresh MIInfer whole-token profile and select one measured optimization;
do not infer the next target from dispatch count alone.
