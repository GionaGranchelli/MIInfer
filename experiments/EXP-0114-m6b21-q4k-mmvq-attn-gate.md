# EXP-0114 — M6-B21 Q4_K×Q8_1 MMVQ recurrent gate

## Question

Can the recurrent `attn_gate` Q4_K projection use the existing Q4_K×Q8_1
MMVQ primitive without changing the external correctness contract?

## Candidate and control

The control is the B19 production path selected with
`MIINFER_Q4K_Q8_1_MMVQ_ATTN_GATE=0`: Q4_K×Q8_K recurrent gate projection.
The candidate quantizes the normalized recurrent input once to Q8_1 and uses
the existing 128-thread/output-row Q4_K×Q8_1 MMVQ projection only for
`attn_gate`. All other paths remain on the accepted B19 configuration.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Compiler: Clang 20.0.0.rocm
Fixture: /tmp/m6a273-reference-p12
```

## Correctness and resources

* Native 16-token generation replay: PASS.
* The 64-layer external observable-contract run completed successfully.
* Decode allocations: 0.
* Candidate/control post-setup and peak device bytes: `17019965780`.

## Results

Three serial process medians, each process containing five timed samples:

| Workload | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 tok/s | 10.7745 | 10.9673 | +1.79% |
| TG128 tok/s | 10.6554 | 10.8248 | +1.59% |

The corresponding median times were 5939.98 ms versus 5835.52 ms for TG64,
and 12012.70 ms versus 11824.70 ms for TG128. Representative layer-0
`gate_projection` timing changed from `0.12128 ms` to `0.08400 ms`; whole-token
repeated throughput is the selection authority.

## Interpretation

The Q4_K×Q8_1 MMVQ candidate consistently reduces the isolated recurrent gate
stage and produces a repeatable 1.6–1.8% whole-token improvement without
increasing memory use or changing the state ownership model.

## Decision

**KEEP; production-selected.** The candidate is now the default. Set
`MIINFER_Q4K_Q8_1_MMVQ_ATTN_GATE=0` to run the former Q4_K×Q8_K control.

## Follow-up

Refresh the post-B21 profile before selecting the next target. Do not infer
that remaining QKV or other Q6_K paths are compatible with this MMVQ layout.
