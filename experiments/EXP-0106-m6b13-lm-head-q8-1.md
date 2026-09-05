# EXP-0106 — M6-B13 Q6_K × Q8_1 LM-head compatibility path

## Question

Can matching the pinned llama.cpp Q6_K × Q8_1 LM-head activation contract
reduce MIInfer's approximately 6.2 ms final vocabulary projection cost?

## Baseline

The accepted MIInfer path uses Q6_K × Q8_K for the final projection. The
post-EXP-0104 profile measured:

```text
final LM head: approximately 6.19 ms/token
```

The pinned external gfx906 path uses Q6_K × Q8_1, so this was tested as an
opt-in M6 representation experiment rather than as an M5 same-contract
comparison.

## Candidate

The candidate quantized the final normalized FP32 vector directly into the
existing canonical Q8_1 block layout and used a Q6_K × Q8_1 GPU GEMV. The
accepted Q6_K × Q8_K path remained the default and was restored after the A/B.

## Environment

```text
GPU:          AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:        /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:        build/mi50-release
ROCm:         7.1.52802-9999
Compiler:     Clang 20.0.0.rocm
Fixture:      /tmp/m6a273-reference
```

## Results

Native generation smoke:

```text
generate16:  PASS
replay:      PASS
allocations: 0
```

P64 same-build benchmark:

| Path | Median | Throughput |
| --- | ---: | ---: |
| Q6_K × Q8_K control | 6777.96 ms | 9.44236 tok/s |
| Q6_K × Q8_1 candidate | 7640.19 ms | 8.37676 tok/s |

The candidate is **11.28% slower** end-to-end. It also adds the Q8_1
representation's smaller-block quantization work, so replacing the current
Q8_K path is not justified by the external representation match alone.

The external observable run preserved the established top-1 and top-5
behavior at the available early checkpoints, but the fixture terminates with
the pre-existing missing `12-l_out-63-` checkpoint. This candidate is rejected
on performance regardless; no production correctness claim is made from the
incomplete late checkpoint set.

## Resources and checks

* Decode-loop allocations: 0.
* Device bytes after setup: `17,018,712,404` with the candidate.
* Native replay: PASS.
* The accepted scalar Q6_K × Q8_K path was restored after the experiment.
* Release CTest after restoration: **20/20 PASS**.

## Interpretation

The Q8_1 representation used by llama.cpp is not itself a sufficient MI50
optimization. A competitive implementation would need the corresponding
optimized MMVQ-style data access and reduction strategy, not this direct
scalar compatibility kernel. The experiment therefore provides a clean
negative result and does not alter the production representation.

## Decision

**REJECT / diagnostic candidate not production-selected.**

## Follow-up

Keep Q6_K × Q8_K in production. Do not repeat a scalar Q8_1 port; revisit the
contract only with a measured MMVQ-equivalent mechanism and a complete external
logit/generation validation path.
