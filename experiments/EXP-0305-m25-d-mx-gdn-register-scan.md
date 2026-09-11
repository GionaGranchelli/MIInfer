# EXP-0305 — M25-D mx GDN register-resident scan

## Hypothesis

The remaining recurrent-prefill cost is largely the per-64-token GDN state
round-trip. Porting mx-llama's fixed Qwen3.8 non-KDA register-resident scan
should preserve the recurrent state while removing those round-trips.

## Motivation

EXP-0300 measured MIInfer's recurrent tail at `813.842 ms/P512` across 48
layers, versus `189.115 ms` for mx's `gated_delta_net_chunked_cuda` family.
EXP-0301 rejected a geometry-only port because its state contract was wrong.
This experiment ports the complete mx Wave64/DPP scan and explicitly adapts
its value-by-key register shard to MIInfer's persistent key-by-value state.

## Baseline

The existing M23 wide recurrent path, with the compact mx projection profile
enabled but the existing MIInfer 64-token GDN chunks:

```text
MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ=1
MIINFER_PREFILL_MX_GDN unset
```

## Candidate

`MIINFER_PREFILL_MX_GDN=1` selects one fixed `64x2` Wave64 block per four
state columns and loops the complete token sequence in registers. The port is
restricted to the Qwen3.8 shape (`state=128`, token count a multiple of 64,
maximum 512) and remains opt-in. It uses mx's gfx906 DPP reduction helpers,
with the existing MIInfer `[head][key][value]` state transposed only while
loading and storing the register shard.

Source provenance:

```text
/home/fedora-workstation/Development/mx-llama.cpp
commit 2e9d29fe736969160f17476ec6f0a6298cee6966
ggml/src/ggml-cuda/gated_delta_net_chunk.cu
MIT; copyright (c) 2023-2026 The ggml authors
```

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M GGUF
- model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm system libraries, release build
- context capacity 1024
- SCLK 1606 MHz, MCLK 1000 MHz
- exact 512-token repeated `The quick brown fox jumps over the lazy dog. ` prompt, ending `The quick`
- full established M23/Mx projection profile and `--max-tokens 0`

## Benchmark

Standalone check:

```text
build/mi50-release/miinfer-m12-gdn-chunk-bench
```

End-to-end A/B command used the switches from EXP-0300 plus
`MIINFER_PREFILL_MX_GDN=1` for the candidate. The A/B order was candidate,
control, candidate, control. The raw telemetry loop retained 350 JSON
samples in `/tmp/mi50-mx-gdn-qualified-GRQlts/telemetry.log`.

## Correctness

- Existing `qwen35-conv-batch-gpu` release CTest passed.
- Standalone CPU recurrent oracle passed for both paths:
  - mx output max error: `1.1e-8`
  - mx final-state max error: `8.2e-8`
- Exact P512 candidate continuation completed successfully and emitted token
  ID `13477` (`brown`), matching the established control continuation.
- All four A/B CLI cases returned status 0 and reported 512 prompt tokens.

## Results

Standalone median timings from the release benchmark:

| path | median us | max output error | max state error |
|---|---:|---:|---:|
| existing M12 chunk | 3945.115 | `1.2e-8` | `1.34e-7` |
| mx register scan | 1087.678 | `1.1e-8` | `8.2e-8` |

The mx scan is `11.52x` faster than the token-at-a-time reference in that
lab shape, while the existing M12 chunk is `3.17x` faster.

Telemetry-qualified P512 A/B:

| order | path | prefill | allocation |
|---|---|---:|---:|
| A1 | mx GDN candidate | 3027.92 ms / 169.09 tok/s | 19,108,282,708 B |
| B1 | M12 GDN control | 3689.13 ms / 138.79 tok/s | 19,108,282,708 B |
| A2 | mx GDN candidate | 3028.10 ms / 169.08 tok/s | 19,108,282,708 B |
| B2 | M12 GDN control | 3683.35 ms / 139.00 tok/s | 19,108,282,708 B |

Median candidate latency is `3028.01 ms` (`169.09 tok/s`) versus
`3686.24 ms` (`138.90 tok/s`) for control: `1.217x`, or `17.3%` faster, and
`658.23 ms/P512` recovered. The telemetry recorded all 350 clock samples at
SCLK/MCLK `1606/1000 MHz`; edge temperature ranged from `34–41 C`.

## Profiling

No PMC capture was taken for this isolated port. The end-to-end delta is
consistent with removing the GDN tail identified by EXP-0300, but the
remaining recurrent FFN and attention tails are still unprofiled in this
candidate.

## Interpretation

The complete mx scan is a valid, state-correct port and materially improves
the current best MIInfer P512 path. It does not reach the project gate of
`200 tok/s`; the remaining gap is approximately `31 tok/s` to that gate and
roughly `52 tok/s` to the pinned mx result. The state-layout adaptation is
essential: applying mx's physical shard indices directly produced a large
state error and was discarded before integration.

## Decision

**KEEP as an opt-in recurrent-prefill candidate; RETEST for longer-generation
qualification.** Do not enable it by default until longer deterministic
continuation and broader context-boundary checks pass.

## Follow-up

Port the next measured mx recurrent bottleneck, starting with the FFN/rocBLAS
tail rather than retuning this already-valid scan. Keep the current control
available for every comparison.
