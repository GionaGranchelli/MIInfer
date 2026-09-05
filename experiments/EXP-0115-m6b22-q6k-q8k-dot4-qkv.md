# EXP-0115 — M6-B22 Q6_K×Q8_K packed-dot4 recurrent QKV

## Question

Can the recurrent `attn_qkv` Q6_K×Q8_K projection use gfx906 packed dot4
operations while preserving its existing quantized representation?

## Candidate and control

The control is the B21 production Q6_K×Q8_K scalar projection, selected with
`MIINFER_Q6K_Q8K_DOT4_QKV=0`. The candidate retains the same Q6_K weights and
Q8_K activation quantization, but assigns one Wave64 to each output row and
accumulates four signed Q6/Q8 elements per lane using gfx906 dot4. No Q8_1
conversion or weight-layout change is involved.

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
* Candidate/control device bytes: `17019965780`.

## Results

Three serial process medians, each process containing five timed samples:

| Workload | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 tok/s | 10.9603 | 11.2805 | +2.92% |
| TG128 tok/s | 10.8261 | 11.1392 | +2.89% |

Representative P64 profile timing for layer 0 QKV changed from `0.284479 ms`
to `0.179999 ms`. Whole-token repeated throughput is the selection metric.

## Interpretation

Keeping the Q6_K×Q8_K contract while packing four values per dot4 operation
reduces the repeated recurrent QKV stage and gives a stable whole-token gain.
The candidate is distinct from the rejected Q6_K×Q8_1 QKV experiment, which
changed the representation/layout contract and failed correctness.

## Decision

**KEEP; production-selected.** The packed Q6_K×Q8_K recurrent QKV path is now
the default. Set `MIINFER_Q6K_Q8K_DOT4_QKV=0` to run the scalar control.

## Follow-up

Refresh the post-B22 profile before selecting the next optimization. Do not
assume the rejected Q8_1 QKV path is a compatible alternative.
