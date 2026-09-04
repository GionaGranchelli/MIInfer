# EXP-0105 — M6-B12 Q6_K packed dot4 projections

## Question

Can the Q6_K × Q8_K projection path use the same gfx906 packed-dot4 approach
that produced a clear Q4_K win?

## Baseline

The accepted scalar Q6_K × Q8_K path remained the production default. The
Q4_K packed-dot4 change was already selected, but Q6_K has different packed
low/high-bit reconstruction and was tested independently.

## Candidate

The candidate used one Wave64 per output row. Each lane reconstructed four
Q6 values from the Q6_K low/high bit planes, loaded four Q8_K values, and
used gfx906 signed dot4 accumulation. The scalar Q6_K launcher remained
available for the control comparison.

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

Native autoregressive smoke with the candidate:

```text
generate16:  PASS
replay:      PASS
allocations: 0
device bytes: 17,018,706,644
```

Native 64-token generation with the candidate:

```text
generate64:  PASS
replay:      PASS
allocations: 0
first decode: 6786.41 ms
second decode: 6790.05 ms
```

Same-build P64 A/B medians:

| Path | Median | Throughput |
| --- | ---: | ---: |
| Scalar control (`MIINFER_Q6K_DOT4=0`) | 6777.96 ms | 9.44236 tok/s |
| Packed dot4 (`MIINFER_Q6K_DOT4=1`) | 6776.41 ms | 9.44453 tok/s |

The measured change is **+0.02%**, below noise and far below the useful
production threshold. The longer benchmark did not provide a valid paired
result because the control process hit a VRAM allocation failure; this does
not change the P64 conclusion.

## Correctness and resources

* Native 16-token generation: PASS with exact replay.
* Native 64-token generation: PASS with exact replay.
* Decode-loop allocations: 0.
* Release CTest after restoring the scalar path: **20/20 PASS**.

## Interpretation

Q6_K reconstruction and dot4 arithmetic are functionally viable, but the
one-Wave64 packed implementation does not improve whole-token throughput.
Unlike Q4_K, this candidate does not justify changing the production path.
The result also provides no evidence that Q6_K's remaining cost is solved by
the Q4-style packed mapping.

## Decision

**REJECT / diagnostic candidate not production-selected.**

## Follow-up

Keep the accepted Q4_K packed-dot4 path and scalar Q6_K path. Select the next
optimization only after a fresh post-EXP-0104 profile identifies measurable
whole-token leverage.
