# EXP-0240 — M11-B Q4_K wave-major 16-token tile rejection

## Hypothesis

The EXP-0239 tile was limited by keeping 16 token accumulators live in each
wave. Assigning one token to each of 16 waves, walking the 16 staged rows with
one accumulator, could preserve weight reuse while reducing register pressure.

## Baseline

Four native Q4_K B=4 launches for 16 tokens, the same exact-shape baseline as
EXP-0239.

## Candidate

A 1024-thread workgroup stages 16 native Q4_K rows and 16 Q8_1 input vectors
in LDS. Each wave owns one token and processes the staged rows one at a time,
writing the token-major output layout. The candidate was standalone only and
was not integrated into causal prefill.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: Q4_K FFN Down, `[17408, 5120]`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Tokens: 16
- Repetitions: 41 interleaved rounds, three launches per timing sample

## Correctness

The candidate matched the B=4 path within `max_abs_error=3.68441e-6` across
all outputs. No non-finite values were observed after correcting the initial
invalid row/token mapping; that first invalid run was discarded.

## Results

| Path | Time | Relative |
| --- | ---: | ---: |
| Four native B=4 launches | 1200.21 us | 1.000x |
| Wave-major 16-token tile | 2193.49 us | 0.547x |

The wave-major mapping is 1.83x slower. Reducing accumulator count does not
offset the 1024-thread workgroup and serial row traversal at this skinny
production shape.

## Decision

**REJECT.** Remove the standalone kernel and benchmark; do not integrate a
wave-major token-tile flag.

## Follow-up

EXP-0239 and EXP-0240 together close the minimal native-layout 16-token Q4_K
reuse mappings. Further work needs a different quantized projection dataflow,
not another accumulator-to-wave assignment of the same tile.
