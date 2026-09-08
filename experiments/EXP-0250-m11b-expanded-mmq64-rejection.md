# EXP-0250 — M11-B decoded Q4_K MMQ64 rejection

## Hypothesis

A grouped 64-row × 64-token kernel using decoded Q4_K metadata and byte-packed
nibbles could preserve exact Q4_K arithmetic while avoiding the native layout's
register-heavy metadata path. The kernel stages 64 expanded rows and a
16-token Q8_1 slab per 256-element weight block.

## Baseline and candidate

- Baseline: sixteen native Q4_K B=4 launches for 64 tokens.
- Candidate: one 256-thread grouped launch over 64 rows and 64 tokens, using
  `Q4KExpandedDeviceBlock` weights and exact existing expanded-dot arithmetic.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, exact Q4_K shape `[17408, 5120]`
- Build: `mi50-release`
- Warmup: 20 baseline/candidate pairs
- Timing: 51 interleaved HIP-event samples per path

## Correctness

The candidate produced finite output and matched the native B=4 baseline with
maximum absolute error `6.4373e-6`.

## Results

| Path | Median | Relative |
| --- | ---: | ---: |
| Sixteen native B=4 launches | 4781.12 us | 1.000x |
| Decoded-metadata MMQ64 candidate | 48751.0 us | 0.098x |

## Interpretation

Exact reuse of the existing Q4_K expanded-dot mapping requires too many scalar
sub-operations per output cell. The 64-row weight staging cannot compensate;
the candidate is more than ten times slower than the native B=4 sequence.

## Decision

**REJECT.** The candidate, temporary benchmark, declaration, and build target
were removed. Production execution is unchanged.

## Follow-up

The remaining viable grouped result is the earlier repacked MMQ tile at only
1.125× for a full 64-token projection. A production attempt would need a new
low-register packed representation and causal integration evidence; another
direct expansion of the current Q4_K arithmetic is not justified.
