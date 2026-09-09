# EXP-0285 — M22 prefill parity closure

## Verdict

**PROVEN HARD CEILING for the current causal B4 layer-major architecture.**
The M22 mx-llama parity gate is not met. The result is an architectural
closure, not a claim that MI50 hardware cannot run faster with a new causal
chunk schedule.

## Fixed comparison

* MIInfer starting commit: `474db556c84975a5ba1f97f9feec592cbd8304d0`
* Reference: mx-llama `2e9d29fe736969160f17476ec6f0a6298cee6966`
* Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* GPU: AMD Instinct MI50/gfx906, SCLK 1606 MHz, MCLK 1000 MHz
* ROCm: 7.1.52802-9999; Clang 20.0.0; Linux 7.1.10-200.fc44.x86_64
* Reference: full offload, Flash Attention, Q8_0 K/F16 V, `-b 2048 -ub 2048`

## Reference repack A/B

Three interleaved repetitions used the exact model and reference build.

| case | repack ON | repack OFF | ON/OFF |
|---|---:|---:|---:|
| P512 PP tok/s | 222.55 | 199.09 | 1.118x |
| P2K PP tok/s | 236.71 | 213.74 | 1.107x |
| P8K PP tok/s | 229.39 | 207.87 | 1.103x |
| P512 TG64 tok/s | 25.38 | 22.77 | 1.114x |
| P2K TG64 tok/s | 25.19 | 22.65 | 1.112x |
| P8K TG64 tok/s | 23.82 | 21.51 | 1.108x |

Repacking is material, but the canonical-weight reference still exceeds
MIInfer M12 by roughly 4.3x at P512. Repack alone is not the explanation.

## Original P512 attribution and Amdahl evidence

The default B4 profile measured 11,161.86 ms / 45.87 PP tok/s. In its sampled
selected chunk, deferred tail was 966.63 ms (78.8% of component time), batch
preparation was 245.78 ms (20.0%), and all other sampled families were below
0.3% each. The selected recurrent B4 family sums were:

| family | selected B4 GPU ms |
|---|---:|
| GDN core | 7.87 |
| SSM output and residual | 5.97 |
| FFN gate/up and SwiGLU | 16.87 |
| FFN down and residual | 16.64 |

The whole-tail measurement is the valid Amdahl bucket; selected-family values
are dispatch attribution and must not be added to it.

## Candidates

| candidate | primitive result | whole-runtime result | decision |
|---|---|---|---|
| M13 Q4/Q6 matrix mapping | 0.860–0.863x Q4; 0.695–0.719x Q6 vs B4 | not integrated | REJECT |
| raw int8 GEMM ceiling | optimistic 1.387x on FFN-down shape | arithmetic ceiling only | REJECT |
| FFN gate/up FP16/rocBLAS | dense GEMM; 1.5568x at B64 FFN-down lab shape | P512 45.87 → 55.72 tok/s (+21.5%) | KEEP opt-in |
| QKV + FFN dense path | no qualified continuation | P512 57.31 tok/s but divergent output | REJECT |
| SSM output dense path | no reproducible P2K win | P512 57.31, P2K 52.59; +VRAM | REJECT |
| shared dense FFN-down source | avoids OOM only by recopying per layer/chunk | P512 38.69 tok/s; 31.98 GB peak | REJECT |

The FFN candidate's post-change profile measured 9,188.69 ms / 55.72 PP
tok/s and reduced the sampled whole-tail bucket to 717.35 ms. It preserved
the exact eight-token replay and measured TG at 35.07 tok/s versus 35.13 for
control (-0.2%). It therefore passes the experimental decode regression gate,
but does not approach the P512 211.5 tok/s near-reference threshold.

## Final best MIInfer PP ladder

The best available result at each rung uses the opt-in FFN candidate where
measured and the qualified M12 path elsewhere.

| prompt | MIInfer PP tok/s | mx PP tok/s | MIInfer / mx |
|---:|---:|---:|---:|
| P512 | 55.72 | 222.64 | 0.250x |
| P2K | 52.69 | 236.72 | 0.223x |
| P8K | 44.89 | 229.20 | 0.196x |
| P16K | 36.18 | 219.91 | 0.165x |
| P32K | 33.96 | 202.84 | 0.167x |
| P64K | 28.37 | 175.75 | 0.161x |
| P128K | 20.55 | 138.39 | 0.149x |

No rung reaches the 95% threshold. The gap grows rather than closes with
context, so M22.1, M22.2, M22.3, M22.4, and final parity all fail.

## Correctness, capacity, and serving

* 23/23 CTest tests pass after the profiler change.
* FFN-candidate deterministic replay: PASS; exact IDs
  `561 3841 13477 37550 14330 42903 4906 13`.
* 8K/16K/32K continuation and replay: PASS in EXP-0270.
* 64K/128K: capacity and one-request functional smoke PASS; full replay/TG64
  qualification remains open because prefill is impractical.
* 128K allocation: 27,342,143,828 internal bytes; no OOM on load.
* lifecycle and auth remain PASS as recorded in EXP-0268 and EXP-0272;
  `scripts/test-serve.sh` passed against the rebuilt binary.
* The CPack archive was regenerated and `scripts/test-package.sh` passed.
* `git diff --check`: PASS. `graphify update .`: PASS.

## Decision

KEEP the measurement-correctness profiler and the FFN gate/up path as clearly
opt-in research code. Do not enable dense projections by default, duplicate
the full model, add workers, or claim long-context production support.

The measured B4 causal dependency, the rejected B8/B64/M13 mappings, the
repack A/B result, and the optimistic raw-int8 ceiling together prove that
local projection substitutions cannot recover the remaining 4–7x gap. M22 is
closed for this architecture. A future milestone must expose a larger
causally valid prompt chunk and rework the GDN scan/dataflow before another
kernel-family campaign is justified.
