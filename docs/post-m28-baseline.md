# Post-M28 production baseline

Date: 2026-09-29  
Production checkpoint: `e68c0f20b0eceb20cfe8fede9125b7feef029e6b` (iteration 26)  
Closeout/evidence commit: `665f996b372178cd63cc83037a8e0ade63df10e4`

## Decision

M28 is qualified and V2-0043 is **PRIMARY GOAL PASS** at iteration 26. The
candidate after iteration 26 was not promoted: real-model parity failed, and
its first tolerance-breaking difference was FP32-to-FP16 KQ-fragment storage
carried into V accumulation. Preserve `e68c0f20` as the production baseline.
The numerical limitation is future V2-0044 attention-frontier work, not an
open blocker for this goal.

The source diff from `e68c0f20` to the closeout head is empty for
`gfx906/`, `include/`, `src/`, and `tools/`; the later commit records the
experiment disposition and generated knowledge graph only.

## Performance evidence

Target: Qwen3.8-27B-Q4_K_M, model SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, one
MI50/gfx906, Wave64, SCLK/MCLK 1606/1000 MHz.

Iteration 26's five-pair isolated P8192 suffix-attention runs measured a
combined median of 200.612 ms, versus 241.06 ms for iteration 22 (16.8% lower)
and about 42.5% below the then-current control. The exact saved P8192 serving
request measured 38.123261 s median across three requests; pinned mx-llama.cpp
measured 38.936731 s median on the matched raw formatted prompt. These are the
qualification measurements documented in
[`EXP-V2-0043`](../experiments/EXP-V2-0043-reference-shaped-gqa-attention-rewrite.md).

Fresh-build smoke on 2026-09-29 used `MIINFER_PRESET=m25_hi_qualified`,
`MIINFER_V2_0043_GQA_ATTENTION=1`, no session reuse, runtime context 16,384,
and the exact saved 8,192-token request. It returned HTTP 200, processed 8,192
prompt tokens and one completion token (token ID `248068`), and took 38.141 s
client wall / 38.132 s internal prefill. A separate short request generated
six tokens successfully. This is a single smoke, not a repeated performance
qualification.

The narrow decode A/B in [`EXP-V2-0042`](../experiments/EXP-V2-0042-p8192-trace-and-sampling-regression.md)
also does not support a ~20 ms/token loss: MIInfer (`3f412d9b`) measured
34.032 ms/token at P64 and 35.953 at P2048, versus pinned mx at 39.571 and
40.192. Generated text differed across engines, so this is timing evidence,
not a token-equivalence claim.

## Build and correctness checks

- Clean Release build from closeout head `665f996b`: passed. Build used the
  `vllm-gfx906-7.2.1` container, ROCm 7.2.1, HIP Clang 20.0.0, and GNU 16.2.1.
- Host tests: 11/11 passed.
- GPU tests run through the host dynamic loader to work around the container's
  newer glibc requirement: 9 passed. Five model integration tests skipped
  because no model path was supplied; they are not claimed as model tests.
- Fresh-build short generation and exact P8192 smoke: passed as described
  above.
- After stopping the test server, `/dev/kfd` had no MIInfer owner and GPU use
  returned to 0%; clocks were 1606/1000 MHz and junction temperature 39 C.

## Next frontier

Only after this closeout is merged and post-merge checks pass, create the
V2-0044 attention-frontier branch from the merged checkpoint. First revalidate
the measured P8192 gap and inspect the exact pinned mx/upstream execution
contracts. Keep iteration 26 immutable; do not reopen the rejected parity
variants or touch decode, MMQ, power tuning, or unrelated prefill paths.
