# EXP-0411: Q8 GQA6 KV reuse with production arithmetic

## Objective

Test whether one CTA can stage each Q8 K/V four-key tile once per KV head and
reuse it across all six associated query heads while preserving the production
Q8 split-K arithmetic order.

## Evidence and hypothesis

EXP-0410's native GQA6 candidate changed both the QK reduction order and the
online softmax/value update granularity. Its 64K full-model logits failed
parity, so its latency result is not actionable. The production
`qwen35_splitk_suffix_attn_stage1_quant_kernel<true,true,4>` is the arithmetic
reference: each 64-lane group handles two query heads, performs the same
lane-local QK partial products and XOR reduction, applies `(key_scale * scale)`
after reduction, and updates online max/sum/value once per key in order.

This candidate adds only cross-head Q8 K/V staging. A CTA covers one query
token, split, and KV head. Three 64-lane groups cover the KV head's six query
heads as pairs. The CTA stages each four-key K/V tile and its scales into LDS;
each pair then runs the production arithmetic sequence against that staged
tile. The selected target remains gfx906/Wave64.

## Gates

- Keep the selector opt-in; baseline dispatch and arithmetic remain unchanged.
- Build with the pinned ROCm 7.2.1 image and test correctness before treating
  latency as a result.
- Compare all float32 output logits, not only top tokens; require finite output,
  max absolute error <= 0.01, cosine >= 0.9999, and matching first two tokens.
- Use a same-GPU comparison for a qualified timing claim. Two-card concurrent
  timing is a screening result because the cards differ.

## Status

### Implementation

The opt-in selector is `MIINFER_M31_Q8_GQA6_PROD_ARITH=1`. One CTA covers a
query token, split, and KV head. Three Wave64 groups process the six query
heads in pairs while sharing staged four-key Q8 K/V tiles and scales in LDS.
Each group retains the production QK reduction, scale placement, and per-key
online softmax/value update order. The graph cache key includes this selector.
The production dispatch remains the default.

Build passed on Machinist with the pinned image
`localhost/miinfer-m31-rocm721-libdw:20261014T094012Z`, digest
`sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
The target `miinfer-m31-fp16-full-model` exited 0. Binary SHA-256:
`84aa8b788d3bcec90fb8263243220bd1c8035c4d1c3269045279faef572aca2a`.
The source diff from base `91fff9a1053d544ba021531707db5884b8711830` has
SHA-256 `8a9bf8ece3d417e52a98733dddae2e9cfe3767469fd0076e287c8237d4778ffd`.

### Correctness and timing

Both guarded 8K runs used the same binary, fixed prompt, Q8/Q8 cache, and GPU
`0000:04:00.0`. Production took 54.336 s; the candidate took 43.429 s. Both
processed 8,192 tokens at capacity 65,536, emitted 248,320 finite logits, and
selected first/second tokens `11`/`443`. The complete logits matched exactly.
This is one sequential sample per path; the 20.1% lower candidate time is
promising, not a stable estimate.

For the 64K screen, production ran on GPU2 `0000:88:00.0` while the candidate
ran concurrently on GPU0 `0000:04:00.0`. Both used the same binary, model,
fixed prompt, Q8/Q8 cache, and capacity 69,632, and each processed 65,536
tokens with 248,320 finite logits. Production took 1,882.718 s; the candidate
took 938.992 s. Both selected tokens `271`/`248068`. Their complete
993,280-byte float32 logits files have identical SHA-256:
`9a33d3f3b190126806dee63074189df0f83b78503eac9f2f80e881babb16f482`.
This establishes bitwise full-logit agreement. The 2.00x timing ratio is only
a cross-card, concurrent screen and is not the qualified performance claim.
Raw captures are on Machinist at
`/home/machinist/m31-q8-prod-arith-64k-parallel-20261014/`; the control log,
candidate observed-result record, metadata, exit codes, environment, guard
samples, logits, and hash comparison are retained there.

The 8K guard samples peaked at 70°C for production and 72°C for candidate. In
the 64K screen, the production guard peaked at 68°C and candidate at 79°C;
neither reached the 80°C warning. Both workloads exited 0 and left no KFD
processes. Candidate output was captured from the live process because the
runner removed its container before saving the normal run log; its observed
result, environment, raw logits, and guard samples were preserved. The runner
then returned 1 because its cleanup attempted to remove each container twice;
this did not change either workload exit status.

### Reference-engine screening

No external kernel code was copied. Relevant source revisions inspected were
Strata `a0116b2`, milpster/gfx906-llama-cpp `1373e889`, mx-llama.cpp `71a0624`,
and upstream llama.cpp `91c631b`; the checked-out trees report MIT licensing.
Strata QSA is not compatible with this frozen Qwen3.8 GGUF: its tensor
directory has no indexer tensors, and its documented path uses two KV heads
while this model has four. Its gfx906 QSA query swizzle reported only 1.19%
full-model gain. Milpster's K/Q register prefetch is a possible future lead,
but only after MIInfer profiling confirms LDS return stalls; its reported
31% is a kernel-specific observation, not a model-level gain.

### Next gate

Run production and candidate sequentially on one MI50 at 65,536 tokens to
qualify the timing. Keep the two-card measurements as correctness and early
screening evidence only. Then push the verified implementation and evidence.
