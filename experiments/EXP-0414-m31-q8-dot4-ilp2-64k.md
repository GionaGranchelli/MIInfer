# EXP-0414 — Q8 dot4 ILP2 full 64K cold prefill

## Question

Does interleaving two Q8 integer dot accumulators make the opt-in dot4 attention
path a faster, correct full-model cold-prefill path at 65,536 tokens?

## Decision

**REJECT for correctness.** The full-model candidate completed 65,536 tokens in
854.657 seconds and produced finite logits, but its logits fail the existing
gate by a wide margin against the exact Q8 production result. The same top-1
token does not qualify the output. Keep the path opt-in and disabled by default.

## Frozen workload and provenance

- Model: `Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Prompt: `m31-64k-context.txt`, SHA-256
  `fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`.
- Processed tokens: exactly 65,536; capacity: 69,632; Q8 KV mode.
- GPU: Machinist MI50, BDF `0000:88:00.0`, unique ID
  `0x2152414173497dd3`, gfx906 `sramecc-:xnack-`.
- Runtime image: `localhost/miinfer-dev:rocm-7.2.1`.
- Environment: `MIINFER_M31_Q8_DOT4_ATTENTION=1`,
  `MIINFER_M31_Q8_DOT4_ILP2=1`, `ROCR_VISIBLE_DEVICES=2`.
- Binary mounted when the process started: SHA-256
  `5e5b4a616cb8bb356322b78858ed6a7ffa2190858bd5cff699be7abac148b4ce`.
  A later rebuild replaced the path while this process was running; the
  in-flight process continued using the original executable inode.
- Source revision used to build the running binary:
  `1a96034b5ce9417dd2919a486f461f91d0b4f0c0` on
  `codex/m31-prefill-attn-candidate`.
- Source SHA-256 at the checked candidate checkout: kernel
  `84a230686aa1f7f58c3375e93149b16d3bcc5a588cbecbabfeef1e7cdecb1d81`,
  attention dispatch
  `aba93b3197ae2414cfb294b682ad6958c1f78013f2750e51dd09dbeebb9f6525`,
  benchmark
  `808cf0a55c497abf281c7582dfcc161e8b8142c2943b84f74652b7389baea8f3`.
- Candidate logits: SHA-256
  `2ce0fbc887f1865d48e7bbc3cb5c8f6a96572efcac7171273b8730e3651d5b7f`.

## Result

The benchmark exited successfully and reported. Source dispatch is eligible
for dot4 after the first 1,024-token cold tile (`base_position > 0`); this run
did not include a profiler trace counting the selected kernels.

```text
q8_only prefill_seconds=854.657 first_token=271 decode_seconds=0.083 second_token=248068
q8_logits_finite=1 q8_logits_count=248320
```

Compared with the exact 64K production Q8 output (SHA-256
`9a33d3f3b190126806dee63074189df0f83b78503eac9f2f80e881babb16f482`):

- max absolute logit error: `6.48067093` (limit `0.01`)
- cosine similarity: `0.94919340` (minimum `0.9999`)
- top-1 token: `271` on both paths
- result: **FAIL**

The prior same-GPU Q8 production run took 1,883.543 seconds, so the candidate
is about 2.20x faster by raw wall time. This is indicative only: the control
was from an earlier run/build, and the candidate fails correctness. It is not
a qualified speedup. The fastest verified correct 64K result remains the
FP16-cache path at 487.521 seconds. The exact Q8 4Q LDS candidate took
939.534 seconds and matched production logits, so this dot4 result is faster
than that Q8 candidate but unusable.

The same dot4 family had already failed a same-build 2K full-model check:
max absolute error `0.751942`, cosine `0.999789`. Layer scans showed local
attention error growing from `7.15e-7` at layer 3 to `2.10e-5` at layer 63.
ILP2 interleaves integer accumulators and combines their exact integer sums;
it changes instruction dependency chains but does not recover precision lost
by Q8 query quantization or the changed attention/reduction arithmetic.

## Resource and run status

The 64K candidate completed on its own with exit status 0. The paired GPU0
control was stopped early and exited 137; it produced no valid timing and is
not used in this comparison. The candidate's thermal samples stayed well
below the 80 C warning and 85 C stop thresholds. Final telemetry showed the
GPU idle after completion.

## Next step

Do not spend another 64K run on Q8 dot4/ILP2. A new attention candidate should
preserve the production FP16 query/score precision while changing the matrix
execution, then pass a same-card full-logit 2K/8K gate before any 64K run.
The most direct distinct direction under review is gfx906 MFMA-based QK.
