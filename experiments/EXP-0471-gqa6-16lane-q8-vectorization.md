# EXP-0471 — GQA6 16-lane Q8 vectorization screen — A/B INVALID

## Hypothesis

The direct-global Q8 Stage1 path issues one 8-byte K and one 8-byte V load
per lane for each query head. On gfx906, remapping each head from 32 lanes × 8
dimensions to 16 lanes × 16 dimensions could halve VMEM instruction count per
head while preserving the same bytes, full context, Q8 KV, causal mask, and
ordered online-softmax updates. This targets cache request and instruction
issue; the 99.628% historical L2 hit rate does not support an HBM-bandwidth
claim.

## Schedule distinction

One CTA still owns one token, Split-K partition, and KV head. It does not tile
query rows or keys together, change softmax grouping, share K/V between query
heads, or use matrix instructions. Four 16-lane groups per wave represent four
query heads; two waves cover the six GQA6 query heads (two groups are masked).
This differs from EXP-0417/0427/0452/0457 cross-token or Q×K tiling,
EXP-0458 four-key blocked softmax, EXP-0461 two-key Wave64, EXP-0465 Q2×K32,
and EXP-0459 pair-head K/V reuse.

## Gates

1. Keep production dispatch unchanged; candidate is opt-in and graph-cache
   configuration must include its selector.
2. Compile for the pinned gfx906 image and inspect the exact candidate object.
   Require zero scratch/spills, zero LDS, and at most 80 VGPR. Otherwise stop
   before GPU execution.
3. If the resource gate passes, run one bounded same-GPU2 8K full-model
   control/candidate screen. Preserve logits, token outputs, route trace,
   actual device BDF, and the 80 C warning / 85 C stop guard.
4. Run the 73,099 + 1,024 restored suffix only if the 8K output is finite,
   numerically acceptable, and materially faster.

## Prior evidence

EXP-0463's direct-global profile had 16 Stage1 dispatches, zero LDS/spills, and
99.628% L2 hits; external-memory counters were unavailable. The actor census
models 926.247 GB of repeated lane-level K/V load payload for the restored
suffix, but this is not physical traffic. Adjacent-row reuse would halve that
logical payload, yet the existing reuse schedule families lost on the full
model. This candidate tests a different instruction width without relying on
cross-row reuse.

## Outcome

**A/B invalid; no schedule conclusion.** The first bounded full-model screen
used unequal attention routes: the control dispatched direct-global Stage1 112
times, while the candidate dispatched the dim16 Stage1 128 times. The candidate
also replaced the cold position-zero tile, while the control retained the
existing cold path. The measured slowdown and logits difference therefore do
not isolate the lane mapping. No restored 73K suffix run was made. Candidate
source was removed pending a corrected, matched rerun.

## Build and resource gate

- Source base: `9d2705f3cb081556271621c7c87de33cdbcc5fdb`.
- Pinned image: `localhost/miinfer-m31-rocm721-libdw@sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
- Build command: `cmake --build /build --target miinfer-m31-fp16-full-model -j8`
  inside the pinned image, with source mounted at `/src` and the build
  directory at `/build`. Build succeeded.
- Candidate specialization with `UNROLL=4`: VGPR 126, SGPR 43, spills 0,
  private bytes 0, LDS 0. Rejected by the <=80 VGPR resource gate; no GPU run.
- Bounded follow-up with `UNROLL=1`: VGPR 76, SGPR 45, spills 0, private
  bytes 0, LDS 0, maximum workgroup 128. It passed the compile/resource gate.
- Fresh `UNROLL=1` gfx906 object:
  `/home/machinist/m31-gqa6-16lane-rewrite/build/CMakeFiles/miinfer_hip.dir/gfx906/kernels/qwen3_primitives.hip.o.0.hipv4-amdgcn-amd-amdhsa--gfx906`
  SHA-256 `e3d5161f0f8814fc1327e11352a0655064b556eb1dc15eaee0a65e1c9021d77c`.

## Bounded 8K screen

- Device: GPU2, PCI BDF `0000:88:00.0`, unique ID `0x2152414173497dd3`,
  architecture `gfx906:sramecc-:xnack-`. GPU0 was not used.
- Model: `/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Input: first 8,192 Qwen-tokenizer tokens from
  `/home/machinist/m31-perf-recovery-0001/build/mi50-release/m31-64k-context.txt`;
  198,637 tokens were available. File SHA-256
  `fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`.
- Both runs used the same image, binary, model, prompt, and device, sequentially.
  `ROCR_VISIBLE_DEVICES=2`, `MIINFER_HIP_GRAPH=1`,
  `MIINFER_M31_Q8_GQA6_PROD_ARITH=1`,
  `MIINFER_M31_Q8_GQA6_DIRECT_GLOBAL=1`, and
  `MIINFER_M31_TRACE_ATTENTION_ROUTE=1` were fixed. Control set
  `MIINFER_M31_Q8_GQA6_DIM16_LANES=0`; candidate set it to `1`. Other
  packed-LDS, DPP, block-softmax, wave64, sparse-window, paired-token, and dot4
  route selectors were set to `0`.
- Command shape for each run:
  `timeout --signal=TERM --kill-after=10s 600s podman run --rm --device=/dev/kfd --device=/dev/dri ... <pinned-image> /workspace/miinfer-m31-fp16-full-model /model.gguf --q8-only --tokens 8192 --prompt-file /prompt.txt --logits-out /out/<case>-logits.f32`.
- Both processes printed `device_bdf=0000:88:00.0`, processed exactly 8,192
  tokens, wrote 248,320 finite logits, and exited 0. Route trace showed 112
  direct-global control dispatches (the cold position-zero tile uses the
  existing cold path) and 128 dim16 candidate dispatches (including the cold
  position-zero tile). This dispatch mismatch invalidates the A/B.
- Control: 42.887 s prefill, first generated token 11, following token 443.
- Candidate: 45.628 s prefill, first generated token 11, following token 443.
  The observed 6.4% slowdown is confounded and is not a schedule result.
- Final-logit comparison: finite on both sides; max absolute difference
  0.9532957; RMS difference 0.1441769; cosine 0.998938424. The top-1 token
  matched at 11 and the next generated token matched at 443. Top-two logit
  indices changed from `[11, 271]` to `[11, 17044]`. This output difference
  is also confounded by the cold-route mismatch; it must not be used to reject
  the suffix schedule.
- GPU2 junction temperature ranged from 32 C to 64 C; no 80 C warning or 85 C
  stop occurred.
- Raw logs, thermal samples, input hashes, and logits remain on Machinist at
  `/home/machinist/m31-gqa6-16lane-rewrite/results/EXP-0471-unroll1-8k/`.
  Control logits SHA-256:
  `13d62d4b8f9a5869879e00ce875081a1df445ac460e18795f770abc9d78a431b`.
  Candidate logits SHA-256:
  `39026c41525ace7f3356ff5b0f5bb6fbf25caadbfc847c6265e8951240f697ea`.

## Decision

The `UNROLL=1` specialization passed compilation and resource limits. The
8K experiment does not establish its speed or correctness because the cold
tile route differed. Reintroduce the opt-in selector only behind the same
`base_position > 0 || prefill_state != nullptr` guard as production direct
global. Rebuild and require equal candidate/control route counts before
accepting another 8K comparison. Do not run the 73K restored suffix unless the
corrected screen passes numerical and speed gates.
