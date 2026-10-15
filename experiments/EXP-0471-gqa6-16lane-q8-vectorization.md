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

## Initial screen (invalid A/B)

**A/B invalid; no schedule conclusion.** The first bounded full-model screen
used unequal attention routes: the control dispatched direct-global Stage1 112
times, while the candidate dispatched the dim16 Stage1 128 times. The candidate
also replaced the cold position-zero tile, while the control retained the
existing cold path. The measured slowdown and logits difference therefore do
not isolate the lane mapping. No restored 73K suffix run was made. Candidate
source remained opt-in while the corrected matched rerun was prepared;
production dispatch remained unchanged.

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
- After correcting per-KV query-head assignment, the pinned target rebuilt
  and linked successfully. Corrected code object SHA-256:
  `f4c915572d28a078be1d846bd25d85c13be3835c30caf9cdfba6ce864d881ce7`;
  host HIP object SHA-256:
  `e544a2a21ce80abc0617814f4d015fb4949e31df9acf1cc4aee67037a1f63a17`.
  The candidate specialization is VGPR 76, SGPR 45, zero spills/private bytes,
  zero LDS, maximum workgroup 128. The linked benchmark binary SHA-256 is
  `37827b37728b8a8ceed924ae41cafe4847a947cdfef5067c68a535bd0573e5b8`.

## Initial bounded 8K screen

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

## Interim decision (before corrected rerun)

The first A/B is invalid. A second attempt was stopped before control
completion after an incorrect interpretation of `rocm-smi` GPU counts. The
candidate also had a head-mapping bug that was fixed and rebuilt before the
completed matched run documented below.

## Corrected rerun (completed)

- The opt-in dispatch uses the same `(base_position > 0 || prefill_state != nullptr)`
  condition as production. The runner pins index 2 and checks it resolves to
  BDF `0000:88:00.0` before each case.
- The runner checks every KFD PID with `rocm-smi --showpidgpus`; it refuses to
  launch if a PID maps to the selected device, if the mapping cannot be parsed,
  or if selected-device VRAM exceeds 1 GiB. It requires both logits files and
  equal nonzero control/candidate route counts before reporting a valid run.
- The first invocation was deliberately stopped by the agent at the parent's
  direction after 92 s, while control was active, based on a mistaken reading
  of the `rocm-smi` `GPU(s)` column. It exited 137, wrote no logits, and did
  not start the candidate. Preserve that attempt as aborted; it has no timing
  or correctness result.
- The persistent restore server PID 2631926 maps through `rocm-smi
  --showpidgpus` to GPU index 0 / BDF `0000:04:00.0`, while the candidate is
  pinned to GPU index 2 / BDF `0000:88:00.0`. Its 30.1 GB allocation does not
  establish contention on the candidate GPU. The server was not stopped.
- Review found a candidate mapping error before the completed run: the old formula
  assigned four 16-lane groups per wave to a six-query-head KV group, causing
  overlapping and misrouted heads. The source now assigns three heads per
-  wave and masks the fourth group. The corrected build passed the resource
  gate above.
- The corrected run completed in control-then-candidate order. Control prefill
  was 42.912 s; candidate prefill was 45.396 s (one sample per side), so the
  candidate was 5.8% slower. Both exited 0, processed 8,192 tokens, and
  reported 248,320 finite logits. The first generated token was 11 and the
  second was 443 on both sides. Route counts matched at 112/112.
- Logit comparison: max absolute difference 0.7885966, RMS difference
  0.1105030, cosine similarity 0.99933815. Top-1 stayed token 11; top-2
  changed from token 271 to 17044. Thus generated tokens matched in this
  sample, while the numeric difference remains visible in the evidence.
- The corrected run's inputs, control/candidate logs, thermal samples and
  logits are under
  `/home/machinist/m31-gqa6-16lane-rewrite/results/EXP-0471-unroll1-8k-matched/`.
  Input hashes: model `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`,
  prompt `fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`.
  Control/candidate logits hashes:
  `13d62d4b8f9a5869879e00ce875081a1df445ac460e18795f770abc9d78a431b`,
  `653cf72be5294dbf207ab5c72bf8f2f78893c68c9d888e007a3feb884018bc00`.
  Control/candidate logs hashes:
  `b987e7b4b078201be177349de3ef55cacd64b6b79f2a79eb30d077633e740479`,
  `b1bbb3598c8752accc960e91454e804c6c655a3de091b5887c4301ffaac98f8e`.
  The interrupted run is separately preserved at
  `/home/machinist/m31-gqa6-16lane-rewrite/results/EXP-0471-unroll1-8k-matched-aborted-20261015T151235/`.

## Final decision

**Reject the DIM16 schedule for performance.** The corrected route-matched
screen was 5.8% slower, one sample per side. It passed finite-logit and route
count checks and generated the same two tokens, but it did not meet the
required clear speed-gain gate. No 73K restored suffix run was made. Production
dispatch remains unchanged; do not promote this schedule.
