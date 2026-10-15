# EXP-0468 — Sparse local-window plus absolute anchors (GPU2 kill screen)

## Claim and scope

Test one explicitly approximate GQA6 Stage1 candidate. For causal key length
`T <= 2,048`, attend to every key. For `T > 2,048`, attend to the latest
2,048 keys plus absolute positions `0, 512, 1,024, ...` strictly below
`T - 2,048`. This retains all canonical Q8 KV entries and persistent-state
positions but changes which keys contribute to attention. It is experimental,
opt-in, and cannot replace the exact production default based on this screen.

## Frozen comparison

- Source base: `c7806fcca074f04fc51eac1ba2263d534e318be4`.
- Control and candidate: same pinned ROCm 7.2.1 binary, model, restored
  73,099-token session, tokenizer-backed 1,024-token suffix, GPU2 BDF
  `0000:88:00.0`, 32 Split-K partitions, existing Stage2 and one HIP Graph
  replay tile. GPU0 remains excluded.
- Control: existing exact direct-global Q8 GQA6 Stage1.
- Candidate: opt-in sparse selector, same Q8 cache/scales, FP32 per-key
  recurrence, partial workspace, and Stage2.

## Implementation gates (before GPU)

1. Selector maps a sorted, unique set; for `T <= 2,048`, it is exactly
   `[0, T)`. For larger T, every key in `[T-2,048, T)` appears once and each
   512-position anchor strictly before that window appears once.
2. A host assert oracle covers T at 1, 2,047, 2,048, 2,049, 73,099,
   anchor/window boundaries, ordinal ordering, and disjoint 32-way ceil-sized
   split partition including empty splits.
3. Candidate graph identity is distinct from direct-global. Route tracing must
   prove the requested candidate and `graph_state=1`; otherwise stop.
4. Pinned target builds for gfx906. Inspect VGPR, SGPR, spills, scratch, LDS,
   workgroup, and disassembly for absolute Q8 KV and scale indexing before
   using GPU2. GPU2 preflight verifies BDF and temperatures; warn at 80 C and
   stop at 85 C.

## GPU kill-screen gates

One same-binary control/candidate pair only. Both must restore 73,099 and end
at 74,123, process all 1,024 suffix tokens without truncation, replay the
captured graph, and show the expected route in all 16 GQA layers. Record finite
logits and recurrent state, max-absolute and cosine deltas, generated tokens,
per-route suffix wall time, and GPU telemetry. Candidate passes this screen
only if no output is non-finite, generated behavior is not catastrophic, and
suffix time is at least 10% lower than control. Catastrophic means failure to
produce a completion, invalid state/logits, or immediate nonsensical/repetitive
behavior; token equality alone is not a correctness proof for approximate
attention.

If it fails speed or outputs catastrophically, stop. If it passes, report that
a real coding-agent task with facts older than the local window remains
required before promotion. No parameter sweep and no cold 64K run is in scope
for this screen.

## Decision

## Result

The pinned build used `localhost/miinfer-dev:rocm-7.2.1` at the image digest
below and targeted gfx906. Both `miinfer-m31-fp16-full-model` and
`miinfer-gqa6-sparse-window-oracle` built successfully; the host assert oracle
passed. For the candidate Stage1 instantiation
`qwen35_splitk_suffix_attn_stage1_quant_kernel<true,true,1,true,false,false,true,false,false,false,false,true>`,
gfx906 metadata reports 54 VGPR, 48 SGPR, 0 LDS, no VGPR/SGPR spills, and no
dynamic stack. Its code object is ELF64-amdgpu; disassembly at symbol address
`0xc0000` contains the candidate kernel instructions. The selector maps
absolute key positions before Q8 K/V and scale loads.

**Performance kill-screen passed; semantic suitability remains unqualified.**
One same-binary pair completed on GPU2 with exit code 0. The executable
reported BDF `0000:88:00.0`, gfx906. Both paths restored 73,099 prefix tokens,
processed the same 1,024-token prompt suffix, replayed one HIP Graph tile, and
ended at position 74,123. Route traces showed 16/16 exact direct-global
control dispatches and 16/16 sparse candidate dispatches with
`graph_state=1`. No truncation was reported.

| Path | Restore | Suffix prefill | First two generated token IDs |
| --- | ---: | ---: | --- |
| Exact direct-global control | 2.840 s | 20.891 s | 18369, 198 |
| Sparse local-window + anchors | 2.927 s | 5.134 s | 198, 9 |

The one-sample suffix speed ratio was **4.069×** (75.4% less suffix time),
well above the 10% kill-screen threshold. This is not a repeated benchmark or
a general performance claim. All checked outputs were finite, but the
approximate path materially diverged: logits max-absolute delta 17.4687,
logit cosine 0.7955, recurrent-state max-absolute delta 21.5996, and
recurrent-state cosine 0.8628. The first two generated token IDs differed.
Those two tokens do not establish catastrophic behavior or coding-task
correctness. No coding-agent task was run, so the candidate remains unsuitable
for promotion pending a focused task with facts older than the local window.
The default remains exact and the candidate remains opt-in.

Peak GPU2 junction temperature in the guard telemetry was 56 C, below the 80 C
warning threshold. GPU0 remained excluded. Raw run output and thermal telemetry
are at
`/home/machinist/m31-gqa6-longctx-results/EXP-0468-sparse-window-gpu2/pair-1024/`.

The first-run summary labeled the sparse time field
`q8_gqa6_packed_lds_suffix_median_seconds` due to a benchmark reporting-label
fallback; the dispatch traces and per-route record identify it as the sparse
path. The source label is corrected in the follow-up commit; timings are
unchanged.

Artifacts: source commit `f6d57b3caef815782b1ec89e4f129795125a9687`;
ROCm image digest `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`;
model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`;
session SHA-256 `bbbe9543d7ccbb740cc3db1cda7bc959e8e95926587a31b9cb8e3510c557de96`;
tokenized prompt SHA-256 `fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`;
measured binary SHA-256 `cf695585b68b1056bc316aa6e3ffc4d1b38c8a44c86b0252cb521699fbeaf98a`.

**Next decision:** do not run another GPU timing sample or cold 64K prefill for
this candidate. Decide whether one real coding-task semantic screen is worth
the measured speed/quality tradeoff. Until then, retain it as experimental and
keep the exact production path unchanged.
