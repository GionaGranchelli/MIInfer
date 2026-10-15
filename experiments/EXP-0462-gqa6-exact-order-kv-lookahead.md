# EXP-0462 — GQA6 exact-order Q8 K/V lookahead, GPU2

## Objective

Test whether overlapping next-tile Q8 K/V loads and QK score calculation with
the current tile's ordered online-softmax/PV recurrence improves the
restored long-context suffix on one MI50. Preserve direct-global GQA6, `(64,3)`
workgroup geometry, four-key tiles, split intervals, Q8 cache data, and the
exact order and arithmetic of every per-key softmax/PV update.

Base: `ee56020db3e1d4f1f0657d81e5c3c173fb86195f`.
Candidate branch: `perf/m31-gqa6-software-pipeline-gpu2`.
Isolated worktree: `/tmp/m31-gqa6-software-pipeline-gpu2`.

## Current bottleneck and evidence

EXP-0412 measured 26.747 s of GQA blocks in a 30.165 s restored 1,024-token
suffix; Stage1 was 25.759 s (85.5% of total), while Stage2 was 20 ms. The
current direct-global GQA6 route is the control. EXP-0461 confirms that this
frontier remains attention-heavy and that the dual-key schedule is rejected;
its control took 20.882 s on GPU2 for the same restored 73,099+1,024 workload.

In the current Stage1 Q8 path, each iteration loads K/V and scales for four
keys, calculates all four QK scores, then applies four dependent online
softmax/PV updates in key order before advancing to the next iteration. This
leaves a possible interval in which the next tile's K/QK work is not issued
while the current tile retires its recurrence. No current Q8 Stage1 stall
profile proves that this interval limits performance.

## Prior art and distinction

EXP-V2-0020 rejected register double-buffered FP16 attention: U=1 and U=2
were 18.7% and 23.1% slower, and a U=4 double-buffer attempt spilled heavily.
Its explanation was reduced load-level parallelism and register-copy/resource
cost. That evidence is a strong warning against this proposal.

This candidate keeps the current Q8 production route's four-key vector burst,
CTA geometry, GQA mapping, and serial recurrence. It would hold one current
tile plus next-tile K/scale and score state, issuing next-tile loads and QK
before retiring the current tile's recurrence. It must not reduce the issue
width to U=1/2 or combine/reassociate softmax updates. This is a narrow
reopening of the same broad software-pipeline family; proceed only if current
Q8 ISA exposes a short load-to-use wait and the candidate remains resource
healthy. If that evidence is absent or the resource gate fails, record REJECT
without any GPU run.

## One independent variable

Enable exact-order next-tile K/scale/QK lookahead in the existing direct-global
Q8 GQA6 kernel. Do not change CTA shape, tile size, key split, arithmetic,
cache representation, graph topology, or model parameters. The selector stays
opt-in and production default remains unchanged.

## Gates

- **ISA/source gate:** establish the current Q8 path's K-load-to-use waits and
  show that lookahead moves next-tile K/QK instructions ahead of the current
  recurrence without changing the recurrence sequence.
- **Resource gate:** compile the exact gfx906 candidate in the pinned image;
  require zero spills/private scratch and zero LDS. Record VGPR/SGPR and
  calculated resident waves against direct-global. Reject before inference if
  resource allocation drops resident waves or adds more than 12 VGPR without
  an evidence-backed reason.
- **Correctness gate:** same-GPU2 restored session at prefix 73,099 and
  tokenized suffix 1,024. Require finite outputs, nonempty 48-layer state
  trace, logits/state max-abs <=0.01, cosine >=0.9999, matching top two tokens,
  and final position 74,123. Exactness is diagnostic.
- **Performance gate:** one guarded same-GPU2 direct-global control/candidate
  pair in the pinned environment, with route trace in all 16 layers. Keep only
  if the candidate is at least 10% faster. No cold-64K run before a verified
  real coding-task result.
- **Safety:** GPU2 must be idle and verified as BDF `0000:88:00.0`; keep the
  80 C warning and 85 C stop guard. Never use GPU0.

## Source/ISA result and disposition

The current Q8 direct-global code object was inspected from the already built
`qwen3-gfx906.o` on Machinist; no GPU kernel was launched. The pinned image
digest was `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.
The object SHA-256 was
`3e01f6055a9aac9c402ed23459f7303f739b8067d61ecac3c1fcc26f5603ce0d` and the
disassembly SHA-256 was
`1e28955c844f81753a7fefe70612c8787ea92cef73aa5565f160e6f06d2877ca`.

The direct-global Stage1 symbol is
`qwen35_splitk_suffix_attn_stage1_quant_kernel<true,true,4,true,false,false,true,false,false,false,false>`.
Its metadata reports workgroup 192, wave64, VGPR 60, SGPR 50, zero VGPR/SGPR
spills, zero private bytes, and zero LDS. In its key/PV loop, a
`global_load_dwordx2` at `0xC6244` is followed by seven independent
instructions, then `s_waitcnt vmcnt(0)` at `0xC626C`; the Q8 unpack consumes
the loaded register immediately afterward. This is a real short load-to-use
dependency in the Q8 object, but static ISA alone does not show that it stalls
often enough to limit the workload. There is no current Q8 Stage1 stall
profile.

The new proposal still depends on the same core premise rejected by
EXP-V2-0020: register-held next-tile data should overlap global loads with
attention compute. Retaining the four-key burst and changing FP16 to the
current Q8 GQA6 route is not enough new causal evidence to reopen that family.
The original experiment also shows a U=4 double-buffer spilling risk, while
this source/resource pass did not compile a candidate.

**Disposition: LEARN; stop before implementation and GPU use.** The ISA
confirms a short Q8 load-to-use distance, not a measured stall. First collect
a Q8-specific stall/wait profile for the exact direct-global full-model route.
Only reconsider a pipeline if that profile shows meaningful VMEM dependency
stall and the new schedule changes the prior premise beyond register
double-buffering. No source selector, code object, or timing result is added
for EXP-0462.
