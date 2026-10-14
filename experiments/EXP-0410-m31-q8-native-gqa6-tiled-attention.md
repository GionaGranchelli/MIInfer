# EXP-0410: Native-Q8 GQA6 tiled attention

## Question

Can the production Q8 attention math load each KV-head's Q8 K/V tile once and
reuse it across all six GQA query heads, while preserving production QK score
arithmetic and full-model logits?

## Candidate

The opt-in `MIINFER_M31_Q8_GQA6_NATIVE_TILED=1` path assigns one CTA to one KV
head and four query tokens. The CTA covers the 24 query rows for the six query
heads associated with that KV head (32 row slots, with the final eight masked).
It stages native Q8 keys and values, then reuses them across all six heads.
The query stays FP32. For every Q/K dimension it computes
`key = float(q8_byte) * key_scale`, followed by
`dot = fmaf(q_fp32, key, dot)`, then applies the production attention scale.
There is no Q quantization, residual plane, or integer-dot approximation.

This is an isolated opt-in kernel candidate. Production dispatch remains the
default unless explicitly enabled. The suffix graph cache key includes the
selector so a graph captured with another attention path is not reused.

## Offline checks

- Base: `3805063`.
- Build: `docker.io/mixa3607/rocm-gfx906:7.14-complete`, local image digest
  `sha256:622c025824d465faa0f5e59eea4f723845032011df273c3b11e25031965fc428`.
- Configure and target `miinfer-m31-fp16-full-model`: PASS for gfx906.
- HIP compiler: Clang 23.0.0; build target compiled and linked.
- Candidate kernel resources: 55 SGPR, 85 VGPR, zero VGPR spill, zero scratch,
  two waves per SIMD. The compiler reports zero statically allocated LDS;
  launch-time dynamic shared memory is supplied by the wrapper.
- `git diff --check`: PASS.
- Initial pinned-image configure failed because CMake did not search the
  image's `/opt/rocm-7.2.1` prefix. Re-running with
  `CMAKE_PREFIX_PATH=/opt/rocm-7.2.1` fixed discovery; no source or build-image
  changes were needed.

## GPU gate

For every full-model gate, stop on non-finite logits, cosine below `0.9999`,
top-1 or second-token mismatch, or the thermal stop threshold. Report latency
only if the correctness gate passes. A 64K run requires the same-GPU production
control and candidate with identical build, model, prompt, and capacity.

## 64K GPU0 production-control attempt (stopped)

- The pinned binary `7453c42f...` ran the production Q8-only full-model path
  for `--tokens 65536` against `Qwen3.8-27B-Q4_K_M.gguf` on GPU0
  (`0000:04:00.0`). The configured capacity was 69,632 tokens. The process
  stayed runnable at 196% CPU while GPU0 remained 100% busy, but emitted no
  progress/result line after the binary SHA; stdout was buffered. It held
  about 30.48 GB VRAM.
- After 17:18 elapsed (34:00 process CPU time), this exceeded the agreed
  runtime review bound and was stopped with `podman stop --time 10`. SIGTERM
  did not stop it, so Podman escalated to SIGKILL. Wrapper/guard exit was 137.
  The guard recorded 911 temperature samples, max junction `72 C`; neither
  80 C warning nor 85 C hard stop fired. GPU0 returned to 0% use, about
  10.9 MB VRAM and `37 C` junction after cleanup.
- No control logits or completed timing were produced by this attempt. The
  existing production Q8 64K logits artifact is
  `/home/machinist/m31-prefill-0002-20261014/q8-control-gpu2-64k.f32`;
  its run log reports 65,536 processed prompt tokens, capacity 69,632,
  finite logits, and 1,883.543 s prefill on GPU2 BDF `0000:88:00.0`. It is a
  prior build/run and therefore can only serve as a preliminary numerical
  reference for the candidate, not a matched performance control.

## GPU2 visibility preflight failure

- The first separate candidate-only 64K launch used
  `ROCR_VISIBLE_DEVICES=GPU-02152414173497dd3` inside the container and failed
  before model load with `hipGetDeviceCount failed: no ROCm-capable device is
  detected`; no candidate logits were written. This first attempt omitted the
  complete GPU device mapping. Follow-up preflights using only `/dev/kfd`
  exposed CPU agents but no GPU, while adding individual card/render nodes
  returned HSA `OUT_OF_RESOURCES`.
- Existing successful run scripts use the complete shape
  `--device /dev/kfd --device /dev/dri --security-opt label=disable`, and pass
  `ROCR_VISIBLE_DEVICES=2` inside the container. A pinned-image `rocminfo`
  preflight with the complete device shape and `ROCR_VISIBLE_DEVICES=0`
  exposed gfx906 UUID `GPU-021678e17348c2f7` (GPU0); the full inference process
  independently confirmed BDF `0000:04:00.0`. GPU2 was not retried after this
  correction, so its availability under the pinned image remains unverified.

## 64K GPU0 native-Q8 candidate-only attempt (stopped at bound)

- Used the pinned image and binary above, model SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, the
  existing tokenizer-backed prompt, 65,536 processed tokens, and 69,632-token
  capacity. The feature environment was confirmed in the live process:
  `MIINFER_M31_Q8_GQA6_NATIVE_TILED=1` and `ROCR_VISIBLE_DEVICES=0`.
- The inference process printed BDF `0000:04:00.0` and entered `q8_only`. It
  remained runnable and compute-bound through 15:01 elapsed (about 29:24 CPU
  time, 195% CPU, GPU0 100% busy, 30.48 GB VRAM). It produced no completed
  timing/logit line or logits file before the 15-minute cap, so there is no
  correctness or performance result from this run.
- The wrapper stopped the tagged container at the cap. SIGTERM did not stop
  inference; Podman escalated to SIGKILL. Exit was 137. Thermal sampling
  recorded a maximum junction of `71 C`; no 80 C warning or 85 C stop fired.
  GPU0 returned to 0% busy, about 10.9 MB VRAM, and `39 C` junction after
  cleanup.
- The candidate-only run cannot be compared against the prior
  `486.268 s` GPU0 control as a valid speed result: the historical control was
  built from a different source revision and used a different attention/KV
  mode. The planned full-logit comparison against the stored Q8 reference
  could not run because this candidate did not produce logits.

## Pinned 2K full-model gate

- Machinist GPU0 was idle before launch: BDF `0000:04:00.0`, gfx906, unique ID
  `0x21678e17348c2f7`. The process itself printed
  `device_bdf=0000:04:00.0 architecture=gfx906:sramecc+:xnack-`.
- Image digest:
  `localhost/miinfer-m31-rocm721-libdw@sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`;
  local image ID `f5f33d8e8efbe77f0a08335140f9c9a162202067189499be9dbddd2eb214afa4`.
  Runtime HIP reports ROCm 7.2.53211; pinned image uses Clang 22.0.0.
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Binary SHA-256:
  `7453c42f11a4773ab427f8e74cae8e2505a45f2d9b558722bcf5abb05761bb82`.
- The pair used 2,048 deterministic token IDs from the benchmark's existing
  sequence and capacity 65,536. Q8/Q8 control ran first, then the candidate.
- Correctness passed: finite logits, max absolute difference `0.181160450`,
  cosine `0.999952746`, and matching first/second token IDs (`14` / `16`).
- Prefill: production control `9.344 s`; candidate `8.949 s`; one-pair ratio
  `1.044x` (4.4% faster). This is a single short-prompt pair, so it is not a
  qualified speed claim. The difference is modest and order-sensitive; a
  long-context paired run is needed to tell whether GQA load reuse matters at
  the target context length.
- Thermal wrapper sampled junction temperature 99 times, max `58 C`; no 80 C
  warning, 85 C stop, telemetry failure, or timeout. GPU0 returned to 0% use
  and about 10.9 MB VRAM after the run.

## Concurrent 64K full-model screen (candidate fails correctness)

- Ran production Q8/Q8 control and native GQA6 Q8 candidate concurrently from
  the same pushed source/binary and pinned image, using the same model,
  tokenizer-backed prompt, 65,536 processed prompt tokens, and 69,632-token
  capacity. Control ran on GPU2 BDF `0000:88:00.0`; candidate ran on GPU0 BDF
  `0000:04:00.0`. Both inference processes printed their own BDF and
  `prompt_tokens_processed=65536`. This is a concurrent cross-card screen,
  not a definitive same-card A/B timing comparison.
- Both wrappers exited `0`; no timeout or thermal guard fired. Candidate
  completed at 945.473 s prefill (69.3 processed prompt tokens/s), first token
  `94450`, 0.078 s decode, and second token `76`. Production control completed
  at 1,954.231 s prefill (33.5 tokens/s), first token `271`, 0.083 s decode,
  and second token `248068`. The raw timing ratio is `2.067x` in favor of the
  candidate, but correctness failed, so this speed result is not actionable.
- Both full logit arrays contained 248,320 finite `float32` values. Compared
  candidate to this run's concurrent production control: maximum absolute
  difference `9.979642868`, RMSE `1.264166975`, cosine `0.908032863`, and
  top-1 indices `94450` versus `271`. The generated first-token IDs also
  differ (`94450` versus `271`). Reject this candidate for production use;
  do not present its timing as a valid improvement.
- Thermal telemetry sampled 313 times for the candidate (maximum junction
  `71 C`) and 630 times for control (maximum `68 C`); both remained below the
  80 C warning and 85 C stop thresholds. At completion GPU0 and GPU2 returned
  to idle utilization and about 10.9 MB VRAM each.
- The complete run logs, guard records, exit codes, thermal samples, and raw
  logits are preserved under
  `results/m31-native-q8-tile/parallel-64k-20261014/`. SHA-256:
  candidate logits `9ed191a625f26f0183f0b79b3b2941921da388276f7a4b8373d97d719bcd922d`;
  control logits `9a33d3f3b190126806dee63074189df0f83b78503eac9f2f80e881babb16f482`.
- Next gate: localize the earliest context length where full logits diverge,
  using matched builds, token prefixes, model, and capacity. Start with the
  already available 8K/16K checkpoints on separate MI50 cards, verify each
  process BDF, and use the same thermal guard. Do not perform another timing
  sweep unless correctness passes.

## Provenance

Prior source review examined the public gfx906 llama.cpp fork at
`1373e889...`, MX-Llama at `71a0624...`, and Strata at `a0116b2...`. Only the
general dataflow idea of sharing KV loads across GQA heads informed this
candidate. No source code was copied from those projects; this implementation
is written against MIInfer's existing kernel and cache interfaces. No new
third-party license terms or source files are introduced.
