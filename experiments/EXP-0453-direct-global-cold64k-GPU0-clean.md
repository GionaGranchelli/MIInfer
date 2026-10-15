# EXP-0453 — Direct-global GQA6 cold 64K screen on GPU0

**Verdict: exact and faster in one matched pair; promising, not yet a repeated performance result.** Direct-global Q8 GQA6 reduced the timed 65,536-token prefill from 730.222 s to 685.076 s on the same MI50, a 45.146 s reduction (6.18%, 1.066x). The logits were bit-identical. This independently reproduces the size of the earlier 6.18% observation without concurrent GPU work.

## Method

- Same sequential A/B protocol: packed-U128 control, then direct-global candidate.
- Host: Machinist; GPU0 BDF `0000:04:00.0`, gfx906. GPU2 and GPU1 had no inference workload.
- Source: `28e467f07704985ec02486c553da4300a63615d8`; binary SHA-256 `cca0c79ac9b40d0b33dcf896f8aaa3962c7b8c146711253862ca7f622add5207`.
- Image: `localhost/miinfer-m31-rocm721-libdw@sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Prompt SHA-256: `fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`.
- Both paths processed exactly 65,536 prompt tokens, with capacity 69,632. Each produced 248,320 finite logits and 1,008 expected attention-route trace entries.
- Both returned first token `271`, second token `248068`, and logits SHA-256 `9a33d3f3b190126806dee63074189df0f83b78503eac9f2f80e881babb16f482`.
- Thermal guard was attached for both runs. Peak junction temperature was 76 C for control and 79 C for candidate; no warning or stop was recorded. Both container exit codes were zero, and no KFD PIDs remained afterward.

| Path | Timed prefill | Throughput | Start → finish | Peak junction |
| --- | ---: | ---: | --- | ---: |
| Packed-U128 control | 730.222 s | 89.75 tok/s | 00:31:20–00:44:24 UTC | 76 C |
| Direct-global | 685.076 s | 95.66 tok/s | 00:44:24–00:56:42 UTC | 79 C |

## Limits and interpretation

This is one serial pair, with control always first. The candidate started warmer (44 C versus 24 C), so thermal and run-order state were not perfectly matched. No stage profile was collected during this run; the result attributes the difference to the selected direct-global route only as a measured A/B association, not as a proven limiter analysis. Do not call this a median or a production-wide speedup. The earlier contaminated pair measured 730.607 s versus 685.429 s and had the same exact outputs, but is excluded from attribution because GPU2 work overlapped the control and a HIP build overlapped the candidate.

The clean pair confirms that direct-global preserves the tested full-model computation and is worth retaining as an opt-in path. The approximately 6% cold-prefill gain is materially short of the 64K target; continue with the GQA6 attention rewrite rather than treating this as mission completion.

## Raw evidence

Machinist: `/home/machinist/m31-gqa6-longctx-results/EXP-0453-direct-global-cold64k-GPU0-clean/` (`control/` and `directglobal/` contain run logs, metadata, guard telemetry, and logits; `summary.txt` contains the output hashes). The binary logits are retained on the host and are not committed.
