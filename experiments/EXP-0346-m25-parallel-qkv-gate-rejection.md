# EXP-0346 — M25 parallel recurrent QKV/Gate composition

**Status:** REJECTED; candidate removed
**Milestone:** M25 stretch follow-up
**Date:** 2026-09-12
**MIInfer source:** `5048b4b`
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Hypothesis

The oracle materializes recurrent QKV/Z independently from the other
attention projections. Running the recurrent QKV projection on an auxiliary
stream while the main stream prepares and runs Gate might expose useful GPU
overlap at the full P512 execution-contract level.

## Candidate

For Mx Q6_K recurrent QKV layers only, the candidate created a second Mx Q8
activation buffer, quantized the normalized input with the non-affine format,
and ran QKV on a nonblocking auxiliary stream. The main stream quantized the
same normalized input for Gate and launched Gate concurrently, then waited for
QKV before completing the stage. Q4_K recurrent QKV layers remained on the
qualified serialized path.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- P512 repeated-fox prompt, context capacity 1024
- SCLK/MCLK fixed at `1606/1000 MHz`
- clean `env -i` with the complete `m25_hi_qualified` vector explicitly set

## Trial and failure

An initial screen incorrectly combined the candidate selector with
`MIINFER_PRESET=m25_hi_qualified`. The preset deliberately clears unknown
MIInfer selectors, so that screen measured the control path for both labels
and was discarded.

The corrected trial supplied the full qualified vector directly. The control
completed at `2525.97 ms`. The candidate loaded the model and reached prefill,
but produced no latency line within the 180-second process timeout and was
terminated. It produced no correctness result and no usable performance
sample.

The implementation also exposed a workspace-ownership error: shared
wide-prefill attaches `prefill_mmq_q8` after construction, while the
candidate's additional Q8 buffer was allocated only in the per-layer,
non-shared allocation branch. The active shared path therefore did not have a
valid second Q8 workspace. The candidate was removed before any result was
interpreted as evidence about the oracle's GPU schedule.

The raw logs and telemetry are retained outside the repository at:

```text
/tmp/m25-parallel-qkv-gate-corrected-20260912/
```

The corrected control reported `device_allocated_bytes=21993242964` and
`device_vram_free_bytes=11484004352`. Telemetry held SCLK/MCLK at
`1606/1000 MHz` during the trial.

## Decision

**REJECT.** No runtime branch, selector, stream, event, or extra workspace is
retained. This is a failed implementation contract, not a measured rejection
of the oracle's independent QKV/Gate schedule. The previous EXP-0345 input
overlap remains rejected on its valid `+0.16%` P512 result.

## Follow-up

Do not retry this candidate until a complete shared-workspace design exists.
If revisited, first run one representative Q6 layer with an explicitly wired
workspace and finite-output check, then measure the whole stage with outer
events before attempting end-to-end timing.
