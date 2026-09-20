# EXP-0358 — M26-C current decode-route differential

**Status:** ACTIVE — route mapping complete; measurement gate open
**Date opened:** 2026-09-20
**Starting commit:** `9180e6f5b41c6ece62d68a2f4c153a4a480429af`
**Route scaffold commit:** `1b5c0d497c378f4b2bd00d14bbc2fe2a5225d3a4`
**Measurement implementation base HEAD:** `7bca95720521be6d8cd17b3961d4b5580eedfa96`
**Scope:** Current code only. M27 remains untouched.

## Question

How much of the MI50 P512 decode cost differs between the current legacy
batch-major and interactive layer-major routes after both begin from the same
semantic state? Attribute the difference by whole-token operator family,
dispatch and memory traffic, route selection, and position transition.

This does not compare current code with historical M8/M9 results. No kernel or
selector change is allowed until attribution is closed, except a correctness
fix required to establish equivalent state.

## Stop gate

Stop when at least 90% of the measured route delta is attributed, no removable
family accounts for at least 3 ms/token, the remaining cost is shown to be
required by the current contracts, or equivalent comparison proves impossible
without changing semantics. Do not proceed into M27 work.

## Contract mapping

`Qwen35RuntimeEngine.generate()` selects `generate_fresh()` when
`MIINFER_PREFILL_LAYER_MAJOR` is off and `generate_layer_major()` when it is
on. The former processes `P-1` prompt inputs then consumes the final prompt
token through `step()`; the latter prefills all P prompt tokens and computes
the first output token from final hidden state. For P512, both can reach the
semantic boundary after all 512 inputs have been processed, at position 512,
with the next generated token available. Their built-in timings do not match:
legacy includes the final prompt token in a normal decode step, while
interactive counts the LM-head first-token operation with its decode time.

Legacy direct decode synchronizes and copies one token D2H per step. Legacy
queued graph replays at positions beginning 511 and counts graph launches per
token. Interactive direct starts its post-prefill decode at position 512;
interactive graph replays from there and bulk-copies generated tokens. Graph
capture time is tracked separately. These existing timing fields cannot be
compared as-is; the harness must time a common post-P512 boundary and report
prefill/first-token/graph setup separately.

`--check-graph-state` already runs graph then direct in one runtime and checks
token parity plus byte-identical recurrent state, convolution history, and
active K/V snapshots. `snapshot_decode_buffers()` has no corresponding import
helper, and session checkpoints reject differing runtime identities. Thus no
existing cross-route state restore is available.

The `m25_interactive` preset changes more than scheduling: it selects wide,
full-layer-major Mx prefill/repacked resident weights, Mx GDN prefill, wide Mx
attention decode, and `MIINFER_MX_MMV=1`. The latter selects one-token Mx MMV
dispatches in `gfx906/kernels/kquant_wave_layout.hip`. Constructor-owned layer
configuration and weight representation prevent toggling the route safely on
one already-created engine. A serial reload/import diagnostic is required to
avoid two simultaneous model copies on the MI50.

## Initial contract matrix

| ID | Route | Decode execution | Status |
|---|---|---|---|
| A | legacy / batch-major | direct step | Supported by current selector |
| B | legacy / batch-major | queued HIP graph | Supported by current selector |
| C | interactive / layer-major | direct step | Technically selectable; verify exact supported execution |
| D | interactive / layer-major | queued HIP graph | Current no-stream interactive path |

Run only contracts that can start from the same post-P512 semantic state.
Record unsupported combinations and why. Do not silently substitute.

## Required measurements

1. Establish one canonical P512 state and next token. Compare imported state
   snapshots and next-token/logit behavior before timing. Include position,
   token, KV dtype/layout, recurrent layout, convolution history, and all
   active KV entries in the equivalence record. Keep restore/setup outside the
   steady-state timing boundary.
2. Run A–D where supported, with matched P512 state, generated-token count,
   model/build, selectors, warmup, and fixed MI50 clocks at SCLK/MCLK
   `1606/1000 MHz`. Report first-token, steady decode, graph capture, and
   preparation separately; retain raw outputs.
3. Attribute one complete token by semantic family (recurrent projections,
   GDN/state, FFN, attention/QKV/KV/attention O, norms/conversions, LM
   head/argmax, dispatch/synchronization/runtime). Avoid summing overlapping
   spans. Record dispatches, synchronizations, H2D/D2H bytes, allocation and
   VRAM deltas, and available GPU memory counters per route.
4. Audit route selection from process entry through constructor flags and
   kernel launch. Record active/replaced/disabled/unreachable selector status.
5. Verify token and state equivalence across the boundaries 512→513,
   2048→2049, and 8192→8193. Record any capacity or graph-support limits.
6. Sample hardware continuously with `scripts/sample-gpu.sh`. Any material
   SCLK/MCLK deviation from 1606/1000 makes the run diagnostic only; preserve
   it with the telemetry rather than silently discarding it.

Use interleaved A/B order and repeated runs for timing conclusions. Preserve
raw run logs, telemetry, exact command/environment, model hash, compiler,
ROCm, VRAM, temperature, and clocks.

## Baseline and result

No M26-C A/B measurements have been collected yet. No route delta is claimed.
M26-B closed at commit `9180e6f5b41c6ece62d68a2f4c153a4a480429af` after
proving its historical comparison non-comparable; its result is not a current
route baseline.

The route mapping/documentation scaffold was committed separately at
`1b5c0d497c378f4b2bd00d14bbc2fe2a5225d3a4`. The measurement implementation
started from the subsequent clean documentation commit
`7bca95720521be6d8cd17b3961d4b5580eedfa96`; no harness or performance-candidate
changes were included in either starting commit.

## Decision

OPEN — harness feasibility and state equivalence must be established before
timing or selecting any optimization.
