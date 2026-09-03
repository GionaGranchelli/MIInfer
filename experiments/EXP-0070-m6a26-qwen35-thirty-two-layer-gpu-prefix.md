# EXP-0070 — M6-A26 Qwen3.8 thirty-two-layer stateful GPU prefix

## Status

RETEST — the 32-layer executor composes and runs through P64, but one
recurrent-state checkpoint is just outside the existing external max-absolute
error envelope.

## Question

Can the common qwen35 GPU executor compose layers 0–31, including eight
recurrent/full-attention groups, with persistent state and deterministic reset
replay?

## Candidate

Extended `miinfer-m6a21-qwen35-gpu-hybrid-block` with `--prefix32`. The mode
uses the existing ordered `GpuLayerRef` array and `run_prefix()` executor for
the real layer pattern from L0 through L31. All layer weights, recurrent
states, attention caches, and output buffers are allocated before the
stateful loop. No new kernel or layer-specific executor was introduced.

## Environment and command

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* fixture: `/tmp/m6a1-qwen38-reference`
* reference: pinned llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`
* device: MI50 / gfx906

```bash
cmake --build build/mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference --prefix32
```

## Results

All 32 layer outputs were externally compared at P0, P1, P2, P4, P8, P16,
P32, and P64. Outputs were finite and passed the existing output envelope;
the maximum observed output error was `0.557602`.

The only external state-gate failure was:

```text
position=64 layer=30
max_abs=0.0523846
rms=0.000225921
relative_rms=0.031586
```

The current state gate is `max_abs <= 0.05`, so the strict validation run
stops at this bounded L30/P64 discrepancy. A diagnostic continuation showed
that all output checks still pass, poisoned reset/replay fingerprints are
exact, and no decode-loop allocations occur.

```text
device_bytes_after_setup=7629790720
peak_device_bytes=7629790720
allocations_during_decode=0
poisoned_reset_replay=PASS
```

The `prefix32` mode is not declared a correctness pass because the existing
state envelope was not changed after observing the result.

## Interpretation

The common executor now reaches 32 real layers and eight full-attention KV
caches without a composition or allocation failure. The remaining issue is a
single late recurrent-state numerical outlier, not a reset or determinism
failure. It must be localized or evaluated against a documented external
state-error envelope before A26 can close.

## Decision

**RETEST / implementation retained, production inference unchanged.**

## Follow-up

Investigate L30/P64 state error and report the state error curve/index before
either correcting the numerical path or approving a justified external
reference tolerance. Do not weaken the gate silently.
