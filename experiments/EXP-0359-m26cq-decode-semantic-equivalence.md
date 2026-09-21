# EXP-0359 — M26-CQ decode semantic equivalence

**Status:** COMPLETE — `ROUTES_NOT_COMPARABLE`  
**Date:** 2026-09-21  
**Scope:** Correctness only; no timing qualification or route changes.

## Question

When legacy/batch-major and interactive/layer-major decode consume the exact
same token at each position from the same P512 state, do their state and output
differences satisfy an already-established MIInfer correctness contract?

## Method and environment

The common state is the varied-text P512 checkpoint from EXP-0358, at semantic
position 512 with pending token `2193`. Teacher inputs were taken from the
qualified legacy/current route trajectory. The 68-token sequence contains the
pending P512 token followed by the next 67 legacy predictions; both routes read
the same file and `step()` receives those tokens at positions 512–579. The
interactive route never feeds its own prediction back into the sequence.

Both runs used Release executable SHA256
`8655d6ca24426a0d8f3c102e285b400293eedd1da84a9d9e8ef2506c032bbe05`, model
SHA256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`,
AMD Instinct MI50 / gfx906, HIP `7.1.52802-9999`, ROCm Clang `20.0.0.rocm`,
and CMake `4.3.0`. The default run had `MIINFER_PRESET` unset; the interactive
run used `MIINFER_PRESET=m25_interactive`. Both selected direct decode. Import,
restore, model setup, and all diagnostic copies are outside any performance
qualification; printed durations are instrumentation overhead and are not
results.

Teacher input file SHA256:
`2bb7690ceb3806c4254c8c8b676bd64ed3248b4186ccd322f8763a7b0267b580`.
The exact token file, route logs, 12 complete state snapshots per route,
per-position logits/final norms, layer-boundary captures, and comparison logs
are retained in
`/home/fedora-workstation/Development/mi50-artifacts/exp0358-current-route/20260921/m26cq/`.

## Structural contract

| Property | Observation |
|---|---|
| Input token and decode position | Identical: both consume the same 68 IDs at absolute positions 512–579. Processed-history fields match exactly at every saved checkpoint. |
| RoPE and cache position | `step()` passes the same absolute position through `run_prefix()`; attention writes K/V at that position and evaluates the causal prefix through `position + 1`. |
| Active KV length/indexing | Exact snapshot shapes and active lengths match at every checkpoint; each active K/V prefix has `position` entries per head. |
| Recurrent state layout | Both snapshots identify `gdn-state-transposed`; recurrent dimensions and logical serialized ordering match. |
| Convolution history | Element count and serialized logical ordering match. |
| Transition ordering | Both execute the same per-position `step()`/`run_prefix()` ordering, including recurrent update and KV append before next-position consumption. |

No structural contract mismatch was observed. The routes select different
projection/FFN implementations and therefore produce different numeric state.

## Numerical and output results

The existing M9 observable gate is 64/64 teacher-forced argmax matches plus
logits cosine `>= 0.9995` against its pinned host oracle for its qualified
position-64 fixture. It does not define pairwise route tolerances for recurrent
state, convolution history, KV, final norm, or logits at a P512 frontier. The
M9 autoregressive replay requirement is within-route deterministic replay. No
accepted cross-route internal-state tolerance exists, and no tolerance is
introduced here.

State metrics below are pairwise route comparisons of canonical FP32 snapshot
fields. Full max-absolute, RMS, relative-L2, finite-status, per-layer and
per-head output is in `checkpoint-comparisons.log`. Final-normalized hidden and
logit-vector metrics are pairwise route comparisons at the listed input
position. Every compared field is finite.

| Input position | Recurrent state rel-L2 | Conv history rel-L2 | Active K rel-L2 | Active V rel-L2 | Final-norm cosine | Logit cosine | Top-1 equal |
|---:|---:|---:|---:|---:|---:|---:|---|
| 512 | 0.000825121 | 0.0105708 | 0.000933337 | 0.000813528 | 0.999744157 | 0.999773389 | yes |
| 519 | 0.00255598 | 0.0212775 | 0.00267569 | 0.00242221 | 0.999739826 | 0.999894872 | yes |
| 527 | 0.00283767 | 0.0226476 | 0.00387625 | 0.00352193 | 0.999408719 | 0.999792311 | yes |
| 543 | 0.00357926 | 0.0235375 | 0.00625528 | 0.00623007 | 0.999735004 | 0.999758134 | yes |
| 559 | 0.00536155 | 0.0447915 | 0.00888467 | 0.00943350 | 0.999301920 | 0.999853548 | yes |
| 564 | 0.00486901 | 0.0250457 | 0.00917543 | 0.00965281 | 0.999558013 | 0.999919451 | yes |
| 565 | 0.00484720 | 0.0261308 | 0.00924880 | 0.00969223 | 0.999810673 | 0.999976327 | yes |
| 566 | 0.00477305 | 0.0256299 | 0.00928250 | 0.00971833 | 0.999682893 | 0.999961342 | yes |
| 567 | 0.00493284 | 0.0273702 | 0.00934953 | 0.00979271 | 0.999309854 | 0.999910492 | **no** |
| 568 | 0.00487142 | 0.0251713 | 0.00937349 | 0.00981049 | 0.999697618 | 0.999830479 | yes |
| 569 | 0.00482981 | 0.0248212 | 0.00943780 | 0.00985358 | 0.999309734 | 0.999871176 | yes |
| 575 | 0.00541461 | 0.0305191 | 0.0101236 | 0.0107165 | 0.994520366 | 0.999083600 | yes |

At absolute input position 567, the shared teacher-forced input is token `13`.
Legacy ranks `16` then `15`, with logits `19.2184944` and `19.2085495` and a
top-1/top-2 margin of `0.00994491577`. Interactive ranks `15` then `16`, with
logits `19.2626877` and `19.2543755` and margin `0.00831222534`. Thus the
greedy flip is a near tie. The logit-vector max absolute error is `0.253678799`,
RMS is `0.0403499532`, and relative L2 is `0.0134012579` at that position.
At position 566, both rank token `13` first (margin `8.65`).

Final-normalized hidden cosine declines to `0.994520366` by processed position
576, and pairwise logits cosine there is `0.999083600`. Across the same shared
inputs, full recurrent-state relative L2 grows from `0.000825` at position 512
to `0.005415` at 575; convolution-history relative L2 goes from `0.01057` to
`0.03052`, and K/V relative L2 from `0.00093`/`0.00081` to
`0.01012`/`0.01072`. This is accumulated drift, not a one-step token-only
ordering difference. The selected per-layer hidden and recurrent operator
comparisons show that divergence predates the greedy flip and grows through
later layers and positions; token identity alone does not establish state
equivalence.

## Layer/operator localization

At the first consumed decode input (absolute position 512), layer-0 input and
normalized input are exact. The first recurrent projection differences are
already present: QKV max absolute error `0.00115203857` (RMS
`0.000219307162`) and gate-projection max absolute error `0.0243429542` (RMS
`0.00787791412`). Layer-0 recurrent-output max error is `4.91142273e-5`; its
layer output differs by max `0.0736160278`. The recurrent-state snapshot shows
layer-0 state max error `0.00059223175` and convolution-history max error
`0.00115203857` after that first input.

At the position-567 prediction flip, layer-0 input and normalized input remain
exact under teacher forcing. Layer-0 QKV differs by max `0.00170898438`, gate
projection by `0.146288514`, recurrent output by `0.000111103058`, and layer
output by `0.0424194336`. The hidden-input/output trace covers every layer at
all 12 checkpoints; detailed recurrent QKV/gate/stage captures accompany it.
The first stage mismatch is route-selected projection arithmetic, not cache
restore or a changed recurrent/KV layout. In the existing source trace,
legacy uses native Q4 wave GEMV and interactive uses resident Mx Q8 MMQ for
recurrent projections, followed by other route-specific SSM-output and FFN
paths; see EXP-0358's selector audit.

## Commands

Prepare the teacher-forced IDs from the canonical P512 state and the legacy
TG128 state:

```bash
rtk python3 scripts/prepare-m26cq-teacher-inputs.py \
  /home/fedora-workstation/Development/mi50-artifacts/exp0358-current-route/20260920/p512-varied.state \
  /home/fedora-workstation/Development/mi50-artifacts/exp0358-current-route/20260920/legacy-current-tg128.state \
  68 /home/fedora-workstation/Development/mi50-artifacts/exp0358-current-route/20260921/m26cq/teacher-inputs-68.txt
```

Build the diagnostic CLI:

```bash
rtk cmake --build --preset mi50-release --target miinfer -j2
```

Run both routes with the same teacher-input file (legacy leaves
`MIINFER_PRESET` unset; interactive sets `MIINFER_PRESET=m25_interactive`):

```bash
build/mi50-release/miinfer run /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --context 1024 --max-tokens 68 \
  --m26c-import-state /home/fedora-workstation/Development/mi50-artifacts/exp0358-current-route/20260920/p512-varied.state \
  --m26c-decode-route direct \
  --m26c-output-state /path/to/route-final.state \
  --m26c-logits-output /path/to/route-logits \
  --m26c-teacher-forced-inputs /home/fedora-workstation/Development/mi50-artifacts/exp0358-current-route/20260921/m26cq/teacher-inputs-68.txt \
  --m26c-checkpoints-dir /path/to/route-checkpoints \
  --m26c-layer-path-prefix /path/to/route-layers \
  --m26c-layer-path-position 567
```

Compare outputs:

```bash
rtk python3 scripts/compare-m26cq-checkpoints.py LEGACY_CHECKPOINT_DIR INTERACTIVE_CHECKPOINT_DIR CHECKPOINT_LOG
rtk python3 scripts/compare-m26cq-logits.py LEGACY_LOGIT_PREFIX INTERACTIVE_LOGIT_PREFIX
rtk python3 scripts/compare-m26cq-layer-path.py LEGACY_LAYER_PREFIX INTERACTIVE_LAYER_PREFIX all
```

## Decision

`ROUTES_NOT_COMPARABLE`

The structural contracts match, but the current evidence has no accepted
cross-route numerical contract for the differing recurrent/convolution/KV
state or P512 outputs. Teacher forcing shows a small-margin greedy flip on top
of route-state drift that is already present at the first decode input and grows
under shared inputs. The M9 oracle gate is not a valid pairwise tolerance for
these P512 route states. No correctness bug is established, and no loose
tolerance is adopted to qualify the comparison.

**Next action:** close the EXP-0358 route-differential question, keep the
qualified no-preset route canonical, and plan subsequent M26-E/M26-F work using
that route. Do not repair the experimental interactive route solely to create
an A/B timing comparison.
