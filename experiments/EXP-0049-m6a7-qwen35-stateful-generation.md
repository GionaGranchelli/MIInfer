# EXP-0049 — M6-A7 Qwen3.8-27B stateful generation

## Question

Can the verified recurrent and full-attention layer executors advance their
state correctly when composed across positions 0–64 of the complete
Qwen3.8-27B trunk?

## Hypothesis

Persistent Gated DeltaNet states and per-full-attention-layer KV histories can
be carried through the 64-layer hybrid model while retaining external-reference
final normalization, logits, and greedy-token behavior at selected positions.

## Baseline and candidate

The candidate is the host-only
`miinfer-m6a7-qwen35-stateful-generation` harness. It uses the production
Q4_K_M GGUF and the pinned llama.cpp fixture from M6-A1. It composes the
48 recurrent layers and 16 full-attention layers in their verified pattern.

The state is advanced continuously for positions 0–64. To avoid repeating the
expensive vocabulary projection at every unselected host position, positions
9–15, 17–31, and 33–63 use the pinned reference token as teacher forcing;
greedy tokens are checked at positions 0–8 and checkpoint positions 16, 32,
and 64. All positions still execute the complete stateful 64-layer forward.

## Correctness contract

The harness checks recurrent state inputs, final normalization, and logits
against the external fixture at positions 0, 1, 2, 4, 8, 16, 32, and 64.
It checks embedding equality at the same checkpoints, finite execution, and
greedy-token agreement wherever logits are evaluated. Internal recurrent
checkpoint comparisons use the measured A7 diagnostic envelope; final norm
and logits use the established external envelope. No old MIInfer trajectory
is used as the correctness authority.

## Environment

```text
model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
sha256:    7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
reference: llama.cpp c0bc8591e8815c63cb01dd3f051a8b0df02501c9
fixture:   /tmp/m6a1-qwen38-reference
build:     host Release, MIINFER_ENABLE_HIP=OFF
```

## Command

```bash
build/host-release/miinfer-m6a7-qwen35-stateful-generation \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference
```

## Results

```text
positions executed:       0–64
recurrent layers:         48
full-attention layers:    16
external checkpoints:     0, 1, 2, 4, 8, 16, 32, 64
state/logit failures:     0
greedy-token mismatches:  0
non-finite failures:      0
result:                   PASS
```

Representative external differences:

| Position | Final norm max abs | Logits max abs |
|---------:|-------------------:|---------------:|
| 0        | 0.454920           | 0.470857       |
| 1        | 0.745953           | 0.480025       |
| 4        | 0.524737           | 0.375549       |
| 8        | 0.223267           | 0.363101       |
| 16       | 0.291717           | 0.448721       |
| 32       | 0.167904           | 0.201371       |
| 64       | 0.398890           | 0.494943       |

All values remain within the established external envelope, and the checked
greedy IDs match the fixture.

## Decision

**KEEP / M6-A7 complete for host stateful bring-up.** The recurrent states and
full-attention KV histories survive a 0–64 multi-position replay, with the
complete 64-layer composition remaining externally compatible at all selected
checkpoints.

This is a host correctness milestone, not a GPU or performance result. The
selected positions after the first eight are stateful teacher-forced replay,
so this record does not claim 64 independently sampled generated tokens.

## Follow-up

M6-B0 should establish the same-model MI50 llama.cpp baseline, followed by
M6-B1 MIInfer whole-token profiling and a real GPU stateful-generation path.
