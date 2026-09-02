# EXP-0048 — M6-A6 Qwen3.8-27B full 64-layer forward

## Question

Can the concrete recurrent and full-attention executors compose across the
complete 64-layer Qwen3.8-27B trunk and produce externally compatible final
normalization and logits for one token?

## Baseline and candidate

The candidate is the host-only `miinfer-m6a6-qwen35-full-forward` harness. It
uses the selected Q4_K_M GGUF, the verified layer-0/3 composition, and the
actual layer pattern: recurrent layers `0,1,2,4,5,6,...,60,61,62` with full
attention at `3,7,11,...,59,63`. It then applies `output_norm.weight` and
`output.weight` and checks the external position-zero fixture.

The source GGUF and reference are unchanged from M6-A1:

```text
model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
sha256:    7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
reference: llama.cpp c0bc8591e8815c63cb01dd3f051a8b0df02501c9
fixture:   /tmp/m6a1-qwen38-reference
```

## Correctness contract

The real Q4_K_M embedding is compared with the captured reference embedding.
Every layer is executed in order and checked for finite values and checkpoint
agreement. Final norm and logits are compared with the external F32 fixture;
the first external greedy token must also match. This host bring-up uses the
existing Q8 activation/dequantized-weight approximation, so accumulated layer
checkpoints use a wider diagnostic envelope than the isolated A3/A4 checks.
The final norm/logit envelope and external top-1 are the decisive full-forward
checks; this is not yet a GPU or multi-token acceptance.

## Command

```bash
cmake -S . -B build/host-release -DMIINFER_ENABLE_HIP=OFF \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/host-release \
  --target miinfer-m6a6-qwen35-full-forward -j2
build/host-release/miinfer-m6a6-qwen35-full-forward \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference
```

## Result

```text
64/64 layer executions: PASS
recurrent layers:       48
full-attention layers:  16
embedding:               max_abs 0
final norm:              max_abs 0.454920, rmse 0.0791744
logits:                  max_abs 0.470857, rmse 0.0863484
external argmax:         11 == 11
non-finite values:       0
```

## Decision

**KEEP / M6-A6 complete for host one-token bring-up.** The complete main trunk
and final vocabulary projection execute successfully against the pinned
external fixture. Production GPU execution and stateful multi-token generation
remain outside this milestone.

## Follow-up

M6-A7 should add stateful multi-position generation for the complete 64-layer
model, validate recurrent and full-attention state evolution, and compare
selected logits and greedy tokens at positions 0, 1, 2, 4, 8, 16, 32, and 64.
