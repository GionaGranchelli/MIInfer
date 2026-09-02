# EXP-0047 — M6-A5 Qwen3.8-27B four-layer hybrid block

## Question

Can the verified recurrent layers 0–2 and full-attention layer 3 compose into the real Qwen3.8-27B hybrid block with stateful multi-position execution?

## Reference and baseline

The external authority is llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`, using the local Q4_K_M GGUF and the M6-A1 fixture. The fixture includes contiguous checkpoints for positions 0 through 8, including each layer's input/output, recurrent state input, convolution output, and layer-3 full-attention intermediates.

## Candidate

`miinfer-m6a5-qwen35-hybrid-block` is a host-only composition harness. It executes layers 0, 1, and 2 as Gated DeltaNet layers with per-layer convolution history and recurrent state, then executes layer 3 as causal full attention with layer-3 KV history. The layer-3 FFN tail is included. Existing GPU and production paths remain unchanged.

## Correctness gates

For positions 0–8, the harness checks recurrent state continuity, recurrent QKV/convolution/output checkpoints, recurrent layer outputs, full-attention residuals, and final hybrid-block output. It rejects non-finite values and tolerance violations.

## Command

```bash
cmake -S . -B build/host-only -DMIINFER_ENABLE_HIP=OFF -DBUILD_TESTING=ON
cmake --build build/host-only --target miinfer-m6a5-qwen35-hybrid-block -j2
build/host-only/miinfer-m6a5-qwen35-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference
```

## Decision

KEEP — host composition validated at positions 0–8. The harness completed the
three recurrent layers (0–2) followed by the full-attention layer (3) at every
tested position, including recurrent-state continuity and layer-3 KV history.
The checks passed within the documented reference tolerances; the largest
accumulated differences are expected from the host Q4_K_M approximation used
by this bring-up harness. This is not a claim of full-model or GPU support.

## Result

```text
positions checked:       0–8 (9 positions)
recurrent layers:        0, 1, 2
full-attention layer:    3
state continuity:        PASS
attention KV history:    PASS
hybrid outputs:          9/9
non-finite values:       0
```

Command completed successfully with exit status 0. The recurrent QKV,
convolution, recurrent output, residual, and layer-output checkpoints passed;
the full-attention residual and layer-output checkpoints also passed.

## Follow-up

After acceptance, M6-A6 should extend the same concrete layer executors to the complete 64-layer forward and final logits.
