# M31-0003 — llama.cpp comparison limits

## Existing measurements are directional, not a controlled A/B

Historical references include multiple llama.cpp revisions and build
configurations: Test A at `73a43d1`, M31-0001 at `9c2e0e4`, the toolbox repo
at `a708a27` with an uninitialized llama gitlink, and an older M6 run at
`c0bc859`. They must not be merged into one reference curve.

The M31 Z840 funnel reports MIInfer/llama.cpp prefill at 8K/32K/64K as
216.971/227.319, 152.100/200.746, and 107.254/173.832 tok/s. The prompts and
conditions were not fully matched, so the ratios are directional. Historical
128K values (about 52.5 vs 136.6 tok/s prefill; about 11.05 vs 24.3 tok/s TG32)
also compare different hosts/configurations and do not identify a kernel root
cause. No new comparison was run here.

## Implementation differences that are safe to state

- llama.cpp has its own scheduler and recurrent/full-attention graph path for
  Qwen3.5; source topology alone does not establish which HIP kernels execute
  or why they are faster.
- MIInfer uses direct persistent KV-pool views and has a separate recurrent
  state/checkpoint design. No like-for-like memory peak was measured across
  the hosts.
- The MIInfer fused gate/up weights occupy 7.13 GB, near the scale of the
  historical weight-residency difference, but are not proven removable.
- EXP-0170's simple gate/up repeat candidate regressed TG64 by 8.37%; do not
  retry without new causal evidence. EXP-V2-0009 retained its fused route.

Portable ideas worth evaluating later include memory-conscious KV ownership,
reducing host round trips in decode where sampling semantics permit, and
batching compatible dispatch. These are hypotheses for MIInfer, not proof that
copying llama.cpp architecture is correct. Do not replace MIInfer with
llama.cpp. Any code adaptation needs license review and attribution; this
investigation made no third-party code adaptation.

## Minimum valid future comparison

Use one host and MI50, identical model bytes/quantization, exact prompt and
output count, comparable initial temperature/power policy, same context/KV
format where possible, and record build/runtime identity. Separate prefill,
decode, peak VRAM and output parity. First compare short guarded runs; do not
use 128K as the first attribution experiment.
