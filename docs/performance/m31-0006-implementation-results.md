# M31-0006 — Implementation and promotion decision

## Outcome

`KERNEL_BASELINE_MEASURED / NO_CANDIDATE_PROMOTED`.

The production HIP-Graph decode attention path now has a reproducible bounded
measurement harness, exact-context regression evidence, and one short real
model graph decode result. The tested reduced-split schedules lost. No
production performance code or default scheduling was changed; there is no
measured kernel improvement to promote, and long-context production impact
remains unquantified.

## Evidence-based ranking

1. **Do not reduce active splits.** Split-8 at 8K lost ~1.8x to split-16 across
   three adjacent samples. Split-16 and split-32 at 32K lost ~2.7x and ~1.3x
   in graph replay against split-64. Correctness passed, but performance did
   not.
2. **Investigate repeated KV reads/coalescing next, with counters.** The model
   has 8 GiB of unique 128K K+V state; naive per-query-head reloads could
   represent up to 48 GiB logical reads per generated token. This is a
   source-derived upper bound, not measured DRAM traffic.
3. **Attribute the constant ~159 MB microbenchmark allocation floor.** It is
   independent of context and is not explained by the 1.585 MB static split
   scratch. Identify HIP/runtime/module allocations before adding retained
   graph configurations or more workspace.

The synthetic 128K direct-attention pair is 3.38 ms, with graph replay 3.30 ms.
The short model smoke reported 31.59 tok/s at an 8-token prompt. These are
different contexts and scopes; do not divide or extrapolate them into an
attention share or long-context production speedup.

## Short production integration

On the Z840 MI50, `PrefillV2Model::generate` processed an 8-token deterministic
prompt and generated 4 tokens with HIP Graph enabled. It reported prefill
505.641 ms, 3 decode-forward tokens in 94.9625 ms, or 31.591 tok/s. The
complete guarded process lasted about 52.9 s including model load. Maximum
sampled junction temperature was 47°C, below the 80°C warning and 85°C stop
limits; post-run VRAM/KFD state was clear. This is a smoke, not a repeated
performance qualification, and there was no competing candidate to compare.

## Reproducibility identifiers

- Base checkout HEAD: `54d2fbed6e1b5f8dd8ff7c7293bff5144fe09fc7`; source worktree
  was dirty and preserved.
- Exact build-input archive:
  [source-snapshot.tar](../../results/m31-0006-current-tree-build/source-snapshot.tar),
  SHA-256 `47a49541c13a06f8c3d2e2aa5fa566ebc385a5d7459fc350ed8649e7445ef37d`.
- Attention harness source SHA-256
  `43585a0bf87412e3502c1b7cbc4df64de25338576c13ba8fe1f3ea5b1fa5d2ed`;
  production smoke harness source SHA-256
  `516cb68e016b473ce7a5af6320f1e4f4a7d3aa37635383090e4171d6d2dd755c`.
- Host: Z840; MI50 `gfx906`, PCI BDF `0000:06:00.0`; pinned OCI image
  `localhost/miinfer-dev:rocm-7.2.1`, manifest
  `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`;
  HIP runtime/driver `70253211`.
- Model: `Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Raw measurements and hashes: see
  [graph-decode-baseline](m31-0006-graph-decode-baseline.md) and
  [split-scheduling](m31-0006-split-scheduling.md).

## Next engineering step

Do not reopen temperature/cooling or spend more time on split-count sweeps.
If M31 continues, add instrumentation for per-layer attention time and GPU
memory traffic on a representative production decode, then test a single
KV-reuse/coalescing hypothesis against the unchanged baseline. Require
same-buffer parity and a repeatable component gain before considering an
end-to-end run.
