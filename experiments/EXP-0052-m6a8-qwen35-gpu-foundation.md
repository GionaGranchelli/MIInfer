# EXP-0052 — M6-A8 Qwen3.8-27B GPU foundation

**Status:** KEEP — foundation slice complete  
**Milestone:** M6-A8  
**Date:** 2026-09-02  
**Baseline commit:** `ced6c0e`  
**Candidate:** qwen35 model boundary plus layer-0 GPU RMSNorm fixture

## 1. Question

Can MIInfer load the real qwen35 artifact through a dedicated model boundary
and execute one real Qwen3.8 operation on gfx906 against the pinned fixture?

## 2. Hypothesis

A small qwen35-specific loader/configuration boundary can reuse GGUF parsing,
gfx906 validation, and the existing generic F32 RMSNorm primitive without
changing the old Qwen3-8B production path.

## 3. Change

Added `Qwen35Model` metadata/tensor access, strict validation for the selected
configuration, and `miinfer-m6a8-qwen35-gpu-foundation`. The tool executes the
real layer-0 `blk.0.attn_norm.weight` GPU RMSNorm fixture.

No Q4_K projection, DeltaNet, full-attention, or production decode path was
added in this slice.

## 4. Artifact and fixture

```text
model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
sha256: 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
fixture: /tmp/m6a1-qwen38-reference
```

## 5. Result

```text
model=Qwen3.8-27B
architecture=qwen35
device=AMD Instinct MI60 / MI50 (gfx906:sramecc+:xnack-)
hidden=5120
main_layers=64
rms_norm=blk.0.attn_norm.weight
max_abs_error=0
M6-A8 qwen35 GPU foundation RMSNorm PASS
```

## 6. Checks

* M6-A2 compatibility audit: PASS;
* host build: PASS;
* real MI50 fixture execution: PASS;
* old MI50 CTest: 19/19 PASS;
* host CTest: 8/8 PASS;
* `git diff --check`: PASS.

## 7. Decision

**KEEP.** This validates the qwen35 model boundary and one real GPU operation.
It does not constitute Qwen3.8 inference support or a performance benchmark.

## 8. Next task

Implement and externally validate the first qwen35 GPU projection/operation
whose tensor contract is not covered by the old Q4_0 path, then promote to a
single GPU recurrent or full-attention layer. Resume M6-B1 only after a real
qwen35 multi-layer GPU forward exists.
