# EXP-0051 — M6-B1 Qwen3.8-27B MIInfer GPU profile readiness

**Status:** MEASUREMENT-ONLY — prerequisite not met  
**Milestone:** M6  
**Date:** 2026-09-02  
**Baseline commit:** `aea0a63e666c4bdb64e7abafdd8c9cd3fc614ee6`  
**Candidate commit:** documentation only

## 1. Question

Can the current MIInfer production GPU path execute the selected Qwen3.8-27B
Q4_K_M artifact so that M6-B1 can measure whole-token GPU performance?

## 2. Hypothesis

M6-A7 completed a host-only stateful path. The existing HIP runtime is the
earlier Qwen3-8B executor and will reject the qwen35 artifact before GPU work.

## 3. Artifact

```text
/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
size: 17,106,775,008 bytes (15.93 GiB)
sha256: 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
architecture: qwen35
```

## 4. Checks

The read-only M6-A2 audit loads the real GGUF and reports 866 tensors,
including Q4_K, Q5_K, Q6_K, Q8_0, and F32 contracts.

The current model loader rejects the artifact at
`src/qwen3_model.cpp:83-85`:

```text
MIInfer model inspection failed:
UNSUPPORTED MODEL CONFIGURATION: general.architecture is not qwen3
```

The current GPU plan is the prior-model plan. It hard-codes old Qwen3
workspace dimensions (`q_output_fp16=4096`, `k_output_fp16=1024`,
`ffn_intermediate_fp16=12288`) in `src/model_plan.cpp:142-148` and exposes
only `execute_qwen3_*` GPU entry points.

The M6-A3 through M6-A7 qwen35 tools link only `miinfer_model`; they are host
reference/bring-up executables and do not provide a HIP execution path.

## 5. Results

| Check | Result |
|---|---|
| Qwen3.8 GGUF discovery and hash | PASS |
| M6-A2 tensor/metadata audit | PASS |
| Existing host CTest | 8/8 PASS |
| Existing MIInfer model inspection, `--no-gpu` | Explicit qwen35 rejection |
| MIInfer qwen35 GPU forward | Not executable; no implementation |
| MIInfer qwen35 GPU profile | Not measurable |

No MIInfer throughput, GPU-event, dispatch, synchronization, allocation, or
VRAM result is reported. Running a host path as a GPU baseline would be
misleading.

## 6. Interpretation

M6-B0 is a valid external baseline, but M6-B1 has not started as a performance
measurement. The missing prerequisite is a qwen35 MI50 execution path covering
the selected model's Q4_K/Q5_K/Q6_K projections, hybrid layer pattern, Gated
DeltaNet state, full-attention KV cache, and final Q6_K output head.

The existing Qwen3-8B HIP path is not a safe fallback: its loader, tensor
contracts, dimensions, and layer executor target a different architecture.

## 7. Decision

**DEFER M6-B1.** No MIInfer performance claim is made. Preserve M6-B0 as the
external control and continue with the smallest GPU bring-up slice.

## 8. Next task

### M6-A8 — qwen35 MI50 GPU execution foundation

Add a qwen35-specific model/configuration and GPU-plan boundary without
changing the old qwen3 production path. Reuse only validated GGUF parsing,
device validation, arena ownership, and existing primitives where their tensor
contracts fit. Start with one GPU projection/operation fixture from the pinned
M6-A1 reference, then promote to one GPU recurrent or full-attention layer.

M6-B1 can resume only after a real qwen35 GPU forward exists and passes the
external correctness gates.
