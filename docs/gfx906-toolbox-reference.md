# gfx906 Toolbox Reference and MIInfer Follow-up

## Status

Roadmap input only. No MIInfer runtime or dependency change has been made.

## External source pin

- Repository: `kyuz0/mi50-gfx906-toolboxes`
- Pin: `a708a2790fa51303e3d4f5af9e53c045177181a3`
- Relevant path: ROCm 7.2.1 `llama.cpp` toolbox
- Discovery date: 2026-10-02

The source pin is intentionally exact. Later refreshes require a new recorded
pin rather than silently following the external repository.

## What the toolbox contributes

The project is an enablement and reproducibility stack rather than an inference
engine competing directly with MIInfer. Relevant mechanisms include:

1. A gfx906-native llama.cpp build with HIP enabled and
   `GGML_HIP_ROCWMMA_FATTN=ON`.
2. Rebuilt ROCm libraries for gfx906, including rocBLAS/Tensile and RCCL, plus
   extracted gfx906 Tensile artifacts for modern ROCm releases.
3. A patched gfx906 vLLM stack using gfx906-specific Triton and FlashAttention
   forks.
4. A vLLM LDS compatibility correction that reduces a V1 prefix-prefill
   `BLOCK_M` from 128 to 64 to stay within gfx906's 64 KiB LDS limit.
5. Container/toolbox packaging that makes the environment substantially easier
   for another MI50 owner to reproduce.

These are external implementation facts, not MIInfer adoption decisions.

## Candidate A — M29.0A external-reference calibration

### Question

Has the strongest practical upstream-class gfx906 baseline moved materially
since V2-0045, and if so, why?

### Contract

Use:

- 1 x AMD Instinct MI50 32 GB, gfx906/Wave64
- 225 W cap
- 1606 MHz SCLK / 1000 MHz MCLK
- exact MIInfer `Qwen3.8-27B-Q4_K_M` model artifact and SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- deterministic equivalent prompt/decode token streams
- V2-0045 comparable points: P512, P1024, P2048, P4096, P8192 and TG128
- at least five valid samples per comparable cell
- clocks, temperature, power, VRAM, process ownership, build/source provenance,
  and raw timing retained

Capture the exact llama.cpp submodule revision resolved by the toolbox pin and
the ROCm/HIP compiler, rocBLAS/Tensile provenance, and build flags.

### Stop / attribution gate

- If equivalent medians stay within 3% of the pinned V2-0045 upstream control
  and no qualitative capability changes, record the refreshed baseline and stop.
- If a cell moves by more than 3% reproducibly, isolate source revision,
  ROCm/Tensile, ROCWMMA attention, compiler/build flags, or another measured
  cause before changing MIInfer.
- Do not infer a MIInfer optimization from an external aggregate result.
- Do not compare the toolbox vLLM 500-request/512-output-token throughput run
  directly with MIInfer's single-stream TG128 model-forward measurement.

## Candidate B — gfx906 reference environment

If Candidate A demonstrates that the toolbox environment is stable and useful,
retain a provenance-pinned external reference environment for future M31
requalification.

The purpose is reproducibility, not framework adoption. MIInfer may use the
environment to build or run comparison implementations and to archive gfx906
userspace components that disappear from newer ROCm distributions.

Rebuilt rocBLAS/Tensile/RCCL artifacts should remain outside the core MIInfer
dependency set unless a controlled experiment demonstrates a direct need or
benefit for MIInfer.

## Candidate C — MIInfer OCI/Podman distribution

After the reference environment is qualified, build a minimal reproducible
MIInfer image/toolbox that:

- pins the required ROCm/gfx906 userspace and compiler/runtime provenance;
- contains the released MIInfer executable and required runtime libraries;
- exposes `miinfer doctor` or equivalent hardware/runtime validation;
- validates gfx906, available VRAM, ROCm/HIP compatibility, and model hash;
- runs a bounded `miinfer serve` health/generation smoke;
- does not pull PyTorch, vLLM, Triton, FlashAttention, or llama.cpp into the
  production execution path unless separately authorized.

This is productization/reproducibility work. It must not change the custom
single-model execution architecture merely to match the toolbox layout.

## Explicit non-goals

This discovery does not authorize:

- replacing MIInfer with llama.cpp or vLLM;
- adding continuous batching to M29;
- adopting Triton as a production dependency;
- porting the external FlashAttention kernel without profiling evidence;
- copying the `BLOCK_M=64` workaround into MIInfer without an exact-shape
  experiment;
- starting a new unbounded kernel optimization campaign;
- changing M29/M30/M31/V3 ordering.

## Roadmap placement

- **M29.0A:** bounded current external-reference calibration.
- **M31:** reuse the refreshed reference as part of final single-MI50 frontier
  qualification.
- **Cross-cutting release/productization:** OCI/Podman reproducibility path once
  the reference environment is understood.

The external project is therefore useful as a control and packaging blueprint,
not as a replacement architecture.
