# EXP-0456 — Reject FP32 MFMA attention on gfx906

**Status:** REJECT — target instruction is unavailable on MI50/gfx906  
**Control source:** `8b90661ea8ca0d1930749b761f0085efba5067f7`  
**Candidate:** FP32 MFMA score tile, six query heads × sixteen keys  
**GPU:** Machinist GPU0, PCI BDF `0000:04:00.0` (gfx906)

## Hypothesis

Replace serial scalar QK accumulation for a six-query × sixteen-key tile with
`v_mfma_f32_16x16x4f32`, then preserve the existing Q8 scales, online-softmax,
Split-K intervals, and stage-2 combine. The candidate only makes sense if the
target GPU can execute the MFMA instruction.

## Bounded compiler gate

On the pinned ROCm 7.2.1 derived image
`sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`:

- A minimal HIP kernel using `__builtin_amdgcn_mfma_f32_16x16x4f32` fails for
  `--offload-arch=gfx906` with `needs target feature mai-insts`.
- Forcing `-Xclang -target-feature -Xclang +mai-insts` allows compilation but
  prints five warnings that `+mai-insts` is unrecognized and ignored for this
  target.
- Extracting the `hipv4-amdgcn-amd-amdhsa--gfx906` bundle shows the opcode as
  raw words with `llvm-objdump --mcpu=gfx906`; it decodes as
  `v_mfma_f32_16x16x4f32` only when incorrectly disassembled as gfx908.
- This is not an executable gfx906 code path. Compiler acceptance after
  forcing an unsupported feature is not hardware support.

## Decision

Reject before implementing the six-query × sixteen-key kernel. No MI50 kernel
was launched, no GPU runtime was used, and no correctness or performance claim
was produced. Preserve the production direct-global GQA6 path unchanged.

The earlier reference review in
[`EXP-M31-FP16-GQA-Q32-CTA-SCREEN.md`](EXP-M31-FP16-GQA-Q32-CTA-SCREEN.md)
also records that the gfx906 reference fork gates MFMA on CDNA rather than
Vega/gfx906. Reopen only if a verified gfx906 ISA feature/build toolchain
produces an executable MFMA object without forcing an unsupported feature.
