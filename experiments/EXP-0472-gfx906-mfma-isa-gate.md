# EXP-0472 — gfx906 MFMA compile feasibility gate

## Question and scope

Can the pinned MIInfer toolchain compile the intended QK matrix instruction for
the exact MI50 target (`gfx906`)? This is a compile-only gate. It does not add
MIInfer dispatch, emit a candidate model kernel, or run any GPU work.

## Frozen compiler and command

- Host: Machinist
- Image: `localhost/miinfer-dev:rocm-7.2.1`
- Image digest: `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`
- Image ID: `sha256:50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Compiler: AMD clang 22.0.0git, ROCm 7.2.1
- Probe SHA-256: `7dc3cd10ae657f25e71960b4020e906358093dc2bbbda916f547997ca0b59611`
- Target: `--offload-arch=gfx906`
- Command inside the image: `hipcc -O3 --offload-arch=gfx906 -c /work/probe.hip -o /out/probe.o`

The source is [`tools/mfma_gfx906_compile_gate.hip`](tools/mfma_gfx906_compile_gate.hip).
It calls `__builtin_amdgcn_mfma_f32_16x16x4f32` and stores all accumulator
lanes so the result cannot be discarded. No forced target feature was used.

## Result

**REJECT — gfx906 target feature gate.** The exact-target compile stops with:

```text
/work/probe.hip:14:11: error: '__builtin_amdgcn_mfma_f32_16x16x4f32' needs target feature mai-insts
1 error generated when compiling for gfx906.
```

No device object was produced, so MFMA ISA and resource metadata (VGPR, SGPR,
spills, LDS) are not applicable. No GPU was accessed. Do not force
`mai-insts`: an object that assumes an unadvertised instruction feature is not
a viable MI50 kernel.

This matches LLVM's target feature definitions: gfx906 maps to the MI50 and
gfx908 is a separate target; LLVM's target table lists `mai-insts` for the
newer matrix-instruction target family, not gfx906. The LLVM target-feature
review also shows gfx906 without and gfx908 with `mai-insts`:

- [LLVM AMDGPU target guide](https://llvm.org/docs/AMDGPUUsage.html)
- [LLVM target-feature review](https://reviews.llvm.org/D138870)

The MFMA Q×K branch stops here. A future attention candidate must use
instructions available on gfx906 and must first demonstrate a useful
dataflow without repeating the rejected scalar Q×K or GQA head-reuse routes.
