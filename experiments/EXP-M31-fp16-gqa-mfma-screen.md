# EXP-M31: FP16 GQA QK MFMA screen

**Verdict: rejected before GPU execution.** The ROCm 7.2.1 build linked, but the extracted gfx906 device code contains no `v_mfma_f32_16x16x16f16` instruction. The candidate therefore did not pass the required machine-code dispatch gate. No 2K/8K inference, performance result, or correctness result was produced.

## Hypothesis and mapping

Replace the serial FP16 Q×K dot products in the cached FP16 GQA suffix path with gfx906 MFMA. The existing CTA has eight logical 32-lane rows (`dim3(32,8)`), which pair into four physical Wave64s. Each wave owns one 16-key × 16-query output tile; the four waves cover a 32 × 32 score tile. For each MFMA instruction, lane `l` supplies four A values at `(key=l%16, K=4*floor(l/16)+i)` and four B values at `(K=l%16, query=4*floor(l/16)+i)`, `i=0..3`; the lane receives four FP32 output values at `(key=l%16, query=4*floor(l/16)+i)`. Sixteen K=16 steps cover head dimension 256.

This layout was derived from `ggml/src/ggml-cuda/mma.cuh` and `fattn-mma-f16.cuh` in the gfx906 llama.cpp checkout. Those files are under the ggml authors' MIT license; the prototype reuses the documented lane/layout mapping and contains no copied code. The reference license is at `/tmp/gfx906-llama-cpp-m31/LICENSE` in the local development environment.

## Implementation screened

- Branch: `codex/m31-fp16-mfma`
- Candidate source commit: `dea6c453f557dd151c3a02505f0a717484356eaf`
- Latest evidence/compile-fix commit: `f65394123ed5764f42c6562740977bfc3d93541e`
- Entry: `MIINFER_M31_FP16_QK_MFMA`, default off; ordinary FP16 dispatch remains the control.
- GQA and MFMA route log markers are gated on presence of the candidate variable for future runtime verification.

The target compiled and linked in `localhost/miinfer-dev:rocm-7.2.1`, digest `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`. Clang required `-Xclang -target-feature -Xclang +mai-insts` to accept the builtin but warned this feature is not recognized for gfx906 and ignored it. After extracting `.hip_fatbin` from `qwen3_primitives.hip.o` and unbundling target `hipv4-amdgcn-amd-amdhsa--gfx906`, `llvm-objdump -d -C --mcpu=gfx906` reported zero occurrences of `v_mfma` across the object. This is why compile success alone was not accepted.

## Resource and gate result

Machinist GPU0 was identified as gfx906 BDF `0000:04:00.0`; junction temperature was 23°C at preflight and no inference process was active. GPU2 was not used by this lane. The pinned model and prompt were not loaded, and no GPU inference was started. The 2K exact-logit gate failed at the prerequisite code-generation gate, so the 8K and 64K gates were not run.

## Next action

Do not merge or enable this path. Revisit MFMA only if a supported ROCm/gfx906 toolchain emits the instruction for this kernel and disassembly proves it before any MI50 time is spent. Otherwise continue with an ISA path confirmed available on gfx906 or another prefill bottleneck.
