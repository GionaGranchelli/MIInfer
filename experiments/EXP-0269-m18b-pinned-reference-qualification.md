# EXP-0269 — M18-B pinned llama.cpp-gfx906 qualification

## Hypothesis

The pinned gfx906 llama.cpp checkout can provide the primary runtime-only PP
and TG reference for the exact MIInfer Qwen3.8-27B-Q4_K_M model.

## Baseline

Checkout: `/home/fedora-workstation/Development/iacopPBK-llama.cpp-gfx906`

Pinned commit:
`125db33d6d352b5c65357eaf37e8f7ae2fe6fbd8`

Requested configuration: full GPU offload, Flash Attention, Q8_0 K cache,
F16 V cache, `-b 2048`, `-ub 2048`. The checkout was rebuilt with ROCm 7.1,
Clang 20.0.0, `GGML_BACKEND_DL=ON`, and `GGML_NATIVE=OFF`.

## Model and hardware

* Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* GPU: AMD Instinct MI50, gfx906, SCLK 1606 MHz, MCLK 1000 MHz

## Verification

The exact runtime-only command was:

```bash
env HSA_OVERRIDE_GFX_VERSION=9.0.6 HIP_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/fedora-workstation/Development/iacopPBK-llama.cpp-gfx906/build-mi50/bin:/usr/lib64/rocm/lib:/usr/lib64 \
  /home/fedora-workstation/Development/iacopPBK-llama.cpp-gfx906/build-mi50/bin/llama-bench \
  -m /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf -ngl 99 -fa 1 \
  -ctk q8_0 -ctv f16 -b 2048 -ub 2048 -p 8 -n 64 -r 1 -o json
```

The pinned binary loaded its gfx906 backend but failed model loading. The
pinned source contains no `qwen35`/`Qwen3.8` implementation, so this is a
model/commit compatibility blocker, not an MIInfer timing result. No
side-by-side PP/TG claim is made from an incompatible model.

## Decision

BLOCKED FOR COMPARISON at this exact pin. Preserve the pin and model as the
requested reference contract; qualify a comparison only after an approved
reference commit supports the same GGUF. MIInfer’s runtime-only measurements
are retained separately in `results/m18-runtime/`.
