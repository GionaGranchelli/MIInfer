# M29-0001 — gfx906 external-reference calibration

Decision: `MATERIAL_MOVEMENT_UNRESOLVED`

The requested equivalent timing matrix could not be produced because the exact qualified model is not loadable by the llama.cpp binary resolved by the exact toolbox pin. No substitute model or newer source revision was used.

## Required identity and provenance

- Host: Machinist X99, one MI50 gfx906/Wave64
- MIInfer branch: `m29/gfx906-reference-calibration`
- MIInfer HEAD: `e61c2488f790cc83029c16a35c1312701ddc3f04`
- Toolbox parent pin: `a708a2790fa51303e3d4f5af9e53c045177181a3`
- Resolved llama.cpp submodule: `52f1096f21b0e12e4532f34224991e06cf21a69a`
- Image ID: `ec7af7dcb79974488d687d5c0b2f544a947131fe6c6ece20c7a3b2eee74fc442`
- Image digest: `sha256:396ac51f9f386d7144ca6242b7ef235f9e50f86142f84e2a9f5de4f3adb5e4f4`
- ROCm base: 7.2.1
- HIP: `7.2.53211-e1a6bc5663`
- AMD clang reported by `hipcc`: `22.0.0git`, ROCm 7.2.1 toolchain
- Exact model: `Qwen3.8-27B-Q4_K_M.gguf`
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`

The toolbox source tree had pre-existing modifications in `llama.cpp/build-toolbox.llama.sh` and `llama.cpp/get_models.py`; they were preserved and not changed by this calibration attempt.

## Reproduction and failure

The benchmark was invoked with the required gfx906 visibility and V2-style parameters, omitting the invalid `-dev 0` syntax rejected by this build:

```text
ROCR_VISIBLE_DEVICES=1 llama-bench \
  -m /home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf \
  -r 5 -b 2048 -ub 512 -ngl 999 -t 24 \
  -ctk f16 -ctv f16 -fa 1 -p 512 -n 0 -o jsonl
```

The container correctly exposed exactly one gfx906 device (`ROCm0`, 32,752 MiB), but model loading failed before timing:

```text
llama_model_load: error loading model: missing tensor 'blk.64.ssm_conv1d.weight'
llama_model_load_from_file_impl: failed to load model
common_init_from_params: failed to load model '/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf'
Failed to load the model
```

The GGUF header identifies the model family with `qwen35.*` metadata and contains `ssm_conv1d` tensors. The pinned llama.cpp revision predates later Qwen3.5-related changes that are not present in `52f1096`; this is the measured attribution for the capability mismatch. The model hash was independently verified before classification.

Therefore P512/P1024/P2048/P4096/P8192 and TG128 have no valid samples, medians, or movement percentages. The decision is unresolved capability/provenance, not a performance movement claim.

## Raw evidence

- `raw/host-environment-before.json` — SHA-256 `71a0f6b7adaa48b7c5e64b57717b77caeb8d081c36b09d0ac1b11b1ba0498cac`
- `raw/provenance-rerun.txt` — SHA-256 `a4fe8aea88da792c05b0c44ffe69959356893f613ca2288f4653d9b0e0f5d487`
- `raw/toolbox-build-provenance.txt` — SHA-256 `9a6c87d9f9b8ac8fb2172e9f654f5c4b7e5493ca0393b6a5057fefef025bd2ee`
- `raw/exact-model-load-failure.log` — SHA-256 `1ec9c2ef72b106f6c867e1b85970644d63e4f939c7b2b3ef93b8016d59f19893`
- `raw/p512-model-load-failure.stderr` — SHA-256 `133f17b36dbfc6828856bc5bf27a6b4d13bd10d3cd00163f42c1c36179af3361`
- `raw/p512-model-load-failure.jsonl` — empty stdout, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `raw/model-metadata.txt` — SHA-256 `82b94a4cf54e20f1e1bd3bd5d0149b607c94942a7ecfacc49748d79f4a9b19cc`
- `raw/p512-prefill.stderr` — original invalid-device diagnostic, SHA-256 `71c084a99437ec6e715606836d2bc8f7bc14bf5acb572eb717ce252cd9d8ca4b`
- `raw/p512-prefill.jsonl` — original invalid-device/help output, SHA-256 `25b97076ac6f9309b0f243244cfc1f53207d8421fdb2f2d3195382f02c2a8135`

No llama.cpp or MIInfer production code was changed. A later requalification requires either a newly recorded toolbox/source pin that loads this exact model or a separately authorized model artifact; neither is introduced here.
