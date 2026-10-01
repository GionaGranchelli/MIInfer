# MIInfer v0.2.0

MIInfer v0.2.0 is a Linux inference runtime specialized for a single AMD
Instinct MI50 (`gfx906`). It provides production support for
`Qwen3.8-27B-Q4_K_M`, an OpenAI-compatible API, terminal `run` and `chat`, exact
prefix reuse, and a bundled local Web UI.

The qualified single-MI50 prefill/decode matrix beat both pinned mx-llama.cpp
and upstream llama.cpp in all nine measured cells. The detailed workloads and
the forced-token decode methodology are recorded in
[`release.md`](release.md); decode results are model-forward performance, not
HTTP-generation throughput.

## Linux / gfx906 artifact

- Archive: `miinfer-0.2.0-gfx906-Linux.tar.gz`
- SHA-256: `822fa647cec33efc34689630c0137870b5d61689e77a00b281bdba186a0f4519`
- Canonical source: `94fad71ee19f539ce2ec0c7e100ad97d031dbefa`
- Build: Release, GCC 16.2.1, HIP Clang 20.0.0, target `gfx906`

The archive contains runtime executables, an installer, license, and user
documentation. It does not bundle ROCm or model weights. See
[`release.md`](release.md) for installation instructions.
