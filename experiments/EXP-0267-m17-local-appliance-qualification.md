# EXP-0267 — M17 Local Appliance Qualification

## Hypothesis

A supported MI50 host can use MIInfer from the packaged artifact without a
source checkout or compilation.

## Candidate

The release archive ships `install.sh`, `miinfer doctor`, GGUF discovery,
`miinfer serve --model`, and a small static Web UI at `/`.

## Qualification

```bash
scripts/test-package.sh build/mi50-release/miinfer-0.2.0-gfx906-Linux.tar.gz
scripts/test-serve.sh /tmp/miinfer-install/bin/miinfer /path/to/Qwen3.8-27B-Q4_K_M.gguf
```

The package gate extracts and installs the artifact into a fresh directory,
then checks the installed CLI. The serving gate checks the Web UI route and
uses the public OpenAI-compatible endpoint under the qualified bounded queue.

## Decision

PASS on the qualified MI50 with Qwen3.8-27B-Q4_K_M:

* `23/23` CTest passed, including the GPU suite.
* CPack produced the `0.2.0` gfx906 archive.
* `scripts/test-package.sh` passed after extraction and a fresh install.
* `miinfer doctor --model` passed GPU, HIP, model, VRAM, and port checks.
* `scripts/test-serve.sh` passed using the freshly installed binary; it checks
  the Web UI route plus the existing queue, control-plane, and shutdown gate.

Decision: KEEP. M17 is complete without changing the GPU/runtime path.
