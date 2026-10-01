# V2-0048A — Final artifact release gate

## Decision

**RELEASE BLOCKED. Do not promote this artifact or move the existing `v0.2.0`
tag.** The production default of 1,024 generated tokens still produces
incomplete visible output for ordinary coding requests in terminal chat and
the authenticated OpenAI-compatible API. Build, packaging, installed doctor,
exact-prefix safety, and request stability passed.

## Artifact and build identity

- Canonical source SHA: `2ee2ee3fb29c461617b21d136a093e0e24c7134d`
- Version: `0.2.0` (candidate only)
- Artifact: `miinfer-0.2.0-gfx906-Linux.tar.gz`
- Size: 958,806 bytes
- SHA-256: `2ab21fd56278a19b471b4bc221dc4b1627f91eae97b445f0e849ac140d13935d`
- Build: Release; GCC 16.2.1; HIP Clang 20.0.0; target `gfx906`
- Binary reported the canonical source SHA prefix and `Git dirty: false`.

## Qualification results

- CTest: **26/26 passed**, including all GPU-required tests and the
  package-archive smoke test.
- Model-backed package smoke: **PASS** for
  `Qwen3.8-27B-Q4_K_M.gguf`.
- Installed `miinfer doctor --model`: **PASS** for gfx906/MI50, model, ROCm,
  production profile, and VRAM.
- Authenticated default API coding request: **FAIL quality gate**. The
  `parse_csv_ints` request returned 1,024 completion tokens and 4,067 visible
  characters, ending mid-test (`try { parse`) without closing the code block.
  A second byte-identical request behaved the same. Both HTTP requests
  succeeded and the exact-prompt retry did not fault.
- Default terminal chat coding requests: **FAIL quality gate**. A C++ parser
  response hit 1,024 tokens with inconsistent/incomplete code; a Python LRU
  cache response also hit 1,024 and ended mid-example. A separate parser
  request returned no visible answer after 1,024 tokens. The simpler LRU task
  confirms this is not limited to an unusually elaborate manual prompt.
- Authenticated stability sequence: **20/20 HTTP 200**, each completed at the
  requested four-token limit. Server remained responsive and shut down cleanly.
  Logs show safe full-prefill fallback (`reused_prefix_tokens=0`); the exact
  prompt replay did not reproduce the previous invalid-logits crash.
- Performance matrix was not rerun; the qualified V2-0045 measurements remain
  unchanged.

## Interpretation and next gate

The archive is technically healthy, and the exact-prefix crash fix holds.
However, terminal chat and API generation can consume the full configured
1,024-token budget without completing a visible coding answer. This meets the
release-blocking condition for ordinary coding requests. Keep the merged
source correction on `main`, but do not call this artifact `v0.2.0` release,
retag `v0.2.0`, publish it, or proceed to the post-release roadmap step.

The next work should stay narrowly scoped to generation-budget/visible-output
behavior. Re-run these exact artifact-level coding probes after a correction;
only build and tag a new release once both defaults return complete visible
answers and the package, doctor, prefix-replay, and stability gates still pass.

Structured measurements are in
[`results/v2-0048a/final-artifact-qualification.json`](../results/v2-0048a/final-artifact-qualification.json).
