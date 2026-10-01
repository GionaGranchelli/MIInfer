# V2-0048A — Generation budget and exact-prefix fault

## Status

**V2-0048A PASS.** Diagnosis confirmed; minimal correction is qualified on one
candidate. This is not a promotion or publication of a release.
V2-0048 and its packaged artifact remain unchanged. The candidate starts from
`origin/main` at `0032bb59cbf8c3d23ccd0eaeb5970004408503d0`; local `main` remains
untouched at V2-0046 commit `16f18b9f783f8f97c31df2196a3eddac57d76933`.

## Evidence on the preserved V2-0048 package

The package SHA-256 is
`2eb4fc2e8dc6625c0524b8cb4f3c6f2a1ba7f8d7052a3a9e0f8d7b1b32f37efb`.

An authenticated localhost server at 16K reproduced the fault with a minimal
two-request sequence. Request 1 generated four tokens and cached its prompt.
Request 2 used the identical prompt and failed before decode with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. The runtime then reported the
failure at `hipMemcpyAsync` of logits in `src/prefill_v2/model.cpp:891`.

At baseline, prefix matching accepts a prompt equal to the cached prefix. That
sets the unmatched suffix length to zero, so no prefill chunk updates
`last_chunk`; generation nevertheless computes logits from
`d_pong_ + (last_chunk - 1) * kHidden`. The underflowed device pointer explains
the aperture fault. The same failure reproduces with `max_tokens=4`, proving
the 1,024-token budget is not its cause.

Separately, the preserved package returned a 256-token default code response
that ended mid-implementation. An ordinary code prompt with an explicit
1,024-token budget returned a complete implementation and explanation,
`finish_reason=stop`, using 1,002 completion tokens. This supports 1,024 as a
useful default for both chat and the API; it does not claim every task fits
within that budget. The earlier V2-0048 five-turn chat evidence also showed
three turns reaching the 512-token cap with partial or blank visible output.

## Minimal correction

- Use one 1,024-token default for terminal chat and OpenAI-compatible requests.
- Do not take prefix reuse when the cached prefix is the entire prompt. The
  checkpoint does not contain the final hidden activation required for logits;
  full prefill reconstructs it safely.

These findings identify separate causes: an inadequate default generation
budget for some ordinary reasoning/code tasks, and a zero-suffix exact-prefix
reuse bug that can crash the server independently of the generation budget.

## Candidate and qualification

- Corrected source commit: `a2da93f27efe70c9a78116287ea820a72231b61e`.
- Candidate archive:
  `/tmp/miinfer-v2-0048a-build/miinfer-0.2.0-gfx906-Linux.tar.gz`, 958,858
  bytes, SHA-256
  `c98f10dbe8a74bd1cb1d5aa139f840c3f431c1f37617bfb46513456083653807`.
- Installed candidate identity reported version `0.2.0`, commit `a2da93f27efe`,
  `Git dirty: false`, Release, GCC 16.2.1, HIP Clang 20.0.0, gfx906.
- CTest: 25 passed, 0 failed; the package smoke test was skipped in this
  pre-package invocation. The package archive smoke passed both without and
  with the Qwen model argument, including the model-backed first-run checks.
- Default terminal chat at 16K returned a complete C++ implementation and
  edge-case explanation, using 413 generated tokens.
- Authenticated default API code request at 16K returned HTTP 200, 1,826
  visible characters, 446 completion tokens, and `finish_reason=stop`.
- The same prompt was then repeated 19 times with `max_tokens=4`. All 19
  returned HTTP 200 and completed; verbose logs show requests 2–20 completed
  prefill with `reused_prefix_tokens=0`, confirming the exact-prefix fallback.
  The server remained alive through request 20 and shut down cleanly; no HSA
  fault occurred. Empty content on these 4-token stress requests is expected
  and is not counted as an output-quality check.
- The original V2-0048 archive hash remains
  `2eb4fc2e8dc6625c0524b8cb4f3c6f2a1ba7f8d7052a3a9e0f8d7b1b32f37efb`.
  Candidate SHA, old archive SHA, and test summary are also recorded in
  [`focused-qualification.json`](../results/v2-0048a/focused-qualification.json).

V2-0048A passes its scoped release-correction gates. Both main refs are
untouched; no release tag, promotion, or publication was performed.
