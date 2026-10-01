# V2-0048A — Generation budget and exact-prefix fault

## Status

Diagnosis confirmed; minimal correction implemented, qualification pending.
V2-0048 and its packaged artifact remain unchanged. `main` remains at
`0032bb59cbf8c3d23ccd0eaeb5970004408503d0`.

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

## Qualification still required

- Build one candidate from the corrected source commit and preserve its
  identity.
- Verify default API and terminal-chat responses are visibly complete on the
  tested ordinary code task.
- Re-run the identical-prefix request and a bounded authenticated request
  stability sequence on that package at 16K.
- Do not promote or publish unless those gates pass.
