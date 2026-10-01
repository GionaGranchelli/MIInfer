# V2-0048B — Generation completion contract

## Status

Qualification is in progress. Starting canonical source was
`d405f2e353d8fc695f3f1e9b2bd1f866416b2378`; work is isolated on
`rewrite/v2-0048b-generation-completion`. No release tag or promotion has been
performed.

The exact regression prompts and their original surfaces are retained in
[`tests/fixtures/v2-0048b-generation-prompts.json`](../tests/fixtures/v2-0048b-generation-prompts.json).

## Natural-completion measurements before changing the default

The four original cases were allowed to run with a 4,096-token explicit
ceiling. Each stopped naturally below the ceiling; 8,192 was not needed.

| Case | Prompt tokens | Generated | Visible result | Prefill / decode | Outcome |
|---|---:|---:|---|---:|---|
| Terminal C++ parser | not retained | 1,068 | Complete visible answer; source had quality defects | 581.5 / 38,698.6 ms | Natural stop |
| Formerly blank terminal C++ parser | 110 | 2,021 | 897 visible tokens, 3,155 chars; 1,119 reasoning tokens; visible output began at token 1,124 | 638.0 / 100,526 ms | Natural stop |
| Terminal Python LRU cache | not retained | 1,776 | Complete code and executable example | 579.6 / 77,048.7 ms | Natural stop |
| API `parse_csv_ints` | 55 | 1,250 | 4,750 visible chars, closing code fence | 559.4 / 52,048.5 ms | Natural stop |

The longest observed case used 2,021 tokens. A 4,096-token shared default
therefore leaves 2,075 tokens of observed headroom (about 2.03x the longest
measurement). The selected default is a ceiling, not a generation target.

## Current-source behavior checks

All checks below used the in-progress Release build at
`/tmp/miinfer-v2-0048b-build/miinfer` on the qualified Qwen model and MI50.

- `miinfer run`, omitted limit, C++ parser: 56 prompt tokens, 1,428 generated,
  1,425 visible, 5,648 visible characters, `finish_reason=stop`; no reasoning
  tags leaked. Output closed with a code fence.
- `miinfer chat`, omitted limit, former blank-answer prompt: 2,021 generated,
  897 visible, 3,155 visible characters, 1,119 reasoning tokens, visible start
  at token 1,124, `finish_reason=stop`.
- `miinfer chat`, omitted limit, “Say hello briefly”: 6 generated, 2 visible,
  `finish_reason=stop`.
- Authenticated API, omitted limit, C++ parser: 1,250 generated, 4,750 visible
  characters, `finish_reason=stop`; an identical second request returned the
  same complete response. Server logs report full prefill on the exact repeat
  and no HSA fault.
- Authenticated API, omitted limit, “Say hello in one word”: 5 completion
  tokens, `finish_reason=stop`.
- Explicit 32-token API requests stopped at 32 and reported `length` in both
  non-streaming and streaming modes; SSE ended with `[DONE]`.
- Explicit 256-token API requests stopped at 256 and reported `length` in both
  non-streaming and streaming modes; SSE ended with `[DONE]`.
- Explicit 32-token `run` and `chat` requests both emitted
  `[response truncated at output-token limit]` and reported `length`. The chat
  case emitted no visible content because its 32-token budget remained in
  reasoning; it did not silently appear successful.
- Authenticated 16K API prefix/stability check: a prompt calibrated to exactly
  512 input tokens (the checkpoint boundary) was repeated 20 times, each with
  `max_tokens=4`. All 20 returned HTTP 200, exactly 4 completion tokens, and
  `finish_reason=length`; repeated requests followed safe full-prefill fallback
  (`reused_prefix_tokens=0`). `/healthz` and `/readyz` remained healthy, and the
  server shut down cleanly without an HSA fault.

## Implementation

- One shared 4,096-token implicit output ceiling is applied to `run`, `chat`,
  and the OpenAI-compatible API, reduced to remaining context capacity.
- Explicit output limits remain explicit and are rejected if they cannot fit
  the configured context; arithmetic checks avoid context underflow.
- Generation results carry stop-token, output-limit, or cancellation reason.
- API `finish_reason` and final SSE chunks distinguish `stop` from `length`;
  tool-call responses retain `tool_calls`.
- Terminal `run` and `chat` show a truncation notice when the output limit is
  reached while preserving reasoning filtering.

## Pending release gates

## Final canonical artifact gate

**V2-0048B: RELEASE PASS.** The source was merged to `main` at
`94fad71ee19f539ce2ec0c7e100ad97d031dbefa`; the archive and all final checks
below refer to that exact canonical source SHA.

- Version: `0.2.0`
- Artifact: `miinfer-0.2.0-gfx906-Linux.tar.gz`
- Size: 963,353 bytes
- SHA-256: `822fa647cec33efc34689630c0137870b5d61689e77a00b281bdba186a0f4519`
- Build identity: Release, Git dirty `false`, GCC 16.2.1, HIP Clang 20.0.0,
  target `gfx906`; installed binary reports source `94fad71ee19f`.

Exact-artifact gates:

- CTest: **26/26 passed**, including package-archive smoke.
- Package smoke without model: **PASS** through CTest.
- Package smoke with Qwen3.8-27B-Q4_K_M: **PASS**, including installed
  `doctor`, first-run CLI, and authenticated API checks.
- Installed `miinfer doctor --model`: **PASS** for MI50/gfx906, model, ROCm,
  production runtime, and VRAM.
- Default terminal chat, all original terminal fixtures: C++ parser 1,068
  generated / 1,064 visible; formerly blank parser 2,021 / 897 visible; Python
  LRU 1,776 / 1,422 visible. Each visibly completed and ended with `stop`.
- Default API `parse_csv_ints`: two identical requests each returned HTTP 200,
  1,250 completion tokens, 4,750 visible characters, and `finish_reason=stop`.
- Short terminal chat and short API defaults stopped naturally at 6 and 5
  generated tokens respectively (`stop`).
- Explicit `max_tokens=32` and `256`: API non-streaming and streaming all
  stopped at the exact requested count and returned `finish_reason=length`; SSE
  completed with `[DONE]`. Terminal `run` and `chat` at 32 both displayed the
  truncation warning and reported `length`.
- Authenticated 16K exact-prompt stability: **20/20 HTTP 200**, repeated exact
  512-token input, four output tokens per request, expected `length`; server
  health/readiness stayed OK and shutdown was clean. Full-prefill fallback
  (`reused_prefix_tokens=0`) prevented zero-suffix logits use; no HSA fault.

The previous published `v0.2.0` tag pointed to `0da41d1e47b9` and an archive
that failed the completion gate. After these exact-artifact gates passed, the
tag was moved to the canonical source SHA above. No performance, kernel,
prefix-reuse design, or roadmap work was added under V2-0048B.
