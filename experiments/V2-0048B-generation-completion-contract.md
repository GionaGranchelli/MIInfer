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
| Terminal C++ parser | 56 | 1,068 | Complete visible answer; source had quality defects | 581.5 / 38,698.6 ms | Natural stop |
| Formerly blank terminal C++ parser | 110 | 2,021 | 897 visible tokens, 3,155 chars; 1,119 reasoning tokens; visible output began at token 1,124 | 638.0 / 100,526 ms | Natural stop |
| Terminal Python LRU cache | recorded in captured run | 1,776 | Complete code and executable example | 579.6 / 77,048.7 ms | Natural stop |
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

- Re-run full CTest after the final source/evidence commit.
- Build a fresh package from the clean committed candidate SHA; record filename,
  SHA-256, and build identity.
- Run package smoke with and without the model, installed `doctor`, the exact
  artifact’s default coding/chat/API checks, and the authenticated 20-request
  16K stability sequence against the installed archive.
- Record the final V2-0048B release decision. Do not promote or tag unless every
  exact-artifact gate passes.
