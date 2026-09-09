# EXP-0273 — M20-A constrained Hermes qualification

## Setup

Hermes was run against the local OpenAI-compatible endpoint with an isolated
`HERMES_HOME`, no project rules, no toolsets, `model.max_tokens=128`, and
`model.context_length=65536`. MIInfer served the same Qwen3.8 GGUF with
`--context 65536 --experimental-context`.

## Results

Hermes first exposed two real integration constraints:

1. Without `model.max_tokens`, Hermes refused the custom provider before
   making inference requests because its provider output cap was unknown.
2. Hermes requires at least 64K context, so a 16K MIInfer profile was rejected
   before inference even with an explicit output cap.

The final 64K run completed two ordinary-chat requests before the controlled
stop. The first request sent 7,961 prompt tokens and generated 121 tokens:
PP 28.647 tok/s, TTFT 277,906 ms, TG 24.3172 tok/s. The second sent 8,049
prompt tokens and generated 188 tokens. A third request was cancelled during
prefill by the controlled stop. Raw Hermes/server logs are in
`results/hermes/20260909-qualification-110554/`.

This proves constrained Hermes submission and cancellation at a 64K-configured
endpoint, but not Telegram delivery, restart/reset, or a clean multi-turn
qualification. Hermes’ actual requests were approximately 8K tokens, not
128K; configuration capacity is not evidence of prompt length.

## Decision

PARTIAL. The endpoint accepts constrained Hermes traffic and exposes the real
latency/cancellation behavior. Production Hermes qualification remains open
until the long-context performance and external delivery gates are completed.
