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

The final 64K run reached MIInfer’s queued request and `prefill_started` state.
Raw Hermes/server logs are in `results/hermes/20260909-qualification-110554/`.
The request did not complete within the recorded run window, so Hermes
delivery, multi-turn, reset, and Telegram gates are not claimed as passed.

## Decision

PARTIAL / BLOCKED. The endpoint is protocol-compatible enough for Hermes to
construct and submit a real request, but Hermes usability is not qualified on
this prefill curve. Keep ordinary-chat qualification separate from tool-heavy
agent behavior and do not treat a configured 128K value as proof of a 128K
Hermes context.
