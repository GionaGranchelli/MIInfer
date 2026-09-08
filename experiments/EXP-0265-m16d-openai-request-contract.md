# EXP-0265 — M16-D OpenAI request contract qualification

## Hypothesis

A proper JSON parser and a host-testable ChatML builder can replace ad-hoc HTTP
string search without changing the qualified serving queue or GPU runtime.

## Candidate

`miinfer_openai_api` uses nlohmann/json to validate `messages`, optional
`model`, `stream`, and `max_tokens`, then constructs ordered ChatML.

## Correctness

The host contract test passes user, system, and multi-turn messages; escaped
quotes, backslashes, newlines, UTF-8, unknown fields, clamping, malformed JSON,
wrong types, missing fields, empty messages, and unsupported roles.

## Decision

KEEP. M16-D is complete. M16-C may measure concurrent clients against the
frozen single-worker runtime; do not add continuous batching first.
