# EXP-0272 — M20-B API authentication and serving hardening

## Contract

Secrets may be supplied through `MIINFER_API_KEY` or `--api-key-file PATH`.
Non-loopback binding requires a configured key unless `--allow-insecure` is
explicitly provided. `/healthz` and `/readyz` remain public; `/v1/models`,
`/v1/chat/completions`, and `/metrics` require Bearer authentication when a
key is configured.
The bundled Web UI is intended for local unauthenticated use; authenticated
external serving should use an OpenAI-compatible client that sends the Bearer
key.

## Verification

On the physical MI50 with `MIINFER_API_KEY` configured:

* `/healthz`: HTTP 200 without credentials.
* `/v1/models`: HTTP 401 without credentials, HTTP 401 with the wrong Bearer,
  HTTP 200 with the correct Bearer.
* `/metrics`: HTTP 401 without credentials.
* Invalid credentials return structured OpenAI-style JSON with
  `authentication_error`.
* Non-loopback startup without a key fails; explicit insecure override remains
  available for controlled testing.

Raw status codes and response bodies are in
`results/m20-auth/20260909-112004/`.

## Decision

KEEP. The implementation uses constant-time key comparison and does not place
raw secrets in benchmark records or command examples.
