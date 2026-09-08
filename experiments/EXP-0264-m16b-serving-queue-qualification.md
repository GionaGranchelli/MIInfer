# EXP-0264 — M16-B bounded serving queue qualification

## Hypothesis

One bounded parsed-request queue and one GPU worker provide safe, responsive
MI50 serving without concurrent runtime mutation.

## Candidate

`cade649` with `scripts/test-serve.sh` on the Qwen3.8-27B Q4_K_M model.

## Correctness

The qualified-machine gate passed:

* malformed chat request: HTTP 400;
* slow incomplete request: HTTP 408;
* eight pending requests plus one active request: accepted;
* overflow: HTTP 503 and queue rejection metric;
* `/healthz` and `/metrics` during inference: HTTP 200;
* SIGTERM: queued requests receive HTTP 503 and the server exits cleanly.

## Decision

KEEP. M16-B is complete. M16-D owns proper JSON/OpenAI message parsing; do not
add concurrent GPU execution before its request contract is qualified.
