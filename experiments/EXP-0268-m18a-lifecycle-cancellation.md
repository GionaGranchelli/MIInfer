# EXP-0268 — M18-A lifecycle, cancellation, and request telemetry

## Hypothesis

The single-worker server can regain deterministic lifecycle control with a
self-pipe wakeup, bounded queue, host-boundary cancellation, and structured
request telemetry without interrupting a HIP kernel.

## Candidate

`miinfer serve` now exposes `--context`, `--experimental-context`,
`--api-key-file`, and `--allow-insecure`. SIGINT/SIGTERM wakes `poll()`
through a self-pipe; the worker checks shutdown/client cancellation between
token, chunk, layer, and decode dispatch boundaries.

## Environment

* AMD Instinct MI50/gfx906, Wave64, 32 GB HBM2
* ROCm/HIP 7.1.52802-9999
* Clang 20.0.0
* Linux 7.1.10-200.fc44
* Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`

## Verification

```bash
cmake --build build/mi50-release -j2 --target miinfer-openai-api-test miinfer-host-test
./build/mi50-release/miinfer-openai-api-test
./build/mi50-release/miinfer-host-test
```

Physical gates passed:

* Ctrl-C/SIGTERM during model startup exits without requiring `kill -9`.
* SIGTERM during an 8K prefill exits cleanly. The request reached
  `received → queued → dequeued → parsed → tokenized → prefill_started → cancelled`.
* The cancelled request recorded 7,009 prompt tokens, 42,210 request bytes,
  zero generated tokens, and `cancelled=true`.
* A queued client receives HTTP 503 during shutdown.
* An oversized prompt and `max_tokens > 4096` are rejected with HTTP 400.
* The request log contains bounded fields for body/message/prompt sizes,
  tokenization, queue wait, prefill, TTFT, decode, cancellation, finish reason,
  and error.

The durable client-disconnect transcript is in
`results/m18-lifecycle/20260909-111922/`; it records the same 7,009-token
prefill cancelled after the client closed its socket.

## Decision

KEEP. Cancellation is host-boundary based; HIP kernels are never interrupted
mid-dispatch. The physical cancellation transcript is retained in the session
run log, while the repeatable queue gate remains in `scripts/test-serve.sh`.
