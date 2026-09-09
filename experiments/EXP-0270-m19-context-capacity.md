# EXP-0270 — M19 dynamic context capacity qualification

## Hypothesis

Context capacity can be selected at model-load time instead of remaining a
single compile-time cache size.

## Contract

Supported values are `1024`, `8192`, `16384`, `32768`, `65536`, and `131072`.
Values above 1024 require `--experimental-context`; no value is silently
clamped. The normal qualified context remains 1024.

## Physical verification

On the MI50, each value in the ladder was started with:

```bash
miinfer serve --model /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --context N --experimental-context
```

All six servers reached `/healthz` and exited cleanly on SIGTERM. Each startup
log reported matching `configured_context_length=N` and
`runtime_context_capacity=N`. Raw logs and hardware captures are in
`results/m19-context/20260909-allocation-sweep/`.

The rebuilt 131072 allocation reported 2,396 device allocations and
27,342,143,828 bytes internally, while `rocm-smi` showed the corresponding
process/driver VRAM capture. The model remained loadable and the process shut
down cleanly. The detailed bounded ledger is in
`results/m19-context/20260909-context/131072-ledger/ledger.json`. This proves
dynamic allocation, not 128K inference correctness.

## Long-context inference evidence

* P1024: HTTP 200, 809 prompt tokens, 200-token request, 25.07 s.
* P8192: HTTP 200, 7,009 prompt tokens, 200-token request, 239.998 s.
* P16384: HTTP 200, 16,009 prompt tokens, one generated token, 439.770 s
  wall time and 36.405 prompt tokens/s using the M12 path. Raw request and
  response data are in `results/m19-context/20260909-context/16384-retry/`.
* P32768: HTTP 200, 32,009 prompt tokens, one generated token, 939.161 s
  prefill and 34.0826 prompt tokens/s using M12. Raw data are in
  `results/m19-context/20260909-context/32768/`.
* P65536: HTTP 200, 64,009 prompt tokens, one generated token, 2,256.570 s
  prefill and 28.3657 prompt tokens/s using M12. The first decode token took
  1.57063 ms; raw data are in `results/m19-context/20260909-context/65536/`.
* P131072: model load succeeded and an HTTP request with 131,009 prompt tokens
  returned HTTP 200 after 6,373.820 s. Prefill measured 20.5548 tok/s and the
  runtime generated one continuation token. Raw data are in
  `results/m19-context/20260909-context/131072-functional/`.
* The 8K run completed prefill and generation; the same shape was also used
  to verify cancellation during prefill. The 8K TG64 run is recorded under
  `results/m19-context/20260909-context/8192-tg64/`.
* The valid 16K TG64 run completed with 16,009 prompt tokens, 442.436 s
  prefill at 36.1838 tok/s, and 64 generated tokens at 24.6454 tok/s. Raw
  data are in `results/m19-context/20260909-context/16384-tg64/`. A concurrent
  duplicate launch hit OOM before readiness and is retained separately as
  contaminated evidence under `16384-tg64-retry/`.

The original ladder entries were functional smoke measurements rather than
full correctness qualification. The replay re-evaluation below closes that
gap for 8K, 16K, and 32K. The measured PP collapse from 34.0826 tok/s at 32K
to 28.3657 tok/s at 64K and 20.5548 tok/s at 128K still rules out practical
production promotion on this hardware.

## Qualification re-evaluation

The missing replay evidence was subsequently closed for the first three
experimental rungs using the M12 layer-major path. Each replay used a fresh
server process, completed a non-EOS TG64 request, shut down cleanly, and was
compared against the earlier response body:

| context | prompt tokens | generated | PP tok/s | TG tok/s | replay |
|---:|---:|---:|---:|---:|:---:|
| 8K | 8,009 | 64 | 38.37 | 25.05 | PASS |
| 16K | 16,009 | 64 | 36.38 | 24.64 | PASS |
| 32K | 32,022 | 60 | 33.96 | 23.03 | PASS |

The response hashes matched exactly for all three replay pairs. Raw requests,
responses, telemetry, metrics, hardware captures, and commands are under
`results/m19-context/20260909-context/{8192,16384,32768}-tg64-requalification/`.
These three sizes are qualified for functional continuation and deterministic
replay on this binary/configuration, while remaining experimental and
non-production because the PP curve is impractical.

The 64K and 128K runs remain unqualified for deterministic/KV/state and
steady-state TG64. Their measured PP results are sufficient to classify the
current practical performance gate as FAIL: 28.37 tok/s at 64K and 20.55
tok/s at 128K, with the 128K request taking 106.2 minutes to prefill. This is
a performance/dataflow blocker, not a capacity/OOM blocker.

## Decision

KEEP the dynamic capacity implementation. 8K, 16K, and 32K now pass the
functional continuation and deterministic replay gates. The 128K capacity and
one-request functional smoke gates pass, but 128K correctness and practical
serving fail qualification today; do not advertise it as a qualified context.
