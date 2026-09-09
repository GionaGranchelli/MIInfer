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

The 131072 allocation retained approximately 27.996 GB VRAM while resident;
the model remained loadable and the process shut down cleanly. This proves
dynamic allocation, not 128K inference correctness.

## Long-context inference evidence

* P1024: HTTP 200, 809 prompt tokens, 200-token request, 25.07 s.
* P8192: HTTP 200, 7,009 prompt tokens, 200-token request, 239.998 s.
* P16384: HTTP 200, 16,009 prompt tokens, one generated token, 439.770 s
  wall time and 36.405 prompt tokens/s using the M12 path. Raw request and
  response data are in `results/m19-context/20260909-context/16384-retry/`.
* The 8K run completed prefill and generation; the same shape was also used
  to verify cancellation during prefill.

The 16K–128K prompt correctness gates remain experimental and are not promoted
by the allocation sweep alone.

## Decision

KEEP the dynamic capacity implementation. Do not advertise 128K as qualified;
it is a working allocation ceiling pending long-context KV/state correctness.
