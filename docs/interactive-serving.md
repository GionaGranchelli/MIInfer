# Interactive serving

`m25_hi_qualified` is the P512 benchmark preset. `m25_interactive` adds
`MIINFER_MX_MMV=1` and `MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE=1` to
its resident/wide prefill configuration. This is an experimental serving
candidate; see EXP-0350 for validation results and limitations.

```bash
MIINFER_PRESET=m25_interactive MIINFER_API_KEY=miinfer \
  build/mi50-release/miinfer serve \
  --model /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --context 16384 --experimental-context --port 8080
```

The context limit includes prompt plus requested output tokens. Use a larger
capacity only when the conversation requires it. The preset does not change
the existing explicit context qualification policy.

Wide prefill uses the full-layer-major contract in bounded, offset-aware chunks.
For example, 3991 tokens use 7 x 512 + 384 + 23; 9317 use 18 x 512 + 64 + 37.
Attention KV, recurrent state and convolution history persist between chunks.

The HTTP server enables experimental single-session reuse only with
`MIINFER_SESSION_REUSE=1`. It retains one GPU checkpoint at the largest completed
512-token wide-prefill boundary: recurrent state, convolution history, and the
KV prefix. A matching later request restores that checkpoint and replays only
the suffix. It does not retain scalar-decoded state, which is known not to be
token-equivalent when handed to wide prefill.

Each in-memory checkpoint stores the exact token prefix, whose length is the
absolute boundary, and execution-contract version 1 alongside the recurrent
state, convolution history, and attention KV prefix. Increment the version when
the saved-state layout or wide-prefill numerical contract changes. The cache
lives only inside its model/runtime engine; changing model, quantization, or
preset requires a new engine and cannot reuse the old checkpoint.

When enabled, reuse requires the incoming token sequence to strictly extend the
single checkpoint prefix. Mismatch, reset, cancellation, or generation failure
falls back to full replay. A prompt shorter than 512 tokens has no reusable
checkpoint. The server reports `cache_hit` and reused/new token counts for every
request.

`miinfer_request_latency` is emitted after response output, alongside the existing
engine diagnostics. It records prompt/common-prefix/reused/new-prefill counts,
prefill and graph-capture time, first decode computation time, `TTFT_wall_ms`,
steady decode throughput, total request wall time and `cache_hit`.
Wall time begins when the complete HTTP request is received, includes queueing,
and ends at the first successful nonempty content/tool-call SSE write. For
nonstreaming responses it ends at response delivery. No emitted output is
reported as null TTFT. Socket-write completion is the server-side measurement;
it is not a client acknowledgement. Tool-enabled output is currently buffered
until parsing completes, so its TTFT can include the entire generation.

The interactive preset limits HIP graph capture to positions below 4096. Long
Pi generations use normal decode after that point, avoiding per-position graph
preparation overhead. Override `MIINFER_HIP_GRAPH_MAX_POSITION` only for
benchmarking.

For the M26 real-context decode curve, use the built-in Qwen3.8 runtime
benchmark. It measures 128 decode tokens after each fresh context and reports
median decode milliseconds and tokens per second:

```bash
MIINFER_PRESET=m25_interactive build/mi50-release/miinfer run \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --context 16384 --decode-curve --curve-iterations 5 --no-stream
```

This is decode-only timing; prefill and model loading are excluded from the
reported decode values. Keep GPU clocks and the preset fixed when comparing
runs.

For one sampled long-context token, disable graphs and enable the existing
stage-event attribution:

```bash
MIINFER_PRESET=m25_interactive \
MIINFER_DECODE_PROFILE=1 MIINFER_DECODE_PROFILE_POSITION=12288 \
MIINFER_HIP_GRAPH=0 \
build/mi50-release/miinfer run \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --context 16384 --decode-curve --curve-context 12288 \
  --curve-iterations 1 --no-stream
```

The output reports sampled recurrent and attention stage timings. Profiling is
diagnostic and synchronizing; do not use its throughput as a qualification
number.

Decode step latency includes graph capture. The legacy
`time_to_first_token_ms` field remains an internal prefill-plus-first-decode
measurement; use `TTFT_wall_ms` for interactive comparisons. Prometheus TTFT
and request-duration counters use the wall measurements.

Run the same-process GPU regression check with:

```bash
MIINFER_PRESET=m25_interactive build/mi50-release/miinfer run \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt 'The quick brown fox jumps over the lazy dog. ' \
  --context 16384 --check-session
```

This exercises checkpoint suffix replay versus reset/full replay at 512, 640,
3991, 8192 and 16000 prompt tokens. It is a correctness screen, not a
performance qualification or a substitute for real Pi tool-loop testing.

For the targeted wide-versus-scalar recurrent-state diagnostic, add
`MIINFER_INTERACTIVE_VALIDATE=1`. It enables validation for the first recurrent
layer, including the Mx wide path, and prints maximum output, state, and convolution-history errors;
it is intentionally too slow for serving.
