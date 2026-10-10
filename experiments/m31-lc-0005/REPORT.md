# M31-LC-0005 — First Route-Divergence Isolation

**Disposition: UNRESOLVED.** The first eager decode route difference is localized and independently reproduced, but the captured run used HIP Graph disabled. The graph-enabled token mismatch remains causally unverified, and no accepted numerical tolerance exists. Exact cross-route token parity remains required; no tolerance was widened and no production kernel was changed.

## Capture identity and safety

The matched Z840 pair used model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, the frozen 2,048-token prompt SHA-256 `5acd37f00d9e670f4b4e06d46fa12349e1dee06e356e6e73a186d74672f5bb47`, 2,304-token capacity, image digest `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`, and MI50 BDF `0000:06:00.0`. The binary came from source `d4905bd5be1a4c3c32e466f319cee12cdd313f56` and has SHA-256 `a71929408b3792db8c7a37dc2034fd6c4cce498d6c6a894849c426f13926c894`.

Both guarded calls passed, raw-logit captures were valid, and process groups exited cleanly. Peak junction temperature was 67°C. The actual run records and metrics say `use_hip_graph=false`. The generated `source-identity.json` incorrectly retained the default `use_hip_graph=true`; the harness now records the selected value, with a regression assertion. Treat each call's effective configuration as authoritative for this capture.

The captured decode call processes the common first generated token, `220`, so its history prefix matches the frozen history `[220, 248046, 198, 248045]`. Both eager routes generated `[220, 16, 15, 15, 15]`; this diagnostic trajectory differs from the graph-enabled route outputs recorded in `z840-causal-replay` (`control`: `[220, 248046, 198, 248045, 271]`; `mmq_only`: `[220, 248046, 198, 248045, 74455]`). The capture therefore localizes the first eager decode operation but does not explain the later graph-enabled token split.

## First eager decode difference

At layer 0, decode position 0, the two routes have byte-identical input, recurrent state before and after GDN, attention normalization, beta, decay, QKV, Q/K/V convolutions, recurrent output, and `post_normalized` FFN input. The captured F32 Gate/Up/SwiGLU outputs differ by relative L2 `0.0032676983` (RMSE `3.29774e-5`, max absolute `0.00131023`). The difference propagates through FFN Down (relative L2 `0.00306066`) and yields a layer-output difference of relative L2 `0.000472100`.

Both production quantizers were reproduced from the captured `post_normalized` input and matched the captured bytes exactly. Their 160 groups also contain bitwise-identical Q8 integers and FP16 scales; their dequantized activation vectors are bitwise identical. What differs is the affine sum used by the Q4_K minimum correction:

- Fused Q8_1 recomputes `d * sum(q)` from the quantized integers.
- MMQ Mx Q8_1 stores and consumes the original input sum in FP16.

Those correction sums differ in all 160 groups (relative L2 `0.00854805`, RMSE `0.00553863`, maximum absolute `0.0360107`). The fused reference and MMQ reference, each consuming its captured production bytes and canonical layer-0 GGUF Q4_K weights, differ by relative L2 `0.0032676756`, nearly the same as the GPU outputs. Each GPU route agrees with its own reference to about `1e-7` relative L2. This points to the difference in affine-sum contracts, rather than MMQ accumulation error, as the cause in this eager decode capture.

The canonical Gate and Up tensors are Q4_K with dimensions `[5120, 17408]`, each 50,135,040 bytes. Their raw bytes, the production activation buffers, both routes' captured outputs and state, run records, telemetry, and hashes are retained under [`results/m31-lc-0005/z840-eager-decode-trace-v2`](../../results/m31-lc-0005/z840-eager-decode-trace-v2), with a `SHA256SUMS` inventory in each evidence directory. The repeatable analysis and assert-based layout checks are in [`analyze_gateup.py`](analyze_gateup.py); its output is `gateup-analysis.json` in that evidence directory.

## Layer-0 QKV decomposition

The separate layer-0 QKV analysis reports:

| Contribution | Relative L2 | RMSE | Maximum absolute |
| --- | ---: | ---: | ---: |
| Activation quantization, B vs A | 0.0028577486 | 0.00581758 | 0.0355568 |
| GPU arithmetic, C vs B | 1.90388e-7 | 3.87294e-7 | 7.62939e-6 |
| Total, C vs A | 0.0028577553 | 0.00581759 | 0.0355587 |

The captured production quantizer bytes match the independent quantizer implementation. The result isolates the measured QKV discrepancy to activation quantization within the tested reference, but does not classify it as acceptable: the project has no verified QKV bound for this contract. Raw captures and reference results are under [`results/m31-lc-0005/z840-causal-replay`](../../results/m31-lc-0005/z840-causal-replay).

## Stopping decision

This bounded run reached the first eager decode difference, and the independent operation references explain it. Stop with **UNRESOLVED** because:

1. The parity failure that motivated this milestone was observed with HIP Graph enabled. The trace hook intentionally skips graph-capture calls because host file writes cannot safely run during graph capture. Eager and graph token trajectories differ, so the eager result cannot be promoted to a graph-route conclusion.
2. No independently justified tolerance exists for the FFN route difference or the QKV contract. The exact token-parity gate remains blocked by the graph-enabled outputs above.

The next evidence needed is a graph-compatible post-replay capture of layer-0 Gate/Up bytes and outputs under the same forced token history. Until then, classify neither route as numerically passing and keep the production behavior unchanged.
