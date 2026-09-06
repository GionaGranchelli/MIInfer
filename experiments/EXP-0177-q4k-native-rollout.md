# EXP-0177 — Native Q4_K layout rollout across projection families

## Hypothesis

Expanding the validated gfx906-native `Q4KWaveTile` representation from FFN Down (EXP-0174/EXP-0176) to the remaining compatible Q4_K projection families will systematically recover execution time and narrow the decode performance gap against pinned llama.cpp on AMD Instinct MI50 (32 GB).

For Candidate 1 (FFN Gate & Up):
Converting the 128 Q4_K FFN Gate and Up tensors (`[columns=5120, rows=17408]`) to the native wave-tile representation while sharing a single `Q8_1` quantization of `post_normalized` will reduce decode latency by ~8.0–8.5 ms/token, increasing generation throughput from ~14.56 tok/s to ~16.5 tok/s.

## Motivation

EXP-0174 proved that the native `Q4KWaveTile` primitive (two 64-word planes + 4 metadata blocks) delivers a ~1.91x speedup over canonical/expanded GEMV in isolation, yielding a 2.75 ms/token reduction across 32 Q4_K Down projections (+4.2% TG64).

Under the qualified sustained operating point established in EXP-0176 (manual DPM SCLK 1606 MHz, MCLK 1000 MHz), we now execute a staged rollout campaign across all compatible Q4_K projection families.

## Phase 0: External llama.cpp baseline refresh

Pinned commit: `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`
Hardware state: manual DPM 1606/1000 MHz, 100% SCLK/MCLK residency, junction 95–97 °C.
Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`

Measurements (5 repetitions each):
- **PP512:** `[185.64, 187.443, 187.574, 187.534, 187.601]`, **Median = 187.534 tok/s** (5.3323 ms/tok)
- **TG64:** `[22.3167, 22.4015, 22.4167, 22.4182, 22.4397]`, **Median = 22.4167 tok/s** (44.6096 ms/tok)
- **TG128:** `[22.4742, 22.4769, 22.4950, 22.5328, 22.5344]`, **Median = 22.4950 tok/s** (44.4543 ms/tok)
- **TG256:** `[22.5158, 22.5252, 22.5531, 22.5544, 22.6058]`, **Median = 22.5531 tok/s** (44.3398 ms/tok)

### Parity Target

Current MIInfer (EXP-0176 Down-only baseline):
- TG64: 14.5556 tok/s (68.7021 ms/tok)
- TG128: 14.3171 tok/s (69.8466 ms/tok)

Gap to llama.cpp parity:
- **TG64:** 68.7021 - 44.6096 = **24.0925 ms/tok** reduction required (1.540x speedup needed)
- **TG128:** 69.8466 - 44.4543 = **25.3923 ms/tok** reduction required (1.571x speedup needed)

## Phase 1: Workload analysis & Amdahl ceiling

Profiling baseline at position 63 (manual 1606/1000 MHz):
Total token GPU time: 71.1741 ms (Layer sum: 67.9199 ms; 48 Recurrent = 49.4205 ms; 16 Attention = 18.4995 ms).

Inventory of remaining Q4_K projection families:

| Priority | Family | Count | Shape [col, row] | Measured ms/tok | Expected Saving @ 1.91x | Target Throughput |
|---|---|---|---|---|---|---|
| **1** | **FFN Gate & Up** | 128 (64 gate + 64 up) | [5120, 17408] | **17.70 ms** | **~8.43 ms** | ~16.5 tok/s |
| **2** | **Full-Attn Q** | 16 | [5120, 12288] | **3.35 ms** | **~1.60 ms** | ~17.0 tok/s |
| **3** | **Recurrent Attn Gate** | 48 | [5120, 6144] | **2.84 ms** | **~1.35 ms** | ~17.4 tok/s |
| **4** | **Full-Attn Out** | 16 | [6144, 5120] | **1.28 ms** | **~0.61 ms** | ~17.6 tok/s |
| **5** | **Full-Attn K** | 16 | [5120, 1024] | **0.55 ms** | **~0.26 ms** | ~17.7 tok/s |
| **Total** | All remaining Q4_K | 224 | — | **25.72 ms** | **~12.25 ms** | — |

*Note:* Full Q4_K rollout can deliver ~12.25 ms/tok out of the ~24.09 ms/tok gap, bringing MIInfer to ~58.9 ms/tok (~17.0 tok/s). The remaining ~11.8 ms gap belongs to non-Q4_K operations (e.g. Q6_K Down, recurrent SSM, attention, layernorms).

## Candidate 1: FFN Gate & Up Rollout

### Implementation Scope
- Generalized host packing `pack_q4k_wave_tensor` to handle 2D Q4_K matrices where `columns % 1024 == 0` and `rows % 2 == 0`.
- Retained exact bitwise metadata reconstruction checks and per-byte payload verification.
- Generalized device kernel `wave_gemv<NumTiles>` for `NumTiles = columns / 1024` (`NumTiles = 5` for Gate/Up).
- Integrated `d_ffn_gate_native` and `d_ffn_up_native` into `RecurrentLayer` and `FullAttentionLayer`.
- Environment control: `MIINFER_Q4K_NATIVE_GATE_UP` (0 or 1).
- Memory cost: 128 tensors * 17408 rows * 5 tiles * 640 bytes = 7.13 GiB device bytes (vs canonical 6.45 GiB), net increase +713,031,680 bytes (+680.00 MiB).

### Correctness
- Release CTest: **20/20 PASS (100%)**
- Deterministic Replay: **PASS** (`state_fingerprint=7170666995386517766`, exact bitwise match with EXP-0174)
- Decode Loop Allocations: **0** across all runs
- Observable Contract: `--prefix64-observable-contract` **PASS** with `poisoned_reset_replay=PASS`

### Hardware State Validation
Driver state: `MANUAL`, SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz).
Continuous 250 ms telemetry (`scripts/sample-gpu.sh`):
- TG64 suite: 1280 samples, **100.0% SCLK residency at 1606 MHz**, **100.0% MCLK residency at 1000 MHz**, max junction 83.0 °C, max edge 57.0 °C.
- TG128 suite: 1909 samples, **100.0% SCLK residency at 1606 MHz**, **100.0% MCLK residency at 1000 MHz**, max junction 89.0 °C, max edge 62.0 °C.

### A/B Benchmark Results (Interleaved 5 Pairs)

#### TG64
- Control runs (Down only): `[14.5389, 14.5330, 14.5449, 14.5457, 14.5625]` tok/s. **Median = 14.5449 tok/s** (4400.18 ms)
- Native runs (Down + Gate/Up): `[15.8443, 15.8468, 15.8345, 15.8569, 15.8354]` tok/s. **Median = 15.8443 tok/s** (4039.30 ms)
- **Throughput gain:** **+8.93%**
- **Latency reduction:** **5.6388 ms/token** (360.88 ms per 64 tokens)

#### TG128
- Control runs (Down only): `[14.3180, 14.3039, 14.3207, 14.3178, 14.3159]` tok/s. **Median = 14.3178 tok/s** (8939.91 ms)
- Native runs (Down + Gate/Up): `[15.5607, 15.5635, 15.5794, 15.5690, 15.5529]` tok/s. **Median = 15.5635 tok/s** (8224.35 ms)
- **Throughput gain:** **+8.70%**
- **Latency reduction:** **5.5903 ms/token** (715.56 ms per 128 tokens)

### P63 GPU Profile Breakdown
- Control (Down only): Total GPU = 71.1741 ms, Layer sum = 67.9199 ms
- Candidate 1 (Down + Gate/Up): Total GPU = 65.7055 ms, Layer sum = 62.4427 ms
- Delta: **-5.4686 ms/token** total GPU, **-5.4772 ms/token** layer sum
- FFN Gate+Up stage per layer: reduced from ~0.276 ms to 0.190 ms (-0.086 ms/layer * 64 layers = -5.54 ms/token)

### Gap to llama.cpp Status
- Initial Gap (TG64): 24.1432 ms/token
- Remaining Gap (TG64): 18.5045 ms/token (**23.4% of gap closed**)
- Initial Gap (TG128): 25.3887 ms/token
- Remaining Gap (TG128): 19.7984 ms/token (**22.0% of gap closed**)

### Decision
**KEEP**. Candidate 1 achieves a verified +8.7–8.9% throughput improvement across both TG64 and TG128, with 100% clock stability, zero decode allocations, exact replay, and contract compliance.

---

## Candidate 2: Full-Attention Q Projection Rollout

### Implementation Scope
- Converted 16 full-attention Q projection tensors (`[columns=5120, rows=12288]`, Q4_K) to gfx906-native `Q4KWaveTile` format (`NumTiles = 5`).
- Input `normalized` (5120 floats) quantized with `launch_q8_1_quantize_f32` into pre-allocated `q8_1` buffer.
- Replaced canonical `project(q_weight, ...)` with `launch_q4k_wave_gemv(d_q_native, q8_1, qfull, 12288, 5120)`.
- Replaced canonical `d_q` allocation with `d_q_native` when enabled: +60.00 MiB device memory across 16 tensors (19.52 GiB -> 19.58 GiB).
- Preserved decoupled quantization: attention stage 3 (`k_projection`) continues independent Q8_K quantization of `normalized`.
- Environment control: `MIINFER_Q4K_NATIVE_Q` (0 or 1).

### Correctness
- Release CTest: **20/20 PASS (100%)**
- Observable Contract (`--prefix64-observable-contract`): **PASS**
  - `poisoned_reset_replay=PASS`
  - 64/64 positions teacher-forced exact argmax match with reference
  - Position 64 top5 overlap: 5/5
  - Logits cosine similarity: 0.999601
- Autoregressive Generation (`--generate16`): **PASS** (`replay=PASS`, 0 allocations)

### Hardware State Validation
Driver state: `MANUAL`, SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz).
Continuous 250 ms telemetry (`scripts/sample-gpu.sh`):
- TG64 suite: 1457 samples, **100.0% SCLK residency at 1606 MHz**, **100.0% MCLK residency at 1000 MHz**, max junction 83.0 °C, max edge 56.0 °C.
- TG128 suite: 2060 samples, **100.0% SCLK residency at 1606 MHz**, **100.0% MCLK residency at 1000 MHz**, max junction 88.0 °C, max edge 61.0 °C.

### A/B Benchmark Results (Interleaved 5 Pairs)

#### TG64
- Control runs (Down + Gate/Up, Q=0): `[15.8559, 15.8344, 15.8356, 15.8177, 15.8458]` tok/s. **Median = 15.8356 tok/s** (4041.52 ms)
- Native runs (Down + Gate/Up + Q=1): `[16.3970, 16.4067, 16.3952, 16.4082, 16.3908]` tok/s. **Median = 16.3970 tok/s** (3903.14 ms)
- **Throughput gain:** **+3.55%**
- **Latency reduction:** **2.1622 ms/token** (138.38 ms per 64 tokens)

#### TG128
- Control runs (Down + Gate/Up, Q=0): `[15.5608, 15.5825, 15.5836, 15.5957, 15.5750]` tok/s. **Median = 15.5825 tok/s** (8214.35 ms)
- Native runs (Down + Gate/Up + Q=1): `[16.1023, 16.1270, 16.1076, 16.1144, 16.1146]` tok/s. **Median = 16.1144 tok/s** (7943.22 ms)
- **Throughput gain:** **+3.41%**
- **Latency reduction:** **2.1182 ms/token** (271.13 ms per 128 tokens)

### P63 GPU Profile Breakdown
- Control (Q=0): Total GPU = 65.7822 ms, Layer sum = 62.5204 ms
- Native (Q=1): Total GPU = 63.5116 ms, Layer sum = 60.2556 ms
- Delta: **-2.2706 ms/token** total GPU, **-2.2648 ms/token** layer sum
- Full-attention Q stage per layer: reduced from 0.20928 ms to 0.07360 ms (**2.84x kernel speedup**, -0.1357 ms/layer * 16 layers = -2.17 ms/token)

### Cumulative Gap to llama.cpp Status
- Initial Gap (TG64): 24.1432 ms/token
- Remaining Gap after Candidate 2 (TG64): **16.3770 ms/token** (**32.2% of total gap closed**)
- Initial Gap (TG128): 25.3887 ms/token
- Remaining Gap after Candidate 2 (TG128): **17.6021 ms/token** (**30.7% of total gap closed**)

### Decision
**KEEP**. Candidate 2 delivers a clean +3.4–3.6% whole-model throughput increase (-2.12 to -2.16 ms/tok saving), perfectly matching Amdahl expectations for the 16 Q projection layers with 0 decode allocations, exact contract compliance, and 100% clock stability.

## Candidate 3: Recurrent Attention Gate Rollout

### Implementation Scope
- Converted 48 recurrent attention gate projection tensors (`[columns=5120, rows=6144]`, Q4_K) to gfx906-native `Q4KWaveTile` format (`NumTiles = 5`).
- Input `normalized` (5120 floats) quantized with `launch_q8_1_quantize_f32` into pre-allocated `q8_1` buffer.
- Replaced canonical `project(gate_weight, ...)` with `launch_q4k_wave_gemv(d_attn_gate_native, q8_1, gate, 6144, 5120)`.
- Memory conservation: Canonical `d_gate` buffer is skipped when `d_attn_gate_native` is allocated, resulting in +90.00 MiB device memory across 48 tensors (19.58 GiB -> 19.67 GiB).
- Environment control: `MIINFER_Q4K_NATIVE_ATTN_GATE` (0 or 1).

### Correctness
- Release CTest: **20/20 PASS (100%)**
- Observable Contract (`--prefix64-observable-contract`): **PASS**
  - `poisoned_reset_replay=PASS`
  - 64/64 positions teacher-forced exact argmax match with reference
  - Position 64 top5 overlap: 5/5
  - Logits cosine similarity: 0.999601
- Autoregressive Generation (`--generate16`): **PASS** (`replay=PASS`, 0 allocations, bit-identical state fingerprint `15471922765589133669`)

### Hardware State Validation
Driver state: `MANUAL`, SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz).
Continuous 250 ms telemetry (`scripts/sample-gpu.sh`):
- TG64 suite: 1460 samples, **100.0% SCLK residency at 1606 MHz**, **100.0% MCLK residency at 1000 MHz**, max junction 83.0 °C, max edge 56.0 °C.
- TG128 suite: 2042 samples, **100.0% SCLK residency at 1606 MHz**, **100.0% MCLK residency at 1000 MHz**, max junction 89.0 °C, max edge 62.0 °C.

### A/B Benchmark Results (Interleaved 5 Pairs)

#### TG64
- Control runs (Down + Gate/Up + Q, ATTN_GATE=0): `[16.3933, 16.4068, 16.4110, 16.4147, 16.4263]` tok/s. **Median = 16.4110 tok/s** (3899.82 ms)
- Native runs (Down + Gate/Up + Q + ATTN_GATE=1): `[16.6237, 16.6215, 16.6242, 16.6127, 16.6237]` tok/s. **Median = 16.6237 tok/s** (3849.92 ms)
- **Throughput gain:** **+1.30%**
- **Latency reduction:** **0.7797 ms/token** (49.90 ms per 64 tokens)

#### TG128
- Control runs (Down + Gate/Up + Q, ATTN_GATE=0): `[16.1098, 16.1267, 16.1240, 16.1153, 16.1178]` tok/s. **Median = 16.1178 tok/s** (7941.54 ms)
- Native runs (Down + Gate/Up + Q + ATTN_GATE=1): `[16.3110, 16.2988, 16.2952, 16.3152, 16.3109]` tok/s. **Median = 16.3109 tok/s** (7847.50 ms)
- **Throughput gain:** **+1.20%**
- **Latency reduction:** **0.7347 ms/token** (94.04 ms per 128 tokens)

### P63 GPU Profile Breakdown
- Control (ATTN_GATE=0): Total GPU = 63.5116 ms, Layer sum = 60.2556 ms
- Native (ATTN_GATE=1): Total GPU = 62.5276 ms, Layer sum = 59.2719 ms
- Delta: **-0.9840 ms/token** total GPU, **-0.9837 ms/token** layer sum
- Recurrent Attention Gate stage per layer: reduced from 0.0592 ms to 0.0424 ms (-0.0168 ms/layer * 48 layers = -0.81 ms/token)

### Cumulative Gap to llama.cpp Status
- Initial Gap (TG64): 24.1432 ms/token
- Remaining Gap after Candidate 3 (TG64): **15.5454 ms/token** (**35.6% of total gap closed**)
- Initial Gap (TG128): 25.3887 ms/token
- Remaining Gap after Candidate 3 (TG128): **16.8543 ms/token** (**33.6% of total gap closed**)

### Decision
**KEEP**. Candidate 3 adds another steady +1.2–1.3% throughput gain (-0.73 to -0.78 ms/tok saving) across 48 recurrent layers with 0 decode allocations, bit-exact replay, and 100% clock residency.

## Candidate 4: Full-Attention Output Projection Rollout

### Implementation Scope
- Converted 16 full-attention output projection tensors (`[columns=6144, rows=5120]`, Q4_K) to gfx906-native `Q4KWaveTile` format (`NumTiles = 6`).
- Input `gated_attention` (6144 floats) quantized with `launch_q8_1_quantize_f32` into pre-allocated `q8_1` buffer.
- Replaced canonical `project(o_weight, ...)` with `launch_q4k_wave_gemv(d_o_native, q8_1, projected, 5120, 6144)`.
- Memory conservation: Canonical `d_o` buffer is skipped when `d_o_native` is allocated, resulting in +30.00 MiB device memory across 16 tensors (19.67 GiB -> 19.70 GiB).
- Environment control: `MIINFER_Q4K_NATIVE_ATTN_OUT` (0 or 1).

### Correctness
- Release CTest: **20/20 PASS (100%)**
- Observable Contract (`--prefix64-observable-contract`): **PASS**
  - `poisoned_reset_replay=PASS`
  - 64/64 positions teacher-forced exact argmax match with reference
  - Position 64 top5 overlap: 5/5
  - Logits cosine similarity: 0.999648
- Autoregressive Generation (`--generate16`): **PASS** (`replay=PASS`, 0 allocations)

### Hardware State Validation
Driver state: `MANUAL`, SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz).
Continuous 250 ms telemetry (`scripts/sample-gpu.sh`):
- TG64 suite: 1453 samples, **100.0% SCLK residency at 1606 MHz**, **100.0% MCLK residency at 1000 MHz**, max junction 83.0 °C, max edge 56.0 °C.
- TG128 suite: 2035 samples, **100.0% SCLK residency at 1606 MHz**, **100.0% MCLK residency at 1000 MHz**, max junction 89.0 °C, max edge 61.0 °C.

### A/B Benchmark Results (Interleaved 5 Pairs)

#### TG64
- Control runs (Down + Gate/Up + Q + AttnGate, ATTN_OUT=0): `[16.6087, 16.5762, 16.5882, 16.6011, 16.5989]` tok/s. **Median = 16.5989 tok/s** (3855.67 ms)
- Native runs (Down + Gate/Up + Q + AttnGate + ATTN_OUT=1): `[16.8942, 16.8856, 16.8936, 16.8949, 16.8926]` tok/s. **Median = 16.8936 tok/s** (3788.41 ms)
- **Throughput gain:** **+1.78%**
- **Latency reduction:** **1.0509 ms/token** (67.26 ms per 64 tokens)

#### TG128
- Control runs (Down + Gate/Up + Q + AttnGate, ATTN_OUT=0): `[16.3079, 16.2985, 16.3066, 16.3003, 16.3069]` tok/s. **Median = 16.3066 tok/s** (7849.58 ms)
- Native runs (Down + Gate/Up + Q + AttnGate + ATTN_OUT=1): `[16.5761, 16.5821, 16.5677, 16.5667, 16.6039]` tok/s. **Median = 16.5761 tok/s** (7721.95 ms)
- **Throughput gain:** **+1.65%**
- **Latency reduction:** **0.9971 ms/token** (127.63 ms per 128 tokens)

### P63 GPU Profile Breakdown
- Control (ATTN_OUT=0): Total GPU = 62.5276 ms, Layer sum = 59.2719 ms
- Native (ATTN_OUT=1): Total GPU = 61.4607 ms, Layer sum = 58.1967 ms
- Delta: **-1.0669 ms/token** total GPU, **-1.0752 ms/token** layer sum
- Full-Attention Output stage per layer: reduced from 0.1088 ms to 0.0440 ms (**2.47x speedup**, -0.0648 ms/layer * 16 layers = -1.04 ms/token)

### Cumulative Gap to llama.cpp Status
- Initial Gap (TG64): 24.1432 ms/token
- Remaining Gap after Candidate 4 (TG64): **14.5843 ms/token** (**39.6% of total gap closed**)
- Initial Gap (TG128): 25.3887 ms/token
- Remaining Gap after Candidate 4 (TG128): **15.8734 ms/token** (**37.5% of total gap closed**)

### Decision
**KEEP**. Candidate 4 delivers a clean +1.65–1.78% throughput increase (-1.00 to -1.05 ms/tok), bringing MIInfer within 15 ms/tok of pinned llama.cpp.

---

## Candidate 5: Full-Attention K Projection Rollout

### Implementation Scope
- Converted 16 full-attention K projection tensors (`[columns=5120, rows=1024]`, Q4_K) to gfx906-native `Q4KWaveTile` format (`NumTiles = 5`).
- Exploited activation reuse: `normalized` was already quantized to `Q8_1` during stage 1 (Q projection); stage 3 reuses `q8_1` directly without re-quantization overhead.
- Stage 5 (`v_weight`, Q6_K) performs independent Q8_K quantization of `normalized`.
- Memory conservation: Canonical `d_k` buffer skipped when `d_k_native` allocated (+5.00 MiB device memory, total 19.71 GiB).
- Environment control: `MIINFER_Q4K_NATIVE_K` (0 or 1).

### Correctness
- Release CTest: **20/20 PASS (100%)**
- Observable Contract (`--prefix64-observable-contract` with all 5 families active): **PASS**
  - `poisoned_reset_replay=PASS`
  - 64/64 positions teacher-forced exact argmax match with reference
  - Position 64 top5 overlap: 5/5
  - Logits cosine similarity: 0.999607
  - Position 64 max error: 10.1854
- Autoregressive Generation (`--generate16`): **PASS** (`replay=PASS`, 0 decode allocations, decode tok/s = 17.22)

### Hardware State Validation
Driver state: `MANUAL`, SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz).
Continuous 250 ms telemetry (`scripts/sample-gpu.sh`):
- TG64 suite: 1462 samples, **100.0% SCLK residency at 1606 MHz**, **100.0% MCLK residency at 1000 MHz**, max junction 83.0 °C, max edge 55.0 °C.
- TG128 suite: 2027 samples, **100.0% SCLK residency at 1606 MHz**, **100.0% MCLK residency at 1000 MHz**, max junction 89.0 °C, max edge 61.0 °C.

### A/B Benchmark Results (Interleaved 5 Pairs)

#### TG64
- Control runs (K=0): `[16.8864, 16.8976, 16.8820, 16.8944, 16.8703]` tok/s. **Median = 16.8944 tok/s** (3788.24 ms)
- Native runs (K=1): `[16.9938, 17.0223, 17.0270, 17.0267, 17.0207]` tok/s. **Median = 17.0223 tok/s** (3759.77 ms)
- **Throughput gain:** **+0.76%**
- **Latency reduction:** **0.4448 ms/token** (28.47 ms per 64 tokens)
- **Milestone:** Crossed the 17.0 tok/s threshold on sustained TG64.

#### TG128
- Control runs (K=0): `[16.5934, 16.5726, 16.5836, 16.5852, 16.5793]` tok/s. **Median = 16.5836 tok/s** (7719.38 ms)
- Native runs (K=1): `[16.6893, 16.6867, 16.7037, 16.6959, 16.6995]` tok/s. **Median = 16.6959 tok/s** (7666.57 ms)
- **Throughput gain:** **+0.68%**
- **Latency reduction:** **0.4126 ms/token** (52.81 ms per 128 tokens)

### P63 GPU Profile Breakdown
- Control (K=0): Total GPU = 61.4607 ms, Layer sum = 58.1967 ms
- Native (K=1): Total GPU = 61.3207 ms, Layer sum = 58.0631 ms
- Delta: **-0.1400 ms/token** total GPU, **-0.1336 ms/token** layer sum
- Full-Attention K stage per layer: reduced from 0.03824 ms to 0.01488 ms (**2.57x speedup**, -0.0234 ms/layer * 16 layers = -0.37 ms/token)

### Decision
**KEEP**. Candidate 5 completes the Q4_K layout rollout campaign across all compatible projection families, breaking 17.0 tok/s on TG64 with zero regressions and zero decode allocations.

---

## EXP-0177 Campaign Summary & Parity Reconciliation

### Full Rollout Scorecard

| Milestone / Stage | Repacked Tensors | TG64 tok/s | TG64 ms/tok | TG128 tok/s | TG128 ms/tok | Total Latency Saved | VRAM (Device Bytes) |
|---|---|---|---|---|---|---|---|
| **EXP-0176 (Down only baseline)** | 32 | 14.5449 | 68.7526 | 14.3178 | 69.8431 | — (baseline) | 18.80 GiB |
| **+ Candidate 1: FFN Gate & Up** | +128 (160) | 15.8443 | 63.1139 | 15.5635 | 64.2529 | -5.61 ms/tok | 19.52 GiB (+680 MiB) |
| **+ Candidate 2: Full-Attn Q** | +16 (176) | 16.3970 | 60.9866 | 16.1144 | 62.0564 | -2.14 ms/tok | 19.58 GiB (+60 MiB) |
| **+ Candidate 3: Recurrent Attn Gate** | +48 (224) | 16.6237 | 60.1550 | 16.3109 | 61.3086 | -0.76 ms/tok | 19.67 GiB (+90 MiB) |
| **+ Candidate 4: Full-Attn Out** | +16 (240) | 16.8936 | 59.1939 | 16.5761 | 60.3277 | -1.02 ms/tok | 19.70 GiB (+30 MiB) |
| **+ Candidate 5: Full-Attn K** | +16 (256) | **17.0223** | **58.7464** | **16.6959** | **59.8951** | -0.43 ms/tok | 19.71 GiB (+5 MiB) |
| **Full Q4_K Campaign Delta** | **224 new (256 total)** | **+17.03%** | **-10.0062 ms** | **+16.61%** | **-9.9480 ms** | **~10.0 ms/tok** | **+915 MiB net** |

### Parity Gap Reconciliation vs Pinned llama.cpp

- Pinned llama.cpp baseline (manual DPM 1606/1000 MHz):
  - **TG64:** 22.4167 tok/s (44.6096 ms/tok)
  - **TG128:** 22.4950 tok/s (44.4543 ms/tok)
- Initial MIInfer gap (Down-only):
  - **TG64:** 24.1430 ms/tok
  - **TG128:** 25.3888 ms/tok
- Final MIInfer gap (Full Q4_K rollout):
  - **TG64:** 58.7464 - 44.6096 = **14.1368 ms/tok** (**41.4% of total gap closed**)
  - **TG128:** 59.8951 - 44.4543 = **15.4408 ms/tok** (**39.2% of total gap closed**)

### Architectural Invariants Preserved
1. Zero decode-loop heap allocations across all candidates (`allocations_during_decode = 0`).
2. Exact deterministic replay verified bitwise on all candidate runs.
3. 100% SCLK residency at 1606 MHz and 100% MCLK residency at 1000 MHz confirmed by continuous 250 ms telemetry (no throttling, junction < 90 °C).
4. `--prefix64-observable-contract` passed across all 64 teacher-forced positions with 100% argmax agreement and 0.9996+ logits cosine similarity.
5. All 20/20 CTest unit and integration tests passing.
