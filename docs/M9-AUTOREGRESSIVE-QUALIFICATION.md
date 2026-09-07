# Milestone M9 — Real Autoregressive Generation Qualification

## 1. Scope & Objective

Milestone M9 requires qualifying real autoregressive greedy generation across multiple sequence lengths (64, 256, 512 tokens), validating:
1. Exact deterministic trajectory replay between consecutive cold-reset runs.
2. Full 64-layer state fingerprint match across runs.
3. Zero dynamic memory allocations during token generation.
4. Latency and tokens/second measurements under locked 1606 MHz SCLK / 1000 MHz MCLK.

---

## 2. Autoregressive Test Results

### 64 Tokens Generation (`--generate64`)

- **Prompt Token:** 11 (fixture prompt)
- **Generated Token Count:** 64
- **First Decode Latency:** 2134.56 ms (29.98 tok/s)
- **Second Decode Latency (Replay):** 2122.01 ms (30.16 tok/s)
- **Replay Determinism:** PASS (exact bitwise sequence match)
- **End State Fingerprint:** `11386477227812553549` (identical in both runs)
- **Decode Allocations:** 0
- **Device Bytes:** 18,886,426,964 bytes

### 256 Tokens Generation (`--generate256`)

- **Prompt Token:** 11
- **Generated Token Count:** 256
- **First Decode Latency:** 8575.97 ms (29.85 tok/s)
- **Second Decode Latency (Replay):** 8581.85 ms (29.83 tok/s)
- **Replay Determinism:** PASS (exact bitwise sequence match across 256 tokens)
- **End State Fingerprint:** `13010542232188166775` (identical in both runs)
- **Decode Allocations:** 0
- **Device Bytes:** 18,886,426,964 bytes

### 512 Tokens Generation (`--generate512`)

- **Prompt Token:** 11
- **Generated Token Count:** 512
- **First Decode Latency:** 17507.50 ms (29.24 tok/s)
- **Second Decode Latency (Replay):** 17635.10 ms (29.03 tok/s)
- **Replay Determinism:** PASS (exact bitwise sequence match across 512 tokens)
- **End State Fingerprint:** `678289901342068333` (identical in both runs)
- **Decode Allocations:** 0
- **Device Bytes:** 18,886,426,964 bytes

---

## 3. Stability & Determinism Verdict

Across all three test horizons (64, 256, 512 tokens):
1. **Determinism:** 100% bitwise token agreement between independent generation passes from reset state.
2. **State Integrity:** Cumulative layer state fingerprints (recurrent cell states and attention KV entries) matched bit-for-bit between runs.
3. **Memory Safety:** Exactly 0 bytes allocated during token decode; zero memory leaks or growth.
4. **Conclusion:** QUALIFIED PASS.
