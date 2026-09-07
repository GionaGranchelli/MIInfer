#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf}
FIXTURE=/tmp/m6a273-reference-p12
BINARY=./build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block

if [[ ! -x "$BINARY" ]]; then
    echo "Binary $BINARY not found or not executable" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "Model $MODEL not found" >&2
    exit 1
fi
if [[ ! -d "$FIXTURE" ]]; then
    echo "Fixture $FIXTURE not found" >&2
    exit 1
fi

BASE_FLAGS="MIINFER_Q4K_NATIVE_DOWN=1 MIINFER_Q4K_NATIVE_GATE_UP=1 MIINFER_Q4K_NATIVE_Q=1 MIINFER_Q4K_NATIVE_ATTN_GATE=1 MIINFER_Q4K_NATIVE_ATTN_OUT=1 MIINFER_Q4K_NATIVE_K=1 MIINFER_Q5K_NATIVE_SSM_OUT=1 MIINFER_KQUANT_NATIVE_QKV=1 MIINFER_KQUANT_NATIVE_V=1 MIINFER_Q6K_NATIVE_DOWN=1 MIINFER_HIP_GRAPH=1 MIINFER_FUSED_GATE_UP_SWIGLU=1 MIINFER_FUSED_RECURRENT_CORE=1 MIINFER_FAST_ARGMAX=1 MIINFER_TILED_ONLINE_ATTENTION=1 MIINFER_FUSED_ROPE_NORM=1 MIINFER_FUSED_ADD_RMS_NORM=1 MIINFER_FUSED_INTERLAYER_NORM=1 MIINFER_Q6K_NATIVE_LM_HEAD=1 MIINFER_COMBINED_QKV_GATE=1 MIINFER_COMBINED_ATTN_QK=1 MIINFER_SWIGLU_PAIRED=1 MIINFER_KQUANT_FAST_ARITH=1 MIINFER_Q6K_SIMD_UNPACK=1"

# Config A: Baseline M8 decode loop (host-side token step)
FLAGS_A="${BASE_FLAGS} MIINFER_DEVICE_TOKEN_CHAIN=0"

# Config B: Candidate M9 decode loop (device-side token chaining)
FLAGS_B="${BASE_FLAGS} MIINFER_DEVICE_TOKEN_CHAIN=1"

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR="bench/results/m9-qualification/${TIMESTAMP}"
mkdir -p "$OUTDIR"

echo "=========================================================================="
echo "          Milestone M9 Primary Gate Qualification Suite                   "
echo "=========================================================================="
echo "Model:     $MODEL"
echo "Binary:    $BINARY"
echo "Outdir:    $OUTDIR"
echo "Timestamp: $TIMESTAMP"

# 1. Correctness: Observable Contract
echo "--- 1. Running 64-layer observable contract verification ---"
env $FLAGS_B "$BINARY" "$MODEL" "$FIXTURE" --prefix64-observable-contract > "${OUTDIR}/observable_contract.log" 2>&1
grep "observable position=64" "${OUTDIR}/observable_contract.log" || cat "${OUTDIR}/observable_contract.log"

# Start continuous 250ms hardware telemetry
echo "--- 2. Starting continuous 250ms hardware telemetry logging ---"
scripts/sample-gpu.sh "${OUTDIR}/telemetry.jsonl" 250 &
SAMPLER_PID=$!
trap 'kill -TERM "$SAMPLER_PID" 2>/dev/null || true' EXIT

# 3. 5-Pair Interleaved A/B Benchmark on TG64
echo "--- 3. Running 5-Pair Interleaved A/B Benchmark on TG64 ---"
for i in 1 2 3 4 5; do
    echo "Pair $i: Run Config A (M8 unchained baseline)..."
    env $FLAGS_A "$BINARY" "$MODEL" "$FIXTURE" --bench64 > "${OUTDIR}/bench64-A-${i}.log" 2>&1
    grep "benchmark_tokens=" "${OUTDIR}/bench64-A-${i}.log" || cat "${OUTDIR}/bench64-A-${i}.log"
    sleep 3

    echo "Pair $i: Run Config B (M9 chained candidate)..."
    env $FLAGS_B "$BINARY" "$MODEL" "$FIXTURE" --bench64 > "${OUTDIR}/bench64-B-${i}.log" 2>&1
    grep "benchmark_tokens=" "${OUTDIR}/bench64-B-${i}.log" || cat "${OUTDIR}/bench64-B-${i}.log"
    sleep 3
done

# 4. Candidate M9 TG128 Benchmark (5 runs)
echo "--- 4. Running Candidate M9 TG128 Benchmark (5 runs) ---"
for i in 1 2 3 4 5; do
    echo "TG128 run $i..."
    env $FLAGS_B "$BINARY" "$MODEL" "$FIXTURE" --bench128 > "${OUTDIR}/bench128-B-${i}.log" 2>&1
    grep "benchmark_tokens=" "${OUTDIR}/bench128-B-${i}.log" || cat "${OUTDIR}/bench128-B-${i}.log"
    sleep 3
done

# 5. Candidate M9 TG256 Benchmark (5 runs)
echo "--- 5. Running Candidate M9 TG256 Benchmark (5 runs) ---"
for i in 1 2 3 4 5; do
    echo "TG256 run $i..."
    env $FLAGS_B "$BINARY" "$MODEL" "$FIXTURE" --bench256 > "${OUTDIR}/bench256-B-${i}.log" 2>&1
    grep "benchmark_tokens=" "${OUTDIR}/bench256-B-${i}.log" || cat "${OUTDIR}/bench256-B-${i}.log"
    sleep 3
done

kill -TERM "$SAMPLER_PID" 2>/dev/null || true
trap - EXIT
wait "$SAMPLER_PID" 2>/dev/null || true

echo "--- 6. Computing Benchmark Summary & Qualification Analysis ---"
python3 - << 'PYEOF' "$OUTDIR"
import sys, os, re, statistics

outdir = sys.argv[1]

def parse_runs(prefix, count, tokens):
    toks, lats = [], []
    replays, allocs = [], []
    for i in range(1, count + 1):
        fpath = os.path.join(outdir, f"{prefix}-{i}.log")
        with open(fpath) as f:
            content = f.read()
        m_tok = re.search(r'median_tok_s=([\d.]+)', content)
        m_lat = re.search(r'median_ms=([\d.]+)', content)
        m_rep = re.search(r'replay=(\w+)', content)
        m_alc = re.search(r'allocations_during_decode=(\d+)', content)
        if m_tok and m_lat:
            t = float(m_tok.group(1))
            m = float(m_lat.group(1)) / tokens
            toks.append(t)
            lats.append(m)
        if m_rep: replays.append(m_rep.group(1))
        if m_alc: allocs.append(int(m_alc.group(1)))
    return toks, lats, replays, allocs

toks_a, lats_a, reps_a, alc_a = parse_runs("bench64-A", 5, 64)
toks_b, lats_b, reps_b, alc_b = parse_runs("bench64-B", 5, 64)
toks_128, lats_128, reps_128, alc_128 = parse_runs("bench128-B", 5, 128)
toks_256, lats_256, reps_256, alc_256 = parse_runs("bench256-B", 5, 256)

med_a = statistics.median(toks_a)
med_b = statistics.median(toks_b)
med_lat_a = statistics.median(lats_a)
med_lat_b = statistics.median(lats_b)

med_128 = statistics.median(toks_128)
med_lat_128 = statistics.median(lats_128)
med_256 = statistics.median(toks_256)
med_lat_256 = statistics.median(lats_256)

delta_tok = med_b - med_a
pct_gain = (med_b - med_a) / med_a * 100.0
lat_saved = med_lat_a - med_lat_b

pen_128 = (med_b - med_128) / med_b * 100.0
pen_256 = (med_b - med_256) / med_b * 100.0

print("\n" + "="*75)
print("             MIINFER M9 PRIMARY GATE QUALIFICATION RESULTS")
print("="*75)
print(f"Config A (M8 Baseline TG64):  {med_a:.4f} tok/s | {med_lat_a:.3f} ms/token")
print(f"  Runs (tok/s): {', '.join(f'{x:.2f}' for x in toks_a)}")
print(f"  Allocations: {sum(alc_a)} | Replay: {'ALL PASS' if all(r == 'PASS' for r in reps_a) else 'FAIL'}")
print()
print(f"Config B (M9 Candidate TG64): {med_b:.4f} tok/s | {med_lat_b:.3f} ms/token")
print(f"  Runs (tok/s): {', '.join(f'{x:.2f}' for x in toks_b)}")
print(f"  Allocations: {sum(alc_b)} | Replay: {'ALL PASS' if all(r == 'PASS' for r in reps_b) else 'FAIL'}")
print(f"  Speedup vs M8 Baseline:     {pct_gain:+.2f}% ({delta_tok:+.4f} tok/s, {lat_saved:+.3f} ms saved/tok)")
print()
print(f"Candidate M9 TG128:           {med_128:.4f} tok/s | {med_lat_128:.3f} ms/token")
print(f"  Runs (tok/s): {', '.join(f'{x:.2f}' for x in toks_128)}")
print(f"  Context scaling penalty (TG64 -> TG128): {pen_128:.2f}% (Threshold: <= 1.50%)")
print()
print(f"Candidate M9 TG256:           {med_256:.4f} tok/s | {med_lat_256:.3f} ms/token")
print(f"  Runs (tok/s): {', '.join(f'{x:.2f}' for x in toks_256)}")
print(f"  Context scaling penalty (TG64 -> TG256): {pen_256:.2f}%")
print()

tg64_pass = med_b >= 32.00
tg128_pass = med_128 >= 31.50 and pen_128 <= 1.50
alloc_pass = all(a == 0 for a in alc_b + alc_128 + alc_256)
replay_pass = all(r == 'PASS' for r in reps_b + reps_128 + reps_256)

print("GATE CRITERIA CHECKLIST:")
print(f"  [ {'PASS' if tg64_pass else 'FAIL'} ] Primary Target TG64 >= 32.00 tok/s: {med_b:.4f} tok/s")
print(f"  [ {'PASS' if tg128_pass else 'FAIL'} ] Secondary Target TG128 >= 31.50 tok/s: {med_128:.4f} tok/s (scaling penalty {pen_128:.2f}%)")
print(f"  [ {'PASS' if alloc_pass else 'FAIL'} ] Zero decode allocations: {'PASS' if alloc_pass else 'FAIL'}")
print(f"  [ {'PASS' if replay_pass else 'FAIL'} ] Bit-exact deterministic replay: {'PASS' if replay_pass else 'FAIL'}")
print("="*75 + "\n")
PYEOF

# Telemetry check
echo "--- 7. Hardware Telemetry Audit ---"
python3 - << 'PYEOF' "$OUTDIR"
import sys, os, json

outdir = sys.argv[1]
telem_file = os.path.join(outdir, "telemetry.jsonl")

sclk_vals, mclk_vals, temp_vals, power_vals = [], [], [], []

with open(telem_file) as f:
    for line in f:
        try:
            d = json.loads(line)
            card0 = d.get('rocm_smi', {}).get('card0', {})
            sclk_str = card0.get('sclk clock speed:', '')
            mclk_str = card0.get('mclk clock speed:', '')
            sclk = 1606 if '1606' in sclk_str else (int(sclk_str.strip('()Mhz')) if 'Mhz' in sclk_str else 0)
            mclk = 1000 if '1000' in mclk_str else (int(mclk_str.strip('()Mhz')) if 'Mhz' in mclk_str else 0)
            temp = float(card0.get('Temperature (Sensor edge) (C)', 0))
            pwr = float(card0.get('Current Socket Graphics Package Power (W)', 0))
            sclk_vals.append(sclk)
            mclk_vals.append(mclk)
            temp_vals.append(temp)
            power_vals.append(pwr)
        except Exception:
            pass

if sclk_vals:
    total = len(sclk_vals)
    pct_1606 = sum(1 for s in sclk_vals if s >= 1600) / total * 100.0
    pct_1000 = sum(1 for m in mclk_vals if m >= 1000) / total * 100.0
    print(f"Telemetry Audit ({total} samples @ 250ms interval):")
    print(f"  SCLK >= 1600 MHz locked: {pct_1606:.2f}% ({sum(1 for s in sclk_vals if s >= 1600)}/{total})")
    print(f"  MCLK >= 1000 MHz locked: {pct_1000:.2f}% ({sum(1 for m in mclk_vals if m >= 1000)}/{total})")
    print(f"  Edge Temperature: Avg {sum(temp_vals)/total:.1f} C, Min {min(temp_vals):.1f} C, Max {max(temp_vals):.1f} C")
    print(f"  Package Power:    Avg {sum(power_vals)/total:.1f} W, Min {min(power_vals):.1f} W, Max {max(power_vals):.1f} W")
    if pct_1606 >= 95.0 and pct_1000 >= 95.0:
        print("  [ PASS ] Telemetry Clock Lock >= 95% Verified")
    else:
        print("  [ FAIL ] Hardware clock instability detected")
PYEOF

echo "Milestone M9 qualification completed successfully."
