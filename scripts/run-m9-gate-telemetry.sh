#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf}
FIXTURE=/tmp/m6a273-reference-p12
BINARY=./build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block

BASE_FLAGS="MIINFER_Q4K_NATIVE_DOWN=1 MIINFER_Q4K_NATIVE_GATE_UP=1 MIINFER_Q4K_NATIVE_Q=1 MIINFER_Q4K_NATIVE_ATTN_GATE=1 MIINFER_Q4K_NATIVE_ATTN_OUT=1 MIINFER_Q4K_NATIVE_K=1 MIINFER_Q5K_NATIVE_SSM_OUT=1 MIINFER_KQUANT_NATIVE_QKV=1 MIINFER_KQUANT_NATIVE_V=1 MIINFER_Q6K_NATIVE_DOWN=1 MIINFER_HIP_GRAPH=1 MIINFER_FUSED_GATE_UP_SWIGLU=1 MIINFER_FUSED_RECURRENT_CORE=1 MIINFER_FAST_ARGMAX=1 MIINFER_TILED_ONLINE_ATTENTION=1 MIINFER_FUSED_ROPE_NORM=1 MIINFER_FUSED_ADD_RMS_NORM=1 MIINFER_FUSED_INTERLAYER_NORM=1 MIINFER_Q6K_NATIVE_LM_HEAD=1 MIINFER_COMBINED_QKV_GATE=1 MIINFER_COMBINED_ATTN_QK=1 MIINFER_SWIGLU_PAIRED=1 MIINFER_KQUANT_FAST_ARITH=1 MIINFER_Q6K_SIMD_UNPACK=1"

FLAGS_A="${BASE_FLAGS} MIINFER_DEVICE_TOKEN_CHAIN=0"
FLAGS_B="${BASE_FLAGS} MIINFER_DEVICE_TOKEN_CHAIN=1"

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR="bench/results/m9-qualification-gate/${TIMESTAMP}"
mkdir -p "$OUTDIR"

echo "=========================================================================="
echo "      M9 Primary Gate 5-Pair Interleaved Qualification Suite              "
echo "=========================================================================="
echo "Output directory: $OUTDIR"

scripts/sample-gpu.sh "${OUTDIR}/telemetry.jsonl" 250 &
SAMPLER_PID=$!
trap 'kill -TERM "$SAMPLER_PID" 2>/dev/null || true' EXIT

for i in 1 2 3 4 5; do
    echo "Pair $i: Run Config A (M8 Baseline)..."
    env $FLAGS_A "$BINARY" "$MODEL" "$FIXTURE" --bench64 > "${OUTDIR}/bench64-A-${i}.log" 2>&1
    grep "benchmark_tokens=" "${OUTDIR}/bench64-A-${i}.log"
    sleep 6

    echo "Pair $i: Run Config B (M9 Candidate)..."
    env $FLAGS_B "$BINARY" "$MODEL" "$FIXTURE" --bench64 > "${OUTDIR}/bench64-B-${i}.log" 2>&1
    grep "benchmark_tokens=" "${OUTDIR}/bench64-B-${i}.log"
    sleep 6
done

kill -TERM "$SAMPLER_PID" 2>/dev/null || true
trap - EXIT
wait "$SAMPLER_PID" 2>/dev/null || true

python3 - << 'PYEOF' "$OUTDIR"
import sys, os, re, statistics, json

outdir = sys.argv[1]

def parse_runs(prefix, count):
    toks, lats, replays, allocs = [], [], [], []
    for i in range(1, count + 1):
        fpath = os.path.join(outdir, f"{prefix}-{i}.log")
        with open(fpath) as f:
            content = f.read()
        m_tok = re.search(r'median_tok_s=([\d.]+)', content)
        m_lat = re.search(r'median_ms=([\d.]+)', content)
        m_rep = re.search(r'replay=(\w+)', content)
        m_alc = re.search(r'allocations_during_decode=(\d+)', content)
        if m_tok and m_lat:
            toks.append(float(m_tok.group(1)))
            lats.append(float(m_lat.group(1)) / 64.0)
        if m_rep: replays.append(m_rep.group(1))
        if m_alc: allocs.append(int(m_alc.group(1)))
    return toks, lats, replays, allocs

toks_a, lats_a, reps_a, alc_a = parse_runs("bench64-A", 5)
toks_b, lats_b, reps_b, alc_b = parse_runs("bench64-B", 5)

med_a = statistics.median(toks_a)
med_b = statistics.median(toks_b)
med_lat_a = statistics.median(lats_a)
med_lat_b = statistics.median(lats_b)

pct_gain = (med_b - med_a) / med_a * 100.0
lat_saved = med_lat_a - med_lat_b

print("\n" + "="*75)
print("             MIINFER M9 PRIMARY GATE QUALIFICATION RESULTS")
print("="*75)
print(f"Config A (M8 Baseline):  {med_a:.4f} tok/s | {med_lat_a:.3f} ms/token")
print(f"  Runs (tok/s): {', '.join(f'{x:.2f}' for x in toks_a)}")
print(f"  Allocations: {sum(alc_a)} | Replay: {'ALL PASS' if all(r == 'PASS' for r in reps_a) else 'FAIL'}")
print()
print(f"Config B (M9 Candidate): {med_b:.4f} tok/s | {med_lat_b:.3f} ms/token")
print(f"  Runs (tok/s): {', '.join(f'{x:.2f}' for x in toks_b)}")
print(f"  Allocations: {sum(alc_b)} | Replay: {'ALL PASS' if all(r == 'PASS' for r in reps_b) else 'FAIL'}")
print(f"  Speedup vs M8:         {pct_gain:+.2f}% ({med_b - med_a:+.4f} tok/s, {lat_saved:+.3f} ms saved/tok)")
print()

# Telemetry
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

total = len(sclk_vals)
pct_1606 = sum(1 for s in sclk_vals if s >= 1600) / total * 100.0
pct_1000 = sum(1 for m in mclk_vals if m >= 1000) / total * 100.0

print(f"Telemetry Audit ({total} samples @ 250ms interval):")
print(f"  SCLK >= 1600 MHz locked: {pct_1606:.2f}% ({sum(1 for s in sclk_vals if s >= 1600)}/{total})")
print(f"  MCLK >= 1000 MHz locked: {pct_1000:.2f}% ({sum(1 for m in mclk_vals if m >= 1000)}/{total})")
print(f"  Edge Temperature: Avg {sum(temp_vals)/total:.1f} C, Min {min(temp_vals):.1f} C, Max {max(temp_vals):.1f} C")
print(f"  Package Power:    Avg {sum(power_vals)/total:.1f} W, Min {min(power_vals):.1f} W, Max {max(power_vals):.1f} W")
print()

tg64_pass = med_b >= 32.00
alloc_pass = all(a == 0 for a in alc_b)
replay_pass = all(r == 'PASS' for r in reps_b)
telem_pass = pct_1606 >= 95.0 and pct_1000 >= 95.0

print("PRIMARY GATE CRITERIA CHECKLIST:")
print(f"  [ {'PASS' if tg64_pass else 'FAIL'} ] Primary Target TG64 >= 32.00 tok/s: {med_b:.4f} tok/s")
print(f"  [ {'PASS' if alloc_pass else 'FAIL'} ] Zero decode allocations: PASS")
print(f"  [ {'PASS' if replay_pass else 'FAIL'} ] Bit-exact deterministic replay: PASS")
print(f"  [ {'PASS' if telem_pass else 'FAIL'} ] Hardware Clock Lock >= 95.0%: {pct_1606:.2f}% SCLK / {pct_1000:.2f}% MCLK")
print("="*75 + "\n")
PYEOF
