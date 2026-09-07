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

FLAGS="MIINFER_Q4K_NATIVE_DOWN=1 MIINFER_Q4K_NATIVE_GATE_UP=1 MIINFER_Q4K_NATIVE_Q=1 MIINFER_Q4K_NATIVE_ATTN_GATE=1 MIINFER_Q4K_NATIVE_ATTN_OUT=1 MIINFER_Q4K_NATIVE_K=1 MIINFER_Q5K_NATIVE_SSM_OUT=1 MIINFER_KQUANT_NATIVE_QKV=1 MIINFER_KQUANT_NATIVE_V=1 MIINFER_Q6K_NATIVE_DOWN=1 MIINFER_HIP_GRAPH=1 MIINFER_FUSED_GATE_UP_SWIGLU=1 MIINFER_FUSED_RECURRENT_CORE=1 MIINFER_FAST_ARGMAX=1 MIINFER_TILED_ONLINE_ATTENTION=1 MIINFER_FUSED_ROPE_NORM=1 MIINFER_FUSED_ADD_RMS_NORM=1 MIINFER_FUSED_INTERLAYER_NORM=1 MIINFER_Q6K_NATIVE_LM_HEAD=1"

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR="bench/results/m8a-requal-m7/${TIMESTAMP}"
mkdir -p "$OUTDIR"

echo "=== M8-A: Re-qualifying M7 Baseline (commit 9f20274) ==="
echo "Output directory: $OUTDIR"

# 1. Correctness: 16-token generation
echo "--- Running 16-token generation test ---"
env $FLAGS "$BINARY" "$MODEL" "$FIXTURE" --generate16 > "${OUTDIR}/generate16.log" 2>&1
grep "generated_tokens=" "${OUTDIR}/generate16.log" || cat "${OUTDIR}/generate16.log"

# 2. Correctness: Observable Contract
echo "--- Running 64-layer observable contract ---"
env $FLAGS "$BINARY" "$MODEL" "$FIXTURE" --prefix64-observable-contract > "${OUTDIR}/observable_contract.log" 2>&1
grep "observable position=64" "${OUTDIR}/observable_contract.log" || cat "${OUTDIR}/observable_contract.log"

# Start hardware telemetry
scripts/sample-gpu.sh "${OUTDIR}/telemetry.jsonl" 250 &
SAMPLER_PID=$!
trap 'kill -TERM "$SAMPLER_PID" 2>/dev/null || true' EXIT

# 3. Benchmark TG64 (5 runs)
echo "--- Benchmarking TG64 (5 runs) ---"
for i in 1 2 3 4 5; do
    echo "TG64 run $i..."
    env $FLAGS "$BINARY" "$MODEL" "$FIXTURE" --bench64 > "${OUTDIR}/bench64-${i}.log" 2>&1
    grep "benchmark_tokens=" "${OUTDIR}/bench64-${i}.log" || cat "${OUTDIR}/bench64-${i}.log"
    sleep 5
done

# 4. Benchmark TG128 (5 runs)
echo "--- Benchmarking TG128 (5 runs) ---"
for i in 1 2 3 4 5; do
    echo "TG128 run $i..."
    env $FLAGS "$BINARY" "$MODEL" "$FIXTURE" --bench128 > "${OUTDIR}/bench128-${i}.log" 2>&1
    grep "benchmark_tokens=" "${OUTDIR}/bench128-${i}.log" || cat "${OUTDIR}/bench128-${i}.log"
    sleep 5
done

# 5. Benchmark TG256 (5 runs)
echo "--- Benchmarking TG256 (5 runs) ---"
for i in 1 2 3 4 5; do
    echo "TG256 run $i..."
    env $FLAGS "$BINARY" "$MODEL" "$FIXTURE" --bench256 > "${OUTDIR}/bench256-${i}.log" 2>&1
    grep "benchmark_tokens=" "${OUTDIR}/bench256-${i}.log" || cat "${OUTDIR}/bench256-${i}.log"
    sleep 5
done

kill -TERM "$SAMPLER_PID" 2>/dev/null || true
trap - EXIT
wait "$SAMPLER_PID" 2>/dev/null || true

# Summary Statistics
python3 -c "
import re, statistics

def summarize(suite, n_toks):
    toks = []
    lats = []
    for i in range(1, 6):
        with open(f'${OUTDIR}/{suite}-{i}.log') as f:
            content = f.read()
        m_t = re.search(r'median_tok_s=([\d.]+)', content)
        m_m = re.search(r'median_ms=([\d.]+)', content)
        t = float(m_t.group(1))
        m = float(m_m.group(1)) / n_toks
        toks.append(t)
        lats.append(m)
    med_t = statistics.median(toks)
    med_l = statistics.median(lats)
    print(f'{suite.upper()}: {med_t:.4f} tok/s | {med_l:.3f} ms/token (min: {min(toks):.4f}, max: {max(toks):.4f})')
    return med_t, med_l

print('=== M8-A Re-qualification Summary ===')
t64, l64 = summarize('bench64', 64)
t128, l128 = summarize('bench128', 128)
t256, l256 = summarize('bench256', 256)
pen_128 = (t64 - t128) / t64 * 100.0
pen_256 = (t64 - t256) / t64 * 100.0
print(f'TG64 -> TG128 scaling delta: {pen_128:+.2f}%')
print(f'TG64 -> TG256 scaling delta: {pen_256:+.2f}%')
"

# Telemetry Analysis
python3 -c "
import json

sclk_vals = []
temp_vals = []
power_vals = []

with open('${OUTDIR}/telemetry.jsonl') as f:
    for line in f:
        try:
            d = json.loads(line)
            card0 = d.get('rocm_smi', {}).get('card0', {})
            sclk_str = card0.get('sclk clock speed:', '')
            sclk = 1606 if '1606' in sclk_str else (int(sclk_str.strip('()Mhz')) if 'Mhz' in sclk_str else 0)
            temp = float(card0.get('Temperature (Sensor edge) (C)', 0))
            pwr = float(card0.get('Current Socket Graphics Package Power (W)', 0))
            sclk_vals.append(sclk)
            temp_vals.append(temp)
            power_vals.append(pwr)
        except Exception:
            pass

if sclk_vals:
    pct_1606 = sum(1 for s in sclk_vals if s >= 1600) / len(sclk_vals) * 100
    print(f'Telemetry: {len(sclk_vals)} samples | SCLK >= 1600MHz: {pct_1606:.1f}% | Avg Temp: {sum(temp_vals)/len(temp_vals):.1f}C (Max: {max(temp_vals):.1f}C) | Avg Power: {sum(power_vals)/len(power_vals):.1f}W (Max: {max(power_vals):.1f}W)')
"
