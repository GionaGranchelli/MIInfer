#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'usage: %s [tg64|tg128|both] [MODEL_PATH]\n' "$0" >&2
    exit 2
fi

MODE=$1
MODEL=${2:-/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf}
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

BASE_FLAGS="MIINFER_Q4K_NATIVE_DOWN=1 MIINFER_Q4K_NATIVE_GATE_UP=1 MIINFER_Q4K_NATIVE_Q=1 MIINFER_Q4K_NATIVE_ATTN_GATE=1 MIINFER_Q4K_NATIVE_ATTN_OUT=1 MIINFER_Q4K_NATIVE_K=1 MIINFER_Q5K_NATIVE_SSM_OUT=1 MIINFER_KQUANT_NATIVE_QKV=1 MIINFER_KQUANT_NATIVE_V=1 MIINFER_Q6K_NATIVE_DOWN=1 MIINFER_HIP_GRAPH=1 MIINFER_FUSED_GATE_UP_SWIGLU=1"

run_suite() {
    local TOKENS=$1
    local SUITE_NAME="tg${TOKENS}"
    local TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
    local OUTDIR="bench/results/exp0183-fused-recurrent-${SUITE_NAME}/${TIMESTAMP}"
    mkdir -p "$OUTDIR"

    echo "=== Running EXP-0183 Fused Recurrent Core A/B benchmark for TG${TOKENS} ==="
    echo "Results directory: $OUTDIR"

    # Start telemetry
    scripts/sample-gpu.sh "${OUTDIR}/telemetry.jsonl" 250 &
    local SAMPLER_PID=$!
    trap 'kill -TERM "$SAMPLER_PID" 2>/dev/null || true' EXIT

    for i in 1 2 3 4 5; do
        echo "[Pair $i/5] Running Control (EXP-0182, FUSED_RECURRENT_CORE=0)..."
        env $BASE_FLAGS MIINFER_FUSED_RECURRENT_CORE=0 \
            "$BINARY" "$MODEL" "$FIXTURE" "--bench${TOKENS}" > "${OUTDIR}/control-${i}.log" 2>&1
        grep "benchmark_tokens=" "${OUTDIR}/control-${i}.log" || cat "${OUTDIR}/control-${i}.log"
        sleep 5

        echo "[Pair $i/5] Running Candidate (EXP-0183, FUSED_RECURRENT_CORE=1)..."
        env $BASE_FLAGS MIINFER_FUSED_RECURRENT_CORE=1 \
            "$BINARY" "$MODEL" "$FIXTURE" "--bench${TOKENS}" > "${OUTDIR}/candidate-${i}.log" 2>&1
        grep "benchmark_tokens=" "${OUTDIR}/candidate-${i}.log" || cat "${OUTDIR}/candidate-${i}.log"
        sleep 5
    done

    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    trap - EXIT
    wait "$SAMPLER_PID" 2>/dev/null || true

    echo "=== Summary for TG${TOKENS} ==="
    echo "Control runs (FUSED_RECURRENT_CORE=0):"
    for i in 1 2 3 4 5; do
        awk -F' ' '{print "  Run " '"$i"' ": " $0}' "${OUTDIR}/control-${i}.log" | grep "benchmark_tokens="
    done
    echo "Candidate runs (FUSED_RECURRENT_CORE=1):"
    for i in 1 2 3 4 5; do
        awk -F' ' '{print "  Run " '"$i"' ": " $0}' "${OUTDIR}/candidate-${i}.log" | grep "benchmark_tokens="
    done

    # Compute statistics
    python3 -c "
import re, statistics

def extract(path):
    with open(path) as f:
        content = f.read()
    m_tok_s = re.search(r'median_tok_s=([\d.]+)', content)
    m_ms = re.search(r'median_ms=([\d.]+)', content)
    tok_s = float(m_tok_s.group(1))
    ms = float(m_ms.group(1))
    gpu_ms = ms / int('${TOKENS}')
    return tok_s, gpu_ms

ctrl_toks = []
ctrl_lat = []
cand_toks = []
cand_lat = []

for i in range(1, 6):
    ct, cl = extract(f'${OUTDIR}/control-{i}.log')
    ctrl_toks.append(ct)
    ctrl_lat.append(cl)
    kt, kl = extract(f'${OUTDIR}/candidate-{i}.log')
    cand_toks.append(kt)
    cand_lat.append(kl)

med_ctrl_tok = statistics.median(ctrl_toks)
med_cand_tok = statistics.median(cand_toks)
med_ctrl_lat = statistics.median(ctrl_lat)
med_cand_lat = statistics.median(cand_lat)

pct_tps = (med_cand_tok - med_ctrl_tok) / med_ctrl_tok * 100.0
lat_saved = med_ctrl_lat - med_cand_lat
pct_lat = (med_ctrl_lat - med_cand_lat) / med_ctrl_lat * 100.0

print('---------------------------------------------------------')
print('Statistical Analysis (5 repeated benchmark runs):')
print(f'Control  (RECURRENT=0): {med_ctrl_tok:.4f} tok/s | {med_ctrl_lat:.3f} ms/token (median)')
print(f'Candidate(RECURRENT=1): {med_cand_tok:.4f} tok/s | {med_cand_lat:.3f} ms/token (median)')
print(f'Throughput Delta:       {pct_tps:+.2f}%')
print(f'Latency Saved:          {lat_saved:+.3f} ms/token ({pct_lat:+.2f}%)')
print('---------------------------------------------------------')
"

    # Analyze telemetry
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
}

case "$MODE" in
    tg64)
        run_suite 64
        ;;
    tg128)
        run_suite 128
        ;;
    both)
        run_suite 64
        run_suite 128
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 1
        ;;
esac
