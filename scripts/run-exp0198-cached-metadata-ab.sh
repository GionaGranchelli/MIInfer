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

BASE_FLAGS="MIINFER_Q4K_NATIVE_DOWN=1 MIINFER_Q4K_NATIVE_GATE_UP=1 MIINFER_Q4K_NATIVE_Q=1 MIINFER_Q4K_NATIVE_ATTN_GATE=1 MIINFER_Q4K_NATIVE_ATTN_OUT=1 MIINFER_Q4K_NATIVE_K=1 MIINFER_Q5K_NATIVE_SSM_OUT=1 MIINFER_KQUANT_NATIVE_QKV=1 MIINFER_KQUANT_NATIVE_V=1 MIINFER_Q6K_NATIVE_DOWN=1 MIINFER_HIP_GRAPH=1 MIINFER_FUSED_GATE_UP_SWIGLU=1 MIINFER_FUSED_RECURRENT_CORE=1 MIINFER_FAST_ARGMAX=1 MIINFER_TILED_ONLINE_ATTENTION=1 MIINFER_FUSED_ROPE_NORM=1 MIINFER_FUSED_ADD_RMS_NORM=1 MIINFER_FUSED_INTERLAYER_NORM=1 MIINFER_Q6K_NATIVE_LM_HEAD=1 MIINFER_COMBINED_QKV_GATE=1 MIINFER_COMBINED_ATTN_QK=1 MIINFER_SWIGLU_PAIRED=1"

run_suite() {
    local TOKENS=$1
    local SUITE_NAME="tg${TOKENS}"
    local TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
    local OUTDIR="bench/results/exp0198-cached-metadata-${SUITE_NAME}/${TIMESTAMP}"
    mkdir -p "$OUTDIR"

    echo "=== Running EXP-0198 Register-Cached Metadata A/B benchmark for TG${TOKENS} ==="
    echo "Results directory: $OUTDIR"

    # Start telemetry
    scripts/sample-gpu.sh "${OUTDIR}/telemetry.jsonl" 250 &
    local SAMPLER_PID=$!
    trap 'kill -TERM "$SAMPLER_PID" 2>/dev/null || true' EXIT

    for i in 1 2 3 4 5; do
        echo "[Pair $i/5] Running Control (EXP-0197 Baseline: Uncached Global Pointer Metadata)..."
        env $BASE_FLAGS MIINFER_KQUANT_FAST_ARITH=0 MIINFER_Q6K_SIMD_UNPACK=0 \
            "$BINARY" "$MODEL" "$FIXTURE" "--bench${TOKENS}" > "${OUTDIR}/control-${i}.log" 2>&1
        grep "benchmark_tokens=" "${OUTDIR}/control-${i}.log" || cat "${OUTDIR}/control-${i}.log"
        sleep 5

        echo "[Pair $i/5] Running Candidate (EXP-0198: Register-Cached Tile Metadata Decoders)..."
        env $BASE_FLAGS MIINFER_KQUANT_FAST_ARITH=1 MIINFER_Q6K_SIMD_UNPACK=1 \
            "$BINARY" "$MODEL" "$FIXTURE" "--bench${TOKENS}" > "${OUTDIR}/candidate-${i}.log" 2>&1
        grep "benchmark_tokens=" "${OUTDIR}/candidate-${i}.log" || cat "${OUTDIR}/candidate-${i}.log"
        sleep 5
    done

    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    trap - EXIT
    wait "$SAMPLER_PID" 2>/dev/null || true

    echo "=== Summary for TG${TOKENS} ==="
    echo "Control runs (Uncached Global Pointer Metadata):"
    for i in 1 2 3 4 5; do
        awk -F' ' '{print "  Run " '"$i"' ": " $0}' "${OUTDIR}/control-${i}.log" | grep "benchmark_tokens="
    done
    echo "Candidate runs (EXP-0198 Register-Cached Metadata Decoders):"
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

pct_speedup = ((med_cand_tok - med_ctrl_tok) / med_ctrl_tok) * 100.0
lat_saved_ms = med_ctrl_lat - med_cand_lat

wins = sum(1 for c, k in zip(ctrl_lat, cand_lat) if k < c)

print('--------------------------------------------------')
print(f'TG${TOKENS} Control Median:   {med_ctrl_tok:.4f} tok/s  ({med_ctrl_lat:.3f} ms/tok)')
print(f'TG${TOKENS} Candidate Median: {med_cand_tok:.4f} tok/s  ({med_cand_lat:.3f} ms/tok)')
print(f'Delta: {pct_speedup:+.2f}% throughput ({lat_saved_ms:+.3f} ms/tok saved)')
print(f'Pairwise wins: {wins}/5')
print('--------------------------------------------------')
"

    # Telemetry check
    if [[ -f "${OUTDIR}/telemetry.jsonl" ]]; then
        python3 -c "
import json, re

with open('${OUTDIR}/telemetry.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

sclks, mclks, temps, powers = [], [], [], []
for entry in lines:
    card = entry.get('rocm_smi', {}).get('card0', {})
    sclk_str = card.get('sclk clock speed:', '')
    mclk_str = card.get('mclk clock speed:', '')
    temp_str = card.get('Temperature (Sensor edge) (C)', '')
    power_str = card.get('Current Socket Graphics Package Power (W)', '')
    m_sclk = re.search(r'\((\d+)Mhz\)', sclk_str)
    m_mclk = re.search(r'\((\d+)Mhz\)', mclk_str)
    if m_sclk: sclks.append(int(m_sclk.group(1)))
    if m_mclk: mclks.append(int(m_mclk.group(1)))
    if temp_str: temps.append(float(temp_str))
    if power_str: powers.append(float(power_str))

if sclks:
    locked_1606 = sum(1 for s in sclks if s == 1606)
    pct = (locked_1606 / len(sclks)) * 100.0
    print(f'Telemetry: {len(sclks)} samples ({pct:.2f}% locked at 1606MHz)')
    print(f'SCLK min/max: {min(sclks)}/{max(sclks)} MHz, MCLK min/max: {min(mclks)}/{max(mclks)} MHz')
    print(f'Temp min/max/mean: {min(temps):.1f} / {max(temps):.1f} / {sum(temps)/len(temps):.1f} C')
    print(f'Power min/max/mean: {min(powers):.1f} / {max(powers):.1f} / {sum(powers)/len(powers):.1f} W')
"
    fi
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
        echo ""
        sleep 10
        run_suite 128
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 1
        ;;
esac
