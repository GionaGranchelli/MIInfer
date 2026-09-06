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

run_suite() {
    local TOKENS=$1
    local SUITE_NAME="tg${TOKENS}"
    local TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
    local OUTDIR="bench/results/exp0177-k-${SUITE_NAME}/${TIMESTAMP}"
    mkdir -p "$OUTDIR"

    echo "=== Running EXP-0177 Candidate 5 A/B benchmark for TG${TOKENS} ==="
    echo "Results directory: $OUTDIR"

    # Start telemetry
    scripts/sample-gpu.sh "${OUTDIR}/telemetry.jsonl" 250 &
    local SAMPLER_PID=$!
    trap 'kill -TERM "$SAMPLER_PID" 2>/dev/null || true' EXIT

    for i in 1 2 3 4 5; do
        echo "[Pair $i/5] Running Control (Down+GateUp+Q+AttnGate+AttnOut, K=0)..."
        MIINFER_Q4K_NATIVE_DOWN=1 MIINFER_Q4K_NATIVE_GATE_UP=1 MIINFER_Q4K_NATIVE_Q=1 MIINFER_Q4K_NATIVE_ATTN_GATE=1 MIINFER_Q4K_NATIVE_ATTN_OUT=1 MIINFER_Q4K_NATIVE_K=0 \
            "$BINARY" "$MODEL" "$FIXTURE" "--bench${TOKENS}" > "${OUTDIR}/control-${i}.log" 2>&1
        grep "benchmark_tokens=" "${OUTDIR}/control-${i}.log" || cat "${OUTDIR}/control-${i}.log"
        sleep 5

        echo "[Pair $i/5] Running Candidate (Down+GateUp+Q+AttnGate+AttnOut+K)..."
        MIINFER_Q4K_NATIVE_DOWN=1 MIINFER_Q4K_NATIVE_GATE_UP=1 MIINFER_Q4K_NATIVE_Q=1 MIINFER_Q4K_NATIVE_ATTN_GATE=1 MIINFER_Q4K_NATIVE_ATTN_OUT=1 MIINFER_Q4K_NATIVE_K=1 \
            "$BINARY" "$MODEL" "$FIXTURE" "--bench${TOKENS}" > "${OUTDIR}/native-${i}.log" 2>&1
        grep "benchmark_tokens=" "${OUTDIR}/native-${i}.log" || cat "${OUTDIR}/native-${i}.log"
        sleep 5
    done

    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    trap - EXIT
    wait "$SAMPLER_PID" 2>/dev/null || true

    echo "=== Summary for TG${TOKENS} ==="
    echo "Control runs (K=0):"
    for i in 1 2 3 4 5; do
        awk -F' ' '{print "  Run " '"$i"' ": " $0}' "${OUTDIR}/control-${i}.log" | grep "benchmark_tokens="
    done
    echo "Native runs (K=1):"
    for i in 1 2 3 4 5; do
        awk -F' ' '{print "  Run " '"$i"' ": " $0}' "${OUTDIR}/native-${i}.log" | grep "benchmark_tokens="
    done

    # Telemetry check
    echo "Telemetry summary:"
    python3 -c "
import json, sys

total = 0
sclk_1606 = 0
mclk_1000 = 0
max_edge = 0
max_junc = 0
max_mem = 0

with open('${OUTDIR}/telemetry.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
            rs = d.get('rocm_smi', {})
            card = rs.get('card0', {})
            sclk_str = card.get('sclk clock speed:', '')
            mclk_str = card.get('mclk clock speed:', '')
            edge = card.get('Temperature (Sensor edge) (C)')
            junc = card.get('Temperature (Sensor junction) (C)')
            mem = card.get('Temperature (Sensor memory) (C)')

            total += 1
            if '1606' in str(sclk_str): sclk_1606 += 1
            if '1000' in str(mclk_str): mclk_1000 += 1
            if edge is not None: max_edge = max(max_edge, float(edge))
            if junc is not None: max_junc = max(max_junc, float(junc))
            if mem is not None: max_mem = max(max_mem, float(mem))
        except Exception:
            pass

if total > 0:
    print(f'  Total samples: {total}')
    print(f'  SCLK 1606MHz residency: {sclk_1606}/{total} ({sclk_1606/total*100:.1f}%)')
    print(f'  MCLK 1000MHz residency: {mclk_1000}/{total} ({mclk_1000/total*100:.1f}%)')
    print(f'  Max edge temp: {max_edge:.1f} C')
    print(f'  Max junction temp: {max_junc:.1f} C')
    print(f'  Max memory temp: {max_mem:.1f} C')
"
}

if [[ "$MODE" == "tg64" || "$MODE" == "both" ]]; then
    run_suite 64
fi

if [[ "$MODE" == "tg128" || "$MODE" == "both" ]]; then
    run_suite 128
fi
