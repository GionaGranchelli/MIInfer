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

BASE_FLAGS="MIINFER_Q4K_NATIVE_DOWN=1 MIINFER_Q4K_NATIVE_GATE_UP=1 MIINFER_Q4K_NATIVE_Q=1 MIINFER_Q4K_NATIVE_ATTN_GATE=1 MIINFER_Q4K_NATIVE_ATTN_OUT=1 MIINFER_Q4K_NATIVE_K=1 MIINFER_Q5K_NATIVE_SSM_OUT=1"

run_suite() {
    local TOKENS=$1
    local SUITE_NAME="tg${TOKENS}"
    local TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
    local OUTDIR="bench/results/exp0179-q6k-${SUITE_NAME}/${TIMESTAMP}"
    mkdir -p "$OUTDIR"

    echo "=== Running EXP-0179 Native Q6_K Wave64 A/B benchmark for TG${TOKENS} ==="
    echo "Results directory: $OUTDIR"

    # Start telemetry
    scripts/sample-gpu.sh "${OUTDIR}/telemetry.jsonl" 250 &
    local SAMPLER_PID=$!
    trap 'kill -TERM "$SAMPLER_PID" 2>/dev/null || true' EXIT

    for i in 1 2 3 4 5; do
        echo "[Pair $i/5] Running Control (Native Q4K+Q5K, Q6K=0)..."
        env $BASE_FLAGS MIINFER_KQUANT_NATIVE_QKV=0 MIINFER_KQUANT_NATIVE_V=0 MIINFER_Q6K_NATIVE_DOWN=0 \
            "$BINARY" "$MODEL" "$FIXTURE" "--bench${TOKENS}" > "${OUTDIR}/control-${i}.log" 2>&1
        grep "benchmark_tokens=" "${OUTDIR}/control-${i}.log" || cat "${OUTDIR}/control-${i}.log"
        sleep 5

        echo "[Pair $i/5] Running Candidate (Native Q4K+Q5K + Native Q6K=1)..."
        env $BASE_FLAGS MIINFER_KQUANT_NATIVE_QKV=1 MIINFER_KQUANT_NATIVE_V=1 MIINFER_Q6K_NATIVE_DOWN=1 \
            "$BINARY" "$MODEL" "$FIXTURE" "--bench${TOKENS}" > "${OUTDIR}/candidate-${i}.log" 2>&1
        grep "benchmark_tokens=" "${OUTDIR}/candidate-${i}.log" || cat "${OUTDIR}/candidate-${i}.log"
        sleep 5
    done

    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    trap - EXIT
    wait "$SAMPLER_PID" 2>/dev/null || true

    echo "=== Summary for TG${TOKENS} ==="
    echo "Control runs (Q6K=0):"
    for i in 1 2 3 4 5; do
        awk -F' ' '{print "  Run " '"$i"' ": " $0}' "${OUTDIR}/control-${i}.log" | grep "benchmark_tokens="
    done
    echo "Candidate runs (Q6K=1):"
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

ctrl_toks, ctrl_ms = [], []
cand_toks, cand_ms = [], []

for i in range(1, 6):
    t, m = extract(f'${OUTDIR}/control-{i}.log')
    ctrl_toks.append(t)
    ctrl_ms.append(m)
    t, m = extract(f'${OUTDIR}/candidate-{i}.log')
    cand_toks.append(t)
    cand_ms.append(m)

ctrl_toks_mean = statistics.mean(ctrl_toks)
cand_toks_mean = statistics.mean(cand_toks)
ctrl_ms_mean = statistics.mean(ctrl_ms)
cand_ms_mean = statistics.mean(cand_ms)

toks_delta_pct = (cand_toks_mean - ctrl_toks_mean) / ctrl_toks_mean * 100
ms_delta = cand_ms_mean - ctrl_ms_mean

print(f'\n--- Statistics for TG${TOKENS} ---')
print(f'Control Tok/s:   mean={ctrl_toks_mean:.4f} median={statistics.median(ctrl_toks):.4f} min={min(ctrl_toks):.4f} max={max(ctrl_toks):.4f}')
print(f'Candidate Tok/s: mean={cand_toks_mean:.4f} median={statistics.median(cand_toks):.4f} min={min(cand_toks):.4f} max={max(cand_toks):.4f}')
print(f'Throughput Delta: {toks_delta_pct:+.2f}%')
print(f'Control GPU ms:   mean={ctrl_ms_mean:.4f} ms/token')
print(f'Candidate GPU ms: mean={cand_ms_mean:.4f} ms/token')
print(f'GPU ms Delta:     {ms_delta:+.4f} ms/token (saving = {-ms_delta:.4f} ms/token)')
"

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
