#!/usr/bin/env bash
set -euo pipefail

MODEL="/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf"
FIXTURE="/tmp/m6a273-reference-p12"

BIN_PINNED="/home/fedora-workstation/Development/mi50-artifacts/m6b0-llama-c0bc8591/bin/llama-bench"
BIN_UPSTREAM="/home/fedora-workstation/Development/upstream-llama-build/bin/llama-bench"
BIN_MX="/home/fedora-workstation/Development/mx-llama-build/bin/llama-bench"
BIN_MIINFER="./build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block"

for b in "$BIN_PINNED" "$BIN_UPSTREAM" "$BIN_MX" "$BIN_MIINFER"; do
    if [[ ! -x "$b" ]]; then
        echo "Binary not found or not executable: $b" >&2
        exit 1
    fi
done

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR="bench/results/m7-frontier/${TIMESTAMP}"
mkdir -p "$OUTDIR"

echo "=== M7 Competitive Frontier Benchmark ==="
echo "Model: $MODEL"
echo "Results directory: $OUTDIR"

# Start continuous telemetry
scripts/sample-gpu.sh "${OUTDIR}/telemetry.jsonl" 250 &
SAMPLER_PID=$!
trap 'kill -TERM "$SAMPLER_PID" 2>/dev/null || true' EXIT

run_llama_candidate() {
    local NAME=$1
    local BIN=$2
    local EXTRA_ARGS=$3
    local TOKENS=$4
    local LOG="${OUTDIR}/${NAME}-tg${TOKENS}.log"

    echo "Running ${NAME} (TG${TOKENS})..."
    "$BIN" -m "$MODEL" -r 5 -p 0 -n "$TOKENS" -b 2048 -ub 512 -ngl 999 -t 24 -ctk f16 -ctv f16 -fa on -o jsonl $EXTRA_ARGS > "$LOG" 2>&1
    grep -E "avg_ts|samples_ts" "$LOG" || cat "$LOG"
    sleep 5
}

run_miinfer() {
    local TOKENS=$1
    local LOG="${OUTDIR}/miinfer-exp0179-tg${TOKENS}.log"

    echo "Running MIInfer EXP-0179 (TG${TOKENS})..."
    MIINFER_Q4K_NATIVE_DOWN=1 \
    MIINFER_Q4K_NATIVE_GATE_UP=1 \
    MIINFER_Q4K_NATIVE_Q=1 \
    MIINFER_Q4K_NATIVE_ATTN_GATE=1 \
    MIINFER_Q4K_NATIVE_ATTN_OUT=1 \
    MIINFER_Q4K_NATIVE_K=1 \
    MIINFER_Q5K_NATIVE_SSM_OUT=1 \
    MIINFER_KQUANT_NATIVE_QKV=1 \
    MIINFER_KQUANT_NATIVE_V=1 \
    MIINFER_Q6K_NATIVE_DOWN=1 \
        "$BIN_MIINFER" "$MODEL" "$FIXTURE" "--bench${TOKENS}" > "$LOG" 2>&1
    grep "benchmark_tokens=" "$LOG" || cat "$LOG"
    sleep 5
}

for TOKENS in 64 128 256; do
    echo "============================================="
    echo "       BENCHMARK REGIME: TG${TOKENS}"
    echo "============================================="

    # 1. Pinned Vanilla llama.cpp (c0bc8591)
    run_llama_candidate "llama-pinned-c0bc859" "$BIN_PINNED" "" "$TOKENS"

    # 2. Upstream llama.cpp (73a43d1f)
    run_llama_candidate "llama-upstream-73a43d1" "$BIN_UPSTREAM" "" "$TOKENS"

    # 3. mx-llama.cpp (2e9d29fe) with native repack
    run_llama_candidate "mx-llama-repack" "$BIN_MX" "" "$TOKENS"

    # 4. mx-llama.cpp (2e9d29fe) without repack (to isolate repack delta)
    run_llama_candidate "mx-llama-norepack" "$BIN_MX" "--no-repack 1" "$TOKENS"

    # 5. MIInfer EXP-0179
    run_miinfer "$TOKENS"
done

kill -TERM "$SAMPLER_PID" 2>/dev/null || true
trap - EXIT
wait "$SAMPLER_PID" 2>/dev/null || true

echo "=== Processing Benchmark Results ==="

python3 -c "
import json, re, statistics, glob

OUTDIR = '${OUTDIR}'
TOKENS_LIST = [64, 128, 256]

CANDIDATES = [
    ('llama-pinned-c0bc859', 'Pinned llama.cpp (c0bc8591)'),
    ('llama-upstream-73a43d1', 'Upstream llama.cpp (73a43d1f)'),
    ('mx-llama-norepack', 'mx-llama.cpp (no repack)'),
    ('mx-llama-repack', 'mx-llama.cpp (repack) [GFX906 Frontier]'),
    ('miinfer-exp0179', 'MIInfer EXP-0179 (Native Wave64)'),
]

results = {}

for name, label in CANDIDATES:
    results[name] = {}
    for tok in TOKENS_LIST:
        log_path = f'{OUTDIR}/{name}-tg{tok}.log'
        try:
            with open(log_path) as f:
                content = f.read()
            if 'miinfer' in name:
                m_tok_s = re.search(r'median_tok_s=([\d.]+)', content)
                m_ms = re.search(r'median_ms=([\d.]+)', content)
                tok_s = float(m_tok_s.group(1))
                ms_total = float(m_ms.group(1))
                ms_tok = ms_total / tok
                results[name][tok] = {'tok_s': tok_s, 'ms_tok': ms_tok}
            else:
                for line in content.splitlines():
                    if line.startswith('{') and 'avg_ts' in line:
                        d = json.loads(line)
                        if d.get('n_gen') == tok:
                            samples = d.get('samples_ts', [])
                            avg_ts = d.get('avg_ts')
                            med_ts = statistics.median(samples) if samples else avg_ts
                            results[name][tok] = {'tok_s': med_ts, 'ms_tok': 1000.0 / med_ts}
                            break
        except Exception as e:
            results[name][tok] = {'tok_s': 0.0, 'ms_tok': 0.0, 'err': str(e)}

print('\n' + '='*80)
print('          M7 COMPETITIVE BENCHMARK SUMMARY (AMD INSTINCT MI50)')
print('='*80)
print(f'{\"Candidate\":<45} | {\"TG64 (tok/s)\":<12} | {\"TG128 (tok/s)\":<13} | {\"TG256 (tok/s)\":<13}')
print('-'*80)

for name, label in CANDIDATES:
    t64 = results[name].get(64, {}).get('tok_s', 0.0)
    t128 = results[name].get(128, {}).get('tok_s', 0.0)
    t256 = results[name].get(256, {}).get('tok_s', 0.0)
    print(f'{label:<45} | {t64:<12.2f} | {t128:<13.2f} | {t256:<13.2f}')

print('='*80)
print(f'{\"Candidate\":<45} | {\"TG64 (ms/tok)\":<12} | {\"TG128 (ms/tok)\":<13} | {\"TG256 (ms/tok)\":<13}')
print('-'*80)

for name, label in CANDIDATES:
    m64 = results[name].get(64, {}).get('ms_tok', 0.0)
    m128 = results[name].get(128, {}).get('ms_tok', 0.0)
    m256 = results[name].get(256, {}).get('ms_tok', 0.0)
    print(f'{label:<45} | {m64:<12.2f} | {m128:<13.2f} | {m256:<13.2f}')

print('='*80)
"

# Telemetry check
echo "Telemetry Summary:"
python3 -c "
import json

OUTDIR = '${OUTDIR}'
total = 0
sclk_1606 = 0
mclk_1000 = 0
max_edge = 0
max_junc = 0
max_mem = 0

with open(f'{OUTDIR}/telemetry.jsonl') as f:
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
