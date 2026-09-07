#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf}
BINARY=./build/mi50-release/miinfer
PORT=8999

OUTDIR="bench/results/m10-streaming-audit/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTDIR"

echo "=========================================================================="
echo "          M10-A: Streaming Decode Throughput Audit                        "
echo "=========================================================================="
echo "Binary: $BINARY"
echo "Model:  $MODEL"
echo "Outdir: $OUTDIR"

PROMPT="The AMD Instinct MI50 is a high-performance GPU featuring"

# 1. Non-streaming CLI runs
echo "--- 1. Measuring Non-Streaming Decode (on-device chained) ---"
for i in 1 2 3 4 5; do
    echo "Non-stream run $i..."
    "$BINARY" run "$MODEL" --prompt "$PROMPT" --max-tokens 64 --no-stream > "${OUTDIR}/non-stream-${i}.log" 2>&1
    grep "Decode Tokens:" "${OUTDIR}/non-stream-${i}.log"
    sleep 2
done

# 2. Terminal streaming CLI runs
echo "--- 2. Measuring Terminal Streaming Decode (per-token on_token) ---"
for i in 1 2 3 4 5; do
    echo "Terminal stream run $i..."
    "$BINARY" run "$MODEL" --prompt "$PROMPT" --max-tokens 64 > "${OUTDIR}/term-stream-${i}.log" 2>&1
    grep "Decode Tokens:" "${OUTDIR}/term-stream-${i}.log"
    sleep 2
done

# 3. SSE Streaming via HTTP server
echo "--- 3. Starting server for SSE Streaming Decode ---"
"$BINARY" serve "$MODEL" --port "$PORT" > "${OUTDIR}/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait for server to listen
while ! curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    sleep 1
done
echo "Server ready on port $PORT."

echo "--- 4. Measuring HTTP SSE Streaming Decode ---"
python3 - << 'PYEOF' "$PORT" "$OUTDIR"
import sys, time, json, urllib.request, statistics

port = sys.argv[1]
outdir = sys.argv[2]
url = f"http://127.0.0.1:{port}/v1/chat/completions"

prompt = "The AMD Instinct MI50 is a high-performance GPU featuring"
payload = json.dumps({
    "model": "qwen3.5-27b",
    "messages": [{"role": "user", "content": prompt}],
    "stream": True,
    "max_tokens": 64
}).encode('utf-8')

results = []

for run in range(1, 6):
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first_chunk_t = None
    last_chunk_t = None
    token_count = 0
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode('utf-8').strip()
            if not line_str.startswith("data: "):
                continue
            data = line_str[6:]
            if data == "[DONE]":
                last_chunk_t = time.perf_counter()
                break
            if first_chunk_t is None:
                first_chunk_t = time.perf_counter()
            token_count += 1

    total_time = last_chunk_t - t0
    ttft = (first_chunk_t - t0) * 1000.0 if first_chunk_t else 0.0
    decode_time = last_chunk_t - first_chunk_t if first_chunk_t and last_chunk_t else total_time
    tok_s = (token_count / decode_time) if decode_time > 0 else 0.0

    print(f"  SSE Run {run}: tokens={token_count} total={total_time*1000.0:.1f}ms TTFT={ttft:.1f}ms decode_tok_s={tok_s:.2f} tok/s")
    results.append(tok_s)
    with open(f"{outdir}/sse-stream-{run}.json", "w") as f:
        json.dump({"run": run, "tokens": token_count, "ttft_ms": ttft, "decode_tok_s": tok_s, "total_ms": total_time * 1000.0}, f)
    time.sleep(2)

print(f"SSE Streaming Median: {statistics.median(results):.2f} tok/s")
PYEOF

kill -TERM "$SERVER_PID" 2>/dev/null || true
trap - EXIT
wait "$SERVER_PID" 2>/dev/null || true

echo "--- 5. Comparison Summary ---"
python3 - << 'PYEOF' "$OUTDIR"
import sys, os, re, json, statistics

outdir = sys.argv[1]

def get_cli_runs(prefix):
    toks, lats = [], []
    for i in range(1, 6):
        with open(os.path.join(outdir, f"{prefix}-{i}.log")) as f:
            content = f.read()
        m = re.search(r'Decode Tokens:\s+\d+\s+tokens\s+\([\d.]+\s+ms,\s+([\d.]+)\s+tok/s\)', content)
        if m:
            toks.append(float(m.group(1)))
    return toks

non_stream = get_cli_runs("non-stream")
term_stream = get_cli_runs("term-stream")

sse_stream = []
for i in range(1, 6):
    with open(os.path.join(outdir, f"sse-stream-{i}.json")) as f:
        d = json.load(f)
        sse_stream.append(d["decode_tok_s"])

med_ns = statistics.median(non_stream)
med_ts = statistics.median(term_stream)
med_sse = statistics.median(sse_stream)

print("\n" + "="*70)
print("            M10-A STREAMING DECODE SCORECARD")
print("="*70)
print(f"1. Non-streaming CLI (chained):  {med_ns:6.2f} tok/s  (100.0%)  [Reference]")
print(f"2. Terminal streaming (on_token): {med_ts:6.2f} tok/s  ({med_ts/med_ns*100.0:5.1f}%)  [Delta: {med_ts - med_ns:+5.2f} tok/s]")
print(f"3. HTTP SSE streaming (OpenAI):   {med_sse:6.2f} tok/s  ({med_sse/med_ns*100.0:5.1f}%)  [Delta: {med_sse - med_ns:+5.2f} tok/s]")
print("="*70 + "\n")
PYEOF
