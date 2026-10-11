#!/usr/bin/env bash
set -euo pipefail

model="/models/${M31_MODEL_FILE:?missing mounted model filename}"
export MIINFER_API_KEY=miinfer MIINFER_SESSION_REUSE=1
# The 128K engine capacity only leaves room for the exact 64K start plus turn history.
/workspace/MIInfer/build/m31-accel-0001/miinfer serve \
  --model "$model" --host 127.0.0.1 --port 8087 \
  --context 131072 --experimental-context --session-dir /captures/session-cache \
  > /captures/server.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

ready=0
for _ in {1..180}; do
  kill -0 "$server_pid" || { cat /captures/server.log; exit 1; }
  if python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8087/readyz", timeout=1)' 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
(( ready == 1 )) || { echo 'server readiness timeout' >&2; cat /captures/server.log; exit 1; }

python3 /workspace/MIInfer/bench/m31_accel_0001_agent_workload.py \
  --model-path "$model" --model Qwen3.8-27B-Q4_K_M \
  --token-counter /workspace/MIInfer/build/m31-agent64k-host/miinfer-m31-agent-token-count \
  --server-log /captures/server.log --output /captures/agent-workload.json
