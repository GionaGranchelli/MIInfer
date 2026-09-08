#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <miinfer> <model.gguf>" >&2
    exit 2
fi

miinfer_bin=$1
model_path=$2
port=18080
"$miinfer_bin" serve "$model_path" --port "$port" >/tmp/miinfer-serve-test.log 2>&1 &
server_pid=$!
cleanup() { kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 90); do
    curl --silent --fail "http://127.0.0.1:$port/readyz" >/dev/null && break
    sleep 1
done
curl --silent --fail "http://127.0.0.1:$port/readyz" >/dev/null

python3 - "$port" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
request = b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}'
with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
    sock.sendall(request)
    assert b" 400 " in sock.recv(4096)
with socket.create_connection(("127.0.0.1", port), timeout=15) as sock:
    sock.sendall(b"G")
    time.sleep(11)
    assert b" 408 " in sock.recv(4096)
PY

echo "serve smoke test passed"
