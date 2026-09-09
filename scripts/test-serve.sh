#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <miinfer> <model.gguf>" >&2
    exit 2
fi

miinfer_bin=$1
model_path=$2
port=$(python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)
"$miinfer_bin" serve --model "$model_path" --port "$port" >/tmp/miinfer-serve-test.log 2>&1 &
server_pid=$!
cleanup() { kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 90); do
    curl --silent --fail "http://127.0.0.1:$port/readyz" >/dev/null && break
    sleep 1
done
curl --silent --fail "http://127.0.0.1:$port/readyz" >/dev/null
curl --silent --fail "http://127.0.0.1:$port/" | grep -q '/v1/chat/completions'

python3 - "$port" "$server_pid" <<'PY'
import json
import os
import socket
import sys
import time
import urllib.request

port = int(sys.argv[1])
server_pid = int(sys.argv[2])
def read_response(sock):
    chunks = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)

request = b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}'
with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
    sock.sendall(request)
    assert b" 400 " in sock.recv(4096)
oversized_output = json.dumps({"messages": [{"role": "user", "content": "x"}], "max_tokens": 4097}).encode()
request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " + str(len(oversized_output)).encode() + b"\r\n\r\n" + oversized_output
with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
    sock.sendall(request)
    response = read_response(sock)
    assert b" 400 " in response and b"invalid_request_error" in response
oversized_prompt = json.dumps({"messages": [{"role": "user", "content": "hello " * 1100}], "max_tokens": 1}).encode()
request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " + str(len(oversized_prompt)).encode() + b"\r\n\r\n" + oversized_prompt
with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
    sock.sendall(request)
    response = read_response(sock)
    assert b" 400 " in response and b"context_length_exceeded" in response
with socket.create_connection(("127.0.0.1", port), timeout=15) as sock:
    sock.sendall(b"G")
    time.sleep(11)
    assert b" 408 " in sock.recv(4096)

body = json.dumps({"messages": [{"role": "user", "content": "hello " * 600}], "max_tokens": 1}).encode()
request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
busy = socket.create_connection(("127.0.0.1", port), timeout=5)
busy.sendall(request)
time.sleep(1)
clients = []
for _ in range(9):
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.sendall(request)
    clients.append(sock)
clients[-1].settimeout(5)
assert b" 503 " in clients[-1].recv(4096)
assert urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5).status == 200
metrics = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read()
assert b"miinfer_queue_rejected_total 0" not in metrics
os.kill(server_pid, 15)
busy.close()
queued = 0
for sock in clients[1:-1]:
    sock.settimeout(15)
    try:
        queued += b" 503 " in sock.recv(4096)
    finally:
        sock.close()
assert queued >= 1
PY

echo "serve smoke test passed"
