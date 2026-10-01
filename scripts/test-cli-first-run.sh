#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s MIINFER-BINARY MODEL.gguf\n' "$0" >&2
    exit 2
fi

binary=$(realpath "$1")
model=$(realpath "$2")
if [[ ! -x "$binary" || ! -f "$model" ]]; then
    printf 'binary must be executable and model must exist\n' >&2
    exit 2
fi

stage=$(mktemp -d)
server_pid=''
cleanup() {
    status=$?
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ $status -eq 0 ]]; then
        rm -rf "$stage"
    else
        printf 'first-run test failed; diagnostics retained at %s\n' "$stage" >&2
    fi
    return "$status"
}
trap cleanup EXIT
export XDG_CONFIG_HOME="$stage/config"

"$binary" doctor --model "$model" > "$stage/doctor.txt"
grep -q '^Runtime  PASS' "$stage/doctor.txt"
"$binary" config set model-dir "$(dirname "$model")"
"$binary" config set default-model "$(basename "$model")"
"$binary" models > "$stage/models.txt"
grep -q "$(basename "$model")" "$stage/models.txt"
"$binary" inspect "$(basename "$model")" --json > "$stage/inspect.json"
grep -q 'Qwen3.8-27B' "$stage/inspect.json"

printf 'Testing run --prompt...\n' >&2
"$binary" run "$model" --prompt 'Reply with one short greeting.' \
    --max-tokens 8 --temperature 0 --stats > "$stage/run.txt" 2> "$stage/run-stats.txt"
grep -q 'generated_tokens=' "$stage/run-stats.txt"
if grep -Eq '<think|</think>|<tool_call|<function=' "$stage/run.txt"; then
    printf 'run leaked internal reasoning or tool protocol\n' >&2
    exit 1
fi
printf 'Testing automatic production profile with a stale selector set...\n' >&2
MIINFER_PRESET=not-a-public-preset "$binary" run "$model" \
    --prompt 'Reply with one short greeting.' --max-tokens 4 --temperature 0 --no-stream > /dev/null
printf 'Testing positional prompt...\n' >&2
"$binary" run "$model" 'Reply with one short greeting.' \
    --max-tokens 4 --temperature 0 --no-stream > /dev/null
printf 'Testing piped prompt...\n' >&2
printf 'Reply with one short greeting.\n' |
    "$binary" run "$model" --max-tokens 4 --temperature 0 --no-stream > /dev/null

printf 'Testing configured default-model chat...\n' >&2
printf 'Say hello briefly.\nWhat did I ask you to do?\n/exit\n' |
    "$binary" chat --max-tokens 12 --stats > "$stage/chat.txt" 2> "$stage/chat-stats.txt"
grep -Eq 'reused_prefix=[1-9][0-9]*' "$stage/chat-stats.txt"
if grep -Eq '<think|</think>|<tool_call|<function=' "$stage/chat.txt"; then
    printf 'chat leaked internal reasoning or tool protocol\n' >&2
    exit 1
fi

printf 'Testing configured default-model server and API completion...\n' >&2
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
model_id=$(basename "${model%.gguf}")
"$binary" serve --host 127.0.0.1 --port "$port" > "$stage/server-out.txt" 2> "$stage/server.log" &
server_pid=$!
ready=0
for _ in $(seq 1 120); do
    if curl --silent --fail "http://127.0.0.1:$port/readyz" > "$stage/ready.json"; then
        ready=1
        break
    fi
    sleep 1
done
if [[ $ready -ne 1 ]]; then
    printf 'server did not become ready\n' >&2
    cat "$stage/server.log" >&2
    exit 1
fi
grep -q '"ready":true' "$stage/ready.json"
curl --silent --show-error --fail "http://127.0.0.1:$port/healthz" \
    > "$stage/health.json"
grep -q '"status":"ok"' "$stage/health.json"
curl --silent --show-error --fail "http://127.0.0.1:$port/v1/models" \
    > "$stage/models-api.json"
grep -q "\"id\":\"$model_id\"" "$stage/models-api.json"
curl --silent --show-error --fail "http://127.0.0.1:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one short greeting.\"}],\"max_tokens\":8,\"temperature\":0}" \
    > "$stage/completion.json"
grep -q '"role":"assistant"' "$stage/completion.json"
grep -q '"content":"' "$stage/completion.json"

if env -u MIINFER_API_KEY "$binary" serve --model "$model" \
    --host 0.0.0.0 --port "$port" > "$stage/lan-out.txt" 2> "$stage/lan-error.txt"; then
    printf 'unauthenticated LAN serving was unexpectedly accepted\n' >&2
    exit 1
fi
grep -q 'non-loopback serving requires' "$stage/lan-error.txt"

printf 'first-run CLI test passed for %s\n' "$(basename "$model")"
