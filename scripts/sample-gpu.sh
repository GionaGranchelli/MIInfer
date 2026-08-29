#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'usage: %s OUTPUT.jsonl [INTERVAL_MS]\n' "$0" >&2
    exit 2
fi

output_path=$1
interval_ms=${2:-250}
if ! [[ "$interval_ms" =~ ^[0-9]+$ ]] || (( interval_ms < 1 )); then
    printf 'INTERVAL_MS must be a positive integer\n' >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$(dirname "$output_path")"
interval_seconds=$(awk -v milliseconds="$interval_ms" 'BEGIN { printf "%.3f", milliseconds / 1000 }')

json_quote() {
    local value=${1-}
    if command -v jq >/dev/null 2>&1; then
        printf '%s' "$value" | jq -Rs .
        return
    fi
    local first=1
    printf '"'
    while IFS= read -r line || [[ -n "$line" ]]; do
        if (( first == 0 )); then
            printf '\\n'
        fi
        escaped_line="$(printf '%s' "$line" | sed 's/\\/\\\\/g;s/"/\\"/g')"
        printf '%s' "$escaped_line"
        first=0
    done <<< "$value"
    printf '"'
}

trap 'exit 0' TERM INT
: > "$output_path"
while true; do
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    payload='{"status":"UNAVAILABLE"}'
    if command -v rocm-smi >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
        raw_payload="$(rocm-smi --json --showproductname --showmeminfo vram --showclocks \
            --showtemp --showpower --showmaxpower 2>/dev/null || true)"
        normalized_payload="$(printf '%s' "$raw_payload" | jq -c . 2>/dev/null || true)"
        if [[ -n "$normalized_payload" ]]; then
            payload=$normalized_payload
        fi
    fi
    if command -v jq >/dev/null 2>&1; then
        jq -cn --arg timestamp "$timestamp" --argjson monitoring "$payload" \
            '{timestamp_utc: $timestamp, rocm_smi: $monitoring}' >> "$output_path"
    else
        printf '{"timestamp_utc":%s,"rocm_smi":{"status":"UNAVAILABLE"}}\n' \
            "$(json_quote "$timestamp")" >> "$output_path"
    fi
    sleep "$interval_seconds"
done
