#!/usr/bin/env bash
set -euo pipefail

gpu_index=$2
count=0
if [[ -n ${M31_FAKE_COUNTER_FILE:-} && -r $M31_FAKE_COUNTER_FILE ]]; then
    read -r count <"$M31_FAKE_COUNTER_FILE"
fi
if [[ -n ${M31_FAKE_DELAY_AFTER:-} && $count -ge $M31_FAKE_DELAY_AFTER ]]; then
    sleep "${M31_FAKE_DELAY_SECONDS:-4}"
fi
if [[ -n ${M31_FAKE_FAIL_AFTER:-} && $count -ge $M31_FAKE_FAIL_AFTER ]]; then
    exit 1
fi
if [[ ${M31_FAKE_OUTPUT:-} == malformed ]]; then
    printf 'GPU[%s] : junction temperature unavailable\n' "$gpu_index"
    exit 0
fi

printf 'GPU[%s] : PCI Bus: %s\n' "$gpu_index" "${M31_FAKE_PCI_BDF:-0000:06:00.0}"

if [[ -n ${M31_FAKE_TEMPS:-} ]]; then
    read -r -a temperatures <<<"$M31_FAKE_TEMPS"
    index=$((count < ${#temperatures[@]} ? count : ${#temperatures[@]} - 1))
    temperature=${temperatures[$index]}
    if [[ -n ${M31_FAKE_COUNTER_FILE:-} ]]; then
        printf '%s\n' "$((count + 1))" >"$M31_FAKE_COUNTER_FILE"
    fi
else
    temperature=${M31_FAKE_TEMP_C:-33.0}
fi

printf 'GPU[%s] : Temperature (Sensor junction) (C): %s\n' "$gpu_index" "$temperature"
