#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: M31_GPU_INDEX=N M31_GPU_PCI_BDF=BDF M31_WARN_JUNCTION_C=C M31_MAX_JUNCTION_C=C $0 EXPECTED_SECONDS OUTPUT_FILE COMMAND [ARG ...]" >&2
    exit 2
}

[[ $# -ge 3 ]] || usage
expected_seconds=$1
output_file=$2
shift 2
[[ $expected_seconds =~ ^[1-9][0-9]*$ ]] || usage
[[ ! -e $output_file ]] || { echo "refusing to overwrite $output_file" >&2; exit 2; }
for required in timeout setsid ps awk sleep; do
    command -v "$required" >/dev/null || { echo "$required is required" >&2; exit 2; }
done
[[ -r /proc/uptime ]] || { echo "/proc/uptime is required for monotonic timestamps" >&2; exit 2; }

thermal_enabled=0
if [[ -n ${M31_GPU_INDEX:-} || -n ${M31_GPU_PCI_BDF:-} || -n ${M31_WARN_JUNCTION_C:-} || -n ${M31_MAX_JUNCTION_C:-} ]]; then
    [[ ${M31_GPU_INDEX:-} =~ ^[0-9]+$ \
        && ${M31_GPU_PCI_BDF:-} =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]$ \
        && ${M31_WARN_JUNCTION_C:-} =~ ^[0-9]+([.][0-9]+)?$ \
        && ${M31_MAX_JUNCTION_C:-} =~ ^[0-9]+([.][0-9]+)?$ ]] || usage
    command -v awk >/dev/null || { echo "awk is required for the thermal guard" >&2; exit 2; }
    awk -v warning="$M31_WARN_JUNCTION_C" -v limit="$M31_MAX_JUNCTION_C" \
        'BEGIN { exit !(warning < limit) }' || usage
    rocm_smi_bin=${M31_ROCM_SMI_BIN:-rocm-smi}
    command -v "$rocm_smi_bin" >/dev/null || { echo "$rocm_smi_bin is required for the thermal guard" >&2; exit 2; }
    thermal_enabled=1
fi

worker_pid=
worker_pgid=
worker_started=0
worker_waited=0
cleanup_started=0
cleanup_complete=0
worker_status=0
stop_status=0
warning_written=0
detected_seconds=
warning_seconds=$((expected_seconds * 3 / 2))
abort_seconds=$((expected_seconds * 2))

mono_now() {
    local now
    read -r now _ </proc/uptime
    printf '%s' "$now"
}

start_seconds=$(mono_now)

event() {
    local name=$1 now latency=""
    now=$(mono_now)
    if [[ $name == detected ]]; then
        [[ -n $detected_seconds ]] || detected_seconds=$now
    elif [[ -n ${detected_seconds:-} && ( $name == signal_sent || $name == worker_exit || $name == descendants_exited || $name == cleanup_complete ) ]]; then
        latency=" detected_to_${name}_ms=$(awk -v start="$detected_seconds" -v now="$now" 'BEGIN { printf "%.0f", (now-start)*1000 }')"
    fi
    printf 'GUARD_EVENT event=%s monotonic_s=%s since_start_ms=%s worker_pid=%s worker_pgid=%s%s %s\n' \
        "$name" "$now" "$(awk -v start="$start_seconds" -v now="$now" 'BEGIN { printf "%.0f", (now-start)*1000 }')" \
        "${worker_pid:-none}" "${worker_pgid:-none}" "$latency" "${2:-}" >>"$output_file" || \
        printf 'THERMAL_GUARD_LOG_ERROR event=%s\n' "$name" >&2
}

read_device_sample() {
    local telemetry
    telemetry=$(timeout --signal=KILL 3 "$rocm_smi_bin" -d "$M31_GPU_INDEX" --showbus --showtemp 2>&1) || return 1
    awk -v wanted="GPU[$M31_GPU_INDEX]" -v expected_pci="$M31_GPU_PCI_BDF" \
        '$1 == wanted && /PCI Bus:/ { bus_count++; bus=$NF }
         $1 == wanted && /Temperature \(Sensor junction\) \(C\):/ { n++; value=$NF }
         END {
           if (bus_count != 1 || bus != expected_pci || n != 1 || value !~ /^[0-9]+([.][0-9]+)?$/) exit 1
           print bus, value
         }' <<<"$telemetry"
}

if (( thermal_enabled )); then
    if ! device_sample=$(read_device_sample); then
        : >"$output_file"
        event detected "reason=initial_telemetry_unavailable_or_stale"
        printf 'THERMAL_GUARD_ERROR reason=missing_or_invalid_or_stale_junction_or_pci_identity gpu=%s expected_pci=%s\n' \
            "$M31_GPU_INDEX" "$M31_GPU_PCI_BDF" >>"$output_file"
        event cleanup_complete "worker_started=false"
        exit 86
    fi
    read -r sampled_pci junction_c <<<"$device_sample"
fi

: >"$output_file"

group_has_live_members() {
    local processes
    processes=$(ps -eo pgid=,stat=) || return 2
    awk -v wanted="$worker_pgid" '$1 == wanted && $2 !~ /Z/ { found=1 } END { exit !found }' <<<"$processes"
}

worker_is_live() {
    local state
    if ! state=$(ps -o stat= -p "$worker_pid" 2>/dev/null); then
        [[ -e /proc/$worker_pid ]] && return 2
        return 1
    fi
    [[ -n $state && $state != Z* ]]
}

wait_group_for() {
    local seconds=$1 deadline now state
    deadline=$(awk -v now="$(mono_now)" -v seconds="$seconds" 'BEGIN { printf "%.2f", now + seconds }')
    while :; do
        if group_has_live_members; then
            state=0
        else
            state=$?
            (( state == 1 )) && return 0
            return 2
        fi
        now=$(mono_now)
        awk -v now="$now" -v deadline="$deadline" 'BEGIN { exit !(now < deadline) }' || return 1
        sleep 0.05 || return 2
    done
}

reap_worker() {
    (( worker_started && !worker_waited )) || return 0
    if worker_is_live; then
        return 1
    else
        local state=$?
        (( state == 1 )) || return 1
    fi
    set +e
    wait "$worker_pid"
    worker_status=$?
    set -e
    worker_waited=1
    event worker_exit "status=$worker_status"
}

stop_worker_group() {
    local reason=$1 term_result term_wait_result kill_result kill_wait_result
    (( worker_started )) || return 0
    if (( cleanup_started )); then
        return 0
    fi
    cleanup_started=1
    if kill -TERM -- "-$worker_pgid" 2>/dev/null; then term_result=sent; else term_result=not_found; fi
    event signal_sent "signal=TERM result=$term_result reason=$reason"
    if wait_group_for 2; then
        term_wait_result=exited
    else
        term_wait_result=timeout_or_inspection_error
    fi
    if [[ $term_wait_result != exited ]]; then
        if kill -KILL -- "-$worker_pgid" 2>/dev/null; then kill_result=sent; else kill_result=not_found; fi
        event signal_sent "signal=KILL result=$kill_result reason=$reason term_grace_seconds=2"
        if wait_group_for 2; then kill_wait_result=exited; else kill_wait_result=timeout_or_inspection_error; fi
        if [[ $kill_wait_result != exited ]]; then
            event cleanup_failed "reason=process_group_not_confirmed_dead after_kill_grace_seconds=2"
            cleanup_complete=0
            return 1
        fi
    fi
    if ! reap_worker; then
        event cleanup_failed "reason=worker_exit_not_confirmed"
        cleanup_complete=0
        return 1
    fi
    if group_has_live_members; then
        event cleanup_failed "reason=process_group_still_has_live_members"
        cleanup_complete=0
        return 1
    else
        local group_status=$?
        if (( group_status != 1 )); then
            event cleanup_failed "reason=process_group_inspection_failed"
            cleanup_complete=0
            return 1
        fi
    fi
    event descendants_exited "process_group=$worker_pgid group_empty_observed=true"
    event cleanup_complete "process_group=$worker_pgid"
    cleanup_complete=1
    return 0
}

on_exit() {
    local status=$?
    trap - EXIT INT TERM HUP
    if (( worker_started && !cleanup_complete && !cleanup_started )); then
        event detected "reason=monitor_exit exit_status=$status"
        cleanup_started=0
        stop_worker_group monitor_exit || status=87
    elif (( worker_started && !cleanup_complete )); then
        event monitor_exit "exit_status=$status cleanup_retry=true"
        cleanup_started=0
        stop_worker_group monitor_exit || status=87
    fi
    exit "$status"
}

on_signal() {
    local signal=$1 status=$2
    trap - INT TERM HUP
    stop_status=$status
    if (( worker_started )); then
        event detected "reason=monitor_signal signal=$signal"
        stop_worker_group "monitor_signal_$signal" || stop_status=87
    fi
    exit "$stop_status"
}

trap on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

if (( thermal_enabled )); then
    printf 'THERMAL_GUARD_BEGIN gpu=%s pci=%s warn_junction_c=%s max_junction_c=%s initial_junction_c=%s\n' \
        "$M31_GPU_INDEX" "$sampled_pci" "$M31_WARN_JUNCTION_C" "$M31_MAX_JUNCTION_C" "$junction_c" >>"$output_file"
    if awk -v temp="$junction_c" -v warning="$M31_WARN_JUNCTION_C" 'BEGIN { exit !(temp >= warning) }'; then
        printf 'THERMAL_WARNING reason=initial_temperature junction_c=%s warning_c=%s\n' \
            "$junction_c" "$M31_WARN_JUNCTION_C" >>"$output_file"
        warning_written=1
    fi
    if awk -v temp="$junction_c" -v limit="$M31_MAX_JUNCTION_C" 'BEGIN { exit !(temp >= limit) }'; then
        event detected "reason=initial_temperature junction_c=$junction_c limit_c=$M31_MAX_JUNCTION_C"
        printf 'THERMAL_GUARD_ABORT reason=initial_temperature junction_c=%s limit_c=%s\n' \
            "$junction_c" "$M31_MAX_JUNCTION_C" >>"$output_file"
        event cleanup_complete "worker_started=false"
        trap - EXIT INT TERM HUP
        exit 86
    fi
fi

setsid "$@" >>"$output_file" 2>&1 &
worker_pid=$!
worker_pgid=$worker_pid
worker_started=1
event worker_started "command=$*"

while :; do
    if worker_is_live; then
        :
    else
        worker_state=$?
        if (( worker_state != 1 )); then
            event detected "reason=worker_inspection_failed"
            stop_worker_group worker_inspection_failed || exit 87
            exit 87
        fi
        reap_worker
        if group_has_live_members; then
            event detected "reason=worker_exited_with_descendants"
            stop_worker_group worker_exited_with_descendants || exit 87
        else
            group_status=$?
            if (( group_status != 1 )); then
                event detected "reason=process_group_inspection_failed"
                stop_worker_group process_group_inspection_failed || exit 87
                exit 87
            fi
            event descendants_exited "process_group=$worker_pgid group_empty_observed=true"
            event cleanup_complete "process_group=$worker_pgid"
            cleanup_complete=1
        fi
        exit "$worker_status"
    fi

    now=$(mono_now)
    elapsed=$(awk -v start="$start_seconds" -v now="$now" 'BEGIN { printf "%.2f", now - start }')
    if (( !warning_written )) && awk -v elapsed="$elapsed" -v threshold="$warning_seconds" 'BEGIN { exit !(elapsed >= threshold) }'; then
        printf 'WATCHDOG_WARNING elapsed_sec=%s expected_sec=%s threshold=1.5x\n' \
            "$warning_seconds" "$expected_seconds" >>"$output_file"
        warning_written=1
    fi
    if awk -v elapsed="$elapsed" -v threshold="$abort_seconds" 'BEGIN { exit !(elapsed >= threshold) }'; then
        event detected "reason=runtime_timeout elapsed_sec=$elapsed limit_sec=$abort_seconds"
        printf 'WATCHDOG_ABORT expected_sec=%s abort_sec=%s threshold=2.0x\n' \
            "$expected_seconds" "$abort_seconds" >>"$output_file"
        stop_worker_group runtime_timeout || exit 87
        exit 124
    fi

    if (( thermal_enabled )); then
        if ! device_sample=$(read_device_sample); then
            event detected "reason=telemetry_unavailable_or_stale"
            printf 'THERMAL_GUARD_ERROR reason=missing_or_invalid_or_stale_junction_or_pci_identity gpu=%s expected_pci=%s\n' \
                "$M31_GPU_INDEX" "$M31_GPU_PCI_BDF" >>"$output_file"
            stop_worker_group telemetry_unavailable_or_stale || exit 87
            exit 86
        fi
        read -r sampled_pci junction_c <<<"$device_sample"
        printf 'THERMAL_SAMPLE gpu=%s pci=%s junction_c=%s warn_c=%s limit_c=%s\n' \
            "$M31_GPU_INDEX" "$sampled_pci" "$junction_c" "$M31_WARN_JUNCTION_C" "$M31_MAX_JUNCTION_C" >>"$output_file"
        if (( !warning_written )) && awk -v temp="$junction_c" -v warning="$M31_WARN_JUNCTION_C" 'BEGIN { exit !(temp >= warning) }'; then
            printf 'THERMAL_WARNING reason=temperature_limit junction_c=%s warning_c=%s\n' \
                "$junction_c" "$M31_WARN_JUNCTION_C" >>"$output_file"
            warning_written=1
        fi
        if awk -v temp="$junction_c" -v limit="$M31_MAX_JUNCTION_C" 'BEGIN { exit !(temp >= limit) }'; then
            event detected "reason=temperature_limit junction_c=$junction_c limit_c=$M31_MAX_JUNCTION_C"
            printf 'THERMAL_GUARD_ABORT reason=temperature_limit junction_c=%s limit_c=%s\n' \
                "$junction_c" "$M31_MAX_JUNCTION_C" >>"$output_file"
            stop_worker_group temperature_limit || exit 87
            exit 86
        fi
    fi
    sleep 0.2
done
