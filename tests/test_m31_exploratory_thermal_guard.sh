#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runner="$repo_root/scripts/run-m31-exploratory.sh"
fake_rocm_smi="$repo_root/tests/fixtures/fake-rocm-smi.sh"
temp_dir=$(mktemp -d)
runner_pid=
sentinel_pid=
cleanup() {
    [[ -z $runner_pid ]] || kill -KILL "$runner_pid" 2>/dev/null || true
    [[ -z $sentinel_pid ]] || kill "$sentinel_pid" 2>/dev/null || true
    while IFS= read -r pid; do kill -KILL "$pid" 2>/dev/null || true; done < <(find "$temp_dir" -name '*.pid' -type f -exec cat {} \; 2>/dev/null)
    rm -rf "$temp_dir"
}
trap cleanup EXIT

sleep() {
    if [[ ${M31_TEST_MONITOR_SLEEP_FAIL:-0} == 1 && ${1:-} == 0.2 ]]; then return 42; fi
    if [[ ${M31_TEST_MONITOR_WAIT_FAIL:-0} == 1 && ${1:-} == 0.05 && ${M31_TEST_MONITOR_WAIT_FAILED:-0} == 0 ]]; then
        export M31_TEST_MONITOR_WAIT_FAILED=1
        return 42
    fi
    command sleep "$@"
}
export -f sleep

guard_env=(M31_GPU_INDEX=1 M31_GPU_PCI_BDF=0000:06:00.0 M31_WARN_JUNCTION_C=80 M31_MAX_JUNCTION_C=85 M31_ROCM_SMI_BIN="$fake_rocm_smi")
is_live() {
    local state
    state=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
    [[ -n $state && $state != Z* ]]
}
assert_dead() {
    if is_live "$1"; then
        printf 'process %s is still alive\n' "$1" >&2
        return 1
    fi
}
assert_event_order() {
    awk '
      /event=detected / && !detected { detected=$2 }
      /event=signal_sent / && /signal=TERM/ { term=$2 }
      /event=signal_sent / && /signal=KILL/ { kill=$2 }
      /event=worker_exit / { exited=$2 }
      /event=descendants_exited / { descendants=$2 }
      /event=cleanup_complete / { cleanup=$2 }
      /event=GUARD_EVENT/ {
        for (i=1; i<=NF; i++) if ($i ~ /^monotonic_s=/) {
          split($i, a, "="); if (a[2]+0 < previous+0) exit 2; previous=a[2]
        }
      }
      END {
        if (!detected || !term || !exited || !descendants || !cleanup || detected+0 > term+0 || term+0 > exited+0 || exited+0 > descendants+0 || descendants+0 > cleanup+0) exit 1
      }
    ' "$1"
}
assert_bounded_shutdown() {
    awk '
      function metric(key, i, parts) {
        for (i=1; i<=NF; i++) if ($i ~ "^" key "=") { split($i, parts, "="); return parts[2]+0 }
        return 0
      }
      /event=signal_sent / && /signal=TERM/ { term++ }
      /event=signal_sent / && /signal=TERM/ { term_ms=metric("detected_to_signal_sent_ms") }
      /event=signal_sent / && /signal=KILL/ { kill++; kill_ms=metric("detected_to_signal_sent_ms") }
      /event=worker_exit / { worker++; worker_ms=metric("detected_to_worker_exit_ms") }
      /event=descendants_exited / { descendants++; descendants_ms=metric("detected_to_descendants_exited_ms") }
      /event=cleanup_complete / { cleanup++ }
      /event=detected / { detected=1 }
      /event=cleanup_complete / {
        for (i=1; i<=NF; i++) if ($i ~ /^detected_to_cleanup_complete_ms=/) {
          split($i, a, "="); latency=a[2]+0
        }
      }
      END {
        if (!detected || term != 1 || worker != 1 || descendants != 1 || cleanup != 1 || !latency || latency > 4500) exit 1
        if ((expect_kill && kill != 1) || (!expect_kill && kill != 0)) exit 1
        if (expect_kill == 1 && (kill_ms < 1900 || kill_ms > 2600)) exit 1
        if (expect_kill == 2 && kill_ms > 1000) exit 1
        printf "CPU_GUARD_TIMING detected_to_TERM_ms=%d detected_to_KILL_ms=%d detected_to_worker_exit_ms=%d detected_to_descendants_exit_ms=%d detected_to_cleanup_ms=%d\n", term_ms, kill_ms, worker_ms, descendants_ms, latency
      }
    ' expect_kill="${2:-0}" "$1"
}

env "${guard_env[@]}" M31_FAKE_TEMP_C=33 "$runner" 10 "$temp_dir/cool.log" sleep 0.1
grep -q 'THERMAL_GUARD_BEGIN gpu=1 pci=0000:06:00.0 warn_junction_c=80 max_junction_c=85 initial_junction_c=33' "$temp_dir/cool.log"
grep -q 'event=cleanup_complete' "$temp_dir/cool.log"

# A TERM-aware worker exits before the bounded escalation deadline.
set +e
env "${guard_env[@]}" M31_FAKE_TEMPS='33 85' M31_FAKE_COUNTER_FILE="$temp_dir/term-count" "$runner" 30 "$temp_dir/term.log" \
    bash -c 'trap "exit 0" TERM; while :; do sleep 0.05; done'
term_status=$?
set -e
[[ $term_status -eq 86 ]]
grep -q 'event=signal_sent .*signal=TERM' "$temp_dir/term.log"
! grep -q 'event=signal_sent .*signal=KILL' "$temp_dir/term.log"
assert_event_order "$temp_dir/term.log"
assert_bounded_shutdown "$temp_dir/term.log" 0

# Ignored TERM forces KILL; descendants share the managed session/process group.
set +e
env "${guard_env[@]}" M31_FAKE_TEMPS='33 85' M31_FAKE_COUNTER_FILE="$temp_dir/kill-count" M31_WORKER_PIDFILE="$temp_dir/worker.pid" M31_CHILD_PIDFILE="$temp_dir/child.pid" M31_GRANDCHILD_PIDFILE="$temp_dir/grandchild.pid" "$runner" 30 "$temp_dir/kill.log" \
    bash -c 'trap "" TERM; echo $$ >"$M31_WORKER_PIDFILE"; (trap "" TERM; (trap "" TERM; exec sleep 30) & echo $! >"$M31_GRANDCHILD_PIDFILE"; wait) & echo $! >"$M31_CHILD_PIDFILE"; wait'
kill_status=$?
set -e
[[ $kill_status -eq 86 ]]
grep -q 'event=signal_sent .*signal=KILL' "$temp_dir/kill.log"
assert_dead "$(<"$temp_dir/worker.pid")"
assert_dead "$(<"$temp_dir/child.pid")"
assert_dead "$(<"$temp_dir/grandchild.pid")"
assert_event_order "$temp_dir/kill.log"
assert_bounded_shutdown "$temp_dir/kill.log" 1

# A monitor wait/poll failure during cleanup must still fail closed and kill the group.
set +e
env "${guard_env[@]}" M31_FAKE_TEMPS='33 85' M31_FAKE_COUNTER_FILE="$temp_dir/poll-failure-count" M31_TEST_MONITOR_WAIT_FAIL=1 M31_WORKER_PIDFILE="$temp_dir/poll-failure.pid" "$runner" 30 "$temp_dir/poll-failure.log" \
    bash -c 'trap "" TERM; echo $$ >"$M31_WORKER_PIDFILE"; exec /bin/sleep 30'
poll_failure_status=$?
set -e
[[ $poll_failure_status -eq 86 ]]
assert_dead "$(<"$temp_dir/poll-failure.pid")"
assert_event_order "$temp_dir/poll-failure.log"
assert_bounded_shutdown "$temp_dir/poll-failure.log" 2

# A worker that exits on its own keeps its status and needs no signal.
set +e
env "${guard_env[@]}" M31_FAKE_TEMP_C=33 "$runner" 10 "$temp_dir/exited.log" bash -c 'exit 7'
exited_status=$?
set -e
[[ $exited_status -eq 7 ]]
grep -q 'event=worker_exit .*status=7' "$temp_dir/exited.log"
! grep -q 'event=signal_sent' "$temp_dir/exited.log"
grep -q 'event=cleanup_complete' "$temp_dir/exited.log"

# A worker can exit while a nested descendant survives; the process group is still cleaned.
set +e
env "${guard_env[@]}" M31_FAKE_TEMPS='33 33 85' M31_FAKE_COUNTER_FILE="$temp_dir/orphan-count" M31_WORKER_PIDFILE="$temp_dir/orphan.pid" "$runner" 30 "$temp_dir/orphan.log" \
    bash -c '(trap "" TERM; exec sleep 30) & echo $! >"$M31_WORKER_PIDFILE"; exit 9'
orphan_status=$?
set -e
[[ $orphan_status -eq 9 ]]
orphan_pid=$(<"$temp_dir/orphan.pid")
assert_dead "$orphan_pid"
grep -q 'reason=worker_exited_with_descendants' "$temp_dir/orphan.log"
grep -q 'event=cleanup_complete' "$temp_dir/orphan.log"

# Stopping the isolated worker group must not signal its caller/container-like process group.
sleep 30 &
sentinel_pid=$!
set +e
env "${guard_env[@]}" M31_FAKE_TEMPS='33 85' M31_FAKE_COUNTER_FILE="$temp_dir/group-count" "$runner" 30 "$temp_dir/group.log" sleep 30
group_status=$?
set -e
[[ $group_status -eq 86 ]]
is_live "$sentinel_pid"
grep -q 'event=cleanup_complete' "$temp_dir/group.log"
kill "$sentinel_pid" 2>/dev/null || true
sentinel_pid=

# Telemetry failure and a sample that exceeds the 3s freshness deadline fail closed.
set +e
env "${guard_env[@]}" M31_FAKE_TEMPS='33' M31_FAKE_COUNTER_FILE="$temp_dir/unavailable-count" M31_FAKE_FAIL_AFTER=1 M31_WORKER_PIDFILE="$temp_dir/unavailable.pid" "$runner" 30 "$temp_dir/unavailable.log" \
    bash -c 'echo $$ >"$M31_WORKER_PIDFILE"; exec sleep 30'
unavailable_status=$?
set -e
[[ $unavailable_status -eq 86 ]]
grep -q 'reason=telemetry_unavailable_or_stale' "$temp_dir/unavailable.log"
assert_dead "$(<"$temp_dir/unavailable.pid")"
assert_event_order "$temp_dir/unavailable.log"
assert_bounded_shutdown "$temp_dir/unavailable.log" 0

set +e
env "${guard_env[@]}" M31_FAKE_TEMPS='33' M31_FAKE_COUNTER_FILE="$temp_dir/stale-count" M31_FAKE_DELAY_AFTER=1 M31_FAKE_DELAY_SECONDS=4 M31_WORKER_PIDFILE="$temp_dir/stale.pid" "$runner" 30 "$temp_dir/stale.log" \
    bash -c 'echo $$ >"$M31_WORKER_PIDFILE"; exec sleep 30'
stale_status=$?
set -e
[[ $stale_status -eq 86 ]]
grep -q 'reason=telemetry_unavailable_or_stale' "$temp_dir/stale.log"
assert_dead "$(<"$temp_dir/stale.pid")"
assert_event_order "$temp_dir/stale.log"
assert_bounded_shutdown "$temp_dir/stale.log" 0

# An unexpected monitor-loop command failure must invoke EXIT cleanup.
env "${guard_env[@]}" M31_FAKE_TEMP_C=33 M31_TEST_MONITOR_SLEEP_FAIL=1 M31_WORKER_PIDFILE="$temp_dir/monitor.pid" "$runner" 30 "$temp_dir/monitor.log" \
    bash -c 'echo $$ >"$M31_WORKER_PIDFILE"; exec /bin/sleep 30' &
runner_pid=$!
for _ in {1..100}; do [[ -s $temp_dir/monitor.pid ]] && break; sleep 0.05; done
[[ -s $temp_dir/monitor.pid ]]
monitor_worker=$(<"$temp_dir/monitor.pid")
set +e
wait "$runner_pid"
monitor_status=$?
set -e
runner_pid=
[[ $monitor_status -eq 42 ]]
assert_dead "$monitor_worker"
grep -q 'reason=monitor_exit exit_status=42' "$temp_dir/monitor.log"
assert_event_order "$temp_dir/monitor.log"
assert_bounded_shutdown "$temp_dir/monitor.log" 0

# Invalid initial telemetry refuses to start any worker.
set +e
env "${guard_env[@]}" M31_FAKE_OUTPUT=malformed M31_WORKER_PIDFILE="$temp_dir/never.pid" "$runner" 10 "$temp_dir/bad.log" true
bad_status=$?
set -e
[[ $bad_status -eq 86 ]]
grep -q 'reason=initial_telemetry_unavailable_or_stale' "$temp_dir/bad.log"
[[ ! -e $temp_dir/never.pid ]]

printf 'PASS: bounded TERM/KILL shutdown, nested descendants, natural exit, isolated process group, stale/unavailable telemetry, unexpected monitor exit, and monotonic latency events\n'
