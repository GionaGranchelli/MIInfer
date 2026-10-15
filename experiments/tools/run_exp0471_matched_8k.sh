#!/usr/bin/env bash
set -u

out=/home/machinist/m31-gqa6-16lane-rewrite/results/EXP-0471-unroll1-8k-matched
prompt=/home/machinist/m31-perf-recovery-0001/build/mi50-release/m31-64k-context.txt
model=/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf
image=localhost/miinfer-m31-rocm721-libdw@sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f
gpu_index=2
expected_bdf=0000:88:00.0

preflight_gpu() {
    local case=$1 bus_report actual_bdf pid_report pid map_report declared indices used_bytes

    bus_report=$(rocm-smi -d "$gpu_index" --showbus 2>&1) || {
        echo "preflight=fail case=$case reason=rocm_smi_showbus_failed" | tee -a "$out/status.log"
        return 1
    }
    actual_bdf=$(printf '%s\n' "$bus_report" | sed -n 's/.*PCI Bus: //p' | head -n 1)
    if [ "$actual_bdf" != "$expected_bdf" ]; then
        echo "preflight=fail case=$case reason=bdf_mismatch expected=$expected_bdf actual=${actual_bdf:-unknown}" | tee -a "$out/status.log"
        return 1
    fi

    pid_report=$(rocm-smi --showpids 2>&1) || {
        echo "preflight=fail case=$case reason=rocm_smi_showpids_failed" | tee -a "$out/status.log"
        return 1
    }
    printf '%s\n' "$pid_report" | grep -q 'KFD process information:' || {
        echo "preflight=fail case=$case reason=unrecognized_kfd_output" | tee -a "$out/status.log"
        return 1
    }

    while read -r pid; do
        [ -n "$pid" ] || continue
        map_report=$(rocm-smi --showpidgpus "$pid" 2>&1) || {
            echo "preflight=fail case=$case reason=pid_map_failed pid=$pid" | tee -a "$out/status.log"
            return 1
        }
        declared=$(printf '%s\n' "$map_report" | sed -n "s/^PID $pid is using \\([0-9][0-9]*\\) DRM device(s):$/\\1/p")
        indices=$(printf '%s\n' "$map_report" | awk -v pid="$pid" '
            $0 ~ "^PID " pid " is using " {inside=1; next}
            inside && /^=+/ {exit}
            inside && $1 ~ /^[0-9]+$/ {print $1}
        ')
        if [ -z "$declared" ] || [ "$(printf '%s\n' "$indices" | awk 'NF {n++} END {print n+0}')" -ne "$declared" ]; then
            echo "preflight=fail case=$case reason=unparseable_pid_map pid=$pid" | tee -a "$out/status.log"
            return 1
        fi
        if printf '%s\n' "$indices" | grep -qx "$gpu_index"; then
            echo "preflight=fail case=$case reason=selected_gpu_has_kfd_pid pid=$pid gpu=$gpu_index bdf=$actual_bdf" | tee -a "$out/status.log"
            return 1
        fi
        echo "preflight=allow case=$case pid=$pid gpu_indices=$(printf '%s' "$indices" | tr '\n' ',')" >>"$out/status.log"
    done < <(printf '%s\n' "$pid_report" | awk '$1 ~ /^[0-9]+$/ {print $1}')

    used_bytes=$(rocm-smi -d "$gpu_index" --showmeminfo vram 2>&1 | sed -n 's/.*VRAM Total Used Memory (B): //p' | head -n 1)
    if ! [[ "$used_bytes" =~ ^[0-9]+$ ]] || [ "$used_bytes" -gt 1073741824 ]; then
        echo "preflight=fail case=$case reason=unattributed_or_busy_vram used_bytes=${used_bytes:-unknown}" | tee -a "$out/status.log"
        return 1
    fi
    echo "preflight=pass case=$case gpu=$gpu_index bdf=$actual_bdf used_vram_bytes=$used_bytes" | tee -a "$out/status.log"
}

run_case() {
    local label=$1
    local dim16=$2
    local name="m31-exp0471-unroll1-8k-matched-$label"
    local pid temp rc

    preflight_gpu "$label" || return 2
    echo "case=$label dim16=$dim16 started=$(date --iso-8601=seconds)" \
        | tee -a "$out/status.log"
    timeout --signal=TERM --kill-after=10s 600s podman run --rm --name "$name" \
        --security-opt label=disable --device=/dev/kfd --device=/dev/dri \
        -e ROCR_VISIBLE_DEVICES=2 -e MIINFER_HIP_ARCHITECTURE=gfx906 \
        -e MIINFER_TARGET_ARCH=gfx906 -e MIINFER_HIP_GRAPH=1 \
        -e MIINFER_M31_Q8_GQA6_PROD_ARITH=1 -e MIINFER_M31_Q8_GQA6_DIRECT_GLOBAL=1 \
        -e MIINFER_M31_Q8_GQA6_DIM16_LANES="$dim16" \
        -e MIINFER_M31_Q8_GQA6_PACKED_LDS=0 -e MIINFER_M31_Q8_GQA6_PACKED_LDS128=0 \
        -e MIINFER_M31_Q8_GQA6_DPP_REDUCE=0 -e MIINFER_M31_Q8_GQA6_BLOCK_SOFTMAX4=0 \
        -e MIINFER_M31_Q8_GQA6_WAVE64_KV_BROADCAST=0 \
        -e MIINFER_M31_Q8_GQA6_DUAL_KEY_WAVE64=0 \
        -e MIINFER_M31_Q8_GQA6_SPARSE_WINDOW=0 \
        -e MIINFER_M31_Q8_GQA6_PAIRED_TOKEN=0 -e MIINFER_M31_Q8_DOT4_ATTENTION=0 \
        -e MIINFER_M31_TRACE_ATTENTION_ROUTE=1 \
        -v "$model:/model.gguf:ro" -v "$prompt:/prompt.txt:ro" \
        -v /home/machinist/m31-gqa6-16lane-rewrite/build:/workspace:ro \
        -v "$out:/out:rw" "$image" \
        /workspace/miinfer-m31-fp16-full-model /model.gguf --q8-only --tokens 8192 \
        --prompt-file /prompt.txt --logits-out "/out/$label-logits.f32" \
        >"$out/$label.log" 2>&1 &
    pid=$!

    while kill -0 "$pid" 2>/dev/null; do
        temp=$(rocm-smi -d 2 --showtemp 2>/dev/null | awk '/Sensor junction/ {print $NF; exit}')
        if [ -n "$temp" ]; then
            printf '%s case=%s gpu2_junction_c=%s\n' "$(date --iso-8601=seconds)" "$label" "$temp" \
                >>"$out/thermal.log"
            if awk -v t="$temp" 'BEGIN {exit !(t >= 85)}'; then
                echo "thermal_stop case=$label gpu2_junction_c=$temp" >>"$out/thermal.log"
                podman kill "$name" >/dev/null 2>&1 || true
            elif awk -v t="$temp" 'BEGIN {exit !(t >= 80)}'; then
                echo "thermal_warning case=$label gpu2_junction_c=$temp" >>"$out/thermal.log"
            fi
        fi
        sleep 5
    done

    wait "$pid"
    rc=$?
    echo "case=$label exit_code=$rc finished=$(date --iso-8601=seconds)" \
        | tee -a "$out/status.log"
    return "$rc"
}

mkdir -p "$out"
echo "runner_started=$(date --iso-8601=seconds)" >"$out/status.log"
run_case control 0 || exit $?
[ -s "$out/control-logits.f32" ] || { echo "result=invalid reason=control_logits_missing" | tee -a "$out/status.log"; exit 3; }
grep -q 'prompt_tokens_processed=8192' "$out/control.log" \
    && grep -q 'q8_logits_finite=1 q8_logits_count=248320' "$out/control.log" \
    || { echo "result=invalid reason=control_output_gate" | tee -a "$out/status.log"; exit 3; }
run_case candidate 1 || exit $?
[ -s "$out/candidate-logits.f32" ] || { echo "result=invalid reason=candidate_logits_missing" | tee -a "$out/status.log"; exit 3; }
grep -q 'prompt_tokens_processed=8192' "$out/candidate.log" \
    && grep -q 'q8_logits_finite=1 q8_logits_count=248320' "$out/candidate.log" \
    || { echo "result=invalid reason=candidate_output_gate" | tee -a "$out/status.log"; exit 3; }
control_routes=$(grep -c '^miinfer_q8_gqa6_packed_lds_dispatch=1 .*direct_global=1' "$out/control.log" || true)
candidate_routes=$(grep -c '^miinfer_q8_gqa6_dim16_lanes_dispatch=1 ' "$out/candidate.log" || true)
if [ "$control_routes" -eq 0 ] || [ "$control_routes" -ne "$candidate_routes" ]; then
    echo "result=invalid reason=route_count_mismatch control=$control_routes candidate=$candidate_routes" | tee -a "$out/status.log"
    exit 3
fi
echo "route_count_gate=pass control=$control_routes candidate=$candidate_routes" | tee -a "$out/status.log"
control_seconds=$(sed -n 's/.*q8_only prefill_seconds=\([0-9.]*\).*/\1/p' "$out/control.log" | tail -n 1)
candidate_seconds=$(sed -n 's/.*q8_only prefill_seconds=\([0-9.]*\).*/\1/p' "$out/candidate.log" | tail -n 1)
if ! [[ "$control_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || ! [[ "$candidate_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "result=invalid reason=prefill_time_missing" | tee -a "$out/status.log"
    exit 3
fi
echo "measurement_gate=pass control_prefill_s=$control_seconds candidate_prefill_s=$candidate_seconds" | tee -a "$out/status.log"
sha256sum "$model" "$prompt" >"$out/inputs.sha256"
