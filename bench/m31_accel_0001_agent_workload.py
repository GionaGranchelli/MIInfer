#!/usr/bin/env python3
"""Run eight production-path coding-agent turns from an exact 64K token prompt."""

import argparse
import json
import math
import os
import struct
import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

import v2_0045_agent_workload as legacy


TARGET_PROMPT_TOKENS = 65536
TURN_PLANS = (
    ("docs/interactive-serving.md", 1, 80, "Trace how a chat request becomes prompt tokens and generated output."),
    ("README.md", 1, 30, "Map the project mission, supported target, and correctness constraints."),
    ("docs/architecture.md", 1, 30, "Explain the runtime layers and identify the model execution path."),
    ("include/miinfer/prefill_v2/model.hpp", 1, 40, "Inspect persistent state, generation statistics, and public APIs."),
    ("src/prefill_v2/model.cpp", 760, 800, "Review prefix restore, suffix prefill, and checkpoint capture."),
    ("src/openai_api.cpp", 280, 320, "Review ChatML construction and tool-result serialization."),
    ("tools/miinfer_cli.cpp", 5180, 5220, "Trace server tokenization and request telemetry."),
    ("docs/M30-0000-agent-workload-v1.md", 1, 40, "Summarize the prior workload contract and propose a bounded follow-up."),
)


class TokenCounter:
    """Keep the MIInfer tokenizer loaded and count production ChatML requests."""

    def __init__(self, executable, model):
        self.process = subprocess.Popen(
            [str(executable), str(model)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)

    def count(self, payload):
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        result = self.process.stdout.readline().strip()
        if not result or result.startswith("ERROR "):
            raise RuntimeError(f"MIInfer tokenizer failed: {result or 'counter exited'}")
        return int(result)

    def ids(self, payload):
        self.process.stdin.write("IDS " + json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        result = self.process.stdout.readline().strip().split()
        if not result or result[0] != "IDS":
            raise RuntimeError(f"MIInfer tokenizer failed to return IDs: {' '.join(result) or 'counter exited'}")
        ids = [int(value) for value in result[1:]]
        if not ids or any(value < 0 or value > 0xFFFFFFFF for value in ids):
            raise RuntimeError("MIInfer tokenizer returned invalid token IDs")
        return ids

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        return self.process.wait(timeout=10)


class ServerLogTail:
    """Read only new per-request server records to keep metrics attributed."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"server log does not exist: {self.path}")
        self.offset = self.path.stat().st_size

    def take_request(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                lines = [line.decode("utf-8", "replace") for line in stream.readlines()]
                self.offset = stream.tell()
            latency, detail = None, None
            for line in lines:
                if line.startswith("miinfer_request_latency "):
                    latency = json.loads(line.split(" ", 1)[1])
                elif line.startswith("miinfer_request {"):
                    detail = json.loads(line[len("miinfer_request "):])
            if latency and detail:
                if latency.get("request_id") != detail.get("request_id"):
                    raise RuntimeError("server telemetry request IDs do not match")
                return latency, detail
            time.sleep(0.05)
        raise RuntimeError("server did not write both request telemetry records")


def body_for(model, messages, tool_choice, max_tokens):
    return {
        "model": model, "messages": messages, "tools": legacy.TOOLS,
        "tool_choice": tool_choice, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "top_k": 1, "stream": True,
    }


def sse_choices(lines):
    finish_reason = None
    done = False
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            done = True
            break
        choices = (json.loads(data).get("choices") or [{}])
        choice = choices[0]
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
        yield choice
    if not done or finish_reason not in {"stop", "length", "tool_calls"}:
        raise RuntimeError("incomplete SSE response: missing [DONE] or accepted finish_reason")


def sse_self_check():
    valid = [
        'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    if len(list(sse_choices(valid))) != 2:
        raise RuntimeError("valid SSE self-check failed")
    require_no_repeated_token_loop({"repeated_token_loop_detected": False})
    for telemetry in ({"repeated_token_loop_detected": True}, {}):
        try:
            require_no_repeated_token_loop(telemetry)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("missing or detected repeated-token loop unexpectedly passed")
    require_disk_session_restore({
        "disk_session_restore": True, "disk_restore_prefix_tokens": 1100,
        "disk_restore_ms": 1.5,
    }, 1100)
    for telemetry in (
            {"disk_session_restore": False, "disk_restore_prefix_tokens": 1100, "disk_restore_ms": 1.5},
            {"disk_session_restore": True, "disk_restore_prefix_tokens": 1099, "disk_restore_ms": 1.5},
            {"disk_session_restore": True, "disk_restore_prefix_tokens": 1100, "disk_restore_ms": float("nan")},
            {"disk_session_restore": True, "disk_restore_prefix_tokens": 1100, "disk_restore_ms": -1.0},
            {}):
        try:
            require_disk_session_restore(telemetry, 1100)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("invalid disk-restore telemetry unexpectedly passed")
    try:
        list(sse_choices(valid[:-1]))
    except RuntimeError:
        pass
    else:
        raise RuntimeError("truncated SSE self-check unexpectedly passed")
    metric_detail = {"prompt_tokens": 1100, "generated_tokens": 2}
    if not request_metric_counts_match(1100, 2, metric_detail, 1100):
        raise RuntimeError("matching request metric self-check failed")
    if request_metric_counts_match(1100, 1, metric_detail, 1100):
        raise RuntimeError("generated-token metric mismatch unexpectedly passed")
    if request_metric_counts_match(1100, 2, metric_detail, 1099):
        raise RuntimeError("tokenizer/server prompt mismatch unexpectedly passed")
    if abs(aggregate_decode_rate([
            {"generated_tokens": 2, "decode_ms": 100, "decode_tokens_per_second": 10},
            {"generated_tokens": 5, "decode_ms": 200, "decode_tokens_per_second": 20},
    ]) - 50.0 / 3.0) > 1e-9:
        raise RuntimeError("decode-rate aggregation self-check failed")
    guard_samples = parse_guard_vram_samples(
        'M31_GPU_SAMPLE {"timestamp_unix_ns":100,"junction_temperature_c":40,"vram_used_bytes":1200}\n'
        'M31_GPU_SAMPLE {"timestamp_unix_ns":200,"junction_temperature_c":41,"vram_used_bytes":1800}\n'
        'M31_GPU_SAMPLE {"timestamp_unix_ns":300,"junction_temperature_c":42,"vram_used_bytes":900}\n')
    summary = request_vram_summary(
        {"request_start_unix_ns": 150, "request_end_unix_ns": 250,
         "gpu_before": {"vram_used_bytes": 1000},
         "gpu_after": {"vram_used_bytes": 1100}}, guard_samples)
    if (summary["peak_sampled_vram_bytes"] != 1800
            or summary["guard_telemetry_samples_in_window"] != 1
            or summary["boundary_vram_sample_count"] != 2
            or summary["vram_sample_count"] != 3):
        raise RuntimeError("guard telemetry request-window peak self-check failed")
    for invalid in ("", "M31_GPU_SAMPLE {bad json}\n",
                    ' prefix M31_GPU_SAMPLE {"timestamp_unix_ns":100,"junction_temperature_c":40,"vram_used_bytes":1}\n',
                    'M31_GPU_SAMPLE {"timestamp_unix_ns":100,"junction_temperature_c":40,"vram_used_bytes":-1}\n',
                    'M31_GPU_SAMPLE {"timestamp_unix_ns":100,"vram_used_bytes":1}\n'):
        try:
            parse_guard_vram_samples(invalid)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("absent or malformed guard telemetry unexpectedly passed")
    try:
        request_vram_summary(
            {"request_start_unix_ns": 150, "request_end_unix_ns": 250,
             "gpu_before": {"vram_used_bytes": 1000},
             "gpu_after": {"vram_used_bytes": 1100}},
            [{"timestamp_unix_ns": 100, "junction_temperature_c": 40,
              "vram_used_bytes": 1200}])
    except RuntimeError:
        pass
    else:
        raise RuntimeError("request without in-window guard samples unexpectedly passed")
    cold_gate = prefix_reuse_request_gates([{
        "server_latency": {"request_id": "cold", "reused_prefix_tokens": 0,
                            "suffix_tokens_dispatched": 1100},
        "server_request": {"prompt_tokens": 1100, "generated_tokens": 0,
                            "finish_reason": "stop", "gqa_kv_reused_tokens": 0,
                            "gdn_checkpoint_position": 0,
                            "runtime_context_capacity": 4096,
                            "gdn_final_position_min": 1100,
                            "gdn_final_position_max": 1100},
    }])
    if not cold_gate[0]["passed"]:
        raise RuntimeError("cold request position/capacity self-check failed")
    cold_start_checks = prefix_reuse_request_gates([
        {"server_latency": {"request_id": "stale-session", "reused_prefix_tokens": 1,
                            "suffix_tokens_dispatched": 1099},
         "server_request": {"prompt_tokens": 1100, "generated_tokens": 0,
                            "finish_reason": "stop", "gqa_kv_reused_tokens": 1,
                            "gdn_checkpoint_position": 1,
                            "runtime_context_capacity": 4096,
                            "gdn_final_position_min": 1100,
                            "gdn_final_position_max": 1100}},
        {"server_latency": {"request_id": "short-cold-dispatch", "reused_prefix_tokens": 0,
                            "suffix_tokens_dispatched": 1099},
         "server_request": {"prompt_tokens": 1100, "generated_tokens": 0,
                            "finish_reason": "stop", "gqa_kv_reused_tokens": 0,
                            "gdn_checkpoint_position": 0,
                            "runtime_context_capacity": 4096,
                            "gdn_final_position_min": 1100,
                            "gdn_final_position_max": 1100}},
    ])
    if [check["passed"] for check in cold_start_checks] != [False, False]:
        raise RuntimeError("cold-start dispatch self-check failed")
    reuse_checks = prefix_reuse_request_gates([
        {"server_latency": {"request_id": "first", "reused_prefix_tokens": 1000,
                            "suffix_tokens_dispatched": 100},
         "server_request": {"prompt_tokens": 1100, "generated_tokens": 2,
                             "finish_reason": "length", "gqa_kv_reused_tokens": 1000,
                             "gdn_checkpoint_position": 1000,
                             "runtime_context_capacity": 4096,
                             "gdn_final_position_min": 1101,
                             "gdn_final_position_max": 1101}},
        {"server_latency": {"request_id": "short-prefix", "reused_prefix_tokens": 1099,
                            "suffix_tokens_dispatched": 20},
         "server_request": {"prompt_tokens": 1120, "generated_tokens": 1,
                             "finish_reason": "length", "gqa_kv_reused_tokens": 1099,
                             "gdn_checkpoint_position": 1099,
                             "runtime_context_capacity": 4096,
                             "gdn_final_position_min": 1120,
                             "gdn_final_position_max": 1120}},
        {"server_latency": {"request_id": "empty-suffix", "reused_prefix_tokens": 1120,
                            "suffix_tokens_dispatched": 0},
         "server_request": {"prompt_tokens": 1120, "generated_tokens": 1,
                             "finish_reason": "length", "gqa_kv_reused_tokens": 1120,
                             "gdn_checkpoint_position": 1120,
                             "runtime_context_capacity": 4096,
                             "gdn_final_position_min": 1120,
                             "gdn_final_position_max": 1120}},
        {"server_latency": {"request_id": "bad-position", "reused_prefix_tokens": 1120,
                            "suffix_tokens_dispatched": 1},
         "server_request": {"prompt_tokens": 1121, "generated_tokens": 1,
                             "finish_reason": "length", "gqa_kv_reused_tokens": 1120,
                             "gdn_checkpoint_position": 1119,
                             "runtime_context_capacity": 4096,
                             "gdn_final_position_min": 1121,
                             "gdn_final_position_max": 1121}},
        {"server_latency": {"request_id": "capacity-overflow", "reused_prefix_tokens": 1121,
                            "suffix_tokens_dispatched": 1},
         "server_request": {"prompt_tokens": 1122, "generated_tokens": 1,
                             "finish_reason": "length", "gqa_kv_reused_tokens": 1121,
                             "gdn_checkpoint_position": 1121,
                             "runtime_context_capacity": 1121,
                             "gdn_final_position_min": 1122,
                             "gdn_final_position_max": 1122}},
        {"server_latency": {"request_id": "bad-final-position", "reused_prefix_tokens": 1122,
                            "suffix_tokens_dispatched": 1},
         "server_request": {"prompt_tokens": 1123, "generated_tokens": 1,
                             "finish_reason": "length", "gqa_kv_reused_tokens": 1122,
                             "gdn_checkpoint_position": 1122,
                             "runtime_context_capacity": 4096,
                             "gdn_final_position_min": 1122,
                             "gdn_final_position_max": 1122}},
    ], previous_prompt_tokens=1000)
    if [check["passed"] for check in reuse_checks] != [True, False, False, False, False, False]:
        raise RuntimeError("per-request prefix reuse self-check failed")
    bad_dispatch = prefix_reuse_request_gates([{
        "server_latency": {"request_id": "dispatch-gap", "reused_prefix_tokens": 1000,
                            "suffix_tokens_dispatched": 119},
        "server_request": {"prompt_tokens": 1120, "generated_tokens": 1,
                            "finish_reason": "length", "gqa_kv_reused_tokens": 1000,
                            "gdn_checkpoint_position": 1000,
                            "runtime_context_capacity": 4096,
                            "gdn_final_position_min": 1121,
                            "gdn_final_position_max": 1121},
    }], previous_prompt_tokens=1000)
    if bad_dispatch[0]["passed"] or bad_dispatch[0]["dispatch_accounting_passed"]:
        raise RuntimeError("incomplete prompt dispatch unexpectedly passed")
    if prefix_reuse_request_gates([], previous_prompt_tokens=1000)[0]["passed"]:
        raise RuntimeError("missing prefix reuse telemetry unexpectedly passed")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "workload.json"
        write_result(path, {"active_turn": 1, "active_phase": "planned_read_tool_call"})
        if json.loads(path.read_text(encoding="utf-8"))["active_turn"] != 1:
            raise RuntimeError("atomic workload-artifact self-check failed")
        failed = json.loads(path.read_text(encoding="utf-8"))
        mark_workload_failed(path, failed, RuntimeError("injected self-check failure"))
        failure = json.loads(path.read_text(encoding="utf-8"))
        if failure.get("status") != "failed" or failure.get("failure", {}).get("message") != "injected self-check failure":
            raise RuntimeError("failure-artifact self-check failed")
        payload = body_for("test", [{"role": "user", "content": "frozen"}], "required", 2)
        record = save_request_input(path, 1, 1, payload, list(range(7)), "self-check", 1, "b" * 64)
        saved = path.parent / record["path"]
        token_ids = json.loads((path.parent / record["token_ids_path"]).read_text(encoding="utf-8"))
        if (saved.read_bytes() != json.dumps(payload).encode("utf-8")
                or legacy.sha256(saved.read_bytes()) != record["sha256"]
                or json.loads(saved.read_text(encoding="utf-8")) != payload
                or token_ids["request_sha256"] != record["sha256"]
                or token_ids["workload_id"] != "self-check"
                or token_ids["sequence_index"] != 1
                or token_ids["model_sha256"] != "b" * 64
                or token_ids["token_count"] != 7
                or token_ids["token_ids"] != list(range(7))
                or token_ids["token_ids_sha256"] != legacy.sha256(struct.pack("<7I", *range(7)))):
            raise RuntimeError("request-input artifact round-trip self-check failed")


def prefix_reuse_request_gates(requests, previous_prompt_tokens=None):
    if not requests:
        return [{"request_index": None, "request_id": None,
                 "expected_reused_prefix_tokens": previous_prompt_tokens,
                 "reused_prefix_tokens": 0, "suffix_tokens_dispatched": 0,
                 "passed": False}]
    checks = []
    expected_prefix_tokens = previous_prompt_tokens
    for request_index, request in enumerate(requests, 1):
        latency = request["server_latency"]
        detail = request["server_request"]
        reused = latency.get("reused_prefix_tokens", 0)
        suffix = latency.get("suffix_tokens_dispatched", 0)
        prompt_tokens = detail.get("prompt_tokens")
        gqa_reused = detail.get("gqa_kv_reused_tokens")
        gdn_position = detail.get("gdn_checkpoint_position")
        runtime_capacity = detail.get("runtime_context_capacity")
        generated_tokens = detail.get("generated_tokens")
        finish_reason = detail.get("finish_reason")
        gdn_final_min = detail.get("gdn_final_position_min")
        gdn_final_max = detail.get("gdn_final_position_max")
        valid_prompt_tokens = type(prompt_tokens) is int and prompt_tokens > 0
        dispatch_accounting_passed = (
            valid_prompt_tokens and type(reused) is int and type(suffix) is int
            and 0 <= reused <= prompt_tokens and suffix >= 0
            and reused + suffix == prompt_tokens)
        state_position_passed = (
            type(gqa_reused) is int and type(gdn_position) is int
            and gqa_reused == reused and gdn_position == reused
            and (expected_prefix_tokens is None or reused == expected_prefix_tokens))
        context_capacity_passed = (
            type(runtime_capacity) is int and valid_prompt_tokens
            and runtime_capacity >= prompt_tokens)
        valid_generated_tokens = type(generated_tokens) is int and generated_tokens >= 0
        expected_final_position_min = None
        expected_final_position_max = None
        if valid_prompt_tokens and valid_generated_tokens and isinstance(finish_reason, str):
            expected_final_position_min = prompt_tokens + max(0, generated_tokens - 1)
            expected_final_position_max = prompt_tokens + generated_tokens
        gdn_final_position_passed = (
            expected_final_position_min is not None
            and expected_final_position_max is not None
            and type(gdn_final_min) is int and type(gdn_final_max) is int
            and gdn_final_min == gdn_final_max
            and expected_final_position_min <= gdn_final_min <= expected_final_position_max)
        cold_start_passed = (expected_prefix_tokens is not None or (
            valid_prompt_tokens and reused == 0 and suffix == prompt_tokens
            and gqa_reused == 0 and gdn_position == 0))
        checks.append({
            "request_index": request_index,
            "request_id": latency.get("request_id"),
            "expected_reused_prefix_tokens": expected_prefix_tokens,
            "reused_prefix_tokens": reused,
            "suffix_tokens_dispatched": suffix,
            "dispatch_accounting_passed": dispatch_accounting_passed,
            "gqa_kv_reused_tokens": gqa_reused,
            "gdn_checkpoint_position": gdn_position,
            "runtime_context_capacity": runtime_capacity,
            "state_position_passed": state_position_passed,
            "context_capacity_passed": context_capacity_passed,
            "expected_gdn_final_position_min": expected_final_position_min,
            "expected_gdn_final_position_max": expected_final_position_max,
            "gdn_final_position_min": gdn_final_min,
            "gdn_final_position_max": gdn_final_max,
            "gdn_final_position_passed": gdn_final_position_passed,
            "cold_start_passed": cold_start_passed,
            "passed": valid_prompt_tokens and cold_start_passed and state_position_passed
                and dispatch_accounting_passed and context_capacity_passed
                and gdn_final_position_passed
                and (expected_prefix_tokens is None or suffix > 0),
        })
        expected_prefix_tokens = prompt_tokens if valid_prompt_tokens else -1
    return checks


def parse_guard_vram_samples(text):
    samples = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("M31_GPU_SAMPLE "):
            if "M31_GPU_SAMPLE" in line:
                raise RuntimeError(f"malformed guarded telemetry record on line {line_number}")
            continue
        try:
            sample = json.loads(line[len("M31_GPU_SAMPLE "):])
        except json.JSONDecodeError as error:
            raise RuntimeError(f"malformed guarded telemetry on line {line_number}") from error
        timestamp = sample.get("timestamp_unix_ns")
        temperature = sample.get("junction_temperature_c")
        used = sample.get("vram_used_bytes")
        if (type(timestamp) is not int or timestamp < 0
                or type(temperature) not in (int, float) or not math.isfinite(temperature)
                or temperature < 0
                or type(used) is not int or used < 0):
            raise RuntimeError(f"invalid guarded telemetry fields on line {line_number}")
        samples.append({"timestamp_unix_ns": timestamp,
                        "junction_temperature_c": temperature,
                        "vram_used_bytes": used})
    if not samples:
        raise RuntimeError("guarded telemetry contains no valid VRAM samples")
    return samples


def request_vram_summary(request, guard_samples):
    start = request.get("request_start_unix_ns")
    end = request.get("request_end_unix_ns")
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise RuntimeError("request has invalid Unix-nanosecond window")
    in_window = [sample for sample in guard_samples
                 if start <= sample["timestamp_unix_ns"] <= end]
    if not in_window:
        raise RuntimeError("guarded telemetry has no VRAM sample in the request window")
    boundary = [snapshot["vram_used_bytes"]
                for snapshot in (request.get("gpu_before"), request.get("gpu_after"))
                if isinstance(snapshot, dict)
                and type(snapshot.get("vram_used_bytes")) is int
                and snapshot["vram_used_bytes"] >= 0]
    window_values = [sample["vram_used_bytes"] for sample in in_window]
    return {
        "peak_sampled_vram_bytes": max(boundary + window_values),
        "guard_telemetry_samples_in_window": len(in_window),
        "boundary_vram_sample_count": len(boundary),
        "vram_sample_count": len(boundary) + len(window_values),
        "vram_sample_sources": ["request_boundary", "guard_telemetry_window"],
    }


def write_result(path, result):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_request_input(output_path, turn, request, payload, token_ids,
                       workload_id, sequence_index, model_sha256):
    data = json.dumps(payload).encode("utf-8")
    token_bytes = struct.pack(f"<{len(token_ids)}I", *token_ids)
    artifact_dir = output_path.parent / f"{output_path.stem}.inputs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"turn-{turn:02d}-request-{request:02d}.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=artifact_dir,
                                         prefix=f".{artifact_path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, artifact_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    token_ids_path = artifact_dir / f"turn-{turn:02d}-request-{request:02d}.tokens.json"
    token_ids_data = {
        "schema": "miinfer-chatml-token-ids-v2",
        "model": payload.get("model"),
        "model_sha256": model_sha256,
        "workload_id": workload_id,
        "sequence_index": sequence_index,
        "turn": turn,
        "request": request,
        "request_sha256": legacy.sha256(data),
        "token_count": len(token_ids),
        "token_ids_sha256": legacy.sha256(token_bytes),
        "token_ids": token_ids,
    }
    write_result(token_ids_path, token_ids_data)
    return {
        "path": str(artifact_path.relative_to(output_path.parent)),
        "sha256": legacy.sha256(data),
        "bytes": len(data),
        "token_ids_path": str(token_ids_path.relative_to(output_path.parent)),
        "token_ids_sha256": token_ids_data["token_ids_sha256"],
        "workload_id": workload_id,
        "sequence_index": sequence_index,
        "tokenizer_prompt_tokens": len(token_ids),
    }


def mark_workload_failed(output_path, result, error):
    result["status"] = "failed"
    result["workload_complete"] = False
    result["failure"] = {
        "type": type(error).__name__, "message": str(error),
        "active_turn": result.get("active_turn"),
        "active_phase": result.get("active_phase"),
        "active_request": result.get("active_request"),
    }
    write_result(output_path, result)


def request_metric_counts_match(prompt_delta, generated_delta, detail, tokenizer_prompt_tokens=None):
    prompt_tokens = detail.get("prompt_tokens")
    generated_tokens = detail.get("generated_tokens")
    return (type(prompt_delta) is int and type(generated_delta) is int
            and type(prompt_tokens) is int and type(generated_tokens) is int
            and prompt_delta == prompt_tokens and generated_delta == generated_tokens
            and (tokenizer_prompt_tokens is None or tokenizer_prompt_tokens == prompt_tokens))


def aggregate_decode_rate(requests):
    timed = [(row["decode_tokens_per_second"], row["decode_ms"])
             for row in requests
             if row.get("decode_tokens_per_second", 0) > 0 and row.get("decode_ms", 0) > 0]
    elapsed_ms = sum(ms for _, ms in timed)
    return sum(rate * ms for rate, ms in timed) / elapsed_ms if elapsed_ms else 0.0


def source_evidence(root):
    parts, manifest = [], []
    for relative, limit in legacy.M30_SOURCES:
        data = (root / relative).read_bytes()
        excerpt = data[:limit]
        manifest.append({"path": relative, "source_sha256": legacy.sha256(data),
                         "excerpt_sha256": legacy.sha256(excerpt),
                         "excerpt_bytes": len(excerpt)})
        text = excerpt.decode("utf-8", "replace")
        parts.append(f"\n--- read_file tool result: {relative} ---\n{text}")
    return "\n".join(parts), manifest


def initial_messages(evidence):
    system = ("You are a careful coding-agent assistant working in MIInfer. Use repository "
              "evidence, distinguish facts from assumptions, and preserve the prior conversation.")
    user = (
        "Deterministic repository investigation transcript. The following repository excerpts "
        "are context already present at session start.\n\n" + evidence +
        "\n\nTask: review MIInfer's production persistent-context serving path. Start by making "
        "exactly one read_file tool call for docs/interactive-serving.md lines 1-80. Do not "
        "modify files. Use the evidence above as prior coding-agent context."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def exact_initial_request(root, model_id, counter):
    source, manifest = source_evidence(root)
    evidence = ""
    messages = initial_messages(evidence)
    target_payload = body_for(model_id, messages, "required", 192)
    count = counter.count(target_payload)
    if count >= TARGET_PROMPT_TOKENS:
        raise RuntimeError("fixed coding-agent instructions already exceed the 65,536-token target")

    # Re-read actual source excerpts as deterministic tool-result history until the target is bracketed.
    unit = "\n\nDeterministic prior repository reads (cycle of tracked MIInfer source):\n" + source
    while count < TARGET_PROMPT_TOKENS:
        evidence += unit
        messages = initial_messages(evidence)
        count = counter.count(body_for(model_id, messages, "required", 192))
    if count == TARGET_PROMPT_TOKENS:
        return messages, manifest, count

    low, high = 0, len(evidence)
    nearest = (count, len(evidence))
    while low <= high:
        middle = (low + high) // 2
        candidate = initial_messages(evidence[:middle])
        candidate_count = counter.count(body_for(model_id, candidate, "required", 192))
        if candidate_count == TARGET_PROMPT_TOKENS:
            return candidate, manifest, candidate_count
        if abs(candidate_count - TARGET_PROMPT_TOKENS) < abs(nearest[0] - TARGET_PROMPT_TOKENS):
            nearest = (candidate_count, middle)
        if candidate_count < TARGET_PROMPT_TOKENS:
            low = middle + 1
        else:
            high = middle - 1

    # Token merges can shift at excerpt boundaries; inspect nearby source-text prefixes exactly.
    begin = max(0, min(low, high) - 128)
    end = min(len(evidence), max(low, high) + 128)
    for length in range(begin, end + 1):
        candidate = initial_messages(evidence[:length])
        candidate_count = counter.count(body_for(model_id, candidate, "required", 192))
        if candidate_count == TARGET_PROMPT_TOKENS:
            return candidate, manifest, candidate_count
    raise RuntimeError(
        "could not form an exactly 65,536-token ChatML prompt from repository text; "
        f"nearest checked count was {nearest[0]}. No token-ID or byte-count padding was used."
    )


def stream_completion(url, key, payload, request_input, server_log, guard_telemetry,
                      required_disk_restore_prefix_tokens=None):
    tokenizer_prompt_tokens = request_input["tokenizer_prompt_tokens"]
    headers = {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if required_disk_restore_prefix_tokens is not None:
        if type(required_disk_restore_prefix_tokens) is not int or required_disk_restore_prefix_tokens <= 0:
            raise RuntimeError("required disk-restore prefix must be a positive integer")
        headers["X-MIInfer-Require-Disk-Session-Restore"] = "1"
    before = legacy.metrics(url, key)
    gpu_before = legacy.gpu_snapshot()
    started = time.perf_counter()
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    content, reasoning, calls = [], [], {}
    first_delta_ms = None
    request_start_unix_ns = time.time_ns()
    with urllib.request.urlopen(request, timeout=1800) as response:
        for choice in sse_choices(response):
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                if first_delta_ms is None:
                    first_delta_ms = (time.perf_counter() - started) * 1000.0
            content.append(delta.get("content", ""))
            reasoning.append(delta.get("reasoning_content", ""))
            for item in delta.get("tool_calls", []):
                call = calls.setdefault(item["index"], {"id": "", "type": "function",
                    "function": {"name": "", "arguments": ""}})
                if item.get("id"):
                    call["id"] = item["id"]
                function = item.get("function", {})
                call["function"]["name"] += function.get("name", "")
                call["function"]["arguments"] += function.get("arguments", "")
    request_end_unix_ns = time.time_ns()
    wall_ms = (time.perf_counter() - started) * 1000.0
    after = legacy.metrics(url, key)
    request_count = after["miinfer_inference_requests_total"] - before["miinfer_inference_requests_total"]
    if request_count != 1:
        raise RuntimeError(f"expected one server inference request, observed {request_count}")
    latency, detail = server_log.take_request()
    require_no_repeated_token_loop(detail)
    if required_disk_restore_prefix_tokens is not None:
        require_disk_session_restore(detail, required_disk_restore_prefix_tokens)
    prompt_tokens = after["miinfer_prompt_tokens_total"] - before["miinfer_prompt_tokens_total"]
    generated_tokens = after["miinfer_generated_tokens_total"] - before["miinfer_generated_tokens_total"]
    if not request_metric_counts_match(prompt_tokens, generated_tokens, detail,
                                       tokenizer_prompt_tokens):
        raise RuntimeError("tokenizer, metrics, and server request token counts disagree")
    gpu_after = legacy.gpu_snapshot()
    vram_summary = request_vram_summary({
        "request_start_unix_ns": request_start_unix_ns,
        "request_end_unix_ns": request_end_unix_ns,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }, guard_telemetry())
    return {
        "content": "".join(content), "reasoning_content": "".join(reasoning),
        "tool_calls": [calls[index] for index in sorted(calls)],
        "client_ttft_ms": first_delta_ms, "client_wall_ms": wall_ms,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "request_count": request_count, "server_latency": latency,
        "server_request": detail, "gpu_before": gpu_before,
        "gpu_after": gpu_after, "request_input": request_input,
        "request_start_unix_ns": request_start_unix_ns,
        "request_end_unix_ns": request_end_unix_ns,
        "vram_samples": vram_summary,
    }


def require_no_repeated_token_loop(server_request):
    if server_request.get("repeated_token_loop_detected") is not False:
        raise RuntimeError("server did not prove the request was free of a repeated-token loop")


def require_disk_session_restore(server_request, expected_prefix_tokens):
    prefix_tokens = server_request.get("disk_restore_prefix_tokens")
    restore_ms = server_request.get("disk_restore_ms")
    if (server_request.get("disk_session_restore") is not True
            or type(prefix_tokens) is not int or prefix_tokens != expected_prefix_tokens
            or type(restore_ms) not in (int, float) or not math.isfinite(restore_ms) or restore_ms < 0):
        raise RuntimeError("server did not prove the required persisted-session prefix was restored from disk")


def require_planned_read(response, path, start_line, end_line):
    calls = response["tool_calls"]
    if len(calls) != 1 or calls[0]["function"]["name"] != "read_file":
        raise RuntimeError("expected exactly one planned read_file tool call")
    arguments = json.loads(calls[0]["function"]["arguments"] or "{}")
    if (arguments.get("path") != path
            or int(arguments.get("start_line", 1)) != start_line
            or int(arguments.get("end_line", start_line + 160)) != end_line):
        raise RuntimeError(f"agent tool call did not match planned read: {path}:{start_line}-{end_line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="pinned Qwen3.8 GGUF used by MIInfer tokenizer")
    parser.add_argument("--token-counter", default="build/miinfer-m31-agent-token-count")
    parser.add_argument("--server-log", help="read-only MIInfer server stderr log")
    parser.add_argument("--guard-telemetry", help="run-guarded.sh telemetry.log with structured samples")
    parser.add_argument("--url", default="http://127.0.0.1:8087/v1/chat/completions")
    parser.add_argument("--api-key", default="miinfer")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--output", default="results/m31-accel-0001/agent-workload.json")
    parser.add_argument("--self-check", action="store_true", help="construct and count the exact initial prompt; no GPU request")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model_path = (root / args.model_path).resolve()
    counter_path = (root / args.token_counter).resolve()
    counter = TokenCounter(counter_path, model_path)
    result = None
    output_path = None
    try:
        messages, manifest, exact_count = exact_initial_request(root, args.model, counter)
        assert len(TURN_PLANS) == 8 and all(start > 0 and end >= start
                                             and (root / path).is_file()
                                             for path, start, end, _ in TURN_PLANS)
        assert exact_count == TARGET_PROMPT_TOKENS
        if args.self_check:
            initial_ids = counter.ids(body_for(args.model, messages, "required", 192))
            if len(initial_ids) != exact_count:
                raise RuntimeError("token-ID self-check disagrees with the exact initial prompt count")
            sse_self_check()
            print(json.dumps({"self_check": "PASS", "turns": len(TURN_PLANS),
                              "initial_prompt_tokens": exact_count,
                              "initial_token_ids": len(initial_ids),
                              "sse_completion": "PASS", "atomic_artifact": "PASS",
                              "source_files": len(manifest)}, sort_keys=True))
            return 0

        if not args.server_log:
            parser.error("--server-log is required unless --self-check is selected")
        if not args.guard_telemetry:
            parser.error("--guard-telemetry is required unless --self-check is selected")
        model_sha256 = os.environ.get("M31_MODEL_SHA256")
        if not model_sha256 or len(model_sha256) != 64:
            raise RuntimeError("guarded launcher must provide the verified M31_MODEL_SHA256")
        workload_id = uuid.uuid4().hex
        server_log = ServerLogTail(root / args.server_log)
        guard_telemetry_path = root / args.guard_telemetry
        if not guard_telemetry_path.is_file():
            raise RuntimeError(f"guarded telemetry does not exist: {guard_telemetry_path}")
        # Fail before inference if the host guard did not provide its preflight sample.
        parse_guard_vram_samples(guard_telemetry_path.read_text(encoding="utf-8"))
        guard_telemetry = lambda: parse_guard_vram_samples(
            guard_telemetry_path.read_text(encoding="utf-8"))
        output_path = root / args.output
        result = {
            "workload": "m31-accel-0001-agent64k-v1",
            "workload_id": workload_id,
            "qualification_sha": os.environ.get("M31_SOURCE_SHA") or subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                text=True, check=True).stdout.strip(),
            "initial_context_target_tokens": TARGET_PROMPT_TOKENS,
            "initial_context_tokenizer_verified": exact_count,
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "runner_sha256": legacy.sha256(Path(__file__).read_bytes()),
            "expected_turn_count": len(TURN_PLANS), "turns_completed": 0,
            "workload_complete": False, "status": "in_progress",
            "active_turn": None, "active_phase": None, "requests_completed": 0,
            "in_progress_turn": None, "active_request": None,
            "prefix_reuse_correctness": "UNQUALIFIED: no matched cold replay",
            "source_manifest": manifest, "turns": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_result(output_path, result)
        previous_prompt_tokens = None
        request_sequence = 0
        for index, (tool_path, start_line, end_line, task) in enumerate(TURN_PLANS, 1):
            turn = {"turn": index, "requests": [], "tool_results": []}
            result["active_turn"] = index
            result["in_progress_turn"] = turn
            if index > 1:
                messages.append({"role": "user", "content":
                    f"Turn {index}: continue the same MIInfer coding-agent investigation. Make exactly one "
                    f"read_file call for {tool_path} lines {start_line}-{end_line}; do not answer in prose yet."})
            result["active_phase"] = "planned_read_tool_call"
            payload = body_for(args.model, messages, "required", 192)
            request_sequence += 1
            request_input = save_request_input(output_path, index, 1, payload,
                                               counter.ids(payload), workload_id,
                                               request_sequence, model_sha256)
            if index == 1 and request_input["tokenizer_prompt_tokens"] != exact_count:
                raise RuntimeError("frozen initial request count changed after prompt construction")
            result["active_request"] = request_input
            write_result(output_path, result)
            call_response = stream_completion(args.url, args.api_key, payload,
                                              request_input, server_log, guard_telemetry)
            turn["requests"].append(call_response)
            result["requests_completed"] += 1
            result["active_request"] = None
            write_result(output_path, result)
            require_planned_read(call_response, tool_path, start_line, end_line)
            if index == 1 and call_response["prompt_tokens"] != exact_count:
                raise RuntimeError("live server token count differs from the exact MIInfer tokenizer check")
            turn["tool_results"].extend(legacy.append_tool_exchange(root, messages, call_response))
            messages.append({"role": "user", "content":
                f"Turn {index} task: {task} Use source references and distinguish verified behavior "
                "from inference. Give at least five numbered observations and a concrete next step, "
                "with at least 100 words."})
            completion = None
            for attempt in range(3):
                result["active_phase"] = f"report_completion_{attempt + 1}"
                payload = body_for(args.model, messages, "auto", 256)
                request_sequence += 1
                required_disk_restore_prefix_tokens = None
                if request_sequence == 2:
                    required_disk_restore_prefix_tokens = turn["requests"][0]["server_request"]["prompt_tokens"]
                request_input = save_request_input(
                    output_path, index, attempt + 2, payload, counter.ids(payload),
                    workload_id, request_sequence, model_sha256)
                if required_disk_restore_prefix_tokens is not None:
                    request_input["required_disk_restore_prefix_tokens"] = required_disk_restore_prefix_tokens
                result["active_request"] = request_input
                write_result(output_path, result)
                completion = stream_completion(args.url, args.api_key, payload,
                                               request_input, server_log, guard_telemetry,
                                               required_disk_restore_prefix_tokens)
                turn["requests"].append(completion)
                result["requests_completed"] += 1
                result["active_request"] = None
                write_result(output_path, result)
                if not completion["tool_calls"]:
                    break
                turn["tool_results"].extend(legacy.append_tool_exchange(root, messages, completion))
                messages.append({"role": "user", "content":
                    "Use the tool result and finish the evidence-based report in at least 100 words."})
            if completion is None or completion["tool_calls"]:
                raise RuntimeError(f"turn {index} did not produce a final assistant response")
            messages.append({"role": "assistant", "content": completion["content"]})
            detail = [item["server_request"] for item in turn["requests"]]
            latency = [item["server_latency"] for item in turn["requests"]]
            vram_summaries = [item["vram_samples"] for item in turn["requests"]]
            generated = sum(row["generated_tokens"] for row in detail)
            reuse_checks = prefix_reuse_request_gates(
                turn["requests"], previous_prompt_tokens=previous_prompt_tokens)
            previous_prompt_tokens = detail[-1]["prompt_tokens"]
            turn.update({
                "context_tokens": detail[0]["prompt_tokens"],
                "generated_tokens_total": generated,
                "ttft_ms": turn["requests"][0]["client_ttft_ms"],
                "decode_tok_per_sec": aggregate_decode_rate(detail),
                "wall_ms": sum(item["client_wall_ms"] for item in turn["requests"]),
                "server_total_ms": sum(row["total_ms"] for row in detail),
                "reused_prefix_tokens": sum(row["reused_prefix_tokens"] for row in latency),
                "suffix_tokens_dispatched": sum(row["suffix_tokens_dispatched"] for row in latency),
                "replayed_tokens": sum(row["suffix_tokens_dispatched"] for row in latency),
                "peak_sampled_vram_bytes": max(
                    summary["peak_sampled_vram_bytes"] for summary in vram_summaries),
                "guard_telemetry_samples_in_window": sum(
                    summary["guard_telemetry_samples_in_window"] for summary in vram_summaries),
                "boundary_vram_sample_count": sum(
                    summary["boundary_vram_sample_count"] for summary in vram_summaries),
                "vram_sample_count": sum(summary["vram_sample_count"] for summary in vram_summaries),
                "vram_sample_sources": ["request_boundary", "guard_telemetry_window"],
                "external_vram_samples_complete": all(
                    summary["guard_telemetry_samples_in_window"] > 0
                    and summary["boundary_vram_sample_count"] == 2
                    for summary in vram_summaries),
                "prefix_reuse_request_gates": reuse_checks,
                "prefix_reuse_gate": bool(reuse_checks) and all(check["passed"] for check in reuse_checks),
                "minimum_generation_gate": generated >= 128,
                "minimum_visible_answer_gate": len(completion["content"]) >= 300,
            })
            result["turns"].append(turn)
            result["turns_completed"] = len(result["turns"])
            result["active_turn"] = None
            result["active_phase"] = None
            result["in_progress_turn"] = None
            write_result(output_path, result)
        result["workload_complete"] = len(result["turns"]) == len(TURN_PLANS)
        result["status"] = "complete" if result["workload_complete"] else "incomplete"
        result["all_turns_minimum_generation_gate"] = all(
            turn["minimum_generation_gate"] for turn in result["turns"])
        result["all_turns_visible_answer_gate"] = all(
            turn["minimum_visible_answer_gate"] for turn in result["turns"])
        result["all_turns_external_vram_samples_complete"] = all(
            turn["external_vram_samples_complete"] for turn in result["turns"])
        result["all_turns_prefix_reuse_gate"] = all(
            turn["prefix_reuse_gate"] for turn in result["turns"])
        result["conversation_message_count"] = len(messages)
        write_result(output_path, result)
        print(json.dumps(result, indent=2))
        return 0 if (result["all_turns_minimum_generation_gate"]
                     and result["all_turns_visible_answer_gate"]
                     and result["all_turns_external_vram_samples_complete"]
                     and result["all_turns_prefix_reuse_gate"]) else 1
    except Exception as error:
        if output_path is not None and result is not None and output_path.exists():
            mark_workload_failed(output_path, result, error)
        raise
    finally:
        counter.close()


if __name__ == "__main__":
    raise SystemExit(main())
