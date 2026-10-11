#!/usr/bin/env python3
"""Run eight production-path coding-agent turns from an exact 64K token prompt."""

import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.request
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
    try:
        list(sse_choices(valid[:-1]))
    except RuntimeError:
        pass
    else:
        raise RuntimeError("truncated SSE self-check unexpectedly passed")
    cold_gate = prefix_reuse_request_gates([{
        "server_latency": {"request_id": "cold", "reused_prefix_tokens": 0,
                            "suffix_tokens_dispatched": 1100},
        "server_request": {"prompt_tokens": 1100, "gqa_kv_reused_tokens": 0,
                            "gdn_checkpoint_position": 0,
                            "runtime_context_capacity": 4096},
    }])
    if not cold_gate[0]["passed"]:
        raise RuntimeError("cold request position/capacity self-check failed")
    reuse_checks = prefix_reuse_request_gates([
        {"server_latency": {"request_id": "first", "reused_prefix_tokens": 1000,
                            "suffix_tokens_dispatched": 100},
         "server_request": {"prompt_tokens": 1100, "gqa_kv_reused_tokens": 1000,
                             "gdn_checkpoint_position": 1000,
                             "runtime_context_capacity": 4096}},
        {"server_latency": {"request_id": "short-prefix", "reused_prefix_tokens": 1099,
                            "suffix_tokens_dispatched": 20},
         "server_request": {"prompt_tokens": 1120, "gqa_kv_reused_tokens": 1099,
                             "gdn_checkpoint_position": 1099,
                             "runtime_context_capacity": 4096}},
        {"server_latency": {"request_id": "empty-suffix", "reused_prefix_tokens": 1120,
                            "suffix_tokens_dispatched": 0},
         "server_request": {"prompt_tokens": 1120, "gqa_kv_reused_tokens": 1120,
                             "gdn_checkpoint_position": 1120,
                             "runtime_context_capacity": 4096}},
        {"server_latency": {"request_id": "bad-position", "reused_prefix_tokens": 1120,
                            "suffix_tokens_dispatched": 1},
         "server_request": {"prompt_tokens": 1121, "gqa_kv_reused_tokens": 1120,
                             "gdn_checkpoint_position": 1119,
                             "runtime_context_capacity": 4096}},
        {"server_latency": {"request_id": "capacity-overflow", "reused_prefix_tokens": 1121,
                            "suffix_tokens_dispatched": 1},
         "server_request": {"prompt_tokens": 1122, "gqa_kv_reused_tokens": 1121,
                             "gdn_checkpoint_position": 1121,
                             "runtime_context_capacity": 1121}},
    ], previous_prompt_tokens=1000)
    if [check["passed"] for check in reuse_checks] != [True, False, False, False, False]:
        raise RuntimeError("per-request prefix reuse self-check failed")
    if prefix_reuse_request_gates([], previous_prompt_tokens=1000)[0]["passed"]:
        raise RuntimeError("missing prefix reuse telemetry unexpectedly passed")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "workload.json"
        write_result(path, {"active_turn": 1, "active_phase": "planned_read_tool_call"})
        if json.loads(path.read_text(encoding="utf-8"))["active_turn"] != 1:
            raise RuntimeError("atomic workload-artifact self-check failed")


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
        valid_prompt_tokens = type(prompt_tokens) is int and prompt_tokens > 0
        state_position_passed = (
            type(gqa_reused) is int and type(gdn_position) is int
            and gqa_reused == reused and gdn_position == reused
            and (expected_prefix_tokens is None or reused == expected_prefix_tokens))
        context_capacity_passed = (
            type(runtime_capacity) is int and valid_prompt_tokens
            and runtime_capacity >= prompt_tokens)
        checks.append({
            "request_index": request_index,
            "request_id": latency.get("request_id"),
            "expected_reused_prefix_tokens": expected_prefix_tokens,
            "reused_prefix_tokens": reused,
            "suffix_tokens_dispatched": suffix,
            "gqa_kv_reused_tokens": gqa_reused,
            "gdn_checkpoint_position": gdn_position,
            "runtime_context_capacity": runtime_capacity,
            "state_position_passed": state_position_passed,
            "context_capacity_passed": context_capacity_passed,
            "passed": valid_prompt_tokens and state_position_passed
                and context_capacity_passed
                and (expected_prefix_tokens is None or suffix > 0),
        })
        expected_prefix_tokens = prompt_tokens if valid_prompt_tokens else -1
    return checks


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


def stream_completion(url, key, model, messages, tool_choice, max_tokens, server_log):
    payload = body_for(model, messages, tool_choice, max_tokens)
    before = legacy.metrics(url, key)
    gpu_before = legacy.gpu_snapshot()
    started = time.perf_counter()
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Accept": "text/event-stream",
    })
    content, reasoning, calls = [], [], {}
    first_delta_ms = None
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
    wall_ms = (time.perf_counter() - started) * 1000.0
    after = legacy.metrics(url, key)
    request_count = after["miinfer_inference_requests_total"] - before["miinfer_inference_requests_total"]
    if request_count != 1:
        raise RuntimeError(f"expected one server inference request, observed {request_count}")
    latency, detail = server_log.take_request()
    prompt_tokens = after["miinfer_prompt_tokens_total"] - before["miinfer_prompt_tokens_total"]
    if prompt_tokens != detail["prompt_tokens"]:
        raise RuntimeError("metrics prompt-token delta disagrees with its server request record")
    return {
        "content": "".join(content), "reasoning_content": "".join(reasoning),
        "tool_calls": [calls[index] for index in sorted(calls)],
        "client_ttft_ms": first_delta_ms, "client_wall_ms": wall_ms,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": after["miinfer_generated_tokens_total"] - before["miinfer_generated_tokens_total"],
        "request_count": request_count, "server_latency": latency,
        "server_request": detail, "gpu_before": gpu_before,
        "gpu_after": legacy.gpu_snapshot(),
    }


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
    try:
        messages, manifest, exact_count = exact_initial_request(root, args.model, counter)
        assert len(TURN_PLANS) == 8 and all(start > 0 and end >= start
                                             and (root / path).is_file()
                                             for path, start, end, _ in TURN_PLANS)
        assert exact_count == TARGET_PROMPT_TOKENS
        if args.self_check:
            sse_self_check()
            print(json.dumps({"self_check": "PASS", "turns": len(TURN_PLANS),
                              "initial_prompt_tokens": exact_count,
                              "sse_completion": "PASS", "atomic_artifact": "PASS",
                              "source_files": len(manifest)}, sort_keys=True))
            return 0

        if not args.server_log:
            parser.error("--server-log is required unless --self-check is selected")
        server_log = ServerLogTail(root / args.server_log)
        result = {
            "workload": "m31-accel-0001-agent64k-v1",
            "qualification_sha": os.environ.get("M31_SOURCE_SHA") or subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                text=True, check=True).stdout.strip(),
            "initial_context_target_tokens": TARGET_PROMPT_TOKENS,
            "initial_context_tokenizer_verified": exact_count,
            "expected_turn_count": len(TURN_PLANS), "turns_completed": 0,
            "workload_complete": False, "status": "in_progress",
            "active_turn": None, "active_phase": None, "requests_completed": 0,
            "in_progress_turn": None,
            "prefix_reuse_correctness": "UNQUALIFIED: no matched cold replay",
            "source_manifest": manifest, "turns": [],
        }
        output_path = root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_result(output_path, result)
        previous_prompt_tokens = None
        for index, (tool_path, start_line, end_line, task) in enumerate(TURN_PLANS, 1):
            turn = {"turn": index, "requests": [], "tool_results": []}
            result["active_turn"] = index
            result["in_progress_turn"] = turn
            if index > 1:
                messages.append({"role": "user", "content":
                    f"Turn {index}: continue the same MIInfer coding-agent investigation. Make exactly one "
                    f"read_file call for {tool_path} lines {start_line}-{end_line}; do not answer in prose yet."})
            result["active_phase"] = "planned_read_tool_call"
            write_result(output_path, result)
            call_response = stream_completion(args.url, args.api_key, args.model,
                                              messages, "required", 192, server_log)
            turn["requests"].append(call_response)
            result["requests_completed"] += 1
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
                write_result(output_path, result)
                completion = stream_completion(args.url, args.api_key, args.model,
                                               messages, "auto", 256, server_log)
                turn["requests"].append(completion)
                result["requests_completed"] += 1
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
            vram_samples = [snapshot.get("vram_used_bytes")
                            for item in turn["requests"]
                            for snapshot in (item["gpu_before"], item["gpu_after"])
                            if isinstance(snapshot.get("vram_used_bytes"), int)]
            decode_times = [(row["generated_tokens"] / row["decode_tokens_per_second"])
                            for row in detail if row.get("decode_tokens_per_second", 0) > 0]
            generated = sum(row["generated_tokens"] for row in detail)
            reuse_checks = prefix_reuse_request_gates(
                turn["requests"], previous_prompt_tokens=previous_prompt_tokens)
            previous_prompt_tokens = detail[-1]["prompt_tokens"]
            turn.update({
                "context_tokens": detail[0]["prompt_tokens"],
                "generated_tokens_total": generated,
                "ttft_ms": turn["requests"][0]["client_ttft_ms"],
                "decode_tok_per_sec": generated / sum(decode_times) if decode_times else 0.0,
                "wall_ms": sum(item["client_wall_ms"] for item in turn["requests"]),
                "server_total_ms": sum(row["total_ms"] for row in detail),
                "reused_prefix_tokens": sum(row["reused_prefix_tokens"] for row in latency),
                "suffix_tokens_dispatched": sum(row["suffix_tokens_dispatched"] for row in latency),
                "replayed_tokens": sum(row["suffix_tokens_dispatched"] for row in latency),
                "peak_sampled_vram_bytes": max(vram_samples) if vram_samples else None,
                "external_vram_samples_complete": len(vram_samples) == 2 * len(turn["requests"]),
                "prefix_reuse_request_gates": reuse_checks,
                "prefix_reuse_gate": index == 1 or all(check["passed"] for check in reuse_checks),
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
    finally:
        counter.close()


if __name__ == "__main__":
    raise SystemExit(main())
