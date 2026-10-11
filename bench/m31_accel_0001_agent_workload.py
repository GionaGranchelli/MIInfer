#!/usr/bin/env python3
"""Run eight production-path coding-agent turns from an exact 64K token prompt."""

import argparse
import json
import os
import subprocess
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
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choice = (event.get("choices") or [{}])[0]
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
            print(json.dumps({"self_check": "PASS", "turns": len(TURN_PLANS),
                              "initial_prompt_tokens": exact_count,
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
            "prefix_reuse_correctness": "UNQUALIFIED: no matched cold replay",
            "source_manifest": manifest, "turns": [],
        }
        for index, (tool_path, start_line, end_line, task) in enumerate(TURN_PLANS, 1):
            turn = {"turn": index, "requests": [], "tool_results": []}
            if index > 1:
                messages.append({"role": "user", "content":
                    f"Turn {index}: continue the same MIInfer coding-agent investigation. Make exactly one "
                    f"read_file call for {tool_path} lines {start_line}-{end_line}; do not answer in prose yet."})
            call_response = stream_completion(args.url, args.api_key, args.model,
                                              messages, "required", 192, server_log)
            require_planned_read(call_response, tool_path, start_line, end_line)
            if index == 1 and call_response["prompt_tokens"] != exact_count:
                raise RuntimeError("live server token count differs from the exact MIInfer tokenizer check")
            turn["requests"].append(call_response)
            turn["tool_results"].extend(legacy.append_tool_exchange(root, messages, call_response))
            messages.append({"role": "user", "content":
                f"Turn {index} task: {task} Use source references and distinguish verified behavior "
                "from inference. Give at least five numbered observations and a concrete next step, "
                "with at least 100 words."})
            completion = None
            for _ in range(3):
                completion = stream_completion(args.url, args.api_key, args.model,
                                               messages, "auto", 256, server_log)
                turn["requests"].append(completion)
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
            first_request_reuse = latency[0]["reused_prefix_tokens"]
            first_request_suffix = latency[0]["suffix_tokens_dispatched"]
            turn.update({
                "context_tokens": detail[0]["prompt_tokens"],
                "generated_tokens_total": generated,
                "ttft_ms": turn["requests"][0]["client_ttft_ms"],
                "decode_tok_per_sec": generated / sum(decode_times) if decode_times else 0.0,
                "wall_ms": sum(item["client_wall_ms"] for item in turn["requests"]),
                "server_total_ms": sum(row["total_ms"] for row in detail),
                "reused_prefix_tokens": sum(row["reused_prefix_tokens"] for row in latency),
                "suffix_tokens_dispatched": sum(row["suffix_tokens_dispatched"] for row in latency),
                "peak_sampled_vram_bytes": max(vram_samples) if vram_samples else None,
                "external_vram_samples_complete": len(vram_samples) == 2 * len(turn["requests"]),
                "prefix_reuse_gate": index == 1 or (first_request_reuse > 0 and first_request_suffix > 0),
                "minimum_generation_gate": generated >= 128,
                "minimum_visible_answer_gate": len(completion["content"]) >= 300,
            })
            result["turns"].append(turn)
            output_path = root / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["all_turns_minimum_generation_gate"] = all(
            turn["minimum_generation_gate"] for turn in result["turns"])
        result["all_turns_visible_answer_gate"] = all(
            turn["minimum_visible_answer_gate"] for turn in result["turns"])
        result["all_turns_external_vram_samples_complete"] = all(
            turn["external_vram_samples_complete"] for turn in result["turns"])
        result["all_turns_prefix_reuse_gate"] = all(
            turn["prefix_reuse_gate"] for turn in result["turns"])
        result["conversation_message_count"] = len(messages)
        (root / args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if (result["all_turns_minimum_generation_gate"]
                     and result["all_turns_visible_answer_gate"]
                     and result["all_turns_external_vram_samples_complete"]
                     and result["all_turns_prefix_reuse_gate"]) else 1
    finally:
        counter.close()


if __name__ == "__main__":
    raise SystemExit(main())
