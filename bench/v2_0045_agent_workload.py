#!/usr/bin/env python3
"""Run a fixed three-turn coding-agent flow against the MIInfer HTTP API."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path


SOURCES = (
    ("README.md", 15000),
    ("docs/interactive-serving.md", 5962),
    ("include/miinfer/prefill_v2/model.hpp", 6000),
    ("src/openai_api.cpp", 3500),
)
M30_SOURCES = (
    ("README.md", 12000),
    ("docs/architecture.md", 10000),
    ("docs/interactive-serving.md", 9000),
    ("include/miinfer/prefill_v2/model.hpp", 10000),
    ("src/prefill_v2/model.cpp", 14000),
    ("src/openai_api.cpp", 7000),
    ("tools/miinfer_cli.cpp", 12000),
)
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a repository file, optionally restricted to a line range.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Find exact text in tracked source and documentation files.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "path": {"type": "string"},
        }, "required": ["query"]},
    }},
]
SERVER_LOG = Path(os.environ.get(
    "MIINFER_SERVER_LOG",
    Path(__file__).resolve().parents[1] / "results/v2-0045/agent-server.log"))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build_context(root, sources=SOURCES):
    parts, manifest = [], []
    for relative, limit in SOURCES:
        data = (root / relative).read_bytes()
        excerpt = data[:limit]
        manifest.append({"path": relative, "file_sha256": sha256(data),
                         "excerpt_sha256": sha256(excerpt), "excerpt_bytes": len(excerpt)})
        parts.append(f"\n--- repository file: {relative} ---\n" + excerpt.decode("utf-8", "replace"))
    return "\nRepository excerpts for this coding-agent task:\n" + "\n".join(parts), manifest


def gpu_snapshot():
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showtemp", "--showclocks"],
            capture_output=True, text=True, timeout=5, check=False)
        output = result.stdout
        patterns = {
            "vram_used_bytes": r"VRAM Total Used Memory \(B\):\s*(\d+)",
            "vram_total_bytes": r"VRAM Total Memory \(B\):\s*(\d+)",
            "junction_c": r"Temperature \(Sensor junction\) \(C\):\s*([\d.]+)",
            "sclk_mhz": r"sclk clock level:.*\((\d+)Mhz\)",
            "mclk_mhz": r"mclk clock level:.*\((\d+)Mhz\)",
        }
        values = {key: (float(found.group(1)) if key == "junction_c" else int(found.group(1)))
                  for key, pattern in patterns.items() if (found := re.search(pattern, output))}
        return values or {"error": result.stderr.strip() or "telemetry unavailable"}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def http_text(url, key):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8")


def metrics(url, key):
    base = url.rsplit("/v1/", 1)[0]
    body = http_text(base + "/metrics", key)
    values = {}
    for name in ("miinfer_prompt_tokens_total", "miinfer_generated_tokens_total",
                 "miinfer_inference_requests_total"):
        match = re.search(rf"^{name}\s+(\d+)$", body, re.M)
        values[name] = int(match.group(1)) if match else 0
    return values


def latest_server_latency():
    if not SERVER_LOG.exists():
        return None
    for line in reversed(SERVER_LOG.read_text(errors="replace").splitlines()):
        if line.startswith("miinfer_request_latency "):
            return json.loads(line.split(" ", 1)[1])
    return None


def attach_server_latency(result):
    events = []
    if SERVER_LOG.exists():
        events = [json.loads(line.split(" ", 1)[1]) for line in SERVER_LOG.read_text(
            errors="replace").splitlines() if line.startswith("miinfer_request_latency ")]
    request_count = sum(len(turn["requests"]) for turn in result["turns"])
    offset = max(0, len(events) - request_count)
    for turn in result["turns"]:
        used = events[offset:offset + len(turn["requests"])]
        offset += len(turn["requests"])
        for request, event in zip(turn["requests"], used):
            request["server_latency"] = event
        turn["reused_prefix_tokens"] = sum(event["reused_prefix_tokens"] for event in used)
        turn["new_prefill_tokens"] = sum(event.get(
            "suffix_tokens_dispatched",
            max(0, event["prompt_tokens"] - event["reused_prefix_tokens"])) for event in used)
        turn["gdn_restore_ms"] = sum(event.get("restore_ms", 0.0) for event in used)
        turn["suffix_prefill_ms"] = sum(event.get("suffix_prefill_ms", 0.0) for event in used)
        turn["prefill_ms"] = sum(event["prefill_ms"] for event in used)
        turn["ttft_ms"] = turn["requests"][0]["client_ttft_ms"]
        turn["minimum_visible_answer_gate"] = len(turn["requests"][-1]["content"]) >= 300
        turn["minimum_generation_gate"] = turn["generated_tokens_total"] >= 128
    result["all_turns_minimum_generation_gate"] = all(
        turn["minimum_generation_gate"] for turn in result["turns"])
    result["all_turns_visible_answer_gate"] = all(
        turn["minimum_visible_answer_gate"] for turn in result["turns"])
    return result


def stream_completion(url, key, model, messages, tool_choice, max_tokens):
    body = json.dumps({
        "model": model, "messages": messages, "tools": TOOLS,
        "tool_choice": tool_choice, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "top_k": 1, "stream": True,
    }).encode()
    before = metrics(url, key)
    gpu_before = gpu_snapshot()
    started = time.perf_counter()
    request = urllib.request.Request(url, data=body, headers={
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
    after = metrics(url, key)
    return {
        "content": "".join(content), "reasoning_content": "".join(reasoning),
        "tool_calls": [calls[index] for index in sorted(calls)],
        "client_ttft_ms": first_delta_ms, "client_wall_ms": wall_ms,
        "prompt_tokens": after["miinfer_prompt_tokens_total"] - before["miinfer_prompt_tokens_total"],
        "generated_tokens": after["miinfer_generated_tokens_total"] - before["miinfer_generated_tokens_total"],
        "request_count": after["miinfer_inference_requests_total"] - before["miinfer_inference_requests_total"],
        "server_latency": latest_server_latency(),
        "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
    }


def execute_tool(root, call):
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return f"Tool error: malformed arguments for {name}; continue without this result."
    if name == "read_file":
        target = (root / args["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("tool attempted to read outside the repository")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        begin = max(1, int(args.get("start_line", 1)))
        end = min(len(lines), int(args.get("end_line", min(len(lines), begin + 160))))
        return f"{args['path']} lines {begin}-{end} of {len(lines)}:\n" + "\n".join(
            f"{index}: {lines[index - 1]}" for index in range(begin, end + 1))
    if name == "search_code":
        query = args["query"]
        base = (root / args.get("path", ".")).resolve()
        if not base.is_relative_to(root):
            raise ValueError("tool attempted to search outside the repository")
        hits = []
        for path in sorted(base.rglob("*")):
            if {".git", "build", "graphify-out", "results"} & set(path.parts):
                continue
            if not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            try:
                for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if query in line:
                        hits.append(f"{path.relative_to(root)}:{line_number}:{line[:300]}")
                        if len(hits) == 80:
                            return "\n".join(hits)
            except OSError:
                continue
        return "\n".join(hits) or "No matches."
    return f"Tool error: unsupported tool {name}; continue without this result."


def append_tool_exchange(root, messages, response):
    calls = response["tool_calls"]
    if not calls:
        raise RuntimeError("required tool call was not returned")
    messages.append({"role": "assistant", "content": response["content"] or None,
                     "tool_calls": calls})
    results = []
    for call in calls:
        output = execute_tool(root, call)
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        results.append({"tool": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                        "result_sha256": sha256(output.encode()),
                        "result_chars": len(output)})
    return results


def m30_plan(index):
    phase = (index - 1) // 10
    paths = (
        ("README.md", 1, 70),
        ("docs/architecture.md", 1, 80),
        ("docs/interactive-serving.md", 1, 80),
        ("include/miinfer/prefill_v2/model.hpp", 1, 100),
        ("src/prefill_v2/model.cpp", 740, 820),
        ("src/openai_api.cpp", 300, 340),
        ("tools/miinfer_cli.cpp", 5150, 5240),
    )
    path, start, end = paths[(index - 1) % len(paths)]
    phases = (
        "orient the repository and identify the production request path",
        "inspect model state ownership and prepare an implementation plan",
        "interpret tests and diagnose a bounded hypothetical failure",
        "review regression evidence and prepare release-readiness notes",
    )
    return path, start, end, (
        f"Turn {index} belongs to phase {phase + 1}: {phases[phase]}. "
        f"Read the requested excerpt, distinguish measured facts from assumptions, "
        "and report a concise evidence-based continuation for the coding-agent session."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8087/v1/chat/completions")
    parser.add_argument("--api-key", default="miinfer")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--output", default="results/v2-0045/agent-workload.json")
    parser.add_argument("--enrich-existing", action="store_true")
    parser.add_argument("--m30", action="store_true", help="run the frozen 40-turn M30 workload")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_path = root / args.output
    if args.enrich_existing:
        result = attach_server_latency(json.loads(output_path.read_text(encoding="utf-8")))
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"refreshed telemetry for {sum(len(turn['requests']) for turn in result['turns'])} requests")
        return 0
    context, source_manifest = build_context(root, M30_SOURCES if args.m30 else SOURCES)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    messages = [
        {"role": "system", "content": "You are a careful coding-agent assistant working in the MIInfer repository. Use repository tools when requested, ground claims in files, and preserve prior conversation context."},
        {"role": "user", "content": context + "\n\nTask: review the current MIInfer serving path. First call read_file for docs/interactive-serving.md lines 1-80. Do not write or modify files."},
    ]
    result = {"qualification_sha": revision,
              "workload": "m30-agent-v1" if args.m30 else "three-turn coding-agent tool/reuse flow",
              "workload_version": "m30-agent-v1" if args.m30 else "v2-0045-three-turn",
              "logical_turn_target": 40 if args.m30 else 3,
              "source_manifest": source_manifest, "context_excerpt_bytes": len(context.encode()),
              "turns": []}
    plans = [
        ("docs/interactive-serving.md", 1, 80,
         "Now explain the serving and prefix reuse contract in at least eight numbered observations. Cite file names and distinguish documented behavior from assumptions. Target at least 220 words."),
        ("src/prefill_v2/model.cpp", 737, 845,
         "Analyze how this implementation performs reuse, suffix prefill, checkpoint capture, and next-token setup. Give a precise eight-part walkthrough with file references and at least 220 words."),
        ("tools/miinfer_cli.cpp", 4470, 4510,
         "Relate the request telemetry and prefix reuse behavior to a practical 10-request stability test. Give an evidence-based final checklist and caveats in at least eight numbered items and 220 words."),
    ]
    if args.m30:
        plans = [m30_plan(index) for index in range(1, 41)]
    for index, (tool_path, start_line, end_line, followup) in enumerate(plans, 1):
        turn = {"turn": index, "requests": []}
        if index > 1:
            messages.append({"role": "user", "content":
                f"Turn {index}: continue the same coding-agent investigation. Make exactly one read_file tool call now with path={tool_path}, start_line={start_line}, end_line={end_line}. Do not answer in prose yet; call the tool only. Keep the prior history."})
        call_response = stream_completion(args.url, args.api_key, args.model, messages, "required", 192)
        turn["requests"].append(call_response)
        tool_results = append_tool_exchange(root, messages, call_response)
        turn["tool_results"] = tool_results
        messages.append({"role": "user", "content": followup})
        completion = None
        for _ in range(3):
            completion = stream_completion(args.url, args.api_key, args.model, messages, "auto", 1024)
            turn["requests"].append(completion)
            if not completion["tool_calls"]:
                break
            turn["tool_results"].extend(append_tool_exchange(root, messages, completion))
            messages.append({"role": "user", "content":
                "Use the tool result and now complete the requested detailed report. Provide at least 220 words."})
        if completion is None or completion["tool_calls"]:
            raise RuntimeError(f"turn {index} did not produce a final assistant response")
        messages.append({"role": "assistant", "content": completion["content"]})
        turn["context_tokens"] = completion["prompt_tokens"]
        turn["generated_tokens_total"] = sum(item["generated_tokens"] for item in turn["requests"])
        turn["ttft_ms"] = turn["requests"][0]["client_ttft_ms"]
        turn["wall_ms"] = sum(item["client_wall_ms"] for item in turn["requests"])
        turn["gpu_after"] = completion["gpu_after"]
        turn["minimum_generation_gate"] = turn["generated_tokens_total"] >= 128
        turn["minimum_visible_answer_gate"] = len(completion["content"]) >= 300
        result["turns"].append(turn)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["all_turns_minimum_generation_gate"] = all(
        turn["minimum_generation_gate"] for turn in result["turns"])
    result["all_turns_visible_answer_gate"] = all(
        turn["minimum_visible_answer_gate"] for turn in result["turns"])
    result["conversation_message_count"] = len(messages)
    result = attach_server_latency(result)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if (result["all_turns_minimum_generation_gate"]
                 and result["all_turns_visible_answer_gate"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
