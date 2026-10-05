#!/usr/bin/env python3
"""Replay the corrected M30 request transcript against a serving endpoint."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MIINFER_SERVER_LOG", "/tmp/m30-fixed-replay-server.err")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v2_0045_agent_workload import (  # noqa: E402
    M30_SOURCES,
    append_tool_exchange,
    attach_server_latency,
    build_context,
    m30_plan,
    stream_completion,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    context, source_manifest = build_context(root, M30_SOURCES)
    messages = [
        {"role": "system", "content": "You are a careful coding-agent assistant working in the MIInfer repository. Use repository tools when requested, ground claims in files, and preserve prior conversation context."},
        {"role": "user", "content": context + "\n\nTask: review the current MIInfer serving path. First call read_file for docs/interactive-serving.md lines 1-80. Do not write or modify files."},
    ]
    result = {
        "workload": "m30-fixed-transcript-replay",
        "workload_version": "m30-agent-v1-fixed-baseline",
        "baseline_qualification_sha": baseline.get("qualification_sha"),
        "logical_turn_target": len(baseline["turns"]),
        "source_manifest": source_manifest,
        "context_excerpt_bytes": len(context.encode()),
        "turns": [],
    }

    completed_requests = 0
    stop_after_turn = False
    for index, baseline_turn in enumerate(baseline["turns"], 1):
        tool_path, start_line, end_line, followup = m30_plan(index)
        if index > 1:
            messages.append({"role": "user", "content":
                f"Turn {index}: continue the same coding-agent investigation. Make exactly one read_file tool call now with path={tool_path}, start_line={start_line}, end_line={end_line}. Do not answer in prose yet; call the tool only. Keep the prior history."})
        turn = {"turn": index, "requests": []}
        for request_index, recorded in enumerate(baseline_turn["requests"]):
            if args.max_requests is not None and completed_requests >= args.max_requests:
                stop_after_turn = True
                break
            turn["requests"].append(stream_completion(
                args.url, "miinfer", "qwen3.8-27b", messages, 
                "required" if request_index == 0 else "auto", 192 if request_index == 0 else 1024))
            if request_index == 0:
                append_tool_exchange(root, messages, recorded)
                messages.append({"role": "user", "content": followup})
            elif recorded["tool_calls"]:
                append_tool_exchange(root, messages, recorded)
                messages.append({"role": "user", "content":
                    "Use the tool result and now complete the requested detailed report. Provide at least 220 words."})
            else:
                messages.append({"role": "assistant", "content": recorded["content"]})
            completed_requests += 1
            if args.max_requests is not None and completed_requests >= args.max_requests:
                stop_after_turn = True
                break
        turn["generated_tokens_total"] = sum(item["generated_tokens"] for item in turn["requests"])
        turn["minimum_generation_gate"] = turn["generated_tokens_total"] >= 128
        result["turns"].append(turn)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if stop_after_turn:
            break

    result["all_turns_minimum_generation_gate"] = all(
        turn["minimum_generation_gate"] for turn in result["turns"])
    result = attach_server_latency(result)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "turns": len(result["turns"]),
        "requests": sum(len(turn["requests"]) for turn in result["turns"]),
        "complete_transcript": args.max_requests is None or completed_requests >= sum(
            len(turn["requests"]) for turn in baseline["turns"]),
        "all_turns_minimum_generation_gate": result["all_turns_minimum_generation_gate"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
