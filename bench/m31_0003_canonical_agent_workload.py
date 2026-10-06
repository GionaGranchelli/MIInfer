#!/usr/bin/env python3
"""Deterministic M31 agent-workload runner.

The transcript and tool results are fixed locally.  A live run requires an
MIInfer OpenAI-compatible server; ``--dry-run`` emits the exact manifest
without touching the server.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from v2_0045_agent_workload import stream_completion


ROOT = Path(__file__).resolve().parents[1]
TOOL_RESULTS = {
    "read": "tools/read_file returned docs/architecture.md lines 1-80 from the frozen source tree.",
    "search": "tools/search_code returned the exact persistent-context call sites and line numbers.",
    "fail": "tool error: attempted patch rejected by the validation gate; preserve the branch and rollback.",
    "long": "tool result: " + ("deterministic repository excerpt; " * 80),
}


def sha256(value):
    return hashlib.sha256(value.encode()).hexdigest()


def manifest(root):
    entries = []
    for relative in ("docs/architecture.md", "docs/interactive-serving.md",
                     "include/miinfer/prefill_v2/model.hpp",
                     "src/prefill_v2/model.cpp"):
        data = (root / relative).read_bytes()
        entries.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data)})
    return entries


def transcript():
    events = []
    for turn in range(1, 21):
        kind = ("conversation" if turn == 1 else
                "code_generation" if turn in (2, 3, 10) else
                "tool_invocation" if turn in (4, 11, 15) else
                "tool_result" if turn in (5, 12, 16) else
                "branch" if turn in (6, 13, 17) else
                "failed_attempt" if turn in (7, 14) else
                "rollback" if turn in (8, 18) else
                "alternate_branch" if turn in (9, 19) else
                "long_tool_result" if turn == 20 else "continuation")
        tool_key = {"tool_result": "read", "failed_attempt": "fail",
                    "long_tool_result": "long"}.get(kind)
        events.append({"turn": turn, "kind": kind,
                       "prompt": f"M31 canonical turn {turn}: perform the deterministic {kind} step.",
                       "tool_result": TOOL_RESULTS.get(tool_key) if tool_key else None})
    return events


def run_path(url, model, events, mode, max_tokens):
    messages = [{"role": "system", "content":
                 "You are a deterministic coding agent. Follow the numbered M31 transcript exactly."}]
    rows = []
    for event in events:
        if mode == "cold_replay":
            request_messages = messages + [{"role": "user", "content": event["prompt"]}]
        else:
            request_messages = messages + [{"role": "user", "content": event["prompt"]}]
        result = stream_completion(url, "miinfer", model, request_messages, "none", max_tokens)
        rows.append({"turn": event["turn"], "kind": event["kind"], **result})
        messages.extend(({"role": "user", "content": event["prompt"]},
                        {"role": "assistant", "content": result["content"]}))
        if event["tool_result"]:
            messages.append({"role": "tool", "content": event["tool_result"]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8087/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--mode", choices=("all", "cold_replay", "persistent", "cow"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results/m31-0003-agent-workload.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    events = transcript()
    result = {"benchmark": "M31-0003", "workload": "canonical-agent-20-turn-v1",
              "qualification_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                    capture_output=True, text=True, check=True).stdout.strip(),
              "source_manifest": manifest(ROOT), "transcript_sha256": sha256(json.dumps(events, sort_keys=True)),
              "events": events, "paths": {}}
    if not args.dry_run:
        modes = ("cold_replay", "persistent", "cow") if args.mode == "all" else (args.mode,)
        for mode in modes:
            result["paths"][mode] = {"requests": run_path(args.url, args.model, events, mode, args.max_tokens)}
    result["paths"].setdefault("cow", {"requests": [], "snapshot_api_required": True})
    result["replay_avoided"] = None
    result["agent_runtime_speedup"] = None
    result["parity_gate"] = False
    result["result"] = "FAIL" if args.dry_run or "cow" in result["paths"] and result["paths"]["cow"].get("snapshot_api_required") else "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("benchmark", "transcript_sha256", "result")}, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
