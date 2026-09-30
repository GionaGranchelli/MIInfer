#!/usr/bin/env python3
"""Run the bounded sequential serving/reuse stability gate."""

import json
from pathlib import Path

from v2_0045_agent_workload import append_tool_exchange, build_context, gpu_snapshot, stream_completion


def main():
    root = Path(__file__).resolve().parents[1]
    context, sources = build_context(root)
    output = root / "results/v2-0045/stability-workload.json"
    base = [
        {"role": "system", "content": "You are a careful coding-agent assistant working in the MIInfer repository. Use repository tools when requested and preserve conversation history."},
        {"role": "user", "content": context},
    ]
    result = {"qualification_sha": __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
        check=True).stdout.strip(), "workload": "10 sequential HTTP inference requests; three tool/reuse cycles",
        "source_manifest": sources, "gpu_before": gpu_snapshot(), "requests": []}

    # The request is sized to the P8192 class and deliberately asks for more
    # than the 128-token cap so the decode leg is exactly 128 tokens.
    long_context = context[:26000]
    long_request = [{"role": "system", "content": "Answer as a careful repository analyst."},
                    {"role": "user", "content": long_context +
                     "\n\nGive a detailed analysis of this repository context in at least 220 words."}]
    first = stream_completion("http://127.0.0.1:8087/v1/chat/completions", "miinfer",
                              "qwen3.8-27b", long_request, "none", 128)
    result["requests"].append({"index": 1, "kind": "P8192-class TG128", **first})
    if not 7800 <= first["prompt_tokens"] <= 8500 or first["generated_tokens"] != 128:
        result["p8192_tg128_gate"] = False
    else:
        result["p8192_tg128_gate"] = True

    for cycle, (path, start, end) in enumerate((
            ("docs/interactive-serving.md", 1, 80),
            ("src/prefill_v2/model.cpp", 737, 845),
            ("tools/miinfer_cli.cpp", 4470, 4510)), 1):
        messages = list(base)
        messages.append({"role": "user", "content":
            f"Cycle {cycle}: first call read_file with path={path}, start_line={start}, end_line={end}. Call the tool only."})
        call = stream_completion("http://127.0.0.1:8087/v1/chat/completions", "miinfer",
                                 "qwen3.8-27b", messages, "required", 192)
        result["requests"].append({"index": len(result["requests"]) + 1,
                                   "kind": f"cycle-{cycle}-tool", **call})
        tool_results = append_tool_exchange(root, messages, call)
        result["requests"][-1]["tool_results"] = tool_results
        messages.append({"role": "user", "content":
            "Explain the relevant behavior in at least 180 words, citing the repository file."})
        answer = stream_completion("http://127.0.0.1:8087/v1/chat/completions", "miinfer",
                                   "qwen3.8-27b", messages, "auto", 256)
        result["requests"].append({"index": len(result["requests"]) + 1,
                                   "kind": f"cycle-{cycle}-answer", **answer})
        if answer["tool_calls"]:
            raise RuntimeError(f"cycle {cycle} unexpectedly emitted another tool call")
        messages.append({"role": "assistant", "content": answer["content"]})
        if cycle == 3:
            for followup in range(3):
                messages.append({"role": "user", "content":
                    f"Follow-up {followup + 1}: give one more concise evidence-based implication."})
                extra = stream_completion("http://127.0.0.1:8087/v1/chat/completions", "miinfer",
                                          "qwen3.8-27b", messages, "none", 128)
                result["requests"].append({"index": len(result["requests"]) + 1,
                                           "kind": f"cycle-3-followup-{followup + 1}", **extra})
                messages.append({"role": "assistant", "content": extra["content"]})
    result["gpu_after"] = gpu_snapshot()
    result["request_count_gate"] = len(result["requests"]) == 10
    result["all_requests_successful"] = all(item["request_count"] == 1 for item in result["requests"])
    result["three_reuse_cycles_gate"] = all(
        any(request.get("server_latency", {}).get("cache_hit")
            for request in result["requests"] if request["kind"].startswith(f"cycle-{cycle}-"))
        for cycle in range(1, 4))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"requests", "source_manifest"}}, indent=2))
    return 0 if all((result["p8192_tg128_gate"], result["request_count_gate"],
                     result["all_requests_successful"], result["three_reuse_cycles_gate"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
