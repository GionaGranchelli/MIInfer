#!/usr/bin/env python3
"""Freeze the six M31-LC-0009 token-ID prompts from a pinned Qwen tokenizer."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

from tokenizers import Tokenizer


MODEL_SHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
TOKENIZER_REVISION = "Qwen/Qwen3.8-27B@366a23fd246985933964e9d21ac63eb3a93190ab"
VOCAB_LIMIT = 248320

PROMPTS = [
    {
        "id": "ordinary-64-01",
        "category": "ordinary_text",
        "tokens": 64,
        "context": "Ice floats because liquid water is denser than solid water. A lake cools from its surface, and ice forms above the deeper liquid. ",
        "question": "Explain in one plain sentence why this surface ice helps fish survive winter.",
        "acceptance": "Answer says the ice layer slows heat loss and leaves liquid water below; no unsupported exact temperature claim.",
        "filler": "Context note: the question concerns heat moving through a lake. ",
    },
    {
        "id": "coding-64-01",
        "category": "coding",
        "tokens": 64,
        "context": "for i in range(len(xs) - 1):\n    check(xs[i])\n",
        "question": "Give the corrected range expression.",
        "acceptance": "The expression is range(len(xs)); it handles empty lists and checks every element.",
        "filler": "Input note: Python range excludes its stop value. ",
    },
    {
        "id": "reasoning-512-01",
        "category": "reasoning",
        "tokens": 512,
        "context": "Three repair crews A, B, and C must visit sites North, East, and West once each. A cannot visit North. B must visit East. C visits the remaining site. State the assignment as A=site, B=site, C=site. ",
        "question": "What is the only assignment satisfying every rule? Give only the three assignments.",
        "acceptance": "Exact assignment A=West, B=East, C=North.",
        "filler": "Dispatch note {i}: crew B remains reserved for East; record constraints as written and do not infer additional restrictions.\n",
    },
    {
        "id": "ordinary-512-01",
        "category": "ordinary_text",
        "tokens": 512,
        "context": "A neighborhood library tested evening hours for four weeks. Attendance rose on Tuesday and Thursday. The Saturday visit count did not change. Most new evening visitors borrowed language-learning books and used the public computers. Staff reported a longer closing checklist but no safety incidents. The library plans to keep Tuesday hours, review Thursday staffing, and leave Saturday unchanged. ",
        "question": "Summarize the decision and the two supporting observations in three concise bullet points.",
        "acceptance": "Retains Tuesday continuation, Thursday staffing review, Saturday unchanged; cites evening attendance and visitor use; does not invent safety issues.",
        "filler": "Observation {i}: the four-week pilot log records attendance, borrowing, computer use, staffing, and safety separately.\n",
    },
    {
        "id": "coding-2048-01",
        "category": "coding",
        "tokens": 2048,
        "context": "A Python service applies partial updates to a settings dictionary. The contract says a missing key leaves its old value unchanged, while an explicit JSON null sets the value to null. The implementation currently uses `updates.get(key)` and treats every `None` as a deletion. Example: old state is `{'theme': 'dark', 'timezone': 'UTC'}`; update `{'theme': None}` must preserve the key with value None, and an update without `timezone` must preserve UTC. ",
        "question": "Give the smallest Python condition that distinguishes an absent key from an explicit None value. Explain in one sentence.",
        "acceptance": "Uses `if key not in updates` or equivalent membership check; explains absent versus explicit null.",
        "filler": "Update case {i}: partial-update tests distinguish an omitted field from a field explicitly present with a null value.\n",
    },
    {
        "id": "reasoning-2048-01",
        "category": "reasoning",
        "tokens": 2048,
        "context": "A warehouse ledger has 120 packages. Each package is classified exactly once as delivered, held, or returned. The audit rules say 84 were delivered and three times as many were held as returned. A package is held only if it has not been delivered or returned. ",
        "question": "How many packages were held and how many were returned? Show the equation and give the two counts.",
        "acceptance": "Solves H+R=36 and H=3R, yielding held=27 and returned=9.",
        "filler": "Ledger row {i}: package status is one of delivered, held, or returned; categories are mutually exclusive and exhaustive.\n",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_for(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def freeze(tokenizer: Tokenizer, item: dict) -> dict:
    head = ids_for(tokenizer, "<|im_start|>user\n" + item["context"])
    tail = ids_for(
        tokenizer,
        "\nQuestion: " + item["question"]
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n",
    )
    room = item["tokens"] - len(head) - len(tail)
    if room < 0:
        raise ValueError(f"{item['id']}: task text exceeds its target length")

    filler_ids = []
    i = 1
    while len(filler_ids) < room:
        filler_ids.extend(ids_for(tokenizer, item["filler"].format(i=i)))
        i += 1
    token_ids = head + filler_ids[:room] + tail
    if len(token_ids) != item["tokens"] or max(token_ids) >= VOCAB_LIMIT:
        raise ValueError(f"{item['id']}: invalid token count or out-of-range token")

    packed = struct.pack(f"<{len(token_ids)}I", *token_ids)
    prompt_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return {
        **item,
        "prompt_text": prompt_text,
        "input_ids": token_ids,
        "prompt_ids_encoding": "little-endian uint32",
        "prompt_ids_sha256": hashlib.sha256(packed).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if sha256(args.tokenizer_json) != TOKENIZER_SHA256:
        raise SystemExit("tokenizer SHA-256 does not match the pinned revision")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    if (tokenizer.token_to_id("<|im_start|>") != 248045
            or tokenizer.token_to_id("<|im_end|>") != 248046
            or tokenizer.token_to_id("<think>") != 248068):
        raise SystemExit("unexpected Qwen chat token IDs")

    records = [freeze(tokenizer, item) for item in PROMPTS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode() for r in records)
    corpus_path = args.output_dir / "corpus.jsonl"
    corpus_path.write_bytes(data)
    manifest = {
        "schema_version": 1,
        "model": "Qwen3.8-27B-Q4_K_M.gguf",
        "model_sha256": MODEL_SHA256,
        "tokenizer": TOKENIZER_REVISION,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_matches_gguf_metadata": "PENDING Z840 VERIFICATION",
        "corpus_sha256": hashlib.sha256(data).hexdigest(),
        "prompt_count": len(records),
        "prompt_id_encoding": "little-endian uint32; raw bytes hashed per prompt",
        "token_counts": {r["id"]: len(r["input_ids"]) for r in records},
    }
    (args.output_dir / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
