#!/usr/bin/env python3
"""Validate the frozen LC-0009 JSONL and write runner-ready little-endian IDs."""

import hashlib
import json
import struct
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: materialize_corpus.py CORPUS.jsonl MANIFEST.json OUTPUT_DIR")
    corpus_path, manifest_path, output_dir = map(Path, sys.argv[1:])
    corpus_bytes = corpus_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(corpus_bytes).hexdigest() != manifest["corpus_sha256"]:
        raise SystemExit("corpus SHA-256 does not match manifest")

    records = [json.loads(line) for line in corpus_bytes.splitlines()]
    if len(records) != manifest["prompt_count"]:
        raise SystemExit("corpus prompt count does not match manifest")
    materialized = []
    for record in records:
        ids = record["input_ids"]
        if len(ids) != record["tokens"] or len(ids) != manifest["token_counts"][record["id"]]:
            raise SystemExit(f"{record['id']}: token count mismatch")
        if any(not isinstance(token, int) or token < 0 or token >= 248320 for token in ids):
            raise SystemExit(f"{record['id']}: invalid token ID")
        packed = struct.pack(f"<{len(ids)}I", *ids)
        if hashlib.sha256(packed).hexdigest() != record["prompt_ids_sha256"]:
            raise SystemExit(f"{record['id']}: prompt ID SHA-256 mismatch")
        path = output_dir / f"{record['id']}.u32le"
        materialized.append((path, packed, len(ids), record["prompt_ids_sha256"]))
    if len({path for path, _, _, _ in materialized}) != len(materialized):
        raise SystemExit("duplicate prompt IDs in corpus")
    if any(path.exists() for path, _, _, _ in materialized):
        raise SystemExit("refusing to overwrite an existing prompt ID file")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, packed, count, digest in materialized:
        path.write_bytes(packed)
        print(f"{path.stem} tokens={count} sha256={digest} file={path}")


if __name__ == "__main__":
    main()
