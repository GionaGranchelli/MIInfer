#!/usr/bin/env python3
"""Validate the small, raw-file M6-A1 reference bundle."""

import argparse
import json
import math
import struct
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    root = args.fixture
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    assert manifest["reference"]["implementation"] == "llama.cpp"
    assert manifest["reference"]["commit"] == "c0bc8591e8815c63cb01dd3f051a8b0df02501c9"
    positions = manifest["workload"]["checkpoint_positions"]
    vocab = manifest["model"]["n_vocab"]
    assert positions == sorted(set(positions))
    prompt_tokens = [int(line) for line in (root / manifest["artifacts"]["prompt_tokens"]).read_text().splitlines()]
    generated_tokens = [int(line) for line in (root / manifest["artifacts"]["generated_tokens"]).read_text().splitlines()]
    assert prompt_tokens == manifest["workload"]["prompt_tokens"]
    assert generated_tokens == manifest["workload"]["generated_tokens"]
    assert (root / manifest["artifacts"]["generated_text"]).stat().st_size > 0

    for artifact in manifest["artifacts"]["logits"]:
        path = root / artifact["file"]
        data = path.read_bytes()
        assert len(data) == artifact["elements"] * 4, path
        values = struct.unpack("<%df" % artifact["elements"], data)
        assert all(math.isfinite(value) for value in values), path
        assert artifact["elements"] == vocab

    for artifact in manifest["artifacts"]["states"]:
        path = root / artifact["file"]
        assert artifact["opaque"] and path.stat().st_size > 0, path

    for record in manifest["artifacts"]["tensors"]:
        path = root / record["file"]
        assert record["type"] == "f32"
        expected = math.prod(record["shape"]) * 4
        assert record["bytes"] == expected and path.stat().st_size == expected, path
        values = struct.unpack("<%df" % math.prod(record["shape"]), path.read_bytes())
        assert all(math.isfinite(value) for value in values), path

    print("M6-A1 fixture valid:", root)
    print("  reference:", manifest["reference"]["commit"])
    print("  positions:", positions)
    print("  tensors:", len(manifest["artifacts"]["tensors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
