#!/usr/bin/env python3
"""Compare the pinned Hugging Face tokenizer vocabulary with GGUF token metadata."""

import hashlib
import json
import struct
import sys
from pathlib import Path


TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"  # gitleaks:allow public tokenizer checksum
VOCAB_SIZE = 248320
SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF metadata")
    return data


def read_u32(stream) -> int:
    return struct.unpack("<I", read_exact(stream, 4))[0]


def read_u64(stream) -> int:
    return struct.unpack("<Q", read_exact(stream, 8))[0]


def read_string(stream) -> str:
    return read_exact(stream, read_u64(stream)).decode("utf-8")


def skip_value(stream, kind: int) -> None:
    if kind in SCALAR_SIZE:
        stream.seek(SCALAR_SIZE[kind], 1)
    elif kind == 8:
        stream.seek(read_u64(stream), 1)
    elif kind == 9:
        element_kind = read_u32(stream)
        count = read_u64(stream)
        if element_kind in SCALAR_SIZE:
            stream.seek(SCALAR_SIZE[element_kind] * count, 1)
        elif element_kind == 8:
            for _ in range(count):
                stream.seek(read_u64(stream), 1)
        else:
            raise ValueError(f"unsupported GGUF metadata array type: {element_kind}")
    else:
        raise ValueError(f"unsupported GGUF metadata type: {kind}")


def read_metadata(path: Path) -> dict:
    result = {}
    with path.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise ValueError("model is not GGUF")
        version = read_u32(stream)
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version: {version}")
        read_u64(stream)  # tensor count
        count = read_u64(stream)
        for _ in range(count):
            key = read_string(stream)
            kind = read_u32(stream)
            if key in {"general.architecture", "tokenizer.ggml.model", "tokenizer.ggml.tokens"}:
                if key == "tokenizer.ggml.tokens":
                    if kind != 9 or read_u32(stream) != 8:
                        raise ValueError("GGUF tokenizer tokens are not a string array")
                    result[key] = [read_string(stream) for _ in range(read_u64(stream))]
                elif kind == 8:
                    result[key] = read_string(stream)
                else:
                    raise ValueError(f"unexpected GGUF type for {key}: {kind}")
            else:
                skip_value(stream, kind)
    return result


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_tokenizer_gguf.py TOKENIZER.json MODEL.gguf")
    tokenizer_path, model_path = map(Path, sys.argv[1:])
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if tokenizer_hash != TOKENIZER_SHA256:
        raise ValueError(f"tokenizer SHA-256 mismatch: {tokenizer_hash}")
    tokenizer = json.loads(tokenizer_path.read_text())
    tokens = [None] * VOCAB_SIZE
    for text, token_id in tokenizer["model"]["vocab"].items():
        if tokens[token_id] is not None:
            raise ValueError(f"duplicate tokenizer ID {token_id}")
        tokens[token_id] = text
    for entry in tokenizer.get("added_tokens", []):
        token_id, text = entry["id"], entry["content"]
        if tokens[token_id] not in (None, text):
            raise ValueError(f"conflicting tokenizer entry for ID {token_id}")
        tokens[token_id] = text
    metadata = read_metadata(model_path)
    gguf_tokens = metadata["tokenizer.ggml.tokens"]
    if metadata.get("general.architecture") != "qwen35" or len(gguf_tokens) != VOCAB_SIZE:
        raise ValueError("GGUF architecture or vocabulary size does not match the pinned corpus")
    differences = [(i, tokens[i], gguf_tokens[i]) for i in range(VOCAB_SIZE)
                   if tokens[i] is not None and tokens[i] != gguf_tokens[i]]
    unmapped = [i for i, token in enumerate(tokens) if token is None]
    if unmapped and unmapped != list(range(unmapped[0], VOCAB_SIZE)):
        raise ValueError("GGUF IDs absent from tokenizer JSON are not a reserved trailing range")
    for text, token_id in (("<|im_start|>", 248045), ("<|im_end|>", 248046), ("<think>", 248068)):
        if tokens[token_id] != text or gguf_tokens[token_id] != text:
            raise ValueError(f"special-token mapping mismatch for {text} at {token_id}")
    if differences:
        raise ValueError(f"tokenizer/ GGUF vocabulary differs at {len(differences)} IDs; first={differences[:5]}")
    print(json.dumps({
        "tokenizer_sha256": tokenizer_hash,
        "gguf_architecture": metadata["general.architecture"],
        "gguf_vocab_size": len(gguf_tokens),
        "tokenizer_defined_ids": sum(token is not None for token in tokens),
        "gguf_unmapped_reserved_ids": sum(token is None for token in tokens),
        "gguf_unmapped_reserved_id_range": [unmapped[0], unmapped[-1]] if unmapped else [],
        "gguf_reserved_token_examples": [gguf_tokens[unmapped[0]], gguf_tokens[unmapped[-1]]] if unmapped else [],
        "special_tokens": {"<|im_start|>": 248045, "<|im_end|>": 248046, "<think>": 248068},
        "vocabulary_ids_equal": True,
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, UnicodeError, ValueError, KeyError, IndexError, struct.error) as error:
        raise SystemExit(f"tokenizer/GGUF verification failed: {error}")
