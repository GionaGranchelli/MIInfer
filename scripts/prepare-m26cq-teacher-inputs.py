#!/usr/bin/env python3
"""Build a fixed teacher-forcing sequence from the qualified legacy trajectory."""

import argparse
import struct


def read_header(path):
    with open(path, "rb") as stream:
        magic = stream.read(8)
        assert magic == b"M26CSTAT"
        version, blocks, hidden, state, context, position, current, layers, history, outputs = (
            struct.unpack("<IIIIQQIIQQ", stream.read(56))
        )
        assert version == 1 and history == position
        for _ in range(3):
            length = struct.unpack("<Q", stream.read(8))[0]
            stream.seek(length, 1)
        ids = struct.unpack(f"<{history}I", stream.read(history * 4))
        output_ids = struct.unpack(f"<{outputs}I", stream.read(outputs * 4))
    return position, current, ids, output_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_p512_state")
    parser.add_argument("legacy_trajectory_state")
    parser.add_argument("count", type=int)
    parser.add_argument("output")
    args = parser.parse_args()

    start_pos, pending, prefix, _ = read_header(args.canonical_p512_state)
    trajectory_pos, _, trajectory_prefix, generated = read_header(args.legacy_trajectory_state)
    assert start_pos == 512 and trajectory_pos >= start_pos
    assert trajectory_prefix[:start_pos] == prefix
    assert 1 <= args.count <= len(generated) + 1
    tokens = (pending, *generated[:args.count - 1])
    with open(args.output, "w", encoding="ascii") as output:
        output.writelines(f"{token}\n" for token in tokens)
    print(f"teacher_inputs={len(tokens)} start_position={start_pos} first_token={tokens[0]} "
          f"last_position={start_pos + len(tokens) - 1} source=legacy-current-trajectory")


if __name__ == "__main__":
    main()
