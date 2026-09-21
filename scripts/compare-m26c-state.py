#!/usr/bin/env python3
"""Compare M26-C route snapshots; semantic float buffers require bitwise parity."""

import argparse
import math
import struct
import sys


def read_exact(file, size):
    data = file.read(size)
    if len(data) != size:
        raise ValueError("truncated M26-C snapshot")
    return data


def scalar(file, fmt):
    size = struct.calcsize(fmt)
    return struct.unpack("<" + fmt, read_exact(file, size))[0]


def string(file):
    size = scalar(file, "Q")
    if size > 4096:
        raise ValueError("invalid identity length")
    return read_exact(file, size).decode()


def compare_bytes(left, right, label, size, floats=False, summaries=None, group=None,
                  layer_errors=None, words=False):
    remaining = size
    offset = 0
    maximum = 0.0
    sum_sq = left_sq = right_sq = 0.0
    finite = True
    count = 0
    mismatch_at = None
    mismatch_words = None
    while remaining:
        chunk_size = min(1 << 20, remaining)
        chunk_size -= chunk_size % 4 if floats else 0
        a = read_exact(left, chunk_size)
        b = read_exact(right, chunk_size)
        if a != b:
            if mismatch_at is None:
                local = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
                mismatch_at = offset + local
                if words:
                    word_offset = local - local % 4
                    mismatch_words = (mismatch_at // 4,
                                      struct.unpack_from("<I", a, word_offset)[0],
                                      struct.unpack_from("<I", b, word_offset)[0])
        if floats:
            for (x,), (y,) in zip(struct.iter_unpack("<f", a), struct.iter_unpack("<f", b)):
                count += 1
                if not math.isfinite(x) or not math.isfinite(y):
                    finite = False
                    maximum = math.inf
                else:
                    delta = x - y
                    maximum = max(maximum, abs(delta))
                    sum_sq += delta * delta
                    left_sq += x * x
                    right_sq += y * y
        offset += chunk_size
        remaining -= chunk_size
    if floats:
        if summaries is not None and group is not None:
            prior = summaries.get(group, (0.0, True, 0.0, 0.0, 0.0, 0, True))
            summaries[group] = (max(prior[0], maximum), prior[1] and mismatch_at is None,
                                prior[2] + sum_sq, prior[3] + left_sq, prior[4] + right_sq,
                                prior[5] + count, prior[6] and finite)
        if layer_errors is not None:
            layer_errors.append((label, maximum, math.sqrt(sum_sq / count) if count else 0.0,
                                 math.sqrt(sum_sq / left_sq) if left_sq else math.nan,
                                 finite, mismatch_at is None))
    elif mismatch_at is not None:
        if mismatch_words is None:
            print(f"{label}: exact=False first_byte={mismatch_at}")
        else:
            index, a, b = mismatch_words
            print(f"{label}: exact=False first_index={index} left={a} right={b}")
    else:
        print(f"{label}: exact=True")
    return mismatch_at is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy")
    parser.add_argument("interactive")
    args = parser.parse_args()
    try:
        with open(args.legacy, "rb") as left, open(args.interactive, "rb") as right:
            if read_exact(left, 8) != read_exact(right, 8):
                raise ValueError("snapshot magic differs")
            fixed_fmt = "IIIIQQIIQQ"
            lfixed = struct.unpack("<" + fixed_fmt, read_exact(left, struct.calcsize("<" + fixed_fmt)))
            rfixed = struct.unpack("<" + fixed_fmt, read_exact(right, struct.calcsize("<" + fixed_fmt)))
            if any(lfixed[i] != rfixed[i] for i in range(len(lfixed)) if i != 6):
                raise ValueError(f"snapshot structure differs: left={lfixed}, right={rfixed}")
            equal = lfixed[6] == rfixed[6]
            if not equal:
                print(f"current token differs: left={lfixed[6]} right={rfixed[6]}")
            model_hashes = (string(left), string(right))
            quantizers = (string(left), string(right))
            contracts = (string(left), string(right))
            if model_hashes[0] != model_hashes[1] or quantizers[0] != quantizers[1]:
                raise ValueError("model hash or quantization identity differs")
            position, current_token = lfixed[5], lfixed[6]
            layer_count, token_count, output_count = lfixed[7], lfixed[8], lfixed[9]
            equal &= compare_bytes(left, right, "processed token history", token_count * 4,
                                   words=True)
            equal &= compare_bytes(left, right, "generated token IDs", output_count * 4,
                                   words=True)
            summaries = {}
            layer_errors = []
            for layer in range(layer_count):
                kind_left = scalar(left, "B")
                kind_right = scalar(right, "B")
                if kind_left != kind_right:
                    raise ValueError(f"layer {layer}: kind differs")
                if kind_left == 1:
                    lcounts = (scalar(left, "Q"), scalar(left, "Q"))
                    rcounts = (scalar(right, "Q"), scalar(right, "Q"))
                    if lcounts != rcounts:
                        raise ValueError(f"layer {layer}: recurrent shapes differ")
                    equal &= compare_bytes(left, right, f"layer {layer} recurrent state", lcounts[0] * 4,
                                           True, summaries, "recurrent_state", layer_errors)
                    equal &= compare_bytes(left, right, f"layer {layer} convolution history", lcounts[1] * 4,
                                           True, summaries, "convolution_history", layer_errors)
                elif kind_left == 2:
                    lshape = (scalar(left, "I"), scalar(left, "Q"))
                    rshape = (scalar(right, "I"), scalar(right, "Q"))
                    if lshape != rshape:
                        raise ValueError(f"layer {layer}: attention shapes differ")
                    for head in range(lshape[0]):
                        equal &= compare_bytes(left, right, f"layer {layer} head {head} K", lshape[1] * 4,
                                               True, summaries, "active_KV_K", layer_errors)
                        equal &= compare_bytes(left, right, f"layer {layer} head {head} V", lshape[1] * 4,
                                               True, summaries, "active_KV_V", layer_errors)
                else:
                    raise ValueError(f"layer {layer}: invalid kind {kind_left}")
            if left.read(1) or right.read(1):
                raise ValueError("trailing snapshot bytes")
            result = "PASS" if equal else "FAIL"
            for family, (max_abs, exact, sum_sq, left_sq, right_sq, count, finite) in summaries.items():
                rel_l2 = math.sqrt(sum_sq / left_sq) if left_sq else math.nan
                print(f"{family}: max_abs={max_abs:.9g} rms={math.sqrt(sum_sq / count):.9g} "
                      f"relative_l2={rel_l2:.9g} finite={finite} exact={exact}")
            for label, max_abs, rms, relative_l2, finite, exact in layer_errors:
                print(f"{label}: max_abs={max_abs:.9g} rms={rms:.9g} "
                      f"relative_l2={relative_l2:.9g} finite={finite} exact={exact}")
            print(f"m26c_state_compare={result} position={position} current_token={current_token} "
                  f"output_tokens={output_count} model_sha256={model_hashes[0]}")
            print(f"left_contract={contracts[0]}")
            print(f"right_contract={contracts[1]}")
            if not equal:
                return 1
    except (OSError, ValueError, UnicodeDecodeError) as error:
        print(f"m26c_state_compare=FAIL {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
