#!/usr/bin/env python3
"""Compare captured layer-0 fused and MMQ Gate/Up decode operations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROWS, COLUMNS, Q4_BLOCK, Q8_BLOCK = 17408, 5120, 144, 36


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_f32(path, count):
    raw = Path(path).read_bytes()
    if len(raw) != count * 4:
        raise ValueError(f"{path}: expected {count * 4} bytes, got {len(raw)}")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite values")
    return values


def stats(reference, actual):
    a, b = np.asarray(reference, np.float64), np.asarray(actual, np.float64)
    delta = b - a
    norm = float(np.linalg.norm(a))
    return {
        "bitwise_equal": np.array_equal(np.asarray(reference, "<f4"),
                                        np.asarray(actual, "<f4")),
        "relative_l2": float(np.linalg.norm(delta) / norm) if norm else None,
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def quantize_q8_1(values):
    values = np.asarray(values, dtype=np.float32)
    packed = bytearray()
    for x in values.reshape(-1, 32):
        scale = np.float32(np.max(np.abs(x)) / np.float32(127.0))
        inverse = np.float32(1.0 / scale) if scale else np.float32(0)
        q = np.copysign(np.floor(np.abs(x * inverse) + np.float32(0.5)), x).astype(np.int8)
        q = np.clip(q, -127, 127).astype(np.int8)
        s = np.float32(np.sum(q.astype(np.int32)) * scale)
        packed.extend(np.asarray([scale, s], dtype="<f2").tobytes())
        packed.extend(q.tobytes())
    return bytes(packed)


def quantize_mx_affine(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1, 128)
    packed = bytearray()
    for block in values:
        ds = []
        qs = []
        for group in block.reshape(4, 32):
            maximum = np.max(np.abs(group))
            scale = np.float32(maximum / np.float32(127.0)) if maximum else np.float32(0)
            q = np.zeros(32, dtype=np.int8) if not scale else np.clip(
                np.copysign(np.floor(np.abs(group / scale) + np.float32(0.5)), group),
                -127, 127).astype(np.int8)
            # The HIP kernel uses a pairwise 16, 8, 4, 2, 1 reduction.
            sums = group.copy()
            width = 16
            while width:
                sums[:width] += sums[width:2 * width]
                width //= 2
            ds.extend((scale, sums[0]))
            qs.append(q)
        packed.extend(np.asarray(ds, dtype="<f2").tobytes())
        packed.extend(np.concatenate(qs).tobytes())
    return bytes(packed)


def decode_q8_1(raw):
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(COLUMNS // 32, Q8_BLOCK)
    scales = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    sums = blocks[:, 2:4].copy().view("<f2").reshape(-1).astype(np.float32)
    q = blocks[:, 4:].copy().view(np.int8).reshape(-1, 32)
    return (q.astype(np.float32) * scales[:, None]).reshape(-1), q, scales, sums


def decode_mx_affine(raw):
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(COLUMNS // 128, 144)
    header = blocks[:, :16].copy().view("<f2").reshape(-1, 4, 2).astype(np.float32)
    q = blocks[:, 16:].copy().view(np.int8).reshape(-1, 4, 32)
    return (q.astype(np.float32) * header[:, :, 0, None]).reshape(-1), q, header[:, :, 0], header[:, :, 1]


def q4_values(blocks):
    """Decode canonical GGUF Q4_K blocks to their dequantized F32 weights."""
    nrow, nblock = blocks.shape[:2]
    ql = blocks[..., 16:]
    index = np.arange(256)
    group, lane = index // 32, index % 32
    pair, part = group // 2, group % 2
    half, in_half = lane // 16, lane % 16
    byte = pair * 32 + (in_half // 4) * 4 + (half != 0) * 16 + in_half % 4
    q = (ql[..., byte] >> (4 * part)) & 15
    scale_bytes = blocks[..., 4:16]
    scales = np.empty((nrow, nblock, 8), dtype=np.float32)
    minimums = np.empty_like(scales)
    for g in range(8):
        if g < 4:
            scales[..., g] = scale_bytes[..., g] & 63
            minimums[..., g] = scale_bytes[..., g + 4] & 63
        else:
            scales[..., g] = (scale_bytes[..., g + 4] & 15) | ((scale_bytes[..., g - 4] >> 6) << 4)
            minimums[..., g] = (scale_bytes[..., g + 4] >> 4) | ((scale_bytes[..., g] >> 6) << 4)
    d = blocks[..., :2].copy().view("<f2").reshape(nrow, nblock).astype(np.float32)
    dmin = blocks[..., 2:4].copy().view("<f2").reshape(nrow, nblock).astype(np.float32)
    weights = np.empty((nrow, nblock, 256), dtype=np.float32)
    for g in range(8):
        values = q[..., g * 32:(g + 1) * 32].astype(np.float32)
        weights[..., g * 32:(g + 1) * 32] = (
            (d * scales[..., g])[..., None] * values
            - (dmin * minimums[..., g])[..., None])
    return weights


def project(raw, x, mx_header=None, integer_q=None, mmq=False):
    """FP64 reference using canonical weights and captured production Q8 bytes."""
    matrix = np.frombuffer(raw, dtype=np.uint8).reshape(ROWS, COLUMNS // 256, Q4_BLOCK)
    output = np.empty(ROWS, dtype=np.float32)
    q_input = None if integer_q is None else np.asarray(integer_q).reshape(-1, 32)
    for first in range(0, ROWS, 64):
        blocks = matrix[first:first + 64]
        weights = q4_values(blocks)
        if not mmq:
            output[first:first + len(blocks)] = (
                weights.astype(np.float64).reshape(len(blocks), -1)
                @ x.astype(np.float64)).astype(np.float32)
            continue
        # MMQ's affine Q8 header stores half(d, sum(x)); its Q4_K minimum
        # correction consumes that captured sum, matching mx_load_input.
        acc = np.zeros(len(blocks), dtype=np.float64)
        for b in range(COLUMNS // 256):
            qblock = blocks[:, b]
            ql = qblock[:, 16:]
            scale_bytes = qblock[:, 4:16]
            d = qblock[:, :2].copy().view("<f2").reshape(-1).astype(np.float64)
            dmin = qblock[:, 2:4].copy().view("<f2").reshape(-1).astype(np.float64)
            for g in range(8):
                group, lane = g, np.arange(32)
                pair, part = group // 2, group % 2
                half, in_half = lane // 16, lane % 16
                byte = pair * 32 + (in_half // 4) * 4 + (half != 0) * 16 + in_half % 4
                qw = ((ql[:, byte] >> (4 * part)) & 15).astype(np.int32)
                if g < 4:
                    s = (scale_bytes[:, g] & 63).astype(np.float64)
                    m = (scale_bytes[:, g + 4] & 63).astype(np.float64)
                else:
                    s = ((scale_bytes[:, g + 4] & 15) | ((scale_bytes[:, g - 4] >> 6) << 4)).astype(np.float64)
                    m = ((scale_bytes[:, g + 4] >> 4) | ((scale_bytes[:, g] >> 6) << 4)).astype(np.float64)
                qi = q_input[b * 8 + g].astype(np.int32)
                dot = np.sum(qw * qi[None, :], axis=1, dtype=np.int64)
                if mx_header is None:
                    raise ValueError("MMQ reference requires decoded affine Mx header")
                input_scale = mx_header[b * 8 + g, 0]
                input_sum = mx_header[b * 8 + g, 1]
                acc += d * input_scale * s * dot - dmin * m * input_sum
        output[first:first + len(blocks)] = acc.astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("non-finite Gate/Up reference output")
    return output


def silu_mul(gate, up):
    g = gate.astype(np.float64)
    output = (g / (1.0 + np.exp(-g)) * up.astype(np.float64)).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("non-finite SwiGLU reference output")
    return output


def self_check():
    vector = np.linspace(-1, 1, COLUMNS, dtype=np.float32)
    assert len(quantize_q8_1(vector)) == (COLUMNS // 32) * Q8_BLOCK
    assert len(quantize_mx_affine(vector)) == (COLUMNS // 128) * 144
    block = np.zeros((1, 1, Q4_BLOCK), dtype=np.uint8)
    block[0, 0, 0:4] = np.frombuffer(np.asarray([1, 0], dtype="<f2").tobytes(),
                                      dtype=np.uint8)
    block[0, 0, 4:16] = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    block[0, 0, 16:] = 0x21
    decoded = q4_values(block).reshape(-1)
    assert np.array_equal(decoded[::32], np.tile([1, 2], 4))
    assert np.isfinite(decoded).all()


def main():
    self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    d = args.run_dir
    route = {}
    for name in ("control", "mmq_only"):
        prefix = d / f"{name}-optrace.decode-pos0.layer0"
        fields = ("layer_input", "attn_norm", "beta", "decay", "qkv", "q_conv",
                  "k_conv", "v_conv", "gated_output", "post_normalized",
                  "ffn_activation", "ffn_down", "layer_output")
        route[name] = {field: read_f32(prefix.with_name(prefix.name + f".{field}.f32"),
                                       17408 if field.startswith("ffn_") and field != "ffn_down"
                                       else 5120 if field in ("layer_input", "attn_norm", "post_normalized", "ffn_down", "layer_output")
                                       else 48 if field in ("beta", "decay") else 10240 if field == "qkv"
                                       else 2048 if field in ("q_conv", "k_conv") else 6144)
                      for field in fields}
    common = ("layer_input", "attn_norm", "beta", "decay", "qkv", "q_conv", "k_conv",
              "v_conv", "gated_output", "post_normalized")
    common_identity = {field: stats(route["control"][field], route["mmq_only"][field])
                       for field in common}
    if not all(row["bitwise_equal"] for row in common_identity.values()):
        raise ValueError("route inputs/stages diverge before FFN Gate/Up")
    for field in ("state_before", "state_after_gdn"):
        left = (d / f"control-optrace.decode-pos0.layer0.{field}.bin").read_bytes()
        right = (d / f"mmq_only-optrace.decode-pos0.layer0.{field}.bin").read_bytes()
        if len(left) != 3_145_728:
            raise ValueError(f"unexpected recurrent {field} byte count: {len(left)}")
        if left != right:
            raise ValueError(f"route recurrent {field} states differ before/at Gate/Up")
    fused_q_path = d / "control-optrace.decode-pos0.layer0.gateup_input_q8_1.bin"
    mmq_q_path = d / "mmq_only-optrace.decode-pos0.layer0.gateup_input_mx_q8.bin"
    fused_raw, mmq_raw = fused_q_path.read_bytes(), mmq_q_path.read_bytes()
    post = route["control"]["post_normalized"]
    if quantize_q8_1(post) != fused_raw:
        raise ValueError("captured Q8_1 bytes do not match independent production quantizer")
    if quantize_mx_affine(post) != mmq_raw:
        raise ValueError("captured Mx Q8 bytes do not match independent production quantizer")
    fused_x, fused_q, fused_d, fused_sum_q = decode_q8_1(fused_raw)
    mmq_x, mmq_q, mmq_d, mmq_sum = decode_mx_affine(mmq_raw)
    if not np.array_equal(fused_q, mmq_q.reshape(-1, 32)):
        raise ValueError("fused and MMQ routes quantized the same input to different Q8 integers")
    if not np.array_equal(fused_d, mmq_d.reshape(-1)):
        raise ValueError("fused and MMQ routes stored different Q8 scales")
    prompt_hash = sha(d / "prompt.ids")
    if prompt_hash != "5acd37f00d9e670f4b4e06d46fa12349e1dee06e356e6e73a186d74672f5bb47":
        raise ValueError("capture does not use the frozen M31-LC-0002 prompt")
    capture_records = {name: json.loads((d / f"{name}-pair-1.record.json").read_text())
                       for name in ("control", "mmq_only")}
    for name, record in capture_records.items():
        if (record["effective_config"].get("model_sha256")
                != "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
                or record["expected_prompt_tokens"] != 2048
                or record["expected_context_capacity_tokens"] != 2304
                or record["requested_environment"].get("MIINFER_HIP_GRAPH") != "0"
                or record["guard"]["exit_code"] != 0
                or not record["guard"]["clean_process_group"]
                or not record["raw_logits_capture"]["valid"]):
            raise ValueError(f"{name} run does not satisfy capture identity/guard gates")
    token_histories = [record["generated_token_ids"] for record in capture_records.values()]
    if token_histories[0] != token_histories[1] or token_histories[0][0] != 220:
        raise ValueError("route histories differ before the captured first decode step")
    source_identity = json.loads((d / "source-identity.json").read_text())
    if (source_identity["source_sha"] != "d4905bd5be1a4c3c32e466f319cee12cdd313f56"
            or source_identity["binary_sha256"] != "a71929408b3792db8c7a37dc2034fd6c4cce498d6c6a894849c426f13926c894"
            or source_identity["model_sha256"] != "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
            or source_identity["container_digest"]
            != "sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e"
            or source_identity["gpu"]["pci_bdf"] != "0000:06:00.0"):
        raise ValueError("source, model, image or GPU identity mismatch")
    refs = {}
    gate_raw = (d / "ffn_gate.q4k.bin").read_bytes()
    up_raw = (d / "ffn_up.q4k.bin").read_bytes()
    if len(gate_raw) != ROWS * (COLUMNS // 256) * Q4_BLOCK or len(up_raw) != len(gate_raw):
        raise ValueError("canonical Q4_K weight capture has an unexpected byte count")
    for tensor, raw in (("gate", gate_raw), ("up", up_raw)):
        refs[f"{tensor}_full_precision"] = project(raw, post)
        refs[f"{tensor}_fused_q8_1"] = project(raw, fused_x)
        refs[f"{tensor}_mmq_mx_q8"] = project(raw, mmq_x, np.stack((mmq_d, mmq_sum), axis=-1).reshape(-1, 2), mmq_q, True)
    fused_ref = silu_mul(refs["gate_fused_q8_1"], refs["up_fused_q8_1"])
    mmq_ref = silu_mul(refs["gate_mmq_mx_q8"], refs["up_mmq_mx_q8"])
    mmq_gate = read_f32(d / "mmq_only-optrace.decode-pos0.layer0.ffn_gate.f32", ROWS)
    mmq_up = read_f32(d / "mmq_only-optrace.decode-pos0.layer0.ffn_up.f32", ROWS)
    mmq_activation = route["mmq_only"]["ffn_activation"]
    fused_activation = route["control"]["ffn_activation"]
    records = {}
    for tensor in ("gate", "up"):
        records[tensor] = {
            "activation_quantization_fused": stats(refs[f"{tensor}_full_precision"], refs[f"{tensor}_fused_q8_1"]),
            "activation_quantization_mmq": stats(refs[f"{tensor}_full_precision"], refs[f"{tensor}_mmq_mx_q8"]),
            "mmq_kernel_vs_quantized_reference": stats(refs[f"{tensor}_mmq_mx_q8"], mmq_gate if tensor == "gate" else mmq_up),
        }
    result = {
        "schema": "m31-lc-0005-gateup-attribution-v1",
        "case": {"position": 0, "layer": 0, "decode_step": 1,
                 "model_sha256": "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
                 "prompt_ids_sha256": prompt_hash,
                 "forced_history_prefix": [220], "hip_graph": False},
        "capture_identity": {
            "source_sha": source_identity["source_sha"],
            "binary_sha256": source_identity["binary_sha256"],
            "image_digest": source_identity["container_digest"],
            "gpu_bdf": source_identity["gpu"]["pci_bdf"],
            "source_manifest_graph_setting_mismatch":
                source_identity["sampler_config"]["use_hip_graph"] != "false",
            **{name: {"tokens": record["generated_token_ids"],
                   "raw_logits_valid": record["raw_logits_capture"]["valid"],
                   "guard_exit": record["guard"]["exit_code"],
                   "clean_process_group": record["guard"]["clean_process_group"],
                   "effective_hip_graph": record["effective_config"]["use_hip_graph"]}
               for name, record in capture_records.items()}},
        "common_stages_bitwise_equal_through_post_normalized": common_identity,
        "quantized_inputs": {
            "fused_q8_1": {"sha256": sha(fused_q_path), "bytes": len(fused_raw),
                            "reproduced_from_post_normalized": True},
            "mmq_mx_q8": {"sha256": sha(mmq_q_path), "bytes": len(mmq_raw),
                          "reproduced_from_post_normalized": True},
            "decoded_activation_difference": stats(fused_x, mmq_x),
            "integer_q_bitwise_equal": True,
            "half_scale_bitwise_equal": True,
            "fused_quantized_activation_sum_vs_mmq_original_input_sum": stats(
                (fused_q.astype(np.int32).sum(axis=1) * fused_d).astype(np.float32),
                mmq_sum.reshape(-1)),
            "sum_groups_with_different_correction": int(np.count_nonzero(
                (fused_q.astype(np.int32).sum(axis=1) * fused_d).astype(np.float32)
                != mmq_sum.reshape(-1))),
            "q8_1_s_metadata_sha256": hashlib.sha256(fused_sum_q.tobytes()).hexdigest(),
        },
        "canonical_q4k_weights": {"gate_sha256": hashlib.sha256(gate_raw).hexdigest(),
                                  "up_sha256": hashlib.sha256(up_raw).hexdigest(),
                                  "bytes_each": len(gate_raw), "shape": [ROWS, COLUMNS]},
        "projection_reference": records,
        "activation_outputs": {
            "fused_gpu_vs_own_q8_1_reference": stats(fused_ref, fused_activation),
            "mmq_gpu_vs_own_mx_q8_reference": stats(mmq_ref, mmq_activation),
            "gpu_fused_vs_gpu_mmq": stats(fused_activation, mmq_activation),
            "reference_fused_vs_reference_mmq": stats(fused_ref, mmq_ref),
        },
        "downstream": {"ffn_down": stats(route["control"]["ffn_down"], route["mmq_only"]["ffn_down"]),
                       "layer_output": stats(route["control"]["layer_output"], route["mmq_only"]["layer_output"])},
        "first_differing_comparable_stage": "ffn_activation",
        "first_differing_route_contract": "Q4_K affine minimum correction uses quantized activation sum in fused Q8_1 and captured original-input sum in MMQ Mx Q8_1",
        "self_check": "PASS",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
