"""Import the pinned BF16 MTP tensors from a local llama.cpp-style GGUF, without downloading them.

    python tools/mtp_import.py --gguf ~/LLMs/mtp-Qwen3.8-Flash-Next-BF16.gguf --out mtp-bf16

Writes the same raw BF16 tensors and mtp-manifest.json as mtp_fetch.py, for mtp_pack.py and mtp_rt.py.
The GGUF's embedding and output tensors are not needed. A shared BF16 MTP GGUF also works. Quantized
GGUFs are refused: every reconstructed tensor must match mtp_fetch.py's pinned checkpoint SHA256.
The reader has no gguf-py dependency; large tensors are copied in bounded chunks, not loaded into RAM.

GGUF conversion applies Gemma's 1+w to F32 norms. Its rounding can erase tiny BF16 bits: the importer
enumerates exact BF16 inverses and accepts only a full-tensor pinned hash match, with a bounded search.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_reader import GGUFFile, TensorInfo  # noqa: E402
import mtp_fetch  # noqa: E402

CHUNK_BYTES = 8 << 20
MAX_NORM_CANDIDATES = 4096
PREFIX = "blk.48."

# Shapes are the pinned checkpoint's row-major shapes, not GGUF's innermost-first dimensions.
SHAPES = {
    "mtp.fc_embedding.weight": [2560, 2560],
    "mtp.fc_hidden.weight": [2560, 2560],
    "mtp.hyper_connection_mixer.hc_norm.weight": [10240],
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": [320, 10240],
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": [4, 10240],
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": [10240],
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": [320, 10240],
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.mlp.experts.down_proj": [512, 2560, 640],
    "mtp.layers.0.mlp.experts.gate_up_proj": [512, 1280, 2560],
    "mtp.layers.0.mlp.gate.weight": [512, 2560],
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": [2560, 640],
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": [640, 2560],
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": [640, 2560],
    "mtp.layers.0.mlp.shared_expert_gate.weight": [1, 2560],
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": [4, 10240],
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": [10240],
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": [320, 10240],
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": [640, 2560],
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": [128],
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": [128],
    "mtp.layers.0.self_attn.k_norm.weight": [256],
    "mtp.layers.0.self_attn.k_proj.weight": [512, 2560],
    "mtp.layers.0.self_attn.o_proj.weight": [2560, 6144],
    "mtp.layers.0.self_attn.q_norm.weight": [256],
    "mtp.layers.0.self_attn.q_proj.weight": [12288, 2560],
    "mtp.layers.0.self_attn.v_proj.weight": [512, 2560],
    "mtp.pre_fc_norm_embedding.weight": [2560],
    "mtp.pre_fc_norm_hidden.weight": [10240],
}

DIRECT = {
    "mtp.hyper_connection_mixer.hc_norm.weight": "nextn.hc_head_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": "nextn.hc_head_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": "nextn.hc_head_up.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": "hc_attn_inject.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": "hc_attn_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": "hc_attn_down.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": "hc_attn_up.weight",
    "mtp.layers.0.mlp.experts.down_proj": "ffn_down_exps.weight",
    "mtp.layers.0.mlp.gate.weight": "ffn_gate_inp.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": "hc_ffn_inject.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": "hc_ffn_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": "hc_ffn_down.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": "hc_ffn_up.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": "indexer.k_norm.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": "indexer.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight": "attn_k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight": "attn_k.weight",
    "mtp.layers.0.self_attn.o_proj.weight": "attn_output.weight",
    "mtp.layers.0.self_attn.q_norm.weight": "attn_q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight": "attn_q.weight",
    "mtp.layers.0.self_attn.v_proj.weight": "attn_v.weight",
    "mtp.pre_fc_norm_embedding.weight": "nextn.enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": "nextn.hnorm.weight",
}


@dataclass(frozen=True)
class Entry:
    name: str
    shape: list[int]
    sources: tuple[TensorInfo, ...]
    operation: str
    column_half: int = 0

    @property
    def byte_count(self) -> int:
        return math.prod(self.shape) * 2


def build_plan(gguf: GGUFFile) -> list[Entry]:
    """Validate the complete tensor directory before creating output files."""
    if gguf.metadata.get("general.architecture") != "qwen4exp":
        raise ValueError("MTP import requires a qwen4exp GGUF")
    tensors = {t.name: t for t in gguf.tensors}
    if len(tensors) != len(gguf.tensors):
        raise ValueError("duplicate GGUF tensor names")
    if set(SHAPES) != set(mtp_fetch.SHA256):
        raise ValueError("importer's tensor contract differs from the pinned checkpoint hashes")
    file_bytes = gguf.path.stat().st_size

    def tensor(short: str, shape: list[int], types: tuple[str, ...]) -> TensorInfo:
        name = PREFIX + short
        t = tensors.get(name)
        if t is None:
            raise ValueError(f"missing BF16 MTP tensor: {name}")
        if t.type_name not in types:
            raise ValueError(f"{name}: {t.type_name}; import requires {' or '.join(types)}, not a quantized MTP")
        if t.shape != list(reversed(shape)):
            raise ValueError(f"{name}: GGUF shape {t.shape}, expected {list(reversed(shape))}")
        size = t.expected_bytes()
        if size is None or t.offset < 0 or gguf.data_start + t.offset + size > file_bytes:
            raise ValueError(f"{name}: truncated or invalid tensor data range")
        return t

    entries = []
    for name, shape in sorted(SHAPES.items()):
        if name in DIRECT:
            norm = len(shape) == 1
            router = name.endswith("mlp.gate.weight") or name.endswith("mlp.shared_expert_gate.weight")
            types = ("BF16", "F32") if norm or router else ("BF16",)
            source = tensor(DIRECT[name], shape, types)
            op = "norm" if norm else ("f32_bf16" if source.type_name == "F32" else "copy")
            entries.append(Entry(name, shape, (source,), op))
        elif name in ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight"):
            rows, cols = shape
            source = tensor("nextn.eh_proj.weight", [rows, 2 * cols], ("BF16",))
            entries.append(Entry(name, shape, (source,), "split_columns", int(name == "mtp.fc_hidden.weight")))
        elif name.endswith("experts.gate_up_proj"):
            experts, rows, cols = shape
            if rows % 2:
                raise ValueError("gate_up_proj must have an even output dimension")
            parts = tuple(tensor(short, [experts, rows // 2, cols], ("BF16",)) for short in
                          ("ffn_gate_exps.weight", "ffn_up_exps.weight"))
            entries.append(Entry(name, shape, parts, "expert_gate_up"))
        elif name.endswith("indexer.index_qk_proj.weight"):
            rows, cols = shape
            q = tensors.get(PREFIX + "indexer.q_proj.weight")
            if q is None or len(q.shape) != 2:
                raise ValueError("missing 2-D MTP indexer query projection")
            q_rows = q.shape[1]
            if not 0 < q_rows < rows:
                raise ValueError("invalid MTP indexer query/key split")
            parts = (tensor("indexer.q_proj.weight", [q_rows, cols], ("BF16",)),
                     tensor("indexer.k_proj.weight", [rows - q_rows, cols], ("BF16",)))
            entries.append(Entry(name, shape, parts, "concat_rows"))
        else:
            raise ValueError(f"no import mapping for {name}")
    return entries


def _chunks(fh: BinaryIO, start: int, size: int) -> Iterator[bytes]:
    fh.seek(start)
    left = size
    while left:
        raw = fh.read(min(CHUNK_BYTES, left))
        if not raw:
            raise ValueError("GGUF tensor data ended before its declared size")
        left -= len(raw)
        yield raw


def _exact_bf16(values: np.ndarray) -> bytes:
    values = np.ascontiguousarray(values, dtype="<f4")
    bits = values.view("<u4")
    if not np.isfinite(values).all() or (bits & 0xFFFF).any():
        raise ValueError("F32 tensor cannot be converted to BF16 losslessly")
    return (bits >> 16).astype("<u2").tobytes()


def recover_norm(raw: bytes, source_type: str, want: str) -> bytes:
    """Undo 1+w only when the reconstructed BF16 bytes match the pinned checkpoint."""
    if source_type == "BF16":
        if hashlib.sha256(raw).hexdigest() == want:
            return raw
        raise ValueError("BF16 norm does not match the pinned checkpoint")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("non-finite F32 norm")
    # Some producers leave raw w rather than applying 1+w. Hashes determine the convention.
    try:
        direct = _exact_bf16(values)
    except ValueError:
        direct = None
    if direct is not None and hashlib.sha256(direct).hexdigest() == want:
        return direct
    unshifted = values - np.float32(1.0)
    try:
        base = _exact_bf16(unshifted)
    except ValueError:
        base = None
    if base is not None and hashlib.sha256(base).hexdigest() == want:
        return base

    # Enumerate finite BF16 values whose F32 1+w is bit-identical to the GGUF value. Rounded tiny
    # norms can have several inverses; subtraction alone cannot recover their original bits.
    all_bits = np.arange(65536, dtype="<u2")
    decoded = (all_bits.astype("<u4") << 16).view("<f4")
    finite = np.isfinite(decoded)
    all_bits, decoded = all_bits[finite], decoded[finite]
    shifted_bits = (decoded + np.float32(1.0)).view("<u4")
    order = np.argsort(shifted_bits)
    shifted_bits, all_bits = shifted_bits[order], all_bits[order]
    raw_bits = values.view("<u4")
    left, right = np.searchsorted(shifted_bits, raw_bits, side="left"), np.searchsorted(
        shifted_bits, raw_bits, side="right")
    counts = right - left
    if (counts == 0).any():
        raise ValueError("F32 norm is not an exact BF16 Gemma 1+w conversion")
    ambiguous = np.flatnonzero(counts > 1)
    combinations = math.prod(int(counts[i]) for i in ambiguous)
    if combinations > MAX_NORM_CANDIDATES:
        raise ValueError(f"ambiguous Gemma norm: {combinations} possible BF16 inverses; refusing import")
    reconstructed = all_bits[left].copy()
    candidates = [all_bits[left[i]:right[i]] for i in ambiguous]
    for choice in itertools.product(*candidates):
        reconstructed[ambiguous] = choice
        blob = reconstructed.astype("<u2", copy=False).tobytes()
        if hashlib.sha256(blob).hexdigest() == want:
            return blob
    raise ValueError("norm cannot be reconstructed to the pinned checkpoint SHA256")


def entry_chunks(fh: BinaryIO, gguf: GGUFFile, entry: Entry) -> Iterator[bytes]:
    if entry.operation == "norm":
        t = entry.sources[0]
        raw = b"".join(_chunks(fh, gguf.data_start + t.offset, t.expected_bytes()))
        yield recover_norm(raw, t.type_name, mtp_fetch.SHA256[entry.name])
    elif entry.operation == "f32_bf16":
        t = entry.sources[0]
        for raw in _chunks(fh, gguf.data_start + t.offset, t.expected_bytes()):
            yield _exact_bf16(np.frombuffer(raw, dtype="<f4"))
    elif entry.operation in ("copy", "concat_rows"):
        for t in entry.sources:
            yield from _chunks(fh, gguf.data_start + t.offset, t.expected_bytes())
    elif entry.operation == "expert_gate_up":
        experts, rows, cols = entry.shape
        part_bytes = rows // 2 * cols * 2
        # Checkpoint [expert, gate rows followed by up rows, hidden]. Concatenating the two whole
        # GGUF tensors would put every expert's gate before every expert's up and corrupt the draft.
        for expert in range(experts):
            for t in entry.sources:
                yield from _chunks(fh, gguf.data_start + t.offset + expert * part_bytes, part_bytes)
    elif entry.operation == "split_columns":
        rows, cols = entry.shape
        source_row_bytes = 2 * cols * 2
        batch_rows = max(1, CHUNK_BYTES // source_row_bytes)
        start = gguf.data_start + entry.sources[0].offset
        for row in range(0, rows, batch_rows):
            count = min(batch_rows, rows - row)
            raw = b"".join(_chunks(fh, start + row * source_row_bytes, count * source_row_bytes))
            source = np.frombuffer(raw, dtype="<u2").reshape(count, 2 * cols)
            yield source[:, entry.column_half * cols:(entry.column_half + 1) * cols].tobytes()
    else:
        raise ValueError(f"unknown import operation {entry.operation}")


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def import_tensors(source: Path, out: Path) -> list[dict]:
    gguf = GGUFFile(source)
    plan = build_plan(gguf)
    # Check all small norms first, before copying GB of expert tensors or changing output.
    with source.open("rb") as fh:
        for entry in plan:
            if entry.operation == "norm":
                list(entry_chunks(fh, gguf, entry))
    tdir = out / "tensors"
    tdir.mkdir(parents=True, exist_ok=True)
    manifest, stamps = [], {}
    with source.open("rb") as fh:
        for entry in plan:
            path = tdir / (entry.name + ".bin")
            want = mtp_fetch.SHA256[entry.name]
            kept = path.is_file() and path.stat().st_size == entry.byte_count and mtp_fetch.sha256_of(path) == want
            if not kept:
                tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    digest, total = hashlib.sha256(), 0
                    with tmp.open("wb") as dst:
                        for raw in entry_chunks(fh, gguf, entry):
                            dst.write(raw)
                            digest.update(raw)
                            total += len(raw)
                    if total != entry.byte_count or digest.hexdigest() != want:
                        raise ValueError(f"{entry.name}: imported bytes do not match the pinned checkpoint "
                                         f"{mtp_fetch.PINNED_REVISION}; refusing this GGUF")
                    os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)
            st = path.stat()
            stamps[entry.name] = [st.st_size, st.st_mtime_ns, want]
            manifest.append(dict(name=entry.name, dtype="BF16", shape=entry.shape, bytes=entry.byte_count,
                                 file=path.relative_to(out).as_posix(), sha256=want, shard=source.name,
                                 source_gguf=str(source.resolve()), source_tensors=[t.name for t in entry.sources],
                                 import_operation=entry.operation, pinned_revision=mtp_fetch.PINNED_REVISION))
            print(f"{entry.name}: {'kept' if kept else 'imported and verified'} {entry.byte_count / 1e6:.1f} MB",
                  flush=True)
    _atomic_json(out / "mtp-manifest.json", manifest)
    _atomic_json(tdir / "verified.json", stamps)
    print(f"verified {len(manifest)} pinned BF16 MTP tensors ({sum(e.byte_count for e in plan) / 1e9:.3f} GB)")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True, help="local full or shared BF16 MTP GGUF")
    ap.add_argument("--out", type=Path, required=True, help="raw tensor directory for mtp_pack.py")
    args = ap.parse_args()
    try:
        import_tensors(args.gguf, args.out)
    except (OSError, ValueError, KeyError) as exc:
        print(f"MTP import refused: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
