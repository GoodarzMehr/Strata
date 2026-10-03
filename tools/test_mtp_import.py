"""Offline MTP GGUF import tests: layouts, rounded Gemma norms, and pinned-byte refusal.

    python tools/test_mtp_import.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_import as M  # noqa: E402


TINY = {
    "mtp.fc_embedding.weight": [4, 4],
    "mtp.fc_hidden.weight": [4, 4],
    "mtp.hyper_connection_mixer.hc_norm.weight": [8],
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": [2, 8],
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": [8, 2],
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": [2, 8],
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": [8],
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": [2, 8],
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": [8, 2],
    "mtp.layers.0.mlp.experts.down_proj": [3, 4, 2],
    "mtp.layers.0.mlp.experts.gate_up_proj": [3, 4, 4],
    "mtp.layers.0.mlp.gate.weight": [3, 4],
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": [4, 2],
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": [2, 4],
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": [2, 4],
    "mtp.layers.0.mlp.shared_expert_gate.weight": [1, 4],
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": [2, 8],
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": [8],
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": [2, 8],
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": [8, 2],
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": [6, 4],
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": [2],
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": [2],
    "mtp.layers.0.self_attn.k_norm.weight": [2],
    "mtp.layers.0.self_attn.k_proj.weight": [2, 4],
    "mtp.layers.0.self_attn.o_proj.weight": [4, 4],
    "mtp.layers.0.self_attn.q_norm.weight": [2],
    "mtp.layers.0.self_attn.q_proj.weight": [8, 4],
    "mtp.layers.0.self_attn.v_proj.weight": [2, 4],
    "mtp.pre_fc_norm_embedding.weight": [4],
    "mtp.pre_fc_norm_hidden.weight": [8],
}


def f32(bits):
    return (np.asarray(bits, dtype="<u2").astype("<u4") << 16).view("<f4")


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path, tensors):
    """Minimal fixture writer: tensor tuples are (name, GGUF shape, type id, bytes)."""
    header = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), 1))
    header += string("general.architecture") + struct.pack("<I", 8) + string("qwen4exp")
    offset = 0
    data = bytearray()
    for name, shape, type_id, raw in tensors:
        header += string(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        header += struct.pack("<IQ", type_id, offset)
        data += raw
        data += b"\0" * ((-len(raw)) % 32)
        offset = len(data)
    header += b"\0" * ((-len(header)) % 32)
    path.write_bytes(header + data)


def fixture_tensors():
    expected = {}
    for index, (name, shape) in enumerate(sorted(TINY.items())):
        values = (np.arange(math.prod(shape), dtype="<f4") / 8 + np.float32(index * 2)).reshape(shape)
        bits = (values.view("<u4") >> 16).astype("<u2")
        expected[name] = bits
    # These original BF16 values cannot be recovered by just subtracting 1 from the F32 norm.
    expected["mtp.layers.0.mlp_hyper_connection.hc_norm.weight"][0] = 46809
    expected["mtp.pre_fc_norm_hidden.weight"][0] = 13406
    tensors = []
    for name, short in M.DIRECT.items():
        bits = expected[name]
        router = name.endswith("mlp.gate.weight") or name.endswith("mlp.shared_expert_gate.weight")
        if bits.ndim == 1:
            type_id, raw = 0, (f32(bits) + np.float32(1)).tobytes()
        elif router:
            type_id, raw = 0, f32(bits).tobytes()
        else:
            type_id, raw = 30, bits.tobytes()
        tensors.append((M.PREFIX + short, list(reversed(bits.shape)), type_id, raw))
    eh = np.concatenate((expected["mtp.fc_embedding.weight"], expected["mtp.fc_hidden.weight"]), axis=1)
    tensors.append((M.PREFIX + "nextn.eh_proj.weight", list(reversed(eh.shape)), 30, eh.tobytes()))
    gu = expected["mtp.layers.0.mlp.experts.gate_up_proj"]
    for short, part in (("ffn_gate_exps.weight", gu[:, :2]), ("ffn_up_exps.weight", gu[:, 2:])):
        tensors.append((M.PREFIX + short, list(reversed(part.shape)), 30, part.tobytes()))
    qk = expected["mtp.layers.0.self_attn.indexer.index_qk_proj.weight"]
    for short, part in (("indexer.q_proj.weight", qk[:4]), ("indexer.k_proj.weight", qk[4:])):
        tensors.append((M.PREFIX + short, list(reversed(part.shape)), 30, part.tobytes()))
    return tensors, {name: bits.tobytes() for name, bits in expected.items()}


class Import(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source, self.out = self.root / "shared-BF16.gguf", self.root / "import"
        self.tensors, self.expected = fixture_tensors()
        self.hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in self.expected.items()}
        self.patches = [mock.patch.object(M, "SHAPES", TINY),
                        mock.patch.object(M.mtp_fetch, "SHA256", self.hashes),
                        mock.patch.object(M, "CHUNK_BYTES", 32)]
        for patch in self.patches:
            patch.start()
        write_gguf(self.source, self.tensors)

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def run_import(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return M.import_tensors(self.source, self.out)

    def test_all_tensors_match_checkpoint_layout_and_verify(self):
        manifest = self.run_import()
        self.assertEqual(len(manifest), 31)
        for item in manifest:
            with self.subTest(name=item["name"]):
                self.assertEqual(item["shape"], TINY[item["name"]])
                self.assertEqual(item["dtype"], "BF16")
                self.assertEqual((self.out / item["file"]).read_bytes(), self.expected[item["name"]])
                self.assertEqual(item["sha256"], self.hashes[item["name"]])
        with mock.patch.object(M.mtp_fetch, "REPO", M.mtp_fetch.PINNED), mock.patch.object(
                M.mtp_fetch, "sha256_of", side_effect=AssertionError("verified bytes were hashed again")):
            self.assertEqual(M.mtp_fetch.verify(str(self.out)), [])
        self.assertEqual(json.loads((self.out / "mtp-manifest.json").read_text()), manifest)

    def test_gate_up_is_fused_per_expert_and_eh_is_split_per_row(self):
        self.run_import()
        gu = np.fromfile(self.out / "tensors/mtp.layers.0.mlp.experts.gate_up_proj.bin", dtype="<u2").reshape(3, 4, 4)
        source = {name: raw for name, _, _, raw in self.tensors}
        gate = np.frombuffer(source[M.PREFIX + "ffn_gate_exps.weight"], dtype="<u2").reshape(3, 2, 4)
        up = np.frombuffer(source[M.PREFIX + "ffn_up_exps.weight"], dtype="<u2").reshape(3, 2, 4)
        for expert in range(3):
            np.testing.assert_array_equal(gu[expert, :2], gate[expert])
            np.testing.assert_array_equal(gu[expert, 2:], up[expert])
        self.assertNotEqual(self.expected["mtp.fc_embedding.weight"], self.expected["mtp.fc_hidden.weight"])
        for name in ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight"):
            self.assertEqual((self.out / "tensors" / (name + ".bin")).read_bytes(), self.expected[name])

    def test_verified_existing_files_are_reused(self):
        self.run_import()
        times = {name: (self.out / "tensors" / (name + ".bin")).stat().st_mtime_ns for name in self.expected}
        self.run_import()
        self.assertEqual(times, {name: (self.out / "tensors" / (name + ".bin")).stat().st_mtime_ns
                                 for name in self.expected})

    def test_wrong_pinned_tensor_does_not_replace_existing_file_or_manifest(self):
        name = "mtp.fc_embedding.weight"
        self.out.mkdir()
        (self.out / "tensors").mkdir()
        target = self.out / "tensors" / (name + ".bin")
        target.write_bytes(b"keep existing output")
        (self.out / "mtp-manifest.json").write_text("existing manifest")
        index = next(i for i, t in enumerate(self.tensors) if t[0].endswith("nextn.eh_proj.weight"))
        n, shape, kind, raw = self.tensors[index]
        self.tensors[index] = (n, shape, kind, bytes([raw[0] ^ 1]) + raw[1:])
        write_gguf(self.source, self.tensors)
        with self.assertRaisesRegex(ValueError, "do not match the pinned checkpoint"):
            self.run_import()
        self.assertEqual(target.read_bytes(), b"keep existing output")
        self.assertEqual((self.out / "mtp-manifest.json").read_text(), "existing manifest")
        self.assertEqual(list(self.out.rglob("*.tmp")), [])

    def test_quantized_experts_wrong_shapes_and_missing_data_are_refused_before_writes(self):
        gate = next(i for i, t in enumerate(self.tensors) if t[0].endswith("ffn_gate_exps.weight"))
        bad_cases = []
        modified = list(self.tensors)
        name, shape, kind, raw = modified[gate]
        modified[gate] = (name, shape, 12, raw)
        bad_cases.append((modified, "not a quantized MTP"))
        modified = list(self.tensors)
        modified[gate] = (name, [shape[0], shape[1] + 1, shape[2]], kind, raw)
        bad_cases.append((modified, "GGUF shape"))
        bad_cases.append((self.tensors[:gate] + self.tensors[gate + 1:], "missing BF16"))
        for tensors, message in bad_cases:
            with self.subTest(message=message):
                write_gguf(self.source, tensors)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_import()
                self.assertFalse(self.out.exists())
        write_gguf(self.source, self.tensors)
        self.source.write_bytes(self.source.read_bytes()[:-40])
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.run_import()
        self.assertFalse(self.out.exists())

    def test_inexact_f32_router_is_refused(self):
        index = next(i for i, t in enumerate(self.tensors) if t[0].endswith("ffn_gate_inp.weight"))
        name, shape, kind, raw = self.tensors[index]
        values = np.frombuffer(raw, dtype="<f4").copy()
        values[0] = np.nextafter(values[0], np.float32(100))
        self.tensors[index] = name, shape, kind, values.tobytes()
        write_gguf(self.source, self.tensors)
        with self.assertRaisesRegex(ValueError, "losslessly"):
            self.run_import()
        self.assertFalse((self.out / "mtp-manifest.json").exists())


class Norms(unittest.TestCase):
    def test_gemma_rounding_is_recovered_only_by_the_pinned_hash(self):
        original = np.asarray([46809, 13406, 0x3F80, 0xBF00], dtype="<u2").tobytes()
        shifted = (f32(np.frombuffer(original, dtype="<u2")) + np.float32(1)).tobytes()
        want = hashlib.sha256(original).hexdigest()
        naive = ((np.frombuffer(shifted, dtype="<f4") - 1).view("<u4") >> 16).astype("<u2").tobytes()
        self.assertNotEqual(naive, original)
        self.assertEqual(M.recover_norm(shifted, "F32", want), original)
        with self.assertRaisesRegex(ValueError, "pinned checkpoint SHA256"):
            M.recover_norm(shifted, "F32", "0" * 64)

    def test_raw_bf16_and_raw_f32_norms_are_supported_with_verified_hashes(self):
        raw = np.asarray([0x3F80, 0xBF00], dtype="<u2").tobytes()
        want = hashlib.sha256(raw).hexdigest()
        self.assertEqual(M.recover_norm(raw, "BF16", want), raw)
        self.assertEqual(M.recover_norm(f32(np.frombuffer(raw, dtype="<u2")).tobytes(), "F32", want), raw)
        with self.assertRaisesRegex(ValueError, "does not match"):
            M.recover_norm(raw, "BF16", "0" * 64)

    def test_oversized_ambiguity_and_nonfinite_norms_are_refused(self):
        with self.assertRaisesRegex(ValueError, "ambiguous Gemma norm"):
            M.recover_norm(np.array([1.], dtype="<f4").tobytes(), "F32", "0" * 64)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            M.recover_norm(np.array([np.nan], dtype="<f4").tobytes(), "F32", "0" * 64)


if __name__ == "__main__":
    unittest.main()
