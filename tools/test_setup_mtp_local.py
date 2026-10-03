"""Local MTP selection must reuse verified tensors or fail without a surprise download."""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup


class LocalMTP(unittest.TestCase):
    def test_local_gguf_selects_importer(self):
        with mock.patch.object(setup, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            setup.prepare_mtp_source(Path("mtp"), "/models/mtp-BF16.gguf", {"example": "value"})
        args = run.call_args.args[0]
        self.assertEqual(Path(args[1]).name, "mtp_import.py")
        self.assertEqual(args[2:], ["--gguf", "/models/mtp-BF16.gguf", "--out", "mtp"])
        self.assertEqual(run.call_args.kwargs["env"], {"example": "value"})
        self.assertEqual(run.call_count, 1)

    def test_import_failure_is_not_replaced_by_download(self):
        with mock.patch.object(setup, "run", side_effect=SystemExit(3)) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                setup.prepare_mtp_source(Path("mtp"), "/models/wrong.gguf")
        self.assertEqual(failure.exception.code, 3)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(Path(run.call_args.args[0][1]).name, "mtp_import.py")

    def test_default_download_path_is_preserved(self):
        with mock.patch.object(setup, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            setup.prepare_mtp_source(Path("mtp"))
        self.assertEqual(Path(run.call_args.args[0][1]).name, "mtp_fetch.py")
        self.assertEqual(run.call_args.args[0][2:], ["fetch", "--out", "mtp"])


if __name__ == "__main__":
    unittest.main()
