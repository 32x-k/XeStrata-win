# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""setup.verify_sha256: a file with the wrong SHA-256 that this run downloaded is deleted (the next run downloads it
again); one the person already had (--gguf-dir, or put in the models folder by hand) is left as it is."""
import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

BODY = b"not the pinned file"
WRONG = hashlib.sha256(b"the pinned file").hexdigest()


class ShaKeep(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.f = self.d / "model.gguf"
        self.f.write_bytes(BODY)
        setup.FETCHED.clear()

    def tearDown(self):
        setup.FETCHED.clear()

    def check(self, size=len(BODY), sha=WRONG):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            setup.verify_sha256(self.f, size, sha)
        return out.getvalue()

    def test_own_file_is_kept(self):
        out = self.check()
        self.assertTrue(self.f.exists())
        self.assertEqual(self.f.read_bytes(), BODY)
        self.assertIn("left as it is", out)

    def test_own_file_of_the_wrong_size_is_kept(self):
        out = self.check(size=len(BODY) + 1)
        self.assertTrue(self.f.exists())
        self.assertIn("not the pinned file", out)

    def test_downloaded_file_is_deleted(self):
        setup.FETCHED.add(self.f)
        setup.mark(self.f)
        out = self.check()
        self.assertFalse(self.f.exists())
        self.assertFalse(self.f.with_name(self.f.name + ".done").exists())
        self.assertIn("deleted", out)

    def test_right_file_is_marked(self):
        setup.verify_sha256(self.f, len(BODY), hashlib.sha256(BODY).hexdigest())
        self.assertTrue(self.f.exists())
        self.assertIn("sha256 ", self.f.with_name(self.f.name + ".done").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
