"""H100 fresh-install items for DeepSeek in the Terminal: the screens' packages, the Advanced list's own flags, the disk in "fits"."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from strata_cli.launch import cmd, params as P  # noqa: E402
from strata_cli.launch.catalog import TIGHT, verdict  # noqa: E402

CARD = {"index": 0, "name": "RTX 5070", "vram_gb": 12.0, "vendor": "nvidia", "usable": True}


class Screens(unittest.TestCase):
    def test_missing_package_is_named_not_a_traceback(self):
        import importlib.util as u
        real = u.find_spec
        with mock.patch.object(u, "find_spec", lambda n, *a, **k: None if n == "rich" else real(n, *a, **k)):
            self.assertEqual(cmd._screens_missing(), "rich")
        with mock.patch.object(u, "find_spec", lambda n, *a, **k: None if n == "textual" else real(n, *a, **k)):
            self.assertEqual(cmd._screens_missing(), "textual")

    def test_setup_installs_the_screens_packages(self):
        import setup
        names = [r.split("==")[0].split(";")[0].strip() for r in setup.runtime_packages()]
        self.assertIn("textual", names)
        self.assertIn("numpy", names)


class Advanced(unittest.TestCase):
    ARGS = ["--pack", "P", "--max-context", "8192", "--eos", "1", "--dspark", "auto", "--dspark-conf", "0.65"]

    def test_deepseeks_own_flags_are_known_before_the_engine_is_installed(self):
        rows = P.build_rows(P.dsv41_help_text(None), self.ARGS, None, {}, {}, {})
        mine = {r.name: r for r in rows if r.name in self.ARGS}
        self.assertEqual(set(mine), {"--pack", "--max-context", "--eos", "--dspark", "--dspark-conf"})
        for r in mine.values():
            self.assertNotIn("not in this engine's list", r.description)
            self.assertTrue(r.known)

    def test_a_real_typing_mistake_is_still_flagged(self):
        rows = P.build_rows(P.dsv41_help_text(None), self.ARGS + ["--dsparkk", "1"], None, {}, {}, {})
        self.assertIn("not in this engine's list", next(r for r in rows if r.name == "--dsparkk").description)


class Disk(unittest.TestCase):
    def setUp(self):
        import strata_mcp as M
        self.s = M.Strata(ROOT)

    def hw(self, disk):
        return {"ram_gb": 64.0, "gpus": [CARD], "engine_vendor": "nvidia", "disk_free_gb": disk}

    def test_fits_in_memory_but_not_on_the_disk_is_tight_with_the_numbers(self):
        w = self.s.dsv41_words(self.hw(37.0), "IST-DAS-2.25bpw")
        self.assertTrue(w.startswith("tight"), w)
        self.assertIn("free disk space", w)
        self.assertIn("37 GB", w)
        self.assertEqual(verdict(w), (TIGHT, "tight: needs more disk space"))

    def test_enough_disk_or_already_installed_is_unchanged(self):
        self.assertNotIn("free disk space", self.s.dsv41_words(self.hw(900.0), "IST-DAS-2.25bpw"))
        self.assertNotIn("free disk space", self.s.dsv41_words(self.hw(37.0), "IST-DAS-2.25bpw", installed=True))
        self.assertNotIn("free disk space", self.s.dsv41_words({**self.hw(None)}, "IST-DAS-2.25bpw"))


if __name__ == "__main__":
    unittest.main()
