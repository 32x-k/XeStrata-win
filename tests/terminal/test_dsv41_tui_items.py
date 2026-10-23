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


class Presets(unittest.TestCase):
    def test_a_saved_preset_is_read_from_where_it_is_written_whatever_the_data_dir(self):
        import tempfile
        from strata_cli.launch import settings as S, store
        with tempfile.TemporaryDirectory() as d, mock.patch.dict("os.environ", {"STRATA_LAUNCH_PRESET_DIR": d}):
            s = S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", data_dir=str(Path(d) / "elsewhere"), name="DS 8K")
            store.save(s, "DS 8K")
            self.assertEqual([p.name for p in store.list_presets()], ["DS 8K"])
            self.assertEqual(store.preset_dir(), Path(d))          # (the folder does not depend on the launch's --data-dir)


class PlanEngine(unittest.TestCase):
    def test_an_installed_deepseek_engine_is_named_in_the_plan_not_downloaded(self):
        import tempfile
        from strata_cli.launch import runner as R
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "engine-dsv41" / "0.1.49-windows-x86_64-cuda-cuda13"
            pkg.mkdir(parents=True)
            exe = pkg / ("strata-dsv41.exe" if sys.platform == "win32" else "strata-dsv41")
            exe.write_bytes(b"x")
            fake = mock.MagicMock()
            fake.s.data_dir = d
            fake.catalog.data_dir.return_value = Path(d)
            fake.catalog.strata.setup_module.side_effect = RuntimeError("no setup")
            with mock.patch.dict("os.environ", {"STRATA_DSV41_EXE": ""}),                     mock.patch("strata_control.engine_component.status", return_value={"usable": False}):
                found, known = R.Runner.dsv41_engine_found(fake)
            self.assertEqual(found, exe)
            self.assertTrue(known)


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
