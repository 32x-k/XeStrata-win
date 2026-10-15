# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for setup.py's Windows parts that run anywhere: the GPUs read from `strata-device --list-devices` (an AMD
GPU's architecture and HIP number), whether an AMD GPU can be used there, and the folders whose model files setup may
move (XeStrata's only, never upstream Strata's).

    python -m unittest tools.test_setup_windows
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

DEVICES = [
    {"ordinal": 0, "name": "AMD Radeon AI PRO R9700", "backend": "HIP", "vendor_id": 4098, "device_id": 0,
     "pci": "0000:66:00.0", "total_bytes": 34208743424, "free_bytes": 34036973568, "integrated": False,
     "compute_units": 32, "arch": "gfx1201", "backend_ordinal": 0},
    {"ordinal": 1, "name": "AMD Radeon(TM) 890M Graphics", "backend": "HIP", "vendor_id": 4098, "device_id": 0,
     "pci": "0000:68:00.0", "total_bytes": 15415410688, "free_bytes": 15239905280, "integrated": True,
     "compute_units": 8, "arch": "gfx1150", "backend_ordinal": 1},
]
LISTING = "\n".join(json.dumps(d) for d in DEVICES) + "\n"


class GpusFromEngine(unittest.TestCase):
    def listed(self):
        with mock.patch.object(setup, "engine_tool", return_value=LISTING) as tool:
            found = setup.gpus_from_engine()
        return found, tool

    def test_amd_gpus_with_architecture_and_hip_number(self):
        found, _ = self.listed()
        self.assertEqual([g["vendor"] for g in found], ["amd", "amd"])
        self.assertEqual([g["gfx"] for g in found], ["gfx1201", "gfx1150"])
        self.assertEqual([g["hip"] for g in found], [0, 1])
        self.assertTrue(found[1]["integrated"])
        self.assertEqual(found[1]["vram_gb"], 0.0)

    def test_every_gpu_listed(self):
        # the caller's selection does not narrow the listing: HIP's numbers are those of every GPU
        with mock.patch.dict("os.environ", {"ONEAPI_DEVICE_SELECTOR": "hip:0", "HIP_VISIBLE_DEVICES": "0"}):
            _, tool = self.listed()
        env = tool.call_args.kwargs["env"]
        self.assertNotIn("ONEAPI_DEVICE_SELECTOR", env)
        self.assertNotIn("HIP_VISIBLE_DEVICES", env)


class AmdOnWindows(unittest.TestCase):
    def problem(self, gfx, archs):
        g = {"vendor": "amd", "gfx": gfx}
        with mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "PACKAGED", {"engine": "engine"}), \
                mock.patch.object(setup, "packaged_engine", return_value={"package": "p", "hip_archs": archs}):
            return setup.gpu_problem(g)

    def test_an_architecture_the_engine_has(self):
        self.assertIsNone(self.problem("gfx1201", ["gfx1150", "gfx1201"]))

    def test_an_architecture_the_engine_lacks(self):
        self.assertIn("no code for gfx1201", self.problem("gfx1201", ["gfx1100"]))

    def test_one_the_runtime_does_not_list(self):
        self.assertIn("HIP runtime", self.problem(None, ["gfx1201"]))


class HipSelector(unittest.TestCase):
    def selector(self, found, sel, win=True):
        cfg: dict = {}
        info = {g["index"]: g for g in found}
        with mock.patch.object(setup, "WIN", win), mock.patch.object(setup, "gpu_info", side_effect=info.get):
            setup.hip_selector(cfg, sel, found)
        return (cfg.get("env") or {}).get("ONEAPI_DEVICE_SELECTOR")

    AMD = [{"index": 0, "hip": 0}, {"index": 1, "hip": 1}]

    def test_the_chosen_hip_gpus(self):
        self.assertEqual(self.selector(self.AMD, [0]), "hip:0")
        self.assertEqual(self.selector(self.AMD, [1, 0]), "hip:1,0")

    def test_one_gpu_or_not_windows(self):
        self.assertIsNone(self.selector(self.AMD[:1], [0]))
        self.assertIsNone(self.selector(self.AMD, [0], win=False))

    def test_a_gpu_of_another_backend_among_them(self):
        # an Intel Arc beside the R9700: a hip: selector would hide it from the engine, and "None" is no number
        mixed = [{"index": 0, "hip": None}, {"index": 1, "hip": 0}]
        self.assertIsNone(self.selector(mixed, [1, 0]))
        self.assertEqual(self.selector(mixed, [1]), "hip:0")


class OtherInstalls(unittest.TestCase):
    def test_upstream_strata_folders_are_not_taken(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mine = base / "XeStrata-main"
            for d in (mine, base / "Strata", base / "Strata-main", base / "XeStrata-main (1)" / "XeStrata-main"):
                d.mkdir(parents=True)
                (d / "setup.py").write_text("", encoding="utf-8")
            with mock.patch.object(setup, "ROOT", mine):
                found = setup.other_installs({})
            # the copy unpacked again, inside its "(1)" folder; never Strata or Strata-main
            self.assertEqual(found, [(base / "XeStrata-main (1)" / "XeStrata-main").resolve()])


if __name__ == "__main__":
    unittest.main()
