"""The three 0.2.7 speed features are on by default and have an off switch: the contract names, where each applies, and how the Terminal carries them.

The app (apps/host/server/native-settings.mjs SPEED_SWITCHES) and Control (strata_control/speed_switches.py) say the same thing; the last test reads the
app's table with node and fails when the two differ.  `strata launch` shows the environment entries as rows of its Advanced table and writes them into
the run config's "env" (`--set STRATA_KVA=0`); a default launch has none of them.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from strata_cli.launch import params as P
from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import speed_switches as W

ROOT = Path(__file__).resolve().parents[2]

OFF = {
    "faster_long_prompts": {"STRATA_KVA": "0"},
    "faster_radeon_9000": {"STRATA_PCIE_FRAC_DEFAULT": "old"},
    "deepseek_speedups": {"DS4_FAST_ATTN": "0", "DS4_ATTN_SLAB": "0", "DS4_COPY_PIN": "0", "DS4_COPY_HUGETLB": "0",
                          "DS4_STORE_SPARE": "0", "DS4_STORE_PUT": "0"},
}
R9700 = {"name": "AMD Radeon AI PRO R9700", "vendor": "amd", "arch": "gfx1201"}
RX7900 = {"name": "AMD Radeon RX 7900 XTX", "vendor": "amd", "arch": "gfx1100"}
STRIX = {"name": "AMD Radeon 8060S", "vendor": "amd", "arch": "gfx1151", "uma": True}
RTX = {"name": "NVIDIA GeForce RTX 4090", "vendor": "nvidia", "arch": "89"}
QWEN = {"arch": "qwen35moe", "backend": "cuda", "args": ["--native", "m.gguf"], "env": {"PATH": "/bin"}}
DEEPSEEK = {"arch": "deepseek41", "exe": "C:/s/engine-dsv41/0.1.52-win/strata-dsv41.exe", "args": ["--pack", "p"], "env": {"DS4_ASYNC_T": "4"}}


def ids(cfg, gpus, backend=None):
    return [s.id for s in W.applicable(cfg, gpus, backend)]


class Contract(unittest.TestCase):
    def test_three_switches_with_the_contract_names(self):
        self.assertEqual([s.id for s in W.SWITCHES], list(OFF))
        for s in W.SWITCHES:
            self.assertEqual(dict(s.off), OFF[s.id], s.id)

    def test_an_effect_text_says_what_turning_it_off_does_in_plain_words(self):
        for s in W.SWITCHES:
            self.assertRegex(s.effect, r"[Tt]urn off", s.id)
            self.assertNotRegex(s.effect, r"STRATA_|DS4_|--|env", s.id)
        self.assertEqual(W.SWITCHES[0].effect, "Reads long prompts faster by approximating part of the work. Turn off if answers about long documents look wrong.")

    def test_where_each_switch_applies(self):
        self.assertEqual(ids(QWEN, [RTX]), ["faster_long_prompts"])
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [RX7900]), ["faster_long_prompts"])
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [STRIX]), ["faster_long_prompts"], "Strix Halo is gfx1151, not gfx12")
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [R9700]), ["faster_long_prompts", "faster_radeon_9000"])
        self.assertEqual(ids({"arch": "qwen35moe"}, [R9700], "hip"), ["faster_long_prompts", "faster_radeon_9000"], "the installed engine's backend when the config has none")
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [{"name": "AMD Radeon RX 9070 XT", "vendor": "amd"}]), ["faster_long_prompts", "faster_radeon_9000"])
        self.assertEqual(ids({**QWEN, "backend": "cuda"}, [RTX, R9700]), ["faster_long_prompts"], "the Radeon switch needs the HIP engine")
        for backend in ("sycl", "metal", "cpu", None):
            self.assertEqual(ids({**QWEN, "backend": backend}, [R9700]), [], backend)
        self.assertEqual(ids(DEEPSEEK, [R9700]), ["deepseek_speedups"])
        self.assertEqual(ids({**DEEPSEEK, "backend": "cuda"}, [RTX]), ["deepseek_speedups"])
        self.assertEqual(ids({"exe": "C:/x/strata-dsv41.exe"}, [RTX]), ["deepseek_speedups"], "found by the engine's name too")

    def test_off_and_on_again_is_the_identical_environment(self):
        env = {"PATH": "/bin", "STRATA_PREFILL": "2", "DS4_ASYNC_T": "4"}
        for s in W.SWITCHES:
            off = W.set_switch(env, s, False)
            self.assertEqual(off, {**env, **OFF[s.id]})
            self.assertTrue(W.is_off(off, s))
            back = W.set_switch(off, s, True)
            self.assertEqual(json.dumps(back), json.dumps(env), s.id)
        self.assertFalse(W.is_off({"DS4_FAST_ATTN": "0"}, "deepseek_speedups"), "only some of the entries reads as on")
        self.assertFalse(W.is_off({"STRATA_KVA": "1"}, "faster_long_prompts"))


class Terminal(unittest.TestCase):
    """`strata launch`: the run config a launch writes."""

    def config(self, base, extra=None):
        return derive_config(base, S.LaunchSettings(extra=extra or {}), bool_flags=frozenset())

    def test_a_default_launch_has_none_of_the_new_entries(self):
        for base in (QWEN, DEEPSEEK):
            env = self.config(base).get("env") or {}
            for name in W.env_names():
                self.assertNotIn(name, env, name)
            self.assertEqual(env, base["env"], "the environment is exactly the installed one")
            self.assertEqual(json.dumps(self.config(base), sort_keys=True), json.dumps(self.config(base, {}), sort_keys=True))

    def test_each_switch_off_by_set_gives_exactly_its_entries(self):
        for sid, entries in OFF.items():
            base = DEEPSEEK if sid == "deepseek_speedups" else QWEN
            cfg = self.config(base, S.settings_from_pairs([f"{k}={v}" for k, v in entries.items()]))
            self.assertEqual(cfg["env"], {**base["env"], **entries}, sid)
            self.assertEqual(cfg["args"], base["args"], "no engine flag changes")

    def test_off_again_puts_the_default_back(self):
        cfg = self.config(QWEN, S.settings_from_pairs(["STRATA_KVA=off"]))
        self.assertEqual(cfg["env"], QWEN["env"])
        already = {**QWEN, "env": {**QWEN["env"], "STRATA_KVA": "0"}}
        self.assertEqual(self.config(already, S.settings_from_pairs(["STRATA_KVA=off"]))["env"], QWEN["env"])

    def test_the_advanced_table_lists_only_the_switches_that_apply(self):
        def names(cfg, gpus, extra=None):
            rows = P.build_rows(None, cfg["args"], None, cfg["env"], None, extra or {}, P.switch_params(W.applicable(cfg, gpus)))
            return [r.name for r in rows if r.name in W.env_names()]
        self.assertEqual(names(QWEN, [RTX]), ["STRATA_KVA"])
        self.assertEqual(names({**QWEN, "backend": "hip"}, [R9700]), ["STRATA_KVA", "STRATA_PCIE_FRAC_DEFAULT"])
        self.assertEqual(names(DEEPSEEK, [R9700]), list(OFF["deepseek_speedups"]), "the six, in the contract order")
        self.assertEqual(names({**QWEN, "backend": "sycl"}, [R9700]), [])

    def test_a_row_says_on_until_it_is_turned_off_and_explains_itself(self):
        rows = P.build_rows(None, QWEN["args"], None, QWEN["env"], None, {}, P.switch_params(W.applicable(QWEN, [RTX])))
        row = next(r for r in rows if r.name == "STRATA_KVA")
        self.assertEqual((row.name, row.default, row.tuned, row.yours, row.known), ("STRATA_KVA", "on", "on", "", True))
        self.assertIn("Faster long prompts", row.description)
        self.assertIn("set 0 to turn off", row.description)
        self.assertIn("Turn off if answers about long documents look wrong", row.description)
        off = P.build_rows(None, QWEN["args"], None, QWEN["env"], None, {"STRATA_KVA": "0"}, P.switch_params(W.applicable(QWEN, [RTX])))
        self.assertEqual(next(r for r in off if r.name == "STRATA_KVA").yours, "0")

    def test_the_six_deepseek_rows_say_they_go_together(self):
        text = W.env_words("DS4_COPY_PIN")
        self.assertIn("all 6 speed-up rows go together", text)
        self.assertEqual({W.env_names()[n].id for n in OFF["deepseek_speedups"]}, {"deepseek_speedups"})


class SameAsTheApp(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_apps_table_has_the_same_names_and_words(self):
        script = ("import {SPEED_SWITCHES} from './apps/host/server/native-settings.mjs';"
                  "console.log(JSON.stringify(SPEED_SWITCHES.map(s=>({id:s.id,label:s.label,hint:s.hint,off:s.off}))))")
        out = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True).stdout
        app = json.loads(out)
        self.assertEqual([a["id"] for a in app], [s.id for s in W.SWITCHES])
        for a, s in zip(app, W.SWITCHES):
            self.assertEqual(a["label"], s.label)
            self.assertEqual(a["hint"], s.effect, s.id)
            self.assertEqual(a["off"], dict(s.off), s.id)


if __name__ == "__main__":
    unittest.main()
