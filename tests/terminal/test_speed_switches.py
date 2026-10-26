"""The 0.2.7 speed features and their switches: the contract names, the defaults (Faster long prompts OFF, the other two ON), where each applies, and how the Terminal carries them.

The app (apps/host/server/native-settings.mjs SPEED_SWITCHES) and Control (strata_control/speed_switches.py) say the same thing; the last test reads the
app's table with node and fails when the two differ.  `strata launch` shows the environment entries as rows of its Advanced table and writes them into
the run config's "env" (`--set STRATA_KVA=1` turns Faster long prompts on); a default launch has none of them.
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

ENV = {      # what each switch writes outside its default
    "faster_long_prompts": {"STRATA_KVA": "1"},
    "faster_radeon_9000": {"STRATA_PCIE_FRAC_DEFAULT": "old"},
    "deepseek_speedups": {"DS4_FAST_ATTN": "0", "DS4_ATTN_SLAB": "0", "DS4_COPY_PIN": "0", "DS4_COPY_HUGETLB": "0",
                          "DS4_STORE_SPARE": "0", "DS4_STORE_PUT": "0"},
}
DEFAULT = {"faster_long_prompts": "off", "faster_radeon_9000": "on", "deepseek_speedups": "on"}
R9700 = {"name": "AMD Radeon AI PRO R9700", "vendor": "amd", "arch": "gfx1201"}
RX7900 = {"name": "AMD Radeon RX 7900 XTX", "vendor": "amd", "arch": "gfx1100"}
STRIX = {"name": "AMD Radeon 8060S", "vendor": "amd", "arch": "gfx1151", "uma": True}
RTX = {"name": "NVIDIA GeForce RTX 4090", "vendor": "nvidia", "arch": "89"}
# A Qwen3.8-Flash-Next run config (the original, Swift, Coder, Unsloth, Gyro, NVFP4): setup writes no "arch" for it.  Qwen3.6-35B-A3B has an engine path of its own: "arch".
QWEN = {"model_name": "qwen3.8-flash-next-iq3_s", "backend": "cuda", "args": ["--native", "m.gguf"], "env": {"PATH": "/bin"}}
QWEN36 = {"arch": "qwen35moe", "model_name": "qwen3.6-35b-a3b-ud-q4_k_xl", "backend": "cuda", "args": ["--native", "m.gguf"], "env": {"PATH": "/bin"}}
DEEPSEEK = {"arch": "deepseek41", "exe": "C:/s/engine-dsv41/0.1.52-win/strata-dsv41.exe", "args": ["--pack", "p"], "env": {"DS4_ASYNC_T": "4"}}


def ids(cfg, gpus, backend=None):
    return [s.id for s in W.applicable(cfg, gpus, backend)]


class Contract(unittest.TestCase):
    def test_three_switches_with_the_contract_names_and_their_defaults(self):
        self.assertEqual([s.id for s in W.SWITCHES], list(ENV))
        for s in W.SWITCHES:
            self.assertEqual(dict(s.env), ENV[s.id], s.id)
            self.assertEqual(s.default, DEFAULT[s.id], s.id)

    def test_an_effect_text_says_what_the_feature_does_in_plain_words(self):
        for s in W.SWITCHES:
            self.assertRegex(s.effect, r"Off by default" if s.default == "off" else r"[Tt]urn off", s.id)
            self.assertNotRegex(s.effect, r"STRATA_|DS4_|--|env", s.id)
        self.assertEqual(W.SWITCHES[0].effect, "Reads very long prompts faster by approximating part of the work. Off by default: answers about long documents can miss or mix up details.")

    def test_where_each_switch_applies(self):
        self.assertEqual(ids(QWEN, [RTX]), ["faster_long_prompts"])
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [RX7900]), ["faster_long_prompts"])
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [STRIX]), ["faster_long_prompts"], "Strix Halo is gfx1151, not gfx12")
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [R9700]), ["faster_long_prompts", "faster_radeon_9000"])
        self.assertEqual(ids({"model_name": "qwen3.8-flash-next-iq3_s"}, [R9700], "hip"), ["faster_long_prompts", "faster_radeon_9000"], "the installed engine's backend when the config has none")
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [{"name": "AMD Radeon RX 9070 XT", "vendor": "amd"}]), ["faster_long_prompts", "faster_radeon_9000"])
        self.assertEqual(ids({**QWEN, "backend": "cuda"}, [RTX, R9700]), ["faster_long_prompts"], "the Radeon switch needs the HIP engine")
        for backend in ("sycl", "metal", "cpu", None):
            self.assertEqual(ids({**QWEN, "backend": backend}, [R9700]), [], backend)
        self.assertEqual(ids(DEEPSEEK, [R9700]), ["deepseek_speedups"])
        self.assertEqual(ids({**DEEPSEEK, "backend": "cuda"}, [RTX]), ["deepseek_speedups"])
        self.assertEqual(ids({"exe": "C:/x/strata-dsv41.exe"}, [RTX]), ["deepseek_speedups"], "found by the engine's name too")

    def test_faster_long_prompts_is_for_qwen38_flash_next_only(self):
        """KVA was measured on the 48-layer / 512-expert Flash-Next shape and the engine declines other shapes: no row for Qwen3.6-35B-A3B (arch qwen35moe) or DeepSeek."""
        self.assertEqual(ids(QWEN36, [RTX]), [], "Qwen3.6-35B-A3B on CUDA: no KVA row")
        self.assertEqual(ids({**QWEN36, "backend": "hip"}, [RX7900]), [], "on HIP too")
        self.assertEqual(ids({**QWEN36, "backend": "hip"}, [R9700]), ["faster_radeon_9000"], "the Radeon switch is not tied to the model's shape")
        for family in ("qwen3.8-flash-next-iq3_s", "swift-1.5-iq3_s", "qwen3.8-flash-next-coder-iq3_s", "qwen3.8-flash-next-unsloth-ud-q4_k_xl",
                       "qwen3.8-flash-next-gyro-gyro-s", "qwen3.8-flash-next-nvfp4-nvfp4"):
            self.assertEqual(ids({**QWEN, "model_name": family}, [RTX]), ["faster_long_prompts"], family)
        self.assertEqual(ids(DEEPSEEK, [RTX]), ["deepseek_speedups"])

    def test_outside_the_default_and_back_is_the_identical_environment(self):
        env = {"PATH": "/bin", "STRATA_PREFILL": "2", "DS4_ASYNC_T": "4"}
        for s in W.SWITCHES:
            other = W.set_switch(env, s, s.default != "on")           # KVA on, the other two off
            self.assertEqual(other, {**env, **ENV[s.id]})
            self.assertTrue(W.is_set(other, s))
            self.assertEqual(W.is_on(other, s), s.default != "on")
            back = W.set_switch(other, s, s.default == "on")
            self.assertEqual(json.dumps(back), json.dumps(env), s.id)

    def test_kva_is_off_by_default_and_an_earlier_builds_zero_shows_as_off(self):
        self.assertFalse(W.is_on({}, "faster_long_prompts"))
        self.assertTrue(W.is_on({"STRATA_KVA": "1"}, "faster_long_prompts"))
        self.assertFalse(W.is_on({"STRATA_KVA": "0"}, "faster_long_prompts"), "stored by the old off switch: harmless, and Off is what it is now")
        self.assertFalse(W.is_on({"STRATA_KVA": "yes"}, "faster_long_prompts"), "only 1 turns it on")
        self.assertEqual(W.set_switch({"PATH": "/bin", "STRATA_KVA": "0"}, "faster_long_prompts", False), {"PATH": "/bin"}, "choosing Off cleans it up")
        self.assertTrue(W.is_on({}, "deepseek_speedups"))
        self.assertTrue(W.is_on({"DS4_FAST_ATTN": "0"}, "deepseek_speedups"), "only some of the entries reads as the default (on)")


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

    def test_each_switch_outside_its_default_by_set_gives_exactly_its_entries(self):
        for sid, entries in ENV.items():          # --set STRATA_KVA=1 turns KVA on; the other two are turned off by their entries
            base = DEEPSEEK if sid == "deepseek_speedups" else QWEN
            cfg = self.config(base, S.settings_from_pairs([f"{k}={v}" for k, v in entries.items()]))
            self.assertEqual(cfg["env"], {**base["env"], **entries}, sid)
            self.assertEqual(cfg["args"], base["args"], "no engine flag changes")

    def test_off_again_puts_the_default_back(self):
        cfg = self.config(QWEN, S.settings_from_pairs(["STRATA_KVA=off"]))
        self.assertEqual(cfg["env"], QWEN["env"])
        turned_on = {**QWEN, "env": {**QWEN["env"], "STRATA_KVA": "1"}}
        self.assertEqual(self.config(turned_on, S.settings_from_pairs(["STRATA_KVA=off"]))["env"], QWEN["env"])
        old = {**QWEN, "env": {**QWEN["env"], "STRATA_KVA": "0"}}
        self.assertEqual(self.config(old, S.settings_from_pairs(["STRATA_KVA=off"]))["env"], QWEN["env"])

    def test_the_advanced_table_lists_only_the_switches_that_apply(self):
        def names(cfg, gpus, extra=None):
            rows = P.build_rows(None, cfg["args"], None, cfg["env"], None, extra or {}, P.switch_params(W.applicable(cfg, gpus)))
            return [r.name for r in rows if r.name in W.env_names()]
        self.assertEqual(names(QWEN, [RTX]), ["STRATA_KVA"])
        self.assertEqual(names({**QWEN, "backend": "hip"}, [R9700]), ["STRATA_KVA", "STRATA_PCIE_FRAC_DEFAULT"])
        self.assertEqual(names(DEEPSEEK, [R9700]), list(ENV["deepseek_speedups"]), "the six, in the contract order")
        self.assertEqual(names({**QWEN, "backend": "sycl"}, [R9700]), [])

    def test_a_row_shows_its_default_until_it_is_changed_and_explains_itself(self):
        rows = P.build_rows(None, QWEN["args"], None, QWEN["env"], None, {}, P.switch_params(W.applicable(QWEN, [RTX])))
        row = next(r for r in rows if r.name == "STRATA_KVA")
        self.assertEqual((row.name, row.default, row.tuned, row.yours, row.known), ("STRATA_KVA", "off", "off", "", True))
        self.assertIn("Faster long prompts", row.description)
        self.assertIn("off by default; set 1 to turn on", row.description)
        self.assertIn("answers about long documents can miss or mix up details", row.description)
        on = P.build_rows(None, QWEN["args"], None, QWEN["env"], None, {"STRATA_KVA": "1"}, P.switch_params(W.applicable(QWEN, [RTX])))
        self.assertEqual(next(r for r in on if r.name == "STRATA_KVA").yours, "1")
        hip = {**QWEN, "backend": "hip"}
        radeon = next(r for r in P.build_rows(None, hip["args"], None, hip["env"], None, {}, P.switch_params(W.applicable(hip, [R9700]))) if r.name == "STRATA_PCIE_FRAC_DEFAULT")
        self.assertEqual(radeon.default, "on")
        self.assertIn("on by default; set old to turn off", radeon.description)

    def test_the_six_deepseek_rows_say_they_go_together(self):
        text = W.env_words("DS4_COPY_PIN")
        self.assertIn("all 6 speed-up rows go together", text)
        self.assertEqual({W.env_names()[n].id for n in ENV["deepseek_speedups"]}, {"deepseek_speedups"})


class SameAsTheApp(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_apps_table_has_the_same_names_and_words(self):
        script = ("import {SPEED_SWITCHES} from './apps/host/server/native-settings.mjs';"
                  "console.log(JSON.stringify(SPEED_SWITCHES.map(s=>({id:s.id,label:s.label,hint:s.hint,default:s.default,env:s.env}))))")
        out = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True).stdout
        app = json.loads(out)
        self.assertEqual([a["id"] for a in app], [s.id for s in W.SWITCHES])
        for a, s in zip(app, W.SWITCHES):
            self.assertEqual(a["label"], s.label)
            self.assertEqual(a["hint"], s.effect, s.id)
            self.assertEqual(a["env"], dict(s.env), s.id)
            self.assertEqual(a["default"], s.default, s.id)


if __name__ == "__main__":
    unittest.main()
