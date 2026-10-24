"""DeepSeek's speed choice (0.2.4.3): Faster (--experts 5, the default on a PC with 16 GB of graphics memory or less) or Full quality (--experts 6)."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.app import LaunchApp
from strata_cli.launch.runner import Runner, apply_experts, derive_config, experts_from_log, used_vram_gb
from strata_cli.launch.screens import SpeedScreen, flow_for
from tests.terminal import test_launch as TL
from tests.terminal.launch_support import Home, text_of
from tests.terminal.test_launch_tui import SIZE, make_ctx, press

run_cli = TL.run_cli


class Words(unittest.TestCase):
    def test_the_default_is_faster_up_to_16_gb_of_graphics_memory(self):
        for gb in (8, 12, 15.9, 16.0):                       # the engine's rule: 16 GiB or less (17,179,869,184 bytes)
            self.assertTrue(S.faster_default(gb), gb)
        for gb in (16.01, 16.4, 20, 24, 48, 0, None):
            self.assertFalse(S.faster_default(gb), gb)

    def test_plain_words_and_the_quality_note(self):
        self.assertEqual(S.speed_words("faster", True), "Faster (default on this PC)")
        self.assertEqual(S.speed_words("faster", False), "Faster")
        self.assertEqual(S.speed_words("full", False), "Full quality: all 6 experts per word")
        self.assertEqual(S.FASTER_NOTE, "Faster uses 5 of 6 experts per word: a small quality change, usually not noticeable.")

    def test_the_flags_text_says_the_same(self):
        self.assertNotIn("--speed", S.flags_text(S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw")))
        self.assertIn("--full-quality", S.flags_text(S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", speed="full")))
        self.assertIn("--speed faster", S.flags_text(S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", speed="faster")))


class EngineFlag(unittest.TestCase):
    def test_experts_replaces_the_saved_value_and_goes_last(self):
        cfg = {"args": ["--pack", "p", "--experts", "6", "--dspark", "auto"]}
        apply_experts(cfg, 5)
        self.assertEqual(cfg["args"], ["--pack", "p", "--dspark", "auto", "--experts", "5"])
        cfg = {"args": ["--experts=5", "--pack", "p"]}
        apply_experts(cfg, 6)
        self.assertEqual(cfg["args"], ["--pack", "p", "--experts", "6"])
        cfg = {"args": ["--pack", "p"]}
        apply_experts(cfg, None)
        self.assertEqual(cfg["args"], ["--pack", "p"])

    def test_derive_config_passes_it(self):
        base = {"exe": "x", "args": ["--pack", "p"]}
        cfg = derive_config(base, S.LaunchSettings(), bool_flags=frozenset(), experts=5)
        self.assertEqual(cfg["args"][-2:], ["--experts", "5"])
        self.assertEqual(derive_config(base, S.LaunchSettings(), bool_flags=frozenset())["args"], ["--pack", "p"])


class Resolve(unittest.TestCase):
    def runner(self, *cards, **kw):
        cat = mock.Mock()
        cat.strata.multi_family.return_value = True
        gpus = [{"index": i, "vram_gb": gb} for i, gb in enumerate(cards)]
        cat.computer.return_value = mock.Mock(usable_gpus=gpus)
        return Runner(S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", **kw), catalog=cat, emit=lambda e: None)

    def test_a_12_gb_pc_gets_faster_and_a_big_one_full_quality_in_the_words(self):
        self.assertEqual(self.runner(12).resolved_speed(), ("faster", True))
        self.assertEqual(self.runner(96).resolved_speed(), ("full", True))
        self.assertEqual(self.runner(8, 8).resolved_speed(), ("faster", True))         # summed over the cards, as the engine sums them
        self.assertEqual(self.runner(12, 12).resolved_speed(), ("full", True))

    def test_nothing_is_passed_to_the_engine_unless_the_person_chose(self):
        for cards in ((12,), (96,), (8, 8)):
            self.assertIsNone(self.runner(*cards).experts_flag(), cards)                # the engine's --experts auto decides: one rule
        self.assertEqual(self.runner(12, speed="full").experts_flag(), 6)
        self.assertEqual(self.runner(96, speed="faster").experts_flag(), 5)

    def test_a_choice_wins_over_the_default_in_the_words(self):
        self.assertEqual(self.runner(12, speed="full").resolved_speed(), ("full", False))

    def test_only_the_cards_the_launch_uses_are_summed(self):
        comp = mock.Mock(usable_gpus=[{"index": 0, "vram_gb": 12}, {"index": 1, "vram_gb": 24}])
        self.assertEqual(used_vram_gb(comp, S.LaunchSettings()), 36)
        self.assertEqual(used_vram_gb(comp, S.LaunchSettings(gpus=[0])), 12)
        self.assertEqual(used_vram_gb(comp, S.LaunchSettings(gpu_roles={"0": "main", "1": "off"})), 12)

    def test_the_engines_own_line_wins_in_the_summary(self):
        r = self.runner(12)
        self.assertIn("Faster (default on this PC)", r.speed_line(5))
        self.assertIn("Full quality (default on this PC)", r.speed_line(6))             # the engine said 6 although the rule here said Faster
        self.assertIn("Full quality: all 6 experts per word", self.runner(12, speed="full").speed_line(5))   # a choice stays what was chosen

    def test_the_number_is_read_back_from_the_engine_log(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "e.log"
            log.write_text("ds4: loading\nds4: experts per word: 6\nlater\nds4: experts per word: 5\n", encoding="utf-8")
            self.assertEqual(experts_from_log(log), 5)
            log.write_text("nothing here\n", encoding="utf-8")
            self.assertIsNone(experts_from_log(log))
            self.assertIsNone(experts_from_log(Path(d) / "missing.log"))

    def test_the_line_has_the_note_only_for_faster(self):
        self.assertIn(S.FASTER_NOTE, self.runner(12).speed_line())
        self.assertIn("Faster (default on this PC)", self.runner(12).speed_line())
        self.assertEqual(self.runner(12, speed="full").speed_line(), "Speed: Full quality: all 6 experts per word")

    def test_another_model_has_no_speed(self):
        cat = mock.Mock()
        cat.strata.multi_family.return_value = False
        r = Runner(S.LaunchSettings(family="qwen", model="IQ3_S"), catalog=cat, emit=lambda e: None)
        self.assertIsNone(r.resolved_speed())
        self.assertIsNone(r.experts_flag())
        self.assertEqual(r.speed_line(), "")


class Flags(unittest.TestCase):
    def plan(self, *args):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "ist-das-2.00bpw", "--family", "deepseek41", *args])
        self.assertEqual(code, 0, err + out)
        return json.loads(out)["plan"]

    def test_plan_names_the_speed_in_plain_words(self):
        with mock.patch.object(Runner, "resolved_speed", return_value=("faster", True)):
            p = self.plan()
        self.assertIn("Faster (default on this PC)", p["speed"])
        self.assertIn(S.FASTER_NOTE, p["speed"])

    def test_full_quality_flag_and_speed_flag(self):
        self.assertEqual(self.plan("--full-quality")["settings"]["speed"], "full")
        self.assertEqual(self.plan("--speed", "full")["settings"]["speed"], "full")
        self.assertEqual(self.plan("--speed", "faster")["settings"]["speed"], "faster")
        self.assertIn("--full-quality", self.plan("--full-quality")["flags"])

    def test_both_flags_say_so_and_full_quality_wins(self):
        p = self.plan("--full-quality", "--speed", "faster")
        self.assertEqual(p["settings"]["speed"], "full")
        self.assertTrue(any("full quality is used" in w for w in p["warnings"]))

    def test_another_model_warns_and_drops_it(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "q2_0", "--full-quality"])
        p = json.loads(out)["plan"]
        self.assertIsNone(p["settings"]["speed"])
        self.assertTrue(any("for DeepSeek" in w for w in p["warnings"]))

    def test_human_plan_prints_a_speed_line(self):
        with mock.patch.object(Runner, "resolved_speed", return_value=("faster", True)):
            code, out, err = run_cli(["launch", "--plan", "--model", "ist-das-2.00bpw", "--family", "deepseek41"])
        self.assertIn("  speed     Faster (default on this PC)", out)


class Screen(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_the_step_exists_only_for_deepseek(self):
        ctx = make_ctx()
        self.assertNotIn("speed", flow_for(ctx))
        ctx.settings.family = "deepseek41"
        flow = flow_for(ctx)
        self.assertEqual(flow[flow.index("context") + 1], "speed")

    def test_the_screen_offers_two_choices_and_remembers_only_a_change(self):
        ctx = make_ctx()
        ctx.settings.family, ctx.settings.model = "deepseek41", "IST-DAS-2.00bpw"
        faster_here = S.faster_default(used_vram_gb(ctx.computer, ctx.settings))
        app = LaunchApp(ctx)

        async def go():
            async with app.run_test(size=SIZE) as pilot:
                await pilot.pause(0.2)
                app.goto("speed")
                await pilot.pause(0.3)
                t = text_of(app)
                self.assertIn("Faster", t)
                self.assertIn("Full quality", t)
                self.assertIn(S.FASTER_NOTE, t)
                screen = app.screen
                self.assertIsInstance(screen, SpeedScreen)
                screen.commit()
                self.assertEqual(ctx.settings.speed, None if faster_here else "faster")
                await press(pilot, "down")
                screen.commit()
                self.assertIn(ctx.settings.speed, (None, "full"))
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
