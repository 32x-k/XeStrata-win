# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The auto policy of an image job: streamed layers beside the text engine only when the RAM holds the image model."""
import os
import tempfile
import unittest
from unittest import mock

from serve import images


class Engine:
    def alive(self):
        return True

    def unload(self):
        pass


class Policy(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        cfg = {}
        for k, mib in (("diffusion_model", 4), ("vae", 1), ("llm", 5)):
            path = os.path.join(self.dir.name, k)
            with open(path, "wb") as f:
                f.truncate(mib << 20)
            cfg[k] = path
        self.gen = mock.Mock(cfg=cfg, policy="auto", reserve=512)
        self.svc = mock.Mock(engine=Engine())
        self.streamed = {"free_mib": 4000, "held_mib": 0, "full_mib": 10000, "least_mib": 2300}

    def policy(self, available, need=None):
        with mock.patch("serve.session_save_retry.available_mib", return_value=available):
            return images.resolve_policy(self.svc, self.gen, need or self.streamed)

    def test_streamed_beside_the_engine_when_the_ram_holds_the_files(self):
        self.assertEqual(self.policy(10 + images.IMAGE_HOST_MIB), "keep")
        self.gen._say.assert_not_called()

    def test_unloaded_when_the_ram_does_not_hold_them(self):
        self.assertEqual(self.policy(9 + images.IMAGE_HOST_MIB), "unload")
        self.assertIn("unload rather than stream", self.gen._say.call_args[0][0])

    def test_the_ram_unknown_keeps_the_former_choice(self):
        self.assertEqual(self.policy(None), "keep")

    def test_full_speed_beside_the_engine_does_not_look_at_the_ram(self):
        need = dict(self.streamed, free_mib=12000)
        self.assertEqual(self.policy(0, need), "keep")

    def test_no_room_at_all_unloads(self):
        need = dict(self.streamed, free_mib=1000)
        self.assertEqual(self.policy(1 << 20, need), "unload")


class PathWords(unittest.TestCase):
    def test_a_path_is_one_word_and_comes_back_as_utf8(self):
        path = "C:\\Users\\Taro Yamada\\AppData\\Local\\Temp\\画像 100%\\img_0.rgba"
        word = images.path_word(path)
        self.assertNotIn(" ", word)
        self.assertTrue(word.isascii())
        back = bytearray()
        i = 0
        while i < len(word):   # what strata-image's path_of does
            if word[i] == "%":
                back.append(int(word[i + 1:i + 3], 16))
                i += 3
            else:
                back.append(ord(word[i]))
                i += 1
        self.assertEqual(back.decode("utf-8"), path)

    def test_a_plain_path_stays_as_it_is(self):
        self.assertEqual(images.path_word("/tmp/strata-images-x/job-1/img_0.rgba"),
                         "/tmp/strata-images-x/job-1/img_0.rgba")


class ParallelTurn(unittest.TestCase):
    """--parallel: a chat registered as running but still waiting for the control lines does not hold the image."""

    def run_turn(self, live, waiting):
        import contextlib
        import threading
        engine = mock.Mock(batch=2, ctl=threading.Lock(), waiting=waiting)
        engine.alive.return_value = True
        svc = mock.Mock(engine=engine, live_reqs={i: None for i in range(live)}, status={"queued": 0},
                        status_lock=threading.Lock())
        svc._turn.return_value = contextlib.nullcontext()
        gen = mock.Mock(policy="keep", reserve=512, spare=1024)
        gen.need.return_value = {"free_mib": 30000, "held_mib": 0, "full_mib": 10000, "least_mib": 2000}
        gen.fit.return_value = 20000
        gen.free_mib.return_value = 30000
        cancel = threading.Event()
        threading.Timer(2.0, cancel.set).start()   # a turn that would wait for ever is ended by the client leaving
        try:
            with images.gpu_turn(svc, gen, {}, cancel) as policy:
                return policy
        except RuntimeError:   # the client gone (ClientGone): it waited
            return None
        finally:
            cancel.set()

    def test_requests_waiting_for_the_lines_do_not_hold_the_image(self):
        self.assertEqual(self.run_turn(live=2, waiting=2), "keep")

    def test_a_request_holding_a_slot_is_waited_for(self):
        self.assertIsNone(self.run_turn(live=2, waiting=1))


if __name__ == "__main__":
    unittest.main()
