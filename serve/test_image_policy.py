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


if __name__ == "__main__":
    unittest.main()
