# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""I6 (tools and images review, 2026-10-22): a picture a tool returned, with an elastic image encoder that is not
running, used to write to a process that did not exist (a 400 "malformed request" that every later turn of the
conversation repeated).  It starts the encoder like a picture of the user does; when it cannot start, the picture
becomes an "[image omitted: why]" note.

    python -m serve.test_vision_tool_image
"""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, Vision
from serve.test_lifecycle import ResidentEngine

ROOT = Path(__file__).resolve().parents[1]


class ElasticVision:
    """An encoder process stand-in: starts, stops, and turns every picture into three image tokens."""
    def __init__(self, d, elastic=True):
        self.elastic, self.elastic_mib, self.elastic_idle_s = elastic, 2048, 60.0
        self.up, self.starts, self.unloads = False, 0, 0
        self.dir = Path(d)
        self.rows = self.dir / "img.sve"
        self.rows.write_bytes(b"rows")

    def alive(self):
        return self.up

    def restart(self):
        self.up, self.starts = True, self.starts + 1

    def unload(self):
        self.up, self.unloads = False, self.unloads + 1

    def encode(self, source):
        assert self.up, "a picture reached an encoder that was not started"
        return self.rows, 3


class VramEngine(ResidentEngine):
    """The resident test engine with an elastic cache: VRAM n frees what the reserve asks for."""
    def vram(self, reserve_mib, timeout=120.0):
        r = 700 if reserve_mib is None else reserve_mib
        return {"reserve_mib": r, "expert_cache_mib": max(500, 6000 - r), "expert_cache_full_mib": 6000}


def tool_turn():
    return [{"role": "user", "content": "look at it"},
            {"role": "tool", "tool_call_id": "t1", "content": [{"type": "image", "source": "x.png"}]}]


class FailingStart(ElasticVision):
    def restart(self):
        raise RuntimeError("the vision encoder did not start: it ended without saying anything (exit code 1)")


class ToolImage(unittest.TestCase):
    def service(self, vision):
        tok = ByteTokenizer()
        engine = VramEngine(tok)
        engine.loaded = True
        return Service(engine, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"), vision=vision)

    def test_an_elastic_encoder_is_started_for_a_tool_picture(self):
        vision = ElasticVision(tempfile.mkdtemp())
        svc = self.service(vision)
        messages = tool_turn()
        svc._note_unreadable_tool_images(messages)
        self.assertEqual(vision.starts, 1)
        self.assertEqual(messages[1]["content"][0]["type"], "image")      # kept: it was read
        svc._note_unreadable_tool_images(messages)                        # the running encoder is used as it is
        self.assertEqual(vision.starts, 1)

    def test_an_encoder_that_cannot_start_leaves_a_note_not_a_400(self):
        svc = self.service(FailingStart(tempfile.mkdtemp()))
        messages = tool_turn()
        svc._note_unreadable_tool_images(messages)                        # no exception
        part = messages[1]["content"][0]
        self.assertEqual(part["type"], "text")
        self.assertIn("[image omitted:", part["text"])
        self.assertIn("did not start", part["text"])

    def test_encode_without_a_running_reader_is_a_clear_value_error(self):
        v = Vision.__new__(Vision)
        v.dir, v.cache, v.proc, v.dead = Path(tempfile.mkdtemp()), {}, None, None
        v.lock = threading.Lock()
        with mock.patch.object(Vision, "normalize", return_value=b"png"), self.assertRaises(ValueError) as cm:
            v.encode(b"bytes")
        self.assertIn("picture reader is not running", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
