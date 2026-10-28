# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/test_sdk_anthropic.py - Anthropic's Python SDK reads what /v1/messages answers.

The server runs on the mock engine, so what is checked is the shape the official client parses.  Skipped without the
`anthropic` package (a developer's tool, not one of XeStrata's dependencies).

    python -m unittest serve.test_sdk_anthropic -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

try:
    import anthropic
except ImportError:
    anthropic = None

ROOT = Path(__file__).resolve().parents[1]
THOUGHT = "Thinking it over</think>\n\nHello there."
# the requests below that ask for no thinking are read without it (the server's default): their scripts open no
# thinking to close
CALLS = ("<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
         "<tool_call>\n<function=b>\n<parameter=x>\n2\n</parameter>\n</function>\n</tool_call>")
TOOLS = [{"name": n, "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}} for n in ("a", "b")]


@unittest.skipUnless(anthropic is not None, "the anthropic package is not installed")
class AnthropicMessages(unittest.TestCase):
    def client(self, script):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=8192), tok,
                      ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return anthropic.Anthropic(base_url=f"http://127.0.0.1:{httpd.server_address[1]}", api_key="none",
                                   max_retries=0)

    def test_signed_thinking_goes_back(self):
        c = self.client(THOUGHT)
        user = [{"role": "user", "content": "hi"}]
        m = c.messages.create(model="m", max_tokens=64, messages=user, thinking={"type": "adaptive"})
        self.assertEqual([b.type for b in m.content], ["thinking", "text"])
        self.assertTrue(m.content[0].signature)
        with c.messages.stream(model="m", max_tokens=64, messages=user, thinking={"type": "adaptive"}) as s:
            streamed = s.get_final_message()
        self.assertEqual(streamed.content[0].signature, m.content[0].signature)
        again = c.messages.create(model="m", max_tokens=64, thinking={"type": "adaptive"}, messages=user + [
            {"role": "assistant", "content": [b.model_dump() for b in m.content]},
            {"role": "user", "content": "and?"}])
        self.assertEqual(again.stop_reason, "end_turn")

    def test_one_call_and_its_result(self):
        c = self.client([CALLS, "done"])
        m = c.messages.create(model="m", max_tokens=300, messages=[{"role": "user", "content": "go"}], tools=TOOLS,
                              tool_choice={"type": "any", "disable_parallel_tool_use": True})
        self.assertEqual(m.stop_reason, "tool_use")
        uses = [b for b in m.content if b.type == "tool_use"]
        self.assertEqual([(u.name, u.id[:6]) for u in uses], [("a", "toolu_")])
        done = c.messages.create(model="m", max_tokens=64, tools=TOOLS, messages=[
            {"role": "user", "content": "go"}, {"role": "assistant", "content": [u.model_dump() for u in uses]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0].id, "content": "1"}]}])
        self.assertEqual(done.content[-1].text, "done")

    def test_structured_output_and_count_tokens(self):
        c = self.client('{"a": 1}')
        user = [{"role": "user", "content": "json"}]
        m = c.messages.create(model="m", max_tokens=64, messages=user, extra_body={"output_config": {"format": {
            "type": "json_schema", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}}}}})
        self.assertEqual(m.content[-1].text, '{"a":1}')
        self.assertGreater(c.messages.count_tokens(model="m", messages=user).input_tokens, 0)

    def test_errors(self):
        c = self.client("ok")
        with self.assertRaises(anthropic.BadRequestError) as cm:
            c.messages.create(model="m", max_tokens=64, messages=[{"role": "user", "content": "go"}], tools=TOOLS,
                              tool_choice={"type": "tool", "name": "zz"})
        self.assertEqual(cm.exception.body["error"]["type"], "invalid_request_error")


if __name__ == "__main__":
    unittest.main()
