# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/test_sdk_openai_chat.py - OpenAI's Python SDK reads what /v1/chat/completions answers.

The server runs on the mock engine (its reply is a script), so what is checked is the API's shape as the official
client parses it, not a model's words.  Skipped without the `openai` package (it is a developer's tool, not one of
XeStrata's dependencies).

    python -m unittest serve.test_sdk_openai_chat -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

try:
    import openai
except ImportError:
    openai = None

ROOT = Path(__file__).resolve().parents[1]
ANSWER = "</think>\n\nHello there."
CALLS = ("</think>\n\n<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
         "<tool_call>\n<function=b>\n<parameter=x>\n2\n</parameter>\n</function>\n</tool_call>")
TOOLS = [{"type": "function", "function": {"name": n, "parameters": {
    "type": "object", "properties": {"x": {"type": "string"}}}}} for n in ("a", "b")]


@unittest.skipUnless(openai is not None, "the openai package is not installed")
class OpenAIChat(unittest.TestCase):
    def client(self, script):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=8192), tok,
                      ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return openai.OpenAI(base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1", api_key="none",
                             max_retries=0)

    def test_text_with_and_without_a_stream(self):
        c = self.client(ANSWER)
        r = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=64)
        self.assertEqual(r.choices[0].message.content, "Hello there.")
        self.assertEqual(r.choices[0].finish_reason, "stop")
        self.assertTrue(r.system_fingerprint)
        self.assertGreater(r.usage.completion_tokens, 0)
        chunks = list(c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}],
                                                max_tokens=64, stream=True, stream_options={"include_usage": True}))
        self.assertEqual("".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices), "Hello there.")
        self.assertEqual(chunks[-1].choices, [])                  # the usage chunk
        self.assertEqual(chunks[-1].usage.completion_tokens, r.usage.completion_tokens)

    def test_a_forced_call_and_one_call_at_most(self):
        c = self.client(CALLS)
        r = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "go"}], tools=TOOLS,
                                      max_tokens=300, tool_choice="required", parallel_tool_calls=False)
        self.assertEqual(r.choices[0].finish_reason, "tool_calls")
        self.assertEqual([t.function.name for t in r.choices[0].message.tool_calls], ["a"])

    def test_n_choices(self):
        r = self.client(ANSWER).chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}],
                                                        max_tokens=64, n=2)
        self.assertEqual([(ch.index, ch.message.content) for ch in r.choices], [(0, "Hello there."),
                                                                                (1, "Hello there.")])
        # streamed, every chunk of the one completion has its id (OpenAI's ChatCompletionChunk.id)
        chunks = list(self.client(ANSWER).chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=64, n=2, stream=True))
        self.assertEqual({ch.choices[0].index for ch in chunks if ch.choices}, {0, 1})
        self.assertEqual(len({ch.id for ch in chunks}), 1)

    def test_structured_output(self):
        r = self.client('</think>\n\n{"a": 1}').chat.completions.create(
            model="m", messages=[{"role": "user", "content": "json"}], max_tokens=64,
            response_format={"type": "json_schema", "json_schema": {"name": "o", "schema": {
                "type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}})
        self.assertEqual(r.choices[0].message.content, '{"a":1}')

    def test_models_and_an_error(self):
        c = self.client(ANSWER)
        ids = [m.id for m in c.models.list()]
        self.assertEqual(c.models.retrieve(ids[0]).id, ids[0])
        with self.assertRaises(openai.NotFoundError):
            c.models.retrieve("no-such-model")
        with self.assertRaises(openai.BadRequestError) as cm:
            c.chat.completions.create(model="m", messages=[{"role": "user", "content": "go"}], tools=TOOLS,
                                      tool_choice={"type": "function", "function": {"name": "zz"}})
        self.assertEqual(cm.exception.body.get("type"), "invalid_request_error")


if __name__ == "__main__":
    unittest.main()
