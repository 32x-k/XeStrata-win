# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/test_sdk_openai_responses.py - OpenAI's Python SDK reads what /v1/responses answers.

The server runs on the mock engine, so what is checked is the shape the official client parses.  Skipped without the
`openai` package (a developer's tool, not one of XeStrata's dependencies).

    python -m unittest serve.test_sdk_openai_responses -v
"""
from __future__ import annotations

import base64
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
THOUGHT = "Thinking it over</think>\n\nHello there."
CALLS = ("</think>\n\n<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
         "<tool_call>\n<function=b>\n<parameter=x>\n2\n</parameter>\n</function>\n</tool_call>")
TOOLS = [{"type": "function", "name": n, "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}
         for n in ("a", "b")]


@unittest.skipUnless(openai is not None, "the openai package is not installed")
class OpenAIResponses(unittest.TestCase):
    def client(self, script):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=8192), tok,
                      ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return openai.OpenAI(base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1", api_key="none",
                             max_retries=0)

    def test_text_and_the_reasoning_summary(self):
        c = self.client(THOUGHT)
        r = c.responses.create(model="m", input="hi", max_output_tokens=64, reasoning={"summary": "auto"})
        self.assertEqual(r.output_text, "Hello there.")
        reasoning = [o for o in r.output if o.type == "reasoning"][0]
        self.assertEqual(reasoning.summary[0].text, "Thinking it over")
        events = list(c.responses.create(model="m", input="hi", max_output_tokens=64, stream=True,
                                         reasoning={"summary": "auto"}))
        kinds = [e.type for e in events]
        self.assertEqual(kinds[0], "response.created")
        self.assertEqual(kinds[-1], "response.completed")
        self.assertIn("response.reasoning_summary_text.delta", kinds)
        self.assertEqual(events[-1].response.output_text, "Hello there.")

    def test_a_forced_call_and_one_call_at_most(self):
        r = self.client(CALLS).responses.create(model="m", input="go", tools=TOOLS, max_output_tokens=300,
                                                tool_choice={"type": "function", "name": "a"},
                                                parallel_tool_calls=False)
        self.assertEqual([o.name for o in r.output if o.type == "function_call"], ["a"])

    def test_history_with_hosted_items_and_a_file(self):
        c = self.client("</think>\n\nok")
        note = base64.b64encode(b"NOTE-BODY").decode()
        r = c.responses.create(model="m", max_output_tokens=16, input=[
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "read this"},
                {"type": "input_file", "filename": "n.txt", "file_data": "data:text/plain;base64," + note}]},
            {"type": "web_search_call", "id": "ws_1", "status": "completed",
             "action": {"type": "search", "query": "q"}},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "found"}]},
            {"type": "message", "role": "user", "content": "go on"}])
        self.assertEqual(r.output_text, "ok")

    def test_errors(self):
        c = self.client("</think>\n\nok")
        with self.assertRaises(openai.BadRequestError) as cm:
            c.responses.create(model="m", input="go", tools=TOOLS, tool_choice={"type": "function", "name": "zz"})
        self.assertEqual(cm.exception.body.get("param"), "tool_choice")
        with self.assertRaises(openai.NotFoundError):
            c.responses.delete("resp_1")


if __name__ == "__main__":
    unittest.main()
