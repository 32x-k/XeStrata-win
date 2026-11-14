# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parameter text restoration: conservative schema hints, root references and identical streaming values."""
import copy
import json
from pathlib import Path
import unittest
import urllib.error
import urllib.request

from serve.frontend import ChatTemplate, OutputParser, _schema_types, param_value, parse_tool_call
from serve.server import ByteTokenizer, MockEngine, Service, serve

try:
    import jsonschema
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False


def parameters(prop, **extra):
    return {"type": "object", "properties": {"x": prop}, "required": ["x"], "additionalProperties": False, **extra}


def body(text):
    return "<function=f>\n<parameter=x>\n" + text + "\n</parameter>\n</function>"


class ParameterSchema(unittest.TestCase):
    def check_value(self, prop, text, want, **extra):
        root = parameters(prop, **extra)
        before = copy.deepcopy(root)
        tool = {"name": "f", "parameters": root}
        expected = {"x": want}
        self.assertEqual(parse_tool_call(body(text), tool).arguments, expected)
        wrapped = "<tool_call>\n" + body(text) + "\n</tool_call>"
        for stream in (False, True):
            for size in (1, 7, len(wrapped)):
                parser = OutputParser(thinking=False, tools=[tool], stream_tools=stream)
                events = []
                for i in range(0, len(wrapped), size):
                    events.extend(parser.feed(wrapped[i:i + size]))
                events.extend(parser.finish())
                self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [expected])
                if stream:
                    self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")), expected)
        self.assertEqual(root, before)
        return root

    def test_type_and_combinations_intersect(self):
        for prop in ({"type": "string", "anyOf": [{"pattern": "^[0-9]+$"}, {"pattern": "^a$"}]},
                     {"type": "string", "anyOf": [{"type": ["string", "integer"]}]},
                     {"allOf": [{"type": ["string", "integer"]}, {"type": ["string", "null"]}]},
                     {"type": ["string", "integer"], "anyOf": [{"type": "string"}, {"type": "null"}],
                      "oneOf": [{"pattern": "^[0-9]+$"}, {"type": "integer"}]}):
            with self.subTest(prop=prop):
                self.assertEqual(_schema_types(prop), {"string"})
                root = self.check_value(prop, "1", "1")
                if HAVE_JSONSCHEMA:
                    jsonschema.validate({"x": "1"}, root)

    def test_root_definitions_not_property_definitions(self):
        prop = {"anyOf": [{"$ref": "#/$defs/text"}, {"type": "null"}],
                "$defs": {"text": {"type": "integer"}}}   # a decoy: this is NOT the reference's base
        for text, want in (("1", "1"), ("null", None), ('"1"', "1"), ("true", "true"), ("[1]", "[1]")):
            with self.subTest(text=text):
                root = self.check_value(prop, text, want, **{"$defs": {"text": {"type": "string"}}})
                if HAVE_JSONSCHEMA:
                    jsonschema.validate({"x": want}, root)
        # Without an explicit root, do not pretend the property's own definitions are the document.
        self.assertIsNone(_schema_types(prop))
        self.assertEqual(param_value("1", prop), 1)

    def test_local_pointer_chains_escapes_and_array_indices(self):
        defs = {"text/name~": {"type": "string"}, "chain": {"$ref": "#/$defs/text~1name~0"},
                "$id": {"type": "integer"},  # a definition name, not a resource identifier
                "list": {"anyOf": [{"type": "string"}, {"type": "null"}]}}
        for ref in ("#/$defs/text~1name~0", "#/$defs/chain", "#/$defs/list/anyOf/0", "#/$defs/%63hain"):
            with self.subTest(ref=ref):
                self.check_value({"$ref": ref}, "1", "1", **{"$defs": defs})

    def test_boolean_reference_targets(self):
        for target in (True, False):
            self.check_value({"$ref": "#/$defs/value"}, "1", 1, **{"$defs": {"value": target}})

    def test_ref_siblings_are_conjoined(self):
        self.check_value({"$ref": "#/$defs/union", "type": "string"}, "1", "1",
                         **{"$defs": {"union": {"type": ["string", "integer"]}}})

    def test_cycles_missing_external_and_anchor_refs_are_unknown(self):
        defs = {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}
        for ref in ("#/$defs/a", "#/$defs/missing", "#text", "other.json#/$defs/text"):
            with self.subTest(ref=ref):
                self.check_value({"$ref": ref}, "1", 1, **{"$defs": defs})
                self.check_value({"type": "string", "$ref": ref}, "1", "1", **{"$defs": defs})

    def test_hash_ref_points_to_the_root_not_the_property(self):
        root = self.check_value({"$ref": "#"}, '{"x":{}}', {"x": {}})
        self.assertEqual(_schema_types(root["properties"]["x"], root=root), {"object"})
        root["properties"]["x"]["type"] = "string"
        self.assertEqual(_schema_types(root["properties"]["x"], root=root), set())

    def test_nested_resources_do_not_resolve_from_the_outer_root(self):
        self.check_value({"$id": "nested", "$ref": "#/$defs/text"}, "1", 1,
                         **{"$defs": {"text": {"type": "string"}}})

    def test_depth_limit_unknown_and_boolean_branches(self):
        deep = {"type": "string"}
        for _ in range(12):
            deep = {"allOf": [deep]}
        for unknown in (True, {}, {"not": {"type": "integer"}}, {"$dynamicRef": "#text"}, deep):
            with self.subTest(unknown=unknown):
                self.assertIsNone(_schema_types(unknown))
                self.check_value(unknown, "1", 1)
                self.check_value({"type": "string", "anyOf": [unknown, False]}, "1", "1")
                self.check_value({"anyOf": [{"type": "string"}, unknown]}, "1", 1)
        self.assertEqual(_schema_types(False), set())
        self.assertEqual(_schema_types({"type": "string", "allOf": [False]}), set())
        self.assertEqual(_schema_types({"anyOf": [False, {"type": "string"}]}), {"string"})
        self.check_value(False, "1", 1)       # parser is not the strict validator

    def test_plain_string_keeps_raw_quotes_and_json_like_text(self):
        for text in ('"1"', "1", "null", "true", "[1]", '{"a":1}', '"a\\nb"'):
            with self.subTest(text=text):
                self.check_value({"type": "string"}, text, text)

    def test_enum_and_const_accept_both_representations(self):
        for prop in ({"enum": ["1", "2"]}, {"const": "1"}, {"type": "string", "enum": ["1", "2"]},
                     {"allOf": [{"enum": ["1", "2"]}, {"const": "1"}]},
                     {"anyOf": [{"const": "1"}, {"const": "2"}]},
                     {"oneOf": [{"const": "1"}, {"const": "2"}]}):
            for text in ("1", '"1"'):
                with self.subTest(prop=prop, text=text):
                    self.check_value(prop, text, "1")
        for text in ("1", '"1"'):
            self.check_value({"$ref": "#/$defs/value"}, text, "1", **{"$defs": {"value": {"const": "1"}}})
        self.check_value({"enum": ["a\nb"]}, '"a\\nb"', "a\nb")

    def test_ambiguous_or_unrecognized_quoted_string_keeps_original(self):
        for prop in ({"enum": ["1", '"1"']}, {"enum": ['"1"']},
                     {"type": "string", "anyOf": [{"const": "1"}, {}]}):
            with self.subTest(prop=prop):
                self.check_value(prop, '"1"', '"1"')
        self.check_value({"const": "1"}, '"other"', '"other"')

    def test_regular_non_string_values_and_numeric_intersections(self):
        for prop, text, want in (({"type": "integer"}, "1", 1), ({"type": "number"}, "1.5", 1.5),
                                 ({"type": "boolean"}, "true", True), ({"type": "array"}, "[1]", [1]),
                                 ({"type": "object"}, '{"a":1}', {"a": 1}),
                                 ({"type": ["string", "integer"]}, "1", 1),
                                 ({"type": "number", "allOf": [{"type": "integer"}]}, "1", 1),
                                 ({"type": ["string", "number"], "enum": [1, "foo"]}, "1.0", 1.0),
                                 ({"type": ["string", "number"], "enum": [1.0, "foo"]}, "1", 1),
                                 ({"type": ["string", "number"], "const": 1}, "1e0", 1.0),
                                 ({"type": ["string", "integer"]}, "1.0", 1.0)):
            with self.subTest(prop=prop):
                self.check_value(prop, text, want)
        self.assertEqual(_schema_types({"type": "number", "allOf": [{"type": "integer"}]}), {"integer"})

    def test_only_unconstrained_string_streams_before_parameter_end(self):
        for prop, immediate in (({"type": "string"}, True), ({"const": "1"}, False)):
            parser = OutputParser(thinking=False, tools=[{"name": "f", "parameters": parameters(prop)}],
                                  stream_tools=True)
            events = parser.feed('<tool_call>\n<function=f>\n<parameter=x>\n"1"')
            partial = "".join(e.text for e in events if e.kind == "tool_args")
            self.assertEqual(partial != '{"x":', immediate)


@unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
class StrictParameterSchema(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, ""), self.tok,
                           ChatTemplate(Path(__file__).with_name("test_chat_template.jinja")))
        self.httpd = serve(self.svc, port=0)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, root, text, stream):
        self.svc.engine = MockEngine(self.tok, "<tool_call>\n" + body(text) + "\n</tool_call>")
        request = {"messages": [{"role": "user", "content": "Call f."}], "reasoning_effort": "none",
                   "stream": stream, "tools": [{"type": "function", "function": {
                       "name": "f", "strict": True, "parameters": root}}]}
        req = urllib.request.Request(self.url, json.dumps(request).encode(), {"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.read().decode()

    def test_strict_success_uses_real_root_refs_and_same_stream_value(self):
        cases = [(parameters({"type": "string", "anyOf": [{"pattern": "^[0-9]+$"}, {"pattern": "^a$"}]}), "1"),
                 (parameters({"anyOf": [{"$ref": "#/$defs/text"}, {"type": "null"}]},
                             **{"$defs": {"text": {"type": "string"}}}), "1")]
        for prop in ({"enum": ["1", "2"]}, {"const": "1"}):
            cases.extend((parameters(prop), text) for text in ("1", '"1"'))
        for root, text in cases:
            jsonschema.validate({"x": "1"}, root)
            for stream in (False, True):
                with self.subTest(root=root, text=text, stream=stream):
                    status, raw = self.request(root, text, stream)
                    self.assertEqual(status, 200, raw)
                    if stream:
                        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                        self.assertFalse(any("error" in c for c in chunks), raw)
                        args = "".join(t.get("function", {}).get("arguments", "") for c in chunks
                                       for choice in c.get("choices", [])
                                       for t in choice.get("delta", {}).get("tool_calls", []))
                        self.assertIn("data: [DONE]", raw)
                    else:
                        args = json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                    self.assertEqual(json.loads(args), {"x": "1"})

    def test_decimal_integer_representation_keeps_a_valid_numeric_enum(self):
        root = parameters({"type": ["string", "number"], "enum": [1, "foo"]})
        jsonschema.validate({"x": 1.0}, root)
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, raw = self.request(root, "1.0", stream)
                self.assertEqual(status, 200, raw)
                self.assertNotIn("structured_output_failed", raw)
                if stream:
                    chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                    args = "".join(t.get("function", {}).get("arguments", "") for c in chunks
                                   for choice in c.get("choices", [])
                                   for t in choice.get("delta", {}).get("tool_calls", []))
                else:
                    args = json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                self.assertIsInstance(json.loads(args)["x"], float)
                self.assertEqual(json.loads(args), {"x": 1.0})

    def test_invalid_types_enum_and_patterns_still_fail_strict(self):
        for prop, text in (({"type": "integer"}, '"wrong"'), ({"const": "1"}, '"other"'), (False, "1"),
                           ({"type": "string", "anyOf": [{"pattern": "^a$"}]}, "1")):
            for stream in (False, True):
                with self.subTest(prop=prop, stream=stream):
                    status, raw = self.request(parameters(prop), text, stream)
                    self.assertEqual(status, 200 if stream else 502, raw)
                    self.assertIn("structured_output_failed", raw)
                    if stream:
                        self.assertNotIn('"tool_calls":', raw)
                        self.assertIn("data: [DONE]", raw)


if __name__ == "__main__":
    unittest.main()
