# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Standard-library synthetic tests; no engine, GPU, corpus or external tools."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from tools import profile_pool_timing as p


def toy():
    values = {key: "0.000/0/0" for key in p.FIELDS}
    values.update(embed="1.000/1/4", routing_wait="2.000/1/0", cpu_pool="3.000/2/8")
    row = " ".join(f"{key}={value}" for key, value in values.items())
    return ("strata generate: 条件未確認\nplanned dump rows: 50\n" +
            "\n".join(f"pool timing position {i}: {row}" for i in range(4)) +
            "\nprompt  : 7 0 9\noutput  : 0 99\n"
            "prefill 2 tokens in 12.4 ms -> 0 tok/s\n"
            "decode 2 tokens in 12.6 ms -> 0 tok/s\n")


class ProfilePoolTimingTests(unittest.TestCase):
    def parse(self, text=None):
        return p.parse_bytes((toy() if text is None else text).encode("utf-8"), "条件.txt")

    def rejects(self, text):
        with self.assertRaises(ValueError):
            self.parse(text)

    def test_actual_rows_zero_ids_and_shorter_than_planned(self):
        run = self.parse()
        self.assertEqual(run["actual_rows"], 4)
        self.assertEqual(run["prompt_ids"], [7, 0, 9])
        self.assertEqual(run["output_ids"], [0, 99])  # EOS identity/cap are not inferred.
        self.assertTrue(run["metadataNotProvided"])
        self.assertEqual(run["sha256"], hashlib.sha256(toy().encode("utf-8")).hexdigest())
        for name in ("prefill", "decode", "all"):
            s = run["scopes"][name]
            self.assertEqual(s["positions"], 4 if name == "all" else 2)
            self.assertEqual(len(s["spans"]), 12)
        s = run["scopes"]["all"]
        self.assertEqual((s["wall_ms"], s["mean_ms"], s["rest"]["wall_ms"]), (25, 6.25, 1))
        self.assertEqual((s["api_counts"], s["payload_bytes"]), (16, 48))
        embed = s["spans"]["embed"]
        self.assertEqual((embed["mean_ms"], embed["share_percent"]), (1, 16))
        bound = embed["independent_ideal_zero_window_upper_bound"]
        self.assertAlmostEqual(bound["speedup"], 1 / .84)
        self.assertTrue(bound["theoretical_not_predicted"])

    def test_truncated_extra_and_noncontiguous_rows(self):
        for text in (toy().replace("pool timing position 3:", "ignored position 3:"),
                     toy().replace("position 3:", "position 4:"),
                     toy() + toy().splitlines()[2].replace("position 0:", "position 4:")):
            self.rejects(text)

    def test_duplicate_rows_and_records(self):
        for line in (toy().splitlines()[2], "prompt : 1", "decode 2 tokens in 12.6 ms"):
            self.rejects(toy() + line + "\n")

    def test_duplicate_missing_unknown_fields_and_shape(self):
        for replacement in ("embed=1/1/4 embed=1/1/4", "", "mystery=1/1/4",
                            "embed=1/1", "embed=1/1/4/5", "embed=1/1.5/4"):
            self.rejects(toy().replace("embed=1.000/1/4", replacement, 1))
        self.rejects(toy().replace("prompt  : 7 0 9", "prompt : []"))

    def test_timer_mismatch_and_missing(self):
        self.rejects(toy().replace("prefill 2", "prefill 3"))
        self.rejects(toy().replace("decode 2", "decode 3"))
        self.rejects(toy().replace("decode 2 tokens in 12.6 ms -> 0 tok/s\n", ""))

    def test_nonfinite_negative_and_overflow(self):
        for bad in ("nan", "inf", "-inf", "-1", "1e309"):
            self.rejects(toy().replace("1.000/1/4", f"{bad}/1/4"))
            self.rejects(toy().replace("12.6 ms", f"{bad} ms"))
        self.rejects(toy().replace("/1/4", "/-1/4"))
        self.rejects(toy().replace("/1/4", "/1/-4"))
        self.rejects(toy().replace("1.000/1/4", "1e308/1/4"))

    def test_rounding_negative_rest_and_impossible_sum(self):
        run = self.parse(toy().replace("12.4 ms", "11.9 ms").replace("12.6 ms", "12.0 ms"))
        s = run["scopes"]["all"]
        self.assertAlmostEqual(s["rest"]["wall_ms"], -.1)
        self.assertNotIn("independent_ideal_zero_window_upper_bound", s["rest"])
        self.rejects(toy().replace("12.4 ms", "11.8 ms"))
        zero = toy().replace("2.000/1/0", "0.000/1/0").replace("3.000/2/8", "0.000/2/8")
        s = self.parse(zero.replace("12.4 ms", "2.0 ms"))["scopes"]["prefill"]
        self.assertNotIn("independent_ideal_zero_window_upper_bound", s["spans"]["embed"])

    def test_zero_scope(self):
        zero = toy().replace("prompt  : 7 0 9", "prompt : 0").replace("output  : 0 99", "output : 0 99 0 99")
        zero = zero.replace("prefill 2 tokens in 12.4", "prefill 0 tokens in 0.0").replace("decode 2 tokens in 12.6", "decode 4 tokens in 25.0")
        self.assertIsNone(self.parse(zero)["scopes"]["prefill"]["mean_ms"])
        self.rejects(zero.replace("prefill 0 tokens in 0.0", "prefill 0 tokens in 1.0"))

    def test_unicode_cli_file_stdout_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            src, out = Path(directory) / "条件.txt", Path(directory) / "結果.json"
            src.write_text(toy(), encoding="utf-8")
            self.assertEqual(p.main([str(src), str(src), "--output", str(out)]), 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(report["runs"]), 2)
            self.assertIn("条件未確認", out.read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                p.main([str(src)])
            self.assertEqual(json.loads(stream.getvalue())["runs"][0]["source_path"], str(src))
            src.write_text(toy().replace("embed=1.000/1/4", "bad=1/1/4"), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                p.main([str(src)])
            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
