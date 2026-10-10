# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Writes prompt.txt: the first 2,000 characters of wikitext-2's test split (its body lines, headings left out), in
Ornith's chat format with a request to summarize.  The text itself (CC BY-SA) is not kept here.

    python make_prompt.py wikitext-2-raw/wiki.test.raw > prompt.txt
"""
import sys


def main():
    lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
    body = [s.strip() for s in lines if s.strip() and not s.strip().startswith("=")]
    text = " ".join(body)[:2000]
    sys.stdout.write("<|im_start|>user\nSummarize the following passage in a few paragraphs, then list three "
                     "questions it raises.\n\n" + text + "<|im_end|>\n<|im_start|>assistant\n")


if __name__ == "__main__":
    main()
