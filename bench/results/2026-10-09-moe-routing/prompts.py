# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Writes one prompt file per conversation, rendered with the model's own chat template.

    python prompts.py CHAT_TEMPLATE.jinja OUT_DIR [NAME=VALUE ...]

NAME=VALUE pairs go to the template as variables (GLM-5.3-Flash: reasoning_effort=high).
Copied from 2026-10-04-llmjp-routing (llm-jp-4.1's llm-jp-harmony-v1, which takes none), with those variables added.
"""
import json
import sys
from datetime import date
from pathlib import Path

import jinja2

CONVERSATIONS = {
    "ja-explain": "量子コンピュータが従来のコンピュータより速く解ける問題と、そうでない問題の違いを、高校生にわかるように説明してください。",
    "ja-business": (
        "社内で週 1 回の定例会議が長引きがちです。"
        "会議を 30 分に収めるための具体的な運営ルールを提案し、導入するときの反発への対処も書いてください。"
    ),
    "ja-summary": (
        "次の文章を 3 行で要約し、筆者の主張に対する反論を 1 つ挙げてください。\n\n"
        "地方の公共交通は、利用者の減少と運転手の不足という二つの問題を同時に抱えている。路線を減らせば利用者はさらに減り、"
        "減便は自家用車への依存を強める。一方で、自動運転やデマンド交通に期待する声もあるが、導入費用と運用の担い手が"
        "見つからない地域が多い。筆者は、交通を独立した事業として採算をとるのではなく、医療や教育と同じ基盤として、"
        "自治体が費用を負担する仕組みに改めるべきだと考える。そのためには、交通の便益を住民の健康や就労の機会として"
        "測り直す必要がある。"
    ),
    "code-python": (
        "Python で、巨大な CSV ファイル（数十 GB）を一定のメモリで読み、列 `user_id` ごとに `amount` の合計と件数を"
        "集計する関数を書いてください。型ヒントとテストも付けてください。"
    ),
    "code-review": (
        "次の C++ のコードの問題点を指摘し、直したコードを示してください。\n\n```cpp\n"
        "std::vector<int>* make(int n) {\n  std::vector<int> v;\n  for (int i = 0; i <= n; i++) v.push_back(i);\n"
        "  return &v;\n}\n\nint sum(std::vector<int>* v) {\n  int s;\n"
        "  for (size_t i = 0; i < v->size(); i++) s += (*v)[i];\n"
        "  return s;\n}\n```"
    ),
    "math": "1 から 100 までの整数のうち、3 でも 5 でも割り切れないものの和を求めてください。途中の考え方も示してください。",
    "en-essay": (
        "Write a short essay (about 300 words) on whether cities should ban cars from their historic centres. "
        "Give arguments on both sides and end with your own view."
    ),
    "ja-creative": "雨の日の図書館を舞台にした、400 字ほどの短い物語を書いてください。最後に小さな驚きを入れてください。",
}


def main():
    template, out = Path(sys.argv[1]).read_text(), Path(sys.argv[2])
    variables = dict(a.split("=", 1) for a in sys.argv[3:])
    out.mkdir(parents=True, exist_ok=True)
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])   # GLM's template uses {% break %}
    env.globals["strftime_now"] = lambda f: date(2026, 10, 4).strftime(f)
    env.filters["tojson"] = lambda x, **_: json.dumps(x, ensure_ascii=False)
    t = env.from_string(template)
    for name, text in CONVERSATIONS.items():
        rendered = t.render(messages=[{"role": "user", "content": text}], add_generation_prompt=True,
                            **variables)
        (out / f"{name}.txt").write_text(rendered)


if __name__ == "__main__":
    main()
