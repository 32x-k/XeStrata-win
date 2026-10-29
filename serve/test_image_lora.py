# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/loras.py and the image endpoint's adapters: the folder's listing (a file's keys read by its header), a
request's "lora" field and prompt tags, the IMG and NEED words the engine is sent, the response, and the models that
take none.

The files are written here, not read from a model: a safetensors' header is the 8-byte length of its JSON and the
JSON; a GGUF's is its metadata's fields and its tensors' infos (the data is never read: the listing knows a weight's
shape from it).

    python -m unittest serve.test_image_lora
"""
from __future__ import annotations

import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from serve import images, loras


def safetensors(path: Path, tensors: dict[str, tuple[int, ...]], meta: dict[str, str] | None = None) -> None:
    """A safetensors file of those tensors (F16) and that metadata; the tensors' data is zeros: only the header is
    read."""
    head: dict = {} if meta is None else {"__metadata__": meta}
    at = 0
    for name, shape in tensors.items():
        n = 1
        for d in shape:
            n *= d
        head[name] = {"dtype": "F16", "shape": list(shape), "data_offsets": [at, at + 2 * n]}
        at += 2 * n
    raw = json.dumps(head).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(at))


def _gguf_string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf(path: Path, tensors: dict[str, tuple[int, ...]], meta: dict[str, str] | None = None) -> None:
    """A GGUF file of version 3 with those tensors (F16) and string metadata (its count and the tensors' infos are the
    header; serve/loras.py reads no further)."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta or {})))
    for k, v in (meta or {}).items():
        out += _gguf_string(k) + struct.pack("<I", 8) + _gguf_string(v)
    for name, shape in tensors.items():
        dims = list(reversed(shape))    # GGUF's shape runs from the innermost dimension
        out += _gguf_string(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims)
        out += struct.pack("<IQ", 1, 0)   # F16, its offset in the data
    path.write_bytes(bytes(out) + bytes(4096))


def qwen_lora(rank: int = 4, blocks: tuple[int, ...] = (0, 1), prefix: str = "transformer.",
              extra: dict[str, tuple[int, ...]] | None = None, meta: dict[str, str] | None = None) -> dict:
    """A Qwen-Image-2.1 adapter's tensors, as diffusers' saves name them (transformer_blocks.N.attn.to_q and the
    like), at `rank` and that prefix."""
    out: dict[str, tuple[int, ...]] = {}
    for b in blocks:
        for name, cols, rows in (("attn.to_q", 256, 256), ("attn.to_out.0", 256, 256),
                                 ("img_mlp.gate_layer", 256, 512), ("img_mlp.proj", 256, 512),
                                 ("img_mlp.out", 512, 256)):
            out[f"{prefix}transformer_blocks.{b}.{name}.lora_A.weight"] = (rank, cols)
            out[f"{prefix}transformer_blocks.{b}.{name}.lora_B.weight"] = (rows, rank)
    out.update(extra or {})
    return out


class Tree(unittest.TestCase):
    """A temporary LoRA folder (the config's "lora_dir")."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name) / "loras"
        self.d.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def folder(self, family: str | None = None) -> loras.Folder:
        cfg = {"diffusion_model": str(self.d / "dit.gguf"), "lora_dir": str(self.d)}
        if family:
            cfg["family"] = family
        return loras.Folder(cfg)

    def qwen(self, name: str = "style/gouache", **kw) -> Path:
        path = self.d / f"{name}.safetensors"
        safetensors(path, qwen_lora(**{k: v for k, v in kw.items() if k != "meta"}),
                    kw.get("meta", {"modelspec.trigger_phrase": "gouache style",
                                    "recommended_strength_model": "0.8"}))
        return path


class Listing(Tree):
    """What the folder holds for a model: the files its DiT takes, each as the listing's fields."""

    def test_a_file_the_dit_takes_is_listed(self):
        self.qwen()
        entries = self.folder().listing()
        self.assertEqual([e["name"] for e in entries], ["style/gouache"])
        e = entries[0]
        self.assertEqual(e["path"], "style/gouache.safetensors", "the path under the folder, with its extension")
        self.assertEqual(e["rank"], 4)
        self.assertEqual((e["scale"], e["trigger"]), (0.8, "gouache style"))
        self.assertEqual(e["targets"], ["attn.to_out.0", "attn.to_q", "img_mlp.gate_layer", "img_mlp.out",
                                        "img_mlp.proj"])
        self.assertGreater(e["size_mib"], 0)

    def test_a_gguf_and_a_fused_gate_up(self):
        gguf(self.d / "fused.gguf", {
            "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight": (2, 256),
            "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight": (1024, 2)},
            {"ss_network_dim": "2"})
        entries = self.folder().listing()
        self.assertEqual([(e["name"], e["rank"], e["targets"]) for e in entries],
                         [("fused", 2, ["img_mlp.gate_up"])])
        self.assertEqual(entries[0]["scale"], 1.0, "no recommended strength: 1.0")
        self.assertIsNone(entries[0]["trigger"])

    def test_another_models_file_and_the_forms_not_applied_are_left_out(self):
        self.qwen("qwen")
        safetensors(self.d / "moss.safetensors", {
            "layers.0.proj.q.lora_A.weight": (8, 256), "layers.0.proj.q.lora_B.weight": (256, 8)})
        safetensors(self.d / "dora.safetensors", qwen_lora(
            extra={"transformer.transformer_blocks.0.attn.to_q.dora_scale": (256,)}))
        safetensors(self.d / "lokr.safetensors", {
            "transformer.transformer_blocks.0.attn.to_q.lokr_w1": (256, 256),
            "transformer.transformer_blocks.0.attn.to_q.lokr_w2_a": (4, 64),
            "transformer.transformer_blocks.0.attn.to_q.lokr_w2_b": (4, 64)})
        safetensors(self.d / "text-encoder.safetensors", {
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (4, 64),
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": (256, 4)})
        safetensors(self.d / "a-norm.safetensors", {
            "transformer.transformer_blocks.0.attn.norm_q.lora_A.weight": (4, 256),
            "transformer.transformer_blocks.0.attn.norm_q.lora_B.weight": (256, 4)})
        (self.d / "broken.safetensors").write_bytes(b"not a header at all")
        (self.d / "notes.txt").write_text("x", encoding="utf-8")
        safetensors(self.d / ".hidden.safetensors", qwen_lora())
        f = self.folder()
        self.assertEqual([e["name"] for e in f.listing()], ["qwen"])
        self.assertEqual(f.skipped["moss"], "layers.0.proj.q is not one of this image model's weights")
        self.assertIn("DoRA", f.skipped["dora"])
        self.assertIn("lokr", f.skipped["lokr"])
        self.assertIn("text encoder", f.skipped["text-encoder"])
        self.assertIn("not one of this image model's weights", f.skipped["a-norm"])
        self.assertIn("header", f.skipped["broken"])

    def test_a_name_with_a_space_and_the_line_word(self):
        self.qwen("my style")
        e = self.folder().listing()[0]
        self.assertEqual(e["name"], "my style")
        self.assertEqual(loras.encode_path("/x/my style.safetensors"), "/x/my%20style.safetensors")
        self.assertEqual(loras.encode_path("C:\\a b\\c:d.safetensors"), "C:\\a%20b\\c:d.safetensors")

    def test_two_files_of_one_stem_keep_their_extension(self):
        safetensors(self.d / "gouache.safetensors", qwen_lora())
        gguf(self.d / "gouache.gguf", {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": (2, 256),
                                       "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": (256, 2)})
        self.assertEqual([e["name"] for e in self.folder().listing()], ["gouache.gguf", "gouache.safetensors"])

    def test_the_listing_follows_the_folder(self):
        self.assertEqual(self.folder().listing(), [])
        self.qwen("late")
        self.assertEqual([e["name"] for e in self.folder().listing()], ["late"],
                         "a new file: the folder's mtime changed")

    def test_a_model_that_takes_no_adapter_lists_none(self):
        self.qwen()
        for family in ("moss-image", "mage-flow", "hunyuan-image-3"):
            self.assertEqual(self.folder(family).listing(), [], family)


class Requests(Tree):
    """A request's names: the "lora" field, local_path, and the prompt's tags."""

    def setUp(self):
        super().setUp()
        self.qwen("style/gouache")
        self.f = self.folder()

    def test_the_fields_forms(self):
        want = [{"name": "style/gouache", "scale": 0.5, "path": str(self.d / "style/gouache.safetensors")}]
        for value in ([{"name": "style/gouache", "scale": 0.5}], {"name": "style/gouache", "scale": 0.5},
                      '[{"name": "style/gouache", "scale": "0.5"}]', {"name": "style/gouache",
                                                                              "scale": "0.5"}):
            self.assertEqual(loras.field(value, self.f), want, value)
        self.assertEqual(loras.field(None, self.f), [])
        self.assertEqual(loras.field([], self.f), [])
        self.assertEqual(loras.field("style/gouache", self.f)[0]["scale"], 1.0, "a name alone: 1.0")
        self.assertEqual(loras.field([{"name": "style/gouache"}], self.f)[0]["scale"], 1.0, "no scale: 1.0")

    def test_a_name_the_model_does_not_have_is_refused_with_the_names_it_has(self):
        with self.assertRaises(loras.LoraError) as e:
            loras.field([{"name": "nope"}], self.f)
        self.assertIn("nope", str(e.exception))
        self.assertIn("style/gouache", str(e.exception))
        scale = {"transformer.transformer_blocks.0.attn.to_q.dora_scale": (256,)}
        safetensors(self.d / "dora.safetensors", qwen_lora(extra=scale))
        with self.assertRaises(loras.LoraError) as e:
            loras.field([{"name": "dora"}], self.f)
        self.assertIn("DoRA", str(e.exception), "a file of the folder, and why it is not this model's")

    def test_the_limits_and_the_repeats(self):
        with self.assertRaises(loras.LoraError):
            loras.field([{"name": "style/gouache"}, {"name": "style/gouache"}], self.f)
        with self.assertRaises(loras.LoraError):
            loras.field([{"name": "style/gouache", "scale": 2.5}], self.f)
        with self.assertRaises(loras.LoraError):
            loras.field([{"name": "style/gouache", "scale": "big"}], self.f)
        with self.assertRaises(loras.LoraError):
            loras.field([{}], self.f)
        with self.assertRaises(loras.LoraError):
            loras.field(7, self.f)
        self.qwen("more", blocks=(0,))
        uses = [{"name": "style/gouache"}] + [{"name": "more"}] * 1
        self.assertEqual(len(loras.field(uses, self.f)), 2)
        with self.assertRaises(loras.LoraError):
            loras.limit([{"name": "x"}] * 5)
        loras.limit([{"name": "x"}] * 4)

    def test_local_path_is_a_file_inside_the_folder_only(self):
        inside = str(self.d / "style/gouache.safetensors")
        use = loras.field([{"local_path": inside, "scale": 0.25}], self.f)[0]
        self.assertEqual((use["name"], use["scale"]), ("style/gouache", 0.25))
        self.assertEqual(loras.field([{"local_path": "style/gouache.safetensors"}], self.f)[0]["name"], "style/gouache")
        outside = Path(self.tmp.name) / "outside.safetensors"
        safetensors(outside, qwen_lora())
        for bad in (str(outside), "../outside.safetensors", str(self.d / "style"), "/etc/passwd", ""):
            with self.assertRaises(loras.LoraError, msg=bad):
                loras.field([{"local_path": bad}], self.f)
        link = self.d / "link.safetensors"
        link.symlink_to(outside)
        with self.assertRaises(loras.LoraError):
            loras.field([{"local_path": str(link)}], self.f)
        with self.assertRaises(loras.LoraError):   # a file of the folder that is not this model's
            loras.field([{"local_path": str(self.d / "style/gouache.safetensors")}],
                        loras.Folder({"lora_dir": str(self.d), "family": "moss-image"}))

    def test_a_scale_text_is_taken_and_said_not_applied(self):
        use = loras.field([{"name": "style/gouache", "scale_text": 0.4}], self.f)[0]
        self.assertEqual(use["scale_text"], 0.4)
        with self.assertRaises(loras.LoraError):
            loras.field([{"name": "style/gouache", "scale_text": -3}], self.f)

    def test_tags_come_out_and_unknown_ones_are_left_out(self):
        text, uses, unknown = loras.strip_tags("a cat <lora:style/gouache:0.5> in <lora:nope:1> snow", self.f)
        self.assertEqual(text, "a cat  in  snow")
        self.assertEqual([(u["name"], u["scale"]) for u in uses], [("style/gouache", 0.5)])
        self.assertEqual(unknown, ["nope"])
        text, uses, unknown = loras.strip_tags("<lora:style/gouache>", self.f)
        self.assertEqual((text, uses[0]["scale"], unknown), ("", 1.0, []))
        text, uses, unknown = loras.strip_tags("<lora:style/gouache:1:0.5>", self.f)
        self.assertEqual((text, uses[0]["scale_text"]), ("", 0.5), "A1111's text encoder's weight: taken, not applied")
        text, uses, unknown = loras.strip_tags("<lora:style/gouache:0.3> and <lora:style/gouache:0.9>", self.f)
        self.assertEqual([u["scale"] for u in uses], [0.3], "the first naming of a name wins")
        self.assertEqual(loras.strip_tags("no tags here", self.f), ("no tags here", [], []))
        with self.assertRaises(loras.LoraError):
            loras.strip_tags("<lora:style/gouache:heavy>", self.f)
        with self.assertRaises(loras.LoraError):
            loras.strip_tags("<lora:style/gouache:1:heavy>", self.f)

    def test_a_model_that_takes_no_adapter_refuses_both(self):
        for family in ("moss-image", "mage-flow", "hunyuan-image-3"):
            f = self.folder(family)
            with self.assertRaises(loras.LoraError) as e:
                loras.field([{"name": "style/gouache"}], f)
            self.assertIn("takes no adapters", str(e.exception))
            with self.assertRaises(loras.LoraError):
                loras.strip_tags("a cat <lora:style/gouache:1>", f)
            self.assertEqual(loras.field(None, f), [], "a request of none is no request for one")


def cfg(**kw) -> dict:
    """A run config's "images" section of Qwen-Image-2.1 Turbo."""
    out = {"exe": "strata-image", "diffusion_model": "dit.gguf", "vae": "vae.safetensors", "llm": "text.gguf",
           "schedule": "turbo"}
    out.update(kw)
    return out


class FakeTokenizer:
    """What prompt_ids asks of a tokenizer: a prompt's ids (20 of them, as the chat form's turn has)."""

    def encode(self, text, parse_special=False, plain=None):
        return [7] * 20


class FakeEngine:
    """The engine's side of a job: the lines it is sent, and zeros of the size asked for as the image."""

    def __init__(self):
        self.lines: list[str] = []

    def alive(self):
        return True

    def ask(self, line, want, timeout, cancel=None, step=None, stage=None):
        self.lines.append(line)
        if want != "IMAGE":
            return {"least_mib": 1, "full_mib": 2, "held_mib": 3, "cap_mib": 4, "host_mib": 5, "free_mib": 6,
                    "lora_n": 2 if " LORA:" in line else 0, "lora_mib": 8 if " LORA:" in line else 0}
        width, height, out = int(line.split()[1]), int(line.split()[2]), line.split()[5]
        Path(out).write_bytes(bytes(width * height * 4))
        r = {"text_s": 0.5, "sampling_s": 1.0, "decode_s": 0.25}
        if " LORA:" in line:
            r.update(lora_n=2, lora_mib=8)
        return r

    def close(self, shutdown=False):
        pass


class Job(Tree):
    """parse_request's adapters, the IMG and NEED words, and the response."""

    def setUp(self):
        super().setUp()
        self.qwen("style/gouache")
        self.qwen("Strong Style", blocks=(0,))
        self.cfg = cfg(lora_dir=str(self.d))
        self.gen = images.ImageGen(self.cfg)
        self.gen.tok, self.gen.engine = FakeTokenizer(), FakeEngine()

    def tearDown(self):
        shutil.rmtree(self.gen.dir, ignore_errors=True)
        super().tearDown()

    def test_the_field_and_the_tags_reach_the_job(self):
        job = images.parse_request({"prompt": "a cat <lora:style/gouache:0.5>", "size": "256x256"}, self.cfg)
        self.assertEqual(job["prompt"], "a cat ")
        self.assertEqual([(u["name"], u["scale"]) for u in job["loras"]], [("style/gouache", 0.5)])
        job = images.parse_request({"prompt": "a cat", "size": "256x256",
                                    "lora": [{"name": "Strong Style", "scale": -0.5}]}, self.cfg)
        self.assertEqual(job["loras"], [{"name": "Strong Style", "scale": -0.5,
                                         "path": str(self.d / "Strong Style.safetensors")}])
        job = images.parse_request({"prompt": "a cat", "size": "256x256", "lora": {"name": "style/gouache"},
                                    "negative_prompt": "blurry <lora:Strong Style:2>"}, self.cfg)
        self.assertEqual([(u["name"], u["scale"]) for u in job["loras"]],
                         [("style/gouache", 1.0), ("Strong Style", 2.0)])
        self.assertEqual(job["negative_prompt"], "blurry ", "the negative prompt's tags come out too")
        self.assertNotIn("lora_unknown", job)

    def test_a_tags_unknown_name_is_left_out_and_said(self):
        job = images.parse_request({"prompt": "a <lora:nope:1> cat", "size": "256x256"}, self.cfg)
        self.assertEqual(job["prompt"], "a  cat")
        self.assertEqual(job["lora_unknown"], ["nope"])
        self.assertNotIn("loras", job, "nothing is applied")

    def test_the_refusals(self):
        with self.assertRaises(images.ImageRequestError):
            images.parse_request({"prompt": "a cat", "lora": [{"name": "nope"}]}, self.cfg)
        with self.assertRaises(images.ImageRequestError):
            images.parse_request({"prompt": "a cat", "lora": [{"name": "style/gouache"}] * 2}, self.cfg)
        with self.assertRaises(images.ImageRequestError):
            images.parse_request({"prompt": "<lora:style/gouache:1>"}, self.cfg)
        with self.assertRaises(images.ImageRequestError):
            images.parse_request({"prompt": "a cat", "lora": [{"name": "style/gouache", "scale": 3}]}, self.cfg)
        mage = {**self.cfg, "family": "mage-flow", "schedule": "mage", "steps": 4}
        with self.assertRaises(images.ImageRequestError) as e:
            images.parse_request({"prompt": "a cat", "lora": [{"name": "style/gouache"}]}, mage)
        self.assertIn("Mage-Flow", str(e.exception))
        hunyuan = {"exe": "strata-image", "family": "hunyuan-image-3", "diffusion_model": "d", "vae": "v",
                   "tokenizer": "t", "system_prompt": "s", "schedule": "hunyuanimage3"}
        with self.assertRaises(images.ImageRequestError) as e:
            images.parse_request({"prompt": "a cat <lora:style/gouache:1>"}, hunyuan)
        self.assertIn("HunyuanImage 3.0", str(e.exception))

    def test_a_scale_text_is_said_not_applied(self):
        job = images.parse_request({"prompt": "a cat", "lora": [{"name": "style/gouache", "scale_text": 0.6}]},
                                   self.cfg)
        self.assertNotIn("scale_text", job["loras"][0])
        self.assertTrue(job["lora_text"])
        made, figures = self.gen.run(job)
        self.assertEqual(figures["lora_text"], "not applied (the image engine applies no text encoder's adapter)")

    def test_the_words_the_engine_is_sent_and_the_response(self):
        job = images.parse_request({"prompt": "a cat", "size": "256x256", "seed": 3,
                                    "lora": [{"name": "Strong Style", "scale": 0.75}]}, self.cfg)
        word = f" LORA:0.75:{loras.encode_path(str(self.d / 'Strong Style.safetensors'))}"
        need = self.gen.need(job)
        self.assertTrue(self.gen.engine.lines[-1].endswith(word), self.gen.engine.lines[-1])
        self.assertEqual((need["lora_n"], need["lora_mib"]), (2, 8))
        made, figures = self.gen.run(job)
        self.assertEqual(len(made), 1)
        img = self.gen.engine.lines[-1]
        self.assertTrue(img.startswith("IMG 256 256 3 14 "), img)
        self.assertTrue(img.endswith(word), img)
        self.assertEqual(loras.encode_path(str(self.d / "Strong Style.safetensors")).replace("%20", " "),
                         str(self.d / "Strong Style.safetensors"), "the path itself, its separators %XX'd")
        self.assertEqual((figures["lora_n"], figures["lora_mib"]), (2, 8))
        body = self.gen.response(made, job, figures, "http://x")
        self.assertEqual(body["data"][0]["lora"], [{"name": "Strong Style", "scale": 0.75}])
        self.assertEqual(body["strata"]["lora_n"], 2)
        job = images.parse_request({"prompt": "a cat", "size": "256x256", "seed": 3}, self.cfg)
        made, figures = self.gen.run(job)
        self.assertNotIn("lora", self.gen.response(made, job, figures, "http://x")["data"][0])
        self.assertNotIn("lora_n", figures, "no adapter: the figures are as they were")
        self.assertNotIn(" LORA:", self.gen.engine.lines[-1])

    def test_the_listing_is_the_models_capabilities(self):
        d = self.gen.defaults()
        self.assertEqual([e["name"] for e in d["loras"]], ["Strong Style", "style/gouache"])
        self.assertEqual(images.model_defaults(self.cfg).get("loras"), None, "the pure defaults carry no folder read")


if __name__ == "__main__":
    unittest.main()
