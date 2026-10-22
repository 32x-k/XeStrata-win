# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/pack_gguf.py - the engine's pack for a DeepSeek-V4.1 GGUF (unsloth's DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL
and the like: dense BF16, routed experts MXFP4, Engram tables MXFP8 or Q6_K), beside the model's files (no weights
copied).

    python tools/ds4/pack_gguf.py --model ~/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-Q8_K_XL

writes <model>/maya-ds4/ in pack.py's format, so the engine reads both:
  ds4.txt       the geometry under config.json's names (the GGUF's deepseek41.* keys translated), model_dir, format gguf
  tensors.txt   every tensor under the checkpoint's names: `file <i> <name>` lines, then
                `t <name> <dtype> <file> <offset> <bytes> <dims...>` - dims outermost first (torch order: a matrix is
                [out][in]), dtype BF16 / F32 / MXFP4 / MXFP8 (ggml's block layouts, 32 values a block); the routed
                experts as one tensor a layer and role ([n_exp][out][in])
  engram.bin    the Engram hash tables, from the GGUF's metadata (pack.py's layout; no tokenizers package needed)
  tokenizer/    the server's tokenizer from the GGUF's vocabulary and merges (pre "deepseek3"), and its chat template

A GGUF has no DSpark drafter: drafter_pack.py makes it from DeepSeek's checkpoint into ../dspark/, and the pack takes it
from there when it is present.  Nor a vision tower: vision_pack.py makes it from DeepSeek's checkpoint into ../vision/
(--vision: another folder), and the pack takes it and its geometry (vision.<key>, image_token_id) from there when it is
present; without it the engine reads text only.

A quant that leaves the two Engram tables out (v1.1.16's Maya-S; Maya's quants hold them since) takes them from a set
of GGUF files that holds them, beside it: --engram, else ../Engram/ (unsloth's MXFP8 shard of them, where setups before
v1.1.17 put it) when it is whole.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import struct
import sys

TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17),
         67: ("MXFP8", 32, 33),          # ggml type -> (name, values a block, bytes a block); 67: unsloth's fork
         43: ("MXFP8", 32, 33),          # llama.cpp's own id (unsloth's GGUFs from 2026-10-19 on)
         # the low-bit expert types (the Maya quants' experts; the engine's ds4_iq kernels)
         10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
         14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
         20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
         # the trellis types (tools/trellis/tqenc.hpp; 145 / 146 are the agentionai fork's ggml ids, 224-227 Strata's own)
         145: ("TQK6", 128, 26), 146: ("TQK7", 128, 30), 224: ("TQK5X", 128, 22), 225: ("TQK4X", 128, 18),
         226: ("TQK5T", 128, 22), 227: ("TQK4T", 128, 18)}
# the Engram set a quant without the tables takes them from by default, beside it (or inside its folder): unsloth's
# MXFP8 shard of them, where setups before v1.1.17 put it for Maya-S (--engram: any other)
ENGRAM_SETS = ("Engram",)


class Gguf:
    """One GGUF shard's metadata and tensor infos (offsets absolute in the file)."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        with path.open("rb") as f:
            self.f = f
            magic = f.read(4)
            if magic != b"GGUF":
                raise SystemExit("%s: not a GGUF file" % path)
            self.version, nt, nkv = struct.unpack("<IQQ", f.read(20))
            self.kv = {}
            for _ in range(nkv):
                k = self._str()
                self.kv[k] = self._val(struct.unpack("<I", f.read(4))[0])
            self.tensors = []
            for _ in range(nt):
                name = self._str()
                nd = struct.unpack("<I", f.read(4))[0]
                dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
                typ, off = struct.unpack("<IQ", f.read(12))
                self.tensors.append((name, typ, dims, off))
            align = int(self.kv.get("general.alignment", 32))
            pos = f.tell()
            self.data_start = (pos + align - 1) // align * align
        del self.f

    def _str(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _val(self, t):
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in fmt:
            n = struct.calcsize(fmt[t])
            return struct.unpack("<" + fmt[t], self.f.read(n))[0]
        if t == 8:
            return self._str()
        if t == 9:
            at, n = struct.unpack("<IQ", self.f.read(12))
            if at in fmt:
                n_b = struct.calcsize(fmt[at])
                return list(struct.unpack("<%d%s" % (n, fmt[at]), self.f.read(n * n_b)))
            return [self._val(at) for _ in range(n)]
        raise SystemExit("%s: metadata type %d" % (self.path, t))


# GGUF name (blk.N. stripped) -> the checkpoint's name under layers.N. (pack.py / the engine)
LAYER = {
    "attn_norm.weight": "attn_norm.weight",
    "attn_sinks.weight": "attn.attn_sink",
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_kv.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_output_a.weight": "attn.wo_a.slice",        # one entry a group (the engine's slices)
    "attn_output_b.weight": "attn.wo_b.weight",
    "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
    "attn_compressor_gate.weight": "attn.compressor.wgate.weight",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.attn_k.weight": "attn.indexer.wk.weight",
    "indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight",
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_base.weight": "hc_attn_base",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_ffn_scale.weight": "hc_ffn_scale",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "exp_probs_b_vl.bias": "ffn.gate.bias_vl",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    "ffn_gate_exps.weight": "ffn.experts.w1",             # [n_exp][ff][dim]: one entry an expert
    "ffn_up_exps.weight": "ffn.experts.w3",
    "ffn_down_exps.weight": "ffn.experts.w2",             # [n_exp][dim][ff]
    "engram_embd.weight": "engram.embed.weight",
    "engram_kv.weight": "engram.wkv.weight",
    "engram_gate.weight": "engram.gate.weight",          # q_weight * k_weight, folded by the converter
}
GLOBAL = {"token_embd.weight": "embed.weight", "output_norm.weight": "norm.weight", "output.weight": "head.weight"}


def geometry(kv: dict, vocab: int) -> dict:
    """config.json's text_config names (what the engine reads) from the GGUF's deepseek41.* keys."""
    a = kv["general.architecture"]
    g = lambda k, d=None: kv.get("%s.%s" % (a, k), d)  # noqa: E731
    nl = g("block_count")
    ratios = list(g("attention.compress_ratios", []))
    out = {
        "vocab_size": vocab,
        "hidden_size": g("embedding_length"),
        "num_hidden_layers": nl,
        "num_attention_heads": g("attention.head_count"),
        "head_dim": g("attention.key_length"),
        "qk_rope_head_dim": g("rope.dimension_count"),
        "q_lora_rank": g("attention.q_lora_rank"),
        "o_lora_rank": g("attention.output_lora_rank"),
        "o_groups": g("attention.output_group_count"),
        "n_routed_experts": g("expert_count"),
        "num_experts_per_tok": g("expert_used_count"),
        "moe_intermediate_size": g("expert_feed_forward_length"),
        "sliding_window": g("attention.sliding_window"),
        "hc_mult": g("hyper_connection.count"),
        "hc_sinkhorn_iters": g("hyper_connection.sinkhorn_iterations"),
        "hc_eps": g("hyper_connection.epsilon"),
        "index_n_heads": g("attention.indexer.head_count"),
        "index_head_dim": g("attention.indexer.key_length"),
        "index_topk": g("attention.indexer.top_k"),
        "candidate_source_layer_id": g("attention.indexer.candidate_source_layer"),
        "candidate_topk_blocks": g("attention.indexer.candidate_top_k_blocks"),
        "candidate_block_size": g("attention.indexer.candidate_block_size"),
        "rope_scaling.original_max_position_embeddings": g("rope.scaling.original_context_length"),
        "rope_scaling.factor": g("rope.scaling.factor"),
        "rope_scaling.beta_fast": g("rope.scaling.yarn_beta_fast"),
        "rope_scaling.beta_slow": g("rope.scaling.yarn_beta_slow"),
        "eos_token_id": kv.get("tokenizer.ggml.eos_token_id", 1),
        "rms_norm_eps": g("attention.layer_norm_rms_epsilon"),
        "routed_scaling_factor": g("expert_weights_scale"),
        "swiglu_limit": (g("swiglu_clamp_exp") or [0.0])[0],
        "rope_theta": g("rope.freq_base"),
        "compress_rope_theta": g("attention.compress_rope_freq_base"),
        "compress_ratios": ratios,
        "kv_source_layer_ids": list(g("attention.kv_source_layers", [])),
        "index_source_layer_ids": list(g("attention.indexer.source_layers", [])),
        "engram_layer_ids": list(g("engram.layers", [])),
        "max_position_embeddings": g("context_length"),
        "dspark_block_size": g("dspark.block_size", 0),
        "dspark_noise_token_id": g("dspark.noise_token_id", 0),
        "dspark_markov_rank": g("dspark.markov_rank", 0),
        "dspark_n_routed_experts": g("dspark.expert_count", 0),
        "dspark_num_experts_per_tok": g("dspark.expert_used_count", 0),
        "dspark_target_layer_ids": list(g("dspark.target_layers", [])),
    }
    clamp = g("swiglu_clamp_exp") or []
    if clamp and any(abs(c - clamp[0]) > 1e-6 for c in clamp):
        raise SystemExit("swiglu_clamp_exp differs between layers %s: the engine takes one swiglu_limit" % clamp[:8])
    return out


def ple_to_engram(kv: dict):
    """llama.cpp's own names for the Engram metadata (unsloth's GGUFs from 2026-10-19 on: ple.*, the hash heads' offsets
    counted across the layers' tables) -> the engram.* keys this pack reads (each layer's offsets from its own table)."""
    a = kv.get("general.architecture")
    g = lambda k, d=None: kv.get("%s.%s" % (a, k), d)  # noqa: E731
    if g("engram.layers") or not g("ple.layers"):
        return
    layers, ngram, heads = list(g("ple.layers")), g("ple.ngram_size"), g("ple.heads_per_ngram")
    offs, sizes = list(g("ple.head_offsets")), list(g("ple.head_vocab_sizes"))
    cols = (ngram - 1) * heads
    if len(offs) != len(layers) * cols or len(sizes) != len(offs):
        raise SystemExit("ple: %d head offsets, %d sizes for %d layers x %d columns" %
                         (len(offs), len(sizes), len(layers), cols))
    per, rows = [], []
    for i in range(len(layers)):
        o, s = offs[i * cols:(i + 1) * cols], sizes[i * cols:(i + 1) * cols]
        per += [x - o[0] for x in o]
        rows.append(o[-1] + s[-1] - o[0])
    for k, v in (("layers", layers), ("max_ngram_size", ngram), ("head_count", heads),
                 ("head_dim", g("embedding_length_per_layer_input")), ("multipliers", list(g("ple.layer_multipliers"))),
                 ("primes", sizes), ("offsets", per), ("num_embeddings", rows)):
        kv["%s.engram.%s" % (a, k)] = v


def split_ple(g: Gguf, kv: dict):
    """llama.cpp's one Engram tensor (per_layer_token_embd: every layer's table, one after the other) -> each layer's
    own blk.N.engram_embd as the earlier GGUFs had it: the same bytes, read in place."""
    a = kv["general.architecture"]
    ple = [t for t in g.tensors if t[0] == "per_layer_token_embd.weight"]
    if not ple:
        return
    name, typ, dims, off = ple[0]
    layers, rows = kv.get("%s.engram.layers" % a, []), kv.get("%s.engram.num_embeddings" % a, [])
    if len(dims) != 2 or typ not in TYPES or len(rows) != len(layers) or sum(rows) != dims[1]:
        raise SystemExit("%s: %s %s does not split in the Engram layers %s of %s rows" % (g.path.name, name, dims, layers,
                                                                                         rows))
    _, per, bpb = TYPES[typ]
    row = dims[0] // per * bpb
    parts, start = [], 0
    for il, n in zip(layers, rows):
        parts.append(("blk.%d.engram_embd.weight" % il, typ, [dims[0], n], off + start * row))
        start += n
    g.tensors = [t for t in g.tensors if t[0] != name] + parts


def short(g: Gguf):
    """Why a GGUF file is shorter than its own tensor directory says (a download still running), or None."""
    need = g.data_start
    for _, typ, dims, off in g.tensors:
        if typ in TYPES:
            _, per, bpb = TYPES[typ]
            nel = 1
            for d in dims:
                nel *= d
            need = max(need, g.data_start + off + nel // per * bpb)
    have = g.path.stat().st_size
    return None if have >= need else "%s: %d of %d bytes" % (g.path.name, have, need)


def engram_set(model: pathlib.Path, explicit: str | None):
    """The Engram tables' GGUF set for a quant that leaves them out: (its folder, its files read), or (None, []).
    explicit (--engram): that folder, whole; else the first whole one of ENGRAM_SETS beside the quant or in its
    folder - one not whole yet (its download still running) is passed over."""
    dirs = ([pathlib.Path(explicit).expanduser().resolve()] if explicit else
            [d for n in ENGRAM_SETS for d in (model / n, model.parent / n)])
    for d in dirs:
        paths = sorted(d.glob("*.gguf")) if d.is_dir() else []
        if not paths:
            if explicit:
                raise SystemExit("%s: no .gguf files (--engram: the folder of the Engram tables' GGUF set)" % d)
            continue
        try:
            egs = [Gguf(p) for p in paths]
            why = next((w for w in map(short, egs) if w), None)
        except (Exception, SystemExit) as e:   # noqa: BLE001 - a header cut short raises anything
            why = str(e) or type(e).__name__
        if why is None:
            return d, egs
        if explicit:
            raise SystemExit("%s: not whole (%s)" % (d, why))
        print("%s: not whole (%s) - not used" % (d, why))
    return None, []


def write_engram(out: pathlib.Path, kv: dict, vocab: int):
    a = kv["general.architecture"]
    g = lambda k, d=None: kv.get("%s.%s" % (a, k), d)  # noqa: E731
    layers = list(g("engram.layers", []))
    if not layers:
        return 0, 0
    ngram, heads, hd = g("engram.max_ngram_size"), g("engram.head_count"), g("engram.head_dim")
    tm = list(g("engram.token_map"))
    mult, primes, offs = list(g("engram.multipliers")), list(g("engram.primes")), list(g("engram.offsets"))
    cols = (ngram - 1) * heads
    nl = len(layers)
    if len(mult) != nl * ngram or len(primes) != nl * cols or len(offs) != nl * cols:
        raise SystemExit("engram: %d multipliers, %d primes, %d offsets for %d layers x %d columns (ngram %d)" %
                         (len(mult), len(primes), len(offs), nl, cols, ngram))
    pad = int(tm[g("engram.pad_token_id")])
    full = tm + [0] * max(0, vocab - len(tm))
    with out.open("wb") as f:
        f.write(struct.pack("<4s6i", b"DSE1", vocab, nl, ngram, heads, hd, pad))
        f.write(struct.pack("<%di" % vocab, *full[:vocab]))
        f.write(struct.pack("<%dq" % len(mult), *mult))
        f.write(struct.pack("<%dq" % len(primes), *primes))
        f.write(struct.pack("<%dq" % len(offs), *offs))
    return nl, cols


def write_tokenizer(out: pathlib.Path, kv: dict, template: str | None):
    tokens = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type") or [1] * len(tokens)
    merges = kv.get("tokenizer.ggml.merges") or []
    if len(set(tokens)) != len(tokens):
        raise SystemExit("the GGUF's vocabulary has duplicate token strings")
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps([int(t) for t in types]))
    special = {"bos": int(kv.get("tokenizer.ggml.bos_token_id", 0)), "eos": int(kv.get("tokenizer.ggml.eos_token_id", 1)),
               "pad": int(kv.get("tokenizer.ggml.padding_token_id", 2))}
    (out / "tokenizer.json").write_text(json.dumps({"pre": "deepseek3", "special_ids": special}, indent=1))
    if template:
        (out / "chat_template.jinja").write_text(template, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the directory of the GGUF's shards")
    ap.add_argument("--out", help="default: <model>/maya-ds4")
    ap.add_argument("--vision", help="vision_pack.py's folder (default: <model>/../vision; taken when it is there)")
    ap.add_argument("--dspark-gguf", help="unsloth's DSpark drafter GGUF (dspark-*-Q8_0.gguf, architecture dflash): its three "
                                          "blocks become mtp.0-2 beside the trunk's tensors (default: the first dspark-*.gguf "
                                          "in <model>/.. when there is one)")
    ap.add_argument("--mmproj", help="unsloth's vision tower GGUF (mmproj-*.gguf, clip / deepseek41v): its tensors become the "
                                     "checkpoint's vision.* / aligner.* (default: the first mmproj*.gguf in <model>/..)")
    ap.add_argument("--tq-experts", nargs="+", help="folders tq_convert.py wrote (tq.json + tq-L<layer>.bin): the routed experts of the layers "
                                         "listed there are read from it as trellis types (TQK5X ..), the others keep their MXFP4")
    ap.add_argument("--dense-override", help="a folder dense_requant.py wrote (dense.json + dense.bin): its tensors replace the same-named "
                                             "dense matrices of the GGUF (K-quants of the attention / shared experts)")
    ap.add_argument("--override-experts", help="a second requantized GGUF folder (requant_gguf.py output): its routed experts replace the main "
                                               "model's for --override-roles (gate,up,down) in --override-layers (default all): mixed-bit packs")
    ap.add_argument("--override-roles", default="down")
    ap.add_argument("--override-layers", default="")
    ap.add_argument("--tq-layers", help="only these layers (comma list) of the --tq-experts folders")
    ap.add_argument("--drafter", help="the DSpark drafter's .json (drafter_pack.py's; default <model>/../dspark/dspark.json)")
    ap.add_argument("--engram", help="the folder of the Engram tables' GGUF set, for a quant that leaves them out "
                                     "(default: <model>/../Engram, when it is whole)")
    ap.add_argument("--shards", nargs="+", help="the GGUF's files in --model (default: every .gguf there - one "
                                                "model's; the setup names them: a folder may hold an earlier layout)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser() if a.out else model / "maya-ds4"
    shards = [model / n for n in a.shards] if a.shards else sorted(model.glob("*.gguf"))
    if not shards or not all(p.is_file() for p in shards):
        raise SystemExit("%s: no .gguf files" % model if not shards else
                         "%s: not there" % next(p for p in shards if not p.is_file()))
    gs = [Gguf(p) for p in shards]
    kv = next((g.kv for g in gs if "general.architecture" in g.kv), None)
    if kv is None or kv["general.architecture"] not in ("deepseek41",):
        raise SystemExit("%s: not a DeepSeek-V4.1 GGUF (architecture %r)" % (model, kv and kv.get("general.architecture")))
    split = int(kv.get("split.count", 1))
    if split != len(shards):
        counts = sorted({m.group(1) for m in (re.search(r"-of-(\d{5})\.gguf$", p.name) for p in shards) if m})
        raise SystemExit("%s: %d of the %d shards are here" % (model, len(shards), split) if len(counts) < 2 else
                         "%s: the shards of more than one GGUF split (-of-%s): name the model's own with --shards" %
                         (model, ", -of-".join(counts)))
    ple_to_engram(kv)
    for g in gs:
        split_ple(g, kv)
    # the Engram tables may come from a shared set of GGUF files beside the quant (the same tensor names: engram_set) -
    # its engram tensors fill in the ones the quant's shards leave out, its engram metadata too.  v1.1.16's Maya-S:
    # unsloth's UD-Q8_K_XL shard that holds only the tables, as it is (00008 of 9; before 2026-10-19 shards 00008 and
    # 00009 of 10, a table each), so no shard count is checked - every Engram layer's tables must be found, below.
    # Looked for only when the quant's shards leave some out.
    have = {t[0] for g in gs for t in g.tensors}
    arch = kv["general.architecture"]
    layers = kv.get("%s.engram.layers" % arch, [])
    lacking = not layers or any("blk.%d.%s.weight" % (il, part) not in have
                                for il in layers for part in ("engram_embd", "engram_kv", "engram_gate"))
    eg_dir, egs = engram_set(model, a.engram) if lacking or a.engram else (None, [])
    eg_from = {}
    if eg_dir is not None:
        eshards = [g.path for g in egs]
        for g in egs:
            for k, v in g.kv.items():
                if ".engram." in k or ".ple." in k or k.endswith(".embedding_length_per_layer_input"):
                    kv.setdefault(k, v)
        ple_to_engram(kv)
        for g in egs:
            split_ple(g, kv)
        for p, g in zip(eshards, egs):
            picked = [t for t in g.tensors if re.match(r"blk\.\d+\.engram_", t[0]) and t[0] not in have]
            if picked:
                g.tensors = picked
                eg_from[p] = g
    for il in kv.get("%s.engram.layers" % arch, []):
        for part in ("engram_embd", "engram_kv", "engram_gate"):
            nm = "blk.%d.%s.weight" % (il, part)
            if nm not in have and not any(t[0] == nm for g in eg_from.values() for t in g.tensors):
                raise SystemExit("%s: missing - not in the quant's shards nor in an Engram set (--engram, or "
                                 "<model>/../Engram/: unsloth's UD-Q8_K_XL shard of the Engram tables)" % nm)
    rel = lambda p: os.path.relpath(p, model).replace(os.sep, "/")  # noqa: E731
    gs_all = gs + list(eg_from.values())
    names = [g.path.name for g in gs] + [rel(p) for p in eg_from]

    tensors, unknown = [], []
    for fi, g in enumerate(gs_all):
        for name, typ, dims, off in g.tensors:
            if typ not in TYPES:
                raise SystemExit("%s: %s has ggml type %d, which the engine does not read" % (g.path.name, name, typ))
            tname, per, bpb = TYPES[typ]
            nel = 1
            for d in dims:
                nel *= d
            if dims[0] % per:
                raise SystemExit("%s: %s rows of %d are not whole %s blocks" % (g.path.name, name, dims[0], tname))
            nbytes = nel // per * bpb
            m = re.match(r"blk\.(\d+)\.(.+)$", name)
            if m and m.group(2) in LAYER:
                ck = "layers.%s.%s" % (m.group(1), LAYER[m.group(2)])
            elif name in GLOBAL:
                ck = GLOBAL[name]
            else:
                unknown.append(name)
                continue
            shape = list(reversed(dims))     # ggml ne0 innermost -> torch order, outermost first
            base = g.data_start + off
            if ck.endswith(".attn.wo_a.slice"):
                # [o_groups * o_lora][in] -> o_groups entries of [o_lora][in] (group g: rows g * o_lora ..)
                n = int(kv["%s.attention.output_group_count" % kv["general.architecture"]])
                if len(shape) != 2 or shape[0] % n:
                    raise SystemExit("%s: %s %s does not split in %d groups" % (g.path.name, name, shape, n))
                part = nbytes // n
                for i in range(n):
                    tensors.append(("%s.%d.weight" % (ck, i), tname, fi, base + i * part, part,
                                    [shape[0] // n, shape[1]]))
                continue
            if ".ffn.experts." in ck:
                # [n_exp][out][in] -> an entry an expert: layers.N.ffn.experts.E.w1.weight
                n, part = shape[0], nbytes // shape[0]
                pre, role = ck.rsplit(".", 1)
                for i in range(n):
                    tensors.append(("%s.%d.%s.weight" % (pre, i, role), tname, fi, base + i * part, part, shape[1:]))
                continue
            tensors.append((ck, tname, fi, base, nbytes, shape))
    if unknown:
        raise SystemExit("tensors the pack does not map (%d): %s" % (len(unknown), ", ".join(unknown[:12])))
    # the DSpark drafter (drafter_pack.py's dspark.bin beside the quants' folders, from DeepSeek's checkpoint): its
    # three blocks as mtp.0-2, so the engine drafts with them
    files = names
    if eg_from:
        print("Engram tables from %s (%d shards)" % (eg_dir, len(eg_from)))
    # experts of a second requantized model replace the main one's for some roles / layers (the byte layouts are per tensor: any mix of types)
    n_ov = 0
    if a.override_experts:
        od = pathlib.Path(a.override_experts).expanduser().resolve()
        roles_ov = {"gate": "w1", "up": "w3", "down": "w2"}
        want = {roles_ov[r] for r in a.override_roles.split(",")}
        lay_ov = None
        if a.override_layers:
            lay_ov = set()
            for part in a.override_layers.split(","):
                lo, _, hi = part.partition("-")
                lay_ov.update(range(int(lo), int(hi or lo) + 1))
        ogs = [Gguf(pth) for pth in sorted(od.glob("*.gguf"))]
        new = []
        for og in ogs:
            try:
                orel = os.path.relpath(og.path, model).replace(os.sep, "/")
            except ValueError:
                raise SystemExit("%s must be on the model's drive" % od)
            files.append(orel)
            ofi = len(files) - 1
            for name, typ, dims, off in og.tensors:
                m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
                if not m or typ not in TYPES:
                    continue
                il, role = int(m.group(1)), {"gate": "w1", "up": "w3", "down": "w2"}[m.group(2)]
                if role not in want or (lay_ov is not None and il not in lay_ov):
                    continue
                tname, per, bpb = TYPES[typ]
                nel = 1
                for d in dims:
                    nel *= d
                nbytes = nel // per * bpb
                shape = list(reversed(dims))
                n, part = shape[0], nbytes // shape[0]
                for i in range(n):
                    new.append(("layers.%d.ffn.experts.%d.%s.weight" % (il, i, role), tname, ofi, og.data_start + off + i * part, part, shape[1:]))
        names_new = {t[0] for t in new}
        tensors[:] = [t for t in tensors if t[0] not in names_new] + new
        n_ov = len(new)
        print("experts overridden from %s: %d tensors (roles %s, layers %s)" % (od, n_ov, a.override_roles, a.override_layers or "all"))
    # dense matrices requantized by dense_requant.py replace the GGUF's BF16 ones
    n_dense = 0
    if a.dense_override:
        dd = pathlib.Path(a.dense_override).expanduser().resolve()
        dj = json.loads((dd / "dense.json").read_text())
        try:
            relf = os.path.relpath(dd / "dense.bin", model).replace(os.sep, "/")
        except ValueError:
            raise SystemExit("%s must be on the model's drive" % dd)
        if relf not in files:
            files.append(relf)
        dfi = files.index(relf)
        pos = {t[0]: i for i, t in enumerate(tensors)}
        for name, (tname, fname, off, nb, shape) in dj.items():
            if name not in pos:
                raise SystemExit("dense override: %s is not in the pack" % name)
            tensors[pos[name]] = (name, tname, dfi, off, nb, shape)
            n_dense += 1
        print("dense override from %s: %d tensors" % (dd, n_dense))
    # the DSpark drafter from unsloth's GGUF (architecture dflash): blk.N.* -> mtp.N.*, the head tensors under the
    # checkpoint's names (fc -> main_proj, enc.output_norm -> main_norm, output_norm -> norm of the last block,
    # conf_proj -> confidence_head.proj, markov_w1 / w2 -> markov_head.embed / head)
    ds_path = pathlib.Path(a.dspark_gguf).expanduser().resolve() if a.dspark_gguf else next(
        iter(sorted(model.parent.glob("dspark-*.gguf"))), None)
    dspark_gguf = False
    if ds_path is not None:
        if not ds_path.is_file():
            raise SystemExit("%s: no DSpark GGUF there" % ds_path)
        dg = Gguf(ds_path)
        if dg.kv.get("general.architecture") != "dflash":
            raise SystemExit("%s: not a DSpark (dflash) GGUF" % ds_path)
        nb = int(dg.kv["dflash.block_count"])
        files.append(rel(ds_path))
        fi = len(files) - 1
        HEAD = {"fc.weight": "mtp.0.main_proj.weight", "enc.output_norm.weight": "mtp.0.main_norm.weight",
                "output_norm.weight": "mtp.%d.norm.weight" % (nb - 1),
                "conf_proj.weight": "mtp.%d.confidence_head.proj.weight" % (nb - 1),
                "markov_w1.weight": "mtp.%d.markov_head.embed.weight" % (nb - 1),
                "markov_w2.weight": "mtp.%d.markov_head.head.weight" % (nb - 1)}
        groups = int(kv["%s.attention.output_group_count" % kv["general.architecture"]])
        bad = []
        for name, typ, dims, off in dg.tensors:
            if typ not in TYPES:
                raise SystemExit("%s: %s has ggml type %d, which the engine does not read" % (ds_path.name, name, typ))
            tname, per, bpb = TYPES[typ]
            nel = 1
            for d in dims:
                nel *= d
            nbytes = nel // per * bpb
            m = re.match(r"blk\.(\d+)\.(.+)$", name)
            if m and m.group(2) in LAYER:
                ck = "mtp.%s.%s" % (m.group(1), LAYER[m.group(2)])
            elif name in HEAD:
                ck = HEAD[name]
            else:
                bad.append(name)
                continue
            shape = list(reversed(dims))
            base = dg.data_start + off
            if ck.endswith(".attn.wo_a.slice"):
                part = nbytes // groups
                for i in range(groups):
                    tensors.append(("%s.%d.weight" % (ck, i), tname, fi, base + i * part, part, [shape[0] // groups, shape[1]]))
            elif ".ffn.experts." in ck:
                n, part = shape[0], nbytes // shape[0]
                pre, role = ck.rsplit(".", 1)
                for i in range(n):
                    tensors.append(("%s.%d.%s.weight" % (pre, i, role), tname, fi, base + i * part, part, shape[1:]))
            else:
                tensors.append((ck, tname, fi, base, nbytes, shape))
        if bad:
            raise SystemExit("DSpark GGUF tensors the pack does not map (%d): %s" % (len(bad), ", ".join(bad[:12])))
        dspark_gguf = True
        print("DSpark drafter from %s: %d blocks" % (ds_path.name, nb))
    # the vision tower from unsloth's mmproj GGUF (clip, projector deepseek41v): llama.cpp's names -> the checkpoint's;
    # the F32 norms / biases / marker rows are read as BF16 by the engine; mlp.w1 is the file's gate and up
    mm_path = pathlib.Path(a.mmproj).expanduser().resolve() if a.mmproj else next(
        iter(sorted(model.parent.glob("mmproj*.gguf"))), None)
    mm_geo = {}
    if mm_path is not None:
        if not mm_path.is_file():
            raise SystemExit("%s: no mmproj GGUF there" % mm_path)
        mg = Gguf(mm_path)
        if mg.kv.get("clip.projector_type") != "deepseek41v":
            raise SystemExit("%s: not a DeepSeek-V4.1 vision GGUF (projector %r)" % (mm_path, mg.kv.get("clip.projector_type")))
        files.append(rel(mm_path))
        fi = len(files) - 1
        FIX = {"v.patch_embd.weight": "vision.patch_embed.proj.weight", "v.patch_embd.bias": "vision.patch_embed.proj.bias",
               "v.post_ln.weight": "vision.norm.weight", "mm.1.weight": "aligner.w1.weight", "mm.1.bias": "aligner.w1.bias",
               "mm.2.weight": "aligner.w2.weight", "mm.2.bias": "aligner.w2.bias", "v.token_embd.img_start": "image_start",
               "v.image_newline": "image_newline", "v.token_embd.img_end": "image_end"}
        BLK = {"ln1.weight": "norm1.weight", "ln2.weight": "norm2.weight", "attn_qkv.weight": "attn.wqkv.weight",
               "attn_qkv.bias": "attn.wqkv.bias", "attn_out.weight": "attn.wo.weight", "attn_out.bias": "attn.wo.bias",
               "ffn_gate.weight": "mlp.gate.weight", "ffn_up.weight": "mlp.up.weight", "ffn_down.weight": "mlp.w2.weight"}
        bad = []
        for name, typ, dims, off in mg.tensors:
            if typ not in TYPES:
                raise SystemExit("%s: %s has ggml type %d, which the engine does not read" % (mm_path.name, name, typ))
            tname, per, bpb = TYPES[typ]
            nel = 1
            for d in dims:
                nel *= d
            m = re.match(r"v\.blk\.(\d+)\.(.+)$", name)
            if m and m.group(2) in BLK:
                ck = "vision.blocks.%s.%s" % (m.group(1), BLK[m.group(2)])
            elif name in FIX:
                ck = FIX[name]
            else:
                bad.append(name)
                continue
            shape = list(reversed(dims))
            if name == "v.patch_embd.weight":    # conv [out][3][14][14] -> [out][3 * 14 * 14], the same bytes
                shape = [shape[0], nel // shape[0]]
            tensors.append((ck, tname, fi, mg.data_start + off, nel // per * bpb, shape))
        if bad:
            raise SystemExit("mmproj tensors the pack does not map (%d): %s" % (len(bad), ", ".join(bad[:12])))
        k = lambda n, d=None: mg.kv.get("clip.vision." + n, d)  # noqa: E731
        toks = kv.get("tokenizer.ggml.tokens", [])
        img_id = next((i for i, t in enumerate(toks) if t == "<｜deepseek_image｜>"), None)
        if img_id is None:
            raise SystemExit("the vocabulary has no <｜deepseek_image｜> token")
        mm_geo = {"vision.num_hidden_layers": k("block_count"), "vision.hidden_size": k("embedding_length"),
                  "vision.num_attention_heads": k("attention.head_count"), "vision.intermediate_size": k("feed_forward_length"),
                  "vision.patch_size": k("patch_size"), "vision.downsample_ratio": k("projector.scale_factor"),
                  "vision.rope_theta": 10000, "vision.max_image_tokens": 1024, "vision.min_pixels": k("image_min_pixels"),
                  "image_token_id": img_id}
        print("vision tower from %s: %d blocks, image token %d" % (mm_path.name, mm_geo["vision.num_hidden_layers"], img_id))
    # trellis experts from tq_convert.py: per layer the three roles stacked [n_exp][rows][row bytes] in one file
    tq_layers = []
    if a.tq_experts:
        by_layer, tq_dirs = {}, {}
        for d in a.tq_experts:
            tqd = pathlib.Path(d).expanduser().resolve()
            for key, ent in json.loads((tqd / "tq.json").read_text()).items():
                m = re.match(r"layers\.(\d+)\.ffn\.experts\.(w[123])$", key)
                if not m:
                    raise SystemExit("tq.json: %s is not a layer's expert role" % key)
                by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = ent
                tq_dirs[(int(m.group(1)), m.group(2))] = tqd
        if a.tq_layers:
            keep = {int(x) for x in a.tq_layers.split(",")}
            by_layer = {il: r for il, r in by_layer.items() if il in keep}
        TQTYPES = {v[0]: k for k, v in TYPES.items()}
        for il, roles in sorted(by_layer.items()):
            if sorted(roles) != ["w1", "w2", "w3"]:
                raise SystemExit("tq.json: layer %d lacks a role (has %s)" % (il, sorted(roles)))
            if len({r[0] for r in roles.values()}) != 1 or roles["w1"][0] != roles["w3"][0]:
                raise SystemExit("tq.json: layer %d: gate and up must be one type" % il)
            drop = re.compile(r"layers\.%d\.ffn\.experts\.\d+\.w[123]\.weight$" % il)
            kept = [t for t in tensors if not drop.match(t[0])]
            if len(kept) == len(tensors):
                raise SystemExit("layer %d has no routed experts in the GGUF to replace" % il)
            tensors[:] = kept
            for role, (tname, fname, off, total, shape) in roles.items():
                if tname not in TQTYPES:
                    raise SystemExit("tq.json: unknown type %s" % tname)
                n_exp, rows, cols = shape
                per = total // n_exp
                f = tq_dirs[(il, role)] / fname
                if not f.is_file() or f.stat().st_size < off + total:
                    raise SystemExit("%s: missing or short" % f)
                try:
                    relf = os.path.relpath(f, model).replace(os.sep, "/")
                except ValueError:
                    raise SystemExit("%s must be on the model's drive" % f)
                if relf not in files:
                    files.append(relf)
                fi = files.index(relf)
                for e in range(n_exp):
                    tensors.append(("layers.%d.ffn.experts.%d.%s.weight" % (il, e, role), tname, fi, off + e * per, per, [rows, cols]))
            tq_layers.append(il)
        print("trellis experts from %s: layers %s" % (", ".join(a.tq_experts), tq_layers))
    drafter = pathlib.Path(a.drafter).expanduser().resolve() if a.drafter else model.parent / "dspark" / "dspark.json"
    if a.drafter and not drafter.is_file():
        raise SystemExit("%s: no drafter there" % drafter)
    drafter_in = (drafter.is_file() and drafter.with_suffix(".bin").is_file()) and not dspark_gguf
    if drafter_in:
        files.append(rel(drafter.with_suffix(".bin")))
        groups = int(kv["%s.attention.output_group_count" % kv["general.architecture"]])
        for name, dt, off, nb, shape in json.loads(drafter.read_text())["tensors"]:
            if name.endswith(".attn.wo_a.weight"):   # (one entry a group, as the trunk's)
                for i in range(groups):
                    tensors.append(("%s.slice.%d.weight" % (name[:-len(".weight")], i), dt, len(files) - 1,
                                    off + i * (nb // groups), nb // groups, [shape[0] // groups, shape[1]]))
                continue
            tensors.append((name, dt, len(files) - 1, off, nb, shape))
    # the vision tower (vision_pack.py's vision.bin, from DeepSeek's checkpoint): its tensors under the checkpoint's
    # names, the picture span's marker rows, and the geometry config.json gives them
    vdir = pathlib.Path(a.vision).expanduser().resolve() if a.vision else model.parent / "vision"
    vision = {}
    if (vdir / "vision.json").is_file() and (vdir / "vision.bin").is_file():
        try:
            rel = os.path.relpath(vdir / "vision.bin", model).replace(os.sep, "/")
        except ValueError:     # (Windows: another drive)
            raise SystemExit("%s: the vision folder must be on the model's drive" % vdir)
        if " " in rel:
            raise SystemExit("%s: the vision folder's path relative to the model has a space" % vdir)
        files.append(rel)
        info = json.loads((vdir / "vision.json").read_text())
        for name, dt, off, nb, shape in info["tensors"]:
            tensors.append((name, dt, len(files) - 1, off, nb, shape))
        vision = info["config"]
    if mm_geo and not vision:
        vision = mm_geo
    vocab = next(t[5][0] for t in tensors if t[0] == "embed.weight")
    geo = geometry(kv, vocab)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "ds4.txt").open("w") as f:
        for k, v in geo.items():
            if v is None:
                continue
            if isinstance(v, bool):
                v = int(v)
            f.write("%s %s\n" % (k, ",".join(str(x) for x in v) if isinstance(v, list) else v))
        for k, v in vision.items():
            f.write("%s %s\n" % (k, v))
        f.write("format gguf\n")
        f.write("model_dir %s\n" % model)
    with (out / "tensors.txt").open("w") as f:
        for i, name in enumerate(files):
            f.write("file %d %s\n" % (i, name))
        for name, dt, fi, off, nb, shape in tensors:
            f.write("t %s %s %d %d %d %s\n" % (name, dt, fi, off, nb, " ".join(map(str, shape))))
    nl, cols = write_engram(out / "engram.bin", kv, vocab)
    # the server's chat template: the GGUF's own (unsloth's, byte-identical to DeepSeek's encoding.py on the 9 cases we
    # checked: chat, thinking, effort, system, multi-turn, tools, tool results).  Project Maya's tools/ds4/chat_template.jinja
    # lost the tool associations and later system instructions (their #299), so it is NOT used here.
    template = kv.get("tokenizer.chat_template")
    if not template:
        sys.exit("%s: the GGUF has no tokenizer.chat_template; refusing to fall back to the old Maya template (#299)" % model)
    write_tokenizer(out / "tokenizer", kv, template)
    by = {}
    for t in tensors:
        by[t[1]] = by.get(t[1], 0) + t[4]
    extras = [w for w, there in (("the DSpark drafter", drafter_in or dspark_gguf), ("the vision tower", bool(vision)), ("trellis experts in layers %s" % tq_layers, bool(tq_layers)), ("%d dense matrices requantized" % n_dense, bool(n_dense)), ("%d experts from %s" % (n_ov, a.override_experts), bool(n_ov))) if there]
    print("%s: %d tensors in %d files (%s), Engram %d layers x %d hash columns%s -> %s" %
          (model.name, len(tensors), len(files), ", ".join("%s %.1f GB" % (k, v / 1e9) for k, v in sorted(by.items())),
           nl, cols, "".join(", " + w for w in extras), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
