// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/strata_image.cpp - strata-image: Qwen-Image-2.1, Mage-Flow or HunyuanImage 3.0 text to image on the GPU.
//
//   strata-image --dit DIT.gguf --vae VAE.safetensors --text TEXT_ENCODER.gguf [--vision MMPROJ.gguf]
//                [--budget-mib M] (--ids 1,2,3 [--drop N] [--width W] [--height H] [--seed S] [--sigmas 1,...,0]
//                [--ref IMAGE]... [--mask IMAGE] [--init IMAGE] [--strength F] [--repeat N] --out IMAGE.rgba | --serve)
//   strata-image --dit DIT.safetensors ... [--dit-q8 auto|0|1] [--edit-dit EDIT_DIT.safetensors --vision MMPROJ.gguf]
//                [--dit-keep 0|1]
//
// The DiT's file says the model: a GGUF is Qwen-Image-2.1's, a safetensors file Mage-Flow's (its bf16 weights as
// Comfy-Org has them; its blocks FP16 on the GPU when they fit the budget with the text encoder's layers, else
// quantized to 8 bits as they are read (MfDit: W8A8 or Q8_0): --dit-q8 auto, the default; 1 always 8 bits, 0 always
// FP16).  A safetensors file whose header is a HunyuanImage 3.0 save - its `format`, or the `comfy_quant` beside
// its matrices, or the model's own `model.wte.weight` beside `patch_embed` - is HunyuanImage 3.0's, and takes no
// --text: its DiT reads the ids itself (`wte`, with three rows its embedders write over), so its entry is
//   strata-image --dit QUANTIZED.safetensors --vae VAE.safetensors [--vision VISION.safetensors] (--ids ... |
//                --serve) [--out F] [--width W] [--height H] [--seed S] [--steps N] [--sigmas 1,...,0]
//                [--guidance G] [--ref IMAGE]... [--budget-mib M]
// and its guidance is a token's value (`--guidance`, the model's own 2.5 unset) rather than a second pass.  Its
// edits (--ref, REF:) need --vision, the model's bf16 shard with its `vision_model.*` tensors (SigLIP2): the ids
// then carry each reference's block before the prompt (hunyuanimage3_pipeline.hpp).  Mage-Flow's edits take
// its edit DiT (--edit-dit) and Qwen3-VL-4B's mmproj (--vision), up to three --ref; --dit-keep 1, the default, keeps the
// DiT not in use in RAM between a generation and an edit (0: read from its file again).
//
// One image: --ids, the prompt in the model's chat form, tokenized (the server's tokenizer); --drop, the leading ids of
// the system turn; --budget-mib, the VRAM the engine may take (default: the free VRAM); --repeat, the image again N
// times in the process (the cache's effect); --sigmas, the sampler's sigmas from 1 down to the terminal 0, one step
// between each two (default Turbo's 8 steps; serve/images.py's base_sigmas for the base model); --ref, an edit's
// reference image (PNG, JPEG, BMP, GIF; its pad in --ids is one <|image_pad|>, which the engine widens to the image's
// tokens), with the vision encoder (--vision, Qwen3-VL's mmproj); --init, img2img's base image for a model without
// edits; --strength, how much of the schedule runs from the base image (diffusers' strength, above 0 and at most 1;
// 1, the default, starts from noise); --mask, a mask over the base image (the --init image, else the first --ref;
// OpenAI's: a PNG whose transparent pixels are where the image changes, the rest kept as the base has it); see
// img_start.hpp.
// Writes RGBA8 rows to --out and a line of times and of the budget's split a run.
//
// --serve: the engine stays, its layers and the last prompt's states kept between requests, and takes lines on stdin
// (serve/images.py), one at a time, answering each on stdout:
//   IMG <width> <height> <seed> <drop> <out path> <ids, commas> [SIGMAS:<sigmas, commas>] [REF:<path>]...
//                            -> STAGE <refs|text|sampling|decode> as each stage starts and STEP <done> <steps> lines,
//                            then IMAGE key=value ... (times, the budget's split); SIGMAS: as --sigmas; REF: an
//                            edit's reference images; MASK:, INIT: and STRENGTH: as --mask, --init and --strength;
//                            NEG:<ids, commas> and CFG:<scale> (above 1):
//                            classifier-free guidance with that negative prompt (its pads as the prompt's); GUID:
//                            <scale> is HunyuanImage 3.0's guidance token instead (it samples once: no NEG:, no CFG:);
//                            PREVIEW:<steps, commas>: partial images after those steps, each a line PREVIEW <index>
//                            <step> <path> before the next STEP, its RGBA8 rows at <out path>.p<index>
//   NEED <width> <height> <n> [REF:<path>]... [MASK:<path>] [INIT:<path>] [NEG:<n>]  the VRAM an image of n prompt
//                            ids needs (from the files); MASK: or INIT:, a base image (with a strength below 1 and
//                            references, MASK:<their first>); NEG: guidance's negative prompt of n ids
//                                                                                          -> NEED least_mib=...
//                            full_mib=... held_mib=... free_mib=... cap_mib=... host_mib=... (si::QiNeed)
//   FREE                     the free VRAM and what the engine holds                        -> FREE free_mib=... held_mib=...
//   BUDGET <MiB>             the VRAM the engine may take (0: the free VRAM at its start)      -> BUDGET budget_mib=...
//   FIT <MiB>                the budget: what the engine holds and the free VRAM, less <MiB>  -> BUDGET budget_mib=...
//   RELEASE [ALL]            frees the layers, activations and buffers (the cache); ALL also the parts' fixed weights
//                            and scratch, the parts opened again on the next IMG                -> RELEASE ok
//   STOP                     (while an IMG runs) ends it after the step it is in               -> ERROR stopped
//   QUIT                     ends the engine
// A request that fails answers ERROR <message> and the engine goes on.  Before the first line it makes a small image
// (its kernels built and the library's first products set up; skipped when the GPU has no room) and lets its buffers
// go, then says READY budget_mib=<M>.  A model whose sequence has a shape of its own (HunyuanImage 3.0, whose ids
// carry the image block) warms up with one of its own: the pipeline says so in warm_up_request.

#include "strata/core/gpu.hpp"
#include "strata/image/hunyuanimage3_file.hpp"
#include "strata/image/hunyuanimage3_pipeline.hpp"
#include "strata/image/qi_image_io.hpp"
#include "strata/image/mf_pipeline.hpp"
#include "strata/image/qi_pipeline.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

namespace si = strata::image;

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> ids;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');)
        if (!tok.empty()) ids.push_back((int32_t) std::stol(tok));
    return ids;
}

// the sampler's sigmas, commas: two or more, from at most 1 down to 0, falling
std::vector<float> parse_sigmas(const std::string& s) {
    std::vector<float> sig;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');)
        if (!tok.empty()) sig.push_back(std::stof(tok));
    bool ok = sig.size() >= 2 && sig.front() <= 1.0f && sig.back() == 0.0f;
    for (size_t i = 1; ok && i < sig.size(); ++i) ok = sig[i] < sig[i - 1];
    if (!ok) throw std::invalid_argument("sigmas: two or more, falling from at most 1 to 0: " + s);
    return sig;
}

si::QiRef read_image(const std::string& path) {
    si::QiRef f;
    std::string err;
    f.rgba = si::load_rgba8(path, f.width, f.height, err);
    if (f.rgba.empty()) throw std::invalid_argument(err);
    return f;
}

// the words left on an IMG line: its REF:<path> images (and SIGMAS:, MASK:, INIT:, STRENGTH:...)
std::vector<si::QiRef> read_refs(std::istringstream& ls, si::QiRequest& r) {
    std::vector<si::QiRef> refs;
    for (std::string word; ls >> word;) {
        if (word.rfind("SIGMAS:", 0) == 0) r.sigmas = parse_sigmas(word.substr(7));
        else if (word.rfind("MASK:", 0) == 0) r.mask = read_image(word.substr(5));
        else if (word.rfind("INIT:", 0) == 0) r.init = read_image(word.substr(5));
        else if (word.rfind("STRENGTH:", 0) == 0) r.strength = std::stod(word.substr(9));
        else if (word.rfind("NEG:", 0) == 0) r.neg_ids = parse_ids(word.substr(4));
        else if (word.rfind("CFG:", 0) == 0) r.cfg = std::stof(word.substr(4));
        else if (word.rfind("GUID:", 0) == 0) r.guidance = std::stof(word.substr(5));
        else if (word.rfind("STEPS:", 0) == 0) r.steps = std::stoll(word.substr(6));
        else if (word.rfind("PREVIEW:", 0) == 0)
            for (const int64_t k : parse_ids(word.substr(8))) r.preview.push_back((int) k);
        else if (word.rfind("REF:", 0) == 0) refs.push_back(read_image(word.substr(4)));
        else throw std::invalid_argument("unknown word " + word);
    }
    return refs;
}

// a NEED line's: the REF:<path> images' sizes; MASK:<path> or INIT:<path> (a base image) sets `mask`, NEG:<n> `n_neg`
std::vector<std::pair<int64_t, int64_t>> ref_sizes(std::istringstream& ls, bool& mask, int64_t& n_neg) {
    std::vector<std::pair<int64_t, int64_t>> sizes;
    for (std::string word; ls >> word;) {
        if (word.rfind("MASK:", 0) == 0 || word.rfind("INIT:", 0) == 0) {
            mask = true;
            continue;
        }
        if (word.rfind("NEG:", 0) == 0) {
            n_neg = std::stoll(word.substr(4));
            continue;
        }
        int64_t w = 0, h = 0;
        if (word.rfind("REF:", 0) != 0 || !si::image_size(word.substr(4), w, h))
            throw std::invalid_argument("not an image: " + word);
        sizes.emplace_back(w, h);
    }
    return sizes;
}

std::string plan_kv(const si::QiTimes& t, const si::QiPlan& p) {
    char b[640];
    std::snprintf(b, sizeof b,
                  "text_s=%.3f sampling_s=%.3f decode_s=%.3f budget_mib=%llu blocks_mib=%llu resident_blocks=%d "
                  "layers_mib=%llu resident_layers=%d activations_mib=%llu mlp_rows=%lld vae_mib=%llu vae_tiled=%d "
                  "text_cached=%d prefix_kv_mib=%llu resident_kv=%d streamed_mib=%llu",
                  t.text, t.sampling, t.decode, (unsigned long long) (p.budget >> 20),
                  (unsigned long long) (p.blocks >> 20), p.resident_blocks, (unsigned long long) (p.layers >> 20),
                  p.resident_layers, (unsigned long long) (p.activations >> 20), (long long) p.mlp_rows,
                  (unsigned long long) (p.vae >> 20), p.vae_tiled ? 1 : 0, p.text_cached ? 1 : 0,
                  (unsigned long long) (p.prefix_kv >> 20), p.resident_kv,
                  (unsigned long long) (p.streamed >> 20));
    return b;
}

void write_file(const std::string& path, const std::vector<uint8_t>& img) {
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(img.data()), (std::streamsize) img.size());
    if (!o) throw std::runtime_error("cannot write " + path);
}

// stdin's lines on a thread of their own, so that STOP reaches a running request
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false};
    void run() {
        for (std::string line; std::getline(std::cin, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "STOP") {
                stop = true;
                continue;
            }
            std::lock_guard<std::mutex> l(mu);
            q.push_back(line);
            cv.notify_all();
        }
        std::lock_guard<std::mutex> l(mu);
        eof = true;
        cv.notify_all();
    }
    bool next(std::string& line) {
        std::unique_lock<std::mutex> l(mu);
        cv.wait(l, [this] { return eof || !q.empty(); });
        if (q.empty()) return false;
        line = q.front();
        q.pop_front();
        return true;
    }
};

int serve(si::ImagePipeline& p) {
    {
        // the warm-up: a 256 x 256 image of one step from token ids the vocabulary has; where the GPU has no room
        // for it now (another program holds it), the first request builds the kernels instead.  A model whose
        // sequence has a shape of its own asks for its own request instead (HunyuanImage 3.0's ids carry the block).
        si::QiRequest r;
        if (!p.warm_up_request(r)) {
            for (int32_t i = 1; i <= 16; ++i) r.ids.push_back(i * 100);
            r.width = r.height = 256;
            r.sigmas = {1.0f, 0.0f};
        }
        try {
            p.generate(r);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-image: no warm-up (%s)\n", e.what());
        }
        p.release(true);   // idle, the engine holds next to nothing on the GPU: the first request opens the parts
    }
    std::printf("READY budget_mib=%llu\n", (unsigned long long) (p.budget() >> 20));
    std::fflush(stdout);
    Lines in;
    std::thread reader([&in] { in.run(); });
    reader.detach();   // ends with the process: a blocked getline cannot be woken
    for (std::string line; in.next(line);) {
        std::istringstream ls(line);
        std::string op;
        ls >> op;
        try {
            if (op == "QUIT") {
                break;
            } else if (op == "IMG") {
                si::QiRequest r;
                std::string out, ids;
                ls >> r.width >> r.height >> r.seed >> r.drop >> out >> ids;
                if (!ls || out.empty()) throw std::invalid_argument("IMG <width> <height> <seed> <drop> <out> <ids>");
                r.ids = parse_ids(ids);
                r.refs = read_refs(ls, r);
                r.on_preview = [&out](int index, int done, const std::vector<uint8_t>& rgba) {
                    const std::string path = out + ".p" + std::to_string(index);
                    write_file(path, rgba);
                    std::printf("PREVIEW %d %d %s\n", index, done, path.c_str());
                    std::fflush(stdout);
                };
                in.stop = false;
                si::QiTimes t;
                si::QiPlan pl;
                const auto img = p.generate(r, &t, &pl, [&in](int done, int steps) {
                    std::printf("STEP %d %d\n", done, steps);
                    std::fflush(stdout);
                    return !in.stop.load();
                }, [](const char* stage) {   // the server's progress (serve/images.py)
                    std::printf("STAGE %s\n", stage);
                    std::fflush(stdout);
                });
                write_file(out, img);
                std::printf("IMAGE %s\n", plan_kv(t, pl).c_str());
            } else if (op == "RATIO") {   // the size a request leaves to the model (serve/images.py)
                si::QiRequest r;
                std::string ids;
                ls >> r.drop >> ids;
                if (!ls) throw std::invalid_argument("RATIO <drop> <ids> [REF:<path>]...");
                r.ids = parse_ids(ids);
                r.refs = read_refs(ls, r);
                double secs = 0.0;
                const int i = p.choose_ratio(r, &secs);
                std::printf("RATIO index=%d ratio_s=%.3f\n", i, secs);
            } else if (op == "NEED") {
                int64_t w = 0, h = 0, n = 0;
                ls >> w >> h >> n;
                if (!ls) throw std::invalid_argument("NEED <width> <height> <n> [REF:<path>]... [MASK:<path>]");
                bool mask = false;
                int64_t n_neg = 0;
                const auto sizes = ref_sizes(ls, mask, n_neg);
                const si::QiNeed nd = p.need(n, w, h, sizes, mask, n_neg);
                std::printf("NEED least_mib=%llu full_mib=%llu held_mib=%llu free_mib=%llu cap_mib=%llu host_mib=%llu\n",
                            (unsigned long long) (nd.least >> 20), (unsigned long long) (nd.full >> 20),
                            (unsigned long long) (nd.held >> 20), (unsigned long long) (nd.free >> 20),
                            (unsigned long long) (nd.cap >> 20), (unsigned long long) (nd.host >> 20));
            } else if (op == "FREE") {
                size_t free = 0, total = 0;
                strata::gpu::mem_info(&free, &total);
                std::printf("FREE free_mib=%llu held_mib=%llu\n", (unsigned long long) (free >> 20),
                            (unsigned long long) (p.held() >> 20));
            } else if (op == "BUDGET") {
                uint64_t mib = 0;
                ls >> mib;
                p.set_budget(mib << 20);
                std::printf("BUDGET budget_mib=%llu\n", (unsigned long long) (p.budget() >> 20));
            } else if (op == "FIT") {
                uint64_t mib = 0;
                ls >> mib;
                p.fit(mib << 20);
                std::printf("BUDGET budget_mib=%llu\n", (unsigned long long) (p.budget() >> 20));
            } else if (op == "RELEASE") {
                std::string what;
                ls >> what;
                p.release(what == "ALL");
                std::printf("RELEASE ok\n");
            } else if (!op.empty()) {
                throw std::invalid_argument("unknown request " + op);
            }
        } catch (const si::QiStopped&) {
            std::printf("ERROR stopped\n");
        } catch (const std::exception& e) {
            if (op == "IMG") {   // a request that failed (short of memory) gives its cache back
                try {
                    p.release(false);
                } catch (const std::exception& e2) {
                    std::fprintf(stderr, "strata-image: release after a failed request: %s\n", e2.what());
                }
            }
            std::string m = e.what();
            for (char& c : m)
                if (c == '\n') c = ' ';
            std::printf("ERROR %s\n", m.c_str());
        }
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // oneMath's rocBLAS backend times hipBLASLt's first algorithms on a shape's first product and keeps the fastest:
    // an image model's few shapes repeat every step (RX 9060 XT: its int8 products 20 to 40% faster), where the text
    // engine's change with each prompt (it leaves the timing off)
    if (std::getenv("ONEMATH_HIPBLASLT_TUNE") == nullptr) {
#if defined(_WIN32)
        _putenv_s("ONEMATH_HIPBLASLT_TUNE", "1");
#else
        setenv("ONEMATH_HIPBLASLT_TUNE", "1", 0);
#endif
    }
    try {
        std::string dit, vae, text, vision, ratio_head, out, ids;
        std::vector<std::string> ref_paths;
        si::QiRequest r;
        uint64_t budget = 0;
        int repeat = 1;
        std::string edit_dit;
        int dit_q8 = -1;   // Mage-Flow: by the budget
        bool serving = false, dit_keep = true, ratio = false;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--dit") dit = val();
            else if (a == "--vae") vae = val();
            else if (a == "--text") text = val();
            else if (a == "--vision") vision = val();
            else if (a == "--ratio-head") ratio_head = val();
            else if (a == "--ratio") ratio = true;
            else if (a == "--ref") ref_paths.push_back(val());
            else if (a == "--mask") r.mask = read_image(val());
            else if (a == "--init") r.init = read_image(val());
            else if (a == "--strength") r.strength = std::stod(val());
            else if (a == "--neg-ids") r.neg_ids = parse_ids(val());
            else if (a == "--cfg") r.cfg = std::stof(val());
            else if (a == "--out") out = val();
            else if (a == "--ids") ids = val();
            else if (a == "--drop") r.drop = std::stoll(val());
            else if (a == "--width") r.width = std::stoll(val());
            else if (a == "--height") r.height = std::stoll(val());
            else if (a == "--seed") r.seed = std::stoull(val());
            else if (a == "--sigmas") r.sigmas = parse_sigmas(val());
            else if (a == "--steps") r.steps = std::stoll(val());
            else if (a == "--guidance") r.guidance = std::stof(val());
            else if (a == "--budget-mib") budget = std::stoull(val()) << 20;
            else if (a == "--repeat") repeat = std::stoi(val());
            else if (a == "--serve") serving = true;
            else if (a == "--dit-q8") {
                const std::string v = val();
                dit_q8 = v == "auto" ? -1 : v != "0";
            }
            else if (a == "--edit-dit") edit_dit = val();
            else if (a == "--dit-keep") dit_keep = val() != "0";
            else throw std::invalid_argument("unknown option " + a);
        }
        // the model is in the DiT's file, not in its extension: both Mage-Flow and HunyuanImage 3.0 are
        // safetensors, and HunyuanImage 3.0's says so in its header (hunyuanimage3_is_dit_file reads the header
        // alone, no weights)
        const bool safetensors = dit.size() > 12 && dit.compare(dit.size() - 12, 12, ".safetensors") == 0;
        const bool hunyuanimage3 = safetensors && si::hunyuanimage3_is_dit_file(dit);
        const bool mage = safetensors && !hunyuanimage3;
        const bool need_text = !hunyuanimage3;
        if (dit.empty() || vae.empty() || (need_text && text.empty()) ||
            (!serving && ((out.empty() && !ratio) || ids.empty()))) {
            std::fprintf(stderr, "usage: %s --dit F --vae F --text F [--vision F] [--budget-mib M] (--ids 1,2,3 "
                         "[--drop N] [--width W] [--height H] [--seed S] [--sigmas 1,...,0] [--ref IMAGE]... "
                         "[--mask IMAGE] [--init IMAGE] [--strength F] [--repeat N] --out IMAGE.rgba | --serve) "
                         "[--dit-q8 auto|0|1] [--edit-dit F] "
                         "[--dit-keep 0|1]\n",
                         argv[0]);
            std::fprintf(stderr, "       %s: HunyuanImage 3.0 (a quantized .safetensors --dit) takes no --text "
                         "(its DiT reads the ids itself); [--steps N] [--guidance G] stand for [--sigmas] [--cfg]; "
                         "its edits (--ref) take --vision, the model's shard with vision_model.*; --ratio with "
                         "--ratio-head F prints the size it chooses (the ids ending with <boi> <img_size_N>) in "
                         "place of an image\n",
                         argv[0]);
            return 2;
        }
        std::unique_ptr<si::ImagePipeline> pipe;
        std::string err;
        if (hunyuanimage3) {
            auto h = std::make_unique<si::HunyuanImage3Pipeline>();
            if (!h->open(dit, vae, err, 0, vision, ratio_head)) throw std::runtime_error(err);
            pipe = std::move(h);
        } else if (mage) {
            auto m = std::make_unique<si::MfPipeline>();
            if (!m->open(dit, vae, text, err, dit_q8, edit_dit, vision, dit_keep)) throw std::runtime_error(err);
            pipe = std::move(m);
        } else {
            auto q = std::make_unique<si::QiPipeline>();
            if (!q->open(dit, vae, text, err, vision)) throw std::runtime_error(err);
            pipe = std::move(q);
        }
        si::ImagePipeline& p = *pipe;
        if (budget > 0) p.set_budget(budget);
        if (serving) return serve(p);
        r.ids = parse_ids(ids);
        for (const std::string& path : ref_paths) {
            si::QiRef f;
            f.rgba = si::load_rgba8(path, f.width, f.height, err);
            if (f.rgba.empty()) throw std::runtime_error(err);
            r.refs.push_back(std::move(f));
        }
        if (ratio) {
            double secs = 0.0;
            const int i = p.choose_ratio(r, &secs);
            std::printf("RATIO index=%d ratio_s=%.3f", i, secs);
            if (const auto* h = dynamic_cast<const si::HunyuanImage3Pipeline*>(&p))
                for (size_t k = 0; k < h->ratio_logits().size(); ++k)
                    std::printf("%s%.6g", k == 0 ? " logits=" : ",", h->ratio_logits()[k]);
            std::printf("\n");
            return 0;
        }
        for (int k = 0; k < repeat; ++k) {
            si::QiTimes t;
            si::QiPlan pl;
            write_file(out, p.generate(r, &t, &pl));
            std::printf("%s\n", plan_kv(t, pl).c_str());
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-image: %s\n", e.what());
        return 1;
    }
    return 0;
}
