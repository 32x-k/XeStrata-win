// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/strata_image.cpp - strata-image: Qwen-Image-2.1 text to image on the GPU.
//
//   strata-image --dit DIT.gguf --vae VAE.safetensors --text TEXT_ENCODER.gguf [--vision MMPROJ.gguf]
//                [--budget-mib M] (--ids 1,2,3 [--drop N] [--width W] [--height H] [--seed S] [--ref IMAGE]...
//                [--repeat N] --out IMAGE.rgba | --serve)
//
// One image: --ids, the prompt in the model's chat form, tokenized (the server's tokenizer); --drop, the leading ids of
// the system turn; --budget-mib, the VRAM the engine may take (default: the free VRAM); --repeat, the image again N
// times in the process (the cache's effect); --ref, an edit's reference image (PNG, JPEG, BMP, GIF; its pad in --ids
// is one <|image_pad|>, which the engine widens to the image's tokens), with the vision encoder (--vision, Qwen3-VL's
// mmproj).  Writes RGBA8 rows to --out and a line of times and of the budget's split a run.
//
// --serve: the engine stays, its layers and the last prompt's states kept between requests, and takes lines on stdin
// (serve/images.py), one at a time, answering each on stdout:
// A path on a request line is UTF-8 with %XX for a byte that is not printable ASCII and for % itself (a space, a
// non-ASCII name), so that one word holds it (serve/images.py's path_word).
//   IMG <width> <height> <seed> <drop> <out path> <ids, commas> [REF:<path>]...   -> STEP <done> <steps> lines,
//                            then IMAGE key=value ... (times, the budget's split); REF: an edit's reference images
//   NEED <width> <height> <n> [REF:<path>]...  the VRAM an image of n prompt ids needs (from the files)
//                                                                                          -> NEED least_mib=...
//                            full_mib=... held_mib=... free_mib=... (si::QiNeed)
//   FREE                     the free VRAM and what the engine holds                        -> FREE free_mib=... held_mib=...
//   BUDGET <MiB>             the VRAM the engine may take (0: the free VRAM at its start)      -> BUDGET budget_mib=...
//   FIT <MiB>                the budget: what the engine holds and the free VRAM, less <MiB>  -> BUDGET budget_mib=...
//   RELEASE [ALL]            frees the layers, activations and buffers (the cache); ALL also the parts' fixed weights
//                            and scratch, the parts opened again on the next IMG                -> RELEASE ok
//   STOP                     (while an IMG runs) ends it after the step it is in               -> ERROR stopped
//   QUIT                     ends the engine
// A request that fails answers ERROR <message> and the engine goes on.  Before the first line it makes a small image
// (its kernels built and the library's first products set up; skipped when the GPU has no room) and lets its buffers
// go, then says READY budget_mib=<M>.

#include "strata/core/device.hpp"
#include "strata/core/gpu.hpp"
#include "strata/image/qi_image_io.hpp"
#include "strata/image/qi_pipeline.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
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

// a path word of a request line: %XX back to its byte
std::string path_of(const std::string& word) {
    auto hex = [&](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::invalid_argument("a path with a bad %XX: " + word);
    };
    std::string out;
    for (size_t i = 0; i < word.size(); ++i) {
        if (word[i] != '%') { out += word[i]; continue; }
        if (i + 2 >= word.size()) throw std::invalid_argument("a path with a bad %XX: " + word);
        out += (char) (hex(word[i + 1]) * 16 + hex(word[i + 2]));
        i += 2;
    }
    return out;
}

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> ids;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');)
        if (!tok.empty()) ids.push_back((int32_t) std::stol(tok));
    return ids;
}

// the REF:<path> words left on a request line: the images read, or (sizes) their sizes only
std::vector<si::QiRef> read_refs(std::istringstream& ls) {
    std::vector<si::QiRef> refs;
    for (std::string word; ls >> word;) {
        if (word.rfind("REF:", 0) != 0) throw std::invalid_argument("unknown word " + word);
        si::QiRef f;
        std::string err;
        f.rgba = si::load_rgba8(path_of(word.substr(4)), f.width, f.height, err);
        if (f.rgba.empty()) throw std::invalid_argument(err);
        refs.push_back(std::move(f));
    }
    return refs;
}

std::vector<std::pair<int64_t, int64_t>> ref_sizes(std::istringstream& ls) {
    std::vector<std::pair<int64_t, int64_t>> sizes;
    for (std::string word; ls >> word;) {
        int64_t w = 0, h = 0;
        if (word.rfind("REF:", 0) != 0 || !si::image_size(path_of(word.substr(4)), w, h))
            throw std::invalid_argument("not an image: " + word);
        sizes.emplace_back(w, h);
    }
    return sizes;
}

std::string plan_kv(const si::QiTimes& t, const si::QiPlan& p) {
    char b[512];
    std::snprintf(b, sizeof b,
                  "text_s=%.3f sampling_s=%.3f decode_s=%.3f budget_mib=%llu blocks_mib=%llu resident_blocks=%d "
                  "layers_mib=%llu resident_layers=%d activations_mib=%llu mlp_rows=%lld vae_mib=%llu vae_tiled=%d "
                  "text_cached=%d",
                  t.text, t.sampling, t.decode, (unsigned long long) (p.budget >> 20),
                  (unsigned long long) (p.blocks >> 20), p.resident_blocks, (unsigned long long) (p.layers >> 20),
                  p.resident_layers, (unsigned long long) (p.activations >> 20), (long long) p.mlp_rows,
                  (unsigned long long) (p.vae >> 20), p.vae_tiled ? 1 : 0, p.text_cached ? 1 : 0);
    return b;
}

void write_file(const std::filesystem::path& path, const std::vector<uint8_t>& img) {
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(img.data()), (std::streamsize) img.size());
    if (!o) throw std::runtime_error("cannot write " + path.string());
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

int serve(si::QiPipeline& p) {
    {   // the warm-up: a 256 x 256 image of one step from token ids the vocabulary has; where the GPU has no room
        // for it now (another program holds it), the first request builds the kernels instead
        si::QiRequest r;
        for (int32_t i = 1; i <= 16; ++i) r.ids.push_back(i * 100);
        r.width = r.height = 256;
        r.sigmas = {1.0f, 0.0f};
        try {
            p.generate(r);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-image: no warm-up (%s)\n", e.what());
        }
        p.release(true);   // idle, the engine holds next to nothing on the GPU: the first request opens the parts
    }
    std::printf("READY budget_mib=%llu\n", (unsigned long long) (p.budget() >> 20));
    std::fflush(stdout);
    // shared with the reader: it outlives serve() when QUIT returns while a getline still blocks
    const auto in = std::make_shared<Lines>();
    std::thread reader([in] { in->run(); });
    reader.detach();   // ends with the process: a blocked getline cannot be woken
    for (std::string line; in->next(line);) {
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
                r.refs = read_refs(ls);
                in->stop = false;
                si::QiTimes t;
                si::QiPlan pl;
                const auto img = p.generate(r, &t, &pl, [&in](int done, int steps) {
                    std::printf("STEP %d %d\n", done, steps);
                    std::fflush(stdout);
                    return !in->stop.load();
                });
                const std::string utf8 = path_of(out);
                write_file(std::filesystem::path(std::u8string(utf8.begin(), utf8.end())), img);
                std::printf("IMAGE %s\n", plan_kv(t, pl).c_str());
            } else if (op == "NEED") {
                int64_t w = 0, h = 0, n = 0;
                ls >> w >> h >> n;
                if (!ls) throw std::invalid_argument("NEED <width> <height> <n> [REF:<path>]...");
                const si::QiNeed nd = p.need(n, w, h, ref_sizes(ls));
                std::printf("NEED least_mib=%llu full_mib=%llu held_mib=%llu free_mib=%llu\n",
                            (unsigned long long) (nd.least >> 20), (unsigned long long) (nd.full >> 20),
                            (unsigned long long) (nd.held >> 20), (unsigned long long) (nd.free >> 20));
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
    strata::core::restrict_hip_to_selector();   // before the HIP runtime starts, as the text engine does
    strata::core::no_hip_resource_cache();
    try {
        std::string dit, vae, text, vision, out, ids;
        std::vector<std::string> ref_paths;
        si::QiRequest r;
        uint64_t budget = 0;
        int repeat = 1;
        bool serving = false;
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
            else if (a == "--ref") ref_paths.push_back(val());
            else if (a == "--out") out = val();
            else if (a == "--ids") ids = val();
            else if (a == "--drop") r.drop = std::stoll(val());
            else if (a == "--width") r.width = std::stoll(val());
            else if (a == "--height") r.height = std::stoll(val());
            else if (a == "--seed") r.seed = std::stoull(val());
            else if (a == "--budget-mib") budget = std::stoull(val()) << 20;
            else if (a == "--repeat") repeat = std::stoi(val());
            else if (a == "--serve") serving = true;
            else throw std::invalid_argument("unknown option " + a);
        }
        if (dit.empty() || vae.empty() || text.empty() || (!serving && (out.empty() || ids.empty()))) {
            std::fprintf(stderr, "usage: %s --dit F --vae F --text F [--vision F] [--budget-mib M] (--ids 1,2,3 "
                         "[--drop N] [--width W] [--height H] [--seed S] [--ref IMAGE]... [--repeat N] --out IMAGE.rgba "
                         "| --serve)\n", argv[0]);
            return 2;
        }
        si::QiPipeline p;
        std::string err;
        if (!p.open(dit, vae, text, err, vision)) throw std::runtime_error(err);
        if (budget > 0) p.set_budget(budget);
        if (serving) return serve(p);
        r.ids = parse_ids(ids);
        for (const std::string& path : ref_paths) {
            si::QiRef f;
            f.rgba = si::load_rgba8(path, f.width, f.height, err);
            if (f.rgba.empty()) throw std::runtime_error(err);
            r.refs.push_back(std::move(f));
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
