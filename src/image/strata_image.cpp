// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/strata_image.cpp - strata-image: Qwen-Image-2.1 text to image on the GPU, one image.
//
//   strata-image --dit DIT.gguf --vae VAE.safetensors --text TEXT_ENCODER.gguf --ids 1,2,3 [--drop N]
//                [--width W] [--height H] [--seed S] [--budget-mib M] [--repeat N] --out IMAGE.rgba
//
// --ids: the prompt in the model's chat form, tokenized (the server's tokenizer); --drop: the leading ids of the
// system turn; --budget-mib: the VRAM the engine may take (default: the free VRAM); --repeat: the image again N times
// in the same process (the cache's effect).  Writes RGBA8 rows to --out and a line of times and of the budget's
// split (JSON) to stdout a run.

#include "strata/image/qi_pipeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        std::string dit, vae, text, out, ids;
        strata::image::QiRequest r;
        uint64_t budget = 0;
        int repeat = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--dit") dit = val();
            else if (a == "--vae") vae = val();
            else if (a == "--text") text = val();
            else if (a == "--out") out = val();
            else if (a == "--ids") ids = val();
            else if (a == "--drop") r.drop = std::stoll(val());
            else if (a == "--width") r.width = std::stoll(val());
            else if (a == "--height") r.height = std::stoll(val());
            else if (a == "--seed") r.seed = std::stoull(val());
            else if (a == "--budget-mib") budget = std::stoull(val()) << 20;
            else if (a == "--repeat") repeat = std::stoi(val());
            else throw std::invalid_argument("unknown option " + a);
        }
        if (dit.empty() || vae.empty() || text.empty() || out.empty() || ids.empty()) {
            std::fprintf(stderr, "usage: %s --dit F --vae F --text F --ids 1,2,3 [--drop N] [--width W] [--height H] "
                         "[--seed S] --out IMAGE.rgba\n", argv[0]);
            return 2;
        }
        std::stringstream ss(ids);
        for (std::string tok; std::getline(ss, tok, ',');)
            if (!tok.empty()) r.ids.push_back((int32_t) std::stol(tok));
        strata::image::QiPipeline p;
        std::string err;
        if (!p.open(dit, vae, text, err)) throw std::runtime_error(err);
        if (budget > 0) p.set_budget(budget);
        for (int k = 0; k < repeat; ++k) {
            strata::image::QiTimes t;
            strata::image::QiPlan pl;
            const auto img = p.generate(r, &t, &pl);
            std::ofstream o(out, std::ios::binary);
            o.write(reinterpret_cast<const char*>(img.data()), (std::streamsize) img.size());
            if (!o) throw std::runtime_error("cannot write " + out);
            std::printf("{\"text_s\": %.3f, \"sampling_s\": %.3f, \"decode_s\": %.3f, \"budget_mib\": %llu, "
                        "\"blocks_mib\": %llu, \"resident_blocks\": %d, \"layers_mib\": %llu, \"resident_layers\": %d, "
                        "\"activations_mib\": %llu, \"mlp_rows\": %lld, \"vae_mib\": %llu, \"vae_tiled\": %s, "
                        "\"text_cached\": %s}\n",
                        t.text, t.sampling, t.decode, (unsigned long long) (pl.budget >> 20),
                        (unsigned long long) (pl.blocks >> 20), pl.resident_blocks, (unsigned long long) (pl.layers >> 20),
                        pl.resident_layers, (unsigned long long) (pl.activations >> 20), (long long) pl.mlp_rows,
                        (unsigned long long) (pl.vae >> 20), pl.vae_tiled ? "true" : "false",
                        pl.text_cached ? "true" : "false");
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-image: %s\n", e.what());
        return 1;
    }
    return 0;
}
