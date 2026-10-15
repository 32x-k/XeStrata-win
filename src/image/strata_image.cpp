// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/strata_image.cpp - strata-image: Qwen-Image-2.1 text to image on the GPU, one image.
//
//   strata-image --dit DIT.gguf --vae VAE.safetensors --text TEXT_ENCODER.gguf --ids 1,2,3 [--drop N]
//                [--width W] [--height H] [--seed S] --out IMAGE.rgba
//
// --ids: the prompt in the model's chat form, tokenized (the server's tokenizer); --drop: the leading ids of the
// system turn.  Writes RGBA8 rows to --out and one line of times (JSON) to stdout.

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
        strata::image::QiTimes t;
        const auto img = p.generate(r, &t);
        std::ofstream o(out, std::ios::binary);
        o.write(reinterpret_cast<const char*>(img.data()), (std::streamsize) img.size());
        if (!o) throw std::runtime_error("cannot write " + out);
        std::printf("{\"text_s\": %.3f, \"sampling_s\": %.3f, \"decode_s\": %.3f}\n", t.text, t.sampling, t.decode);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-image: %s\n", e.what());
        return 1;
    }
    return 0;
}
