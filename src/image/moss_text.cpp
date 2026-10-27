// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_text.cpp - see include/strata/image/moss_text.hpp.

#include "strata/image/moss_text.hpp"

#include "strata/core/plain_model.hpp"

#include <algorithm>
#include <exception>
#include <fstream>

namespace strata::image {

// The engine's blocks from the layer stream: block l is layer l
struct MossText::Source : core::PlainModel::BlockSource {
    LayerStream* layers = nullptr;
    const uint8_t* begin(int l) override { return layers->begin(l); }
    void end(int l) override { layers->end(l); }
};

MossText::MossText() = default;
MossText::~MossText() { close(); }

bool MossText::open(const std::string& gguf, int64_t max_tokens, std::string& err, uint64_t layer_budget) {
    close();
    core::PlainOptions o;
    o.gguf = gguf;
    o.encoder = true;
    o.stream_blocks = true;
    // the prompt in one chunk of its length (the encoder's states do not depend on the chunking, its K/V cells are the
    // prompt's), the windows of one token it never runs
    o.max_cells = max_tokens;
    o.prompt_chunk = max_tokens;
    o.prompt_chunk_given = true;
    o.max_t = 1;
    auto m = std::make_unique<core::PlainModel>();
    auto layers = std::make_unique<LayerStream>();
    auto src = std::make_unique<Source>();
    try {
        if (!m->load(o, err)) return false;
        std::vector<std::vector<uint64_t>> offsets;
        for (int64_t l = 0; l < m->spec().n_layer; ++l) {
            std::vector<LayerStream::Span> spans;
            for (const core::PlainModel::HostSpan& h : m->block_spans(l)) spans.push_back({h.host, h.bytes, 0});
            const int i = layers->add(spans);
            offsets.emplace_back();
            for (size_t k = 0; k < spans.size(); ++k) offsets.back().push_back(layers->offset(i, (int) k));
        }
        core::PlainModel* mp = m.get();
        layers->set_prepare([mp](int l, uint8_t* d) { mp->prepare_block(l, d); });
        layers->configure(std::min(std::max(layer_budget, layers->min_budget()), layers->total()), m->stream());
        src->layers = layers.get();
        m->set_block_source(src.get(), std::move(offsets));
        // the products' kernels loaded and the dense products' forms timed, as the chat's engine does at its start
        if (!m->prompt_warm(err)) return false;
    } catch (const std::exception& e) {
        err = std::string("MOSS-Image's text encoder: ") + e.what();
        return false;
    }
    layers_ = std::move(layers);
    source_ = std::move(src);
    model_ = std::move(m);
    return true;
}

// the layer stream first: it waits for the model's stream (its compute stream), which the model destroys with itself
void MossText::close() {
    layers_.reset();
    source_.reset();
    model_.reset();
}

void MossText::set_layer_budget(uint64_t bytes) {
    if (layers_ != nullptr)
        layers_->configure(std::min(std::max(bytes, layers_->min_budget()), layers_->total()), model_->stream());
}

std::vector<float> MossText::encode(const std::vector<int32_t>& ids, std::string& err) {
    if (model_ == nullptr) {
        err = "MOSS-Image's text encoder is not open";
        return {};
    }
    std::vector<float> out(ids.size() * (size_t) hidden());
    try {
        if (!model_->encode(ids.data(), (int64_t) ids.size(), out.data(), err)) return {};
    } catch (const std::exception& e) {   // a streamed block's copy failed
        err = std::string("MOSS-Image's text encoder: ") + e.what();
        return {};
    }
    return out;
}

int64_t MossText::hidden() const { return model_ != nullptr ? model_->spec().n_embd : 0; }

bool MossText::read_states(const std::string& path, std::vector<float>& rows, int64_t& n, int64_t& width,
                           std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    int32_t hdr[3] = {};
    f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    bool ok = f && hdr[0] == 0x31455453 && hdr[1] > 0 && hdr[2] > 0;
    if (ok) {
        n = hdr[1];
        width = hdr[2];
        rows.resize((size_t) (n * width));
        f.read(reinterpret_cast<char*>(rows.data()), (std::streamsize) (rows.size() * sizeof(float)));
        ok = f && f.peek() == std::char_traits<char>::eof();
    }
    if (!ok) err = path + " is not a states file (STE1)";
    return ok;
}

}  // namespace strata::image
