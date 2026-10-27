// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_text.cpp - see include/strata/image/moss_text.hpp.

#include "strata/image/moss_text.hpp"

#include "strata/core/plain_model.hpp"

namespace strata::image {

MossText::MossText() = default;
MossText::~MossText() = default;

bool MossText::open(const std::string& gguf, int64_t max_tokens, std::string& err) {
    close();
    core::PlainOptions o;
    o.gguf = gguf;
    o.encoder = true;
    // the prompt in one chunk of its length (the encoder's states do not depend on the chunking, its K/V cells are the
    // prompt's), the windows of one token it never runs
    o.max_cells = max_tokens;
    o.prompt_chunk = max_tokens;
    o.prompt_chunk_given = true;
    o.max_t = 1;
    auto m = std::make_unique<core::PlainModel>();
    if (!m->load(o, err)) return false;
    model_ = std::move(m);
    return true;
}

void MossText::close() { model_.reset(); }

std::vector<float> MossText::encode(const std::vector<int32_t>& ids, std::string& err) {
    if (model_ == nullptr) {
        err = "MOSS-Image's text encoder is not open";
        return {};
    }
    std::vector<float> out(ids.size() * (size_t) hidden());
    if (!model_->encode(ids.data(), (int64_t) ids.size(), out.data(), err)) return {};
    return out;
}

int64_t MossText::hidden() const { return model_ != nullptr ? model_->spec().n_embd : 0; }

}  // namespace strata::image
