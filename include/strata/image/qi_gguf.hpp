// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_gguf.hpp - a tensor of an image model's GGUF (the DiT, the text encoder, the vision encoder),
// checked before anything reads it.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <stdexcept>
#include <string>

namespace strata::image {

/// The tensor `name`, or nullptr when the file has none.  #29: a crafted or damaged file whose tensor has no
/// dimensions, a dimension of 0, a type the reader cannot size, or a payload past the end of the file is refused here
/// (std::runtime_error): the loaders take tensor_data() and divide by the shape, which faulted (SIGSEGV) or divided by
/// zero (SIGFPE) on such a file.  The same rule as GgufModel::in_bounds on the language model's paths.
inline const TensorInfo* find_checked(const GgufFile& g, const std::string& name) {
    const TensorInfo* t = g.find(name);
    if (t == nullptr) return nullptr;
    const auto bad = [&](const char* why) {
        return std::runtime_error("image engine: " + name + " in the model file: " + why);
    };
    if (t->shape.empty()) throw bad("no dimensions");
    for (uint64_t d : t->shape)
        if (d == 0) throw bad("a dimension of 0");
    const uint64_t bytes = tensor_payload_bytes(*t);
    if (bytes == 0) throw bad("a type of unknown size");
    if (g.data_start() > g.file_size()) throw bad("the file ends before its tensor data");
    const uint64_t payload = g.file_size() - g.data_start();
    if (t->offset > payload || bytes > payload - t->offset) throw bad("its data runs past the end of the file");
    return t;
}

}  // namespace strata::image
