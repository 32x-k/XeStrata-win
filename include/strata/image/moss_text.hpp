// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/moss_text.hpp - MOSS-Image's text encoder: Qwen3.5-9B (its GGUF: the official one by Unsloth or
// HauhauCS's) through the chat engine's dense model (core::PlainModel, opened as an encoder), the prompt's token ids in
// MOSS-Image's chat form (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n`, one prompt at a time, no
// padding) to every token's state after the final norm, the DiT's context.  Its weights are all in VRAM while it is
// open (no streamed layers): the pipeline opens it for the text and closes it before the DiT where the two do not fit.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {
class PlainModel;
}

namespace strata::image {

class MossText {
public:
    MossText();
    ~MossText();
    MossText(const MossText&) = delete;
    MossText& operator=(const MossText&) = delete;

    /// Opens the GGUF for prompts of up to `max_tokens` tokens (MOSS-Image's 4096 at most).
    bool open(const std::string& gguf, int64_t max_tokens, std::string& err);
    bool is_open() const { return model_ != nullptr; }
    /// Frees the weights and the buffers (open again for the next text).
    void close();
    /// [ids.size()][hidden()] floats (host); empty with `err` on a failure.
    std::vector<float> encode(const std::vector<int32_t>& ids, std::string& err);
    int64_t hidden() const;

private:
    std::unique_ptr<core::PlainModel> model_;
};

}  // namespace strata::image
