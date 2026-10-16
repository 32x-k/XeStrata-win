// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_image_io.hpp - an edit's reference image from a file (PNG, JPEG, BMP, GIF; stb_image,
// third_party/main/stb).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

/// The image's RGBA8 [height][width][4] (an image without alpha opaque); empty with `err` when it cannot be read.
std::vector<uint8_t> load_rgba8(const std::string& path, int64_t& width, int64_t& height, std::string& err);

/// The image's size without decoding it; false when the file is not an image stb_image reads.
bool image_size(const std::string& path, int64_t& width, int64_t& height);

}  // namespace strata::image
