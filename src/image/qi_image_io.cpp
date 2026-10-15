// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_image_io.cpp - see include/strata/image/qi_image_io.hpp.  stb_image's implementation is compiled here
// only (CMakeLists.txt keeps its warnings out of the build's).

#include "strata/image/qi_image_io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"

#include <cstring>

namespace strata::image {

std::vector<uint8_t> load_rgba8(const std::string& path, int64_t& width, int64_t& height, std::string& err) {
    int w = 0, h = 0, n = 0;
    unsigned char* p = stbi_load(path.c_str(), &w, &h, &n, 4);
    if (p == nullptr) {
        const char* why = stbi_failure_reason();
        err = path + ": " + (why ? why : "not an image");
        return {};
    }
    std::vector<uint8_t> out((size_t) w * (size_t) h * 4);
    std::memcpy(out.data(), p, out.size());
    stbi_image_free(p);
    width = w;
    height = h;
    return out;
}

bool image_size(const std::string& path, int64_t& width, int64_t& height) {
    int w = 0, h = 0, n = 0;
    if (stbi_info(path.c_str(), &w, &h, &n) == 0) return false;
    width = w;
    height = h;
    return true;
}

}  // namespace strata::image
