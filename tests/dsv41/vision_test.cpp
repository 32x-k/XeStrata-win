// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_vision.hpp"
#include "strata/core/runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

namespace {
uint16_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float value(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float rounded(float f) { return value(bits(f)); }
struct Tensor { std::vector<int64_t> shape; std::vector<uint16_t> data; };
using Tensors = std::map<std::string, Tensor>;

Tensors fixture(const strata::ds4::VisionGeo& g) {
    Tensors t;
    auto put = [&](const std::string& name, std::vector<int64_t> shape, float fill = 0) {
        size_t n = 1;
        for (int64_t d : shape) n *= size_t(d);
        t[name] = {std::move(shape), std::vector<uint16_t>(n, bits(fill))};
    };
    put("vision.patch_embed.proj.weight", {g.dim, 3});
    put("vision.patch_embed.proj.bias", {g.dim});
    put("vision.norm.weight", {g.dim}, 1);
    const std::string p = "vision.blocks.0.";
    put(p + "norm1.weight", {g.dim}, 1); put(p + "norm2.weight", {g.dim}, 1);
    put(p + "attn.wqkv.weight", {int64_t(g.dim) * 3, g.dim}); put(p + "attn.wqkv.bias", {int64_t(g.dim) * 3});
    put(p + "attn.wo.weight", {g.dim, g.dim}); put(p + "attn.wo.bias", {g.dim});
    put(p + "mlp.w1.weight", {int64_t(g.inter) * 2, g.dim}); put(p + "mlp.w2.weight", {g.dim, g.inter});
    put("aligner.w1.weight", {g.out, int64_t(g.dim) * 4}); put("aligner.w1.bias", {g.out});
    put("aligner.w2.weight", {g.out, g.out}); put("aligner.w2.bias", {g.out});
    for (int d = 0; d < g.dim; ++d) {
        t["vision.patch_embed.proj.weight"].data[size_t(d) * 3 + d % 3] = bits(float(d % 2 + 1) * 0.25f);
        t[p + "attn.wqkv.weight"].data[(size_t(g.dim) * 2 + d) * g.dim + d] = bits(1);
        t[p + "attn.wo.weight"].data[size_t(d) * g.dim + d] = bits(1);
    }
    const int selected[] = {0, 1, 4, 7};
    for (int o = 0; o < g.out; ++o) {
        for (int k = 0; k < 4; ++k)
            t["aligner.w1.weight"].data[size_t(o) * g.dim * 4 + size_t(selected[o]) * 4 + k] = bits(std::ldexp(1.0f, -k));
        t["aligner.w1.bias"].data[o] = bits(float(o - 1) * 0.125f);
        t["aligner.w2.weight"].data[size_t(o) * g.out + o] = bits(1);
        t["aligner.w2.bias"].data[o] = bits(float(o + 1) * 0.0625f);
    }
    return t;
}
strata::ds4::VisionTower::Reader reader(const Tensors& tensors) {
    return [&tensors](const std::string& name, std::vector<int64_t>& shape, void* destination) {
        const auto at = tensors.find(name);
        if (at == tensors.end()) return false;
        shape = at->second.shape;
        if (destination) std::memcpy(destination, at->second.data.data(), at->second.data.size() * 2);
        return true;
    };
}

// Q and K are zero, V and the output projection are identity, and the FFN is zero.
// Attention is therefore a uniform mean, independently in both heads. This closed form
// exercises head layout, BF16 probabilities, residuals and channel-major merging without another GPU implementation.
std::vector<float> expected(const strata::ds4::VisionGeo& g, const std::vector<uint8_t>& rgb, int w, int h) {
    const size_t n = size_t(w) * h;
    std::vector<float> x(n * g.dim), normalized(x.size()), mean(size_t(g.dim));
    auto norm = [&] {
        for (size_t r = 0; r < n; ++r) {
            float sum = 0;
            for (int d = 0; d < g.dim; ++d) sum += x[r * g.dim + d] * x[r * g.dim + d];
            const float scale = 1.0f / std::sqrt(sum / float(g.dim) + g.eps);
            for (int d = 0; d < g.dim; ++d) normalized[r * g.dim + d] = rounded(x[r * g.dim + d] * scale);
        }
    };
    for (size_t r = 0; r < n; ++r)
        for (int d = 0; d < g.dim; ++d)
            x[r * g.dim + d] = rounded((float(rgb[r * 3 + d % 3]) / 255.0f - 0.5f) / 0.5f) * (float(d % 2 + 1) * 0.25f);
    norm();
    for (int d = 0; d < g.dim; ++d) {
        for (size_t r = 0; r < n; ++r) mean[size_t(d)] += rounded(1.0f / float(n)) * normalized[r * g.dim + d];
        mean[size_t(d)] = rounded(mean[size_t(d)]);
    }
    for (size_t r = 0; r < n; ++r)
        for (int d = 0; d < g.dim; ++d) x[r * g.dim + d] += mean[size_t(d)];
    norm();
    const int rows = (h + 1) / 2, cols = (w + 1) / 2, selected[] = {0, 1, 4, 7};
    std::vector<float> out(size_t(rows) * cols * g.out);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col)
            for (int o = 0; o < g.out; ++o) {
                float v = 0;
                for (int k = 0; k < 4; ++k) {
                    const int y = row * 2 + k / 2, xx = col * 2 + k % 2;
                    if (y < h && xx < w) v += std::ldexp(1.0f, -k) * normalized[(size_t(y) * w + xx) * g.dim + selected[o]];
                }
                v = rounded(v + float(o - 1) * 0.125f);
                v = rounded(0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f)));
                out[(size_t(row) * cols + col) * g.out + o] = rounded(v + float(o + 1) * 0.0625f);
            }
    return out;
}
}  // namespace

int main(int argc, char** argv) try {
    strata::ds4::VisionGeo g;
    g.layers = 1; g.dim = 8; g.heads = 2; g.inter = 4; g.patch = 1; g.merge = 2; g.out = 4; g.max_tokens = 12;
    auto tensors = fixture(g);
    strata::ds4::VisionTower tower;
    std::string error;
    auto bad = tensors;
    bad.erase("vision.blocks.0.attn.wo.weight");
    if (tower.load(g, 0, reader(bad), error) || tower.loaded()) return 1;
    if (!tower.load(g, 0, reader(tensors), error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
    if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
        std::puts("vision: all shapes checked before weights or GPU work"); return 0;
    }
    if (strata::core::Runtime::count() == 0) return 77;
    for (const auto shape : {std::array<int, 2>{3, 1}, {2, 2}, {1, 3}}) {
        const int w = shape[0], h = shape[1];
        std::vector<uint8_t> pixels(size_t(w) * h * 3);
        for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = uint8_t((i * 71 + 19) % 256);
        std::vector<uint16_t> actual, repeated;
        int nh = 0, nw = 0;
        if (!tower.encode(pixels.data(), w, h, actual, nh, nw, error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
        const auto want = expected(g, pixels, w, h);
        if (nh != (h + 1) / 2 || nw != (w + 1) / 2 || actual.size() != want.size()) return 1;
        for (size_t i = 0; i < want.size(); ++i) {
            const float tolerance = std::ldexp(1.0f, std::ilogb(std::max(std::abs(want[i]), 0x1p-126f)) - 6);
            if (!std::isfinite(value(actual[i])) || std::abs(value(actual[i]) - want[i]) > tolerance) {
                std::fprintf(stderr, "vision: closed-form mismatch at %zu: %.9g / %.9g\n", i, value(actual[i]), want[i]); return 1;
            }
        }
        if (!tower.encode(pixels.data(), w, h, repeated, nh, nw, error) || repeated != actual) return 1;
        if (tower.encode(pixels.data(), 0, h, repeated, nh, nw, error) || !repeated.empty()) return 1;
    }
    std::puts("vision: uniform attention, two heads, partial merges and repeat encodes passed");
    return 0;
} catch (const strata::core::DeviceError& e) { std::fprintf(stderr, "%s\n", e.what()); return 77; }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
