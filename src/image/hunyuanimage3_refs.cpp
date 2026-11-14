// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_refs.cpp - see include/strata/image/hunyuanimage3_refs.hpp.
//
// The resampling is PIL's (Imaging/Resample.c): for an output pixel at center (o + 0.5) * scale the taps run from
// floor(center - support + 0.5) to floor(center + support + 0.5), weighted by filter((i - center + 0.5) / fs) with
// fs = max(scale, 1) and normalized; LANCZOS is sinc(x) sinc(x / 3) on |x| < 3, BILINEAR 1 - |x| on |x| < 1.  The
// two passes each round to 8 bits, as PIL's RGB images are.  torchvision's antialiased bilinear (the SigLIP2
// processor's) is the same construction; the two agree to a level.
#include "strata/image/hunyuanimage3_refs.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace {

constexpr double kPi = 3.14159265358979323846;   // MSVC's <cmath> has no M_PI without a define

double sinc(double x) {
    if (x == 0.0) return 1.0;
    const double px = kPi * x;
    return std::sin(px) / px;
}

double lanczos_w(double x) { return (x >= -3.0 && x < 3.0) ? sinc(x) * sinc(x / 3.0) : 0.0; }

double bilinear_w(double x) {
    x = std::fabs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

// one axis of PIL's resample over `lines` lines of `n_in` samples (stride_line between lines, stride_in between
// samples) into n_out samples each (dst_line, dst_out the destination's strides)
void resample_axis(const std::vector<float>& src, int64_t lines, int64_t n_in, int64_t stride_line, int64_t stride_in,
                   int64_t n_out, std::vector<float>& dst, int64_t dst_line, int64_t dst_out, bool lanczos,
                   bool round8) {
    const double scale = (double) n_in / (double) n_out, fs = std::max(scale, 1.0);
    const double support = (lanczos ? 3.0 : 1.0) * fs;
    for (int64_t o = 0; o < n_out; ++o) {
        const double center = ((double) o + 0.5) * scale;
        const auto lo = std::max<int64_t>((int64_t) std::floor(center - support + 0.5), 0);
        const auto hi = std::min<int64_t>((int64_t) std::floor(center + support + 0.5), n_in);
        std::vector<double> w((size_t) std::max<int64_t>(hi - lo, 0));
        double sum = 0.0;
        for (int64_t i = lo; i < hi; ++i)
            sum += w[(size_t) (i - lo)] = lanczos ? lanczos_w(((double) i - center + 0.5) / fs)
                                                  : bilinear_w(((double) i - center + 0.5) / fs);
        if (sum == 0.0) sum = 1.0;
        for (int64_t l = 0; l < lines; ++l) {
            double v = 0.0;
            for (int64_t i = lo; i < hi; ++i)
                v += w[(size_t) (i - lo)] * src[(size_t) (l * stride_line + i * stride_in)];
            v /= sum;
            dst[(size_t) (l * dst_line + o * dst_out)] =
                round8 ? (float) std::clamp(std::nearbyint(v), 0.0, 255.0) : (float) v;
        }
    }
}

}  // namespace

std::vector<HunyuanImage3Bucket> hunyuanimage3_buckets(int64_t base_size, int64_t align, int64_t step) {
    if (step <= 0) step = base_size / 16;
    const int64_t lo = base_size / 2, hi = base_size * 2;
    std::vector<std::pair<int64_t, int64_t>> sizes{{base_size, base_size}};
    int64_t h = base_size, w = base_size;
    while (h < hi || w > lo) {
        h = std::min(h + step, hi);
        w = std::max(w - step, lo);
        sizes.emplace_back(h / align * align, w / align * align);
    }
    h = base_size, w = base_size;
    while (h > lo || w < hi) {
        h = std::max(h - step, lo);
        w = std::min(w + step, hi);
        sizes.emplace_back(h / align * align, w / align * align);
    }
    std::stable_sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) {
        return (double) a.first / (double) a.second < (double) b.first / (double) b.second;
    });
    // the four ratios the processor adds, (height, width), when the walk has not got them
    for (auto [eh, ew] : {std::pair<int64_t, int64_t>{1024, 768}, {1280, 720}, {768, 1024}, {720, 1280}}) {
        const double r = (double) eh / (double) ew;
        bool have = false;
        for (const auto& s : sizes) have = have || ((double) s.first / (double) s.second == r);
        if (!have) sizes.emplace_back(eh, ew);
    }
    std::vector<HunyuanImage3Bucket> out;
    out.reserve(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i)
        out.push_back({(int32_t) i, sizes[i].first, sizes[i].second, (double) sizes[i].first / (double) sizes[i].second});
    return out;
}

const HunyuanImage3Bucket& hunyuanimage3_nearest_bucket(const std::vector<HunyuanImage3Bucket>& table, int64_t width,
                                                        int64_t height) {
    if (table.empty()) throw std::invalid_argument("hunyuanimage3_nearest_bucket: an empty table");
    const double r = (double) height / (double) width;
    size_t best = 0;
    for (size_t i = 1; i < table.size(); ++i)
        if (std::fabs(table[i].ratio - r) < std::fabs(table[best].ratio - r)) best = i;
    return table[best];
}

void hunyuanimage3_naflex_size(int64_t width, int64_t height, int64_t patch, int64_t max_patches, int64_t& out_w,
                               int64_t& out_h) {
    // transformers' bisection: the largest scale whose scaled sides (rounded up to the patch, at least one patch)
    // hold at most max_patches patches
    auto scaled = [patch](double scale, int64_t size) {
        const double s = (double) size * scale;
        const int64_t v = (int64_t) std::ceil(s / (double) patch) * patch;
        return std::max(patch, v);
    };
    constexpr double eps = 1e-5;
    double lo = eps / 10.0, hi = 100.0;
    while (hi - lo >= eps) {
        const double mid = (lo + hi) / 2.0;
        const int64_t th = scaled(mid, height), tw = scaled(mid, width);
        if ((th / patch) * (tw / patch) <= max_patches) lo = mid;
        else hi = mid;
    }
    out_h = scaled(lo, height);
    out_w = scaled(lo, width);
}

std::vector<float> hunyuanimage3_resize(const std::vector<float>& rgb, int64_t channels, int64_t W, int64_t H,
                                        int64_t W2, int64_t H2, bool lanczos, bool round8) {
    if (W == W2 && H == H2) return rgb;
    std::vector<float> t((size_t) (channels * H * W2)), out((size_t) (channels * H2 * W2));
    if (W != W2) resample_axis(rgb, channels * H, W, W, 1, W2, t, W2, 1, lanczos, round8);
    else t = rgb;
    if (H == H2) return t;
    for (int64_t c = 0; c < channels; ++c) {
        std::vector<float> plane(t.begin() + c * H * W2, t.begin() + (c + 1) * H * W2), o((size_t) (H2 * W2));
        resample_axis(plane, W2, H, 1, W2, H2, o, 1, W2, lanczos, round8);
        std::copy(o.begin(), o.end(), out.begin() + c * H2 * W2);
    }
    return out;
}

std::vector<float> hunyuanimage3_ref_rgb(const QiRef& f) {
    const int64_t n = f.width * f.height;
    std::vector<float> rgb((size_t) (3 * n));
    for (int64_t q = 0; q < n; ++q)
        for (int c = 0; c < 3; ++c) rgb[(size_t) (c * n + q)] = (float) f.rgba[(size_t) (q * 4 + c)];
    return rgb;
}

std::vector<float> hunyuanimage3_vae_copy(const QiRef& f, int64_t tw, int64_t th) {
    if (f.width <= 0 || f.height <= 0 || tw <= 0 || th <= 0)
        throw std::invalid_argument("hunyuanimage3_vae_copy: an empty image or target");
    // the reference's resize_and_crop: the aspect kept, the side that fits the target exactly and the other rounded
    // (Python's round: to the nearest, ties to even, as nearbyint), then the center crop
    const double tr = (double) th / (double) tw, r = (double) f.height / (double) f.width;
    int64_t rw = 0, rh = 0;
    if (r < tr) {
        rh = th;
        rw = (int64_t) std::nearbyint((double) th / (double) f.height * (double) f.width);
    } else {
        rw = tw;
        rh = (int64_t) std::nearbyint((double) tw / (double) f.width * (double) f.height);
    }
    rw = std::max(rw, tw);
    rh = std::max(rh, th);
    const int64_t top = (int64_t) std::nearbyint((double) (rh - th) / 2.0);
    const int64_t left = (int64_t) std::nearbyint((double) (rw - tw) / 2.0);
    const std::vector<float> big = hunyuanimage3_resize(hunyuanimage3_ref_rgb(f), 3, f.width, f.height, rw, rh, true);
    std::vector<float> out((size_t) (3 * th * tw));
    for (int c = 0; c < 3; ++c)
        for (int64_t y = 0; y < th; ++y)
            for (int64_t x = 0; x < tw; ++x)
                out[(size_t) ((c * th + y) * tw + x)] =
                    big[(size_t) ((c * rh + (y + top)) * rw + (x + left))] / 127.5f - 1.0f;
    return out;
}

std::vector<float> hunyuanimage3_vit_patches(const QiRef& f, int64_t patch, int64_t max_patches, int64_t& ph,
                                             int64_t& pw) {
    if (f.width <= 0 || f.height <= 0) throw std::invalid_argument("hunyuanimage3_vit_patches: an empty image");
    int64_t rw = 0, rh = 0;
    hunyuanimage3_naflex_size(f.width, f.height, patch, max_patches, rw, rh);
    ph = rh / patch;
    pw = rw / patch;
    if (ph * pw > max_patches) throw std::logic_error("hunyuanimage3_vit_patches: more patches than the most");
    const std::vector<float> img = hunyuanimage3_resize(hunyuanimage3_ref_rgb(f), 3, f.width, f.height, rw, rh, false);
    const int64_t D = patch * patch * 3;
    std::vector<float> out((size_t) (max_patches * D), 0.0f);
    for (int64_t py = 0; py < ph; ++py)
        for (int64_t px = 0; px < pw; ++px) {
            float* dst = out.data() + (py * pw + px) * D;
            for (int64_t dy = 0; dy < patch; ++dy)
                for (int64_t dx = 0; dx < patch; ++dx)
                    for (int64_t c = 0; c < 3; ++c)
                        dst[(dy * patch + dx) * 3 + c] =
                            img[(size_t) ((c * rh + (py * patch + dy)) * rw + (px * patch + dx))] / 127.5f - 1.0f;
        }
    return out;
}

}  // namespace strata::image
