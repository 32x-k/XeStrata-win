// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_vision.hpp"
#include "ds4_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>

namespace strata::ds4 {
namespace {

inline float vision_float(uint16_t v) { return sycl::bit_cast<float>(uint32_t(v) << 16); }
inline uint16_t vision_bits(float v) {
    uint32_t u = sycl::bit_cast<uint32_t>(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40u);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

size_t product(std::initializer_list<size_t> dims) {
    size_t n = 1;
    for (size_t d : dims) {
        if (!d || n > std::numeric_limits<size_t>::max() / d) throw std::runtime_error("vision: size overflow");
        n *= d;
    }
    return n;
}

struct VisionWork {
    int device;
    rt::Queue stream;
    std::unique_ptr<prefill::Gemm> gemm;
    std::vector<void*> allocations;
    size_t bytes = 0;
    explicit VisionWork(int id) : device(id), gemm(std::make_unique<prefill::Gemm>()) {
        if (rt::stream_create(&stream) != rt::Result::success) throw std::runtime_error("vision: queue creation failed");
        gemm->init_external(stream.handle, nullptr, 0);
    }
    ~VisionWork() {
        core::OnDevice on(device);
        if (stream) {
            try { static_cast<sycl::queue&>(stream).wait_and_throw(); }
            catch (const std::exception& e) { std::fprintf(stderr, "vision: workspace cleanup: %s\n", e.what()); }
        }
        gemm.reset();
        for (void* p : allocations) rt::free(p);
        if (stream) rt::stream_destroy(stream);
    }
    template<class T> T* take(size_t count) {
        const size_t n = product({count, sizeof(T)});
        T* p = nullptr;
        if (rt::alloc_device(&p, n) != rt::Result::success) throw std::runtime_error("vision: workspace allocation failed");
        allocations.push_back(p);
        bytes += n;
        return p;
    }
};

}  // namespace

struct VisionTower::Impl {
    VisionGeo g;
    int device = -1, patches = 0;
    size_t host = 0, workspace = 0, largest_weight = 0;
    std::map<std::string, std::vector<uint16_t>> tensors;
    const std::vector<uint16_t>& tensor(const std::string& name) const { return tensors.at(name); }

    bool run(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& result, int& nh, int& nw) {
        const int D = g.dim, I = g.inter, P = g.patch, M = g.merge, H = g.heads, hd = D / H;
        if (!rgb || width <= 0 || height <= 0 || width % P || height % P)
            throw std::runtime_error("vision: RGB sides must be positive multiples of the patch size");
        const size_t gh = size_t(height / P), gw = size_t(width / P), N = product({gh, gw});
        const size_t lh = (gh + M - 1) / M, lw = (gw + M - 1) / M, cells = product({lh, lw});
        if (N > size_t(patches) || lh * (lw + 1) + 2 > size_t(g.max_tokens))
            throw std::runtime_error("vision: image exceeds the model's span limit");
        nh = int(lh); nw = int(lw);
        const size_t patch_dim = product({3, size_t(P), size_t(P)}), unfold_dim = product({size_t(D), size_t(M), size_t(M)});
        const size_t xn_count = std::max(N * std::max(size_t(D), patch_dim), cells * size_t(g.out));
        const size_t mid_count = std::max(N * size_t(I), cells * unfold_dim);
        const size_t o_count = std::max(N * size_t(D), cells * size_t(g.out));
        const size_t y_count = std::max(N * size_t(std::max(3 * D, 2 * I)), cells * size_t(g.out));
        const size_t small_count = size_t(std::max(3 * D, g.out));
        const size_t fixed = product({N, patch_dim}) + product({N, size_t(D), 4}) + y_count * 4 +
                             (xn_count + mid_count + o_count + 3 * N * size_t(D) + small_count) * 2;
        core::OnDevice on(device);
        size_t free = 0, total = 0;
        if (rt::mem_info(&free, &total) != rt::Result::success) throw std::runtime_error("vision: memory query failed");
        const size_t limit = core::max_alloc_bytes();
        const auto& dev = core::Runtime::get().device();
        size_t qb = std::min(N, dev.get_info<sycl::info::device::max_work_group_size>());
        // At most one reported workgroup's worth of independent row normalizations; memory can reduce the batch.
        const size_t reserve = free / 32, attention_weight = N * size_t(hd) * 2;
        const size_t input_conversion = std::max({N * std::max({patch_dim, size_t(D), size_t(I)}), cells * unfold_dim,
                                                  cells * size_t(g.out)}) * 2;
        if (free <= fixed + input_conversion + reserve + 2 * attention_weight)
            throw std::runtime_error("vision: not enough memory beside the model for this image");
        const size_t room = free - fixed - input_conversion - reserve;
        qb = std::max(size_t(1), std::min({qb, limit / (N * 4), room / (N * 8 + 1)}));
        const size_t scores = product({qb, N});
        const size_t conversion = std::max(input_conversion, scores * 2);
        const size_t extra = scores * 6 + conversion + reserve;
        if (free <= fixed + extra + 2 * attention_weight) throw std::runtime_error("vision: attention workspace does not fit");
        const size_t weight_bytes = std::min({largest_weight * 2, limit, (free - fixed - extra) / 2});
        const size_t weight_elems = weight_bytes / 2;
        if (weight_bytes < attention_weight || weight_elems < std::max({patch_dim, unfold_dim, size_t(I), size_t(g.out)}))
            throw std::runtime_error("vision: the device cannot hold a weight row and its conversion buffer");
        VisionWork work(device);
        sycl::queue& q = work.stream;
        auto* pixels = work.take<uint8_t>(N * patch_dim);
        auto* x = work.take<float>(N * size_t(D));
        auto* y = work.take<float>(y_count);
        auto* xn = work.take<uint16_t>(xn_count);
        auto* mid = work.take<uint16_t>(mid_count);
        auto* out = work.take<uint16_t>(o_count);
        auto* queries = work.take<uint16_t>(N * size_t(D));
        auto* keys = work.take<uint16_t>(N * size_t(D));
        auto* values = work.take<uint16_t>(N * size_t(D));
        auto* small = work.take<uint16_t>(small_count);
        auto* weights = work.take<uint16_t>(weight_elems);
        auto* s = work.take<float>(scores);
        auto* prob = work.take<uint16_t>(scores);
        workspace = work.bytes + conversion + std::max(weight_bytes, attention_weight);
        auto small_copy = [&](const std::string& name) {
            const auto& t = tensor(name);
            q.memcpy(small, t.data(), t.size() * 2);
        };
        auto linear = [&](const uint16_t* input, const std::string& name, float* output, size_t rows,
                          size_t columns, size_t inner) {
            const auto& t = tensor(name);
            const size_t stripe = std::min(columns, weight_elems / inner);
            for (size_t col = 0; col < columns; col += stripe) {
                const size_t n = std::min(stripe, columns - col);
                q.memcpy(weights, t.data() + col * inner, n * inner * 2);
                work.gemm->bf16(input, weights, output + col, int64_t(rows), int64_t(n), int64_t(inner), int64_t(columns));
            }
        };
        auto normalize = [&](const std::string& name) {
            small_copy(name);
            const float eps = g.eps;
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> item) {
                const size_t r = item[0];
                float sum = 0;
                for (int e = 0; e < D; ++e) { const float v = x[r * D + e]; sum += v * v; }
                const float inv = 1.0f / sycl::sqrt(sum / float(D) + eps);
                for (int e = 0; e < D; ++e) xn[r * D + e] = vision_bits(vision_float(small[e]) * (x[r * D + e] * inv));
            });
        };
        auto add = [&](const std::string& bias, bool initial) {
            if (!bias.empty()) small_copy(bias);
            const bool biased = !bias.empty();
            q.parallel_for(sycl::range<1>(N * size_t(D)), [=](sycl::id<1> item) {
                const size_t i = item[0];
                float v = y[i];
                if (!initial) v = x[i] + v;
                if (biased) v += vision_float(small[i % D]);
                x[i] = v;
            });
        };
        q.memcpy(pixels, rgb, N * patch_dim);
        q.parallel_for(sycl::range<1>(N * patch_dim), [=](sycl::id<1> item) {
            const size_t i = item[0], r = i / patch_dim, e = i % patch_dim, plane = size_t(P) * P;
            const size_t c = e / plane, py = (r / gw) * P + (e % plane) / P, px = (r % gw) * P + e % P;
            const float v = (float(pixels[(py * size_t(width) + px) * 3 + c]) / 255.0f - 0.5f) / 0.5f;
            xn[i] = vision_bits(v);
        });
        linear(xn, "vision.patch_embed.proj.weight", y, N, size_t(D), patch_dim);
        add("vision.patch_embed.proj.bias", true);
        const float theta = g.theta, scale = 1.0f / std::sqrt(float(hd));
        for (int block = 0; block < g.layers; ++block) {
            const std::string prefix = "vision.blocks." + std::to_string(block) + ".";
            normalize(prefix + "norm1.weight");
            linear(xn, prefix + "attn.wqkv.weight", y, N, size_t(D) * 3, size_t(D));
            small_copy(prefix + "attn.wqkv.bias");
            q.parallel_for(sycl::range<1>(N * size_t(H) * size_t(hd / 2)), [=](sycl::id<1> item) {
                const size_t z = item[0], d = z % (hd / 2), head = (z / (hd / 2)) % H, row = z / (size_t(H) * (hd / 2));
                const size_t f = d % (hd / 4);
                const int axis_dim = hd / 2;
                const float freq = 1.0f / sycl::pow(theta, float(2 * f) / float(axis_dim));
                const size_t coordinate = d < size_t(hd / 4) ? row / gw : row % gw;
                const float angle = float(coordinate) * freq;
                const float sn = sycl::sin(angle), cs = sycl::cos(angle);
                const size_t base = head * hd + d, at = head * N * hd + row * hd + d;
                for (int k = 0; k < 2; ++k) {
                    const size_t col = size_t(k) * D + base;
                    const float a = vision_float(vision_bits(y[row * (size_t(D) * 3) + col] + vision_float(small[col])));
                    const float b = vision_float(vision_bits(y[row * (size_t(D) * 3) + col + hd / 2] + vision_float(small[col + hd / 2])));
                    uint16_t* dest = k == 0 ? queries : keys;
                    dest[at] = vision_bits(a * cs - b * sn);
                    dest[at + hd / 2] = vision_bits(b * cs + a * sn);
                }
                for (int k = 0; k < 2; ++k) {
                    const size_t e = d + size_t(k) * (hd / 2), col = 2 * size_t(D) + head * hd + e;
                    values[head * size_t(hd) * N + e * N + row] = vision_bits(y[row * (size_t(D) * 3) + col] + vision_float(small[col]));
                }
            });
            for (size_t q0 = 0; q0 < N; q0 += qb) {
                const size_t nq = std::min(qb, N - q0);
                for (int head = 0; head < H; ++head) {
                    work.gemm->bf16(queries + size_t(head) * N * hd + q0 * hd, keys + size_t(head) * N * hd,
                                    s, int64_t(nq), int64_t(N), hd);
                    q.parallel_for(sycl::range<1>(nq), [=](sycl::id<1> item) {
                        const size_t row = item[0];
                        float largest = -std::numeric_limits<float>::max();
                        for (size_t k = 0; k < N; ++k) largest = sycl::fmax(largest, s[row * N + k] * scale);
                        float sum = 0;
                        for (size_t k = 0; k < N; ++k) sum += sycl::exp(s[row * N + k] * scale - largest);
                        for (size_t k = 0; k < N; ++k) prob[row * N + k] = vision_bits(sycl::exp(s[row * N + k] * scale - largest) / sum);
                    });
                    work.gemm->bf16(prob, values + size_t(head) * hd * N, y, int64_t(nq), hd, int64_t(N));
                    q.parallel_for(sycl::range<1>(nq * size_t(hd)), [=](sycl::id<1> item) {
                        const size_t i = item[0];
                        out[(q0 + i / hd) * D + size_t(head) * hd + i % hd] = vision_bits(y[i]);
                    });
                }
            }
            linear(out, prefix + "attn.wo.weight", y, N, size_t(D), size_t(D));
            add(prefix + "attn.wo.bias", false);
            normalize(prefix + "norm2.weight");
            linear(xn, prefix + "mlp.w1.weight", y, N, size_t(I) * 2, size_t(D));
            q.parallel_for(sycl::range<1>(N * size_t(I)), [=](sycl::id<1> item) {
                const size_t i = item[0], row = i / I, e = i % I;
                const float gate = vision_float(vision_bits(y[row * (size_t(I) * 2) + e]));
                const float up = vision_float(vision_bits(y[row * (size_t(I) * 2) + I + e]));
                mid[i] = vision_bits(vision_float(vision_bits(gate / (1.0f + sycl::exp(-gate)))) * up);
            });
            linear(mid, prefix + "mlp.w2.weight", y, N, size_t(D), size_t(I));
            add("", false);
        }
        normalize("vision.norm.weight");
        q.parallel_for(sycl::range<1>(cells * unfold_dim), [=](sycl::id<1> item) {
            const size_t i = item[0], cell = i / unfold_dim, f = i % unfold_dim, mm = size_t(M) * M;
            const size_t c = f / mm, k = f % mm, row = (cell / lw) * M + k / M, col = (cell % lw) * M + k % M;
            mid[i] = row < gh && col < gw ? xn[(row * gw + col) * D + c] : uint16_t(0);
        });
        linear(mid, "aligner.w1.weight", y, cells, size_t(g.out), unfold_dim);
        const int output_dim = g.out;
        small_copy("aligner.w1.bias");
        q.parallel_for(sycl::range<1>(cells * size_t(output_dim)), [=](sycl::id<1> item) {
            const size_t i = item[0];
            const float v = vision_float(vision_bits(y[i] + vision_float(small[i % output_dim])));
            xn[i] = vision_bits(0.5f * v * (1.0f + sycl::erf(v * 0.70710678118654752f)));
        });
        linear(xn, "aligner.w2.weight", y, cells, size_t(output_dim), size_t(output_dim));
        small_copy("aligner.w2.bias");
        q.parallel_for(sycl::range<1>(cells * size_t(output_dim)), [=](sycl::id<1> item) {
            const size_t i = item[0];
            out[i] = vision_bits(y[i] + vision_float(small[i % output_dim]));
        });
        result.resize(cells * size_t(output_dim));
        q.memcpy(result.data(), out, result.size() * 2).wait_and_throw();
        if (!std::all_of(result.begin(), result.end(), [](uint16_t v) { return std::isfinite(vision_float(v)); }))
            throw std::runtime_error("vision: non-finite image embedding");
        return true;
    }
};

VisionTower::VisionTower() = default;
VisionTower::~VisionTower() = default;
void VisionTower::release() { p_.reset(); }
bool VisionTower::loaded() const { return bool(p_); }
int VisionTower::device() const { return p_ ? p_->device : -1; }
int VisionTower::max_patches() const { return p_ ? p_->patches : 0; }
size_t VisionTower::host_bytes() const { return p_ ? p_->host : 0; }
size_t VisionTower::workspace_bytes() const { return p_ ? p_->workspace : 0; }

bool VisionTower::load(const VisionGeo& geo, int device, const Reader& read, std::string& err) try {
    release();
    const int D = geo.dim, I = geo.inter, P = geo.patch, M = geo.merge;
    if (device < 0 || geo.layers <= 0 || D <= 0 || D > std::numeric_limits<int>::max() / 3 ||
        I <= 0 || I > std::numeric_limits<int>::max() / 2 || P <= 0 || M <= 0 || geo.out <= 0 || geo.heads <= 0 ||
        D % geo.heads || (D / geo.heads) % 4 || geo.max_tokens < 5 || !std::isfinite(geo.theta) || geo.theta <= 0 ||
        !std::isfinite(geo.eps) || geo.eps <= 0)
        throw std::runtime_error("vision: invalid geometry");
    auto next = std::make_unique<Impl>(); next->g = geo; next->device = device;
    size_t cells = 0;
    for (int h = 1; h <= geo.max_tokens - 2; ++h) {
        const int w = (geo.max_tokens - 2) / h - 1;
        if (w > 0) cells = std::max(cells, size_t(h) * w);
    }
    const size_t patches = product({cells, size_t(M), size_t(M)});
    if (patches > size_t(std::numeric_limits<int>::max())) throw std::runtime_error("vision: patch count overflow");
    next->patches = int(patches);
    const size_t patch_columns = product({3, size_t(P), size_t(P)});
    const size_t merged_columns = product({size_t(D), size_t(M), size_t(M)});
    if (patch_columns > size_t(INT64_MAX) || merged_columns > size_t(INT64_MAX))
        throw std::runtime_error("vision: weight dimensions overflow");
    std::map<std::string, std::vector<int64_t>> expected = {
        {"vision.patch_embed.proj.weight", {D, int64_t(patch_columns)}}, {"vision.patch_embed.proj.bias", {D}},
        {"vision.norm.weight", {D}}, {"aligner.w1.weight", {geo.out, int64_t(merged_columns)}},
        {"aligner.w1.bias", {geo.out}}, {"aligner.w2.weight", {geo.out, geo.out}}, {"aligner.w2.bias", {geo.out}}};
    for (int b = 0; b < geo.layers; ++b) {
        const std::string prefix = "vision.blocks." + std::to_string(b) + ".";
        for (const char* name : {"norm1.weight", "norm2.weight", "attn.wo.bias"}) expected[prefix + name] = {D};
        expected[prefix + "attn.wqkv.weight"] = {int64_t(3) * D, D};
        expected[prefix + "attn.wqkv.bias"] = {int64_t(3) * D};
        expected[prefix + "attn.wo.weight"] = {D, D};
        expected[prefix + "mlp.w1.weight"] = {int64_t(2) * I, D};
        expected[prefix + "mlp.w2.weight"] = {D, I};
    }
    for (const auto& item : expected) {
        std::vector<int64_t> shape;
        if (!read(item.first, shape, nullptr) || shape != item.second)
            throw std::runtime_error("vision: missing tensor or unexpected shape: " + item.first);
    }
    for (const auto& item : expected) {
        size_t count = 1;
        for (int64_t dim : item.second) count = product({count, size_t(dim)});
        auto& data = next->tensors[item.first]; data.resize(count);
        std::vector<int64_t> shape;
        if (!read(item.first, shape, data.data())) throw std::runtime_error("vision: tensor read failed: " + item.first);
        next->host += product({count, size_t(2)});
        if (item.second.size() == 2) next->largest_weight = std::max(next->largest_weight, count);
    }
    p_ = std::move(next);
    return true;
} catch (const std::exception& e) { err = e.what(); return false; }

bool VisionTower::encode(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& rows, int& nh, int& nw,
                         std::string& err) try {
    rows.clear(); nh = nw = 0;
    if (!p_) throw std::runtime_error("vision: tower is not loaded");
    return p_->run(rgb, width, height, rows, nh, nw);
} catch (const std::exception& e) { rows.clear(); err = e.what(); return false; }

}  // namespace strata::ds4
