// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/lora_probe.cpp - strata-image-lora-probe: a LoRA adapter file read (its keys brought to the DiT's names,
// its entries and metadata), and the delta's arithmetic on the GPU against a CPU reference.
//
//   strata-image-lora-probe list LORA.safetensors|LORA.gguf      the entries and the metadata
//   strata-image-lora-probe selftest                             y += scale (x A^T) B^T against FP64 (random)
//
// `list` reads no GPU: it is the reader's check.  `selftest` builds a small adapter and two products, with and
// without a per-row scale, and prints how far the GPU's output is from the reference's.
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/image_lora.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace gpu = strata::gpu;

std::string clip(const std::string& s, size_t n = 160) {
    std::string o;
    for (char c : s) {
        if (o.size() >= n) {
            o += "...";
            break;
        }
        o += (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
    }
    return o;
}

int list(const std::string& path) {
    const auto f = strata::image::LoraFile::open(path);
    std::printf("%s: %zu entries, %.1f MiB on the host\n", path.c_str(), f->entries().size(),
                (double) f->host_bytes() / 1048576.0);
    for (const auto& [k, v] : f->metadata()) std::printf("  meta %s = %s\n", k.c_str(), clip(v).c_str());
    for (const auto& e : f->entries()) {
        std::printf("  %-56s %-5s rank %-4lld %s %lld x %lld scale %.4f%s%s\n", e.key.c_str(),
                    strata::image::lora_kind_name(e.kind), (long long) e.rank, "in/out", (long long) e.cols,
                    (long long) e.rows, e.scale(), e.dora.empty() ? "" : " dora",
                    e.extra.empty() ? "" : " +tensors");
    }
    return 0;
}

struct Ref {
    double dot = 0, ng = 0, nw = 0, err = 0, mag = 0, mx = 0;
};

void compare(const char* what, const std::vector<float>& got, const std::vector<double>& want) {
    Ref r;
    for (size_t i = 0; i < got.size(); ++i) {
        r.dot += (double) got[i] * want[i];
        r.ng += (double) got[i] * got[i];
        r.nw += want[i] * want[i];
        const double e = std::fabs((double) got[i] - want[i]);
        r.err += e;
        r.mag += std::fabs(want[i]);
        r.mx = std::max(r.mx, e);
    }
    std::printf("%-22s cos %.7f  rel %.3g  max %.3g\n", what, r.dot / std::sqrt(r.ng * r.nw + 1e-30), r.err / r.mag,
                r.mx);
}

// one entry's delta: y[t][c] += scale * row_scale[t] * sum_r (x[t] . a[r]) b[c][r]
void selftest_case(void* s, bool rows_scaled) {
    namespace si = strata::image;
    constexpr int64_t T = 8, K = 64, R = 8, N = 32;
    std::mt19937 rng(rows_scaled ? 7u : 3u);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    si::LoraEntry e;
    e.key = "selftest";
    e.kind = si::LoraKind::lora;
    e.cols = K;
    e.rows = N;
    e.rank = R;
    std::vector<float> a((size_t) (R * K)), b((size_t) (N * R)), x((size_t) (T * K)), rs((size_t) T, 1.0f);
    for (float& v : a) v = nd(rng);
    for (float& v : b) v = nd(rng);
    for (float& v : x) v = nd(rng);
    for (int64_t t = 0; t < T; ++t) rs[(size_t) t] = std::exp2((float) (t % 3));   // 1, 2, 4
    si::lora_to_f16(a, e.a);
    si::lora_to_f16(b, e.b);
    const float scale = 0.8f;

    si::LoraDeviceEntry dev;
    std::string err;
    if (!dev.upload(e, s, err)) throw std::runtime_error(err);
    strata::prefill::Gemm gemm;
    if (!gemm.init(s, 0, err)) throw std::runtime_error(err);
    auto* dx = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * K) * 2));
    auto* dy = static_cast<float*>(gpu::alloc_device((size_t) (T * N) * 4));
    auto* drs = static_cast<float*>(gpu::alloc_device((size_t) T * 4));
    auto* tf = static_cast<float*>(gpu::alloc_device((size_t) (T * R) * 4));
    auto* t16 = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * R) * 2));
    std::vector<uint16_t> x16((size_t) (T * K));
    si::lora_to_f16(x, x16);
    std::vector<float> zero((size_t) (T * N), 0.0f);
    gpu::copy(dx, x16.data(), x16.size() * 2);
    gpu::copy(dy, zero.data(), zero.size() * 4);
    gpu::copy(drs, rs.data(), rs.size() * 4);
    dev.add(dx, T, dy, N, 0, N, scale, rows_scaled ? drs : nullptr, gemm, tf, t16, s);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) (T * N));
    gpu::copy(got.data(), dy, got.size() * 4);
    // the reference, in FP64 from the same FP16 inputs the GPU took
    std::vector<double> want((size_t) (T * N), 0.0);
    std::vector<double> xd((size_t) (T * K)), ad((size_t) (R * K)), bd((size_t) (N * R));
    for (int64_t i = 0; i < T * K; ++i) xd[(size_t) i] = strata::kernels::f32_from_f16(x16[(size_t) i]);
    for (int64_t i = 0; i < R * K; ++i) ad[(size_t) i] = strata::kernels::f32_from_f16(e.a[(size_t) i]);
    for (int64_t i = 0; i < N * R; ++i) bd[(size_t) i] = strata::kernels::f32_from_f16(e.b[(size_t) i]);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t r = 0; r < R; ++r) {
            double v = 0;
            for (int64_t k = 0; k < K; ++k) v += xd[(size_t) (t * K + k)] * ad[(size_t) (r * K + k)];
            for (int64_t c = 0; c < N; ++c)
                want[(size_t) (t * N + c)] += v * bd[(size_t) (c * R + r)] * scale * (rows_scaled ? rs[(size_t) t] : 1.0);
        }
    compare(rows_scaled ? "row-scaled (W8A8)" : "plain (FP16)", got, want);
    gpu::free(dx);
    gpu::free(dy);
    gpu::free(drs);
    gpu::free(tf);
    gpu::free(t16);
}

int selftest() {
    void* s = strata::core::Runtime::get().create_stream();
    selftest_case(s, false);
    selftest_case(s, true);
    strata::core::Runtime::get().destroy_stream(static_cast<sycl::queue*>(s));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string what = argc > 1 ? argv[1] : "";
        if (what == "list" && argc == 3) return list(argv[2]);
        if (what == "selftest" && argc == 2) return selftest();
        std::fprintf(stderr, "usage: %s list LORA.safetensors|LORA.gguf | %s selftest\n", argv[0], argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-image-lora-probe: %s\n", e.what());
        return 1;
    }
}