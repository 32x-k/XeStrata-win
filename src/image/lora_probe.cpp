// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/lora_probe.cpp - strata-image-lora-probe: a LoRA adapter file read (its keys brought to the DiT's names,
// its entries and metadata), the delta's arithmetic on the GPU against a CPU reference, and the deltas a DiT binds.
//
//   strata-image-lora-probe list LORA.safetensors|LORA.gguf      the entries and the metadata
//   strata-image-lora-probe selftest                             y += scale (x A^T) B^T against FP64 (random)
//   strata-image-lora-probe map DIT.gguf [LORA [scale]]          the weights adapters may take, and where a file's keys
//                                                                land (or that they do not)
//   strata-image-lora-probe step DIT.gguf OUT.bin [LORA [scale]] one forward of a fixed context and noise, its
//                                                                velocity written as raw FP32 (a base run has no LORA)
//   strata-image-lora-probe moss-map DIR [LORA [scale]]          the same two for MOSS-Image's DiT (MossDit)
//   strata-image-lora-probe moss-step DIR OUT.bin [LORA [scale]]
//   strata-image-lora-probe mf-map DIT.safetensors [LORA [scale]]  the same two for Mage-Flow's DiT (MfDit)
//   strata-image-lora-probe mf-step DIT.safetensors OUT.bin [LORA [scale]]
//
// `list` and `map` read no GPU: they are the readers' checks.  `selftest` builds a small adapter and two products, with
// and without a per-row scale, and prints how far the GPU's output is from the reference's.  `step` is how a LoRA's
// delta against the same delta baked into the weights is compared (the parity test of the catalogue's records): the
// same C++ forward on the same noise, the model loaded twice.  MOSS_*_HW, MOSS_*_TEXT: the latent's and the text's
// rows (2 and 4); MOSS_STEP_Q8=1: the DiT's blocks in 8 bits (else FP16, the same on both sides of a parity);
// MOSS_STEP_DROP_TEXT=1: the text embedder's weights dropped and read again, a forward each (the same velocity, an
// adapter's layer kept on the weight through the load); MOSS_STEP_BLOCK_MIB: the blocks' VRAM (the rest streamed).
// MF_STEP_HW, MF_STEP_TEXT: the latent's rows a side (2) and the text's (4); MF_STEP_Q8=1: the blocks in 8 bits (else
// FP16); MF_STEP_BLOCK_MIB: the blocks' VRAM (the rest streamed from RAM); MF_STEP_REFS: an edit's reference images
// (one, at the latent size, its own rows after the target's).
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/image_lora.hpp"
#include "strata/image/mf_dit.hpp"
#include "strata/image/moss_dit.hpp"
#include "strata/image/moss_pipeline.hpp"
#include "strata/image/qi_dit.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
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

// The cosine of the GPU's output and the reference's (an accumulator of 4096 values and more stays FP32 in both), and
// the relative error; true when the cosine is above 0.9999, the selftest's pass.
bool compare(const char* what, const std::vector<float>& got, const std::vector<double>& want) {
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
    const double c = r.dot / std::sqrt(r.ng * r.nw + 1e-30);
    std::printf("%-22s cos %.7f  rel %.3g  max %.3g%s\n", what, c, r.err / r.mag, r.mx,
                c > 0.9999 ? "" : "  WRONG");
    return c > 0.9999;
}

// one entry's delta: y[t][row0 + c] += scale * row_scale[t] * sum_r (x[t] . a[r]) b[src_row + c][r], the entry's
// rows [src_row, src_row + n) at the weight's [row0, row0 + n)
void selftest_case(void* s, int64_t T, int64_t K, int64_t R, int64_t N, bool rows_scaled, int64_t src_row = 0,
                   int64_t row0 = 0, int64_t n = 0) {
    namespace si = strata::image;
    if (n <= 0) n = N - src_row;
    std::mt19937 rng((uint32_t) (rows_scaled ? 7u : 3u) ^ (uint32_t) (T * 31 + K * 17 + R * 13 + N * 7));
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
    dev.add(dx, T, dy, N, src_row, row0, n, scale, rows_scaled ? drs : nullptr, gemm, tf, t16, s);
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
            for (int64_t c = 0; c < n; ++c)
                want[(size_t) (t * N + row0 + c)] +=
                    v * bd[(size_t) ((src_row + c) * R + r)] * scale * (rows_scaled ? rs[(size_t) t] : 1.0);
        }
    char label[64];
    std::snprintf(label, sizeof label, "%s %lldx%lld rank %lld", rows_scaled ? "row-scaled" : "plain",
                  (long long) T, (long long) N, (long long) R);
    const bool ok = compare(label, got, want);
    gpu::free(dx);
    gpu::free(dy);
    gpu::free(drs);
    gpu::free(tf);
    gpu::free(t16);
    if (!ok) throw std::runtime_error(std::string("the delta of ") + label + " is not the reference's");
}

int selftest() {
    void* s = strata::core::Runtime::get().create_stream();
    try {
        selftest_case(s, 8, 64, 8, 32, false);
        selftest_case(s, 8, 64, 8, 32, true);
        // the rows of a fused weight: the entry's second half into the weight's second half
        selftest_case(s, 8, 64, 8, 64, false, 32, 32, 32);
        // a rank the matrix engines' accumulating product takes (K a multiple of 32)
        selftest_case(s, 64, 64, 64, 4096, false);
    } catch (...) {
        strata::core::Runtime::get().destroy_stream(static_cast<sycl::queue*>(s));
        throw;
    }
    strata::core::Runtime::get().destroy_stream(static_cast<sycl::queue*>(s));
    return 0;
}

// The DiT's table of the weights adapters may take, and where each entry of `lora` lands on it (a line a mapped
// weight, MISSING a key the DiT does not take, SHAPE a weight whose rows and columns are not the entry's).
int map(const std::string& dit_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    si::QiDit dit;
    std::string err;
    if (!dit.open(dit_path, nullptr, err, true)) throw std::runtime_error(err);
    const auto targets = dit.lora_targets();
    std::printf("targets %zu\n", targets.size());
    for (const auto& t : targets)
        std::printf("  %-52s rows %lld cols %lld%s\n", t.name.c_str(), (long long) t.rows, (long long) t.cols,
                    t.row0 != 0 ? (" at " + std::to_string(t.row0)).c_str() : "");
    if (lora_path.empty()) return 0;
    const auto f = si::LoraFile::open(lora_path);
    size_t mapped = 0;
    for (const auto& e : f->entries()) {
        const si::QiDit::LoraTarget* t = nullptr;
        for (const auto& c : targets)
            if (c.name == e.key) {
                t = &c;
                break;
            }
        if (t == nullptr) {
            std::printf("MISSING %-46s %-5s rank %-4lld %lld x %lld\n", e.key.c_str(), si::lora_kind_name(e.kind),
                        (long long) e.rank, (long long) e.cols, (long long) e.rows);
        } else if (e.rows != t->rows || e.cols != t->cols) {
            std::printf("SHAPE   %-46s %lld x %lld, the weight %lld x %lld\n", e.key.c_str(), (long long) e.cols,
                        (long long) e.rows, (long long) t->cols, (long long) t->rows);
        } else {
            ++mapped;
            std::printf("mapped  %-46s %-5s rank %-4lld rows %lld at the weight's %lld, scale %.4f\n", e.key.c_str(),
                        si::lora_kind_name(e.kind), (long long) e.rank, (long long) e.rows, (long long) t->row0,
                        e.scale() * scale);
        }
    }
    std::printf("%zu of %zu entries mapped\n", mapped, f->entries().size());
    return mapped == f->entries().size() ? 0 : 1;
}

// One forward of a fixed context and noise (philox_randn of seed 42), the velocity written as raw FP32: the same
// numbers as an adapterless run mean the adapter's delta compares equal to the same delta baked into the weights.
int step(const std::string& dit_path, const std::string& out_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    void* s = strata::core::Runtime::get().create_stream();
    si::QiDit dit;
    std::string err;
    if (!dit.open(dit_path, s, err)) throw std::runtime_error(err);
    if (!lora_path.empty()) {
        std::vector<si::LoraUse> loras;
        loras.push_back({si::LoraFile::open(lora_path), scale});
        if (!dit.set_loras(loras, err)) throw std::runtime_error(err);
    }
    const char* hw_env = std::getenv("QI_STEP_HW");
    const char* text_env = std::getenv("QI_STEP_TEXT");
    const int64_t hw = hw_env != nullptr ? std::strtoll(hw_env, nullptr, 10) : 8;
    const int64_t n_text = text_env != nullptr ? std::strtoll(text_env, nullptr, 10) : 4;
    std::vector<float> ctx((size_t) (n_text * dit.hidden()));
    for (size_t i = 0; i < ctx.size(); ++i) ctx[i] = (float) ((int64_t) ((i * 37) % 101) - 50) / 50.0f;   // exact both sides
    dit.set_context(ctx.data(), n_text);
    const int64_t n = dit.channels() * hw * hw;
    const auto noise = si::philox_randn(42, (uint64_t) n);
    float *x = nullptr, *v = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&v, (size_t) n * 4)) throw std::runtime_error("alloc");
    gpu::copy(x, noise.data(), noise.size() * 4);
    dit.forward(x, 500.0f, hw, hw, false, v);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) n);
    gpu::copy(got.data(), v, got.size() * 4);
    // the prefix cache: a step after the first one reads the context's K and V from it (the adapters' included)
    float* vp = nullptr;
    if (!gpu::alloc_device(&vp, (size_t) n * 4)) throw std::runtime_error("alloc");
    dit.forward(x, 500.0f, hw, hw, true, vp);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> cached((size_t) n);
    gpu::copy(cached.data(), vp, cached.size() * 4);
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; ++i) {
        dot += (double) got[(size_t) i] * cached[(size_t) i];
        na += (double) got[(size_t) i] * got[(size_t) i];
        nb += (double) cached[(size_t) i] * cached[(size_t) i];
    }
    std::printf("prefix cache vs the whole sequence: cos %.7f\n", dot / std::sqrt(na * nb));
    gpu::free(vp);
    std::ofstream fo(out_path, std::ios::binary);
    if (!fo) throw std::runtime_error("cannot write " + out_path);
    fo.write(reinterpret_cast<const char*>(got.data()), (std::streamsize) (got.size() * 4));
    double norm = 0;
    for (float g : got) norm += (double) g * g;
    std::printf("velocity %lld values, norm %.6f, first %.6f %.6f %.6f\n", (long long) n, std::sqrt(norm), got[0],
                got[1], got[2]);
    std::printf("lora %.1f MiB on the GPU%s%s\n", (double) dit.lora_bytes() / 1048576.0,
                dit.lora_key().empty() ? " (none)" : "", dit.lora_key().empty() ? "" : dit.lora_key().c_str());
    std::printf("products: %s\n", strata::prefill::Gemm::i8_usable(s) ? "int8 (W8A8) where the weights allow it"
                                                                     : "FP16/native (no int8 products)");
    gpu::free(x);
    gpu::free(v);
    return 0;
}

}  // namespace

namespace {

// MOSS-Image's table of the weights adapters may take (one line a name: its entry's rows and columns), and where each
// entry of `lora` lands on it (a line a mapped weight, MISSING a key the DiT does not take, SHAPE a weight whose rows
// and columns are not the entry's).
int moss_map(const std::string& dit_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    si::MossDit dit;
    std::string err;
    if (!dit.open(dit_path, nullptr, err, true)) throw std::runtime_error(err);
    const auto targets = dit.lora_targets();
    std::vector<si::MossDit::LoraTarget> groups;   // one per name: the rows and columns of the weights it covers
    for (const auto& t : targets) {
        auto g = std::find_if(groups.begin(), groups.end(), [&](const auto& x) { return x.name == t.name; });
        if (g == groups.end()) {
            groups.push_back(t);
        } else {
            g->rows = std::max(g->rows, t.src_row + t.rows);
        }
    }
    std::printf("targets %zu\n", groups.size());
    for (const auto& g : groups)
        std::printf("  %-52s rows %lld cols %lld\n", g.name.c_str(), (long long) g.rows, (long long) g.cols);
    if (lora_path.empty()) return 0;
    const auto f = si::LoraFile::open(lora_path);
    size_t mapped = 0;
    for (const auto& e : f->entries()) {
        const si::MossDit::LoraTarget* g = nullptr;
        for (const auto& c : groups)
            if (c.name == e.key) {
                g = &c;
                break;
            }
        if (g == nullptr) {
            std::printf("MISSING %-46s %-5s rank %-4lld %lld x %lld\n", e.key.c_str(), si::lora_kind_name(e.kind),
                        (long long) e.rank, (long long) e.cols, (long long) e.rows);
        } else if (e.rows != g->rows || e.cols != g->cols) {
            std::printf("SHAPE   %-46s %lld x %lld, the weight %lld x %lld\n", e.key.c_str(), (long long) e.cols,
                        (long long) e.rows, (long long) g->cols, (long long) g->rows);
        } else {
            ++mapped;
            std::printf("mapped  %-46s %-5s rank %-4lld rows %lld, scale %.4f\n", e.key.c_str(),
                        si::lora_kind_name(e.kind), (long long) e.rank, (long long) e.rows, e.scale() * scale);
        }
    }
    std::printf("%zu of %zu entries mapped\n", mapped, f->entries().size());
    return mapped == f->entries().size() ? 0 : 1;
}

// One forward of a fixed context and noise (torch_cpu_randn of seed 42), the velocity written as raw FP32: the same
// numbers as an adapterless run of a model with the same delta in its weights mean the delta is applied as baked.
int moss_step(const std::string& dit_path, const std::string& out_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    void* s = strata::core::Runtime::get().create_stream();
    si::MossDit dit;
    std::string err;
    const bool q8 = std::getenv("MOSS_STEP_Q8") != nullptr;
    if (!dit.open(dit_path, s, err, false, q8)) throw std::runtime_error(err);
    if (const char* mib = std::getenv("MOSS_STEP_BLOCK_MIB"); mib != nullptr && *mib != 0) {   // VRAM of the blocks
        dit.set_block_budget((uint64_t) std::strtoull(mib, nullptr, 10) << 20);
        std::printf("the blocks' VRAM %s MiB, %d of %d resident\n", mib, dit.resident_blocks(), dit.blocks());
    }
    if (!lora_path.empty()) {
        std::vector<si::LoraUse> loras;
        loras.push_back({si::LoraFile::open(lora_path), scale});
        if (!dit.set_loras(loras, err)) throw std::runtime_error(err);
    }
    const char* hw_env = std::getenv("MOSS_STEP_HW");
    const char* text_env = std::getenv("MOSS_STEP_TEXT");
    const int64_t hw = hw_env != nullptr ? std::strtoll(hw_env, nullptr, 10) : 2;
    const int64_t n_text = text_env != nullptr ? std::strtoll(text_env, nullptr, 10) : 4;
    std::vector<float> ctx((size_t) (n_text * dit.text_dim()));
    for (size_t i = 0; i < ctx.size(); ++i) ctx[i] = (float) ((int64_t) ((i * 37) % 101) - 50) / 50.0f;   // exact both sides
    dit.set_context(0, ctx.data(), n_text);
    const int64_t Cl = dit.latent_channels(), n = Cl * hw * hw;
    const auto noise = si::torch_cpu_randn(42, (uint64_t) n);
    std::vector<float> tokens((size_t) n);
    for (int64_t c = 0; c < Cl; ++c)
        for (int64_t q = 0; q < hw * hw; ++q) tokens[(size_t) (q * Cl + c)] = noise[(size_t) (c * hw * hw + q)];
    float *x = nullptr, *v = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&v, (size_t) n * 4)) throw std::runtime_error("alloc");
    gpu::copy(x, tokens.data(), tokens.size() * 4);
    dit.forward(x, 500.0f, hw, hw, 1, v);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) n);
    gpu::copy(got.data(), v, got.size() * 4);
    if (const char* drop = std::getenv("MOSS_STEP_DROP_TEXT"); drop != nullptr) {
        // the text embedder's weights go (the pipeline gives them up for the blocks' room) and come back: a weight an
        // adapter is bound to keeps its layer across the load, and the velocity is the same again
        const uint64_t before = dit.lora_bytes();
        std::vector<float> rows_before((size_t) (n_text * dit.hidden()));
        gpu::copy(rows_before.data(), dit.text_rows(0), rows_before.size() * 4);
        if (std::strcmp(drop, "1") == 0) dit.drop_text();
        dit.set_context(0, ctx.data(), n_text);
        std::vector<float> rows_after((size_t) (n_text * dit.hidden()));
        gpu::copy(rows_after.data(), dit.text_rows(0), rows_after.size() * 4);
        double rd = 0, ra = 0, rb = 0;
        for (size_t i = 0; i < rows_after.size(); ++i) {
            rd += (double) rows_before[i] * rows_after[i];
            ra += (double) rows_before[i] * rows_before[i];
            rb += (double) rows_after[i] * rows_after[i];
        }
        std::printf("the context's text rows: cos %.7f, the same bytes %s\n", rd / std::sqrt(ra * rb + 1e-30),
                    rows_before == rows_after ? "yes" : "no");
        std::printf("the adapter's A and B: %llu bytes before the text embedder's weights went, %llu after them\n",
                    (unsigned long long) before, (unsigned long long) dit.lora_bytes());
        dit.forward(x, 500.0f, hw, hw, 1, v);
        if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
        std::vector<float> again((size_t) n);
        gpu::copy(again.data(), v, again.size() * 4);
        double dot = 0, na = 0, nb = 0;
        for (int64_t i = 0; i < n; ++i) {
            dot += (double) got[(size_t) i] * again[(size_t) i];
            na += (double) got[(size_t) i] * got[(size_t) i];
            nb += (double) again[(size_t) i] * again[(size_t) i];
        }
        std::printf("the text embedder read again: cos %.7f, the same bytes %s\n", dot / std::sqrt(na * nb + 1e-30),
                    got == again ? "yes" : "no");
    }
    std::ofstream fo(out_path, std::ios::binary);
    if (!fo) throw std::runtime_error("cannot write " + out_path);
    fo.write(reinterpret_cast<const char*>(got.data()), (std::streamsize) (got.size() * 4));
    double norm = 0;
    for (float g : got) norm += (double) g * g;
    std::printf("velocity %lld values, norm %.6f, first %.6f %.6f %.6f\n", (long long) n, std::sqrt(norm), got[0],
                got[1], got[2]);
    std::printf("lora %.1f MiB on the GPU, %d files\n", (double) dit.lora_bytes() / 1048576.0, dit.lora_count());
    std::printf("products: %s; blocks %s\n",
                strata::prefill::Gemm::i8_usable(s) ? "int8 (W8A8) where the weights allow it"
                                                    : "FP16/native (no int8 products)",
                q8 ? "8 bits" : "16 bits");
    gpu::free(x);
    gpu::free(v);
    return 0;
}

// Mage-Flow's table of the weights adapters may take (one line a name: its rows and columns), and where each entry of
// `lora` lands on it (a line a mapped weight, MISSING a key the DiT does not take, SHAPE a weight whose rows and
// columns are not the entry's).
int mf_map(const std::string& dit_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    si::MfDit dit;
    std::string err;
    if (!dit.open(dit_path, nullptr, err, true)) throw std::runtime_error(err);
    const auto targets = dit.lora_targets();
    std::printf("targets %zu\n", targets.size());
    for (const auto& t : targets) std::printf("  %-52s rows %lld cols %lld\n", t.name.c_str(), (long long) t.rows,
                                              (long long) t.cols);
    if (lora_path.empty()) return 0;
    const auto f = si::LoraFile::open(lora_path);
    size_t mapped = 0;
    for (const auto& e : f->entries()) {
        const si::MfDit::LoraTarget* t = nullptr;
        for (const auto& c : targets)
            if (c.name == e.key) {
                t = &c;
                break;
            }
        if (t == nullptr) {
            std::printf("MISSING %-46s %-5s rank %-4lld %lld x %lld\n", e.key.c_str(), si::lora_kind_name(e.kind),
                        (long long) e.rank, (long long) e.cols, (long long) e.rows);
        } else if (e.rows != t->rows || e.cols != t->cols) {
            std::printf("SHAPE   %-46s %lld x %lld, the weight %lld x %lld\n", e.key.c_str(), (long long) e.cols,
                        (long long) e.rows, (long long) t->cols, (long long) t->rows);
        } else {
            ++mapped;
            std::printf("mapped  %-46s %-5s rank %-4lld rows %lld, scale %.4f\n", e.key.c_str(),
                        si::lora_kind_name(e.kind), (long long) e.rank, (long long) e.rows, e.scale() * scale);
        }
    }
    std::printf("%zu of %zu entries mapped\n", mapped, f->entries().size());
    return mapped == f->entries().size() ? 0 : 1;
}

// One forward of a fixed context and noise (torch_cpu_randn of seed 42), the velocity written as raw FP32: the same
// numbers as an adapterless run of a model with the same delta in its weights mean the delta is applied as baked.
int mf_step(const std::string& dit_path, const std::string& out_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    void* s = strata::core::Runtime::get().create_stream();
    si::MfDit dit;
    std::string err;
    const bool q8 = std::getenv("MF_STEP_Q8") != nullptr;
    if (!dit.open(dit_path, s, err, false, q8)) throw std::runtime_error(err);
    if (const char* mib = std::getenv("MF_STEP_BLOCK_MIB"); mib != nullptr && *mib != 0) {   // VRAM of the blocks
        dit.set_block_budget((uint64_t) std::strtoull(mib, nullptr, 10) << 20);
        std::printf("the blocks' VRAM %s MiB, %d of %d resident\n", mib, dit.resident_blocks(), (int) dit.blocks());
    }
    if (!lora_path.empty()) {
        std::vector<si::LoraUse> loras;
        loras.push_back({si::LoraFile::open(lora_path), scale});
        if (!dit.set_loras(loras, err)) throw std::runtime_error(err);
    }
    const char* hw_env = std::getenv("MF_STEP_HW");
    const char* text_env = std::getenv("MF_STEP_TEXT");
    const int64_t hw = hw_env != nullptr ? std::strtoll(hw_env, nullptr, 10) : 2;
    const int64_t n_text = text_env != nullptr ? std::strtoll(text_env, nullptr, 10) : 4;
    std::vector<float> ctx((size_t) (n_text * dit.text_dim()));
    for (size_t i = 0; i < ctx.size(); ++i) ctx[i] = (float) ((int64_t) ((i * 37) % 101) - 50) / 50.0f;   // exact both sides
    dit.set_context(0, ctx.data(), n_text);
    // an edit: a reference the target's size, after its rows (the same on both sides)
    std::vector<float> rz;
    std::vector<std::pair<int64_t, int64_t>> rsz;
    if (const char* refs = std::getenv("MF_STEP_REFS"); refs != nullptr && *refs != 0) {
        for (int64_t j = 0; j < std::strtoll(refs, nullptr, 10); ++j) {
            const auto z = si::torch_cpu_randn(43 + (uint64_t) j, (uint64_t) (hw * hw * dit.latent_channels()));
            rz.insert(rz.end(), z.begin(), z.end());
            rsz.emplace_back(hw, hw);
        }
    }
    const int64_t Cl = dit.latent_channels(), n = Cl * hw * hw;
    if (!rz.empty()) dit.set_refs(rz.data(), rsz);
    const auto noise = si::torch_cpu_randn(42, (uint64_t) n);
    std::vector<float> tokens((size_t) n);
    for (int64_t c = 0; c < Cl; ++c)
        for (int64_t q = 0; q < hw * hw; ++q) tokens[(size_t) (q * Cl + c)] = noise[(size_t) (c * hw * hw + q)];
    float *x = nullptr, *v = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&v, (size_t) n * 4)) throw std::runtime_error("alloc");
    gpu::copy(x, tokens.data(), tokens.size() * 4);
    dit.forward(x, 0.5f, hw, hw, 1, v);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) n);
    gpu::copy(got.data(), v, got.size() * 4);
    std::ofstream fo(out_path, std::ios::binary);
    if (!fo) throw std::runtime_error("cannot write " + out_path);
    fo.write(reinterpret_cast<const char*>(got.data()), (std::streamsize) (got.size() * 4));
    double norm = 0;
    for (float g : got) norm += (double) g * g;
    std::printf("velocity %lld values, norm %.6f, first %.6f %.6f %.6f\n", (long long) n, std::sqrt(norm), got[0],
                got[1], got[2]);
    std::printf("lora %.1f MiB on the GPU, %d files\n", (double) dit.lora_bytes() / 1048576.0, dit.lora_count());
    std::printf("products: %s; blocks %s\n",
                strata::prefill::Gemm::i8_usable(s) ? "int8 (W8A8) where the weights allow it"
                                                    : "FP16/native (no int8 products)",
                q8 ? "8 bits" : "16 bits");
    gpu::free(x);
    gpu::free(v);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string what = argc > 1 ? argv[1] : "";
        if (what == "list" && argc == 3) return list(argv[2]);
        if (what == "selftest" && argc == 2) return selftest();
        if (what == "map" && (argc == 3 || argc == 4 || argc == 5))
            return map(argv[2], argc > 3 ? argv[3] : "", argc == 5 ? std::strtof(argv[4], nullptr) : 1.0f);
        if (what == "step" && (argc == 4 || argc == 5 || argc == 6))
            return step(argv[2], argv[3], argc > 4 ? argv[4] : "", argc == 6 ? std::strtof(argv[5], nullptr) : 1.0f);
        if (what == "moss-map" && (argc == 3 || argc == 4 || argc == 5))
            return moss_map(argv[2], argc > 3 ? argv[3] : "", argc == 5 ? std::strtof(argv[4], nullptr) : 1.0f);
        if (what == "moss-step" && (argc == 4 || argc == 5 || argc == 6))
            return moss_step(argv[2], argv[3], argc > 4 ? argv[4] : "", argc == 6 ? std::strtof(argv[5], nullptr) : 1.0f);
        if (what == "mf-map" && (argc == 3 || argc == 4 || argc == 5))
            return mf_map(argv[2], argc > 3 ? argv[3] : "", argc == 5 ? std::strtof(argv[4], nullptr) : 1.0f);
        if (what == "mf-step" && (argc == 4 || argc == 5 || argc == 6))
            return mf_step(argv[2], argv[3], argc > 4 ? argv[4] : "", argc == 6 ? std::strtof(argv[5], nullptr) : 1.0f);
        std::fprintf(stderr,
                     "usage: %s list LORA.safetensors|LORA.gguf | %s selftest | %s map DIT.gguf [LORA [scale]] | "
                     "%s step DIT.gguf OUT.bin [LORA [scale]] | %s moss-map DIR [LORA [scale]] | "
                     "%s moss-step DIR OUT.bin [LORA [scale]] | %s mf-map DIT [LORA [scale]] | "
                     "%s mf-step DIT OUT.bin [LORA [scale]]\n",
                     argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-image-lora-probe: %s\n", e.what());
        return 1;
    }
}