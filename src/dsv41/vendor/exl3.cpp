// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// EXL3's file layout and codebooks are exllamav3's (MIT, turboderp). The 16-value trellis tiles and 128-point
// Hadamard blocks below belong to that format; they impose no sub-group or work-group width on the device.
#include "strata/dsv41/exl3.hpp"
#include "../ds4_kernels.cuh"
#include "../ds4_numerics.cuh"
#include "../ds4_quant_sycl.hpp"
#include <atomic>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::dsv41k::exl3 {
namespace {
namespace sd = strata::ds4::sycl_detail;
std::atomic<int> errors{0};
thread_local const int* skip = nullptr;

template <typename F> void launch(sycl::queue& s, size_t n, F f) {
    try { sd::parallel(s, n, f); }
    catch (...) { errors.fetch_add(1, std::memory_order_relaxed); throw; }
}
void require(const Mat& m) {
    if (!supported(m)) {
        errors.fetch_add(1, std::memory_order_relaxed);
        throw std::invalid_argument("dsv41: unsupported EXL3 matrix layout");
    }
}
void had128(float* v) {
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
    for (int i = 0; i < 128; ++i) v[i] *= 0.088388347648f;
}
float inner(const Mat& m, int row, int col) {
    const int r = row % 16, c = col % 16;
    const int lane = (c % 8) * 4 + (r % 8) / 2;
    const int j = (r & 1) | ((r >> 3) << 1) | ((c >> 3) << 2);
    const int p = lane * 8 + j, end = (p + 257) * m.K;
    const int words = 8 * m.K, ld = m.ld > 0 ? m.ld : m.n / 16;
    const uint32_t* w = m.trellis + ((size_t) (row / 16) * ld + col / 16) * words;
    const int first = ((end - 16) / 32) % words, last = ((end - 1) / 32) % words;
    const int shift = ((end - 1) / 32 + 1) * 32 - end;
    const uint32_t state = (uint32_t) ((((uint64_t) w[first] << 32) | w[last]) >> shift) & 0xffffu;
    if (m.cb == 2) {
        const uint32_t x = state * 0x83DCD12Du;
        const uint32_t sum = (x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24);
        return sd::f16_value(sd::f16_bits(sycl::fma(sd::f16_value((uint16_t) (0x6400u + sum)),
                                                   sd::f16_value(0x1eee), sd::f16_value(0xc931))));
    }
    uint32_t x = m.cb == 1 ? state * 0xCBAC1FEDu : state * 89226354u + 64248484u;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return sd::f16_value(sd::f16_bits(sd::f16_value((uint16_t) x) + sd::f16_value((uint16_t) (x >> 16))));
}
template <typename Input>
void band(const Mat& m, Input input, int c0, float* out) {
    for (int c = 0; c < 128; ++c) out[c] = 0.0f;
    for (int r0 = 0; r0 < m.k; r0 += 128) {
        float x[128];
        for (int r = 0; r < 128; ++r) x[r] = input(r0 + r) * sd::f16_value(m.suh[r0 + r]);
        had128(x);
        for (int r = 0; r < 128; ++r)
            for (int c = 0; c < 128; ++c) out[c] = sycl::fma(x[r], inner(m, r0 + r, c0 + c), out[c]);
    }
    had128(out);
    for (int c = 0; c < 128; ++c) out[c] *= sd::f16_value(m.svh[c0 + c]);
}
Mat expert(const uint8_t* blob, const ExpertLayout& lay, int role, int embd, int ff) {
    Mat m;
    m.trellis = (const uint32_t*) (blob + lay.trellis[role]);
    m.suh = (const uint16_t*) (blob + lay.suh[role]);
    m.svh = (const uint16_t*) (blob + lay.svh[role]);
    m.k = role == 2 ? ff : embd;
    m.n = role == 2 ? embd : ff;
    m.K = lay.K[role]; m.cb = lay.cb;
    return m;
}
float swi(float g, float u, float limit) { return strata::ds4::gguf::iq::portable::swiglu(g, u, limit); }
struct Scratch {
    sycl::queue queue;
    float* p = nullptr;
    size_t count = 0;
    explicit Scratch(const sycl::queue& q) : queue(q) {}
    ~Scratch() {
        if (p) {
            try { queue.wait_and_throw(); sycl::free(p, queue); }
            catch (...) { std::fputs("dsv41: EXL3 scratch cleanup failed\n", stderr); }
        }
    }
};
float* hidden(sycl::queue& s, size_t count) {
    static thread_local std::vector<std::unique_ptr<Scratch>> cache;
    Scratch* sc = nullptr;
    for (auto& e : cache) if (e->queue == s) { sc = e.get(); break; }
    if (!sc) { cache.push_back(std::make_unique<Scratch>(s)); sc = cache.back().get(); }
    if (count > sc->count) {
        s.wait_and_throw();
        if (sc->p) sycl::free(sc->p, s);
        sc->p = nullptr; sc->count = 0;
        sc->p = sycl::malloc_device<float>(count, s);
        if (!sc->p) throw std::bad_alloc();
        sc->count = count;
    }
    return sc->p;
}
}  // namespace

Mat view(const Mat& m, int c0, int nc) {
    Mat v = m;
    v.ld = m.ld > 0 ? m.ld : m.n / 16;
    v.trellis = m.trellis + (size_t) (c0 / 16) * 8 * m.K;
    v.svh = m.svh + c0; v.n = nc;
    return v;
}
bool supported(const Mat& m) {
    return m.trellis && m.suh && m.svh && m.k > 0 && m.n > 0 && m.k % 128 == 0 && m.n % 128 == 0 &&
           m.K >= 1 && m.K <= 8 && m.cb >= 0 && m.cb <= 2;
}
void set_skip(const int* flag) { skip = flag; }
int mv_splits(const Mat& m) { return supported(m) ? 1 : 0; }
int mv_cols(const Mat& m) { return supported(m) ? kMaxMvCols : 0; }
int launch_errors() { return errors.load(std::memory_order_relaxed); }
bool prepare(sycl::queue& s) { return s.has_property<sycl::property::queue::in_order>(); }
bool several_ok() { return true; }
bool mma_supported() { return false; }  // The portable implementation uses no matrix extension.

bool mv(const MvJob* jobs, int n, sycl::queue& s) {
    if (n <= 0 || n > kMaxMvJobs) return false;
    for (int i = 0; i < n; ++i)
        if (!jobs[i].m || !supported(*jobs[i].m) || jobs[i].ncol < 1 || jobs[i].ncol > kMaxMvCols) return false;
    for (int i = 0; i < n; ++i) {
        const MvJob job = jobs[i]; const Mat m = *job.m;
        const int* flag = skip;
        launch(s, (size_t) job.ncol * (m.n / 128), [=](size_t id) {
            if (flag && *flag) return;
            const int c0 = (int) (id % (m.n / 128)) * 128, row = (int) (id / (m.n / 128));
            auto input = [&](int j) {
                if (job.xf) return job.xf[(size_t) row * job.xf_cs + j];
                const uint8_t* b = (const uint8_t*) job.xq + (size_t) row * job.xq_cs + (size_t) (j / 32) * 36;
                const uint16_t d = (uint16_t) ((unsigned) b[0] | (unsigned) b[1] << 8);
                return sd::f16_value(d) * (float) ((const int8_t*) (b + 4))[j % 32];
            };
            float y[128]; band(m, input, c0, y);
            for (int c = 0; c < 128; ++c)
                job.y[(size_t) row * job.y_cs + c0 + c] = job.alpha * y[c] + (job.bias ? job.bias[c0 + c] : 0.0f);
        });
    }
    return true;
}
void moe_gate_up(const MoeArgs& a, const float* x, float* h, sycl::queue& s) {
    if (!h) h = hidden(s, (size_t) a.k * a.n_ff);
    const int* flag = skip;
    launch(s, (size_t) a.k * (a.n_ff / 128), [=](size_t id) {
        if (flag && *flag) return;
        const int e = (int) (id / (a.n_ff / 128)), c0 = (int) (id % (a.n_ff / 128)) * 128;
        if (!a.plan_ptr[e]) return;
        const auto* b = (const uint8_t*) a.plan_ptr[e];
        float g[128], u[128];
        band(expert(b, a.lay, 0, a.n_embd, a.n_ff), [=](int j) { return x[j]; }, c0, g);
        band(expert(b, a.lay, 1, a.n_embd, a.n_ff), [=](int j) { return x[j]; }, c0, u);
        for (int c = 0; c < 128; ++c) h[(size_t) e * a.n_ff + c0 + c] = swi(g[c], u[c], a.limit);
    });
}
void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, sycl::queue& s) {
    if (!h) h = hidden(s, (size_t) a.k * a.n_ff);
    const int* flag = skip;
    launch(s, (size_t) a.n_embd / 128, [=](size_t id) {
        if (flag && *flag) return;
        const int c0 = (int) id * 128;
        float sum[128] = {};
        for (int e = 0; e < a.k; ++e) {
            if (!a.plan_ptr[e]) continue;
            float y[128];
            const Mat m = expert((const uint8_t*) a.plan_ptr[e], a.lay, 2, a.n_embd, a.n_ff);
            band(m, [=](int j) { return h[(size_t) e * a.n_ff + j]; }, c0, y);
            for (int c = 0; c < 128; ++c) sum[c] = sycl::fma(a.plan_w[e], y[c], sum[c]);
        }
        for (int c = 0; c < 128; ++c) {
            float v = sum[c];
            if (a.cpu_flag && *a.cpu_flag) v += a.cpu_part[c0 + c];
            if (sh_out) v += sh_out[c0 + c];
            out[c0 + c] = v;
        }
    });
}
bool win_experts(const WinArgs& a, sycl::queue& s) {
    if (a.T < 1 || a.k < 1 || a.k > 8 || a.n_embd % 128 || a.n_ff % 128 || !a.h || !a.part || !a.out) return false;
    launch(s, (size_t) a.T * 8 * (a.n_ff / 128), [=](size_t id) {
        const int e = (int) (id / (a.n_ff / 128)), c0 = (int) (id % (a.n_ff / 128)) * 128;
        if (e % 8 >= a.k || !a.plan_ptr[e]) return;
        const auto* b = (const uint8_t*) a.plan_ptr[e];
        const float* x = a.x + (size_t) (e / 8) * a.x_stride;
        float g[128], u[128];
        band(expert(b, a.lay, 0, a.n_embd, a.n_ff), [=](int j) { return x[j]; }, c0, g);
        band(expert(b, a.lay, 1, a.n_embd, a.n_ff), [=](int j) { return x[j]; }, c0, u);
        for (int c = 0; c < 128; ++c) a.h[(size_t) e * a.n_ff + c0 + c] = swi(g[c], u[c], a.limit);
    });
    launch(s, (size_t) a.T * 8 * (a.n_embd / 128), [=](size_t id) {
        const int e = (int) (id / (a.n_embd / 128)), c0 = (int) (id % (a.n_embd / 128)) * 128;
        if (e % 8 >= a.k || !a.plan_ptr[e]) return;
        float y[128];
        band(expert((const uint8_t*) a.plan_ptr[e], a.lay, 2, a.n_embd, a.n_ff),
             [=](int j) { return a.h[(size_t) e * a.n_ff + j]; }, c0, y);
        for (int c = 0; c < 128; ++c) a.part[(size_t) e * a.n_embd + c0 + c] = y[c];
    });
    if (a.wait_ptr) strata::ds4::wait_seq(a.wait_ptr, a.wait_val, s);
    launch(s, (size_t) a.T * a.n_embd, [=](size_t id) {
        const int t = (int) (id / a.n_embd), c = (int) (id % a.n_embd);
        float v = 0.0f;
        for (int e = 0; e < a.k; ++e)
            if (a.plan_ptr[t * 8 + e]) v = sycl::fma(a.plan_w[t * 8 + e], a.part[(size_t) (t * 8 + e) * a.n_embd + c], v);
        if (a.cpu_flag && a.cpu_flag[t]) v += a.cpu_part[(size_t) t * (a.cpu_stride ? a.cpu_stride : a.n_embd) + c];
        if (a.sh_out) v += a.sh_out[id];
        a.out[(size_t) t * a.out_stride + c] = v;
    });
    return true;
}

void set_gate_up(const SetArgs& a, const float* x, int ldx, const int* idx, uint16_t* h, int ldh, sycl::queue& s) {
    launch(s, (size_t) a.n_exp * a.max_rows * (a.n_ff / 128), [=](size_t id) {
        const int c0 = (int) (id % (a.n_ff / 128)) * 128;
        const int row = (int) ((id / (a.n_ff / 128)) % a.max_rows);
        const int e = (int) (id / ((size_t) a.max_rows * (a.n_ff / 128)));
        const int lo = a.bounds[e] - a.row0, count = a.bounds[e + 1] - a.bounds[e];
        if (row >= count || (a.skip_ge > 0 && count >= a.skip_ge)) return;
        const int r = lo + row, xr = idx[r];
        const uint8_t* b = a.wbase + (size_t) e * a.stride;
        auto input = [=](int j) { return a.x16 ? sd::f16_value(a.x16[(size_t) xr * ldx + j]) : x[(size_t) xr * ldx + j]; };
        float g[128], u[128];
        band(expert(b, a.lay, 0, a.n_embd, a.n_ff), input, c0, g);
        band(expert(b, a.lay, 1, a.n_embd, a.n_ff), input, c0, u);
        for (int c = 0; c < 128; ++c) h[(size_t) r * ldh + c0 + c] = sd::f16_bits(swi(g[c], u[c], a.limit));
    });
}
void set_down(const SetArgs& a, const uint16_t* h, int ldh, float* y, int ldy, sycl::queue& s) {
    launch(s, (size_t) a.n_exp * a.max_rows * (a.n_embd / 128), [=](size_t id) {
        const int c0 = (int) (id % (a.n_embd / 128)) * 128;
        const int row = (int) ((id / (a.n_embd / 128)) % a.max_rows);
        const int e = (int) (id / ((size_t) a.max_rows * (a.n_embd / 128)));
        const int lo = a.bounds[e] - a.row0, count = a.bounds[e + 1] - a.bounds[e];
        if (row >= count || (a.skip_ge > 0 && count >= a.skip_ge)) return;
        const int r = lo + row;
        float out[128];
        band(expert(a.wbase + (size_t) e * a.stride, a.lay, 2, a.n_embd, a.n_ff),
             [=](int j) { return sd::f16_value(h[(size_t) r * ldh + j]); }, c0, out);
        for (int c = 0; c < 128; ++c) y[(size_t) r * ldy + c0 + c] = out[c];
    });
}
void gemm_rows(const Mat& m, const uint16_t* x, int ldx, int rows, float* y, int ldy, float beta, sycl::queue& s) {
    require(m);
    launch(s, (size_t) rows * (m.n / 128), [=](size_t id) {
        const int r = (int) (id / (m.n / 128)), c0 = (int) (id % (m.n / 128)) * 128;
        float out[128]; band(m, [=](int j) { return sd::f16_value(x[(size_t) r * ldx + j]); }, c0, out);
        for (int c = 0; c < 128; ++c) {
            float& v = y[(size_t) r * ldy + c0 + c];
            v = beta == 0.0f ? out[c] : sycl::fma(beta, v, out[c]);
        }
    });
}
void reconstruct_inner_rows(const Mat& m, uint16_t* dst, sycl::queue& s) {
    require(m);
    launch(s, (size_t) m.k * m.n, [=](size_t i) { dst[i] = sd::f16_bits(inner(m, (int) (i % m.k), (int) (i / m.k))); });
}
void reconstruct_inner(const Mat& m, uint16_t* dst, sycl::queue& s) {
    require(m);
    launch(s, (size_t) m.k * m.n, [=](size_t i) { dst[i] = sd::f16_bits(inner(m, (int) (i / m.n), (int) (i % m.n))); });
}
void reconstruct_rows(const Mat& m, uint16_t* dst, bool bf16, sycl::queue& s) {
    reconstruct_inner_rows(m, dst, s);
    launch(s, (size_t) m.k * (m.n / 128), [=](size_t id) {
        const int r = (int) (id % m.k), c0 = (int) (id / m.k) * 128;
        float v[128];
        for (int c = 0; c < 128; ++c) v[c] = sd::f16_value(dst[(size_t) (c0 + c) * m.k + r]);
        had128(v);
        for (int c = 0; c < 128; ++c) dst[(size_t) (c0 + c) * m.k + r] = sd::f16_bits(v[c]);
    });
    launch(s, (size_t) m.n * (m.k / 128), [=](size_t id) {
        const int c = (int) (id / (m.k / 128)), r0 = (int) (id % (m.k / 128)) * 128;
        float v[128];
        for (int r = 0; r < 128; ++r) v[r] = sd::f16_value(dst[(size_t) c * m.k + r0 + r]);
        had128(v);
        for (int r = 0; r < 128; ++r) {
            const float w = v[r] * sd::f16_value(m.svh[c]) * sd::f16_value(m.suh[r0 + r]);
            dst[(size_t) c * m.k + r0 + r] = bf16 ? strata::ds4::bf16_rn(w) : sd::f16_bits(w);
        }
    });
}
void rows_rot_f16(const float* x, int ldx, const int* idx, int rows, int cols, const uint16_t* suh, uint16_t* out, sycl::queue& s) {
    launch(s, (size_t) rows * (cols / 128), [=](size_t id) {
        const int r = (int) (id / (cols / 128)), c0 = (int) (id % (cols / 128)) * 128, xr = idx ? idx[r] : r;
        float v[128];
        for (int c = 0; c < 128; ++c) v[c] = x[(size_t) xr * ldx + c0 + c] * sd::f16_value(suh[c0 + c]);
        had128(v);
        for (int c = 0; c < 128; ++c) out[(size_t) r * cols + c0 + c] = sd::f16_bits(v[c]);
    });
}
void rows_rot_out(float* y, int ldy, int rows, int cols, const uint16_t* svh, sycl::queue& s) {
    launch(s, (size_t) rows * (cols / 128), [=](size_t id) {
        const int r = (int) (id / (cols / 128)), c0 = (int) (id % (cols / 128)) * 128;
        float v[128];
        for (int c = 0; c < 128; ++c) v[c] = y[(size_t) r * ldy + c0 + c];
        had128(v);
        for (int c = 0; c < 128; ++c) y[(size_t) r * ldy + c0 + c] = v[c] * sd::f16_value(svh[c0 + c]);
    });
}
void rows_f16(const float* x, int ldx, const int* idx, int rows, int cols, uint16_t* out, sycl::queue& s) {
    launch(s, (size_t) rows * cols, [=](size_t i) {
        const size_t r = i / cols;
        out[i] = sd::f16_bits(x[(idx ? (size_t) idx[r] : r) * ldx + i % cols]);
    });
}
}  // namespace strata::dsv41k::exl3
