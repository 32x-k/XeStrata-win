// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_kernels.cuh"
#include "ds4_mailbox.hpp"
#include "ds4_runtime.hpp"
#include "ds4_sycl.hpp"
#include "strata/core/readback.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
namespace ds = strata::ds4;
namespace rt = ds::rt;
namespace sd = ds::sycl_detail;

void require(rt::Result status) {
    if (status != rt::Result::success) throw std::runtime_error(rt::error_string(status));
}
struct Stream {
    rt::Queue value;
    Stream() { require(rt::stream_create(&value)); }
    ~Stream() { rt::stream_destroy(value); }
};
struct Memory {
    sycl::queue& q;
    std::vector<void*> pointers;
    explicit Memory(sycl::queue& queue) : q(queue) {}
    ~Memory() {
        try {
            q.wait_and_throw();
            for (void* p : pointers) sycl::free(p, q);
        } catch (...) {
            std::fputs("core parity: cannot safely release GPU buffers\n", stderr);
            std::_Exit(1);
        }
    }
    template <class T> T* get(size_t n) {
        T* p = sycl::malloc_device<T>(std::max<size_t>(n, 1), q);
        if (!p) throw std::bad_alloc();
        try { pointers.push_back(p); } catch (...) { sycl::free(p, q); throw; }
        return p;
    }
    template <class T> T* upload(const std::vector<T>& values) {
        auto* p = get<T>(values.size());
        if (!values.empty()) q.memcpy(p, values.data(), values.size() * sizeof(T));
        return p;
    }
};
void close(float actual, double expected, const char* label) {
    if (!std::isfinite(actual) || std::abs(double(actual) - expected) > 1e-5 * std::max(1.0, std::abs(expected))) {
        std::fprintf(stderr, "%s: %.9g != %.17g\n", label, actual, expected);
        throw std::runtime_error(label);
    }
}

void gemv(sycl::queue& q) {
    constexpr int tokens = 3, rows = 7, offset = 3;
    constexpr float guard = 12345.f, alpha = -.75f;
    for (int k : {0, 1, 7, 33, 513}) for (int kind : {0, 1, 2}) {
        const int ldx = k + (kind == 1 ? 5 : 0), ldy = rows + (kind == 0 ? 0 : 4);
        std::vector<float> x(size_t(tokens) * ldx, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> w(size_t(rows) * k);
        std::vector<uint16_t> wh(w.size());
        std::vector<float> y(size_t(offset) + size_t(tokens) * ldy + 5, guard);
        std::vector<double> expected(y.begin(), y.end());
        for (int t = 0; t < tokens; ++t) for (int j = 0; j < k; ++j)
            x[size_t(t) * ldx + j] = float((t * 7 + j * 3) % 31 - 15) / 16.f;
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = float(int(i % 23) - 11) / 32.f;
            wh[i] = sd::f16_bits(w[i]);
        }
        for (int t = 0; t < tokens; ++t) for (int r = 0; r < rows; ++r) {
            double sum = 0;
            for (int j = 0; j < k; ++j) sum += double(w[size_t(r) * k + j]) * x[size_t(t) * ldx + j];
            expected[size_t(offset) + size_t(t) * ldy + r] = sum * (kind == 2 ? 1.f : alpha);
        }
        Memory memory(q);
        auto* dx = memory.upload(x);
        auto* dw = memory.upload(w);
        auto* dh = memory.upload(wh);
        auto* dy = memory.upload(y);
        if (kind == 0) ds::gemv_f16(dh, dx, rows, k, tokens, alpha, dy + offset, q);
        if (kind == 1) ds::gemv_f16_rows(dh, dx, size_t(ldx), rows, k, tokens, alpha, dy + offset, size_t(ldy), q);
        if (kind == 2) ds::gemv_f32(dw, dx, rows, k, tokens, dy + offset, ldy, q);
        strata::core::readback(q, y.data(), dy, y.size() * sizeof(float));
        for (size_t i = 0; i < y.size(); ++i) {
            if (expected[i] == guard && y[i] != guard) throw std::runtime_error("gemv changed a guard");
            close(y[i], expected[i], "gemv value");
        }
        ds::gemv_f16(nullptr, nullptr, rows, k, 0, alpha, nullptr, q);
        ds::gemv_f32(nullptr, nullptr, 0, k, tokens, nullptr, 0, q);
    }
    std::puts("gemv ragged/empty/strides/guards: PASS");
}

void route(sycl::queue& q) {
    constexpr int tokens = 3, experts = 37, picks = 3, ahead = 16, off = 5;
    std::vector<float> logits(size_t(tokens) * 1024, 0.f), bias(experts, 0.f), alt(experts, 0.f);
    for (int t = 0; t < tokens; ++t) for (int i = 0; i < experts; ++i)
        logits[size_t(t) * 1024 + i] = float((i / 2 + t) % 7 - 3) / 4.f;
    alt[0] = 10.f;
    std::vector<int> ids(size_t(tokens) * ds::kPicks, -99);
    std::vector<float> weights(ids.size(), -99.f);
    std::vector<ds::Req> request(1);
    for (auto& row : request[0].ahead) std::fill(std::begin(row), std::end(row), -99);
    ds::VlBits bits{}; bits.w[0] = 2;
    Memory memory(q);
    auto* dl = memory.upload(logits); auto* db = memory.upload(bias); auto* da = memory.upload(alt);
    auto* di = memory.upload(ids); auto* dw = memory.upload(weights); auto* dr = memory.upload(request);
    ds::route(dl, db, experts, picks, tokens, 1.25f, di, dw, q, da, &bits);
    ds::route_ahead(dl, db, experts, ahead, tokens, dr, off, q);
    strata::core::readback(q, ids.data(), di, ids.size() * sizeof(int));
    strata::core::readback(q, weights.data(), dw, weights.size() * sizeof(float));
    strata::core::readback(q, request.data(), dr, sizeof(ds::Req));
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> order(experts); std::iota(order.begin(), order.end(), 0);
        auto score = [&](int i) { return std::sqrt(std::log1p(std::exp(logits[size_t(t) * 1024 + i]))); };
        for (bool is_ahead : {false, true}) {
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                const float sa = score(a) + (!is_ahead && t == 1 ? alt[a] : bias[a]);
                const float sb = score(b) + (!is_ahead && t == 1 ? alt[b] : bias[b]);
                return sa == sb ? a < b : sa > sb;
            });
            float sum = 0.f;
            for (int j = 0; j < picks; ++j) sum += score(order[j]);
            for (int j = 0; j < (is_ahead ? ahead : picks); ++j) {
                const int actual = is_ahead ? request[0].ahead[t][off + j] : ids[size_t(t) * ds::kPicks + j];
                if (actual != order[j]) throw std::runtime_error("route tie order or lookahead offset");
                if (!is_ahead) close(weights[size_t(t) * ds::kPicks + j], 1.25f * score(order[j]) / sum, "route weight");
            }
        }
        for (int j = picks; j < ds::kPicks; ++j)
            if (ids[size_t(t) * ds::kPicks + j] != -99 || weights[size_t(t) * ds::kPicks + j] != -99.f)
                throw std::runtime_error("route wrote unused picks");
        for (int j = 0; j < ds::kAhead; ++j)
            if ((j < off || j >= off + ahead) && request[0].ahead[t][j] != -99)
                throw std::runtime_error("lookahead wrote outside its slice");
    }
    std::puts("route ties/alternate bias/top3/lookahead: PASS");
}

void attention(sycl::queue& queue) {
    constexpr int tokens = 2, heads = 3, d = 33, stride = 3, offset = 5;
    constexpr float guard = 12345.f, scale = .25f;
    std::vector<float> query(size_t(tokens) * heads * d), sinks{-.5f, 0.f, 1.f};
    std::vector<uint16_t> window(size_t(4) * d), compressed(size_t(5) * d);
    std::vector<int> selection{3, 1, -99, 4, 2, -99};
    std::vector<float> output(size_t(offset) + query.size() + 7, guard);
    std::vector<double> expected(output.begin(), output.end());
    for (size_t i = 0; i < query.size(); ++i) query[i] = float(int(i % 17) - 8) / 32.f;
    for (size_t i = 0; i < window.size(); ++i) window[i] = uint16_t(sycl::bit_cast<uint32_t>(float(int(i % 13) - 6) / 8.f) >> 16);
    for (size_t i = 0; i < compressed.size(); ++i) compressed[i] = uint16_t(sycl::bit_cast<uint32_t>(float(int(i % 19) - 9) / 16.f) >> 16);
    ds::AttnRows rows{};
    rows.win0[0] = 1; rows.nwin[0] = 3; rows.nsel[0] = 2;
    // Query 1 sees only the sink: its output must be zero for every head.
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
        std::vector<const uint16_t*> kv;
        kv.reserve(size_t(rows.nwin[t]) + size_t(rows.nsel[t]));
        for (int r = 0; r < rows.nwin[t]; ++r) kv.push_back(window.data() + size_t(rows.win0[t] + r) * d);
        for (int r = 0; r < rows.nsel[t]; ++r) kv.push_back(compressed.data() + size_t(selection[size_t(t) * stride + r]) * d);
        std::vector<double> weights;
        double denominator = std::exp(double(sinks[h]));
        for (const auto* row : kv) {
            double dot = 0;
            for (int e = 0; e < d; ++e) dot += double(query[(size_t(t) * heads + h) * d + e]) * sd::bf16_value(row[e]);
            weights.push_back(std::exp(dot * scale));
            denominator += weights.back();
        }
        for (int e = 0; e < d; ++e) {
            double numerator = 0;
            for (size_t r = 0; r < kv.size(); ++r) numerator += weights[r] * sd::bf16_value(kv[r][e]);
            expected[size_t(offset) + (size_t(t) * heads + h) * d + e] = numerator / denominator;
        }
    }
    Memory memory(queue);
    auto* dq = memory.upload(query); auto* dsink = memory.upload(sinks);
    auto* dw = memory.upload(window); auto* dc = memory.upload(compressed);
    auto* di = memory.upload(selection); auto* dy = memory.upload(output);
    for (int group_heads : {0, 1, 2, 8}) {
        queue.fill(dy, guard, output.size());
        ds::sparse_attn(dq, dw, dc, di, stride, rows, dsink, heads, d, tokens, scale, dy + offset,
                        queue, ds::kKvBf16, group_heads);
        strata::core::readback(queue, output.data(), dy, output.size() * sizeof(float));
        for (size_t i = 0; i < output.size(); ++i) {
            if (expected[i] == guard && output[i] != guard) throw std::runtime_error("attention changed a guard");
            close(output[i], expected[i], "attention value");
        }
    }
    std::puts("attention odd dimensions/head tails/sink/guards: PASS");
}

void attention_scratch(sycl::queue& queue) {
    constexpr int tokens = 2, heads = 3, d = 512, stride = 3, padding = 7;
    constexpr float guard = 12345.f, scale = .0625f;
    constexpr uint16_t scratch_guard = 0x7e55;
    const size_t row_bytes = size_t(d) / 2 + size_t(d) / 16;
    std::vector<uint8_t> packed(4 * row_bytes);
    std::vector<float> decoded(size_t(4) * d), query(size_t(tokens) * heads * d), sinks{-.5f, 0.f, 1.f};
    std::vector<int> selection{3, 1, -99, -99, -99, -99};
    std::vector<float> output(query.size() + size_t(2) * padding, guard);
    std::vector<uint16_t> scratch(size_t(tokens) * stride * d + size_t(2) * padding, scratch_guard);
    std::vector<double> expected(output.begin(), output.end());
    constexpr float magnitudes[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    for (int r = 0; r < 4; ++r) {
        for (int e = 0; e < d; ++e) {
            const int code = (e + 3 * r) % 16;
            const int exponent = (r + e / 16) % 3 - 1;
            packed[size_t(r) * row_bytes + e / 2] |= uint8_t(code << (4 * (e & 1)));
            decoded[size_t(r) * d + e] = std::ldexp((code & 8 ? -1.f : 1.f) * magnitudes[code & 7], exponent);
        }
        for (int b = 0; b < d / 16; ++b)
            packed[size_t(r) * row_bytes + d / 2 + b] = uint8_t(((r + b) % 3 + 6) << 3);
    }
    for (size_t i = 0; i < query.size(); ++i) query[i] = float(int(i % 17) - 8) / 128.f;
    ds::AttnRows rows{};
    rows.nsel[0] = 2;
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
        double weights[2]{}, denominator = std::exp(double(sinks[h]));
        for (int r = 0; r < rows.nsel[t]; ++r) {
            double dot = 0.;
            for (int e = 0; e < d; ++e)
                dot += double(query[(size_t(t) * heads + h) * d + e]) * decoded[size_t(selection[r]) * d + e];
            weights[r] = std::exp(dot * scale);
            denominator += weights[r];
        }
        for (int e = 0; e < d; ++e) {
            double numerator = 0.;
            for (int r = 0; r < rows.nsel[t]; ++r) numerator += weights[r] * decoded[size_t(selection[r]) * d + e];
            expected[padding + (size_t(t) * heads + h) * d + e] = numerator / denominator;
        }
    }
    Memory memory(queue);
    auto* dq = memory.upload(query); auto* dsink = memory.upload(sinks);
    auto* dp = memory.upload(packed); auto* di = memory.upload(selection);
    auto* dy = memory.upload(output); auto* dscratch = memory.upload(scratch);
    for (int mode = 0; mode < 5; ++mode) {
        queue.fill(dy, guard, output.size());
        queue.fill(dscratch, scratch_guard, scratch.size());
        ds::AttnScratch sc{};
        sc.rows = dscratch + padding;
        sc.cap = mode == 2 ? stride - 1 : stride;
        sc.max_t = mode == 3 ? tokens - 1 : tokens;
        const int hp = mode == 1 ? 0 : mode == 4 ? 2 : -1;
        ds::sparse_attn(dq, nullptr, dp, di, stride, rows, dsink, heads, d, tokens, scale,
                        dy + padding, queue, ds::kKvFp4, hp, &sc);
        strata::core::readback(queue, output.data(), dy, output.size() * sizeof(float));
        strata::core::readback(queue, scratch.data(), dscratch, scratch.size() * sizeof(uint16_t));
        for (size_t i = 0; i < output.size(); ++i) {
            if (expected[i] == guard && output[i] != guard) throw std::runtime_error("scratch attention output guard");
            close(output[i], expected[i], "scratch attention value");
        }
        // A device whose local memory cannot hold a tile may use the direct fallback.
        const bool written = mode <= 1 && scratch[padding] != scratch_guard;
        if (mode == 0) std::puts(written ? "attention scratch staging: exercised" : "attention scratch staging: direct fallback");
        for (size_t i = 0; i < scratch.size(); ++i) {
            uint16_t want = scratch_guard;
            if (written && i >= padding && i < size_t(padding) + size_t(2) * d) {
                const size_t at = i - padding;
                want = uint16_t(sycl::bit_cast<uint32_t>(decoded[size_t(selection[at / d]) * d + at % d]) >> 16);
            }
            if (scratch[i] != want) throw std::runtime_error("scratch attention capacity or row guard");
        }
    }
    std::puts("attention FP4 scratch/capacity/empty selection/guards: PASS");
}
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
void attention_generated(sycl::queue& queue) {
    constexpr int tokens = 64, heads = 17, d = 512, stride = 17, padding = 7;
    constexpr float guard = 12345.f, scale = .03125f;
    const size_t row_bytes = size_t(d) / 2 + size_t(d) / 16;
    std::vector<uint8_t> packed(64 * row_bytes);
    std::vector<float> decoded(size_t(64) * d), query(size_t(tokens) * heads * d), sinks(heads);
    std::vector<uint16_t> window(size_t(tokens + 8) * d);
    std::vector<int> selection(size_t(tokens) * stride);
    std::vector<float> output(query.size() + 2 * size_t(padding), guard);
    constexpr float magnitudes[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    for (int row = 0; row < 64; ++row) {
        for (int e = 0; e < d; ++e) {
            const int code = (e + 3 * row) % 16, exponent = (row + e / 16) % 3 - 1;
            packed[size_t(row) * row_bytes + e / 2] |= uint8_t(code << (4 * (e & 1)));
            decoded[size_t(row) * d + e] = std::ldexp((code & 8 ? -1.f : 1.f) * magnitudes[code & 7], exponent);
        }
        for (int b = 0; b < d / 16; ++b)
            packed[size_t(row) * row_bytes + d / 2 + b] = uint8_t(((row + b) % 3 + 6) << 3);
    }
    for (size_t i = 0; i < query.size(); ++i) query[i] = float(int(i % 17) - 8) / 256.f;
    for (size_t i = 0; i < window.size(); ++i)
        window[i] = uint16_t(sycl::bit_cast<uint32_t>(float(int(i % 13) - 6) / 16.f) >> 16);
    for (int head = 0; head < heads; ++head) sinks[head] = float(head % 3 - 1) / 2.f;
    for (int t = 0; t < tokens; ++t) for (int j = 0; j < stride; ++j) selection[size_t(t) * stride + j] = j;
    Memory memory(queue);
    auto* dq = memory.upload(query); auto* dsink = memory.upload(sinks);
    auto* dw = memory.upload(window); auto* dp = memory.upload(packed);
    auto* di = memory.upload(selection); auto* dy = memory.upload(output);
    for (int compression : {0, 2}) {
        ds::AttnRows rows{};
        rows.gen = 1; rows.p0 = 64; rows.win = 8; rows.R = compression; rows.kcap = stride;
        std::vector<double> expected(output.size(), guard);
        for (int t = 0; t < tokens; ++t) for (int head = 0; head < heads; ++head) {
            const int count = 8 + (compression ? stride : 0);
            auto value = [&](int row, int e) -> double {
                return row < 8 ? sd::bf16_value(window[size_t(t + 1 + row) * d + e]) :
                    decoded[size_t(selection[size_t(t) * stride + row - 8]) * d + e];
            };
            std::vector<double> weights(size_t(count), 0.);
            double denominator = std::exp(double(sinks[head]));
            for (int row = 0; row < count; ++row) {
                double dot = 0.;
                for (int e = 0; e < d; ++e) dot += double(query[(size_t(t) * heads + head) * d + e]) * value(row, e);
                weights[size_t(row)] = std::exp(dot * scale);
                denominator += weights[size_t(row)];
            }
            for (int e = 0; e < d; ++e) {
                double numerator = 0.;
                for (int row = 0; row < count; ++row) numerator += weights[size_t(row)] * value(row, e);
                expected[size_t(padding) + (size_t(t) * heads + head) * d + e] = numerator / denominator;
            }
        }
        queue.fill(dy, guard, output.size());
        ds::sparse_attn(dq, dw, dp, di, stride, rows, dsink, heads, d, tokens, scale,
                        dy + padding, queue, ds::kKvFp4);
        strata::core::readback(queue, output.data(), dy, output.size() * sizeof(float));
        for (size_t i = 0; i < output.size(); ++i) {
            if ((i < padding || i >= query.size() + padding) && output[i] != guard)
                throw std::runtime_error("generated attention output guard");
            close(output[i], expected[i], "generated attention double reference");
        }
    }
    std::puts("attention generated R0/R2/ragged heads/FP64/guards: PASS");
}


void engram(sycl::queue& q) {
    // Independent double reference; include partial groups, a single element,
    // the model dimension, zero norms, signed gates, and residual output guards.
    constexpr int tokens = 3, streams = 4, padding = 7;
    constexpr float guard = 12345.f, eps = 1e-20f;
    for (int dim : {1, 17, 257, 5120}) for (bool zero : {false, true}) {
        const size_t count = size_t(tokens) * streams * size_t(dim);
        std::vector<float> input(count + 2 * size_t(padding), guard), kv(size_t(tokens) * (streams + 1) * size_t(dim)), w(size_t(streams) * size_t(dim));
        for (size_t i = 0; i < count; ++i) input[padding + i] = zero ? 0.f : float(int(i % 31) - 15) / 32.f;
        for (size_t i = 0; i < kv.size(); ++i) kv[i] = zero ? 0.f : float(int((i * 13) % 37) - 18) / 64.f;
        for (size_t i = 0; i < w.size(); ++i) w[i] = float(int((i * 7) % 29) - 14) / 32.f;
        std::vector<double> reference(input.begin(), input.end());
        for (int t = 0; t < tokens; ++t) for (int stream = 0; stream < streams; ++stream) {
            const size_t off = size_t(padding) + (size_t(t) * streams + size_t(stream)) * size_t(dim);
            const size_t key = (size_t(t) * (streams + 1) + size_t(stream)) * size_t(dim);
            const size_t value = (size_t(t) * (streams + 1) + streams) * size_t(dim);
            double sh = 0., sk = 0., sd = 0.;
            for (int j = 0; j < dim; ++j) {
                const double h = input[off + size_t(j)], k = kv[key + size_t(j)];
                sh += h * h; sk += k * k; sd += h * double(w[size_t(stream) * size_t(dim) + size_t(j)]) * k;
            }
            const double dot = sd / std::sqrt(sh / dim + double(eps)) / std::sqrt(sk / dim + double(eps)) / std::sqrt(double(dim));
            const double sq = std::copysign(std::sqrt(std::max(std::abs(dot), double(1e-6f))), dot);
            const double gate = 1. / (1. + std::exp(-sq));
            for (int j = 0; j < dim; ++j) reference[off + size_t(j)] += gate * double(kv[value + size_t(j)]);
        }
        Memory memory(q); auto* h = memory.upload(input); auto* keys = memory.upload(kv); auto* weights = memory.upload(w);
        ds::engram_gate(h + padding, keys, weights, streams, dim, tokens, eps, q);
        strata::core::readback(q, input.data(), h, input.size() * sizeof(float));
        for (size_t i = 0; i < input.size(); ++i) {
            if ((i < padding || i >= size_t(padding) + count) && input[i] != guard)
                throw std::runtime_error("Engram residual output guard");
            if (zero && i >= padding && i < size_t(padding) + count && input[i] != 0.f)
                throw std::runtime_error("Engram zero norm output");
            close(input[i], reference[i], "Engram double reference");
        }
    }
    std::puts("Engram ragged/zero norms/residual/FP64: PASS");
}

// The wait a decode layer queues ahead of its down kernels: the stream stays behind the flag the host
// writes and then reads the reply written before it. A stopped mailbox releases the wait only after the CPU
// writer has finished, on the device as on the host.
void reply_wait(sycl::queue& q) {
    using namespace std::chrono_literals;
    struct Words {
        sycl::queue& q;
        unsigned* flag = sycl::malloc_host<unsigned>(1, q);
        float* reply = sycl::malloc_host<float>(1, q);
        float* seen = sycl::malloc_host<float>(1, q);
        explicit Words(sycl::queue& queue) : q(queue) {}
        ~Words() {
            try {
                q.wait_and_throw();
                for (void* p : {static_cast<void*>(flag), static_cast<void*>(reply), static_cast<void*>(seen)})
                    if (p) sycl::free(p, q);
            } catch (...) {
                std::fputs("core parity: cannot safely release the reply words\n", stderr);
                std::_Exit(1);
            }
        }
    } w(q);
    if (!w.flag || !w.reply || !w.seen) throw std::bad_alloc();
    unsigned* const flag = w.flag;
    float* const reply = w.reply;
    float* const seen = w.seen;
    auto queue_reader = [&](unsigned sequence) {
        *flag = 0u;
        *seen = -1.f;
        ds::wait_seq(flag, sequence, q);
        sd::single(q, [=] { *seen = *reply; });
    };
    for (bool bound : {false, true}) {
        const auto control = std::make_shared<ds::MailboxControl>();
        if (bound) ds::bind_mailbox(q, control);
        *reply = 0.f;
        queue_reader(7u);
        std::this_thread::sleep_for(50ms);
        if (q.ext_oneapi_empty()) throw std::runtime_error("reply wait ended before the reply");
        *reply = 93.f;
        std::atomic_ref<unsigned>(*flag).store(7u, std::memory_order_release);
        q.wait_and_throw();
        if (*seen != 93.f) throw std::runtime_error("reply wait did not expose the reply");
        if (!bound) continue;
        control->producer_done.store(false);
        queue_reader(9u);
        control->stop("CPU response failed");
        std::this_thread::sleep_for(50ms);
        if (q.ext_oneapi_empty())
            throw std::runtime_error("cancellation released a reply wait while the CPU producer could still write");
        *reply = 51.f;
        control->producer_finished();
        q.wait_and_throw();
        if (*seen != 51.f) throw std::runtime_error("producer completion did not publish its final write");
        ds::unbind_mailbox(&q);
    }
    std::puts("reply wait publish/cancel: PASS");
}

}  // namespace

int main() try {
    strata::core::restrict_hip_to_selector();
    strata::core::no_hip_resource_cache();
    if (!strata::core::Runtime::count()) return 77;
    Stream stream;
    auto& q = static_cast<sycl::queue&>(stream.value);
    gemv(q);
    route(q);
    attention(q);
    attention_scratch(q);
    attention_generated(q);
    engram(q);
    reply_wait(q);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "ds4 core parity: %s\n", e.what());
    return 1;
}
