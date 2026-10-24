// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_sdpa_onednn.cpp - see qi_sdpa.hpp: oneDNN's SDPA through its graph API on Intel GPUs (Q . K^T, the
// scale, the softmax, . V fused into one partition), a partition compiled once a shape and kept.

#include "qi_sdpa.hpp"

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace strata::image::sdpa {

namespace {

std::mutex g_mu;

namespace dg = dnnl::graph;
using lt = dg::logical_tensor;

void* sdpa_malloc(size_t size, size_t alignment, const void* dev, const void* ctx) {
    return sycl::aligned_alloc_device(alignment, size, *static_cast<const sycl::device*>(dev),
                                      *static_cast<const sycl::context*>(ctx));
}
void sdpa_free(void* ptr, const void*, const void* ctx, void* ev) {
    if (ev != nullptr) static_cast<sycl::event*>(ev)->wait();
    sycl::free(ptr, *static_cast<const sycl::context*>(ctx));
}

// a device's oneDNN engine, its compiled partitions by shape, and the scale (FP16, 1 / sqrt(head_dim)) on it
struct Sdpa {
    bool failed = false;
    std::unique_ptr<dnnl::engine> eng;
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, dg::compiled_partition> parts;
    std::map<int64_t, sycl::half*> scales;
};
std::unordered_map<sycl::device, Sdpa> g_sdpa;

// the graph of one shape: q [1][heads][rows][hd], k and v [1][heads][T][hd], out as q, the engine's strides
dg::compiled_partition compile(const dnnl::engine& eng, int64_t rows, int64_t T, int64_t heads, int64_t hd) {
    const int64_t row = heads * hd;
    const std::vector<int64_t> qd{1, heads, rows, hd}, kd{1, heads, T, hd}, st{rows * row, hd, row, 1},
        kst{T * row, hd, row, 1};
    lt Q(0, lt::data_type::f16, qd, st), K(1, lt::data_type::f16, kd, kst), V(2, lt::data_type::f16, kd, kst);
    lt S(3, lt::data_type::f32, lt::layout_type::any), SS(5, lt::data_type::f32, lt::layout_type::any);
    lt SC(4, lt::data_type::f16, std::vector<int64_t>{1}, std::vector<int64_t>{1});
    lt P(6, lt::data_type::f16, lt::layout_type::any), O(7, lt::data_type::f16, qd, st);
    dg::op mm1(10, dg::op::kind::MatMul, "qk");
    mm1.set_attr<bool>(dg::op::attr::transpose_b, true);
    mm1.add_inputs({Q, K});
    mm1.add_outputs({S});
    dg::op scale(11, dg::op::kind::Multiply, "scale");
    scale.add_inputs({S, SC});
    scale.add_outputs({SS});
    dg::op sm(12, dg::op::kind::SoftMax, "softmax");
    sm.set_attr<int64_t>(dg::op::attr::axis, -1);
    sm.add_inputs({SS});
    sm.add_outputs({P});
    dg::op mm2(13, dg::op::kind::MatMul, "pv");
    mm2.add_inputs({P, V});
    mm2.add_outputs({O});
    dg::graph g(dnnl::engine::kind::gpu);
    g.add_op(mm1);
    g.add_op(scale);
    g.add_op(sm);
    g.add_op(mm2);
    g.finalize();
    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) throw std::runtime_error("no fused SDPA partition");
    return parts[0].compile({Q, K, V, SC}, {O}, eng);
}

}  // namespace

bool onednn(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1,
            int64_t T, int64_t heads, int64_t head_dim, uint16_t* out16) {
    std::lock_guard<std::mutex> lock(g_mu);
    Sdpa& d = g_sdpa[q.get_device()];
    if (d.failed) return false;
    try {
        if (!d.eng) {
            const dg::allocator alloc = dg::sycl_interop::make_allocator(sdpa_malloc, sdpa_free);
            d.eng = std::make_unique<dnnl::engine>(
                dg::sycl_interop::make_engine_with_allocator(q.get_device(), q.get_context(), alloc));
        }
        const auto key = std::make_tuple(q1 - q0, T, heads, head_dim);
        auto it = d.parts.find(key);
        if (it == d.parts.end()) it = d.parts.emplace(key, compile(*d.eng, q1 - q0, T, heads, head_dim)).first;
        sycl::half*& sc = d.scales[head_dim];
        if (sc == nullptr) {
            sc = sycl::malloc_device<sycl::half>(1, q);
            const sycl::half v = 1.0f / std::sqrt((float) head_dim);
            q.memcpy(sc, &v, sizeof v).wait();
        }
        dnnl::stream strm = dnnl::sycl_interop::make_stream(*d.eng, q);
        const lt& Q = it->second.query_logical_tensor(0);
        const lt& K = it->second.query_logical_tensor(1);
        const lt& V = it->second.query_logical_tensor(2);
        const lt& SC = it->second.query_logical_tensor(4);
        const lt& O = it->second.query_logical_tensor(7);
        const int64_t row = heads * head_dim;
        std::vector<dg::tensor> ins{dg::tensor(Q, *d.eng, const_cast<uint16_t*>(q16 + q0 * row)),
                                    dg::tensor(K, *d.eng, const_cast<uint16_t*>(k16)),
                                    dg::tensor(V, *d.eng, const_cast<uint16_t*>(v16)), dg::tensor(SC, *d.eng, sc)};
        std::vector<dg::tensor> outs{dg::tensor(O, *d.eng, out16)};
        dg::sycl_interop::execute(it->second, strm, ins, outs);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-image: oneDNN's attention is not taken on this GPU (%s): the engine's own kernel\n",
                     e.what());
        d.failed = true;
        return false;
    }
}

}  // namespace strata::image::sdpa
