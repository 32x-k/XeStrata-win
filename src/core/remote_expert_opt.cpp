// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/remote_expert_opt.cpp - --remote-expert-opt on the Xe runtime (the SYCL port of upstream
// src/core/remote_expert_opt.cu; upstream's own sycl/ version is a stub).
//
// The helper GPU sums its routed expert rows into one weighted row per token (reduce_experts), so a layer
// returns H floats a token instead of a row a routed entry; accumulate() folds it into the shared sums the
// primary's combine adds after the experts, and copy_cpu_rows zeroes the CPU rows the helper serves (the
// pool skips their experts, so the mapped rows hold stale bytes).  Upstream's one mapped array is one
// host-USM Sink per consuming device here: its kernels cannot read another context's host USM.
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstring>
#include <memory>

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int K = 10;
constexpr int MAXT = strata::kernels::cpu::MAXT;
constexpr int CAP = MAXT * K;   // a window's rows x top-K
struct ReduceMeta { int32_t row[CAP]; float weight[CAP]; };

using f32x4 = float __attribute__((ext_vector_type(4)));

bool failed(std::string& err) {
    err = "remote expert decode optimization: " + std::string(gpu::last_error());
    return false;
}

// (token, channel) each: sum over the token's K routed entries (row < 0: the CPU kept it), as upstream's
// reduce_experts.  Runs on the helper's stream, in its OnDevice scope.
void launch_reduce(const float* parts, const ReduceMeta* meta, float* sum, int tokens, void* stream) {
    Runtime::get().stream(stream).parallel_for(sycl::range<2>((size_t) tokens, (size_t) H),
                                               [=](sycl::id<2> id) {
        const int64_t t = id[0], c = id[1];
        float v = 0.0f;
        for (int j = 0; j < K; ++j) {
            const int i = (int) t * K + j, r = meta->row[i];
            if (r >= 0) v += parts[(int64_t) r * H + c] * meta->weight[i];
        }
        sum[t * H + c] = v;
    });
}

// copy_rows_from_mapped (src/kernels/xe/elementwise.cpp) plus the mask of rows the helper's reduced sums
// already carry: those are written +0.0 too, as upstream's copy_cpu_rows.
void launch_copy_cpu_rows(float* dst, const float* src, int64_t rows, const int32_t* mask,
                          const int32_t* hit_rows, const int32_t* count, void* stream) {
    Runtime::get().stream(stream).parallel_for(sycl::nd_range<1>((size_t) rows * 128, 128),
                                               [=](sycl::nd_item<1> it) {
        const int64_t row = (int64_t) it.get_group(0);
        const volatile int32_t* m = mask;
        const int c = *count;
        bool h = false;
        for (int i = (int) it.get_local_id(0); i < c; i += 128) h |= hit_rows[i] == row;
        const bool skip = m[row] != 0 || sycl::any_of_group(it.get_group(), h);
        f32x4* d = reinterpret_cast<f32x4*>(dst + row * H);
        const volatile f32x4* sr = reinterpret_cast<const volatile f32x4*>(src + row * H);
        for (int i = (int) it.get_local_id(0); i < H / 4; i += 128) d[i] = skip ? f32x4(0.0f) : sr[i];
    });
}
}  // namespace

void RemoteExpertOpt::attach(RemoteExperts& remote) {
    remote.remote_opt_ = this;
    peers_.push_back({&remote});
}

void RemoteExpertOpt::detach(RemoteExperts& remote) {
    remote.remote_opt_ = nullptr;
    peers_.erase(std::remove_if(peers_.begin(), peers_.end(),
                                [&](const Peer& p) { return p.remote == &remote; }), peers_.end());
}

bool RemoteExpertOpt::attach_device(int device, std::string& err) {
    for (const Sink& s : sinks_)
        if (s.device == device) return true;
    try {
        const OnDevice on(device);
        float* sum = nullptr;
        if (!gpu::alloc_host(&sum, (size_t) MAXT * (H + K) * sizeof(float))) return failed(err);
        std::memset(sum, 0, (size_t) MAXT * (H + K) * sizeof(float));
        sinks_.push_back({device, sum, reinterpret_cast<int32_t*>(sum + (int64_t) MAXT * H)});
        return true;
    } catch (const std::exception& e) {
        err = std::string("remote expert decode optimization: ") + e.what();
        return false;
    }
}

bool RemoteExpertOpt::init(std::string& err) {
    if (!attach_device(0, err)) return false;
    for (auto& p : peers_) {
        try {
            const OnDevice on(p.remote->device_);
            if (!gpu::alloc_device(&p.sum, (size_t) MAXT * H * sizeof(float))) return failed(err);
        } catch (const std::exception& e) {
            err = std::string("remote expert decode optimization: ") + e.what();
            return false;
        }
    }
    return true;
}

RemoteExpertOpt::~RemoteExpertOpt() {
    for (auto& p : peers_) {
        try {
            const OnDevice on(p.remote->device_);
            if (p.remote->stream_) gpu::stream_sync(p.remote->stream_);
            if (p.sum) gpu::free(p.sum);
        } catch (const std::exception&) {
        }
        p.remote->remote_opt_ = nullptr;
    }
    for (const Sink& s : sinks_) {
        try {
            const OnDevice on(s.device);
            gpu::free(s.sum);
        } catch (const std::exception&) {
        }
    }
}

size_t RemoteExpertOpt::metadata_bytes() { return sizeof(ReduceMeta); }

const RemoteExpertOpt::Sink* RemoteExpertOpt::sink_for(void* stream) const {
    Runtime* rt = stream != nullptr ? Runtime::owner_of_stream(stream) : &Runtime::get();
    if (rt == nullptr) return nullptr;
    const int dev = rt->ordinal();
    for (const Sink& s : sinks_)
        if (s.device == dev) return &s;
    return nullptr;
}

void RemoteExpertOpt::begin(const float* weights, int token_begin, int tokens) {
    weights_ = weights;
    token_begin_ = token_begin;
    tokens_ = tokens;
    for (const Sink& s : sinks_) {
        std::memset(s.sum + (int64_t) token_begin * H, 0, (size_t) tokens * H * sizeof(float));
        std::memset(s.mask + (int64_t) token_begin * K, 0, (size_t) tokens * K * sizeof(int32_t));
    }
}

void RemoteExpertOpt::prepare(const RemoteExperts& remote, void* metadata) const {
    auto* m = (ReduceMeta*) metadata;
    std::fill(m->row, m->row + tokens_ * K, -1);
    for (size_t i = 0; i < remote.original_row_.size(); ++i)
        m->row[remote.original_row_[i]] = (int32_t) i;
    std::memcpy(m->weight, weights_, (size_t) tokens_ * K * sizeof(float));
}

bool RemoteExpertOpt::reduce(RemoteExperts& remote, const void* metadata, std::string& err) {
    const auto it =
        std::find_if(peers_.begin(), peers_.end(), [&](const Peer& p) { return p.remote == &remote; });
    if (it == peers_.end()) {
        err = "remote expert decode optimization: unknown helper";
        return false;
    }
    try {
        // finish() synchronizes this stream before the host reads h_out_. Where the helper's kernels can
        // write its host USM in place the reduced rows land there directly - a copy fewer per layer.
        float* sum = remote.zero_copy_ ? remote.h_out_ : it->sum;
        launch_reduce(remote.d_out_, (const ReduceMeta*) metadata, sum, tokens_, remote.stream_);
        if (remote.zero_copy_) return true;
        if (!gpu::copy_async(remote.h_out_, it->sum, (size_t) tokens_ * H * sizeof(float), remote.stream_))
            return failed(err);
        return true;
    } catch (const std::exception& e) {
        err = std::string("remote expert decode optimization: ") + e.what();
        return false;
    }
}

void RemoteExpertOpt::accumulate(const RemoteExperts& remote) {
    for (const Sink& s : sinks_) {
        for (int64_t i = 0; i < (int64_t) tokens_ * H; ++i)
            s.sum[(int64_t) token_begin_ * H + i] += remote.h_out_[i];
        for (int i = 0; i < tokens_ * K; ++i)
            s.mask[token_begin_ * K + i] |= remote.owned_[(size_t) i] != 0;
    }
}

void RemoteExpertOpt::copy_rows(float* dst, const float* src, int token_begin, int tokens,
                                const int32_t* hit_rows, const int32_t* count, void* stream) const {
    const Sink* s = sink_for(stream);
    if (s == nullptr) return;
    launch_copy_cpu_rows(dst, src, (int64_t) tokens * K, s->mask + (int64_t) token_begin * K, hit_rows,
                         count, stream);
}

void RemoteExpertOpt::combine(float* dst, float* scratch, int token_begin, int tokens,
                              const uint32_t* skip, uint32_t ring, void* stream) const {
    const Sink* s = sink_for(stream);
    if (s == nullptr) return;
    const float* sum = s->sum + (int64_t) token_begin * H;
    if (skip) {
        // An all-primary device-planned group bypasses the host: ignore its stale sum.
        strata::kernels::copy_or_zero_from_mapped(scratch, sum, (long long) tokens * H, skip, ring, stream);
        sum = scratch;
    }
    strata::kernels::add_inplace(dst, sum, (int64_t) tokens * H, stream);
}

bool RemoteExpertOpt::owns(int64_t layer, int32_t expert) const {
    const bool admit_host = host_admit_.load(std::memory_order_relaxed) > 0;   // after a long prompt (open_host_admission)
    for (const auto& p : peers_)
        if (!(admit_host && p.remote->host_mode()) && p.remote->holds(layer, expert))
            return true;   // its cache's, or the arena's in host mode
    return false;
}

bool RemoteExpertOpt::adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
                            const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
                            ExpertSource& source) {
    if (max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<uint8_t> resident(primary.size(), 0);
    for (size_t i = 0; i < primary.size(); ++i) resident[i] = primary[i] >= 0;
    for (const auto& [i, slot] : pending) resident[(size_t) i] = 1;
    for (const auto& p : peers_) {
        const auto& r = *p.remote;
        for (int32_t l = 0; l < (int32_t) r.layers_present_.size(); ++l)
            for (int32_t e = 0; e < r.n_expert_; ++e)
                if (r.cache_.slot_of(l, e) >= 0) resident[(size_t) l * r.n_expert_ + e] = 1;
    }
    struct Swap { float gain; int32_t layer, in, out; };
    std::string err;
    for (auto& p : peers_) {
        auto& r = *p.remote;
        if (r.host_mode_) continue;   // the arena is not this tier's to swap: it holds everything already
        std::vector<Swap> swaps;
        std::vector<std::pair<float, int32_t>> candidates, victims;
        for (int32_t l = 0; l < (int32_t) r.layers_present_.size(); ++l) {
            candidates.clear();
            victims.clear();
            const size_t base = (size_t) l * r.n_expert_;
            float primary_cold = 1.0e30f;
            for (int32_t e = 0; e < r.n_expert_; ++e) {
                const float u = usage[base + e];
                if (primary[base + e] >= 0) primary_cold = std::min(primary_cold, u);
                if (r.cache_.slot_of(l, e) >= 0) victims.emplace_back(u, e);
                else if (!resident[base + e] && u >= 2.0f) candidates.emplace_back(u, e);
            }
            const size_t n = std::min(candidates.size(), victims.size());
            if (!n) continue;
            std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.first > b.first; });
            std::partial_sort(victims.begin(), victims.begin() + (ptrdiff_t) n, victims.end(),
                              [](auto& a, auto& b) { return a.first < b.first; });
            for (size_t i = 0; i < n; ++i) {
                if (victims[i].first > primary_cold) break;
                if (candidates[i].first < victims[i].first + 1.5f) break;
                swaps.push_back({candidates[i].first - victims[i].first, l, candidates[i].second,
                                 victims[i].second});
            }
        }
        std::sort(swaps.begin(), swaps.end(), [](auto& a, auto& b) { return a.gain > b.gain; });
        if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
        if (swaps.empty()) continue;
        try {
            const OnDevice on(r.device_);
            // a transient source blob (a GGUF read in place) is recycled once this returns: fill from a
            // copy held until the stream has consumed it (upstream's note_async_read, which the SYCL
            // ExpertSource does not have)
            std::vector<std::unique_ptr<uint8_t[]>> staged;
            for (const auto& s : swaps) {
                const uint8_t* in_blob = source.blob(s.layer, s.in);
                const int64_t bytes = (int64_t) lay.blob_bytes(s.layer);
                if (in_blob == nullptr) {
                    err = "remote expert decode optimization: no blob for the swap's incoming expert";
                    return false;
                }
                if (source.transient(s.layer, s.in)) {
                    staged.push_back(std::make_unique<uint8_t[]>((size_t) bytes));
                    std::memcpy(staged.back().get(), in_blob, (size_t) bytes);
                    in_blob = staged.back().get();
                }
                if (!r.cache_.fill_slot(r.cache_.slot_of(s.layer, s.out), in_blob, r.stream_, err, bytes))
                    return false;
            }
            if (!gpu::stream_sync(r.stream_)) return failed(err);
        } catch (const std::exception& e) {
            err = std::string("remote expert decode optimization: ") + e.what();
            return false;
        }
        for (const auto& s : swaps) {
            r.cache_.replace(s.layer, s.out, s.in);
            const size_t base = (size_t) s.layer * r.n_expert_;
            resident[base + s.out] = 0;
            resident[base + s.in] = 1;
        }
    }
    return true;
}
}  // namespace strata::core
