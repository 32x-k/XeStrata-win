// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_stream.cpp - see include/strata/image/qi_stream.hpp.

#include "strata/image/qi_stream.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::image {

namespace {

constexpr uint64_t kAlign = 256;

void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("image engine (layers): ") + what + ": " + gpu::last_error());
}

// A layer's spans into its relay on several threads: one thread's pass from the page cache was the streamed layer's
// longest part.  A quarter of the hardware threads, at most 8, and at least 16 MiB each.
void copy_spans(uint8_t* dst, const std::vector<LayerStream::Span>& spans, uint64_t bytes) {
    static const int most = std::clamp((int) std::thread::hardware_concurrency() / 4, 1, 8);
    const int n = (int) std::clamp<uint64_t>(bytes >> 24, 1, (uint64_t) most);
    auto part = [&](uint64_t lo, uint64_t hi) {   // the bytes [lo, hi) of the spans laid end to end
        uint64_t at = 0;
        for (const LayerStream::Span& sp : spans) {
            const uint64_t a = std::max(lo, at), b = std::min(hi, at + sp.bytes);
            if (a < b) std::memcpy(dst + sp.offset + (a - at), (const uint8_t*) sp.host + (a - at), b - a);
            at += sp.bytes;
        }
    };
    uint64_t total = 0;
    for (const LayerStream::Span& sp : spans) total += sp.bytes;
    if (n == 1) { part(0, total); return; }
    std::vector<std::thread> th;
    for (int k = 1; k < n; ++k) th.emplace_back(part, total * (uint64_t) k / (uint64_t) n, total * (uint64_t) (k + 1) / (uint64_t) n);
    part(0, total / (uint64_t) n);
    for (std::thread& t : th) t.join();
}

}  // namespace

LayerStream::LayerStream() { thread_ = std::thread([this] { worker(); }); }

LayerStream::~LayerStream() {
    {
        std::lock_guard<std::mutex> l(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
    try {
        drop();
    } catch (const std::exception& e) {   // a destructor cannot throw: the GPU's own error is reported
        std::fprintf(stderr, "image engine (layers): %s\n", e.what());
    }
    if (copy_ != nullptr) gpu::stream_destroy(copy_);
}

int LayerStream::add(std::vector<Span> spans) {
    Layer l;
    for (Span& s : spans) {
        s.offset = l.bytes;
        l.bytes += (s.bytes + kAlign - 1) / kAlign * kAlign;
    }
    l.spans = std::move(spans);
    max_bytes_ = std::max(max_bytes_, l.bytes);
    layers_.push_back(std::move(l));
    return (int) layers_.size() - 1;
}

uint64_t LayerStream::total() const {
    uint64_t t = 0;
    for (const Layer& l : layers_) t += l.bytes;
    return t;
}

uint64_t LayerStream::vram() const {
    uint64_t t = 0;
    for (const Layer& l : layers_)
        if (l.d != nullptr) t += l.bytes;
    for (const Slot& s : slots_)
        if (s.d != nullptr) t += max_bytes_;
    return t;
}

void LayerStream::drop() {
    {   // the worker's queue drained
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return queue_.empty(); });
    }
    if (compute_ != nullptr) gpu::stream_sync(compute_);
    if (copy_ != nullptr) gpu::stream_sync(copy_);
    for (Layer& l : layers_) {
        if (l.d != nullptr) gpu::free(l.d);
        l.d = nullptr;
        l.loaded = false;
    }
    for (Slot& s : slots_) {
        if (s.d != nullptr) gpu::free(s.d);
        if (s.ready != nullptr) gpu::event_destroy(s.ready);
        if (s.used != nullptr) gpu::event_destroy(s.used);
        s = Slot{};
    }
    for (int k = 0; k < 2; ++k) {
        if (relay_[k] != nullptr) gpu::free(relay_[k]);
        if (relay_free_[k] != nullptr) gpu::event_destroy(relay_free_[k]);
        relay_[k] = nullptr;
        relay_free_[k] = nullptr;
    }
}

void LayerStream::configure(uint64_t budget, gpu::Stream compute) {
    compute_ = compute;
    if (copy_ == nullptr) copy_ = gpu::stream_create_copy();
    budget_ = budget;
    const uint64_t all = total();
    int R = (int) layers_.size();
    bool streaming = false;
    if (all > budget) {
        streaming = true;
        if (budget < 2 * max_bytes_)
            throw std::runtime_error("image engine (layers): the budget is below two layers (" +
                                     std::to_string(budget >> 20) + " MiB)");
        uint64_t used = 2 * max_bytes_;
        R = 0;
        while (R < (int) layers_.size() && used + layers_[(size_t) R].bytes <= budget) used += layers_[(size_t) R++].bytes;
    }
    // what no longer fits goes; the slots come and go with streaming
    {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return queue_.empty(); });
    }
    gpu::stream_sync(compute_);
    gpu::stream_sync(copy_);
    for (int i = R; i < (int) layers_.size(); ++i) {
        Layer& l = layers_[(size_t) i];
        if (l.d != nullptr) gpu::free(l.d);
        l.d = nullptr;
        l.loaded = false;
    }
    if (!streaming) {
        for (Slot& s : slots_) {
            if (s.d != nullptr) gpu::free(s.d);
            if (s.ready != nullptr) gpu::event_destroy(s.ready);
            if (s.used != nullptr) gpu::event_destroy(s.used);
            s = Slot{};
        }
        if (relay_[1] != nullptr) gpu::free(relay_[1]);
        relay_[1] = nullptr;
    } else {
        for (Slot& s : slots_) {
            if (s.d == nullptr) check(gpu::alloc_device(&s.d, max_bytes_), "a slot");
            if (s.ready == nullptr) check(gpu::event_create(&s.ready), "an event");
            if (s.used == nullptr) {
                check(gpu::event_create(&s.used), "an event");
                check(gpu::event_record(s.used, compute_), "an event");   // a slot never used: free
            }
            s.layer = -1;
            s.issued = false;
        }
    }
    for (int k = 0; k < (streaming ? 2 : 1); ++k) {
        if (relay_[k] == nullptr && (streaming || R > 0)) check(gpu::alloc_host(&relay_[k], max_bytes_), "the relay");
        if (relay_free_[k] == nullptr) {
            check(gpu::event_create(&relay_free_[k]), "an event");
            check(gpu::event_record(relay_free_[k], copy_), "an event");
        }
    }
    resident_ = R;
    streaming_ = streaming;
}

void LayerStream::load_resident(int i) {
    Layer& l = layers_[(size_t) i];
    if (l.d == nullptr) check(gpu::alloc_device(&l.d, l.bytes), "a layer");
    std::lock_guard<std::mutex> lk(relay_mu_[0]);
    gpu::event_sync(relay_free_[0]);
    copy_spans(relay_[0], l.spans, l.bytes);
    check(gpu::copy_async(l.d, relay_[0], l.bytes, copy_), "a layer's copy");
    check(gpu::event_record(relay_free_[0], copy_), "an event");
    // the compute stream waits for the copy where it uses the layer
    check(gpu::stream_wait_event(compute_, relay_free_[0]), "a wait");
    if (prepare_) prepare_(i, l.d);
    l.loaded = true;
}

int LayerStream::slot_of(int i) const {
    for (int k = 0; k < 2; ++k)
        if (slots_[k].layer == i) return k;
    return -1;
}

void LayerStream::request(int i) {
    std::lock_guard<std::mutex> lk(mu_);
    if (slot_of(i) >= 0) return;
    // the slot not holding the layer in use (the one begun last): the two alternate
    Slot& s = slots_[next_slot_];
    next_slot_ ^= 1;
    s.layer = i;
    s.issued = false;
    s.prepared = false;
    queue_.push_back(i);
    cv_.notify_all();
}

void LayerStream::fill(Slot& s, int i) {
    const Layer& l = layers_[(size_t) i];
    const int k = &s == &slots_[0] ? 0 : 1;              // the slot's relay
    std::lock_guard<std::mutex> lk(relay_mu_[k]);
    gpu::event_sync(relay_free_[k]);                    // the relay's last copy is done
    copy_spans(relay_[k], l.spans, l.bytes);
    check(gpu::stream_wait_event(copy_, s.used), "a wait");   // the compute stream past the slot's last layer
    check(gpu::copy_async(s.d, relay_[k], l.bytes, copy_), "a layer's copy");
    check(gpu::event_record(s.ready, copy_), "an event");
    check(gpu::event_record(relay_free_[k], copy_), "an event");
}

void LayerStream::worker() {
    for (;;) {
        int i;
        Slot* s = nullptr;
        {
            std::unique_lock<std::mutex> l(mu_);
            cv_.wait(l, [this] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            i = queue_.front();
            const int k = slot_of(i);
            s = k >= 0 ? &slots_[k] : nullptr;
        }
        std::string failed;
        if (s != nullptr) {
            try {
                fill(*s, i);
            } catch (const std::exception& e) {   // thrown on the caller's thread by begin()
                failed = e.what();
            }
        }
        {
            std::lock_guard<std::mutex> l(mu_);
            if (s != nullptr && s->layer == i) s->issued = true;
            if (!failed.empty()) error_ = failed;
            queue_.pop_front();
        }
        cv_.notify_all();
    }
}

const uint8_t* LayerStream::begin(int i) {
    if (i < resident_) {
        Layer& l = layers_[(size_t) i];
        if (!l.loaded) load_resident(i);
        return l.d;
    }
    request(i);
    int k = -1;
    {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] {
            k = slot_of(i);
            return !error_.empty() || (k >= 0 && slots_[k].issued);
        });
        if (!error_.empty()) {
            const std::string e = error_;
            error_.clear();
            throw std::runtime_error(e);
        }
    }
    if (k < 0 || k > 1) throw std::logic_error("image engine (layers): a begun layer has no slot");
    Slot* s = &slots_[k];
    check(gpu::stream_wait_event(compute_, s->ready), "a wait");
    if (prepare_) {
        bool first;
        {
            std::lock_guard<std::mutex> l(mu_);
            first = !s->prepared;
            s->prepared = true;
        }
        if (first) prepare_(i, s->d);
    }
    // the next streamed layer, wrapping to the first streamed one (the DiT's next step)
    int j = i + 1;
    if (j >= (int) layers_.size()) j = resident_;
    if (j != i && j >= resident_) request(j);
    return s->d;
}

void LayerStream::end(int i) {
    if (i < resident_) return;
    const int k = slot_of(i);
    if (k >= 0) check(gpu::event_record(slots_[k].used, compute_), "an event");
}

}  // namespace strata::image
