// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_stream.hpp - a component's layers (the DiT's blocks, the text encoder's layers) on the GPU
// within a budget: the first layers that fit stay in VRAM, the rest go through two slots, each copied while the layer
// before it computes.
//
// A layer is a list of host spans (the tensors in the mapped file) copied into one device region.  The copy goes from
// the file's pages (the page cache: nothing is held in RAM for it) through a pinned relay buffer of the largest layer
// by DMA on a copy stream; a slot is reused once the compute stream has passed the layer that used it.  Each slot has
// its relay, so one layer's pass into its relay runs beside the other's DMA, and the pass is split over threads.  The layers
// are taken in order, as the DiT and the encoder walk them (the next streamed layer is fetched when one begins), so
// keeping a prefix resident and cycling the rest is what a budget below the whole can do best: an LRU cache over a
// cyclic walk misses every time.
#pragma once

#include "strata/core/gpu.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::image {

class LayerStream {
public:
    struct Span {
        const void* host = nullptr;   // in the mapped file
        uint64_t bytes = 0;
        uint64_t offset = 0;          // within the layer's device region
    };

    LayerStream();
    ~LayerStream();
    LayerStream(const LayerStream&) = delete;
    LayerStream& operator=(const LayerStream&) = delete;

    /// A layer of spans (offsets are assigned, each 256-byte aligned) -> its index.
    int add(std::vector<Span> spans);
    /// The offset of span `k` of layer `i` within its region.
    uint64_t offset(int i, int k) const { return layers_[(size_t) i].spans[(size_t) k].offset; }
    uint64_t bytes(int i) const { return layers_[(size_t) i].bytes; }
    uint64_t total() const;
    int count() const { return (int) layers_.size(); }
    /// The least budget configure takes when not every layer fits: two slots of the largest layer.
    uint64_t min_budget() const { return 2 * max_bytes_; }

    /// Sets the VRAM the layers may take (their resident prefix and, when not all fit, the two slots): a layer resident
    /// before and past the new prefix is freed.  `compute` is the stream the layers are used on.
    void configure(uint64_t budget, gpu::Stream compute);
    /// The layers that stay in VRAM (a prefix), and whether the rest are streamed.
    int resident() const { return resident_; }
    uint64_t vram() const;   // what the layers take now

    /// The device region of layer i for work about to be queued on the compute stream (which is made to wait for its
    /// copy), and the next streamed layer's copy started.
    const uint8_t* begin(int i);
    /// The compute stream's work on layer i is queued: its slot (if streamed) may be reused once that work is done.
    void end(int i);
    /// Work queued on the compute stream on a layer's device region once each time it is copied there, before its
    /// first use (the weights rewritten in place for the products that read them).
    void set_prepare(std::function<void(int, uint8_t*)> prepare) { prepare_ = std::move(prepare); }
    /// Frees every layer's device region and the slots (the budget stays, the layers come back on use).
    void drop();

private:
    struct Layer {
        std::vector<Span> spans;
        uint64_t bytes = 0;
        uint8_t* d = nullptr;   // resident copy
        bool loaded = false;
    };
    struct Slot {
        uint8_t* d = nullptr;
        int layer = -1;                     // the layer it holds (or is being filled with)
        gpu::Event* ready = nullptr;        // its copy done (on the copy stream)
        gpu::Event* used = nullptr;         // the compute stream past its layer
        bool issued = false;                // `ready` recorded for `layer`
        bool prepared = false;              // prepare_ queued on `layer`'s copy
    };
    void load_resident(int i);
    void fill(Slot& s, int i);              // the worker: host -> relay -> slot
    void worker();
    int slot_of(int i) const;
    void request(int i);

    std::vector<Layer> layers_;
    uint64_t budget_ = 0, max_bytes_ = 0;
    int resident_ = 0;
    bool streaming_ = false;
    gpu::Stream compute_ = nullptr, copy_ = nullptr;
    Slot slots_[2];
    int next_slot_ = 0;
    uint8_t* relay_[2] = {};                // slot k's (relay 0 also the resident loads')
    gpu::Event* relay_free_[2] = {};
    std::mutex relay_mu_[2];                // a relay: the worker's fills and the resident loads
    // the worker thread and its queue of layers to fill
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<int> queue_;
    bool stop_ = false;
    std::string error_;   // a fill's failure, thrown where its layer is begun
    std::function<void(int, uint8_t*)> prepare_;
};

}  // namespace strata::image
