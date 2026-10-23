// Strata. src/dsv41/ds4_store.hpp - the RAM tier of the DeepSeek-V4.1 routed experts as a cache of its own: whole experts read
// from the model's files by unbuffered requests (the OS page cache neither used nor disturbed), kept in a fixed arena of slots
// that a replacement policy owns, filled on demand (the CPU lane's picks) and ahead of time (the lookahead's guesses).
//
// An expert (layer, e) is three matrices at three file offsets (the GGUF's one tensor a role and layer).  A slot holds their
// bytes at 4096-aligned places (unbuffered reads land the file range rounded out to 4096 at an aligned address; the matrix's
// first byte is `lead` bytes into it), so mat(slot, r) is the pointer the CPU lane's rows read - the same bytes as the file's
// mapping, so every answer is the same.  Nothing here touches a device.
//
// Memory: slots are allocated in chunks as they are first needed (VirtualAlloc / mmap, private memory, not pinned), never past
// `max_slots` nor past the RAM this process can still get less `floor_bytes` (the owner's session and the system keep theirs).
//
// Policy (pick the slot to give up, among the unpinned ready ones): lru (oldest touch), lfu (lowest decayed hit count), or
// lfu with an LRU tie-break.  A slot whose expert a window is computing is pinned from ensure() to release().
#pragma once

#include "ds4_os.hpp"
#include "ds4_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::ds4 {

class ExpertStore {
public:
    struct LayerGeo {
        int n_exp = 0;
        int file[3] = {-1, -1, -1};
        uint64_t off[3] = {0, 0, 0}, stride[3] = {0, 0, 0};   // matrix r of expert e: file[r], off[r] + e stride[r]
        uint64_t bytes[3] = {0, 0, 0};
    };
    enum Policy { kLru = 0, kLfu = 1, kLfuLru = 2 };
    struct Stats {
        std::atomic<int64_t> hits{0}, misses{0}, prefetched{0}, prefetch_used{0}, prefetch_wasted{0}, evictions{0}, fails{0};
        std::atomic<int64_t> bytes_read{0}, wait_ns{0};
    };

    struct Cfg {
        const std::vector<std::string>* paths = nullptr;   // the files, by the layers' file indices
        int threads = 8;
        size_t budget_bytes = 0;           // the arena at most (grown in chunks as slots are first needed)
        size_t floor_bytes = 8ull << 30;   // RAM left alone
        Policy policy = kLru;
        float decay = 0.97f;               // lfu: heat x decay every `decay_every` accesses of its layer
        int decay_every = 240;
        std::function<uint64_t()> ram_avail;   // the RAM this process can still get (~0: unknown)
    };

    bool init(const Cfg& c, const std::vector<LayerGeo>& layers, const std::vector<os::fd_t>& fds, const std::vector<os::fd_t>& dfds,
              const std::vector<os::fd_t>& pfds) {
        cfg_ = c;
        geo_ = layers;
        fds_ = fds;
        dfds_ = dfds;
        pfds_ = pfds;
        slot_bytes_ = 0;
        for (const LayerGeo& g : geo_) {
            size_t need = 0;
            for (int r = 0; r < 3; ++r) need += (size_t) g.bytes[r] + 2 * os::kDirectAlign;
            slot_bytes_ = std::max(slot_bytes_, need);
        }
        slot_bytes_ = (slot_bytes_ + os::kDirectAlign - 1) & ~(size_t) (os::kDirectAlign - 1);
        if (slot_bytes_ == 0 || c.budget_bytes < slot_bytes_) return false;
        nmax_ = c.budget_bytes / slot_bytes_;
        slots_.reset(new Slot[nmax_]);
        where_.resize(geo_.size());
        heat_.resize(geo_.size());
        hits_in_layer_.assign(geo_.size(), 0);
        for (size_t l = 0; l < geo_.size(); ++l) {
            where_[l].assign((size_t) geo_[l].n_exp, -1);
            heat_[l].assign((size_t) geo_[l].n_exp, 0.0f);
        }
        io_.reset(new IoQueue(std::max(1, c.threads)));
        return true;
    }
    ~ExpertStore() {
        io_.reset();
        for (auto& ch : chunks_) os::anon_free(ch.first, ch.second);
    }

    size_t slot_bytes() const { return slot_bytes_; }
    size_t slots_allocated() const { return nalloc_; }
    size_t slots_max() const { return nmax_; }
    const Stats& stats() const { return st_; }
    double busy_ms() const { return io_ ? io_->busy_ms() : 0.0; }

    /// The slot holding (layer, e) ready to read, or -1.  (Any thread; a slot found stays valid while it is pinned.)
    int find(int layer, int e) const {
        if (layer < 0 || layer >= (int) where_.size()) return -1;   // (a layer the store does not cover: the DSpark drafter's)
        const int s = where_[(size_t) layer][(size_t) e];
        if (s < 0) return -1;
        Slot& sp = slots_[(size_t) s];
        try_ready(sp);
        return sp.state.load(std::memory_order_acquire) == kReady && sp.layer == layer && sp.expert == e ? s : -1;
    }
    const uint8_t* mat(int slot, int r) const { return slots_[(size_t) slot].base + slots_[(size_t) slot].lead[r]; }

    /// Makes the experts es[0..n) of `layer` ready and pinned (reading what is not here), waiting for the reads; the
    /// caller releases them.  False if a read failed or no slot could be had for one (then none of the failed ones is
    /// pinned: ok[i] says which are; the caller reads those another way).
    bool ensure(int layer, const int* es, int n, std::vector<char>& ok) {
        ok.assign((size_t) n, 0);
        if (layer < 0 || layer >= (int) geo_.size()) return false;
        std::vector<int> sl((size_t) n, -1);
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < n; ++i) {
                const int e = es[i];
                touch_heat(layer, e);
                int s = where_[(size_t) layer][(size_t) e];
                if (s >= 0) {
                    Slot& sp = slots_[(size_t) s];
                    try_ready(sp);
                    if (sp.state.load() == kReady && !sp.prefetch) st_.hits.fetch_add(1, std::memory_order_relaxed);
                    if (sp.prefetch) st_.prefetch_used.fetch_add(1, std::memory_order_relaxed);
                    sp.prefetch = false;
                    sp.pin++;
                    sp.tick = ++clock_;
                    sl[(size_t) i] = s;
                    continue;
                }
                st_.misses.fetch_add(1, std::memory_order_relaxed);
                s = start_fill(layer, e, true);
                sl[(size_t) i] = s;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        bool all = true;
        for (int i = 0; i < n; ++i) {
            const int s = sl[(size_t) i];
            if (s < 0) {
                all = false;
                continue;
            }
            Slot& sp = slots_[(size_t) s];
            while (sp.left.load(std::memory_order_acquire) > 0) io_->wait_any(2000);
            std::lock_guard<std::mutex> lk(mu_);
            finish(sp);
            if (sp.state.load() == kReady) ok[(size_t) i] = 1;
            else {
                sp.pin--;
                sl[(size_t) i] = -1;
                all = false;
            }
        }
        st_.wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                              std::memory_order_relaxed);
        return all;
    }
    void release(int layer, const int* es, int n, const std::vector<char>& ok) {
        if (layer < 0 || layer >= (int) geo_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < n; ++i)
            if (ok[(size_t) i]) {
                const int s = where_[(size_t) layer][(size_t) es[i]];
                if (s >= 0 && slots_[(size_t) s].pin > 0) slots_[(size_t) s].pin--;
            }
    }
    /// Starts reading (layer, e) if it is neither here nor on its way (a guess: unpinned, may be given up before it is used).
    void prefetch(int layer, int e) {
        if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) return;
        if (where_[(size_t) layer][(size_t) e] >= 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (where_[(size_t) layer][(size_t) e] >= 0) return;
        if (start_fill(layer, e, false) >= 0) st_.prefetched.fetch_add(1, std::memory_order_relaxed);
    }
    /// Heat of an expert as the policy sees it (tests / the report).
    float heat(int layer, int e) const { return heat_[(size_t) layer][(size_t) e]; }

private:
    static constexpr int kFree = 0, kLoading = 1, kReady = 2, kFailed = 3;
    struct Slot {
        uint8_t* base = nullptr;
        uint32_t lead[3] = {0, 0, 0};
        int layer = -1, expert = -1, pin = 0;
        bool prefetch = false;
        std::atomic<int> state{kFree}, left{0};
        std::atomic<bool> bad{false};
        uint64_t tick = 0;
    };

    void touch_heat(int layer, int e) {
        heat_[(size_t) layer][(size_t) e] += 1.0f;
        if (cfg_.policy != kLru && ++hits_in_layer_[(size_t) layer] % (uint64_t) cfg_.decay_every == 0) {
            // (the layer's heat x decay: a linear pass over 384 floats every 240 accesses of that layer)
            for (float& h : heat_[(size_t) layer]) h *= cfg_.decay;
        }
    }
    // a slot for (layer, e) and its reads started; -1 if there is none to be had.  mu_ held.
    int start_fill(int layer, int e, bool demand) {
        int s = take_slot(demand);
        if (s < 0) {
            st_.fails.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        Slot& sp = slots_[(size_t) s];
        if (sp.layer >= 0 && sp.expert >= 0) {
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            if (sp.prefetch) st_.prefetch_wasted.fetch_add(1, std::memory_order_relaxed);
            st_.evictions.fetch_add(1, std::memory_order_relaxed);
        }
        sp.layer = layer;
        sp.expert = e;
        sp.pin = demand ? 1 : 0;
        sp.prefetch = !demand;
        sp.tick = ++clock_;
        sp.state.store(kLoading, std::memory_order_release);
        sp.bad.store(false);
        sp.left.store(3, std::memory_order_release);
        where_[(size_t) layer][(size_t) e] = s;
        const LayerGeo& g = geo_[(size_t) layer];
        IoQueue::Job jobs[3];
        size_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const uint64_t off = g.off[r] + (uint64_t) e * g.stride[r];
            const uint64_t lead = off % os::kDirectAlign;
            sp.lead[r] = (uint32_t) (at + lead);
            IoQueue::Job& j = jobs[r];
            j.fd = pfds_[(size_t) g.file[r]];
            j.dfd = dfds_[(size_t) g.file[r]];
            j.off = off;
            j.len = g.bytes[r];
            j.dst = sp.base + at;
            j.left = &sp.left;
            j.bad = &sp.bad;
            at += os::direct_span(off, g.bytes[r]);
        }
        st_.bytes_read.fetch_add((int64_t) (g.bytes[0] + g.bytes[1] + g.bytes[2]), std::memory_order_relaxed);
        io_->push(jobs, 3);
        return s;
    }
    // loading -> ready once its reads are done and whole (lock-free; a failed one is cleaned up by finish / take_slot)
    void try_ready(Slot& sp) const {
        int want = kLoading;
        if (sp.left.load(std::memory_order_acquire) == 0 && !sp.bad.load(std::memory_order_acquire))
            sp.state.compare_exchange_strong(want, kReady, std::memory_order_acq_rel);
    }
    void finish(Slot& sp) {   // mu_ held; reads done
        if (sp.state.load() != kLoading) return;
        if (sp.bad.load()) {
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            sp.layer = sp.expert = -1;
            sp.state.store(kFree, std::memory_order_release);
            sp.prefetch = false;
            st_.fails.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        sp.state.store(kReady, std::memory_order_release);
    }
    bool grow() {
        if (nalloc_ >= nmax_) return false;
        constexpr size_t kChunk = 64;   // slots
        const size_t n = std::min(kChunk, nmax_ - nalloc_);
        const size_t bytes = n * slot_bytes_;
        const uint64_t avail = cfg_.ram_avail ? cfg_.ram_avail() : ~uint64_t{0};
        if (avail != ~uint64_t{0} && avail < bytes + cfg_.floor_bytes) return false;
        void* p = os::anon_alloc(bytes, -1, 1);
        if (p == nullptr) return false;
        chunks_.push_back({p, bytes});
        for (size_t i = 0; i < n; ++i) slots_[nalloc_ + i].base = (uint8_t*) p + i * slot_bytes_;
        nalloc_ += n;
        return true;
    }
    // the slot to give up (or a free one): the policy's lowest among the unpinned ready ones.  mu_ held.
    int take_slot(bool demand) {
        for (size_t i = free_hint_; i < nalloc_; ++i)
            if (slots_[i].state.load(std::memory_order_acquire) == kFree) {
                free_hint_ = i + 1;
                return (int) i;
            }
        if (grow()) return take_slot(demand);
        int best = -1;
        double bv = 0;
        for (size_t i = 0; i < nalloc_; ++i) {
            Slot& sp = slots_[i];
            if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {
                try_ready(sp);
                if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);   // (failed: freed)
                if (sp.state.load(std::memory_order_acquire) == kFree) {
                    free_hint_ = std::min(free_hint_, i);
                    return (int) i;
                }
            }
            if (sp.pin > 0 || sp.state.load(std::memory_order_acquire) != kReady) continue;
            double v;
            if (cfg_.policy == kLru) v = (double) sp.tick;
            else {
                const double h = heat_[(size_t) sp.layer][(size_t) sp.expert];
                v = cfg_.policy == kLfu ? h : h * 1e12 + (double) sp.tick;   // lfu with the age as tie-break
                if (cfg_.policy == kLfuLru) v = h + 1e-9 * (double) sp.tick;
            }
            if (best < 0 || v < bv) {
                best = (int) i;
                bv = v;
            }
        }
        if (best < 0) return -1;
        free_hint_ = std::min<size_t>(free_hint_, (size_t) best);
        return best;
    }

    Cfg cfg_;
    std::vector<LayerGeo> geo_;
    std::vector<os::fd_t> fds_, dfds_, pfds_;
    size_t slot_bytes_ = 0, nmax_ = 0, nalloc_ = 0, free_hint_ = 0;
    std::unique_ptr<Slot[]> slots_;
    std::vector<std::pair<void*, size_t>> chunks_;
    std::vector<std::vector<int>> where_;
    std::vector<std::vector<float>> heat_;
    std::vector<uint64_t> hits_in_layer_;
    uint64_t clock_ = 0;
    std::mutex mu_;
    std::unique_ptr<IoQueue> io_;
    Stats st_;
};

}  // namespace strata::ds4
