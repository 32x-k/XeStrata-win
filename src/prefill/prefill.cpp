// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#include "strata/prefill/prefill.hpp"
#include "strata/core/expert_tier.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/device.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/on_device.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/platform/direct_file.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/kernels.hpp"

#include "strata/core/gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// plan v0.3 P6: staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) strata::kernels::cpu::expert_layout().max_blob; }
constexpr int STAGE = 8;           // host->device expert staging ring (chunks below STREAM_ALL_MIN)
// Step 3: from this chunk size on, every non-resident expert of every layer streams in a fixed order through a
// RING_MAX-slot ring (nearly all 512 are routed at such a chunk), so the copy engine keeps working through the
// attention halves instead of waiting for each layer's routing.
constexpr int RING_MAX = 1024;          // the arrays; the ring itself is ring_slots()
// STRATA_TEST_PAGEABLE=1 (upstream cfe09ab5) makes every pinned host allocation with a pageable fallback in
// Prefill::init fail, so the fallbacks (what a container's memlock limit forces) run on a PC that can pin - for ASan /
// MALLOC_CHECK_=3 runs.
// #1454 (upstream 199fd1d5, 55dee8ed): `bo` shares `emb`'s storage (carve), which frees T * N floats per chunk.  The
// planner keeps counting them by default, so the auto chunk and the borrowed cache slots (and with them the prompt
// path's bits) stay as before; STRATA_EMB_REUSE_ACCOUNT=1 lets it use the saved bytes: a larger chunk or more borrowed
// slots where VRAM is the limit (upstream, RTX 3060, IQ3_XXS: chunk 6400 -> 6656, prompt +3.9%).
inline bool emb_reuse_account() {
    static const bool on = [] { const char* v = std::getenv("STRATA_EMB_REUSE_ACCOUNT"); return v != nullptr && v[0] == '1'; }();
    return on;
}
inline bool force_pageable() {
    static const bool v = [] { const char* e = std::getenv("STRATA_TEST_PAGEABLE"); return e != nullptr && e[0] == '1'; }();
    return v;
}
// A host table the GPU reads or writes every layer (the routing ids, the row maps, the group bounds): USM host
// memory that a kernel reads and writes in place (to_device / from_device: copy_i32_from_mapped), so these small copies
// never queue behind the streamed experts on the copy engine (upstream 1b3bb8f8).  A copy from or to pageable memory
// went through that engine: on the RTX 4070 the GPU waited 1.25 s of a 4K prompt at the routing syncs.  The host
// rewrites a table only after the next layer's sync, when the kernel that read it is done.  Pageable, with copies, when
// the allocation fails or STRATA_TEST_PAGEABLE=1.  resize keeps no contents.
template <class T> class HostVec {
public:
    HostVec() = default;
    HostVec(const HostVec&) = delete;
    HostVec& operator=(const HostVec&) = delete;
    ~HostVec() { release(); }
    void resize(size_t n) {
        if (n > cap_) {
            release();
            if (!force_pageable() && strata::gpu::alloc_host(&pinned_, n * sizeof(T))) cap_ = n;
            else { pinned_ = nullptr; fallback_.resize(n); cap_ = n; }
        }
        n_ = n;
    }
    T* data() { return pinned_ != nullptr ? pinned_ : fallback_.data(); }
    size_t size() const { return n_; }
    T& operator[](size_t i) { return data()[i]; }
    void from_device(const T* src, strata::gpu::Stream s) {   // the first size() values, queued on `s`
        if (pinned_ != nullptr) strata::kernels::copy_i32_from_mapped(pinned_, src, (int64_t) n_, s);
        else strata::gpu::copy_async(fallback_.data(), src, n_ * sizeof(T), s);
    }
    void to_device(T* dst, strata::gpu::Stream s) {
        if (pinned_ != nullptr) strata::kernels::copy_i32_from_mapped(dst, pinned_, (int64_t) n_, s);
        else strata::gpu::copy_async(dst, fallback_.data(), n_ * sizeof(T), s);
    }
private:
    static_assert(sizeof(T) == 4, "copied as int32");
    void release() {
        if (pinned_ != nullptr) strata::gpu::free(pinned_);
        pinned_ = nullptr;
        fallback_.clear();
        cap_ = n_ = 0;
    }
    T* pinned_ = nullptr;
    std::vector<T> fallback_;
    size_t cap_ = 0, n_ = 0;
};

constexpr int64_t STREAM_ALL_MIN = 2048;
// STRATA_PREFILL_CPU_SHARE (set_cpu_pool, upstream 978d3558, opt-in there): a chunk below STREAM_ALL_MIN hands the
// decode CPU pool - idle while a prompt is read - the non-resident experts few of its tokens route to, instead of
// streaming them.  `auto` (the default here): the share measured (each layer's CPU time per expert and the GPU's per
// streamed expert, events around its expert work read after the next layer's routing sync, as running means): the
// next layers hand the CPU g / (c + g), where both sides end together, so a GPU that streams faster than the CPU
// computes gives it little.  x: a fixed x.  0: every expert on the GPU.  The CPU rounds differently from the GPU, so
// an answer can differ slightly from the GPU-only one.  Unset, --prefill-cpu-share (set_cpu_share) fixes the share
// where the default arms it: the measured one moves from run to run, and which experts the CPU takes with it, so two
// greedy runs of one prompt parted (2026-10-16, an RTX 4070: logits up to 1.5 apart at the first generated token).
double g_cpu_share_fixed = -1.0;   // set_cpu_share: -1 measured
inline double cpu_share_env() {   // -1: measured
    static const double v = [] {
        const char* e = std::getenv("STRATA_PREFILL_CPU_SHARE");
        if (e == nullptr) return g_cpu_share_fixed;
        if (std::strcmp(e, "auto") == 0) return -1.0;
        return std::clamp(std::strtod(e, nullptr), 0.0, 1.0);
    }();
    return v;
}
inline bool cpu_share_on() { return cpu_share_env() != 0.0; }
double g_pinned_share = 1.0;
int g_ring_override = 0;
// The streamed ring: 384 slots when (nearly) every streamed expert is DMA'd from pinned RAM - measured on Q2_0,
// 8192-token chunks: 96 slots 1153 tok/s, 384 1294 (the next layer's experts arrive during its attention half) -
// and 96 when a large share goes through host copies (IQ3_S on 64 GB, a third unpinned: 96 slots 1216, 256 1070 -
// the host copies are the limit and the bigger ring only takes cache slots).  STRATA_PREFILL_RING overrides.
// On a card with little memory the ring takes at most an eighth of it (384 of IQ3_S's 2.7 MB blobs is 1 GiB, 3% of
// a 32 GB card), so the prompt path's buffers do not crowd out the chunk.  From 8192-token chunks on, up to 512 as
// long as the ring stays within kRingBigBytes (RTX 4070, IQ2_XS's 1.5 MB blobs, a 32K prompt in 8192-token chunks: 384
// slots 1296 tok/s, 512 1364, 768 1376; IQ3_XXS's 2.3 MB blobs, 64K: 384 1043, 512 1034, the ring's slots taken from
// the cache's resident experts).
constexpr uint64_t kRingBigBytes = 800ull << 20;
inline int ring_slots(size_t T) {
    const char* v = std::getenv("STRATA_PREFILL_RING");
    const uint64_t blob_b = (uint64_t) std::max<int64_t>(MAXBLOB(), 1);
    const int pinned = (int64_t) T >= 8192 ? (int) std::clamp<uint64_t>(kRingBigBytes / blob_b, 384, 512) : 384;
    int r = v ? (int) std::strtol(v, nullptr, 10)
              : (g_ring_override > 0 ? g_ring_override : g_pinned_share >= 0.9 ? pinned : 96);
    if (!v) {
        static strata::core::PerDevice<uint64_t> per_device;   // the current device's memory, read once for each GPU
        const int dev = strata::core::current_device();
        const uint64_t mem = per_device.get(strata::core::Runtime::at(dev).device(),
                                            [dev] { return strata::core::device_info(dev).total_bytes; });
        const uint64_t blob = (uint64_t) std::max<int64_t>(MAXBLOB(), 1);
        r = (int) std::min<uint64_t>((uint64_t) r, mem / 8 / blob);
    }
    const int big = std::clamp<int>(r, 16, RING_MAX);
    return (int64_t) T >= STREAM_ALL_MIN ? big : STAGE;
}
// The BF16-weight projections (hyper-connection, SSM alpha/beta, indexer, router, shared gate, PLE key/value) take
// BF16 activations here and FP32 ones in decode.  STRATA_PREFILL_BF16X2 adds each activation's BF16 remainder as a
// second GEMM (Y = W.hi + W.lo, ~16 mantissa bits; upstream 61638c1, 4e0592b): a router that picks its top 10 from
// the x decode would.  2 = all but the hyper-connection's; 1 = the hyper-connection's too (its activations are 10240
// wide and its up projection writes as much: slower); 0 (the default: it changes the prompt path's numbers) = off.
inline int bf16x2_mode() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_PREFILL_BF16X2");
        return e != nullptr ? (int) std::strtol(e, nullptr, 10) : 0;
    }();
    return v;
}
inline bool bf16x2() { return bf16x2_mode() != 0; }
inline bool bf16x2_hc() { return bf16x2_mode() == 1; }
// A small card's prompt chunk is what its buffers leave room for, and every chunk streams the experts again (an
// 8 GB RTX 4070 at a 64K prompt: 2048-token chunks, 291 tok/s against 1,066 with 12 GB).  Two places the buffers
// overlap: the hyper-connection block's rows (xn16, lo, lo16, gated: written and read before a half's GDN, QSA or MoE
// work starts, and xn16 again after it ends) live in the region that work uses (STRATA_PREFILL_HC_IN_REGION=0: their
// own place); the FP16 experts' input rows (Xs) live in the upper half of their down outputs' rows (Dm), each group's
// down written after its gate/up read them, the rows in ascending order (STRATA_PREFILL_XS_IN_DM=0: their own place).
inline bool env_off(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) == 0;
}
inline bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) != 0;
}
inline bool hc_in_region() { static const bool v = !env_off("STRATA_PREFILL_HC_IN_REGION"); return v; }
inline bool xs_in_dm() { static const bool v = !env_off("STRATA_PREFILL_XS_IN_DM"); return v; }
// Experts dequantized (FP16) before one grouped product: G of them take G x 9.4 MiB (--prefill-experts)
int g_expert_group = 16;
constexpr int64_t GU_ELEMS = 1280ll * 2560, D_ELEMS = 2560ll * 640;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    uint64_t failed_bytes = 0;          ///< the take that could not be allocated; the failure message names it
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) { used += bytes; return nullptr; }
        if (base != nullptr) {
            if (used + bytes > cap) { ok = false; failed_bytes = bytes; return nullptr; }
            T* p = reinterpret_cast<T*>(base + used);
            used += bytes;
            return p;
        }
        void* p = nullptr;
        if (!strata::gpu::alloc_device(&p, bytes)) { ok = false; failed_bytes = bytes; return nullptr; }
        owned->push_back(p);
        used += bytes;
        return (T*) p;
    }
};

// upstream 4a9b9041 (#796): what a failed carve had and wanted, for the "do not fit" error
std::string fit_failure(const Alloc& o) {
    size_t fb = 0, tb = 0;
    const bool known = strata::gpu::mem_info(&fb, &tb);
    return " (" + (known ? std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; "
                         : std::string()) +
           std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " + std::to_string(o.failed_bytes >> 20) +
           " MiB)";
}

}  // namespace

void set_cpu_share(double share) { g_cpu_share_fixed = share < 0.0 ? -1.0 : std::min(share, 1.0); }
double cpu_share_setting() { return cpu_share_env(); }

// Step 4 of the prompt-speed plan: the experts the arena could not pin (a third of the streamed ones on IQ3_S) are
// copied into pinned buffers by these threads, ahead of the launches.  Copied in line by the launching thread they
// left the GPU without queued work while each ~2 MB memcpy ran (~15 s of a 32K prompt on IQ3_S).  Job j - a layer's
// j-th unpinned expert, in launch order - lands in host buffer j % kRing, which is free again once the DMA of job
// j - kRing (recorded by the launching thread, `issued`) is done.
bool stager_sleep() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_STAGER_SLEEP");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
struct Stager {
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs
    int kRing = 16;
    // `from` set: a transient source (the GGUF read in place) copies the blob of (l, e) itself (copy_blob), since its
    // blob() pointers do not outlast a chunk's queue
    struct Job { const uint8_t* src; size_t bytes; core::ExpertSource* from = nullptr; int32_t l = 0, e = 0; };
    std::atomic<bool> failed{false};
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<strata::gpu::Event*> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        if (const char* v = std::getenv("STRATA_STAGER_RING")) kRing = (int) std::clamp<long>(std::strtol(v, nullptr, 10), 2, 256);
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = (char) (!force_pageable() && strata::gpu::alloc_host((void**) &buf[i], blob_bytes));
            if (!pinned[i]) {
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (!strata::gpu::event_create(&dma_done[i])) return false;
        }
        device = strata::core::current_device();
        for (int t = 0; t < nthreads; ++t)
            threads.emplace_back([this] {
                // --aux-cpus: the copy threads go to the spare CPUs with the other helpers; STRATA_AUX_STAGER=0 leaves them where
                // they were created (a small spare set would then hold all of them: it limits the copy rate of a long read)
                static const bool keep = [] { const char* v = std::getenv("STRATA_AUX_STAGER"); return v != nullptr && std::strtol(v, nullptr, 10) == 0; }();
                if (keep) strata::aux_cpus::note_owned_thread();
                else strata::aux_cpus::pin_current_thread();
                work();
            });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) strata::gpu::event_destroy(dma_done[i]);
            if (buf[i] && pinned[i]) strata::gpu::free(buf[i]);
        }
    }
    void work() {
        const strata::core::OnDevice on_device(device);   // the stage's GPU, on this worker thread too
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing)   // job j - kRing's DMA from this buffer is queued: sleep until it is (upstream #1057:
                                  // with --mmap-experts the 32 spinning stagers starved the launching thread)
                    for (int v = issued.load(std::memory_order_acquire); v <= j - kRing;
                         v = issued.load(std::memory_order_acquire)) {
                        if (stager_sleep()) issued.wait(v, std::memory_order_acquire);   // STRATA_STAGER_SLEEP=0: spin
                        else std::this_thread::yield();
                    }
                // and done - for a generation's first kRing jobs that is the previous generation's last DMA from
                // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP) or the DMA
                // was a ring entry the routing skipped (an event never recorded returns at once)
                strata::gpu::event_sync(dma_done[b]);
                const Job& jb = jobs[(size_t) j];
                if (jb.from != nullptr) {
                    if (!jb.from->copy_blob(jb.l, jb.e, buf[b])) failed.store(true, std::memory_order_release);
                } else {
                    std::memcpy(buf[b], jb.src, jb.bytes);
                }
                ready[(size_t) j].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        if (js.empty()) return;
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) std::this_thread::yield();
        return buf[j % kRing];   // a failed copy_blob leaves `failed` set: the chunk ends in an error
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, strata::gpu::Stream copy) {
        strata::gpu::event_record(dma_done[j % kRing], copy);
        issued.store(j + 1, std::memory_order_release);
        issued.notify_all();
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        issued.notify_all();
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};

// #1712 (upstream b97a36a4): a chunk's small int32 tables (the QSA step records, the token ids) go up from a pinned
// buffer.  A copy from pageable memory is staged by the runtime; on gfx1100 the first text prompt after a picture one
// was seen to read stale step records that way.  The event holds the next upload back until the last one has left the
// buffer.  No pinned RAM (or a table larger than the buffer): the copy is made from the caller's memory and the stream
// is waited for, so the source is free to change on return all the same (upstream #1996: a pageable source the caller
// changed before the copy read it faulted a 4-hour load on 2x RX 7900 XTX).
struct PinnedUpload {
    int32_t* p = nullptr;
    size_t n = 0;
    strata::gpu::Event* ev = nullptr;
    void init(size_t count) {   // at least `count` ints; a larger count takes a larger buffer once the last copy is out
        if (force_pageable() || (p && count <= n)) return;
        if (p) {
            if (ev && !strata::gpu::event_sync(ev)) return;   // keep the old buffer: begin() then falls back
            release();
        }
        if (!strata::gpu::alloc_host(&p, count * sizeof(int32_t))) {
            p = nullptr;
            return;
        }
        n = count;
        if (!strata::gpu::event_create(&ev)) {
            strata::gpu::free(p);
            p = nullptr;
            n = 0;
            ev = nullptr;
        }
    }
    bool upload(void* dst, const int32_t* src, size_t count, strata::gpu::Stream s) {
        const size_t bytes = count * sizeof(int32_t);
        if (!p || count > n) return strata::gpu::copy_async(dst, src, bytes, s) && strata::gpu::stream_sync(s);
        if (!strata::gpu::event_sync(ev)) return false;   // the previous upload is out of the buffer
        std::memcpy(p, src, bytes);
        return strata::gpu::copy_async(dst, p, bytes, s) && strata::gpu::event_record(ev, s);
    }
    /// For several copies out of one buffer: the pinned buffer once the last upload has left it (nullptr: none of
    /// `count` ints, so copy from the source and wait for the stream), then end() once the copies are queued.
    int32_t* begin(size_t count) { return p && count <= n && strata::gpu::event_sync(ev) ? p : nullptr; }
    bool end(strata::gpu::Stream s) { return strata::gpu::event_record(ev, s); }
    void release() {
        if (ev) strata::gpu::event_destroy(ev);
        if (p) strata::gpu::free(p);
        p = nullptr;
        ev = nullptr;
        n = 0;
    }
};

// STRATA_HANDOFF_PIPE=P (2..16, default 8; 0/1 = the one copy; upstream 1d9758f8, opt-in there): a layer split hands a
// chunk's rows on in P row pieces.  The stage records an event when its chunk's work is done and one after each piece's copy
// into the pinned hand-off buffer, waits on the host only for the first (what on_stage_chunk's checkpoint parts and the
// error check need), and starts the next stage at once: its thread queues piece i's upload once piece i has landed, so
// the uploads overlap the remaining downloads (each GPU's own link).  The wait is the host's, not a queue's: the two
// stages can be GPUs of two backends.  The same bytes land in the same rows before the next stage's first kernel:
// exact.  No device memory; the events and the existing two pinned buffers per stage only.  B70 + RTX 4070, IQ3_XXS,
// serve 32K, --layer-split 24: prompt 1,866 -> 1,868 tok/s, decode 75.8 -> 76.0 (equal), the same answers.
static constexpr int kHandPiecesMax = 16;
static int handoff_pipe_pieces() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_HANDOFF_PIPE");
        const int p = e ? (int) std::strtol(e, nullptr, 10) : 8;
        return p >= 2 ? std::min(p, kHandPiecesMax) : 0;
    }();
    return v;
}
// STRATA_HANDOFF_PIPE: the next stage's chunks run on one thread of the link's own, started once (std::async starts a
// thread per chunk: ~1 ms from the hand-off to the next stage's first call, upstream's nsys on CUDA).  The futures are
// the same (run's result, or what it threw), so the chain's waits do not change.  STRATA_HANDOFF_PIPE_WORKER=0: a
// thread per chunk again, for an A/B of the two halves.
static bool handoff_pipe_worker() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_HANDOFF_PIPE_WORKER");
        return !e || std::strtol(e, nullptr, 10) != 0;
    }();
    return handoff_pipe_pieces() > 0 && v;
}

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    int64_t T = 0, T_max = 0;
    bool borrowed = false;
    strata::gpu::Stream cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    uint16_t *xn16_lo = nullptr, *lo16_lo = nullptr, *mixed_bf_lo = nullptr;   // bf16x2(): the BF16 GEMMs' low parts
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    PinnedUpload steps_up;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    // step 2b (MMQ): the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    float* H = nullptr;
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    int32_t* ids_rel = nullptr;               // row r of the chunk -> r - its group's first row (the gate/up's dst rows)
    HostVec<int32_t> ids_rel_host;
    std::vector<int32_t> grp_start, grp_of, grp_pos;   // the groups over `order`: starts (ng + 1), and expert j's group and place
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    uint8_t* mring = nullptr;                 // MMQ_RING slots of MMQ_SLOT() bytes (chunks below STREAM_ALL_MIN)
    strata::gpu::Event* mgrp_used[2] = {};    // a group of ring slots read by its products, by the group's parity
    bool mgrp_live[2] = {};
    HostVec<int32_t> bounds_host;
    // the MMQ products' pointer tables (mmq_tab_on): expert j's gate|up and down, n_expert pointers each, as two
    // int32 words a pointer (HostVec copies int32)
    int32_t* tab_dev = nullptr;
    HostVec<int32_t> tab_host;
    std::unique_ptr<mmq::Context> mmq_ctx;
    HostVec<int32_t> ids_host, slot_host, src_host;
    std::vector<int32_t> cnt, off;
    uint16_t* dq_gu = nullptr;               // G experts' gate/up, GU_ELEMS apart, and down, D_ELEMS apart
    uint16_t* dq_d = nullptr;
    int32_t* xb_dev = nullptr;               // the experts' first rows in `order`, and the total (grouped products)
    HostVec<int32_t> xb_host;
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies (step 4)
    strata::gpu::Event *copied[RING_MAX] = {}, *used[RING_MAX] = {};
    int used_of[RING_MAX] = {};              // the slot whose `used` event releases this one (a gathered group's last)
    bool stage_live[RING_MAX] = {};
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    strata::gpu::Event* ple_copied[2] = {};          // one runs; the event marks that buffer's upload done
    std::vector<uint32_t> ple_rows[2];
    float* ple_norm = nullptr;
    uint8_t* region = nullptr;               // the attention/MoE scratch region (idle while the PLE block runs)
    uint64_t region_bytes = 0;
    PrefillStats* stats = nullptr;
    // KV streaming: one layer's whole K/V, staged from the host copy per layer and chunk (identity layout)
    strata::kernels::KvHostPools stage;
    int32_t* ident_table = nullptr;
    // set_cpu_pool: a small chunk's CPU experts.  The layer's MoE input (T x N, from `mixed`) and the rows the pool
    // writes (the tail of Dm, in Dm's row order), both in host memory; the pool's activations and jobs
    float *cpu_x = nullptr, *cpu_rows = nullptr;
    size_t cpu_x_n = 0, cpu_rows_n = 0;
    bool cpu_dead = false;   // a pinned buffer of the share could not be had (host memory is short): the GPU does it all
    std::vector<strata::kernels::cpu::NativeAct> cpu_nact;
    std::vector<strata::kernels::cpu::ExpertJobMulti> cpu_jobs;
    // the measured share: running means of the CPU's ms per expert and the GPU's per streamed expert
    double cpu_c_ms = 0, cpu_g_ms = 0, cpu_share_now = 0.5;
    strata::gpu::Event* cpu_ev[2] = {};   // the GPU's expert work of the last CPU-sharing layer
    bool cpu_pend = false;
    double pend_cpu_ms = 0;
    int64_t pend_n_cpu = 0, pend_n_gpu = 0;
    // ... and whether sharing pays at all (auto, upstream f2a3b946, #1282): an eligible layer's wall time per
    // non-resident expert, from before its routing sync to its combine (events, two sets: a layer's start is recorded
    // before the last one's is read); two adjacent layers of the two arms give a ratio (with / without), and the median
    // of the last CPU_RATIOS decides.  Layers differ up to 8x in that cost, the same in every run, which is why adjacent
    // layers are compared rather than a running mean per arm.
    static constexpr int CPU_RATIOS = 5;
    int64_t cpu_layers = 0;     // the eligible layers so far (the arms' schedule)
    int64_t cpu_first_l = -1;   // the model's first eligible layer: 2x the others' cost in both arms, never read
    strata::gpu::Event* cpu_wall[4] = {};
    int pend_wall = 0, pend_set = 0, pend_l = 0;   // pend_wall 1: the last eligible layer ran without the share, 2: with
    int64_t pend_wall_n = 0;
    int last_arm = 0, last_l = 0;
    double last_w = 0;
    double cpu_ratio[CPU_RATIOS] = {};
    int cpu_nratio = 0;
    bool cpu_gate = false;   // the median ratio is below 1: the layers that share are the cheaper ones
    // layer split: the device, and the hand-off to the next stage (two pinned chunk buffers, used in turn).
    // LAYER SPLIT, same process only: allocated in this stage's context, the next stage reads them where they are
    int device = -1;
    float* hand[2] = {};
    // STRATA_HANDOFF_PIPE: per hand-off buffer, "the chunk's work is done" and one "landed" event per piece
    int hand_pieces = 0;
    strata::gpu::Event* hand_done[2] = {};
    strata::gpu::Event* hand_ev[2][kHandPiecesMax] = {};
    // C-4: the chunk's token ids on the device, for one batched embedding gather
    int32_t* tok_dev = nullptr;
    std::vector<int32_t> tok_host;
    PinnedUpload tok_up;
    // the draft K/V pass's token ids, a batch at a time from one host vector (upstream #1996)
    PinnedUpload draft_tok_up;
    // a chunk's picture embedding rows, gathered from the caller's memory (upstream #1996); taken at the first picture
    PinnedUpload emb_up;
};

namespace {
// the staging pool of a streamed session: every page of one layer (same sequence in init and bytes_needed)
// STRATA_KV_STAGE_OWN (A/B only): the staging pool gets its own allocation instead of borrowed expert slots, so a
// streamed run lends the prompt path exactly the slots a resident one does (a lent expert runs on the CPU, which
// rounds differently: without this an A/B compares two expert placements as well as two KV placements)
bool stage_own() { static const bool v = std::getenv("STRATA_KV_STAGE_OWN") != nullptr; return v; }
void take_stage(Alloc& o_borrowed, const core::SessionState& ss, const strata::kernels::QsaShapes& s,
                strata::kernels::KvHostPools& st, bool& ok) {
    const core::QsaState& q0 = ss.qsa_states[ss.qsa_primary()];
    if (q0.kv_mode != 1) return;
    if (stage_own() && o_borrowed.count_only) return;
    Alloc own;
    own.owned = o_borrowed.owned;
    Alloc& o = stage_own() ? own : o_borrowed;
    const size_t rows = (size_t) q0.n_pages * s.n_head_kv * s.page_size;
    if (q0.kv_hybrid) {   // K8V4: the three runs of kKvHybrid; pools_of() then reads as the hybrid pools (mode 3)
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_q4) {
        st.k_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_int8) {
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.v_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
    } else {
        st.k_pool = o.take<uint16_t>(rows * s.head_dim, ok);
        st.v_pool = o.take<uint16_t>(rows * s.head_dim, ok);
    }
}
strata::kernels::QsaAttnPools pools_of(const strata::kernels::KvHostPools& h, const int32_t* table) {
    strata::kernels::QsaAttnPools p;
    p.k_pool = h.k_pool; p.v_pool = h.v_pool; p.k_q = h.k_q; p.v_q = h.v_q; p.k_scale = h.k_scale; p.v_scale = h.v_scale;
    p.k_q4 = h.k_q4; p.v_q4 = h.v_q4;
    p.page_table = table;
    return p;
}
}  // namespace

// set_helper: a helper GPU's share of a small chunk's experts.  Per MoE layer this GPU gathers the rows of the tokens
// the helper's experts read and copies them to `x_ex`, host memory registered on both GPUs; the helper copies them in,
// multiplies them against its cache's blobs (int8 products through a table of the blobs, as the plain models' prompt
// path takes them) and copies its rows to `d_ex`, which this GPU copies into Dm's tail before the combine.  The host
// waits between the two GPUs: they share no events (another backend, as a B70's and an RTX 4070's).
struct Prefill::HelperShare {
    core::RemoteExperts* h = nullptr;
    int prim = -1, dev = -1;       // this GPU, the helper
    int64_t max_rows = 0;          // an expert's rows at most
    int64_t cap = 0;               // a layer's rows at most
    int32_t* tok_dev = nullptr;    // this GPU's: the tokens to gather, the gathered rows
    float* xg = nullptr;
    float* x_ex = nullptr;         // both GPUs': the rows there, the results back
    float* d_ex = nullptr;
    strata::gpu::Stream s = nullptr;   // the helper's
    std::unique_ptr<mmq::Context> ctx;
    float *x = nullptr, *gu = nullptr, *hh = nullptr, *d = nullptr;
    void *xq = nullptr, *xq2 = nullptr, *hq = nullptr;
    int32_t* meta = nullptr;       // src [cap] | bounds [cap + 1] | identity [cap]
    const uint8_t** wtab = nullptr;
    int32_t* h_meta = nullptr;     // the helper's host USM, staging the two above
    const uint8_t** h_wtab = nullptr;
    std::vector<int32_t> toks, tok_of;
    bool live = false;
    int64_t rows = 0;
    ~HelperShare() {
        try {
            if (dev >= 0) {
                const core::OnDevice on(dev);
                if (s != nullptr) strata::gpu::stream_sync(s);
                ctx.reset();
                for (const void* p : {(const void*) x, (const void*) gu, (const void*) hh, (const void*) d, (const void*) xq,
                                      (const void*) xq2, (const void*) hq, (const void*) meta, (const void*) wtab,
                                      (const void*) h_meta, (const void*) h_wtab})
                    strata::gpu::free(p);
                if (s != nullptr) strata::gpu::stream_destroy(s);
            }
            if (prim >= 0) {
                const core::OnDevice on(prim);
                strata::gpu::device_sync();
                strata::gpu::free(tok_dev);
                strata::gpu::free(xg);
            }
            for (float* p : {x_ex, d_ex}) {
                if (p == nullptr) continue;
                if (prim >= 0) strata::gpu::host_unregister_on(prim, p);
                if (dev >= 0) strata::gpu::host_unregister_on(dev, p);
                strata::platform::DirectFile::free_aligned(p);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "prefill: releasing the helper share: %s\n", e.what());
        }
    }
};

bool Prefill::set_helper(core::RemoteExperts* helper, int64_t max_rows, std::string& err) {
    helper_.reset();
    if (helper == nullptr || helper->device() < 0 || max_rows <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.fmt.empty()) { err = "the helper's prompt share needs a native pack"; return false; }
    auto hs = std::make_unique<HelperShare>();
    hs->h = helper;
    hs->prim = core::current_device();
    hs->max_rows = max_rows;
    // a layer's rows at most: an 8192-row share is ~84 MB each of rows and results, and few layers of a chunk below
    // the streamed walk's 2048 tokens give the helper more
    hs->cap = 8192;
    const int64_t cap = hs->cap;
    int64_t gu_w = 0, ff = 0;
    for (const auto& f : lay.fmt) { gu_w = std::max<int64_t>(gu_w, 2 * f.n_ff); ff = std::max<int64_t>(ff, f.n_ff); }
    const size_t rows_b = (size_t) cap * N * sizeof(float);
    try {
        {
            const core::OnDevice on(hs->dev = helper->device());
            if (!mmq::chosen() && !mmq::built_dp4a()) { err = "the helper GPU has no int8 products"; return false; }
            for (const auto& f : lay.fmt)
                if (!mmq::supported(f.gu_type) || !mmq::supported(f.d_type) ||
                    (strata::kernels::cpu::native_mixed_gu(f) && !mmq::supported(f.up_type))) {
                    err = "the helper GPU's int8 products do not take these experts' types";
                    return false;
                }
            hs->s = strata::gpu::stream_create();
            hs->ctx = std::make_unique<mmq::Context>();
            const bool ok = hs->s != nullptr &&
                strata::gpu::alloc_device(&hs->x, rows_b) && strata::gpu::alloc_device(&hs->d, rows_b) &&
                strata::gpu::alloc_device(&hs->gu, (size_t) cap * gu_w * sizeof(float)) &&
                strata::gpu::alloc_device(&hs->hh, (size_t) cap * ff * sizeof(float)) &&
                strata::gpu::alloc_device(&hs->xq, mmq::q8_bytes(cap, N) * 2) &&
                strata::gpu::alloc_device(&hs->xq2, mmq::q8_bytes(cap, N) * 2) &&
                strata::gpu::alloc_device(&hs->hq, mmq::q8_bytes(cap, ff) * 2) &&
                strata::gpu::alloc_device(&hs->meta, (size_t) (3 * cap + 1) * sizeof(int32_t)) &&
                strata::gpu::alloc_device(&hs->wtab, (size_t) cap * sizeof(void*)) &&
                strata::gpu::alloc_host(&hs->h_meta, (size_t) (3 * cap + 1) * sizeof(int32_t)) &&
                strata::gpu::alloc_host(&hs->h_wtab, (size_t) cap * sizeof(void*));
            if (!ok) { err = std::string("the helper GPU's buffers: ") + strata::gpu::last_error(); return false; }
            for (int64_t i = 0; i < cap; ++i) hs->h_meta[2 * cap + 1 + i] = (int32_t) i;   // the identity
        }
        if (!strata::gpu::alloc_device(&hs->tok_dev, (size_t) cap * sizeof(int32_t)) ||
            !strata::gpu::alloc_device(&hs->xg, rows_b)) {
            err = std::string("this GPU's buffers for the helper: ") + strata::gpu::last_error();
            return false;
        }
        for (float** p : {&hs->x_ex, &hs->d_ex}) {
            *p = static_cast<float*>(strata::platform::DirectFile::alloc_aligned(rows_b));
            if (*p == nullptr) { err = "no host memory for the helper's rows"; return false; }
            for (const int d : {hs->prim, hs->dev})
                if (!strata::gpu::host_register_on(d, *p, rows_b)) {
                    err = std::string("registering the helper's rows on GPU ") + std::to_string(d) + ": " +
                          strata::gpu::last_error();
                    return false;
                }
        }
    } catch (const std::exception& e) {
        err = std::string("the helper share: ") + e.what();
        return false;
    }
    hs->tok_of.assign((size_t) 1 << 16, -1);
    helper_ = std::move(hs);
    return true;
}

Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() { release(); }
void Prefill::reset() { release(); impl_ = std::make_unique<Impl>(); stats_ = {}; }
void Prefill::release() {
    if (!impl_) return;
    impl_->stager.reset();
    if (impl_->cs) strata::gpu::stream_sync(impl_->cs);
    if (impl_->copy) strata::gpu::stream_sync(impl_->copy);
    impl_->steps_up.release();
    impl_->tok_up.release();
    impl_->draft_tok_up.release();
    impl_->emb_up.release();
    for (float* p : {impl_->cpu_x, impl_->cpu_rows})
        if (p) strata::gpu::free(p);
    for (strata::gpu::Event* e : impl_->cpu_ev)
        if (e) strata::gpu::event_destroy(e);
    for (strata::gpu::Event* e : impl_->cpu_wall)
        if (e) strata::gpu::event_destroy(e);
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) strata::gpu::event_destroy(impl_->copied[i]);
        if (impl_->used[i]) strata::gpu::event_destroy(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->mgrp_used[b]) strata::gpu::event_destroy(impl_->mgrp_used[b]);
        if (impl_->hand[b]) strata::gpu::free(impl_->hand[b]);
        if (impl_->hand_done[b]) strata::gpu::event_destroy(impl_->hand_done[b]);
        for (strata::gpu::Event* e : impl_->hand_ev[b])
            if (e) strata::gpu::event_destroy(e);
        if (impl_->ple_copied[b]) strata::gpu::event_destroy(impl_->ple_copied[b]);
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty()) strata::gpu::free(impl_->ple_emb_host[b]);
    }
    if (impl_->copy) strata::gpu::stream_destroy(impl_->copy);
    for (void* p : impl_->owned) strata::gpu::free(p);
}

namespace {
constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight

// THE ATTENTION HALF AND THE MoE HALF SHARE THEIR BUFFERS.  A layer runs its attention (GDN or QSA), writes it back
// into the residual, and only then its MoE, so the three sets of scratch are never live at once: one region the size
// of the largest holds them all.  That is ~260 KB of the ~680 KB a prompt token cost - which is what lets a chunk
// grow (every expert is streamed once per chunk, so a bigger chunk streams fewer bytes per token).  The sizes are
// counted with the same `take` sequence `init` uses; a mismatch makes `init` fail with "do not fit", never overlap.
uint64_t gdn_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const strata::kernels::QsaShapes& s) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
// Step 2b: which layers' experts go through MMQ (a native pack's layer whose two weight types it covers; a Strata
// pack's experts keep the FP16 path), whether any layer keeps the FP16 path (a type MMQ lacks), and the largest
// gate/up and down matrices a group buffer slot holds.  STRATA_PREFILL_MMQ=0: the FP16 path everywhere (the A/B).
constexpr int MMQ_GROUP = 16;                  // experts per MMQ launch
// The int8 products' group scratch (GU, H, Hq) holds one group's rows instead of the chunk's T x K when the group
// cap is on (Prefill::set_group_cap: the chunk's planner, where the chunk-sized buffers do not fit the cache): a
// group of up to MMQ_GROUP experts ends early when its rows would pass the cap (an expert alone has up to T rows,
// so at least T).  Chunk-sized, GU/H/Hq took 0.69 GB of an 8192-token chunk's buffers and kept a 12 GB card
// (RTX 4070) at that chunk for a 32K prompt; capped at a chunk's rows / 8 they take 0.1 GB.  Off where the
// chunk-sized buffers fit: the smaller groups cost launches (B70, 1K: prefill 1167 -> 1039 tok/s), and a group's
// tile shape rounds its sums, so the answer can differ from the uncapped run's.  STRATA_MMQ_GROUP_CAP=0 / 1: never /
// always.
bool g_group_cap = false;
inline int64_t mmq_group_rows_cap(size_t T) {
    static const int env = [] { const char* v = std::getenv("STRATA_MMQ_GROUP_CAP"); return v == nullptr ? -1 : (*v != '0' ? 1 : 0); }();
    const bool on = env < 0 ? g_group_cap : env != 0;
    return on ? std::max<int64_t>((int64_t) T, ((int64_t) T * K + 7) / 8) : (int64_t) T * K;
}
// Below STREAM_ALL_MIN the MMQ layers' experts are multiplied where their blobs land: a ring of two groups' slots in
// one allocation (a group's slots MMQ_SLOT apart), the streamed ones copied there from the host and the resident ones
// from the cache, so that a group's products read its experts in place.  A group's slots are released together, when
// its products are done (a gather into separate group buffers was three copies an expert, and the RTX 4070's prompt
// path waited on their launches: 980 ms of 4170).  The streamed walk (STREAM_ALL_MIN on) still gathers.
constexpr int MMQ_RING = 2 * MMQ_GROUP;
inline size_t MMQ_SLOT() { return ((size_t) MAXBLOB() + 255) & ~(size_t) 255; }
// The MMQ products read a resident expert where it is, in its cache slot, through a pointer table (upstream 7d9b50dc's
// STRATA_MMQ_DIRECT; XeStrata's int8 products read no byte past a matrix, so no guard): it is no longer copied into
// the group's buffer or ring slot first.  The same bytes are read, so the same answers.  STRATA_MMQ_TAB=0: copied, as
// before.
inline bool mmq_tab_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MMQ_TAB");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
struct MmqPlan {
    bool any = false, fallback = true;
    std::vector<char> layer;                   // per layer: MMQ
    size_t gu_max = 0, d_max = 0;
    int64_t int8_below = INT64_MAX;            // a layer of a chunk takes MMQ while its experts average fewer rows
};
// STRATA_PREFILL_MMQ: 0 the FP16 path, 1 the int8 products in every layer that takes them; unset, each chunk's layer
// by its experts' rows against what this GPU measured (int8_rows_below).
int mmq_env() {
    const char* env = std::getenv("STRATA_PREFILL_MMQ");
    return env == nullptr || env[0] == '\0' ? -1 : (int) std::strtol(env, nullptr, 10);
}
bool mmq_wanted() { return mmq::chosen() && mmq_env() != 0; }

// From how many rows an expert the FP16 path beats the int8 products on this GPU, timed once at the start on 8 experts
// of the model's first int8 layer (its types and shape, synthetic blocks: any bits time alike) at 16 to 1024 rows an
// expert (256 where the device cannot hold the larger buffers), in the calls a chunk makes: the int8 products from the blocks, against the experts dequantized to FP16 and
// their grouped products.  The geometric middle of the last row count int8 won and the first it lost; INT64_MAX where
// it won at every count, 0 where it lost at 16.  The int8 products were a fixed choice before (FP16 where Intel's XMX
// take it): B70, Qwen3.8 IQ3_XXS, the int8 form 1K 923 -> 1074 tok/s, 4K 1594 -> 1604, 32K 2526 -> 2075.  Where the
// timing cannot run, that fixed choice.  The int8 time grows with the rows and the FP16 one hardly does, so a type whose
// int8 products are cheap crosses later: B70, IQ3_S's IQ4_XS and UD-Q4_K_XL's K-quants still won at 256 rows
// (487/609, 592/669 us), and a 32K chunk, about 600 rows an expert, ran its int8 products: 32K prompt 2,318 -> 1,896
// and 2,125 -> 1,721 tok/s.
int64_t int8_rows_below(const strata::kernels::cpu::NativeFmt& f, size_t blob) {
    namespace k = strata::kernels;
    const int64_t fixed = k::xmx_available(k::XmxType::f16) ? 0 : INT64_MAX;
    constexpr int G = 8;
    constexpr int64_t RS[] = {16, 32, 64, 128, 256, 512, 1024};
    const int64_t FF = f.n_ff, NE = f.n_embd;
    std::vector<void*> owned;
    struct FreeAll {
        std::vector<void*>& o;
        ~FreeAll() { for (void* p : o) strata::gpu::free(p); }
    } free_all{owned};
    auto dev = [&](size_t bytes) {
        void* p = strata::gpu::alloc_device(bytes);
        if (p != nullptr) owned.push_back(p);
        return p;
    };
    // the row buffers for the most rows timed: 1024 an expert, else 256 (a device short of memory)
    int64_t r_top = 1024;
    if (void* probe = strata::gpu::alloc_device((size_t) r_top * G * (NE * 10 + FF * 13 + 4) + mmq::q8_bytes(r_top * G, NE) * 2))
        strata::gpu::free(probe);
    else
        r_top = 256;
    const int64_t RMAX = r_top * G;
    auto* w = static_cast<uint8_t*>(dev((size_t) G * blob));
    auto* dq_gu = static_cast<uint16_t*>(dev((size_t) G * 2 * FF * NE * 2));
    auto* dq_d = static_cast<uint16_t*>(dev((size_t) G * NE * FF * 2));
    auto* x16 = static_cast<uint16_t*>(dev((size_t) RMAX * NE * 2));
    auto* h16 = static_cast<uint16_t*>(dev((size_t) RMAX * FF * 2));
    auto* gu = static_cast<float*>(dev((size_t) RMAX * 2 * FF * 4));
    auto* dn = static_cast<float*>(dev((size_t) RMAX * NE * 4));
    auto* zx = static_cast<float*>(dev((size_t) RMAX * NE * 4));
    void* xq = dev(mmq::q8_bytes(RMAX, NE) * 2);
    void* hq = dev(mmq::q8_bytes(RMAX, FF) * 2);
    auto* bounds = static_cast<int32_t*>(dev((size_t) (G + 1) * 4));
    auto* ident = static_cast<int32_t*>(dev((size_t) RMAX * 4));
    auto* wtab = static_cast<const uint8_t**>(dev((size_t) G * sizeof(void*)));
    auto give_up = [&](const char* why) {
        std::fprintf(stderr, "strata: timing the experts' prompt products: %s (%s); %s\n", why, strata::gpu::last_error(),
                     fixed == 0 ? "the FP16 path" : "the int8 products");
        return fixed;
    };
    for (const void* p : {(const void*) w, (const void*) dq_gu, (const void*) dq_d, (const void*) x16, (const void*) h16,
                          (const void*) gu, (const void*) dn, (const void*) zx, (const void*) xq, (const void*) hq, (const void*) bounds,
                          (const void*) ident, (const void*) wtab})
        if (p == nullptr) return give_up("no device memory for it");
    try {
        void* s = nullptr;   // the compute queue
        const uint8_t* tab[G];
        for (int e = 0; e < G; ++e) tab[e] = w + (size_t) e * blob;
        if (!strata::gpu::memset_async(w, 0x11, (size_t) G * blob, s) || !strata::gpu::memset_async(zx, 0, (size_t) RMAX * NE * 4, s) ||
            !strata::gpu::memset_async(x16, 0, (size_t) RMAX * NE * 2, s) || !strata::gpu::memset_async(h16, 0, (size_t) RMAX * FF * 2, s) ||
            !strata::gpu::copy(static_cast<void*>(wtab), static_cast<const void*>(tab), sizeof tab))
            return give_up("its buffers could not be set");
        mmq::quantize(zx, nullptr, xq, f.gu_type, NE, NE, RMAX, s);
        mmq::quantize(zx, nullptr, hq, f.d_type, FF, FF, RMAX, s);
        mmq::iota(ident, RMAX, s);
        mmq::Context ctx;
        auto time_us = [&](const std::function<void()>& run) {
            run();
            strata::gpu::stream_sync(s);
            double best = 1e30;
            for (int t = 0; t < 3; ++t) {
                const auto t0 = Clock::now();
                for (int i = 0; i < 5; ++i) run();
                strata::gpu::stream_sync(s);
                best = std::min(best, std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / 5);
            }
            return best;
        };
        int64_t won = 0;
        std::string said;
        for (const int64_t r : RS) {
            if (r > r_top) break;
            std::vector<int32_t> hb(G + 1);
            for (int e = 0; e <= G; ++e) hb[(size_t) e] = (int32_t) (e * r);
            if (!strata::gpu::copy(bounds, hb.data(), hb.size() * 4)) return give_up("its row bounds could not be set");
            const int64_t R = G * r;
            const double us8 = time_us([&] {
                mmq::Product a;
                a.type = f.gu_type; a.w_rows = 2 * FF; a.w_cols = NE; a.n = G; a.xq = xq; a.bounds = bounds; a.ids = ident;
                a.total_rows = R; a.max_rows = r; a.rows = R; a.dst = gu; a.ld_dst = 2 * FF; a.wtab = wtab; a.woff = 0;
                ctx.run(a, s);
                mmq::Product b;
                b.type = f.d_type; b.w_rows = NE; b.w_cols = FF; b.n = G; b.xq = hq; b.bounds = bounds; b.ids = ident;
                b.total_rows = R; b.max_rows = r; b.rows = R; b.dst = dn; b.ld_dst = NE; b.wtab = wtab; b.woff = f.down_off;
                ctx.run(b, s);
            });
            const double us16 = time_us([&] {
                for (int e = 0; e < G; ++e)
                    k::iq_dequant_expert_f16(f.gu_type, tab[e], tab[e] + f.up_off, FF, NE, dq_gu + (size_t) e * 2 * FF * NE,
                                             f.d_type, tab[e] + f.down_off, dq_d + (size_t) e * NE * FF, s);
                if (!Gemm::f16_groups(x16, dq_gu, 2 * FF * NE, gu, hb.data(), G, 2 * FF, NE, s))
                    k::xmx_gemm_grouped(x16, dq_gu, 2 * FF * NE, gu, bounds, G, r, 2 * FF, NE, s);
                if (!Gemm::f16_groups(h16, dq_d, NE * FF, dn, hb.data(), G, NE, FF, s))
                    k::xmx_gemm_grouped(h16, dq_d, NE * FF, dn, bounds, G, r, NE, FF, s);
            });
            char buf[64];
            std::snprintf(buf, sizeof buf, "%s%lld: %.0f/%.0f", said.empty() ? "" : ", ", (long long) r, us8, us16);
            said += buf;
            // FP16 only when it is 3% faster: above the crossover the int8 group scratch no longer bounds the chunk, so
            // --prefill auto takes a smaller one.  The RTX 4070's 1024 rows are within 1% either way (int8 1972-1982 us,
            // FP16 1957-1976), and a run that took FP16 there read a 32K serve prompt in 8192-token chunks: 1,687 -> 1,162 tok/s.
            if (us16 * 1.03 < us8) {
                const int64_t below = won == 0 ? 0 : (int64_t) std::llround(std::sqrt((double) won * (double) r));
                std::fprintf(stderr, "strata: the experts' prompt products: int8 below %lld rows an expert, FP16 from "
                                     "there (us int8/FP16 at rows %s)\n", (long long) below, said.c_str());
                return below;
            }
            won = r;
        }
        std::fprintf(stderr, "strata: the experts' prompt products: int8 at every size (us int8/FP16 at rows %s)\n",
                     said.c_str());
        return INT64_MAX;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata: timing the experts' prompt products: %s; %s\n", e.what(),
                     fixed == 0 ? "the FP16 path" : "the int8 products");
        return fixed;
    }
}
const MmqPlan& mmq_plan() {
    // per GPU (the current device's): a layer split's stages can differ, and a stage that took the first GPU's plan
    // ran int8 products its own GPU lacks (B70 + RTX 4070: the 4070's prompt chunk stopped at its first layer)
    static core::PerDevice<MmqPlan> plans;
    return plans.get(core::Runtime::get().device(), [] {
        MmqPlan p;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const bool on = mmq_wanted() && lay.native;
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.fmt[(size_t) l].gu_type, dt = lay.fmt[(size_t) l].d_type;
            // and gate's rows right before up's (the products read them as one matrix)
            // APR: a Hadamard-folded layer takes the FP16 path, which rotates its rows (the int8 products do not)
            if (!mmq::supported(gt) || !mmq::supported(dt) || lay.fmt[(size_t) l].up_off != mmq::matrix_bytes(gt, 640, N) ||
                lay.fmt[(size_t) l].had_gu || lay.fmt[(size_t) l].had_d) {
                p.fallback = true;
                continue;
            }
            p.layer[(size_t) l] = 1;
            p.any = true;
            p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, N));
            p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, N, 640));
        }
        if (p.any && mmq_env() < 0) {   // by the rows a layer's experts average, against this GPU's timing
            int64_t first = 0;
            while (!p.layer[(size_t) first]) ++first;
            p.int8_below = int8_rows_below(lay.fmt[(size_t) first], (size_t) lay.blob_bytes(first));
            if (p.int8_below == 0) {   // FP16 at every size
                p = MmqPlan{};
                p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
            } else if (p.int8_below != INT64_MAX) {
                p.fallback = true;   // both kinds of layer: their scratch at one place (moe_set_bytes)
            }
        }
        return p;
    });
}
// The MoE half's scratch, carve's order.  A layer reads its experts' rows either as FP16 (Xs, Hh) or as int8 (Xq, H,
// Hq), on m.cs, the layer before it done with them: a model with both kinds of layer (the int8 products lack some of
// its types: UD-Q4_K_XL) keeps the larger set's room, not both: 0.42 and 0.40 GB at an 8192-token chunk.  Their sum
// (with the expert groups', expert_group_bytes) cost an RTX 4070 the 8192 chunk at a 32K UD-Q4_K_XL prompt: 6144-token
// chunks streamed 36% more experts, 615 -> 460 tok/s.
uint64_t moe_set_bytes(size_t T, int64_t n_expert) {
    const MmqPlan& mp = mmq_plan();
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    const size_t GR = mp.any && !mp.fallback ? (size_t) mmq_group_rows_cap(T) : T * K;   // as carve()
    a.take<float>(GR * 1280, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    Alloc q8 = a;
    if (mp.fallback) {
        if (!xs_in_dm()) a.take<uint16_t>(T * K * N, ok);
        a.take<uint16_t>(T * K * 640, ok);
    }
    if (mp.any) {
        q8.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), N), ok);
        q8.take<float>(GR * 640, ok);
        q8.take<uint8_t>(mmq::q8_bytes((int64_t) GR, 640), ok);
    }
    return std::max(a.used, q8.used);
}
// The experts' group a layer's products read: G experts dequantized to FP16 (dq_gu, dq_d; 157 MB at 16) where some
// layer keeps the FP16 path, MMQ_GROUP experts' int8 blocks (grp_gu, grp_d) where some layer takes the int8 one and the
// chunk streams every expert.  Both are filled and read on m.cs, one kind a layer, so a model with both kinds of layer
// keeps one place of the larger: their sum still kept an RTX 4070 from its 8192-token chunk on UD-Q4_K_XL (1,251 slots
// lent where its 85% limit is 1,231).  The small chunks' MMQ ring is not here: the copy stream fills it.
// The hyper-connection block's rows, carve's order (in the region with hc_in_region, else beside it).
uint64_t hc_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<uint16_t>(T * D, ok); a.take<float>(T * LR, ok); a.take<uint16_t>(T * LR, ok); a.take<float>(T * D, ok);
    if (bf16x2_hc()) { a.take<uint16_t>(T * D, ok); a.take<uint16_t>(T * LR, ok); }
    return a.used;
}
uint64_t expert_group_bytes(size_t T) {
    const MmqPlan& mp = mmq_plan();
    Alloc dq, q8;
    dq.count_only = q8.count_only = true;
    bool ok = true;
    if (mp.fallback) {
        dq.take<uint16_t>((size_t) (g_expert_group * GU_ELEMS), ok);
        dq.take<uint16_t>((size_t) (g_expert_group * D_ELEMS), ok);
    }
    if (mp.any && (int64_t) T >= STREAM_ALL_MIN) {
        q8.take<uint8_t>(MMQ_GROUP * mp.gu_max, ok);
        q8.take<uint8_t>(MMQ_GROUP * mp.d_max, ok);
    }
    return std::max(dq.used, q8.used);
}
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes) {
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    m.T = chunk; m.cs = stream; m.stats = &stats_;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    m.device = strata::core::current_device();
    if (stage_le_ < 0) stage_le_ = g.n_layers;
    if (stage_lb_ < 0 || stage_lb_ >= stage_le_ || stage_le_ > g.n_layers || (stage_le_ < g.n_layers) != (next_ != nullptr)) {
        err = "prefill: the stage's layer range is wrong";
        return false;
    }
    for (int b = 0; next_ != nullptr && b < 2; ++b)
        if (!m.hand[b] && !strata::gpu::alloc_host((void**) &m.hand[b], (size_t) chunk * D * 4)) {
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    if (next_ != nullptr && handoff_pipe_pieces() > 0 && m.hand_pieces == 0) {   // events only: no device memory
        bool ok = true;
        auto make = [](strata::gpu::Event*& e) { return e != nullptr || strata::gpu::event_create(&e); };
        for (int b = 0; b < 2; ++b) {   // (a retry after a failed init keeps the events it made)
            ok = ok && make(m.hand_done[b]);
            for (int i = 0; ok && i < handoff_pipe_pieces(); ++i) ok = make(m.hand_ev[b][i]);
        }
        if (!ok) { err = "prefill: the layer split's hand-off events (STRATA_HANDOFF_PIPE)"; return false; }
        m.hand_pieces = handoff_pipe_pieces();
    }
    if (m.tok_dev == nullptr) {
        if (!strata::gpu::alloc_device((void**) &m.tok_dev, (size_t) chunk * sizeof(int32_t))) {
            err = "prefill: the token id buffer";
            return false;
        }
        m.owned.push_back(m.tok_dev);
        m.tok_host.resize((size_t) chunk);
        m.tok_up.init((size_t) chunk);
    }
    m.copy = strata::gpu::stream_create_copy();
    if (!m.copy) { err = "prefill: copy stream"; return false; }
    const size_t T = (size_t) chunk;
    m.T_max = chunk;
    m.borrowed = borrow != nullptr;
    bool ok = true;
    // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
    for (int i = 0; i < RING_MAX; ++i) {
        if (!strata::gpu::event_create(&m.copied[i])) ok = false;
        if (!strata::gpu::event_create(&m.used[i])) ok = false;
    }
    for (int p = 0; p < 2; ++p)
        if (!m.mgrp_used[p] && !strata::gpu::event_create(&m.mgrp_used[p])) ok = false;
    if (!m.stager) {
        m.stager = std::make_unique<Stager>();
        const int hw = (int) std::thread::hardware_concurrency();
        const char* stv = std::getenv("STRATA_STAGER_THREADS");   // D-5: the host copy threads of unpinned blobs
        // The GGUF read in place (the low-RAM mode): most of a chunk's blobs are page faults on the SSD, so the copies
        // need many reads in flight (upstream 361537b: 32 threads and a 128-deep ring, measured on an RTX 5070 / NVMe
        // PC).  Here as many threads as the CPU has, up to 32, and a ring 4 x as deep; every other source keeps
        // the defaults.
        bool files = false;
        for (int64_t l = 0; src != nullptr && !files && l < g.n_layers; ++l)
            for (int64_t e = 0; !files && e < g.n_expert; ++e) files = src->transient(l, e);
        // #1353 (upstream 83a7dbc0): ... but a source answers transient() for every blob it copies out of the GGUF,
        // including the ones whose pages are already in the page cache (a Q8 pack on a big-RAM box with a warm cache).
        // Those copies are memcpy from RAM, where 32 threads and a 128-deep ring ran 4-7x slower than the defaults.  So
        // ask the OS: the SSD profile only when fewer than 90% of a sample of the blobs' pages are resident (-1: it
        // cannot tell, keep it).
        if (files && !stv) {
            const char* ev = std::getenv("STRATA_STAGER_SSD");   // 1 / 0: force the profile (A/B, support)
            if (ev != nullptr && ev[0] != '\0') files = ev[0] != '0';
            else if (const double warm = src->cached_share(256); warm >= 0.9) {
                files = false;
                std::fprintf(stderr, "prefill: the GGUF experts are %.0f%% in the page cache; the stager takes the RAM profile\n",
                             warm * 100.0);
            }
        }
        const int threads = stv ? std::clamp((int) std::strtol(stv, nullptr, 10), 1, 32)
                          : files ? std::clamp(hw, 4, 32) : std::max(2, std::min(4, hw / 4));
        if (files && std::getenv("STRATA_STAGER_RING") == nullptr) m.stager->kRing = 4 * threads;
        if (!m.stager->init((size_t) MAXBLOB(), threads)) ok = false;
    }
    m.steps_host.resize(T * strata::kernels::kStepCount);
    m.steps_up.init(T * strata::kernels::kStepCount);
    m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K); m.cnt.resize(m.g->n_expert); m.off.resize(m.g->n_expert + 1);
    for (int b = 0; b < 2; ++b) {
        if (!m.ple_emb_host[b] &&
            (force_pageable() || !strata::gpu::alloc_host((void**) &m.ple_emb_host[b], (size_t) T * N * 4))) {
            m.ple_pageable[b].resize(T * N);          // pageable: the upload is staged before it returns
            m.ple_emb_host[b] = m.ple_pageable[b].data();
        }
        if (!m.ple_copied[b] && !strata::gpu::event_create(&m.ple_copied[b]))
            ok = false;
        m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);
    }
    if (ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {   // KV streaming: the staging pool's identity page table
        const int64_t pages = ss.qsa_states[ss.qsa_primary()].n_pages;
        std::vector<int32_t> ident((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
        if (!strata::gpu::alloc_device((void**) &m.ident_table, ident.size() * 4) ||
            !strata::gpu::copy(m.ident_table, ident.data(), ident.size() * 4))
            ok = false;
        else
            m.owned.push_back(m.ident_table);
    }
    if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    {
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
        m.gemm.init_external(stream, gs, GEMM_SCRATCH);
    }
    if (!carve(T, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit" +
              fit_failure(o);
        return false;
    }
    return true;
}

// Every device buffer of a chunk of T tokens, from the Alloc `alloc` (after the GEMM scratch and workspace): `init`
// once, and `relayout` for a request's own chunk.  The order is `bytes_needed`'s.
bool Prefill::carve(size_t T, void* alloc) {
    Impl& m = *impl_;
    Alloc& o = *static_cast<Alloc*>(alloc);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    bool ok = true;
    // xn: the row scales only (F-1, gr_norm_rs: the FP32 normalized rows are not kept; a T x D buffer was 335 MB of
    // the borrowed cache slots at an 8192-token chunk)
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok); m.xn = o.take<float>(T * HC, ok);
    auto take_hc = [&](Alloc& h) {
        m.xn16 = h.take<uint16_t>(T * D, ok); m.lo = h.take<float>(T * LR, ok); m.lo16 = h.take<uint16_t>(T * LR, ok);
        m.gated = h.take<float>(T * D, ok);
        if (bf16x2_hc()) { m.xn16_lo = h.take<uint16_t>(T * D, ok); m.lo16_lo = h.take<uint16_t>(T * LR, ok); }
    };
    if (!hc_in_region()) take_hc(o);
    m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok);
    // the embedding rows are dead once gr_broadcast has read them on m.cs; the half outputs written into bo come after
    // it on the same in-order queue (the GEMMs' too)
    m.bo = m.emb;
    if (bf16x2()) m.mixed_bf_lo = o.take<uint16_t>(T * N, ok);
    m.steps_dev = o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    m.max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    {
        // one region for the attention half's and the MoE half's scratch (see gdn_set_bytes)
        const uint64_t region = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert),
                                          hc_in_region() ? hc_set_bytes(T) : (uint64_t) 0});
        uint8_t* base = o.take<uint8_t>((size_t) region, ok);
        m.region = base;
        m.region_bytes = region;
        if (hc_in_region()) {
            Alloc h;
            h.base = base; h.cap = region; h.owned = &m.owned;
            take_hc(h);
        }
        Alloc a;
        a.base = base; a.cap = region; a.owned = &m.owned;
        m.qkv = a.take<float>(T * C, ok); m.z = a.take<float>(T * ZV, ok); m.ab = a.take<float>(T * 2 * HV, ok);
        m.gate = a.take<float>(T * HV, ok); m.beta = a.take<float>(T * HV, ok); m.hbuf = a.take<float>(T * C, ok);
        m.y = a.take<float>(T * ZV, ok); m.y_h = a.take<uint16_t>(T * ZV, ok);
        Alloc b;
        b.base = base; b.cap = region; b.owned = &m.owned;
        m.Kc = b.take<float>(T * 512, ok); m.Vc = b.take<float>(T * 512, ok); m.Qf = b.take<float>(T * 12288, ok);
        m.q = b.take<float>(T * ZV, ok); m.idx_raw = b.take<float>(T * 128, ok); m.q_idx = b.take<float>(T * 512, ok);
        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * ZV, ok);
        m.sel_ids = b.take<int32_t>(T * (size_t) m.cap, ok);
        m.sel_scores = b.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
        m.attn_scratch = b.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
        Alloc c;
        c.base = base; c.cap = region; c.owned = &m.owned;
        m.logits = c.take<float>(T * m.g->n_expert, ok); m.w = c.take<float>(T * K, ok); m.ids = c.take<int32_t>(T * K, ok);
        m.slot_dev = c.take<int32_t>(T * K, ok); m.src_dev = c.take<int32_t>(T * K, ok);
        const MmqPlan& mp = mmq_plan();
        // GU, H and Hq: one group's rows where every layer takes the int8 products (mmq_group_rows_cap); the chunk's
        // where some layer's FP16 products write GU by the chunk's rows
        const size_t GR = mp.any && !mp.fallback ? (size_t) mmq_group_rows_cap(T) : T * K;
        m.GU = c.take<float>(GR * 1280, ok);
        m.Dm = c.take<float>(T * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        // the experts' rows: FP16 (Xs, Hh) or int8 (Xq, H, Hq), at the same place (see moe_set_bytes)
        Alloc q8 = c;
        // the upper half of Dm's T*K rows of N floats holds the T*K rows of N halves
        m.Xs = !mp.fallback ? nullptr : xs_in_dm() && m.Dm ? reinterpret_cast<uint16_t*>(m.Dm) + T * K * N
                                                            : c.take<uint16_t>(T * K * N, ok);
        m.Hh = mp.fallback ? c.take<uint16_t>(T * K * 640, ok) : nullptr;
        if (mp.any) {
            m.Xq = q8.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), N), ok);
            m.H = q8.take<float>(GR * 640, ok);
            m.Hq = q8.take<uint8_t>(mmq::q8_bytes((int64_t) GR, 640), ok);
        }
        c.used = std::max(c.used, q8.used);
        if (base == nullptr) ok = false;
    }
    // the experts' group: FP16 (dq_gu, dq_d) and int8 (grp_gu, grp_d) at one place (see expert_group_bytes)
    m.dq_gu = m.dq_d = nullptr;
    m.grp_gu = m.grp_d = m.mring = nullptr;
    if (const uint64_t group_bytes = expert_group_bytes(T)) {
        uint8_t* group = o.take<uint8_t>((size_t) group_bytes, ok);
        Alloc dq, q8;
        dq.base = q8.base = group;
        dq.cap = q8.cap = group_bytes;
        if (group == nullptr) ok = false;
        if (ok && mmq_plan().fallback) {
            m.dq_gu = dq.take<uint16_t>((size_t) (g_expert_group * GU_ELEMS), ok);
            m.dq_d = dq.take<uint16_t>((size_t) (g_expert_group * D_ELEMS), ok);
        }
        if (ok && mmq_plan().any && (int64_t) T >= STREAM_ALL_MIN) {
            m.grp_gu = q8.take<uint8_t>(MMQ_GROUP * mmq_plan().gu_max, ok);
            m.grp_d = q8.take<uint8_t>(MMQ_GROUP * mmq_plan().d_max, ok);
        }
    }
    m.xb_dev = o.take<int32_t>((size_t) m.g->n_expert + 1, ok);
    if (mmq_plan().any) {
        m.ids_identity = o.take<int32_t>(T * K, ok);
        m.ids_rel = o.take<int32_t>(T * K, ok);
        // the bounds: n + 1 absolute, then MMQ_GROUP + 1 relative a group; a group may hold one expert (the rows cap)
        m.bounds_dev = o.take<int32_t>((size_t) (m.g->n_expert + 1 + m.g->n_expert * (MMQ_GROUP + 1)), ok);
        m.tab_dev = o.take<int32_t>((size_t) (4 * m.g->n_expert), ok);
        if ((int64_t) T < STREAM_ALL_MIN) m.mring = o.take<uint8_t>(MMQ_RING * MMQ_SLOT(), ok);
        m.mgrp_live[0] = m.mgrp_live[1] = false;
        // (written at every run's start, not here: when serving, these are live expert-cache slots until a request
        // lends them - a write now would corrupt a resident expert)
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
    }
    m.ring = ring_slots(T);
    if (o.base == nullptr && m.ring > 0) {
        // OWNED buffers: the ring in ONE allocation (upstream 4a9b9041).  384 separate 2.7 MiB allocations took
        // 1,556 MiB on an RTX 4070 (each rounded up to its 2 MiB pages), one of the same total 1,038.  A borrowed
        // region keeps its per-slot layout (and so its price, `bytes_needed`: the loans' slot counts do not move).
        uint8_t* ring_base = o.take<uint8_t>((size_t) m.ring * MMQ_SLOT(), ok);   // slots 256-byte aligned, as alone
        for (int i = 0; ok && i < m.ring; ++i) {
            m.stage_dev[i] = ring_base + (size_t) i * MMQ_SLOT();
            m.stage_live[i] = false;                    // a new buffer: nothing of an earlier layout to wait for
            m.used_of[i] = i;
        }
    } else {
        for (int i = 0; i < m.ring; ++i) {
            m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
            m.stage_live[i] = false;                    // a new buffer: nothing of an earlier layout to wait for
            m.used_of[i] = i;
        }
    }
    m.ple_emb = o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) strata::kernels::NG_HC_DIM, ok);
    take_stage(o, ss, s, m.stage, ok);
    m.T = (int64_t) T;
    return ok;
}

bool Prefill::relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err) {
    Impl& m = *impl_;
    if (!m.borrowed || borrow == nullptr || chunk <= 0 || chunk > m.T_max) {
        err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
        return false;
    }
    if (!strata::gpu::stream_sync(m.cs) || !strata::gpu::stream_sync(m.copy)) {
        err = "prefill: relayout: the stream failed";
        return false;
    }
    bool ok = true;
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    if (ok) m.gemm.rebind(gs, GEMM_SCRATCH);
    if (!ok || !carve((size_t) chunk, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit" +
              fit_failure(o);
        return false;
    }
    return true;
}

int64_t Prefill::chunk() const { return impl_->T; }

bool Prefill::draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                       std::string& err, const char** why) {
    Impl& m = *impl_;
    auto decline = [why](const char* reason) {
        if (why != nullptr) *why = reason;
        return false;
    };
    static const bool off = [] {
        const char* v = std::getenv("STRATA_MTP_BATCH");
        return v != nullptr && v[0] == '0';
    }();
    // A ring (KV streaming: the drafter's window, page p in slot p % n_slots over a host copy) takes the same appends
    // with its own page table and host copy, as a streamed main layer does; the cells written are those the window can
    // still reach (r0 below), which the ring holds, so no two of them share a slot.  STRATA_MTP_BATCH_RING=0: the
    // drafter's own pass for a ring (the A/B).
    static const bool ring_ok = [] {
        const char* v = std::getenv("STRATA_MTP_BATCH_RING");
        return v == nullptr || v[0] != '0';
    }();
    core::QsaState& st = mtp.kv_state_rw();
    if (off) return decline("STRATA_MTP_BATCH=0");
    if (n <= 0 || m.g == nullptr || m.region == nullptr) return decline("no prompt scratch region on this path");
    if (st.kv_mode != 0 && !(st.kv_mode == 2 && ring_ok))
        return decline(st.kv_mode == 2 ? "the drafter's K/V is a ring and STRATA_MTP_BATCH_RING=0"
                                       : "the drafter's K/V mode is not paged or a ring");
    if (mtp.device() != m.device) return decline("the drafter is on another device than this prompt path");
    const core::OnDevice on_device(m.device);
    const auto t0 = Clock::now();
    const core::ModelGeometry& g = *m.g;
    const int64_t Nn = g.n_embd, HCN = g.hc * g.n_embd, KV = g.n_head_kv * g.head_dim;
    if (Nn != N || g.hc != HC || g.hc_lr != LR)   // the hyper-connection kernels' fixed shapes
        return decline("the geometry is not the hyper-connection kernels' fixed shapes");
    const float* w_ne = mtp.tensor_f32("pre_fc_norm_embedding.weight");
    const void* w_fe = mtp.tensor_q8("fc_embedding.weight");
    const float* w_nh = mtp.tensor_f32("pre_fc_norm_hidden.weight");
    const void* w_fh = mtp.tensor_q8("fc_hidden.weight");
    const float* w_hn = mtp.tensor_f32("attn_hyper_connection.hc_norm.weight");
    const uint16_t* w_dn = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_down.weight");
    const uint16_t* w_up = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_up.weight");
    const void* w_k = mtp.tensor_q8("self_attn.k_proj.weight");
    const void* w_v = mtp.tensor_q8("self_attn.v_proj.weight");
    const float* w_kn = mtp.tensor_f32("self_attn.k_norm.weight");
    if (!w_ne || !w_fe || !w_nh || !w_fh || !w_hn || !w_dn || !w_up || !w_k || !w_v || !w_kn)
        return decline("a drafter tensor is missing (Q8_0 projections or norms)");
    const core::NativeEmbed* nemb = core::native_embed();
    const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
    if (!nemb && (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
                  (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
        return decline("the token embedding is not gatherable on this path");
    // the cells the drafter's window can still reach
    const int64_t r0 = std::max<int64_t>(0, mtp.first_needed() - cell0);
    if (r0 >= n) return true;
    // per row: emb/e2 (N), en16 (N half), hn/h2/Rm/gated (HCN), hn16/xn16 (HCN half), lo (LR) + lo16, rs, mixed
    // (N) + mixed_h, K and V (KV each), the token id; up to 256 bytes of alignment for each of the 17 buffers
    const uint64_t per_row = 4 * (2 * Nn + 4 * HCN + LR + HC + Nn + 2 * KV + 1) + 2 * (Nn + 2 * HCN + LR + Nn);
    constexpr uint64_t kAlignBytes = (uint64_t) 17 * 256;
    if (m.region_bytes < kAlignBytes) return decline("too little scratch for a 64-row batch");
    int64_t B = std::min<int64_t>(n - r0, (int64_t) ((m.region_bytes - kAlignBytes) / per_row) & ~(int64_t) 63);
    if (st.kv_mode == 2)   // a ring: one batch's cells must not share a slot (a batch can straddle one page more)
        B = std::min<int64_t>(B, ((st.n_slots - 1) * strata::kernels::qsa_real_shapes().page_size) & ~(int64_t) 63);
    if (B < 64) return decline("too little scratch for a 64-row batch");
    uint8_t* q = m.region;
    auto carve = [&](size_t bytes) { void* p = q; q += (bytes + 255) & ~(size_t) 255; return p; };
    auto* emb = (float*) carve((size_t) B * Nn * 4);
    auto* e2 = (float*) carve((size_t) B * Nn * 4);
    auto* en16 = (uint16_t*) carve((size_t) B * Nn * 2);
    auto* hn = (float*) carve((size_t) B * HCN * 4);
    auto* h2 = (float*) carve((size_t) B * HCN * 4);
    auto* Rm = (float*) carve((size_t) B * HCN * 4);
    auto* gated = (float*) carve((size_t) B * HCN * 4);
    auto* hn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    auto* xn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    auto* lo = (float*) carve((size_t) B * LR * 4);
    auto* lo16 = (uint16_t*) carve((size_t) B * LR * 2);
    auto* rs = (float*) carve((size_t) B * HC * 4);
    auto* mixed = (float*) carve((size_t) B * Nn * 4);
    auto* mixed_h = (uint16_t*) carve((size_t) B * Nn * 2);
    auto* Kc = (float*) carve((size_t) B * KV * 4);
    auto* Vc = (float*) carve((size_t) B * KV * 4);
    auto* tok = (int32_t*) carve((size_t) B * 4);
    if ((uint64_t) (q - m.region) > m.region_bytes) return decline("the batch buffers do not fit the scratch region");
    static const bool timing = std::getenv("STRATA_DRAFT_TIMING") != nullptr;   // debug: where this pass's time goes
    if (timing) strata::gpu::stream_sync(m.cs);
    const auto ti0 = Clock::now();
    if (!mtp.idle(err)) return false;   // the drafter's own stream (its graph uploads) before this writes its K/V
    const double ms_idle = ms_since(ti0);
    const auto tl0 = Clock::now();
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    std::vector<int32_t> tk((size_t) B);
    m.draft_tok_up.init((size_t) B);   // tk is refilled every batch: each batch goes up from the pinned buffer
    for (int64_t b0 = r0; b0 < n; b0 += B) {
        const int64_t nb = std::min(B, n - b0), c0 = cell0 + b0;
        for (int64_t i = 0; i < nb; ++i) tk[(size_t) i] = next_tokens[b0 + i];
        if (!m.draft_tok_up.upload(tok, tk.data(), (size_t) nb, m.cs)) {
            err = "prefill: the draft tokens' upload failed";
            return false;
        }
        // the input branches: the next token's embedding, and this cell's final residual rows
        if (nemb) {
            nemb->gather_dev(tok, nb, emb, m.cs);
        } else {
            const auto* codes = (const uint8_t*) wemb->data;
            const auto* scales = (const float*) (codes + wemb->codes_bytes);
            const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                   : nullptr;
            strata::kernels::embedding_gather_dev(codes, scales, offsets, tok, (int) nb, wemb->ne0, wemb->code_bits,
                                                  wemb->code_bias, wemb->group_elems,
                                                  (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                  (uint64_t) (wemb->ne0 / wemb->group_elems), emb, m.cs);
        }
        rms_rows(emb, w_ne, nb, Nn, Nn, EPS, m.cs);
        to_f16(emb, en16, nb * Nn, m.cs);
        m.gemm.native(en16, mtp.tensor_q8_type("fc_embedding.weight"), w_fe, e2, nb, Nn, Nn);
        if (!strata::gpu::copy_async(hn, R_rows + (size_t) b0 * HCN, (size_t) nb * HCN * 4, m.cs)) {
            err = "prefill: the draft rows' copy failed";
            return false;
        }
        if (mtp.hnorm_per_stream())   // --mtp-hnorm stream: as the drafter's own pass (mtp.cpp)
            strata::kernels::native_qsa_rms_norm_grouped(hn, w_nh, hn, (int) Nn, (int) g.hc, (int) (nb * g.hc), EPS, m.cs);
        else
            rms_rows(hn, w_nh, nb, HCN, HCN, EPS, m.cs);
        to_f16(hn, hn16, nb * HCN, m.cs);
        m.gemm.native(hn16, mtp.tensor_q8_type("fc_hidden.weight"), w_fh, h2, nb * g.hc, Nn, Nn);   // every stream through fc_hidden
        strata::kernels::add_streams_broadcast(h2, e2, Rm, Nn, (int) g.hc, (int) nb, m.cs);
        // the attention hyper-connection's read (its mixed input only: this pass writes nothing back)
        gr_norm_rs(Rm, w_hn, EPS, rs, xn16, nb, m.cs);
        m.gemm.bf16(xn16, w_dn, lo, nb, LR, HCN);
        gr_silu(lo, lo16, nb, m.cs);
        m.gemm.bf16(lo16, w_up, gated, nb, HCN, LR);
        gr_mix_r(Rm, rs, w_hn, gated, mixed, nullptr, nb, m.cs, mixed_h);
        // K and V into the drafter's cache, as the prompt path's QSA layers append theirs
        m.gemm.native(mixed_h, mtp.tensor_q8_type("self_attn.k_proj.weight"), w_k, Kc, nb, KV, Nn);
        m.gemm.native(mixed_h, mtp.tensor_q8_type("self_attn.v_proj.weight"), w_v, Vc, nb, KV, Nn);
        rms_rows(Kc, w_kn, nb * g.n_head_kv, g.head_dim, g.head_dim, EPS, m.cs);
        rope(Kc, nb, g.n_head_kv, g.head_dim, KV, c0, strata::kernels::rope_scaling(), m.cs);
        if (st.kv_rot) {   // rotated as the drafter's own pass stores them (mtp.cpp)
            strata::kernels::fwht256_inplace_cuda(Kc, nb * g.n_head_kv, m.cs);
            strata::kernels::fwht256_inplace_cuda(Vc, nb * g.n_head_kv, m.cs);
        }
        if (st.kv_q4) {
            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, c0, nb, Kc, Vc, s, m.cs, &st.host);
        } else {
            kv_append(Kc, Vc, nb, c0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs, &st.host);
        }
    }
    if (!strata::gpu::stream_sync(m.cs)) {
        err = std::string("prefill: the draft layer's K/V: ") + strata::gpu::last_error();
        return false;
    }
    mtp.ms_prefill += ms_since(t0);
    if (timing)
        std::fprintf(stderr, "strata draft kv: %lld cells from %lld (first needed %lld), batch %lld: drafter idle %.1f ms, "
                     "the batches %.1f ms, all %.1f ms\n", (long long) n, (long long) cell0,
                     (long long) cell0 + (long long) r0, (long long) B, ms_idle, ms_since(tl0), ms_since(t0));
    return true;
}
void Prefill::set_ring_override(int slots) { g_ring_override = std::clamp(slots,0,RING_MAX); }
void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
void Prefill::set_group_cap(bool on) { g_group_cap = on; }
bool Prefill::group_cap() { return g_group_cap; }
void Prefill::set_expert_group(int g) { g_expert_group = std::clamp(g, 1, 64); }
int Prefill::expert_group() { return g_expert_group; }
double Prefill::pinned_share() { return g_pinned_share; }

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    // the same allocation sequence as `init`, counted
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    f(T * N); f(T * D); f(T * HC);
    if (!hc_in_region()) o.take<uint8_t>((size_t) hc_set_bytes(T), ok);
    f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok);
    if (!emb_reuse_account()) f(T * N);   // bo: aliases emb in carve; still counted unless STRATA_EMB_REUSE_ACCOUNT=1
    if (bf16x2()) o.take<uint16_t>(T * N, ok);
    o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    o.take<uint8_t>((size_t) std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                       moe_set_bytes(T, g.n_expert), hc_in_region() ? hc_set_bytes(T) : (uint64_t) 0}),
                    ok);
    if (const uint64_t group_bytes = expert_group_bytes(T)) o.take<uint8_t>((size_t) group_bytes, ok);
    o.take<int32_t>((size_t) g.n_expert + 1, ok);
    if (mmq_plan().any) {
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<int32_t>((size_t) (4 * g.n_expert), ok);
        if ((int64_t) T < STREAM_ALL_MIN) o.take<uint8_t>(MMQ_RING * MMQ_SLOT(), ok);
    }
    for (int i = 0; i < ring_slots(T); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    strata::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    return o.used + (8u << 20);   // alignment slack
}

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0, const uint16_t* X_lo = nullptr) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    if (X_lo) gm.bf16(X_lo, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy, true);
    return true;
}

}  // namespace

std::string experts_path() {
    const MmqPlan& p = mmq_plan();
    if (!p.any) return "FP16";
    std::string s = std::string("int8 (") + mmq::path() + ")";
    if (p.int8_below != INT64_MAX) s += " below " + std::to_string(p.int8_below) + " rows an expert, FP16 above";
    bool lacks = false;
    for (const char c : p.layer) lacks = lacks || !c;
    if (lacks) s += ", FP16 in the layers of types it lacks";
    return s;
}

namespace {
// STRATA_PREFILL_TIMING=1: the prompt path's GPU time by phase.  Events are recorded on the compute stream in order;
// the time between two consecutive marks is charged to the phase of the first, so a gap where the GPU waits (for the
// host's expert grouping, or for an expert's copy) lands on the phase that was waiting.  Events are reused: the marks
// are folded at every MoE layer's host sync, after which all of them have completed.  The folding itself (a query per
// mark, ~500 a layer at 32K tokens) is host time while the GPU is idle: it lands on 'host grouping' (2.7 s of a 32K
// prompt on a B70, where the grouping is 25 ms) and the host line gives it apart.  The wait for the first chunk's PLE
// rows (ple_land, read from the SSD beside layer 0) lands on layer 0's 'hc write'.
enum PfPhase { kPfStart, kPfHc, kPfGdn, kPfQsa, kPfQsaIdx, kPfQsaSel, kPfQsaAttn, kPfRouter, kPfHostGroup, kPfGather,
               kPfWaitCopy, kPfDequant, kPfGemmGU, kPfGemmD, kPfCombine, kPfPle, kPfKvStage, kPfGdnConv, kPfGdnRec, kPfGdnOut,
               kPfHcWrite, kPfCount };
const char* const kPfNames[kPfCount] = {"embed+steps", "hc read", "gdn", "qsa proj", "qsa indexer", "qsa select",
                                        "qsa attn", "router+shared", "host grouping", "gather", "wait copy", "dequant",
                                        "gemm gate/up", "gemm down", "combine", "ple", "kv stage", "gdn conv+gates",
                                        "gdn recurrence", "gdn out proj", "hc write"};
struct PfTimer {
    bool on = env_on("STRATA_PREFILL_TIMING");
    std::vector<strata::gpu::Event*> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPfCount] = {};
    void mark(int phase, strata::gpu::Stream s) {
        if (!on) return;
        if (used == ev.size()) {
            strata::gpu::Event* e = nullptr;
            strata::gpu::event_create(&e, true);
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        strata::gpu::event_record(ev[used], s);
        ++used;
    }
    // every recorded mark has completed (the stream was synchronized): charge the gaps, keep the last mark
    void fold() {
        if (!on || used < 2) return;
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            if (strata::gpu::event_elapsed_ms(&t, ev[i], ev[i + 1])) ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
    }
    ~PfTimer() {
        for (strata::gpu::Event* e : ev) strata::gpu::event_destroy(e);
    }
};
}  // namespace

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    err.clear();
    Impl& m = *impl_;
    const core::OnDevice on_device(m.device);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const auto t_start = Clock::now();
    const int64_t LB = stage_lb_, LE = stage_le_;
    // the next stage reads chunk c while this one reads chunk c + 1 (declared first: an early return waits for it
    // before anything it reads goes away)
    struct NextWait {
        PrefillLink* l;
        ~NextWait() {
            std::string e;
            if (l != nullptr) (void) l->wait(e);
        }
    } next_wait{next_};
    int hand_buf = 0;
    double host_sync_ms = 0, host_chunk_ms = 0, host_setup_ms = 0;   // STRATA_PREFILL_TIMING: the host's share
    double host_route_ms = 0, host_fold_ms = 0, host_group_ms = 0;   // ...at each MoE layer: the routing's sync, the
                                                                     // timer's own folding, the grouping
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    PfTimer pt;
    const strata::gpu::Stream cs = m.cs;
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites with experts:
    // write it again for every prompt (a layout is reused as long as the chunk and the slots are the same)
    if (m.ids_identity != nullptr) mmq::iota(m.ids_identity, m.T * K, m.cs);
    // The PLE rows of a chunk are read from the model file on the host (an SSD read per missed row): the chunk
    // after this one is read on a thread while the GPU runs this one, into the other of two buffers.  The rows
    // depend only on the tokens (the two before a position name its n-grams), so this is the same data.
    const bool ple_on = ss.ple.ready() && LB <= 1 && 1 < LE;
    // the PLE block batched over the chunk: the pinned postops and a BF16 or GGUF-native key (else token by token);
    // STRATA_PLE_BATCH=0 keeps the per-token block (the A/B)
    static const bool ple_batch_env = [] {
        const char* v = std::getenv("STRATA_PLE_BATCH");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool ple_batch = ple_on && ple_batch_env && strata::kernels::ple_native_postops_enabled() &&
                           (ss.ple.w.key_bf16 != nullptr || ss.ple.w.key_native_data != nullptr) &&
                           m.region_bytes / (((uint64_t) 3 * strata::kernels::NG_HC_DIM + (uint64_t) N + 4) * 4 + (uint64_t) N * 2 + 4096) >= 64;
    const int32_t prev0[2] = {prev[0], prev[1]};
    auto ple_gather = [&m, &ss, tokens, n, prev0](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = std::min(m.T, n - c0);
        auto at = [&](int64_t i) { return i < 2 ? prev0[i] : (int32_t) tokens[i - 2]; };   // prev0, then the tokens
        int32_t pv[2] = {at(c0), at(c0 + 1)};
        for (int64_t t = 0; t < T; ++t) {
            const int32_t tok = (int32_t) tokens[c0 + t];
            strata::kernels::ngram_rows(&tok, pv, 1, ss.ple.consts,
                                        m.ple_rows[buf].data() + t * strata::kernels::PLE_N_HEADS);
            pv[0] = pv[1];
            pv[1] = tok;
        }
        return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
    };
    std::string ple_next_err;
    std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
    int ple_buf = 0;

    for (int64_t c0 = 0; c0 < n; c0 += m.T) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        if (std::getenv("STRATA_TRACE")) {   // the stage and the time, one clock for the process (upstream 14d076ee2, 3a2675302)
            static const Clock::time_point t0 = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::fprintf(stderr, "strata trace: prompt chunk %lld of %lld stage [%lld,%lld) %.1f ms\n", (long long) c0,
                         (long long) n, (long long) stage_lb_, (long long) stage_le_, ms);
            std::fflush(stderr);
        }
        const int64_t T = std::min(m.T, n - c0), p0 = pos0 + c0;
        core::progress_at("reading the prompt (batched): preparing the chunk from token", p0);
        ++stats_.chunks;
        pt.mark(kPfStart, cs);
        const auto tsetup = Clock::now();
        // ---- embeddings, broadcast to the four streams - or, in a later stage of a layer split, the rows the
        // previous stage handed on
        if (hand_in_ != nullptr && hand_ev_ != nullptr) {
            // STRATA_HANDOFF_PIPE: each piece's upload is queued once its download on the stage before has landed.
            // The stage before split the chunk it handed on (n rows); one chunk here is all of it, otherwise (a
            // smaller chunk here) every piece is waited for before the one copy.
            const bool whole = c0 == 0 && T == n;
            bool ok = true;
            for (int i = 0; ok && i < hand_pieces_; ++i) {
                int64_t r0 = 0, nr = 0;
                handoff_piece(n, hand_pieces_, i, r0, nr);
                ok = strata::gpu::event_sync(hand_ev_[i]);
                if (ok && whole && nr > 0)
                    ok = strata::gpu::copy_async(m.R + (size_t) r0 * D, hand_in_ + (size_t) r0 * D, (size_t) nr * D * 4,
                                                 m.cs);
            }
            if (ok && !whole) ok = strata::gpu::copy_async(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, m.cs);
            if (!ok) {
                err = std::string("prefill: the layer split's hand-off upload failed (STRATA_HANDOFF_PIPE): ") +
                      strata::gpu::last_error();
                return false;
            }
        } else if (hand_in_ != nullptr) {
            if (!strata::gpu::copy_async(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, m.cs)) {
                err = "prefill: the layer split's hand-off upload failed";
                return false;
            }
        }
        // C-4: the whole chunk's rows in one gather (the same per-element arithmetic as the per-token path, so the
        // same bits); a chunk with picture rows, or a token outside the table, takes the per-token path
        bool batched = hand_in_ == nullptr && m.tok_dev != nullptr;
        const core::NativeEmbed* nemb = core::native_embed();
        const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
        if (batched && nemb == nullptr &&
            (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
             (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
            batched = false;
        for (int64_t t = 0; batched && t < T; ++t) {
            const int64_t tok = tokens[c0 + t];
            if ((embd_rows && embd_rows[p0 + t]) || tok < 0 || (wemb && tok >= wemb->ne1)) batched = false;
            else m.tok_host[(size_t) t] = (int32_t) tok;
        }
        if (batched) {
            if (!m.tok_up.upload(m.tok_dev, m.tok_host.data(), (size_t) T, m.cs)) {
                err = "prefill: the token id upload failed";
                return false;
            }
            if (nemb) {
                nemb->gather_dev(m.tok_dev, T, m.emb, m.cs);
            } else {
                const auto* codes = (const uint8_t*) wemb->data;
                const auto* scales = (const float*) (codes + wemb->codes_bytes);
                const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                       : nullptr;
                strata::kernels::embedding_gather_dev(codes, scales, offsets, m.tok_dev, (int) T, wemb->ne0,
                                                      wemb->code_bits, wemb->code_bias, wemb->group_elems,
                                                      (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                      (uint64_t) (wemb->ne0 / wemb->group_elems), m.emb, m.cs);
            }
        }
        // the picture rows go up from one pinned buffer (upstream #1996: from the caller's pageable memory a later
        // prompt on gfx1100 read stale rows); without one, from the caller's memory with the stream waited for
        // (packed: only the chunk's picture rows, so the buffer is a picture's size, not a chunk's)
        int64_t pic_rows = 0;
        for (int64_t t = 0; hand_in_ == nullptr && !batched && embd_rows && t < T; ++t)
            pic_rows += embd_rows[p0 + t] != nullptr;
        float* rows_stage = nullptr;
        if (pic_rows > 0) {
            m.emb_up.init((size_t) (pic_rows * N));
            rows_stage = reinterpret_cast<float*>(m.emb_up.begin((size_t) (pic_rows * N)));
        }
        const bool rows_pinned = rows_stage != nullptr, rows_pageable = pic_rows > 0 && !rows_pinned;
        int64_t k_row = 0;
        for (int64_t t = 0; hand_in_ == nullptr && !batched && t < T; ++t) {
            const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
            if (row) {
                const float* src = row;
                if (rows_pinned) {
                    std::memcpy(rows_stage + k_row * N, row, (size_t) N * 4);
                    src = rows_stage + k_row * N;
                    ++k_row;
                }
                if (!strata::gpu::copy_async(m.emb + t * N, src, (size_t) N * 4, m.cs)) {
                    err = "prefill: the image embedding upload failed";
                    return false;
                }
            } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.emb + t * N, m.cs, err)) {
                return false;
            }
        }
        if ((rows_pinned && !m.emb_up.end(m.cs)) || (rows_pageable && !strata::gpu::stream_sync(m.cs))) {
            err = "prefill: the image embedding upload failed";
            return false;
        }
        if (hand_in_ == nullptr) gr_broadcast(m.emb, m.R, T, m.cs);
        // ---- the PLE rows of the whole chunk, one batched SSD request on a thread (see ple_gather).  A later chunk's
        // were read ahead during the previous chunk; the first chunk's are read beside layer 0 - they are needed from
        // layer 1 on, and gathering them here first left the GPU idle for the whole read (upstream; RTX 4070, 997
        // tokens: 42 ms)
        if (ple_on && !ple_next.valid())
            ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c0, b = ple_buf] {
                strata::aux_cpus::pin_current_thread();
                return ple_gather(c0, b, ple_next_err);
            });
        bool ple_pending = ple_on;
        // the chunk's rows onto the device just before layer 1 reads them, and the next chunk's gather started
        auto ple_land = [&]() -> bool {
            if (!ple_pending) return true;
            ple_pending = false;
            const auto tp = Clock::now();
            if (!ple_next.get()) {
                err = ple_next_err;
                return false;
            }
            strata::gpu::copy_async(m.ple_emb, m.ple_emb_host[ple_buf], (size_t) T * N * 4, m.cs);
            strata::gpu::event_record(m.ple_copied[ple_buf], m.cs);
            // the pageable fallback (no pinned RAM): waited for, as a pageable source can be read out of order
            // (upstream #1996)
            if (!m.ple_pageable[ple_buf].empty()) strata::gpu::stream_sync(m.cs);
            if (c0 + m.T < n) {
                strata::gpu::event_sync(m.ple_copied[ple_buf ^ 1]);   // the other buffer's upload (a chunk ago) is done
                ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c1 = c0 + m.T, b = ple_buf ^ 1] {
                    strata::aux_cpus::pin_current_thread();
                    return ple_gather(c1, b, ple_next_err);
                });
            }
            ple_buf ^= 1;
            stats_.ms_ple += ms_since(tp);
            return true;
        };
        for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
        // ---- the QSA step records of every position in the chunk
        for (int64_t t = 0; t < T; ++t) strata::kernels::qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);
        if (!m.steps_up.upload(m.steps_dev, m.steps_host.data(), (size_t) T * strata::kernels::kStepCount, m.cs)) {
            err = "prefill: the QSA step records' upload failed";
            return false;
        }

        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < LB; ++l) (core::is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // step 3: this chunk's stream - every non-resident expert of every layer, layer by layer in id order (entry
        // k lands in ring slot k % ring); a copy is issued once the entry `ring` before it is consumed (its slot's
        // `used` event recorded), so the copy stream never waits on an event that is not queued yet
        const strata::kernels::cpu::ExpertLayout& lay0 = strata::kernels::cpu::expert_layout();
        const bool stream_all = m.ring > STAGE && T >= STREAM_ALL_MIN && m.src != nullptr;
        struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; };
        std::vector<StreamEntry> seq;
        std::vector<size_t> seq_start;
        size_t issued = 0, consumed = 0;
        if (stream_all) {
            seq_start.assign((size_t) g.n_layers + 1, 0);
            std::vector<Stager::Job> js;
            for (int64_t l = LB; l < LE; ++l) {
                seq_start[(size_t) l] = seq.size();
                for (int32_t e = 0; e < m.g->n_expert; ++e) {
                    if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                    const bool tr = m.src->transient(l, e);   // copied by the source in the stager (never pinned)
                    const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                    if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                    int job = -1;
                    if (tr || !m.src->pinned(l, e)) {
                        job = (int) js.size();
                        js.push_back({b, (size_t) lay0.blob_bytes(l), tr ? m.src : nullptr, (int32_t) l, e});
                    }
                    seq.push_back({(int32_t) l, e, b, job});
                }
            }
            for (int64_t l = LE; l <= g.n_layers; ++l) seq_start[(size_t) l] = seq.size();
            m.stager->start(std::move(js));
        }
        struct StagerDone {
            Stager* st;
            ~StagerDone() { if (st) st->finish(); }
        } chunk_stager_done{stream_all ? m.stager.get() : nullptr};
        auto issue_until = [&](size_t limit) {
            limit = std::min(limit, seq.size());
            while (issued < limit) {
                const StreamEntry& en = seq[issued];
                const int sl = (int) (issued % (size_t) m.ring);
                const auto th = Clock::now();
                const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                // the slot's last reader, the copy and its mark in one submission (copy_async_after)
                const strata::gpu::Event* after = m.stage_live[sl] ? m.used[m.used_of[sl]] : nullptr;
                if (en.job < 0) {
                    strata::gpu::copy_async_after(m.stage_dev[sl], en.blob, bytes, m.copy, after, m.copied[sl]);
                    ++stats_.experts_dma;
                } else {
                    const uint8_t* hb = m.stager->wait(en.job);
                    strata::gpu::copy_async_after(m.stage_dev[sl], hb, bytes, m.copy, after, m.copied[sl]);
                    m.stager->issued_one(en.job, m.copy);
                }
                m.stage_live[sl] = true;
                stats_.ms_experts_host += ms_since(th);
                ++stats_.experts_streamed;
                ++issued;
            }
        };
        // D-5: the stream is issued by its own host thread, so the thread launching the layers' kernels never waits
        // behind a host copy of an unpinned blob (that wait left the GPU idle: the 'wait copy' / 'dequant' time of the
        // i-quant prompts).  The same copies in the same order into the same slots, and a slot is refilled only once
        // the compute stream has recorded that it is done with it: the same results.  STRATA_PREFILL_ISSUER=0: inline.
        static const bool issuer_on = [] {
            const char* v = std::getenv("STRATA_PREFILL_ISSUER");
            return v == nullptr || std::strtol(v, nullptr, 10) != 0;
        }();
        std::atomic<size_t> a_issued{0}, a_consumed{0};
        std::atomic<bool> a_stop{false};
        double iss_ms = 0;
        int64_t iss_streamed = 0, iss_dma = 0;
        std::thread issuer;
        struct IssuerJoin {
            std::atomic<bool>* stop;
            std::thread* t;
            ~IssuerJoin() { if (t->joinable()) { stop->store(true); t->join(); } }
        } issuer_join{&a_stop, &issuer};
        const bool threaded_issue = stream_all && issuer_on;
        if (threaded_issue) {
            issuer = std::thread([&] {
                const core::OnDevice od(m.device);
                for (size_t idx = 0; idx < seq.size(); ++idx) {
                    while (idx >= a_consumed.load(std::memory_order_acquire) + (size_t) m.ring) {
                        if (a_stop.load(std::memory_order_acquire)) return;
                        std::this_thread::yield();
                    }
                    const StreamEntry& en = seq[idx];
                    const int sl = (int) (idx % (size_t) m.ring);
                    const auto th = Clock::now();
                    const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                    // the slot's last reader, the copy and its mark in one submission (copy_async_after)
                    const strata::gpu::Event* after = m.stage_live[sl] ? m.used[m.used_of[sl]] : nullptr;
                    if (en.job < 0) {
                        strata::gpu::copy_async_after(m.stage_dev[sl], en.blob, bytes, m.copy, after, m.copied[sl]);
                        ++iss_dma;
                    } else {
                        const uint8_t* hb = m.stager->wait(en.job);
                        strata::gpu::copy_async_after(m.stage_dev[sl], hb, bytes, m.copy, after, m.copied[sl]);
                        m.stager->issued_one(en.job, m.copy);
                    }
                    m.stage_live[sl] = true;
                    iss_ms += ms_since(th);
                    ++iss_streamed;
                    a_issued.store(idx + 1, std::memory_order_release);
                }
            });
        } else if (stream_all) {
            issue_until((size_t) m.ring);   // layer 0's first experts, behind the embedding and the PLE
        }
        // the consumer's side: entry k's copy is on the copy stream (the thread issued it), then k is given back
        auto wait_issued = [&](size_t k) {
            if (!threaded_issue) return;
            while (a_issued.load(std::memory_order_acquire) <= k) std::this_thread::yield();
        };
        auto give_back = [&](size_t upto) {
            if (threaded_issue) a_consumed.store(upto, std::memory_order_release);
            else issue_until(upto + (size_t) m.ring);
        };
        host_setup_ms += ms_since(tsetup);
        bool normed = false;   // F-2: the previous half's write already normed R for this half (rs, xn16)
        for (int64_t l = LB; l < LE; ++l) {
            core::progress_beat();   // the serve watchdog: a prompt chunk of 8192 tokens is still moving
            core::progress_at("reading the prompt (batched): layer", l, p0);   // a stall names layer and chunk
            const core::LayerView v(*m.wt, l);
            if (l == std::max<int64_t>(LB, 1) && !ple_land()) return false;   // the PLE rows, read from layer 1 on
            // ---- the PLE block at layer 1, token by token (its conv reads the previous tokens' rows)
            if (l == 1 && ple_on && ple_batch) {
                // the whole chunk at once, in sub-batches carved from the idle scratch region: the key and value
                // projections as GEMMs (a token at a time they re-read ~52 MB of BF16 key per token on the IQ
                // files), the rest with the per-token kernels' arithmetic (native_ple_postops_batch)
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                const strata::kernels::PleWeights& pw = ss.ple.w;
                constexpr int64_t HD = strata::kernels::NG_HC_DIM;
                const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 + (uint64_t) N * (bf16x2() ? 4 : 2) + 4096;
                const int64_t SB = std::min<int64_t>(T, (int64_t) (m.region_bytes / per_token));
                for (int64_t s0 = 0; s0 < T; s0 += SB) {
                    const int64_t nb = std::min(SB, T - s0);
                    uint8_t* q = m.region;
                    auto carve_f = [&](size_t n) { float* p = reinterpret_cast<float*>(q); q += (n * 4 + 255) & ~(size_t) 255; return p; };
                    float* key = carve_f((size_t) nb * HD);
                    float* qn = carve_f((size_t) nb * HD);
                    float* gated = carve_f((size_t) nb * HD);
                    float* val = carve_f((size_t) nb * N);
                    float* gate = carve_f((size_t) nb * 4);
                    uint16_t* e16 = (uint16_t*) carve_f((size_t) nb * N / 2);
                    uint16_t* e16_lo = bf16x2() ? (uint16_t*) carve_f((size_t) nb * N / 2) : nullptr;
                    const float* emb = m.ple_emb + s0 * N;
                    if (pw.key_bf16 != nullptr) {
                        to_bf16(emb, e16, nb * N, m.cs, e16_lo);
                        m.gemm.bf16(e16, pw.key_bf16, key, nb, HD, N);
                        if (e16_lo) m.gemm.bf16(e16_lo, pw.key_bf16, key, nb, HD, N, 0, true);
                    } else {
                        to_f16(emb, e16, nb * N, m.cs);
                        m.gemm.native(e16, pw.key_native_type, pw.key_native_data, key, nb, HD, N);
                        to_bf16(emb, e16, nb * N, m.cs, e16_lo);
                    }
                    m.gemm.bf16(e16, pw.value_bf16, val, nb, N, N);
                    if (e16_lo) m.gemm.bf16(e16_lo, pw.value_bf16, val, nb, N, N, 0, true);
                    try {
                        strata::kernels::native_ple_postops_batch(key, m.R + s0 * D, val, ss.ple.hist, pw, qn, gated,
                                                                  gate, (int) nb, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                }
                stats_.ms_ple += ms_since(tp);
            } else if (l == 1 && ple_on) {
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                for (int64_t t = 0; t < T; ++t) {
                    strata::kernels::PleOut po;
                    po.normalized = m.ple_norm;
                    po.result = m.R + t * D;
                    try {
                        strata::kernels::ple_block(m.ple_emb + t * N, m.R + t * D, ss.ple.hist, ss.ple.w, po,
                                                   ss.ple.scratch, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                    strata::kernels::ple_history_advance(ss.ple.hist, m.ple_norm, m.cs);
                }
                stats_.ms_ple += ms_since(tp);
            }
            for (int half = 0; half < 2; ++half) {
                // ---- the hyper-connection read of this half
                const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                  su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                      *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                if (!wn || !wd || !wu || !wi) return false;
                pt.mark(kPfHc, cs);
                // F-1: the row scales into the start of xn (the FP32 normalized rows are not written; gr_mix_r
                // recomputes them from R)
                if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs, m.xn16_lo);
                normed = false;
                // STRATA_HC_Q8 with the BF16 copies dropped: each from its Q8_0 copy, dequantized on m.cs
                core::WeightRef wdv = *wd, wuv = *wu;
                if (wd->data == nullptr) wdv.data = core::hc_bf16_view(wd, 0, m.cs);
                if (wu->data == nullptr) wuv.data = core::hc_bf16_view(wu, 1, m.cs);
                if (!bf16_proj(m.gemm, &wdv, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo)) return false;
                gr_silu(m.lo, m.lo16, T, m.cs, m.lo16_lo);
                if (!bf16_proj(m.gemm, &wuv, m.lo16, m.gated, T, su, err, 0, m.lo16_lo)) return false;
                if (!bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo)) return false;
                gr_mix_r(m.R, m.xn, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h,
                         m.mixed_bf_lo);

                if (half == 0 && !core::is_qsa_layer(g, l)) {
                    // ======================= GDN =======================
                    const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                          *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                          *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                          *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                          *wsa = need(v, "ssm_a", err);
                    if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                    pt.mark(kPfGdn, cs);
                    float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                    pt.mark(kPfGdnConv, cs);   // "gdn" is the projections in; the rest on their own lines
                    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
                    pt.mark(kPfGdnRec, cs);
                    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs);
                    pt.mark(kPfGdnOut, cs);
                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                    ++gdn_index;
                } else if (half == 0) {
                    // ======================= QSA =======================
                    const core::QsaState& st = ss.qsa_states[qsa_index];
                    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                          *wik = need(v, "indexer.k_proj.weight", err),
                                          *wiq = need(v, "indexer.q_proj.weight", err),
                                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                          *wiqn = need(v, "indexer.q_norm.weight", err),
                                          *wikn = need(v, "indexer.k_norm.weight", err);
                    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                    pt.mark(kPfQsa, cs);
                    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                    if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
                    rope(m.Kc, T, 2, 256, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
                    // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
                    const bool staged = st.kv_mode == 1;
                    if (staged) {
                        pt.mark(kPfKvStage, cs);
                        strata::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host,
                                                            core::qsa_kv_format(st),
                                                            (p0 + s.page_size - 1) / s.page_size, s, m.cs);
                        pt.mark(kPfQsa, cs);
                    }
                    if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
                        // streamed: each half writes its part of the host copy and of the staging pool
                        strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        const strata::kernels::KvHostPools hk = strata::kernels::kv_hybrid_k_half(st.host),
                                                           hv = strata::kernels::kv_hybrid_v_half(st.host),
                                                           sk = strata::kernels::kv_hybrid_k_half(m.stage),
                                                           sv = strata::kernels::kv_hybrid_v_half(m.stage);
                        const bool mirror = st.host.present();
                        kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                                  st.k_q, st.k_q, st.k_scale, st.k_scale, m.cs, mirror ? &hk : nullptr,
                                  staged ? &sk : nullptr);
                        strata::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, m.cs,
                                                      mirror ? &hv : nullptr, staged ? &sv : nullptr);
                    } else {
                        if (st.kv_rot) {   // rotated K and V (kv_q4.hpp), the queries below too, the output back
                            strata::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
                            strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        }
                        if (st.kv_q4)
                            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                                          &st.host, staged ? &m.stage : nullptr);
                        else
                            kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                                      &st.host, staged ? &m.stage : nullptr);
                    }
                    split_q(m.Qf, m.q, T, m.cs);
                    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
                    rope(m.q, T, 24, 256, 6144, p0, strata::kernels::rope_scaling(), m.cs);
                    if (st.kv_rot) strata::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
                    rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
                    rope(m.q_idx, T, 4, 128, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // the indexer appends, token by token; then scores + selection for many queries at once:
                    // a query reads completed blocks (final once completed) and `dead` for its own tail block
                    const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    pt.mark(kPfQsaIdx, cs);
                    // C-2: the chunk's appends in three launches instead of one per token (the same end state:
                    // the queries below read it only after the whole chunk is appended). STRATA_INDEXER_PER_TOKEN=1: the old
                    try {
                        static const bool per_token = std::getenv("STRATA_INDEXER_PER_TOKEN") != nullptr;
                        if (!per_token) {
                            strata::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                                             EPS, ib, s, st.max_cells,
                                                                             strata::kernels::rope_scaling(), m.cs);
                        }
                        for (int64_t t = 0; per_token && t < T; ++t) {
                            const int32_t* step_t = m.steps_dev + t * strata::kernels::kStepCount;
                            strata::kernels::native_qsa_indexer_append(m.idx_raw + t * 128, step_t + strata::kernels::kStepPos, 0,
                                                                       (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                                       strata::kernels::rope_scaling(), m.cs);
                        }
                    } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                    pt.mark(kPfQsaSel, cs);
                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                        const int64_t nb = std::min(m.sel_batch, T - t0);
                        const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                        // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                strata::kernels::kStepNBid)] + 1;
                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise); STRATA_SELECT_OLD=1: the warp kernel
                        static const bool old_sel = std::getenv("STRATA_SELECT_OLD") != nullptr;
                        if (old_sel || !strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
                                                                             steps0, nb, m.max_blocks, s, m.sel_scores,
                                                                             m.cs, active))
                            strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs, active);
                        strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                        m.sel_ids + t0 * m.cap, m.cs);
                    }
                    // STRATA_SEL_OVERLAP (debug, D-1's question): how much do neighbouring queries' selections share?
                    // Per tile of 16 queries: the union of their selected cells against the sum of their widths.
                    if (static const bool ovl = std::getenv("STRATA_SEL_OVERLAP") != nullptr; ovl && qsa_index == 0) {
                        std::vector<int32_t> ids((size_t) (T * m.cap));
                        strata::gpu::copy_async(ids.data(), m.sel_ids, ids.size() * 4, m.cs);
                        strata::gpu::stream_sync(m.cs);
                        double sum_w = 0, sum_u = 0;
                        for (int64_t t0 = 0; t0 + 16 <= T; t0 += 16) {
                            std::vector<int32_t> u;
                            for (int64_t t = t0; t < t0 + 16; ++t) {
                                const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                                sum_w += (double) w;
                                u.insert(u.end(), ids.begin() + t * m.cap, ids.begin() + t * m.cap + w);
                            }
                            std::sort(u.begin(), u.end());
                            sum_u += (double) (std::unique(u.begin(), u.end()) - u.begin());
                        }
                        std::fprintf(stderr, "strata prefill: selection overlap at %lld: 16-query tiles read %.1f%% of the "
                                             "cells one query at a time does\n", (long long) p0, sum_w > 0 ? 100.0 * sum_u / sum_w : 0.0);
                    }
                    // STRATA_IDX_FP16_CHECK: would FP16 pooled indexer keys select the same cells? (the KV-streaming
                    // design's last question). Every query is selected again from the pooled keys and `dead` rounded
                    // to fp16 (exactly what an fp16 store reads back); the agreement with the fp32 selection is
                    // printed cumulatively after each chunk's last QSA layer. Debug: syncs per layer.
                    if (static const bool f16chk = std::getenv("STRATA_IDX_FP16_CHECK") != nullptr; f16chk) {
                        static float *pooled16 = nullptr, *dead16 = nullptr;
                        static int32_t* ids16 = nullptr;
                        static double shared = 0, cells = 0;
                        static long long queries = 0, same = 0, sel_queries = 0;
                        const int64_t rows = st.idx_pooled_rows;
                        if (pooled16 == nullptr &&
                            (!strata::gpu::alloc_device((void**) &pooled16, (size_t) rows * s.idx_dim * 4) ||
                             !strata::gpu::alloc_device((void**) &dead16, (size_t) s.idx_dim * 4) ||
                             !strata::gpu::alloc_device((void**) &ids16, (size_t) (m.T * m.cap) * 4))) {
                            err = "STRATA_IDX_FP16_CHECK: no room for its buffers";
                            return false;
                        }
                        round_f16(st.idx_pooled, pooled16, rows * s.idx_dim, m.cs);
                        round_f16(st.idx_dead, dead16, s.idx_dim, m.cs);
                        for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                            const int64_t nb = std::min(m.sel_batch, T - t0);
                            const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                            strata::kernels::qsa_block_scores(pooled16, dead16, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs);
                            strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                            ids16 + t0 * m.cap, m.cs);
                        }
                        std::vector<int32_t> a((size_t) (T * m.cap)), b((size_t) (T * m.cap));
                        strata::gpu::copy_async(a.data(), m.sel_ids, a.size() * 4, m.cs);
                        strata::gpu::copy_async(b.data(), ids16, b.size() * 4, m.cs);
                        strata::gpu::stream_sync(m.cs);
                        for (int64_t t = 0; t < T; ++t) {
                            const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                            const int32_t *x = a.data() + t * m.cap, *y = b.data() + t * m.cap;
                            int64_t i = 0, j = 0, c = 0;
                            while (i < w && j < w) {
                                if (x[i] == y[j]) { ++c; ++i; ++j; } else if (x[i] < y[j]) ++i; else ++j;
                            }
                            ++queries;
                            same += c == w;
                            if (p0 + t + 1 > m.cap) { ++sel_queries; shared += (double) c; cells += (double) w; }
                        }
                        if (qsa_index + 1 == g.n_qsa_layers())
                            std::fprintf(stderr, "strata prefill: FP16 indexer keys: %lld of %lld selections identical; "
                                                 "where the selection is sparse, %.4f%% of cells shared (%lld queries)\n",
                                         same, queries, cells > 0 ? 100.0 * shared / cells : 100.0, sel_queries);
                    }
                    // STRATA_QSA_DUMP=<file>: append every QSA layer's selected cells for the prompt's last
                    // STRATA_QSA_DUMP_LAST (4096) positions - records of int32 {qsa layer, pos0, T, cap} + T*cap cells,
                    // for tools/qsa_locality.py (how local the sparse attention's reads are: the KV-streaming question)
                    if (static const char* dump = std::getenv("STRATA_QSA_DUMP"); dump != nullptr) {
                        static const long long last = std::getenv("STRATA_QSA_DUMP_LAST")
                                                          ? std::strtoll(std::getenv("STRATA_QSA_DUMP_LAST"), nullptr, 10) : 4096;
                        if (p0 + T > pos0 + n - last) {
                            std::vector<int32_t> h((size_t) (T * m.cap));
                            strata::gpu::copy_async(h.data(), m.sel_ids, h.size() * 4, m.cs);
                            strata::gpu::stream_sync(m.cs);
                            if (std::FILE* f = std::fopen(dump, "ab")) {
                                const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};
                                std::fwrite(hdr, 4, 4, f);
                                std::fwrite(h.data(), 4, h.size(), f);
                                std::fclose(f);
                            }
                        }
                    }
                    const strata::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                                       : core::qsa_attn_pools(st);
                    pt.mark(kPfQsaAttn, cs);
                    // perf-review D-1: the whole chunk on tensor cores, one block per (query, KV head), FP32-level
                    // accuracy but not bitwise (qsa_prompt_attn.hpp). The pools it has no kernel for on this device,
                    // or STRATA_PROMPT_ATTN_OLD=1: the decode kernel, 32 queries at a time
                    static const bool old_attn = std::getenv("STRATA_PROMPT_ATTN_OLD") != nullptr;
                    if (old_attn || !strata::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s,
                                                                            m.attn, T, m.cs))
                        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                            const int64_t nb = std::min(m.attn_batch, T - t0);
                            strata::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                                   m.steps_dev + t0 * strata::kernels::kStepCount, m.cap,
                                                                   s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
                        }
                    if (st.kv_rot || st.kv_hybrid) strata::kernels::fwht256_inplace_cuda(m.attn, T * 24, m.cs);
                    pt.mark(kPfQsa, cs);
                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs);
                    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
                    ++qsa_index;
                } else {
                    // ======================= MoE =======================
                    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                                          *wsu = need(v, "ffn_up_shexp.weight", err),
                                          *wsd = need(v, "ffn_down_shexp.weight", err);
                    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                    pt.mark(kPfRouter, cs);
                    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err, 0, m.mixed_bf_lo)) return false;
                    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
                    // the shared expert and its scalar gate
                    if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                    if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                    swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                    if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                    if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                    m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                    if (m.mixed_bf_lo) m.gemm.bf16(m.mixed_bf_lo, (const uint16_t*) wgi->data, m.sg, T, 1, N, 0, true);
                    // group the (token, k) pairs by expert on the host
                    pt.mark(kPfHostGroup, cs);
                    const auto t_route = Clock::now();
                    m.ids_host.resize((size_t) T * K);
                    m.ids_host.from_device(m.ids, m.cs);
                    const strata::kernels::cpu::ExpertLayout& lay_c = strata::kernels::cpu::expert_layout();
                    bool cpu_maybe = cpu_pool_ != nullptr && cpu_share_on() && !stream_all && m.src != nullptr && !m.cpu_dead &&
                                     lay_c.native && !lay_c.fmt.empty();
                    // The share's two pinned buffers are small (~50 MB) but are had mid-request, when the host may have
                    // nothing left (upstream 04dbe561, 0.1.41 report: tight RAM, chats failing).  A buffer that cannot be
                    // had ends the share for this engine - the GPU takes every expert, as with STRATA_PREFILL_CPU_SHARE=0
                    // - instead of failing the request.
                    auto cpu_share_off = [&](const char* what) {
                        m.cpu_dead = true;
                        cpu_maybe = false;
                        std::fprintf(stderr, "prefill: the CPU share is off for this run: the host could not give %s "
                                             "(RAM is short); the GPU streams every expert\n", what);
                    };
                    bool cpu_arm = true;     // auto: this layer may share (false: the arm without)
                    int cpu_set = -1;        // auto: the wall events' set, -1 when not measured
                    bool cpu_cold = false;   // auto: one-time costs in the window (no reading)
                    // auto shares only while the layers that share cost less per non-resident expert than their
                    // neighbours that do not.  #1282 (upstream, RX 7900 GRE + 5700X3D): every share lost, auto's 0.47 by
                    // 4.5%, the host's issue time per streamed expert rising 107 -> 200 us with the CPU busy - a cost
                    // g / (c + g) cannot see, as both sides are measured with the share on.  The first eligible layers
                    // alternate, without first, until CPU_MIN_RATIOS ratios are in; then the cheaper arm, and every
                    // CPU_PROBE-th layer the other one (a prime: the probes move across the layers from one chunk to the
                    // next), so both stay measured.  Opt-in (STRATA_PREFILL_CPU_GATE=1): the arms without the share cost
                    // more than the gate saved on XeStrata's GPUs, 1K IQ3_XXS prompts 844 -> 833 tok/s on a B70 and
                    // 296 -> 283 on an RTX 4070 (2026-10-11, four interleaved rounds).
                    static const bool cpu_gate_on = [] { const char* e = std::getenv("STRATA_PREFILL_CPU_GATE"); return e != nullptr && e[0] == '1'; }();
                    if (cpu_maybe && cpu_share_env() < 0.0 && cpu_gate_on) {
                        constexpr int64_t CPU_PROBE = 29;
                        constexpr int CPU_MIN_RATIOS = 3;
                        const int64_t i = m.cpu_layers++;
                        cpu_arm = m.cpu_nratio < CPU_MIN_RATIOS ? (i & 1) != 0 : (i % CPU_PROBE == 0) != m.cpu_gate;
                        if (m.cpu_first_l < 0) m.cpu_first_l = l;
                        cpu_cold = l == m.cpu_first_l;
                        if (m.cpu_wall[0] == nullptr)
                            for (strata::gpu::Event*& e : m.cpu_wall)
                                if (!strata::gpu::event_create(&e, true)) {
                                    err = "prefill: CPU share events";
                                    return false;
                                }
                        cpu_set = (int) (i & 1);
                        strata::gpu::event_record(m.cpu_wall[(size_t) 2 * cpu_set], m.cs);
                    }
                    if (cpu_maybe && cpu_arm) {   // the MoE input, for the CPU's experts
                        const size_t want = (size_t) T * N;
                        if (m.cpu_x_n < want) {
                            cpu_cold = true;
                            if (m.cpu_x) strata::gpu::free(m.cpu_x);
                            m.cpu_x = nullptr;
                            m.cpu_x_n = 0;
                            if (!strata::gpu::alloc_host(&m.cpu_x, want * sizeof(float))) {
                                m.cpu_x = nullptr;
                                cpu_share_off("the activation buffer");
                            } else {
                                m.cpu_x_n = want;
                            }
                        }
                    }
                    strata::gpu::stream_sync(m.cs);
                    host_route_ms += ms_since(t_route);
                    const auto t_fold = Clock::now();
                    pt.fold();
                    host_fold_ms += ms_since(t_fold);
                    const auto t_group = Clock::now();
                    if (m.cpu_pend) {   // auto: the last CPU-sharing layer's GPU time is final now
                        m.cpu_pend = false;
                        float g_ms = 0;
                        if (strata::gpu::event_elapsed_ms(&g_ms, m.cpu_ev[0], m.cpu_ev[1])) {
                            constexpr double a = 0.25;
                            const double c = m.pend_cpu_ms / (double) m.pend_n_cpu, gm = g_ms / (double) m.pend_n_gpu;
                            m.cpu_c_ms = m.cpu_c_ms > 0 ? m.cpu_c_ms + a * (c - m.cpu_c_ms) : c;
                            m.cpu_g_ms = m.cpu_g_ms > 0 ? m.cpu_g_ms + a * (gm - m.cpu_g_ms) : gm;
                            m.cpu_share_now = std::clamp(m.cpu_g_ms / (m.cpu_c_ms + m.cpu_g_ms), 0.05, 0.9);
                        }
                    }
                    if (cpu_maybe) {   // the rows buffer, once, at its largest (an expert takes at most MAXT tokens' rows)
                        const size_t cap = (size_t) std::min<int64_t>(T * K, (int64_t) strata::kernels::cpu::MAXT * m.g->n_expert) * N;
                        if (m.cpu_rows_n < cap) {
                            cpu_cold = true;
                            if (m.cpu_rows) strata::gpu::free(m.cpu_rows);
                            m.cpu_rows = nullptr;
                            m.cpu_rows_n = 0;
                            if (!strata::gpu::alloc_host(&m.cpu_rows, cap * sizeof(float))) {
                                m.cpu_rows = nullptr;
                                cpu_share_off("the rows buffer");
                            } else {
                                m.cpu_rows_n = cap;
                            }
                        }
                    }
                    if (m.pend_wall) {   // the last eligible layer's wall, and its ratio to the one before it
                        float w_ms = 0;
                        if (strata::gpu::event_elapsed_ms(&w_ms, m.cpu_wall[(size_t) 2 * m.pend_set], m.cpu_wall[(size_t) 2 * m.pend_set + 1])) {
                            const double w = w_ms / (double) m.pend_wall_n;
                            if (m.last_arm != 0 && m.last_arm != m.pend_wall && m.pend_l > m.last_l) {   // same chunk
                                std::rotate(m.cpu_ratio, m.cpu_ratio + 1, m.cpu_ratio + Impl::CPU_RATIOS);
                                m.cpu_ratio[Impl::CPU_RATIOS - 1] = m.pend_wall == 2 ? w / m.last_w : m.last_w / w;
                                m.cpu_nratio = std::min(m.cpu_nratio + 1, Impl::CPU_RATIOS);
                                double v[Impl::CPU_RATIOS];
                                std::copy(m.cpu_ratio + Impl::CPU_RATIOS - m.cpu_nratio, m.cpu_ratio + Impl::CPU_RATIOS, v);
                                std::nth_element(v, v + m.cpu_nratio / 2, v + m.cpu_nratio);   // even: the upper one
                                m.cpu_gate = v[m.cpu_nratio / 2] < 1.0;
                            }
                            // debug: STRATA_DBG_CPU_GATE=1 prints each reading and what the next layers do
                            if (static const bool dbg = std::getenv("STRATA_DBG_CPU_GATE") != nullptr; dbg)
                                std::fprintf(stderr, "cpu gate: layer %d %s, %lld non-resident, %.2f ms (%.4f a expert) -> %s\n",
                                             m.pend_l, m.pend_wall == 2 ? "shared" : "not shared", (long long) m.pend_wall_n,
                                             w_ms, w, m.cpu_gate ? "share" : "do not share");
                            m.last_arm = m.pend_wall;
                            m.last_l = m.pend_l;
                            m.last_w = w;
                        }
                        m.pend_wall = 0;
                    }
                    std::fill(m.cnt.begin(), m.cnt.end(), 0);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = m.ids_host[(size_t) i];
                        if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
                        ++m.cnt[(size_t) e];
                    }
                    if (stats_.routed.empty()) stats_.routed.assign((size_t) (m.g->n_layers * m.g->n_expert), 0);
                    stats_.routed_k = K;
                    for (int32_t e = 0; e < m.g->n_expert; ++e)
                        stats_.routed[(size_t) (l * m.g->n_expert + e)] += m.cnt[(size_t) e];
                    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
                    int64_t routed = 0;   // the experts this layer routes to: a chunk's layer takes MMQ by their mean rows
                    for (int32_t e = 0; e < m.g->n_expert; ++e) routed += m.cnt[(size_t) e] > 0;
                    const int64_t below = mmq_plan().int8_below;
                    const bool use_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l] &&
                                         (below == INT64_MAX || T * K < below * std::max<int64_t>(routed, 1));
                    // set_helper: the experts the helper GPU computes this layer, the ones it holds that this GPU
                    // would stream and at most max_rows tokens route to (their rows last, after the CPU's)
                    std::vector<char> on_help;
                    int64_t rows_help = 0, n_help = 0;
                    if (helper_ && !stream_all && lay_c.native) {
                        HelperShare& hs = *helper_;
                        for (int32_t e = 0; e < m.g->n_expert; ++e) {
                            const int32_t c = m.cnt[(size_t) e];
                            if (c == 0 || c > hs.max_rows || rows_help + c > hs.cap ||
                                (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) ||
                                !hs.h->holds(l, e))
                                continue;
                            if (on_help.empty()) on_help.assign((size_t) m.g->n_expert, 0);
                            on_help[(size_t) e] = 1;
                            rows_help += c;
                            ++n_help;
                        }
                    }
                    auto helped = [&](int32_t e) { return !on_help.empty() && on_help[(size_t) e]; };
                    // set_cpu_pool: the experts the CPU computes this layer (their rows after the GPU's, before the
                    // helper's: [rows_gpu, rows_gpu + rows_cpu))
                    std::vector<char> on_cpu;
                    std::vector<const uint8_t*> cpu_blob;   // their blobs, taken here (the CPU thread calls no ExpertSource)
                    int64_t rows_cpu = 0, n_cpu = 0, n_stream = 0;
                    if (cpu_maybe) {
                        // the candidates (core/expert_tier, as the other models' prompt path takes them)
                        const strata::core::CpuShareCandidates cc = strata::core::cpu_share_candidates(
                            m.cnt.data(), m.g->n_expert, l, *m.src,
                            [&](int32_t e) {
                                return m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                            },
                            helped);
                        const std::vector<std::pair<int32_t, int32_t>>& cand = cc.cand;   // (tokens, expert)
                        n_stream = cc.missed;
                        const double share = cpu_share_env() >= 0.0 ? cpu_share_env() : cpu_arm ? m.cpu_share_now : 0.0;
                        const size_t take = std::min(cand.size(), (size_t) std::llround(share * (double) n_stream));
                        // the MoE input for the CPU's experts, only when it takes some (a layer whose experts all have
                        // more rows than the CPU takes, MAXT, copied the whole input for nothing: 323 MB at 32K tokens)
                        if (take > 0 && (!strata::gpu::copy_async(m.cpu_x, m.mixed, (size_t) T * N * sizeof(float), m.cs) ||
                                         !strata::gpu::stream_sync(m.cs))) {
                            err = std::string("prefill: the CPU share's input: ") + strata::gpu::last_error();
                            return false;
                        }
                        // an expert without a blob stays on the GPU
                        for (size_t i = 0; i < cand.size() && n_cpu < (int64_t) take; ++i) {
                            const int32_t e = cand[i].second;
                            const uint8_t* b = m.src->blob(l, e);
                            if (b == nullptr) continue;
                            if (on_cpu.empty()) {
                                on_cpu.assign((size_t) m.g->n_expert, 0);
                                cpu_blob.assign((size_t) m.g->n_expert, nullptr);
                            }
                            on_cpu[(size_t) e] = 1;
                            cpu_blob[(size_t) e] = b;
                            rows_cpu += cand[i].first;
                            ++n_cpu;
                        }
                    }
                    auto gpu_side = [&](int32_t e) { return (on_cpu.empty() || !on_cpu[(size_t) e]) && !helped(e); };
                    // The experts, in id order: resident ones from VRAM, the others through the staging ring.  F8:
                    // first, out of that order, the resident experts of at most kIqGemmTileRows rows, whose products
                    // iq_gemm_grouped_f16 takes straight from their GGUF blocks in the cache (dequantized in local
                    // memory to the same FP16 values: the same bits as dequantizing and multiplying, without writing
                    // 9.4 MiB an expert; more rows than one work-group's would decode the weights again per group).
                    // The rows are laid out in this order, so the others' groups stay contiguous.
                    // STRATA_PREFILL_FUSED=0: every expert dequantized.
                    static const bool fused_on = [] {
                        const char* v = std::getenv("STRATA_PREFILL_FUSED");
                        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                    }();
                    const strata::kernels::cpu::NativeFmt* ffmt = lay.native ? &lay.fmt[(size_t) l] : nullptr;
                    const bool fuse_res = fused_on && !use_mmq && ffmt != nullptr && m.host_res && m.cache &&
                                          strata::kernels::xmx_available(strata::kernels::XmxType::f16) &&
                                          ffmt->up_off == (size_t) ffmt->n_ff * ffmt->gu_row &&
                                          strata::kernels::iq_gemm_grouped_ok(ffmt->gu_type, 2 * ffmt->n_ff, ffmt->n_embd) &&
                                          strata::kernels::iq_gemm_grouped_ok(ffmt->d_type, ffmt->n_embd, ffmt->n_ff);
                    auto fusable = [&](int32_t e) {
                        return fuse_res && m.cnt[(size_t) e] <= strata::kernels::kIqGemmTileRows &&
                               m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                    };
                    std::vector<int32_t> order;
                    for (int32_t e = 0; e < m.g->n_expert; ++e)
                        if (m.cnt[(size_t) e] > 0 && gpu_side(e) && fusable(e)) order.push_back(e);
                    const size_t nf = order.size();
                    for (int32_t e = 0; e < m.g->n_expert; ++e)
                        if (m.cnt[(size_t) e] > 0 && gpu_side(e) && !fusable(e)) order.push_back(e);
                    // upstream 62fc7c8a (STRATA_MMQ_GROUP_SORT): an MMQ group's launch covers its largest expert's
                    // rows (max_rows), so a layer whose GPU experts are all resident takes them in falling row count:
                    // the groups' padding shrinks.  XeStrata's int8 products give each row the same bits whatever its
                    // group-mates (no stream-k split), so the order changes no answer.  The streamed walk reads its
                    // entries in id order: only a layer with nothing streamed.  STRATA_MMQ_GROUP_SORT=0: id order.
                    static const bool group_sort = [] {
                        const char* v = std::getenv("STRATA_MMQ_GROUP_SORT");
                        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                    }();
                    if (group_sort && use_mmq && nf == 0 && m.host_res && m.cache && order.size() > (size_t) MMQ_GROUP &&
                        std::all_of(order.begin(), order.end(), [&](int32_t e) {
                            return m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                        }))
                        std::stable_sort(order.begin(), order.end(),
                                         [&](int32_t a, int32_t b) { return m.cnt[(size_t) a] > m.cnt[(size_t) b]; });
                    {
                        int32_t at = 0;
                        for (const int32_t e : order) { m.off[(size_t) e] = at; at += m.cnt[(size_t) e]; }
                        for (int32_t e = 0; e < m.g->n_expert && !on_cpu.empty(); ++e)   // the CPU's block
                            if (on_cpu[(size_t) e]) { m.off[(size_t) e] = at; at += m.cnt[(size_t) e]; }
                        for (int32_t e = 0; e < m.g->n_expert && !on_help.empty(); ++e)   // the helper's, last
                            if (on_help[(size_t) e]) { m.off[(size_t) e] = at; at += m.cnt[(size_t) e]; }
                        m.off[(size_t) m.g->n_expert] = at;
                    }
                    const int64_t rows_gpu = T * K - rows_cpu - rows_help;
                    std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = m.ids_host[(size_t) i];
                        const int32_t p = fill[(size_t) e]++;
                        m.slot_host[(size_t) i] = p;
                        m.src_host[(size_t) p] = (int32_t) (i / K);
                    }
                    m.slot_host.to_device(m.slot_dev, m.cs);
                    m.src_host.to_device(m.src_dev, m.cs);
                    // set_helper: its rows' tokens gathered here and copied to the shared host rows (the host waits
                    // for them: the helper's stream cannot wait on this GPU's), then its products, queued on its own
                    // stream; the results come back at the combine
                    if (rows_help > 0) {
                        HelperShare& hs = *helper_;
                        const auto& hf = lay.fmt[(size_t) l];
                        const int64_t r0 = rows_gpu + rows_cpu;
                        if (hs.tok_of.size() < (size_t) T) hs.tok_of.resize((size_t) T);
                        hs.toks.clear();
                        int32_t* src = hs.h_meta;
                        int32_t* bounds = hs.h_meta + hs.cap;
                        int64_t ne = 0, maxr = 0;
                        for (int32_t e = 0; e < m.g->n_expert; ++e) {
                            if (!helped(e)) continue;
                            bounds[ne] = m.off[(size_t) e] - (int32_t) r0;
                            hs.h_wtab[ne++] = hs.h->blob_on_device(l, e);
                            maxr = std::max<int64_t>(maxr, m.cnt[(size_t) e]);
                        }
                        bounds[ne] = (int32_t) rows_help;
                        for (int64_t p = 0; p < rows_help; ++p) {
                            const int32_t t = m.src_host[(size_t) (r0 + p)];
                            if (hs.tok_of[(size_t) t] < 0) {
                                hs.tok_of[(size_t) t] = (int32_t) hs.toks.size();
                                hs.toks.push_back(t);
                            }
                            src[p] = hs.tok_of[(size_t) t];
                        }
                        for (const int32_t t : hs.toks) hs.tok_of[(size_t) t] = -1;
                        const int64_t U = (int64_t) hs.toks.size();
                        if (!strata::gpu::copy_async(hs.tok_dev, hs.toks.data(), (size_t) U * sizeof(int32_t), m.cs)) {
                            err = std::string("prefill: the helper's tokens: ") + strata::gpu::last_error();
                            return false;
                        }
                        gather_rows(m.mixed, hs.tok_dev, hs.xg, U, N, m.cs);
                        if (!strata::gpu::copy_async(hs.x_ex, hs.xg, (size_t) U * N * sizeof(float), m.cs) ||
                            !strata::gpu::stream_sync(m.cs)) {
                            err = std::string("prefill: the helper's rows: ") + strata::gpu::last_error();
                            return false;
                        }
                        try {
                            const core::OnDevice on(hs.dev);
                            const strata::gpu::Stream q = hs.s;
                            int32_t* d_src = hs.meta;
                            int32_t* d_bounds = hs.meta + hs.cap;
                            const int32_t* d_ident = hs.meta + 2 * hs.cap + 1;
                            const bool up_mixed = strata::kernels::cpu::native_mixed_gu(hf);
                            const int up_type = up_mixed ? hf.up_type : hf.gu_type;
                            const bool up_xq2 = strata::kernels::iq_mmq_type_dup(up_type) != strata::kernels::iq_mmq_type_dup(hf.gu_type);
                            if (!strata::gpu::copy_async(hs.x, hs.x_ex, (size_t) U * N * sizeof(float), q) ||
                                !strata::gpu::copy_async(d_src, src, (size_t) rows_help * sizeof(int32_t), q) ||
                                !strata::gpu::copy_async(d_bounds, bounds, (size_t) (ne + 1) * sizeof(int32_t), q) ||
                                (hs.live == false && !strata::gpu::copy_async(const_cast<int32_t*>(d_ident), hs.h_meta + 2 * hs.cap + 1,
                                                                              (size_t) hs.cap * sizeof(int32_t), q)) ||
                                !strata::gpu::copy_async(static_cast<void*>(hs.wtab), static_cast<const void*>(hs.h_wtab),
                                                         (size_t) ne * sizeof(void*), q)) {
                                err = std::string("prefill: the helper's tables: ") + strata::gpu::last_error();
                                return false;
                            }
                            mmq::quantize(hs.x, d_src, hs.xq, hf.gu_type, N, N, rows_help, q);
                            if (up_mixed && up_xq2) mmq::quantize(hs.x, d_src, hs.xq2, up_type, N, N, rows_help, q);
                            const int64_t ff = hf.n_ff;
                            mmq::Product gu;
                            gu.type = hf.gu_type; gu.w_rows = 2 * ff; gu.w_cols = N; gu.n = (int) ne; gu.xq = hs.xq;
                            gu.bounds = d_bounds; gu.ids = d_ident; gu.total_rows = rows_help; gu.max_rows = maxr;
                            gu.rows = rows_help; gu.dst = hs.gu; gu.ld_dst = 2 * ff; gu.wtab = hs.wtab; gu.woff = 0;
                            if (up_mixed) {   // the gate rows, then the up rows in their own type
                                gu.w_rows = ff;
                                hs.ctx->run(gu, q);
                                gu.type = up_type; gu.xq = up_xq2 ? hs.xq2 : hs.xq; gu.dst = hs.gu + ff; gu.woff = hf.up_off;
                            }
                            hs.ctx->run(gu, q);
                            mmq::swiglu(hs.gu, hs.hh, rows_help, ff, /*interleaved=*/false, q);
                            mmq::quantize(hs.hh, nullptr, hs.hq, hf.d_type, ff, ff, rows_help, q);
                            mmq::Product dn;
                            dn.type = hf.d_type; dn.w_rows = N; dn.w_cols = ff; dn.n = (int) ne; dn.xq = hs.hq;
                            dn.bounds = d_bounds; dn.ids = d_ident; dn.total_rows = rows_help; dn.max_rows = maxr;
                            dn.rows = rows_help; dn.dst = hs.d; dn.ld_dst = N; dn.wtab = hs.wtab; dn.woff = hf.down_off;
                            hs.ctx->run(dn, q);
                            if (!strata::gpu::copy_async(hs.d_ex, hs.d, (size_t) rows_help * N * sizeof(float), q)) {
                                err = std::string("prefill: the helper's results: ") + strata::gpu::last_error();
                                return false;
                            }
                        } catch (const std::exception& e) {
                            err = std::string("prefill: the helper GPU: ") + e.what();
                            return false;
                        }
                        hs.live = true;
                        hs.rows = rows_help;
                    }
                    // set_cpu_pool: the CPU's experts on a thread while this one streams and runs the GPU's; their
                    // unweighted rows land in Dm's row order, so the combine weights them as any other row.  The
                    // blobs are taken here: the thread calls no ExpertSource (upstream 667f2eca)
                    std::future<bool> cpu_fut;
                    double cpu_ms = 0;
                    if (rows_cpu > 0) {
                        // (the rows buffer was had at its largest after the routing sync)
                        const strata::kernels::cpu::NativeFmt& cf = lay.fmt[(size_t) l];
                        m.cpu_jobs.clear();
                        for (int32_t e = 0; e < m.g->n_expert; ++e) {
                            if (!on_cpu[(size_t) e]) continue;
                            strata::kernels::cpu::ExpertJobMulti j;
                            j.blob = cpu_blob[(size_t) e];
                            j.nt = m.cnt[(size_t) e];
                            for (int r = 0; r < j.nt; ++r)   // the activations are set on the thread
                                j.out[r] = m.cpu_rows + (size_t) ((int64_t) m.off[(size_t) e] + r - rows_gpu) * N;
                            m.cpu_jobs.push_back(j);
                        }
                        if (cpu_share_env() < 0.0) {
                            if (m.cpu_ev[0] == nullptr &&
                                (!strata::gpu::event_create(&m.cpu_ev[0], true) || !strata::gpu::event_create(&m.cpu_ev[1], true))) {
                                err = "prefill: CPU share events";
                                return false;
                            }
                            strata::gpu::event_record(m.cpu_ev[0], m.cs);
                        }
                        cpu_fut = std::async(std::launch::async, [&m, &cf, &cpu_ms, T, rows_gpu, rows_cpu, pool = cpu_pool_]() -> bool {
                            const auto t0 = std::chrono::steady_clock::now();
                            if (m.cpu_nact.size() < (size_t) T) m.cpu_nact.resize((size_t) T);
                            std::vector<char> need((size_t) T, 0);   // the tokens the CPU's experts read
                            for (int64_t p = rows_gpu; p < rows_gpu + rows_cpu; ++p) need[(size_t) m.src_host[(size_t) p]] = 1;
                            for (int64_t t = 0; t < T; ++t) {
                                if (!need[(size_t) t]) continue;
                                strata::kernels::cpu::native_quant_act(cf, m.cpu_x + (size_t) t * N, m.cpu_nact[(size_t) t].bytes);
                            }
                            for (auto& j : m.cpu_jobs)
                                for (int r = 0; r < j.nt; ++r) {
                                    const int64_t p = rows_gpu + (int64_t) (j.out[r] - m.cpu_rows) / N;
                                    const int64_t tok = m.src_host[(size_t) p];
                                    j.nact[r] = m.cpu_nact[(size_t) tok].bytes;
                                }
                            pool->run_split_multi_native(cf, m.cpu_jobs.data(), (int) m.cpu_jobs.size());
                            cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                            return true;
                        });
                    }
                    struct CpuJoin {   // an early return waits for the thread (it reads this layer's tables)
                        std::future<bool>& f;
                        ~CpuJoin() { if (f.valid()) f.wait(); }
                    } cpu_join{cpu_fut};
                    const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
                    const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
                    // APR (prism.hadamard): folded gate/up take H (s * x), folded down H (s * h), on the FP16 path
                    // (mmq_plan gives such a layer no int8 products); the CPU's rows rotate in native_quant_act
                    const bool had_gu = lay.native && lay.fmt[(size_t) l].had_gu;
                    const bool had_d = lay.native && lay.fmt[(size_t) l].had_d;
                    const int had_block = lay.native ? lay.fmt[(size_t) l].had_block : 0;
                    const float* had_sx = had_gu ? strata::kernels::hadamard_device_signs(lay.fmt[(size_t) l].gu_signs,
                                                                                          (int) lay.fmt[(size_t) l].n_embd)
                                                 : nullptr;
                    // the sign vectors at the layout's widths (upstream dd32704d4), this device's copies made once
                    const float* had_sh = had_d ? strata::kernels::hadamard_device_signs(lay.fmt[(size_t) l].d_signs,
                                                                                         (int) lay.fmt[(size_t) l].n_ff)
                                                : nullptr;
                    const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
                    const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
                    host_group_ms += ms_since(t_group);
                    pt.mark(kPfGather, cs);
                    if (use_mmq) {
                        // step 2b: the layer's activations as q8_1 rows in expert order, straight from `mixed`
                        mmq::quantize_scatter(m.mixed, m.slot_dev, m.src_dev, m.Xq, mmq_gt, N, N, T, (int) K, m.cs);
                        // each group's rows: absolute bounds (gate/up reads the layer's rows), relative ones (down
                        // reads the group's own quantized H)
                        // the groups: up to MMQ_GROUP experts in `order`, ended early when the rows would pass the
                        // group scratch's cap (mmq_group_rows_cap)
                        const size_t n = order.size();
                        const int64_t cap = mmq_group_rows_cap(T);
                        m.grp_start.clear();
                        m.grp_of.assign(n, 0);
                        m.grp_pos.assign(n, 0);
                        for (size_t j = 0, rows = 0; j < n; ++j) {
                            const int64_t c = m.cnt[(size_t) order[j]];
                            if (m.grp_start.empty() || j - (size_t) m.grp_start.back() >= (size_t) MMQ_GROUP ||
                                (int64_t) rows + c > cap) {
                                m.grp_start.push_back((int32_t) j);
                                rows = 0;
                            }
                            m.grp_of[j] = (int32_t) (m.grp_start.size() - 1);
                            m.grp_pos[j] = (int32_t) (j - (size_t) m.grp_start.back());
                            rows += (size_t) c;
                        }
                        m.grp_start.push_back((int32_t) n);
                        const size_t ng = m.grp_start.size() - 1;
                        m.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                        for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
                        m.bounds_host[n] = (int32_t) rows_gpu;
                        m.ids_rel_host.resize((size_t) (T * K));
                        for (size_t g = 0; g < ng; ++g) {
                            const size_t j0 = (size_t) m.grp_start[g], j1 = (size_t) m.grp_start[g + 1];
                            for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                m.bounds_host[n + 1 + g * (MMQ_GROUP + 1) + i] =
                                    m.bounds_host[std::min(j1, j0 + i)] - m.bounds_host[j0];
                            // the gate/up product writes a group's rows from 0 (GU holds one group): row r -> r - first
                            for (int32_t r = m.bounds_host[j0]; r < m.bounds_host[j1]; ++r)
                                m.ids_rel_host[(size_t) r] = r - m.bounds_host[j0];
                        }
                        m.bounds_host.to_device(m.bounds_dev, m.cs);
                        m.ids_rel_host.to_device(m.ids_rel, m.cs);
                    } else {
                        gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                        // APR: Hadamard-folded gate/up take H (s * x); Xs is this layer's own gathered copy (rotated in
                        // FP16 storage, in float arithmetic)
                        if (had_gu) strata::kernels::hadamard_rows_f16(m.Xs, T * K, N, had_block, had_sx, m.cs);
                        // the grouped products' row bounds, `order`'s first rows and the total
                        m.xb_host.resize(order.size() + 1);
                        for (size_t j = 0; j < order.size(); ++j) m.xb_host[j] = m.off[(size_t) order[j]];
                        m.xb_host[order.size()] = (int32_t) rows_gpu;
                        m.xb_host.to_device(m.xb_dev, m.cs);
                        // F8: the fusable resident experts, kIqGemmGroup a launch
                        std::vector<const void*> fgu, fd;
                        for (size_t j0 = 0; j0 < nf; j0 += (size_t) strata::kernels::kIqGemmGroup) {
                            const size_t j1 = std::min(nf, j0 + (size_t) strata::kernels::kIqGemmGroup);
                            fgu.clear();
                            fd.clear();
                            int64_t maxr = 0;
                            for (size_t j = j0; j < j1; ++j) {
                                const int32_t e = order[j];
                                const uint8_t* b = m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]);
                                fgu.push_back(b);
                                fd.push_back(b + ffmt->down_off);
                                maxr = std::max<int64_t>(maxr, m.cnt[(size_t) e]);
                            }
                            const int G = (int) (j1 - j0);
                            const int64_t r0 = m.xb_host[j0], nr = m.xb_host[j1] - r0;
                            pt.mark(kPfGemmGU, cs);
                            strata::kernels::iq_gemm_grouped_f16(ffmt->gu_type, m.Xs, fgu.data(), ffmt->n_ff, m.GU, 1280,
                                                                 m.xb_dev + j0, G, maxr, 1280, N, m.cs);
                            swiglu_interleaved(m.GU + r0 * 1280, m.Hh + r0 * 640, nr, m.cs);
                            if (had_d) strata::kernels::hadamard_rows_f16(m.Hh + r0 * 640, nr, 640, had_block, had_sh, m.cs);
                            pt.mark(kPfGemmD, cs);
                            strata::kernels::iq_gemm_grouped_f16(ffmt->d_type, m.Hh, fd.data(), 0, m.Dm, N,
                                                                 m.xb_dev + j0, G, maxr, N, 640, m.cs);
                            stats_.experts_resident += G;
                        }
                    }
                    // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                    int stage_next = 0;
                    std::vector<int> stage_of(order.size(), -1);
                    // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
                    std::vector<int> job_of(order.size(), -1);
                    if (!stream_all) {
                        std::vector<Stager::Job> js;
                        for (size_t j = 0; j < order.size(); ++j) {
                            const int32_t e = order[j];
                            if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                            const bool tr = m.src->transient(l, e);   // copied by the source (never pinned)
                            if (!tr && m.src->pinned(l, e)) continue;
                            const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                            if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                            job_of[j] = (int) js.size();
                            js.push_back({b, (size_t) lay.blob_bytes(l), tr ? m.src : nullptr, (int32_t) l, e});
                        }
                        m.stager->start(std::move(js));
                    }
                    StagerDone stager_done{stream_all ? nullptr : m.stager.get()};
                    // MMQ in place (MMQ_RING): expert j's blob in ring slot j % MMQ_RING, its group's slots released
                    // together; else the STAGE ring, a slot at a time
                    const bool mmq_direct = use_mmq && m.mring != nullptr;
                    auto slot_ptr = [&](int sl) { return mmq_direct ? m.mring + (size_t) sl * MMQ_SLOT() : m.stage_dev[sl]; };
                    // mmq_tab_on: the products' pointer tables, a resident expert's entries in its cache slot (`direct`,
                    // when 16-byte aligned) and every other one's where it is copied (its ring slot, or its group buffer)
                    const bool mtab = use_mmq && mmq_tab_on() && m.tab_dev != nullptr && m.host_res && m.cache;
                    std::vector<char> direct(mtab ? order.size() : 0, 0);
                    if (mtab) {
                        const auto& f = lay.fmt[(size_t) l];
                        const size_t n = order.size();
                        m.tab_host.resize(4 * n);
                        auto put = [&](size_t i, const uint8_t* p) {
                            std::memcpy(m.tab_host.data() + 2 * i, static_cast<const void*>(&p), sizeof p);
                        };
                        for (size_t j = 0; j < n; ++j) {
                            const int32_t s = m.host_res[(size_t) l * m.g->n_expert + order[j]];
                            const uint8_t* gu = s >= 0 ? m.cache->device_slot(s) : nullptr;
                            const uint8_t* dn = nullptr;
                            direct[j] = (char) (gu != nullptr && ((uintptr_t) gu | f.down_off) % 16 == 0);
                            if (direct[j]) {
                                dn = gu + f.down_off;
                            } else if (mmq_direct) {
                                gu = slot_ptr((int) (j % MMQ_RING));
                                dn = gu + f.down_off;
                            } else {
                                gu = m.grp_gu + (size_t) m.grp_pos[j] * mmq_gub;
                                dn = m.grp_d + (size_t) m.grp_pos[j] * mmq_db;
                            }
                            put(j, gu);
                            put(n + j, dn);
                        }
                        m.tab_host.to_device(m.tab_dev, m.cs);
                    }
                    auto stage_one = [&](size_t j) -> bool {
                        const int32_t e = order[j];
                        const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                        if (resident) return true;
                        int sl = (int) (j % MMQ_RING);
                        if (!mmq_direct) {
                            sl = stage_next;
                            stage_next = (stage_next + 1) % STAGE;
                        }
                        const int par = (int) (j / MMQ_GROUP % 2);
                        auto wait_slot = [&] {
                            if (mmq_direct) {
                                if (m.mgrp_live[par]) strata::gpu::stream_wait_event(m.copy, m.mgrp_used[par]);
                            } else if (m.stage_live[sl]) {
                                strata::gpu::stream_wait_event(m.copy, m.used[m.used_of[sl]]);
                            }
                        };
                        const auto th = Clock::now();
                        const bool tr = m.src->transient(l, e);
                        const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                        if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                        if (!tr && m.src->pinned(l, e)) {
                            // DMA straight from the page-locked arena: the copy stream only waits for the slot
                            wait_slot();
                            strata::gpu::copy_async(slot_ptr(sl), b, (size_t) lay.blob_bytes(l), m.copy);
                            ++stats_.experts_dma;
                        } else {
                            // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                            const uint8_t* hb = m.stager->wait(job_of[j]);
                            if (m.stager->failed.exchange(false)) { err = "prefill: the expert source could not copy a blob"; return false; }
                            wait_slot();
                            strata::gpu::copy_async(slot_ptr(sl), hb, (size_t) lay.blob_bytes(l), m.copy);
                            m.stager->issued_one(job_of[j], m.copy);
                        }
                        strata::gpu::event_record(m.copied[sl], m.copy);
                        if (!mmq_direct) m.stage_live[sl] = true;
                        stage_of[j] = sl;
                        stats_.ms_experts_host += ms_since(th);
                        ++stats_.experts_streamed;
                        return true;
                    };
                    // The streamed walk gathers an MMQ group in one launch, after one wait on its last streamed copy
                    // (the copy queue is in order), and releases its ring slots with one event (upstream b3096357,
                    // e8de49eb): a wait, a gather and a mark an expert were three submissions each, and the RTX 4070's
                    // copy engine stood idle behind them.  Its slots go back to the issuer only then; a skipped ring
                    // entry first flushes the open group (`flush`), so at most a group's entries are held back.
                    // STRATA_PREFILL_GROUP_GATHER=0: one wait, gather and mark an expert.
                    static const bool group_env = [] {
                        const char* v = std::getenv("STRATA_PREFILL_GROUP_GATHER");
                        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                    }();
                    const bool group_gather = group_env && stream_all && use_mmq && !mmq_direct;
                    mmq::GatherGroup gg;
                    int gg_slots[MMQ_GROUP];
                    int gg_nslots = 0;            // ring slots gathered by the next flush
                    size_t gg_consumed = consumed;   // what the issuer may have back after the next flush
                    auto flush = [&]() {
                        if (gg.n > gg.first) {
                            const auto& f = lay.fmt[(size_t) l];
                            if (gg_nslots > 0) {
                                pt.mark(kPfWaitCopy, cs);
                                strata::gpu::stream_wait_event(m.cs, m.copied[gg_slots[gg_nslots - 1]]);
                                pt.mark(kPfDequant, cs);
                            }
                            if (!mmq::gather_native_group(gg, f.up_off, mmq_gub / 2, f.down_off, mmq_db, m.grp_gu, mmq_gub,
                                                          m.grp_d, mmq_db, m.cs)) {
                                for (int i = gg.first; i < gg.n; ++i) {   // not 16-byte aligned: one at a time
                                    const uint8_t* b = gg.blob[i];
                                    if (b == nullptr) continue;   // read in its cache slot (mmq_tab_on)
                                    mmq::gather_native(b, b + f.up_off, mmq_gub / 2, b + f.down_off, mmq_db,
                                                       m.grp_gu + i * mmq_gub, m.grp_d + i * mmq_db, m.cs);
                                }
                            }
                            if (gg_nslots > 0) {
                                const int rel = gg_slots[gg_nslots - 1];
                                strata::gpu::event_record(m.used[rel], m.cs);
                                for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel;
                            }
                            gg_nslots = 0;
                            gg.first = gg.n;
                        }
                        if (gg_consumed > consumed) {
                            consumed = gg_consumed;
                            give_back(consumed);
                        }
                    };
                    // the MMQ group's last streamed copy into its ring slot (the copy queue is in order): the products
                    // wait for it once, not each copy for its own (upstream 653f0113's grouped staged walk); -1: none yet
                    int grp_last_copy = -1;
                    int64_t grp_rows_last = 0;   // the rows of the last int8 group (GU and H hold that group's)
                    // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                    // expert) is released once the blob is read
                    auto compute = [&](size_t j, const uint8_t* blob_dev, int slot) -> bool {
                        pt.mark(kPfDequant, cs);
                        if (use_mmq) {
                            const size_t q = (size_t) m.grp_pos[j];
                            const bool last_of_group = j + 1 >= (size_t) m.grp_start[(size_t) m.grp_of[j] + 1];
                            const auto& f = lay.fmt[(size_t) l];
                            const bool in_place = mtab && direct[j];   // read in its cache slot: not copied
                            if (mmq_direct) {
                                // a resident expert into its ring slot (the slot's last group is done: same stream)
                                if (slot < 0 && !in_place)
                                    strata::gpu::copy_async(slot_ptr((int) (j % MMQ_RING)), blob_dev,
                                                            (size_t) lay.blob_bytes(l), m.cs);
                            } else if (group_gather) {
                                gg.blob[q] = in_place ? nullptr : blob_dev;
                                gg.n = (int) q + 1;
                                if (slot >= 0) gg_slots[gg_nslots++] = slot;
                                if (!last_of_group) return true;
                                flush();
                                gg = mmq::GatherGroup{};
                            } else if (!in_place) {
                                // gather the expert into its group slot (its GGUF blocks; mmq_plan takes native packs)
                                mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                   mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                                if (slot >= 0) { strata::gpu::event_record(m.used[slot], m.cs); m.used_of[slot] = slot; }
                            }
                            if (!last_of_group) return true;
                            // the group's products: gate/up, swiglu, the group's H to int8, down
                            const size_t j0 = j - q, g = (size_t) m.grp_of[j], n = order.size();
                            const int ngx = (int) (q + 1);
                            const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
                            int64_t maxr = 0;
                            for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                            if (grp_last_copy >= 0) {
                                pt.mark(kPfWaitCopy, cs);
                                strata::gpu::stream_wait_event(m.cs, m.copied[grp_last_copy]);
                                grp_last_copy = -1;
                            }
                            pt.mark(kPfGemmGU, cs);
                            const uint8_t* ring0 = mmq_direct ? slot_ptr((int) (j0 % MMQ_RING)) : nullptr;
                            mmq::Product gu;
                            gu.w = mmq_direct ? ring0 : m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N;
                            gu.expert_bytes = mmq_direct ? MMQ_SLOT() : mmq_gub;
                            // the chunk's rows read (absolute bounds), the group's rows written from 0 (ids_rel)
                            gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = m.ids_rel;
                            gu.total_rows = T * K; gu.max_rows = maxr; gu.rows = nr; gu.dst = m.GU; gu.ld_dst = 1280;
                            const auto* tabs = reinterpret_cast<const uint8_t* const*>(m.tab_dev);
                            if (mtab) { gu.wtab = tabs + j0; gu.woff = 0; }
                            m.mmq_ctx->run(gu, m.cs);
                            mmq::swiglu(m.GU, m.H, nr, 640, !lay.native, m.cs);
                            pt.mark(kPfGemmD, cs);
                            mmq::quantize(m.H, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs);
                            grp_rows_last = nr;
                            mmq::Product dn;
                            dn.w = mmq_direct ? ring0 + f.down_off : m.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640;
                            dn.expert_bytes = mmq_direct ? MMQ_SLOT() : mmq_db;
                            dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
                            dn.ids = m.ids_identity; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = m.Dm + r0 * N;
                            dn.ld_dst = N;
                            if (mtab) { dn.wtab = tabs + n + j0; dn.woff = 0; }
                            m.mmq_ctx->run(dn, m.cs);
                            if (mmq_direct) {
                                const int par = (int) (g % 2);
                                strata::gpu::event_record(m.mgrp_used[par], m.cs);
                                m.mgrp_live[par] = true;
                            }
                            return true;
                        }
                        // dequantize into the group's slot q; the group's products once it is full (or the last)
                        const int G = g_expert_group, q = (int) ((j - nf) % (size_t) G);   // groups from nf on
                        uint16_t *gu16 = m.dq_gu + q * GU_ELEMS, *d16 = m.dq_d + q * D_ELEMS;
                        if (lay.native) {
                            // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                            const auto& f = lay.fmt[(size_t) l];
                            // STRATA_PF_DQ_FUSED=0: the two launches (upstream 75109596 fused them)
                            static const bool dq_fused = [] {
                                const char* v = std::getenv("STRATA_PF_DQ_FUSED");
                                return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                            }();
                            if (dq_fused) {
                                strata::kernels::iq_dequant_expert_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff,
                                                                       f.n_embd, gu16, f.d_type, blob_dev + f.down_off,
                                                                       d16, m.cs);
                            } else {
                                strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff,
                                                                   f.n_embd, gu16, m.cs);
                                strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, d16,
                                                                m.cs);
                            }
                        } else {
                            blob_dequant_f16(blob_dev, gu16, d16, m.cs);
                        }
                        if (slot >= 0) { strata::gpu::event_record(m.used[slot], m.cs); m.used_of[slot] = slot; }
                        if (q + 1 < G && j + 1 < order.size()) return true;
                        const size_t j0 = j - (size_t) q;
                        int64_t maxr = 0;
                        for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                        const int64_t r0 = m.xb_host[j0], nr = m.xb_host[j + 1] - r0;
                        pt.mark(kPfGemmGU, cs);
                        if (!Gemm::f16_groups(m.Xs, m.dq_gu, GU_ELEMS, m.GU, m.xb_host.data() + j0, q + 1, 1280, N, m.cs))
                            strata::kernels::xmx_gemm_grouped(m.Xs, m.dq_gu, GU_ELEMS, m.GU, m.xb_dev + j0, q + 1, maxr,
                                                              1280, N, m.cs);
                        swiglu_interleaved(m.GU + r0 * 1280, m.Hh + r0 * 640, nr, m.cs);
                        if (had_d) strata::kernels::hadamard_rows_f16(m.Hh + r0 * 640, nr, 640, had_block, had_sh, m.cs);
                        pt.mark(kPfGemmD, cs);
                        if (!Gemm::f16_groups(m.Hh, m.dq_d, D_ELEMS, m.Dm, m.xb_host.data() + j0, q + 1, N, 640, m.cs))
                            strata::kernels::xmx_gemm_grouped(m.Hh, m.dq_d, D_ELEMS, m.Dm, m.xb_dev + j0, q + 1, maxr, N,
                                                              640, m.cs);
                        return true;
                    };
                    if (!stream_all) {
                        size_t staged = nf;
                        const size_t lookahead = mmq_direct ? (size_t) MMQ_GROUP : (size_t) STAGE - 1;
                        for (size_t j = nf; j < order.size(); ++j) {
                            while (staged < order.size() && staged <= j + lookahead) {
                                if (!stage_one(staged)) return false;
                                ++staged;
                            }
                            const int32_t e = order[j];
                            if (stage_of[j] < 0) {
                                ++stats_.experts_resident;
                                if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                            } else {
                                if (mmq_direct && group_env) {
                                    grp_last_copy = stage_of[j];   // waited for once, before the group's products
                                } else {
                                    pt.mark(kPfWaitCopy, cs);
                                    strata::gpu::stream_wait_event(m.cs, m.copied[stage_of[j]]);
                                }
                                if (!compute(j, slot_ptr(stage_of[j]), stage_of[j])) return false;
                            }
                        }
                    } else {
                        // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
                        // pick only gives its slot back
                        size_t k = seq_start[(size_t) l];
                        const size_t kend = seq_start[(size_t) l + 1];
                        auto release_to = [&](int32_t e_stop) {
                            if (group_gather && k < kend && seq[k].e < e_stop) flush();
                            while (k < kend && seq[k].e < e_stop) {
                                strata::gpu::event_record(m.used[k % (size_t) m.ring], m.cs);
                                m.used_of[k % (size_t) m.ring] = (int) (k % (size_t) m.ring);
                                consumed = ++k;
                                give_back(consumed);
                            }
                        };
                        for (size_t j = nf; j < order.size(); ++j) {   // from nf on: id order again
                            const int32_t e = order[j];
                            release_to(e);
                            if (k < kend && seq[k].e == e) {
                                const int sl = (int) (k % (size_t) m.ring);
                                pt.mark(kPfWaitCopy, cs);
                                wait_issued(k);
                                if (group_gather) {
                                    gg_consumed = k + 1;   // given back by the flush that gathers it
                                } else {
                                    strata::gpu::stream_wait_event(m.cs, m.copied[sl]);
                                }
                                if (!compute(j, m.stage_dev[sl], sl)) return false;
                                ++k;
                                if (!group_gather) {
                                    consumed = k;
                                    give_back(consumed);
                                }
                            } else {
                                ++stats_.experts_resident;
                                if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                            }
                        }
                        release_to((int32_t) m.g->n_expert);
                    }
                    pt.mark(kPfCombine, cs);
                    if (rows_help > 0) {   // set_helper: its rows into Dm's tail, once its stream is done
                        HelperShare& hs = *helper_;
                        bool ok = false;
                        {
                            const core::OnDevice on(hs.dev);
                            ok = strata::gpu::stream_sync(hs.s);
                        }
                        if (!ok || !strata::gpu::copy_async(m.Dm + (size_t) (rows_gpu + rows_cpu) * N, hs.d_ex,
                                                            (size_t) rows_help * N * sizeof(float), m.cs)) {
                            err = std::string("prefill: the helper GPU's rows: ") + strata::gpu::last_error();
                            return false;
                        }
                        stats_.experts_helper += n_help;
                        if (static bool said = false; !said) {
                            said = true;
                            std::fprintf(stderr, "strata: the helper GPU computes the prompt's experts it holds that at "
                                                 "most %lld tokens route to (%lld of a layer's, %lld rows)\n",
                                         (long long) hs.max_rows, (long long) n_help, (long long) rows_help);
                        }
                    }
                    if (cpu_fut.valid()) {   // set_cpu_pool: the CPU's rows into Dm's tail
                        if (cpu_share_env() < 0.0) strata::gpu::event_record(m.cpu_ev[1], m.cs);   // after the GPU's experts
                        if (!cpu_fut.get()) { err = "prefill: the CPU experts failed"; return false; }
                        if (cpu_share_env() < 0.0 && n_stream > n_cpu) {
                            m.cpu_pend = true;
                            m.pend_cpu_ms = cpu_ms;
                            m.pend_n_cpu = n_cpu;
                            m.pend_n_gpu = n_stream - n_cpu;
                        }
                        stats_.cpu_share = cpu_share_env() >= 0.0 ? cpu_share_env() : m.cpu_share_now;
                        strata::gpu::copy_async(m.Dm + (size_t) rows_gpu * N, m.cpu_rows, (size_t) rows_cpu * N * sizeof(float),
                                                m.cs);
                        stats_.experts_cpu += n_cpu;
                        if (static bool said = false; !said) {
                            said = true;
                            std::fprintf(stderr, "strata: STRATA_PREFILL_CPU_SHARE: the CPU pool computes %lld of a layer's "
                                         "%lld streamed experts (share %.2f)\n", (long long) n_cpu, (long long) n_stream,
                                         stats_.cpu_share);
                        }
                    }
                    // no reading from a layer the share could not change (nothing taken) or with one-time costs
                    if (cpu_set >= 0 && n_stream > 0 && (!cpu_arm || n_cpu > 0) && !cpu_cold) {
                        strata::gpu::event_record(m.cpu_wall[(size_t) 2 * cpu_set + 1], m.cs);
                        m.pend_wall = cpu_arm ? 2 : 1;
                        m.pend_set = cpu_set;
                        m.pend_l = (int) l;
                        m.pend_wall_n = n_stream;
                    }
                    moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
                    // debug: STRATA_DBG_NAN=1 reports the first layer of a chunk whose MoE produced non-finite values
                    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {
                        strata::gpu::stream_sync(m.cs);
                        auto bad = [&](const float* d, int64_t n) {
                            std::vector<float> h((size_t) n);
                            strata::gpu::copy(h.data(), d, (size_t) n * 4);
                            int64_t c = 0;
                            for (float v : h) c += !std::isfinite(v);
                            return c;
                        };
                        // (the CPU's rows of GU and H are never written by the GPU: not counted, upstream 4322e241)
                        // (the int8 path's GU and H hold the last group's rows only)
                        const int64_t bgu = bad(m.GU, (use_mmq ? grp_rows_last : rows_gpu) * 1280), bdm = bad(m.Dm, T * K * N),
                                      bbo = bad(m.bo, T * N);
                        const int64_t bh = use_mmq && m.H ? bad(m.H, grp_rows_last * 640) : -1;
                        static int64_t reported = -1;
                        if ((bgu || bdm || bbo || bh > 0) && reported != stats_.chunks) {
                            reported = stats_.chunks;
                            std::fprintf(stderr, "strata dbg: layer %lld (mmq %d, types %d/%d, %zu experts): non-finite GU %lld "
                                         "H %lld Dm %lld bo %lld of T %lld\n", (long long) l, (int) use_mmq, mmq_gt, mmq_dt,
                                         order.size(), (long long) bgu, (long long) bh, (long long) bdm, (long long) bbo,
                                         (long long) T);
                        }
                    }
                }
                // ---- the hyper-connection write of this half; F-2: fused with the next half's norm when nothing else
                // touches R in between (not the stage's last half, not before the PLE block of layer 1, not under a
                // control vector)
                const int64_t nl = half == 0 ? l : l + 1;
                pt.mark(kPfHcWrite, cs);   // (after layer 0's: the wait for the PLE rows, ple_land)
                // a steered layer's write, vector and next norm as one pass too (upstream 6448f0f7; the same bits);
                // STRATA_CVEC_FUSE=0: gr_write, cvec_apply, then the next half's own norm
                static const bool cvec_fuse = [] {
                    const char* e = std::getenv("STRATA_CVEC_FUSE");
                    return e == nullptr || std::strtol(e, nullptr, 10) != 0;
                }();
                const bool steered = half == 1 && strata::kernels::cvec().covers(l);
                const bool fuse = nl < LE && !(half == 1 && nl == 1 && ple_on) && (!steered || cvec_fuse);
                const core::WeightRef* wnn = nullptr;
                if (fuse) {
                    const core::LayerView vn(*m.wt, nl);
                    wnn = need(vn, half == 0 ? "hc_ffn_norm.weight" : "hc_attn_norm.weight", err);
                    if (!wnn) return false;
                }
                if (wnn && steered) {
                    gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, l, (const float*) wnn->data, EPS, m.xn, m.xn16, T, m.cs,
                                          m.xn16_lo);
                    normed = true;
                } else if (wnn) {
                    gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.xn, m.xn16, T, m.cs,
                                     m.xn16_lo);
                    normed = true;
                } else {
                    gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
                    if (steered)   // --control-vector-scaled
                        strata::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
                }
            }
        }
        if (!ple_land()) return false;   // a stage that ends before layer 1: the rows land anyway, the next gather starts
        if (issuer.joinable()) {
            issuer.join();
            stats_.ms_experts_host += iss_ms;
            stats_.experts_streamed += iss_streamed;
            stats_.experts_dma += iss_dma;
        }
        if (m.stager && m.stager->failed.exchange(false)) {   // a transient source's copy_blob failed
            err = "prefill: the expert source could not copy a blob";
            return false;
        }
        stats_.tokens += T;
        core::progress_at("reading the prompt (batched): finishing the chunk from token", p0);
        pt.mark(kPfStart, cs);
        if (next_ != nullptr) {
            // the rows to the host buffer the next stage read two chunks ago (it has finished: waited below)
            float* h = m.hand[hand_buf];
            const int hp = m.hand_pieces;
            if (hp > 0) {
                // STRATA_HANDOFF_PIPE: the pieces' downloads stay queued; the host waits for the chunk's work only (the
                // state on_stage_chunk saves, the error check), the next stage's uploads for the pieces' events
                bool ok = strata::gpu::event_record(m.hand_done[hand_buf], m.cs);
                for (int i = 0; ok && i < hp; ++i) {
                    int64_t r0 = 0, nr = 0;
                    handoff_piece(T, hp, i, r0, nr);
                    if (nr > 0)
                        ok = strata::gpu::copy_async(h + (size_t) r0 * D, m.R + (size_t) r0 * D, (size_t) nr * D * 4,
                                                     m.cs);
                    ok = ok && strata::gpu::event_record(m.hand_ev[hand_buf][i], m.cs);
                }
                if (!ok || !strata::gpu::event_sync(m.hand_done[hand_buf])) {
                    err = std::string("prefill: the layer split's hand-off: ") + strata::gpu::last_error();
                    return false;
                }
            } else if (!strata::gpu::copy_async(h, m.R, (size_t) T * D * 4, m.cs) ||
                       !strata::gpu::stream_sync(m.cs)) {
                err = std::string("prefill: the layer split's hand-off: ") + strata::gpu::last_error();
                return false;
            }
            // this stage's state is at the chunk's end now (synced) and moves on with the next chunk below
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (!next_->wait(err)) return false;
            if (hp > 0 ? !next_->chunk_pieces(h, m.hand_ev[hand_buf], hp, tokens + c0, T, p0, err)
                       : !next_->chunk(h, tokens + c0, T, p0, err))
                return false;
            hand_buf ^= 1;
            continue;   // the last stage reports the chunk (on_chunk)
        }
        if (const char* dump = std::getenv("STRATA_PREFILL_DUMP_R")) {   // debug: the final residuals, every 64th
            strata::gpu::stream_sync(m.cs);                                  // position (A/B quality of this path)
            if (std::FILE* f = std::fopen(dump, c0 == 0 ? "wb" : "ab")) {
                std::vector<float> row((size_t) D);
                for (int64_t t = (64 - p0 % 64) % 64; t < T; t += 64) {
                    strata::gpu::copy(row.data(), m.R + t * D, (size_t) D * 4);
                    const int64_t pos = p0 + t;
                    std::fwrite(&pos, sizeof pos, 1, f);
                    std::fwrite(row.data(), 4, row.size(), f);
                }
                std::fclose(f);
            }
        }
        if (const char* dump = std::getenv("STRATA_PREFILL_DUMP_R_ALL")) {
            // draft-layer distillation data (opt-in, upstream 48d10a49): every position's final multi-stream residual
            // as BF16 (round-to-nearest-even), rows in position order, appended across chunks and requests:
            // [n][hc*n_embd]
            strata::gpu::stream_sync(m.cs);
            if (std::FILE* f = std::fopen(dump, "ab")) {
                constexpr int64_t kRows = 512;
                std::vector<float> rows((size_t) (kRows * D));
                std::vector<uint16_t> out((size_t) (kRows * D));
                for (int64_t t0 = 0; t0 < T; t0 += kRows) {
                    const int64_t nr = std::min<int64_t>(kRows, T - t0);
                    strata::gpu::copy(rows.data(), m.R + t0 * D, (size_t) (nr * D) * 4);
                    for (int64_t i = 0; i < nr * D; ++i) {
                        uint32_t u;
                        std::memcpy(&u, &rows[(size_t) i], 4);
                        u += 0x7fffu + ((u >> 16) & 1u);
                        out[(size_t) i] = (uint16_t) (u >> 16);
                    }
                    std::fwrite(out.data(), 2, (size_t) (nr * D), f);
                }
                std::fclose(f);
            }
        }
        if (on_chunk || on_stage_chunk) {
            const auto toc = Clock::now();
            if (!strata::gpu::stream_sync(m.cs)) {
                err = std::string("prefill: ") + strata::gpu::last_error();
                return false;
            }
            const auto toc2 = Clock::now();
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (on_chunk && !on_chunk(m.R, T, p0, err)) return false;
            host_sync_ms += std::chrono::duration<double, std::milli>(toc2 - toc).count();
            host_chunk_ms += ms_since(toc2);
        }
    }
    if (next_ != nullptr && !next_->wait(err)) return false;
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (std::getenv("STRATA_DBG_NAN") != nullptr) {   // debug: the state the prompt leaves for the token path
        strata::gpu::stream_sync(m.cs);
        auto bad = [&](const float* d, int64_t n) {
            std::vector<float> h((size_t) n);
            strata::gpu::copy(h.data(), d, (size_t) n * 4);
            int64_t c = 0;
            double mx = 0;
            for (float v : h) { c += !std::isfinite(v); if (std::isfinite(v)) mx = std::max(mx, (double) std::fabs(v)); }
            std::fprintf(stderr, " %lld non-finite (max |x| %.3g)", (long long) c, mx);
        };
        const int64_t last = (n - 1) % m.T;
        std::fprintf(stderr, "strata dbg: prompt end: last residual row");
        bad(m.R + last * D, D);
        if (ss.ple.ready()) { std::fprintf(stderr, "; PLE history"); bad(ss.ple.hist, (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM); }
        std::fprintf(stderr, "; GDN state 0");
        bad(ss.gdn_state, (int64_t) 64 * 1024);
        std::fprintf(stderr, "\n");
    }
    if (!strata::gpu::stream_sync(m.cs)) {
        err = std::string("prefill: ") + strata::gpu::last_error();
        return false;
    }
    stats_.ms_total += ms_since(t_start);
    if (pt.on) {
        pt.fold();
        double total = 0.0;
        for (double v : pt.ms) total += v;
        std::string line;
        char b[96];
        for (int i = 0; i < kPfCount; ++i) {
            if (pt.ms[i] <= 0.0) continue;
            std::snprintf(b, sizeof b, " %s %.0f (%.1f%%)", kPfNames[i], pt.ms[i], total > 0 ? 100.0 * pt.ms[i] / total : 0.0);
            line += b;
        }
        std::fprintf(stderr, "strata prefill timing: %lld tokens, GPU timeline %.0f ms, wall %.0f ms, host staging %.0f ms:%s\n",
                     (long long) n, total, ms_since(t_start), stats_.ms_experts_host, line.c_str());
        std::fprintf(stderr, "strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) %.0f ms, "
                             "waiting for each chunk %.0f ms, after each chunk (the draft layer, progress) %.0f ms, "
                             "PLE %.0f ms; at each MoE layer: the routing's sync %.0f ms, folding these marks %.0f ms "
                             "(the timing's own cost, charged to 'host grouping'), the grouping %.0f ms\n",
                     host_setup_ms, host_sync_ms, host_chunk_ms, stats_.ms_ple, host_route_ms, host_fold_ms, host_group_ms);
    }
    if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr) {   // debug: the GDN states as the prompt path leaves them
        strata::gpu::stream_sync(m.cs);
        std::vector<uint8_t> b((size_t) gdn_floats * 4);
        std::string line;
        char h[8];
        for (int64_t i = 0; i < ss.gdn_alloc; ++i) {   // the GDN layers this session owns
            strata::gpu::copy(b.data(), ss.gdn_state + (size_t) i * gdn_floats, b.size());
            uint64_t x = 1469598103934665603ull;
            for (uint8_t c : b) x = (x ^ c) * 1099511628211ull;
            std::snprintf(h, sizeof(h), "%04llx ", (unsigned long long) (x & 0xffff));
            line += h;
        }
        std::fprintf(stderr, "strata prefill: GDN_HASH %s\n", line.c_str());
    }
    return true;
}

// ---- the next stage in this process (PrefillLink)

void handoff_piece(int64_t T, int P, int i, int64_t& r0, int64_t& nr) {
    const int64_t per = (T + P - 1) / P;
    r0 = std::min<int64_t>(T, (int64_t) i * per);
    nr = std::min<int64_t>(T, r0 + per) - r0;
}

bool PrefillLink::chunk_pieces(const float* rows, gpu::Event* const* landed, int pieces, const int64_t* tokens,
                               int64_t T, int64_t p0, std::string& err) {
    for (int i = 0; i < pieces; ++i)
        if (!strata::gpu::event_sync(landed[i])) {
            err = std::string("prefill: the layer split's hand-off: ") + strata::gpu::last_error();
            return false;
        }
    return chunk(rows, tokens, T, p0, err);
}

// STRATA_HANDOFF_PIPE: one thread that runs the jobs in turn
struct InProcessPrefill::Worker {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::packaged_task<bool()>> jobs;
    bool stop = false;
    std::thread th;
    Worker() {
        th = std::thread([this] {
            strata::aux_cpus::pin_current_thread();
            for (;;) {
                std::packaged_task<bool()> job;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [this] { return stop || !jobs.empty(); });
                    if (jobs.empty()) return;   // stopped, and every job taken has run (no broken promise)
                    job = std::move(jobs.front());
                    jobs.pop_front();
                }
                job();
            }
        });
    }
    std::future<bool> submit(std::packaged_task<bool()> job) {
        std::future<bool> f = job.get_future();
        {
            std::lock_guard<std::mutex> lk(mu);
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
        return f;
    }
    ~Worker() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_one();
        if (th.joinable()) th.join();
    }
};

InProcessPrefill::InProcessPrefill() = default;

InProcessPrefill::~InProcessPrefill() {
    if (run_.valid()) run_.wait();   // the chain's last chunk, before the thread that would run the next one goes
}

bool InProcessPrefill::start(const int64_t* tokens, int64_t T, int64_t p0) {
    if (handoff_pipe_worker()) {
        if (!worker_) worker_ = std::make_unique<Worker>();
        run_ = worker_->submit(
            std::packaged_task<bool()>([this, tokens, T, p0] { return next_->run(tokens, T, p0, err_); }));
    } else
        run_ = std::async(std::launch::async, [this, tokens, T, p0] { return next_->run(tokens, T, p0, err_); });
    return true;
}

bool InProcessPrefill::chunk(const float* rows, const int64_t* tokens, int64_t T, int64_t p0, std::string& err) {
    (void) err;
    next_->accept_rows(rows);
    return start(tokens, T, p0);
}

bool InProcessPrefill::chunk_pieces(const float* rows, gpu::Event* const* landed, int pieces, const int64_t* tokens,
                                    int64_t T, int64_t p0, std::string& err) {
    (void) err;
    next_->accept_rows(rows, landed, pieces);
    return start(tokens, T, p0);
}

bool InProcessPrefill::wait(std::string& err) {
    if (!run_.valid() || run_.get()) return true;
    err = err_;
    return false;
}

}  // namespace strata::prefill
