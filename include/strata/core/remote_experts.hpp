// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExpertOpt;

/// A static, profile-filled expert tier on another GPU. GPU 0 keeps all
/// dense weights and state; results return through the existing pinned CPU rows.
class RemoteExperts {
public:
    RemoteExperts() = default;
    ~RemoteExperts();
    RemoteExperts(const RemoteExperts&) = delete;
    RemoteExperts& operator=(const RemoteExperts&) = delete;

    /// Initialise the device before the host expert arena registers
    /// tens of GiB of portable mapped memory with CUDA.
    static bool preflight(int device, double& free_gib, std::string& err);
    bool open(int device, int slots, int64_t layers, int64_t experts,
              const std::vector<std::pair<int32_t, int32_t>>& ranked,
              const ExpertCache& primary, ExpertSource& source,
              std::vector<uint8_t>& claimed, std::string& err);
    /// --expert-host-device1: the helper holds no VRAM cache at all.  Its kernels read the host
    /// copy's blobs in place - an iGPU's UMA, or a GPU whose kernels read host memory registered
    /// on it (gpu::kernels_read_registered; Level Zero's registrations are copy-only, refused inside).
    /// "Held" is then: in the host copy (a registered address) and in no GPU's expert cache.
    /// open() and this are exclusive; `claimed` (a layer split's stage caches) is only read.
    bool open_host(int device, int64_t layers, int64_t experts, const ExpertCache& primary,
                   ExpertSource& source, const std::vector<uint8_t>& claimed, std::string& err);
    /// open_host(): this helper computes straight from the host arena, it has no VRAM cache.
    bool host_mode() const { return host_mode_; }
    /// Experts open_host() marked held (not GPU-resident, a stable arena address).
    int64_t host_held() const { return host_count_; }
    void close();

    /// `kind` is the primary verifier's classification (-1 = CPU candidate),
    /// or null on the one-token path. Entries already served on CUDA0 are excluded.
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
               int64_t k, const int32_t* kind, const int32_t* primary_res,
               std::string& err);
    bool owns(int64_t index) const { return owned_[(size_t) index] != 0; }
    /// This helper serves (layer, expert): in its VRAM cache, or - host mode - in the arena with a
    /// stable registered address.  begin() takes its rows unless the plan gave them away.
    bool holds(int64_t layer, int32_t expert) const {
        if (host_mode_) {
            const int64_t i = layer * n_expert_ + expert;
            return i >= 0 && i < (int64_t) host_held_.size() && host_held_[(size_t) i] != 0;
        }
        return cache_.slot_of(layer, expert) >= 0;
    }
    bool optimized_decode() const { return remote_opt_ != nullptr; }
    /// The engine device this helper runs on (-1: not open).
    int device() const { return device_; }
    /// (layer, expert)'s blob where this helper's kernels read it: its VRAM slot, or - host mode -
    /// its registered host address in the arena.  Null where it does not hold it.
    const uint8_t* blob_on_device(int64_t layer, int32_t expert) const {
        if (host_mode_) {
            const int64_t i = layer * n_expert_ + expert;
            return i >= 0 && i < (int64_t) host_blob_.size() ? host_blob_[(size_t) i] : nullptr;
        }
        const int32_t slot = cache_.slot_of(layer, expert);
        return slot >= 0 ? cache_.device_slot(slot) : nullptr;
    }
    bool finish(float* out, std::string& err);
    int64_t resident() const { return cache_.resident(); }
    int64_t computed() const { return computed_; }
    int64_t launched_layers() const { return launched_layers_; }
    double gib() const { return cache_.gib(); }
    uint64_t returned_bytes() const { return returned_bytes_; }
    uint64_t full_row_bytes() const { return full_row_bytes_; }
    /// STRATA_PREFILL_HELPER_SOURCE (opt-in): copy up to `n` cached blobs into the host buffers `dst[i]`
    /// on this helper's own stream and wait.  `hit[i]` is set where this helper held (layer, expert); the
    /// bytes are the ones open() filled from the same source, so the prompt path gets identical bytes.
    /// Misses are left to the caller.  False with `err` on a failure; the hits copied so far are complete
    /// (the stream is drained), so the caller falls back for the rest.  Safe from several threads (one
    /// stream, one lock): the prompt path's stager is threaded.
    bool copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                           uint8_t* hit, std::string& err);
    /// host time spent in begin() (staging + launches) and in finish() (waiting for this GPU), cumulative
    double ms_begin() const { return ms_begin_; }
    double ms_wait() const { return ms_wait_; }

private:
    friend class RemoteExpertOpt;
    RemoteExpertOpt* remote_opt_ = nullptr;
    int device_ = -1;
    int64_t n_expert_ = 0;
    int32_t groups_ = 0;
    int64_t computed_ = 0;
    int64_t launched_layers_ = 0;
    uint64_t returned_bytes_ = 0;
    uint64_t full_row_bytes_ = 0;
    double ms_begin_ = 0, ms_wait_ = 0;
    ExpertCache cache_;
    void* stream_ = nullptr;
    void* read_stream_ = nullptr;   ///< STRATA_PREFILL_HELPER_SOURCE: cache -> host copies
    std::mutex read_mu_;            ///< one cache reader at a time (the stager is threaded)
    float* h_x_ = nullptr;     ///< host USM of this GPU's runtime: its kernels read it in place (zero copy)
    float* h_out_ = nullptr;   ///< the same, written in place
    void* h_meta_ = nullptr;
    bool zero_copy_ = false;
    float* d_x_ = nullptr;
    float* d_out_ = nullptr;
    uint8_t* d_q8_ = nullptr;
    float* d_scales_ = nullptr;
    void* d_scratch_ = nullptr;
    void* d_meta_ = nullptr;  ///< one upload of the grouped indices, instead of five small copies
    int32_t* d_start_ = nullptr;
    int32_t* d_dst_ = nullptr;
    int32_t* d_tok_ = nullptr;
    int32_t* d_count_ = nullptr;
    unsigned long long* d_ptr_ = nullptr;
    std::vector<uint8_t> owned_;
    std::vector<uint8_t> layers_present_;
    std::vector<int32_t> group_of_, group_id_;
    std::vector<int32_t> start_, dst_, tok_, original_row_;
    std::vector<unsigned long long> ptr_;

    // host mode (--expert-host-device1): no cache_; the kernels read the host copy's registered
    // addresses in place.  host_blob_ is the address last served; begin() claims a row while the
    // source still answers it, or - FileExpertSource, whose resident experts move with the adaptive
    // tier's exchanges and rotation - one inside a block registered on this GPU (registered_range,
    // host_file_).  Any other answer leaves the row to the CPU.
    bool probe_arena_read(int64_t layers, std::string& err);
    bool host_mode_ = false;
    ExpertSource* host_src_ = nullptr;
    FileExpertSource* host_file_ = nullptr;   ///< host_src_ when it is one: its registered blobs move
    std::vector<uint8_t> host_held_;
    std::vector<const uint8_t*> host_blob_;
    int64_t host_count_ = 0;
};

/// STRATA_PREFILL_HELPER_SOURCE (opt-in): the prompt path's view of the expert tiers, where a blob a helper
/// GPU already holds in its VRAM cache is copied from that cache instead of being read from the files again.
/// Everything else is forwarded to `fallback` unchanged, so its reads, its pinned RAM copy and the prompt
/// path's own decisions keep their meaning.  Nothing is cached here: every call asks the helpers.
///
/// The serve loop builds this only for a serial request path (no batch slots, no layer split), while the
/// helpers are idle: during a prompt read the helper GPUs compute nothing, so their caches can be read
/// without ordering the copies against their kernel streams.
class RemotePrefillSource final : public ExpertSource {
public:
    RemotePrefillSource(ExpertSource& fallback, const std::vector<RemoteExperts*>& helpers)
        : fallback_(fallback), helpers_(helpers) {}

    /// Whether any helper's VRAM cache holds (layer, expert) right now.
    bool from_helper(int64_t layer, int64_t expert) const {
        for (const RemoteExperts* h : helpers_) if (h->holds(layer, (int32_t) expert)) return true;
        return false;
    }
    /// Blobs / bytes this source has taken from the helpers so far (the per-request evidence line).
    uint64_t helper_blobs() const { return helper_blobs_.load(std::memory_order_relaxed); }
    uint64_t helper_bytes() const { return helper_bytes_.load(std::memory_order_relaxed); }

    const uint8_t* blob(int64_t layer, int64_t expert) override { return fallback_.blob(layer, expert); }
    int64_t reads() const override { return fallback_.reads(); }
    bool pinned(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? false : fallback_.pinned(layer, expert);
    }
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override {
        fallback_.begin_layer(layer, ids, k);
    }
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? nullptr : fallback_.device_alias(layer, expert);
    }
    bool make_kernel_readable(std::string& err) override { return fallback_.make_kernel_readable(err); }
    void prefetch(int64_t layer, int64_t expert) override { fallback_.prefetch(layer, expert); }
    uint64_t release(int64_t layer, int64_t expert) override {
        return from_helper(layer, expert) ? 0 : fallback_.release(layer, expert);
    }
    bool pcie_layer(int64_t layer) const override { return fallback_.pcie_layer(layer); }
    /// True where a helper holds the blob: the prompt path's stager must then take the copy_blob route,
    /// which is where the cache-to-host copy happens.  Everywhere else the fallback answers as it always did.
    bool transient(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? true : fallback_.transient(layer, expert);
    }
    double cached_share(int64_t samples) const override { return fallback_.cached_share(samples); }
    bool register_on(int device, std::string& err) override { return fallback_.register_on(device, err); }
    bool registered_on(int device) const override { return fallback_.registered_on(device); }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override {
        return fallback_.advise_pairs(pairs, n);
    }
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override {
        fallback_.prefetch(layer, experts, n);
    }

private:
    void warn(const std::string& err);
    ExpertSource& fallback_;
    std::vector<RemoteExperts*> helpers_;
    std::atomic<uint64_t> helper_blobs_{0}, helper_bytes_{0};
    std::atomic<bool> warned_{false};
};

} // namespace strata::core
