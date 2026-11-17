// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/remote_expert_opt.hpp - the helper-GPU decode optimization (--remote-expert-opt),
// on the Xe runtime (upstream src/core/remote_expert_opt.cu, v0.2.x): the helper's routed expert rows are
// reduced on the helper to one weighted sum per token, so a layer returns H floats a token instead of a
// row a routed entry; the CPU pool then skips the rows the helper serves.
//
// Upstream keeps the sums in mapped CUDA memory every device sees.  SYCL's host USM belongs to one
// context, so the one shared array becomes one host-USM copy per consuming device (`sinks_`): the
// verifier registers its device in init (attach_device) and begin/accumulate write every copy alike.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExperts;
class ExpertSource;

// Remote Expert Decode Optimization v1 on existing helper caches. One instance per serial engine.
class RemoteExpertOpt {
public:
    RemoteExpertOpt() = default;
    ~RemoteExpertOpt();
    RemoteExpertOpt(const RemoteExpertOpt&) = delete;
    RemoteExpertOpt& operator=(const RemoteExpertOpt&) = delete;

    void attach(RemoteExperts& remote); // before RemoteExperts::open / open_host
    /// Undoes attach when the helper's open failed and it stays closed (a refused host-arena
    /// helper): its peer is removed so init() allocates nothing for it.
    void detach(RemoteExperts& remote);
    bool init(std::string& err);        // after all helper caches are open
    /// A verifier's device (Verifier::init): its kernels read their own host-USM copy of the sums.
    /// Idempotent; the copy for device 0 is made by init().
    bool attach_device(int device, std::string& err);
    bool owns(int64_t layer, int32_t expert) const;
    bool adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
               const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
               ExpertSource& source);

    // Bracket the existing pool callback; its signature and dispatch remain unchanged.
    void begin(const float* weights, int token_begin, int tokens);
    void end() { weights_ = nullptr; }
    void copy_rows(float* dst, const float* src, int token_begin, int tokens,
                   const int32_t* hit_rows, const int32_t* count, void* stream) const;
    void combine(float* dst, float* scratch, int token_begin, int tokens,
                 const uint32_t* skip, uint32_t ring, void* stream) const;

private:
    friend class RemoteExperts;
    static size_t metadata_bytes();
    bool active() const { return weights_ != nullptr; }
    void prepare(const RemoteExperts& remote, void* metadata) const;
    bool reduce(RemoteExperts& remote, const void* metadata, std::string& err);
    void accumulate(const RemoteExperts& remote);

    struct Peer { RemoteExperts* remote; float* sum = nullptr; };
    /// One device's view of the accumulated sums: host USM allocated on it, so only its kernels may
    /// read `sum`/`mask` (the mask `K` int32 a token sits behind the `H` floats).
    struct Sink { int device = -1; float* sum = nullptr; int32_t* mask = nullptr; };
    const Sink* sink_for(void* stream) const;

    std::vector<Peer> peers_;
    std::vector<Sink> sinks_;
    const float* weights_ = nullptr;
    int token_begin_ = 0, tokens_ = 0;
};
} // namespace strata::core
