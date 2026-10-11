// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only tests of extracted PRODUCTION orchestration and qsa_step_fill.
// Compile/run through tools/test_session_token_pool_host.py, not the GPU build.
// Fake blocks/pool/composition test handoff ONLY, not production adapter math,
// QSA kernels, PLE, session allocation, or real GPU/error/drain semantics.
#include "strata/core/session.hpp"
#include "strata/core/device.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace strata::core {
uint64_t gdn_state_floats(const ModelGeometry&);
}
namespace strata::gpu {
struct Event { size_t end = 0; };
}

namespace {
using namespace strata::core;
using namespace strata::kernels;
#define CHECK(x) do { if (!(x)) { std::cerr << __FILE__ << ':' << __LINE__ \
    << ": " << #x << '\n'; std::exit(2); } } while (0)

struct Fixture;
Fixture* active = nullptr;
struct Fixture {
    ModelGeometry g;
    WeightTable tables;
    SessionState s;
    Doorbell db;
    SessionLoopScratch scratch;
    strata::gpu::Event event;
    std::array<QsaState, 12> q;
    std::array<std::array<int32_t, kStepCount>, 12> host_step{}, step{};
    std::array<std::array<int32_t, 24>, 12> host_pos{}, pos_dev{};
    std::array<int, 12> kv_visits{};
    std::vector<float> recurrent;
    std::array<float, 4> R{}, mixed{}, block_out{}, hx{};
    std::array<int32_t, 2> ids{}, hids{};
    std::array<float, 2> weights{}, hw{};
    std::array<float, 8> parts{}, answer{};
    std::deque<std::function<void()>> fifo;
    size_t submitted = 0, executed = 0;
    std::vector<std::string> calls;
    std::vector<std::pair<std::string, size_t>> transfers;
    PoolTiming timing{};
    bool timed = false;
    std::string failure, diagnostic;
    bool injected = false, cleanup_failure = false, capturing = false;
    bool use_pool = true, use_hits = true;
    int layer = -1, posts = 0, pools = 0, syncs = 0, token = 0;
    int launch_layer = -1, combine_layer = -1, hit_stamp = -1;
    float* dump_target = nullptr;
    int64_t expected_pos = 0;
    int32_t expected_base = 0;

    Fixture() {
        CHECK(active == nullptr);
        active = this;
        // Keep the real 48/4 topology and 24 heads, but tiny activations/state.
        g.n_embd = 4; g.hc = 1;
        g.ssm_state_size = 2; g.ssm_v_heads = 3;
        g.ssm_conv_channels = 5; g.ssm_d_conv = 4;
        CHECK(g.n_layers == 48 && g.n_qsa_layers() == 12 && g.n_gdn_layers() == 36);
        CHECK(gdn_state_floats(g) == 27);
        recurrent.resize(36 * 27 + 2, -999);
        s.gdn_state = recurrent.data() + 1;
        for (int j = 0; j < 36; ++j) {
            for (int i = 0; i < 27; ++i) s.gdn_state[j * 27 + i] = float(j * 100 + i);
        }
        s.max_cells = 4096; s.k = 2;
        s.layer_hi = 48; s.qsa_alloc = 12; s.gdn_alloc = 36;
        s.qsa_states = q.data();
        for (int j = 0; j < 12; ++j) {
            q[j].max_cells = s.max_cells;
            q[j].host_step = host_step[j].data(); q[j].step = step[j].data();
            q[j].host_pos = host_pos[j].data(); q[j].pos_dev = pos_dev[j].data();
        }
        s.block.R = R.data(); s.R = R.data();
        s.block.mixed = mixed.data(); s.block.block_out = block_out.data();
        s.moe.ids = ids.data(); s.moe.weights = weights.data();
        db.n_embd = 4; db.k = 2;
        db.h_x_f = hx.data(); db.h_ids = hids.data(); db.h_weights = hw.data();
        // Mapped device publication and ringing intentionally unavailable.
        s.db = &db;
        scratch.y_miss = answer.data(); scratch.parts_bytes = sizeof(answer);
        scratch.probe = &event;
        parts.fill(-777); answer.fill(-888);
    }
    ~Fixture() { CHECK(fifo.empty()); active = nullptr; }
    void enqueue(std::function<void()> op) { fifo.push_back(std::move(op)); ++submitted; }
    void drain(size_t end) {
        CHECK(end <= submitted);
        while (executed < end) {
            CHECK(!fifo.empty());
            auto op = std::move(fifo.front()); fifo.pop_front(); ++executed; op();
        }
    }
    bool fail(const std::string& what) {
        if (!injected && layer == 1 && failure == what) {
            injected = true; diagnostic = "injected " + what; return true;
        }
        return false;
    }
    float value(int l, int row, int col) const {
        return float(token * 10000 + l * 100 + row * 10 + col);
    }
    bool run(int64_t pos = 0, int32_t base = 0, bool each = false, float* dump = nullptr) {
        expected_pos = pos; expected_base = base; dump_target = dump;
        std::string err = "stale diagnostic";
        bool ok = session_token_pool(tables, g, pos, base, s, parts.data(),
                                    use_pool ? pool : nullptr, use_hits ? hits : nullptr,
                                    this, scratch, this, each, err, dump, timed ? &timing : nullptr);
        diagnostic = err;
        return ok;
    }
    static void pool(void* user, const float* x, const int32_t* id, const float* w,
                     int64_t n, int64_t k, float* out) {
        auto& f = *static_cast<Fixture*>(user);
        f.calls.push_back("pool");
        CHECK(n == 4 && k == 2 && out == f.answer.data());
        CHECK(x == f.db.h_x_f && id == f.db.h_ids && w == f.db.h_weights);
        CHECK(!f.use_hits || f.launch_layer == f.layer);
        // The preceding post must have executed BEFORE staging can be reused.
        CHECK(f.posts == f.token * 48 + f.layer);
        for (int i = 0; i < 4; ++i) CHECK(x[i] == f.value(f.layer, 0, i));
        for (int j = 0; j < 2; ++j) {
            CHECK(id[j] == f.layer * 3 + j);
            CHECK(w[j] == (j == 0 ? 0.25f : 0.75f));
        }
        if (f.fail("pool throw")) throw std::runtime_error("pool callback original");
        if (f.fail("pool unknown")) throw 17;
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 4; ++i)
                out[j * 4 + i] = f.use_hits && j == 1 ? 0 : f.value(f.layer, j, i);
        ++f.pools;
    }
    static void hits(void* user, void* stream, HitPhase phase, const int32_t* id, int64_t k) {
        auto& f = *static_cast<Fixture*>(user);
        CHECK(stream == &f);
        const int l = f.layer;
        if (phase == HitPhase::Launch) {
            f.calls.push_back("launch");
            CHECK(id == f.hids.data() && k == 2 && id[0] == l * 3);
            if (f.fail("launch throw")) throw std::runtime_error("launch callback original");
            f.launch_layer = l;
            // Retain the pointer: the read happens only when the stream executes.
            f.enqueue([&f, id, l] { CHECK(id[1] == l * 3 + 1); f.hit_stamp = l; });
        } else {
            f.calls.push_back("combine");
            CHECK(id == nullptr && k == 0 && f.launch_layer == l);
            if (f.fail("combine throw")) throw std::runtime_error("combine callback original");
            f.combine_layer = l;
            f.enqueue([&f, l] {
                CHECK(f.hit_stamp == l);
                for (int i = 0; i < 4; ++i) {
                    CHECK(f.parts[4 + i] == 0);
                    f.parts[4 + i] = f.value(l, 1, i) + 500;
                }
            });
        }
    }
};
}  // namespace

namespace strata::kernels {
// PLE stays disabled. A call here would violate this fixture's scope.
bool PleTable::is_open() const { CHECK(false); return false; }
}

// Only operations referenced by the extracted functions are defined. No runtime
// header, runtime object, allocation, thread, Windows API, or driver is linked.
namespace strata::gpu {
const char* last_error() { return active->diagnostic.c_str(); }
bool is_capturing(Stream s) { CHECK(s == active); return active->capturing; }
bool copy_async(void* dst, const void* src, size_t bytes, Stream s) {
    auto& f = *active; CHECK(s == &f);
    std::string what;
    if (f.dump_target != nullptr && src == f.R.data())
        what = dst == f.dump_target ? "dump input" : "dump layer";
    else if (dst == f.hx.data()) what = "D2H mixed";
    else if (dst == f.hids.data()) what = "D2H IDs";
    else if (dst == f.hw.data()) what = "D2H weights";
    else if (dst == f.parts.data()) what = "H2D parts";
    else what = "QSA upload";
    f.calls.push_back(what);
    if (f.fail(what)) return false;
    f.transfers.emplace_back(what, bytes);
    f.enqueue([dst, src, bytes] { std::memcpy(dst, src, bytes); });
    return true;
}
bool event_record(Event* e, Stream s) {
    auto& f = *active; CHECK(s == &f && e == &f.event);
    f.calls.push_back("record");
    if (f.fail("record staging event")) return false;
    e->end = f.submitted; return true;
}
bool event_sync(const Event* e) {
    auto& f = *active; CHECK(e == &f.event);
    f.calls.push_back("wait");
    if (f.fail("wait staging event")) return false;
    f.drain(e->end); return true;
}
bool stream_sync(Stream s) {
    auto& f = *active; CHECK(s == &f);
    f.calls.push_back("sync"); ++f.syncs;
    // Even failed mock waits drain: assert cleanup ATTEMPTS, not a real driver's
    // guarantee of completion after a failed wait (which this test cannot prove).
    f.drain(f.submitted);
    if (f.fail("post sync")) return false;
    if (f.failure == "final drain" || (f.injected && f.cleanup_failure)) {
        f.diagnostic = "cleanup original"; return false;
    }
    return true;
}
}  // namespace strata::gpu

namespace strata::core {
bool block_layer_pre(const WeightTable&, const ModelGeometry& g, int64_t l, int64_t pos,
                     int32_t base, const GdnBuffers& gb, const QsaState& qs, const QsaBuffers&,
                     const MoEBuffers& mb, int64_t k, const BlockBuffers& bb, void* stream,
                     std::string& err, const Doorbell* db, const PleRun* ple, int half, int prefix) {
    auto& f = *active; CHECK(stream == &f);
    f.layer = int(l); f.calls.push_back("pre");
    CHECK(pos == f.expected_pos && base == f.expected_base);
    CHECK(db == nullptr && ple == nullptr && half == 0 && prefix == 0 && k == 2);
    // A pre may have submitted work even if its final validation fails.
    f.enqueue([] {});
    if (f.fail("pre")) { err = "pre original"; return false; }
    if (l % 4 == 3) {
        const int j = int(l / 4); CHECK(&qs == &f.q[j]);
        qsa_step_fill(qs.host_step, pos, qsa_real_shapes());
        for (int h = 0; h < 24; ++h) qs.host_pos[h] = int32_t(pos + base);
        CHECK(strata::gpu::copy_async(qs.step, qs.host_step, qsa_step_bytes(), stream));
        CHECK(strata::gpu::copy_async(qs.pos_dev, qs.host_pos, 24 * sizeof(int32_t), stream));
        f.enqueue([&f, j, pos, base] {
            CHECK(f.step[j][kStepPos] == pos && f.step[j][kStepNKv] == pos + 1);
            for (int h = 0; h < 24; ++h) CHECK(f.pos_dev[j][h] == pos + base);
            CHECK(f.kv_visits[j] == f.token); ++f.kv_visits[j];
        });
    } else {
        const int j = int(l - l / 4);
        CHECK(&qs == &f.q[0]);
        CHECK(gb.state == f.s.gdn_state + j * 27);
        CHECK(gb.conv_state == gb.state + 12);
        float* state = gb.state; float* conv = gb.conv_state;
        f.enqueue([&f, state, conv, j] {
            CHECK(state[0] == j * 100 + f.token);
            CHECK(conv[0] == j * 100 + 12 + f.token * 2);
            ++state[0]; conv[0] += 2;
        });
    }
    CHECK(g.n_layers == 48);
    f.enqueue([&f, l, mb, bb] {
        for (int i = 0; i < 4; ++i) bb.mixed[i] = f.value(int(l), 0, i);
        mb.ids[0] = int(l * 3); mb.ids[1] = int(l * 3 + 1);
        mb.weights[0] = 0.25f; mb.weights[1] = 0.75f;
    });
    return true;
}
bool block_layer_post(const WeightTable&, const ModelGeometry&, int64_t l, int64_t k,
                      const MoEBuffers& mb, const BlockBuffers& bb, const float* parts,
                      void* stream, std::string& err) {
    auto& f = *active; CHECK(stream == &f && k == 2);
    f.calls.push_back("post");
    CHECK(!f.use_hits || f.combine_layer == l);
    f.enqueue([&f, l, mb, bb, parts] {
        CHECK(f.posts == f.token * 48 + l);
        CHECK(mb.ids[0] == l * 3 && mb.ids[1] == l * 3 + 1);
        CHECK(mb.weights[0] == 0.25f && mb.weights[1] == 0.75f);
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 4; ++i) {
                float expected = f.use_pool ? f.value(int(l), j, i) : 0;
                if (f.use_hits && j == 1) expected = f.value(int(l), j, i) + 500;
                CHECK(parts[j * 4 + i] == expected);
            }
        // A fake residual sentinel, NOT moe_combine/adapter arithmetic.
        bb.R[0] += float(l + 1); ++f.posts;
    });
    if (f.fail("post")) { err = "post original"; return false; }
    return true;
}
}  // namespace strata::core

namespace {
static_assert(std::is_trivial_v<PoolTiming> && std::is_standard_layout_v<PoolTiming>, "timing must be POD");

// Independent oracle: counts from the unchanged mock call trace, bytes from successful production
// copy_async arguments. Failed copies contribute an attempt but no submitted bytes; throws count too.
void check_timing(const Fixture& f, uint64_t drains = 1) {
    auto count = [&](const char* name) -> uint64_t {
        return (uint64_t) std::count(f.calls.begin(), f.calls.end(), name);
    };
    auto bytes = [&](const char* name) -> uint64_t {
        uint64_t n = 0;
        for (const auto& transfer : f.transfers) if (transfer.first == name) n += transfer.second;
        return n;
    };
    auto span = [](const PoolTiming::Span& s, uint64_t n, uint64_t b = 0) {
        CHECK(s.count == n && s.bytes == b && std::isfinite(s.ms) && s.ms >= 0);
        if (n == 0) CHECK(s.ms == 0);
    };
    span(f.timing.pre_submit, count("pre"));
    span(f.timing.routing_d2h_submit,
         count("D2H mixed") + count("D2H IDs") + count("D2H weights") + count("record"),
         bytes("D2H mixed") + bytes("D2H IDs") + bytes("D2H weights"));
    span(f.timing.routing_wait, count("wait"));
    span(f.timing.cpu_pool, count("pool"));
    span(f.timing.parts_h2d_submit, count("H2D parts"), bytes("H2D parts"));
    CHECK(count("sync") >= drains);
    span(f.timing.parts_wait, count("sync") - drains);
    span(f.timing.post_submit, count("post"));
    span(f.timing.dump_d2h_submit, count("dump input") + count("dump layer"),
         bytes("dump input") + bytes("dump layer"));
    span(f.timing.final_drain, drains);
    span(f.timing.embedding_submit, 0);
    span(f.timing.head_submit, 0);
    span(f.timing.head_wait, 0);
}

void orchestration() {
    CHECK(gdn_state_floats(ModelGeometry{}) == 817152);
    for (bool each : {false, true}) {
        Fixture f;
        for (int t = 0; t < 2; ++t) {
            f.token = t;
            CHECK(f.run(t + 3, 7, each)); CHECK(f.diagnostic.empty());
            CHECK(f.fifo.empty() && f.executed == f.submitted);
            CHECK(f.posts == (t + 1) * 48 && f.pools == (t + 1) * 48);
            CHECK(f.R[0] == (t + 1) * 1176);
            for (int j = 0; j < 12; ++j) CHECK(f.kv_visits[j] == t + 1);
            for (int j = 0; j < 36; ++j)
                for (int i = 0; i < 27; ++i) {
                    const int delta = i == 0 ? t + 1 : i == 12 ? 2 * (t + 1) : 0;
                    CHECK(f.s.gdn_state[j * 27 + i] == j * 100 + i + delta);
                }
            CHECK(f.recurrent.front() == -999 && f.recurrent.back() == -999);
        }
        CHECK(f.syncs == (each ? 98 : 2));
        // Exact submission/callback ordering, with only QSA uploads/debug sync
        // removed. Execution assertions above additionally catch early reuse.
        std::vector<std::string> filtered;
        for (const auto& c : f.calls) if (c != "QSA upload" && c != "sync") filtered.push_back(c);
        const std::vector<std::string> expected = {
            "pre", "D2H mixed", "D2H IDs", "D2H weights", "record", "wait",
            "launch", "pool", "H2D parts", "combine", "post"};
        CHECK(filtered.size() == 96 * expected.size());
        for (size_t i = 0; i < filtered.size(); ++i) CHECK(filtered[i] == expected[i % expected.size()]);
    }
    for (bool pool : {false, true}) for (bool hits : {false, true}) {
        Fixture f; f.use_hits = hits; f.use_pool = pool;
        CHECK(f.run(0, 0)); CHECK(f.posts == 48);
        if (!pool && !hits) for (float v : f.parts) CHECK(v == 0);
    }
    // Sanity-check the mock itself: no snapshot at submission, and an event
    // waits only to its marked point, not through commands submitted later.
    Fixture f;
    f.answer.fill(3);
    CHECK(strata::gpu::copy_async(f.parts.data(), f.answer.data(), sizeof(f.answer), &f));
    CHECK(strata::gpu::event_record(&f.event, &f));
    f.answer.fill(9);
    f.enqueue([&f] { f.parts[0] = 123; });
    CHECK(strata::gpu::event_sync(&f.event));
    for (float v : f.parts) CHECK(v == 9);
    CHECK(f.fifo.size() == 1);
    CHECK(strata::gpu::stream_sync(&f)); CHECK(f.parts[0] == 123);
}

void guards() {
    using Mutate = std::function<void(Fixture&)>;
    const std::vector<std::pair<Mutate, std::string>> cases = {
        {[](Fixture& f) { f.g.n_layers = 0; }, "invalid geometry or k"},
        {[](Fixture& f) { f.g.qsa_interval = 0; }, "invalid geometry or k"},
        {[](Fixture& f) { f.g.n_embd = 0; }, "invalid geometry or k"},
        {[](Fixture& f) { f.s.k = 0; }, "invalid geometry or k"},
        {[](Fixture& f) { f.s.k = 65; }, "invalid geometry or k"},
        {[](Fixture& f) { f.g.n_expert = 1; }, "invalid geometry or k"},
        {[](Fixture& f) { f.g.n_embd = std::numeric_limits<int64_t>::max(); }, "invalid geometry or k"},
        {[](Fixture& f) { f.s.layer_lo = 1; }, "full single-device"},
        {[](Fixture& f) { f.s.layer_hi = 47; }, "full single-device"},
        {[](Fixture& f) { f.s.qsa_ord0 = 1; }, "full single-device"},
        {[](Fixture& f) { f.s.gdn_ord0 = 1; }, "full single-device"},
        {[](Fixture& f) { --f.s.qsa_alloc; }, "full single-device"},
        {[](Fixture& f) { --f.s.gdn_alloc; }, "full single-device"},
        {[](Fixture& f) { f.s.qsa_states = nullptr; }, "full single-device"},
        {[](Fixture& f) { f.s.gdn_state = nullptr; }, "full single-device"},
        {[](Fixture& f) { f.scratch.y_miss = nullptr; }, "matching initialized scratch"},
        {[](Fixture& f) { f.scratch.probe = nullptr; }, "matching initialized scratch"},
        {[](Fixture& f) { --f.scratch.parts_bytes; }, "matching initialized scratch"},
        {[](Fixture& f) { f.s.db = nullptr; }, "doorbell host staging"},
        {[](Fixture& f) { ++f.db.k; }, "doorbell host staging"},
        {[](Fixture& f) { ++f.db.n_embd; }, "doorbell host staging"},
        {[](Fixture& f) { f.db.h_x_f = nullptr; }, "doorbell host staging"},
        {[](Fixture& f) { f.db.h_ids = nullptr; }, "doorbell host staging"},
        {[](Fixture& f) { f.db.h_weights = nullptr; }, "doorbell host staging"},
        {[](Fixture& f) { f.s.block.R = nullptr; }, "block or router buffers"},
        {[](Fixture& f) { f.s.block.mixed = nullptr; }, "block or router buffers"},
        {[](Fixture& f) { f.s.block.block_out = nullptr; }, "block or router buffers"},
        {[](Fixture& f) { f.s.moe.ids = nullptr; }, "block or router buffers"},
        {[](Fixture& f) { f.s.moe.weights = nullptr; }, "block or router buffers"},
        {[](Fixture& f) { f.q[11].host_step = nullptr; }, "QSA state 11"},
        {[](Fixture& f) { f.q[11].host_pos = nullptr; }, "QSA state 11"},
        {[](Fixture& f) { f.q[11].step = nullptr; }, "QSA state 11"},
        {[](Fixture& f) { f.q[11].pos_dev = nullptr; }, "QSA state 11"},
        {[](Fixture& f) { f.q[11].max_cells = 0; }, "QSA state 11"},
        {[](Fixture& f) { f.capturing = true; }, "inside a capture"},
    };
    for (const auto& c : cases) for (bool timed : {false, true}) {
        Fixture f; f.timed = timed; c.first(f); CHECK(!f.run());
        CHECK(f.diagnostic.find(c.second) != std::string::npos);
        CHECK(f.submitted == 0 && f.calls.empty() && f.syncs == 0);
        if (timed) check_timing(f, 0);
    }
    const std::vector<std::pair<int64_t, int32_t>> positions = {
        {-1, 0}, {0, -1}, {4096, 0}, {4095, 1}, {1, 4095},
        {std::numeric_limits<int32_t>::max(), 1}};
    for (auto p : positions) {
        Fixture f;
        if (p.first > 4096) {
            f.s.max_cells = int64_t(std::numeric_limits<int32_t>::max()) + 10;
            for (auto& q : f.q) q.max_cells = f.s.max_cells;
        }
        CHECK(!f.run(p.first, p.second));
        CHECK(f.diagnostic.find("position is outside") != std::string::npos);
        CHECK(f.submitted == 0 && f.syncs == 0);
    }
    { Fixture f; f.q[11].max_cells = 4; CHECK(!f.run(3, 1));
      CHECK(f.diagnostic.find("QSA state 11") != std::string::npos); CHECK(f.submitted == 0); }
    { Fixture f; CHECK(f.run(4095, 0)); }
    { Fixture f; CHECK(f.run(0, 4095)); }
    { Fixture f; std::string err;
      CHECK(!session_token_pool(f.tables, f.g, 0, 0, f.s, nullptr, nullptr, nullptr,
                               &f, f.scratch, &f, false, err));
      CHECK(err.find("requires parts") != std::string::npos && f.submitted == 0); }
}

void failures() {
    const std::vector<std::pair<std::string, std::string>> matrix = {
        {"pre", "pre layer 1: pre original"},
        {"D2H mixed", "layer 1: D2H mixed: injected D2H mixed"},
        {"D2H IDs", "layer 1: D2H IDs: injected D2H IDs"},
        {"D2H weights", "layer 1: D2H weights: injected D2H weights"},
        {"record staging event", "layer 1: record staging event: injected record staging event"},
        {"wait staging event", "layer 1: wait staging event: injected wait staging event"},
        {"H2D parts", "layer 1: H2D parts: injected H2D parts"},
        {"post", "post layer 1: post original"},
        {"post sync", "layer 1: post sync: injected post sync"},
        {"pool throw", "layer 1: pool callback original"},
        {"pool unknown", "layer 1: unknown exception"},
        {"launch throw", "layer 1: launch callback original"},
        {"combine throw", "layer 1: combine callback original"},
    };
    for (const auto& c : matrix) for (bool cleanup_fails : {false, true}) for (bool timed : {false, true}) {
        Fixture f; f.timed = timed; f.failure = c.first; f.cleanup_failure = cleanup_fails;
        const bool each = c.first == "post sync";
        CHECK(!f.run(0, 0, each)); CHECK(f.injected);
        std::string expected = "session_token_pool: " + c.second;
        if (cleanup_fails) expected += "; session_token_pool: final sync: cleanup original";
        CHECK(f.diagnostic == expected);
        CHECK(f.syncs == (each ? 3 : 1));
        CHECK(f.calls.back() == "sync" && f.fifo.empty() && f.executed == f.submitted);
        if (timed) check_timing(f);
        std::cout << "  failure: " << c.first << (cleanup_fails ? " + failed cleanup" : "")
                  << (timed ? " + timing" : "") << '\n';
    }
    for (bool timed : {false, true}) {
        Fixture f; f.timed = timed; f.failure = "final drain";
        CHECK(!f.run()); CHECK(f.posts == 48 && f.syncs == 1);
        CHECK(f.diagnostic == "session_token_pool: final sync: cleanup original");
        if (timed) check_timing(f);
    }
}

void layer_dump() {
    for (bool each : {false, true}) {
        Fixture f;
        std::array<float, 49 * 4 + 2> ladder;
        f.R = {1, 2, 3, 4};
        for (int t = 0; t < 2; ++t) {
            f.token = t;
            ladder.fill(-999);
            CHECK(f.run(t, 0, each, ladder.data() + 1));
            CHECK(f.fifo.empty() && ladder.front() == -999 && ladder.back() == -999);
            for (int row = 0; row < 49; ++row) {
                CHECK(ladder[1 + row * 4] == 1 + t * 1176 + row * (row + 1) / 2);
                for (int col = 1; col < 4; ++col) CHECK(ladder[1 + row * 4 + col] == col + 1);
            }
        }
    }
    for (bool input : {false, true}) for (bool timed : {false, true}) {
        Fixture f; f.timed = timed;
        std::array<float, 49 * 4> ladder{};
        f.failure = input ? "dump input" : "dump layer";
        if (input) f.layer = 1;   // inject before the first pre; production diagnostic still says layer zero
        CHECK(!f.run(0, 0, false, ladder.data()));
        CHECK(f.injected && f.diagnostic.find(f.failure) != std::string::npos);
        CHECK(f.fifo.empty() && f.syncs == 1 && f.posts == (input ? 0 : 2));
        if (timed) check_timing(f);
    }
    for (int64_t hc : {int64_t(0), std::numeric_limits<int64_t>::max()}) {
        Fixture f;
        float one = 0;
        f.g.hc = hc;
        CHECK(!f.run(0, 0, false, &one));
        CHECK(f.submitted == 0 && f.diagnostic.find("dimensions overflow") != std::string::npos);
    }
}

void timing() {
    for (bool each : {false, true}) for (bool pool : {false, true})
        for (bool hits : {false, true}) for (bool dump : {false, true}) {
            std::vector<std::string> original_calls;
            std::vector<std::pair<std::string, size_t>> original_transfers;
            std::vector<float> original_recurrent;
            std::array<float, 4> original_R{};
            std::array<float, 8> original_parts{};
            std::array<float, 49 * 4> original_ladder{};
            for (bool timed : {false, true}) {
                Fixture f; f.timed = timed; f.use_pool = pool; f.use_hits = hits;
                std::array<float, 49 * 4> ladder{};
                CHECK(f.run(3, 7, each, dump ? ladder.data() : nullptr));
                CHECK(f.diagnostic.empty() && f.posts == 48 && f.fifo.empty());
                if (!timed) {
                    original_calls = f.calls; original_transfers = f.transfers;
                    original_recurrent = f.recurrent; original_R = f.R;
                    original_parts = f.parts; original_ladder = ladder;
                    const PoolTiming::Span untouched[] = {
                        f.timing.pre_submit, f.timing.routing_d2h_submit, f.timing.routing_wait,
                        f.timing.cpu_pool, f.timing.parts_h2d_submit, f.timing.parts_wait,
                        f.timing.post_submit, f.timing.dump_d2h_submit, f.timing.final_drain,
                        f.timing.embedding_submit, f.timing.head_submit, f.timing.head_wait
                    };
                    for (const auto& span : untouched)
                        CHECK(span.ms == 0 && span.count == 0 && span.bytes == 0);
                } else {
                    CHECK(f.calls == original_calls && f.transfers == original_transfers);
                    CHECK(f.recurrent == original_recurrent && f.R == original_R);
                    CHECK(f.parts == original_parts && ladder == original_ladder);
                    check_timing(f);
                    CHECK(f.timing.pre_submit.count == 48 && f.timing.post_submit.count == 48);
                    CHECK(f.timing.routing_d2h_submit.count == 192);
                    CHECK(f.timing.routing_d2h_submit.bytes == 48 * (4 * sizeof(float) +
                          2 * sizeof(int32_t) + 2 * sizeof(float)));
                    CHECK(f.timing.parts_h2d_submit.bytes == 48 * 8 * sizeof(float));
                    CHECK(f.timing.cpu_pool.count == (pool ? 48 : 0));
                    CHECK(f.timing.dump_d2h_submit.bytes == (dump ? 49 * 4 * sizeof(float) : 0));
                }
            }
        }
    // Caller owns reset policy: reusing the POD accumulates, including final drains.
    Fixture f; f.timed = true;
    for (int t = 0; t < 2; ++t) {
        f.token = t;
        CHECK(f.run(t, 0));
        check_timing(f, t + 1);
        CHECK(f.timing.pre_submit.count == (uint64_t) (t + 1) * 48);
    }
}

void qsa_step() {
    // Fixed independent oracle at block/tail and selection-budget boundaries.
    const std::array<std::array<int32_t, 4>, 10> cases = {{
        {{0, 1, 0, 1}}, {{2, 3, 0, 3}}, {{3, 4, 1, 4}}, {{4, 5, 1, 5}},
        {{2047, 2048, 512, 2048}}, {{2048, 2049, 512, 2049}},
        {{2050, 2051, 512, 2051}}, {{2051, 2052, 513, 2051}},
        {{4095, 4096, 1024, 2051}}, {{262143, 262144, 65536, 2051}}
    }};
    for (auto expected : cases) {
        std::array<int32_t, 6> h = {{-91, -7, -7, -7, -7, -92}};
        qsa_step_fill(h.data() + 1, expected[0], qsa_real_shapes());
        for (int i = 0; i < 4; ++i) CHECK(h[i + 1] == expected[i]);
        CHECK(h.front() == -91 && h.back() == -92);
    }
    auto custom = qsa_real_shapes(); custom.idx_block = 8; custom.idx_top_k = 16;
    std::array<int32_t, 4> h{};
    qsa_step_fill(h.data(), 31, custom);
    CHECK((h == std::array<int32_t, 4>{{31, 32, 4, 23}}));
    qsa_step_fill(nullptr, -1, custom);  // documented early null return
    h.fill(-7);
    try { qsa_step_fill(h.data(), -1, custom); CHECK(false); }
    catch (const DeviceError& e) { CHECK(std::string(e.what()) == "qsa: qsa_step_fill: pos < 0"); }
    for (auto v : h) CHECK(v == -7);
}
}  // namespace

int main(int argc, char** argv) {
    CHECK(argc == 2);
    const std::string mode = argv[1];
    if (mode == "orchestration") orchestration();
    else if (mode == "guards") guards();
    else if (mode == "failures") failures();
    else if (mode == "qsa-step") qsa_step();
    else if (mode == "layer-dump") layer_dump();
    else if (mode == "timing") timing();
    else CHECK(false);
    std::cout << "PASS " << mode << '\n';
}
