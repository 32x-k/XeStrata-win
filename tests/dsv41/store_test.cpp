// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Strata. ExpertStore (src/dsv41/ds4_store.hpp) on real expert files, no device: the bytes a slot holds equal the file's, the
// unbuffered reads' rate, the policy's hits on a skewed random access stream.
//   ds4_store_test <gguf> <layer0 gate off> <layer0 up off> <layer0 down off> <expert bytes (gate/up/down)> [budget GB] [policy 0|1|2] [accesses]
// (offsets from the GGUF's tensor directory: tools/dsv41 header scan; the three tensors of a layer are [384][...] so expert e of
// a matrix lies at off + e * bytes)
#include "ds4_store.hpp"
#include "ds4_engram_io.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>

using namespace strata::ds4;

// The CTest path creates its own small file and checks bytes, eviction, prefetch cancellation and
// shrinking. It needs neither model weights nor timing measurements.
static int selftest() {
    namespace fs = std::filesystem;
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path dir = fs::temp_directory_path() / ("ds4-store-test-" + std::to_string(id));
    if (!fs::create_directory(dir)) return 1;
    struct Temp {
        fs::path path;
        ~Temp() {
            try { std::error_code ec; fs::remove_all(path, ec); }
            catch (...) { std::fputs("store test: temporary file cleanup failed\n", stderr); }
        }
    } temp{dir};
    const fs::path path = dir / "experts.bin";
    const int ne = 12;
    const size_t mb = (size_t) os::kDirectAlign + 37;
    const uint64_t offsets[3] = {13, 13 + ne * mb, 13 + 2 * (size_t) ne * mb};
    std::vector<uint8_t> bytes((size_t) offsets[2] + ne * mb + (size_t) os::kDirectAlign);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (uint8_t) ((i * 37 + i / 251) & 255);
    {
        std::ofstream f(path, std::ios::binary);
        f.write((const char*) bytes.data(), (std::streamsize) bytes.size());
        if (!f) return 1;
    }
    const os::fd_t direct = os::open_direct(path.string()), buffered = os::open_par(path.string());
    if (buffered == os::kBadFd) { os::close_fd(direct); return 1; }
    int bad = 0;
    {
        EgRowCache cache;
        cache.init(2, mb);
        std::memset(cache.put(1), 17, mb);
        std::memset(cache.put(2), 29, mb);
        const uint8_t* hit = cache.find(1);
        if (!hit || hit[0] != 17 || hit[mb - 1] != 17) ++bad;
        std::memset(cache.put(3), 41, mb);
        if (cache.find(2) != nullptr || cache.find(1) == nullptr || cache.size() != 2) ++bad;
        cache.init(0, mb);
        if (cache.size() != 0 || cache.find(1) != nullptr || cache.put(1) != nullptr) ++bad;
    }
    // Retained and demand reads must preserve packed rows, interleaved MXFP8 scales, and separate scales. The
    // requested union fits exactly; a smaller budget and a truncated table must both leave the demand path usable.
    for (int mode = 0; mode < 3; ++mode) {
        EgReader reader;
        const fs::path table = dir / "engram.bin";
        {
            std::ofstream f(table, std::ios::binary);
            f.write((const char*) bytes.data(), (std::streamsize) bytes.size());
            if (!f) return 1;
        }
        const std::vector<std::string> paths = {table.string()};
        const size_t kept = reader.preload(paths, mode == 2 ? std::vector<EgSpan>{{0, bytes.size() - 2, 5}} :
            std::vector<EgSpan>{{0, 13, 90}, {0, 23, 90}}, mode == 0 ? 100 : mode == 1 ? 99 : 100);
        if (kept != (mode == 0 ? 100u : 0u)) ++bad;
        if (mode == 0 && !fs::remove(table)) ++bad;   // retained bytes must not silently fall back to a file read
        reader.start(paths, 2);
        EgBatch batch;
        uint8_t packed[41] = {}, codes[64] = {}, scales[2] = {}, split[31] = {}, split_scale[3] = {};
        EgRead r;
        r.file = 0; r.off = 19; r.n = sizeof packed; r.dst = packed;
        batch.reads.push_back(r);
        r.off = 37; r.n = 66; r.dst = codes; r.scales = scales; r.mx8 = true;
        batch.reads.push_back(r);
        r.off = 16; r.n = sizeof split; r.dst = split; r.scales = split_scale; r.mx8 = false;
        r.file2 = 0; r.off2 = 109; r.n2 = sizeof split_scale;
        batch.reads.push_back(r);
        reader.submit(batch);
        reader.wait(batch);
        if (batch.ok != std::vector<uint8_t>({1, 1, 1}) || std::memcmp(packed, bytes.data() + 19, sizeof packed) ||
            std::memcmp(split, bytes.data() + 16, sizeof split) ||
            std::memcmp(split_scale, bytes.data() + 109, sizeof split_scale)) ++bad;
        for (size_t k = 0; k < 2; ++k)
            if (scales[k] != bytes[37 + k * 33] || std::memcmp(codes + k * 32, bytes.data() + 38 + k * 33, 32)) ++bad;
        reader.stop();
    }
    {
        EgReader reader;
        const std::vector<std::string> paths = {path.string()};
        if (reader.preload(paths, {{0, UINT64_MAX - 1, 4}}, 4) != 0 ||
            reader.preload(paths, {{-1, 0, 1}}, 1) != 0 || reader.preload(paths, {{0, 0, 1}}, 0) != 0) ++bad;
    }
    ExpertStore::LayerGeo geo;
    geo.n_exp = ne;
    for (int role = 0; role < 3; ++role) {
        geo.file[role] = 0;
        geo.off[role] = offsets[role];
        geo.stride[role] = geo.bytes[role] = mb;
        geo.tr[role] = geo.blob;
        geo.blob += ((mb + 255) & ~(uint64_t) 255) + ExpertStore::kSlack;
    }
    const size_t slot_bytes = (geo.blob + 2 * ExpertStore::kSlack + os::kDirectAlign - 1) & ~(size_t) (os::kDirectAlign - 1);
    for (int policy : {0, 1}) {
        ExpertStore store;
        ExpertStore::Cfg cfg;
        cfg.threads = 2;
        cfg.budget_bytes = 4 * slot_bytes;
        cfg.floor_bytes = 0;
        cfg.policy = (ExpertStore::Policy) policy;
        cfg.ram_avail = [] { return os::ram_avail(); };
        if (!store.init(cfg, {geo}, {direct}, {buffered})) { ++bad; continue; }
        auto check = [&](int e0, int e1) {
            const int es[2] = {e0, e1};
            std::vector<char> ok;
            store.ensure(0, es, 2, ok);
            for (int j = 0; j < 2; ++j) {
                if (j >= (int) ok.size() || !ok[(size_t) j]) { ++bad; continue; }
                const int slot = store.find(0, es[j]);
                if (slot < 0) { ++bad; continue; }
                for (int role = 0; role < 3; ++role)
                    if (std::memcmp(store.mat(slot, role), bytes.data() + offsets[role] + (size_t) es[j] * mb, mb) != 0) ++bad;
            }
            store.release(0, es, 2, ok);
        };
        for (int e = 0; e < ne; e += 2) check(e, e + 1);
        check(0, 1);
        store.prefetch(0, 8);
        store.prefetch(0, 9);
        store.cancel_stale(0);
        // Drain cancelled guesses before demanding their rows again. An ensure overlapping a
        // cancelled read may return false, leaving the caller to use its ordinary fallback.
        (void) store.preload({}, 0);
        check(8, 9);
        (void) store.shrink_chunks(store.slots_allocated());
        check(0, 11);
        if (store.stats().evictions.load() == 0 || store.stats().fails.load() != 0) ++bad;
    }
    os::close_fd(buffered);
    os::close_fd(direct);
    std::printf("expert store: %s (%d errors)\n", bad ? "FAIL" : "OK", bad);
    return bad ? 1 : 0;
}
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 6) {
        std::printf("usage: ds4_store_test <gguf> <gate off> <up off> <down off> <matrix bytes> [budget GB] [policy] [accesses]\n");
        return 2;
    }
    const std::string path = argv[1];
    const uint64_t offs[3] = {std::strtoull(argv[2], nullptr, 10), std::strtoull(argv[3], nullptr, 10), std::strtoull(argv[4], nullptr, 10)};
    const uint64_t mb = std::strtoull(argv[5], nullptr, 10);
    const double gb = argc > 6 ? std::atof(argv[6]) : 1.0;
    const int policy = argc > 7 ? std::atoi(argv[7]) : 0;
    const int n_acc = argc > 8 ? std::atoi(argv[8]) : 2000;
    const os::fd_t fd = os::open_ro(path), dfd = os::open_direct(path), pfd = os::open_par(path);
    if (fd == os::kBadFd || pfd == os::kBadFd) {
        std::printf("cannot open %s\n", path.c_str());
        return 2;
    }
    std::printf("direct handle: %s\n", dfd != os::kBadFd ? "yes" : "NO (buffered)");
    ExpertStore::LayerGeo geo;
    geo.n_exp = 384;
    uint64_t at = 0;
    for (int r = 0; r < 3; ++r) {
        geo.file[r] = 0;
        geo.off[r] = offs[r];
        geo.stride[r] = mb;
        geo.bytes[r] = mb;
        geo.tr[r] = at;   // the engine's blob layout with the store's 8192-byte gaps
        at += ((mb + 255) & ~(uint64_t) 255) + ExpertStore::kSlack;
    }
    geo.blob = at;
    ExpertStore st;
    ExpertStore::Cfg cfg;
    cfg.budget_bytes = (size_t) (gb * 1e9);
    cfg.threads = std::getenv("STORE_TEST_THREADS") ? std::atoi(std::getenv("STORE_TEST_THREADS")) : 8;
    cfg.policy = (ExpertStore::Policy) policy;
    cfg.floor_bytes = 4ull << 30;
    cfg.ram_avail = [] { return os::ram_avail(); };
    if (!st.init(cfg, {geo}, {dfd}, {pfd})) {
        std::printf("init failed\n");
        return 2;
    }
    std::printf("slot %.2f MB, up to %zu slots\n", st.slot_bytes() / 1e6, st.slots_max());
    // a skewed stream: 70% of the accesses from the 64 hottest experts, the rest uniform; 6 a "token"
    std::mt19937_64 rng(7);
    auto pick = [&] { return rng() % 100 < 70 ? (int) (rng() % 64) : (int) (rng() % 384); };
    std::vector<uint8_t> ref(mb);
    int bad = 0;
    long checked = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int a = 0; a < n_acc; ++a) {
        int es[6];
        for (int i = 0; i < 6; ++i) {
            es[i] = pick();
            for (int j = 0; j < i; ++j)
                if (es[j] == es[i]) es[i] = (es[i] + 1 + j) % 384;
        }
        std::vector<char> ok;
        st.ensure(0, es, 6, ok);
        if (a % 17 == 0 && !std::getenv("STORE_TEST_NOCMP"))   // a sample compared with the file read plainly
            for (int i = 0; i < 6; ++i)
                if (ok[(size_t) i])
                    for (int r = 0; r < 3; ++r) {
                        const int s = st.find(0, es[i]);
                        os::pread_full(fd, ref.data(), mb, offs[r] + (uint64_t) es[i] * mb);
                        ++checked;
                        if (s < 0 || std::memcmp(st.mat(s, r), ref.data(), mb) != 0) ++bad;
                    }
        st.release(0, es, 6, ok);
    }
    // guesses queued behind demand, cancelled, and a shrink: the store must still give the file's bytes
    for (int round = 0; round < 3; ++round) {
        for (int e = 100; e < 140; ++e) st.prefetch(0, e);
        st.cancel_stale(0);
        int es[6] = {5 + round, 17, 200, 201, 202, 250 + round};
        std::vector<char> ok;
        st.ensure(0, es, 6, ok);
        for (int i = 0; i < 6; ++i)
            if (ok[(size_t) i]) {
                const int sl = st.find(0, es[i]);
                for (int r = 0; r < 3; ++r) {
                    os::pread_full(fd, ref.data(), mb, offs[r] + (uint64_t) es[i] * mb);
                    ++checked;
                    if (sl < 0 || std::memcmp(st.mat(sl, r), ref.data(), mb) != 0) ++bad;
                }
            }
        st.release(0, es, 6, ok);
        std::this_thread::sleep_for(std::chrono::milliseconds(800));   // (the guesses read by now)
        const size_t g = st.shrink_chunks(1);
        std::printf("round %d: shrank %zu chunk(s), %zu slots left, cancelled %lld\n", round, g, st.slots_allocated(), (long long) st.stats().cancelled.load());
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto& s = st.stats();
    std::printf("%d accesses of 6 experts in %.2f s: hits %lld misses %lld evictions %lld fails %lld; read %.2f GB (%.2f GB/s); waited %.2f s; "
                "compared %ld matrices, %d differ\n",
                n_acc, dt, (long long) s.hits.load(), (long long) s.misses.load(), (long long) s.evictions.load(), (long long) s.fails.load(),
                s.bytes_read.load() / 1e9, s.bytes_read.load() / 1e9 / dt, s.wait_ns.load() / 1e9, checked, bad);
    return bad == 0 ? 0 : 1;
}
