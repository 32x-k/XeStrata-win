// Host test for include/strata/prefill/copy_events.hpp (#4005): the copied-event rules of the streamed prompt walk.
#include "strata/prefill/copy_events.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::prefill;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    check(copy_event_every_from(0) == 1 && copy_event_every_from(1) == 1 && copy_event_every_from(-5) == 1, "unset / 0 / 1 / negative: every entry");
    check(copy_event_every_from(2) == 2 && copy_event_every_from(16) == 16 && copy_event_every_from(64) == 64, "2..64 as given");
    check(copy_event_every_from(65) == 64 && copy_event_every_from(100000) == 64, "above 64 is 64");

    // default: every entry records (the walk then waits on copied[slot], as before the switch)
    for (size_t total : {1u, 2u, 7u, 100u})
        for (size_t i = 0; i < total; ++i) check(copy_event_due(i, total, 1), "every == 1 records every entry");

    // sparse: the walk's issuer/consumer as a model.  The issuer records per copy_event_due (and, when it must wait for
    // the consumer, the entry before the one it is about to issue); the consumer for entry k waits on
    // copy_event_entry(k, recorded, ...).  Every wait must name a recorded entry at or after k, and the ring must
    // never need an event that a later entry has overwritten.
    std::srand(4005);
    for (int every : {2, 3, 5, 16, 64})
        for (int ring : {2, 4, 8, 96})
            for (size_t total : {1u, 5u, 33u, 300u}) {
                std::vector<long long> rec_of((size_t) ring, -1);
                size_t recorded = 0, issued = 0, consumed = 0;
                long long last_rec = -1;
                bool ok = true;
                auto record = [&](size_t i) {
                    if (rec_of[i % (size_t) ring] >= 0 && (size_t) rec_of[i % (size_t) ring] >= consumed) ok = false;   // overwrote a live event
                    rec_of[i % (size_t) ring] = (long long) i;
                    recorded = i + 1;
                    last_rec = (long long) i;
                };
                auto cover = [&](size_t idx) { if (idx > 0 && last_rec < (long long) idx - 1) record(idx - 1); };
                size_t guard = 0;
                while (consumed < total && ok && ++guard < 100000) {
                    const bool can_issue = issued < total && issued < consumed + (size_t) ring;
                    const bool can_consume = consumed < issued;
                    const bool pick_issue = can_issue && (!can_consume || std::rand() % 2);
                    if (pick_issue) {
                        const size_t idx = issued;
                        if (std::rand() % 4 == 0) cover(idx);   // the issuer had to wait (ring full or stager)
                        if (copy_event_due(idx, total, every)) record(idx);
                        ++issued;
                    } else if (can_consume) {
                        if (issued == total || true) {
                            // a consumer needs recorded > k; when the issuer is blocked it has covered what it issued
                            if (recorded <= consumed) {
                                if (issued < total && !(issued < consumed + (size_t) ring)) cover(issued);
                                if (issued == total) cover(issued);
                            }
                            if (recorded <= consumed) { continue; }   // would spin until the issuer records
                            const size_t r = copy_event_entry(consumed, recorded, [&](size_t e) { return rec_of[e % (size_t) ring] == (long long) e; });
                            if (r < consumed || r >= recorded || rec_of[r % (size_t) ring] != (long long) r) ok = false;
                            ++consumed;
                        }
                    } else {
                        break;
                    }
                }
                char what[128];
                std::snprintf(what, sizeof what, "sparse every=%d ring=%d total=%zu: every wait names a live recorded entry", every, ring, total);
                check(ok && consumed == total, what);
            }

    if (g_fail == 0) std::printf("copy_events_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
