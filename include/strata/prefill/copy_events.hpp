// include/strata/prefill/copy_events.hpp - which streamed expert copies get a "copied" event (#4005).
//
// The streamed prompt walk issues one host-to-device copy per expert entry on the copy stream and the compute stream
// waits for it.  By default every entry records its own event (every == 1) and the compute stream waits on the event of
// the entry it needs.  STRATA_PREFILL_COPY_EVENT_EVERY=N (tester switch) records every N-th entry only, and the
// compute stream waits on the first recorded entry at or after the one it needs: the copy stream is in order, so that
// event covers it.  The two rules are pure functions so a host test can check them without a GPU.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill {

/// The value of STRATA_PREFILL_COPY_EVENT_EVERY: 1 (every entry) unless the text is a number above 1; at most 64.
inline int copy_event_every_from(int parsed) { return parsed > 1 ? (parsed < 64 ? parsed : 64) : 1; }

/// Entry `idx` of `total` records its copied event right after its copy: always with every == 1; else every N-th entry
/// and the last one.
inline bool copy_event_due(size_t idx, size_t total, int every) {
    return every <= 1 || (idx + 1) % (size_t) every == 0 || idx + 1 == total;
}

/// The entry whose recorded event a consumer waits on for entry `k`, when the entries [0, recorded) have had their
/// events decided (`recorded` = 1 + the last recorded entry) and `is_recorded(r)` says whether entry r has one.  The
/// first recorded entry at or after k; the last recorded entry (recorded - 1) when none is found, which covers k as
/// well.  Precondition: k < recorded.
template <class IsRecorded>
inline size_t copy_event_entry(size_t k, size_t recorded, IsRecorded is_recorded) {
    for (size_t r = k; r < recorded; ++r)
        if (is_recorded(r)) return r;
    return recorded - 1;
}

}  // namespace strata::prefill
