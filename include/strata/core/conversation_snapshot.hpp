// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

#include <string>

namespace strata::core {

// Caller synchronizes the device before saving, and after restoring all layers.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
// A nonzero unchanged_tokens is valid only for storage retained from an image
// actually restored into this session, bounded by every subsequent rewrite.
// Equal token IDs alone do not establish that its K/V bytes are unchanged.
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error,
                          int64_t unchanged_tokens = 0, size_t* reused_bytes = nullptr);
bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                   int64_t upto, bool include_index, size_t& bytes, std::string& error);
// No device calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error);
// The first `upto` tokens of an image captured at `captured` tokens: the leading bytes of each buffer (cells come
// first in the identity layout, as conversation_kv_save takes them), the image untouched.  Cells and indexer rows
// past `upto` in the restored pages hold the image's later tokens until a request writes them, as after a whole
// restore followed by a rewind to a checkpoint.
bool conversation_kv_restore_prefix(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                    int64_t captured, int64_t upto, bool include_index, std::string& error);
// Diagnostic read-back after a synchronized restore. Uses 64 KiB of stack
// workspace, compares authoritative bytes and resident draft-ring pages, and
// fingerprints the authoritative payload only. Never changes model state.
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, uint64_t& fingerprint, std::string& error);
// Disk save without a host copy: the layer's authoritative K/V as a streamed source (same bytes and metadata as
// conversation_kv_save).  Caller synchronizes the device first and keeps the state untouched while it is read.
bool conversation_kv_source(SessionKvSource& source, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, std::string& error);
// The bytes of each K/V part (k, v, k_scale, v_scale, pooled) a snapshot of `upto` tokens holds: no state read.
bool conversation_kv_part_sizes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                                std::array<uint64_t, 5>& sizes, std::string& error);
// Disk sessions: the read limits this session can ever restore - its geometry and layer range, at most
// min(max_tokens, its cells) tokens, `max_checkpoints` checkpoints, the exact running-state sizes and the K/V part
// sizes at that many tokens - so session_file_read refuses an oversized or foreign file before it allocates.
bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& session, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
/// Whole-model sizes: `gdn` covers every GDN layer, the indexer sizes are per QSA layer.
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
/// The same for the layers `ss` owns (its carve, upstream #216): `gdn` holds its GDN rows only.
bool conversation_session_sizes(const ModelGeometry& g, const SessionState& ss, ConversationStateSizes& sizes,
                                std::string& error);
bool conversation_checkpoint_validate(const ConversationCheckpoint& checkpoint, const SessionState& session,
                                      const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_save(ConversationCheckpoint& checkpoint, const SessionState& session,
                                  const ModelGeometry& g, std::string& error);
/// STRATA_CKPT_ASYNC=1 (opt-in): conversation_checkpoint_save (and _stage) copies the GDN state into a VRAM stage and
/// moves it into the checkpoint's host vector (the staging) on a background thread, so the copy over the link overlaps
/// what follows (the rest of the prompt, the reply).  This waits for that copy; every reader of a checkpoint's bytes
/// calls it first (restore, snapshot save and restore, the parked copies, the session file, the serve loop at each
/// request, unstage).  No-op when nothing is in flight.
/// `only` (non-null): wait only when the copy in flight is into those bytes (a checkpoint's `gdn.data()`, the staging).
void conversation_checkpoint_join(const void* only = nullptr);
bool conversation_checkpoint_restore(const ConversationCheckpoint& checkpoint, SessionState& session,
                                     const ModelGeometry& g, std::string& error);
/// A checkpoint's save in two steps: `stage` copies the running state into `staging` (pinned host memory of
/// conversation_checkpoint_staged_bytes; the caller has synchronized the device) as one batch of copies on `stream`
/// and waits for it, `unstage` later fills `checkpoint` (its ids set) from it on the host, as
/// conversation_checkpoint_save would have.  The reply waits only for the copies over the link.
size_t conversation_checkpoint_staged_bytes(const SessionState& session, const ModelGeometry& g);
/// `dev_tag` (optional): the tag of the GDN state's VRAM copy (STRATA_CKPT_VRAM_SLOTS; 0: none), for the checkpoint.
bool conversation_checkpoint_stage(void* staging, void* stream, const SessionState& session, const ModelGeometry& g,
                                   size_t tokens, std::string& error, uint64_t* dev_tag = nullptr);
bool conversation_checkpoint_unstage(ConversationCheckpoint& checkpoint, const void* staging,
                                     const SessionState& session, const ModelGeometry& g, std::string& error);

struct ConversationView {
    const std::vector<int32_t>& ids;
    const std::vector<ConversationImageKey>& images;
    const std::vector<ConversationCheckpoint>& checkpoints;
    bool cvec;
};
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error);
// The capture estimate includes retained capacity and transient segment directories;
// only estimate - reuse.bytes() requires additional physical RAM. Capture consumes
// the uniquely owned reusable buffers, including on failure.
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
// Disk save without capturing the K/V on the host: `meta` gets everything but the K/V (running state copied,
// checkpoints as given by the view), `sources` one streamed source per QSA layer then the draft.  Caller has
// synchronized and must not run the session until the file is written.
bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& session,
                                   const ModelGeometry& g, const QsaState& draft, std::string& error);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
enum class ConversationRestore { restored, invalid, transfer_failed };
// Invalid images are rejected before any device call/write. Transfer failure may
// leave partial state: caller MUST NOT continue inference from that session.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);
// The K/V of the image's first `upto` tokens (main layers and the draft layer), not its running state: the caller
// mounts a checkpoint taken at `upto`.  For a checkpoint of a parked conversation that a new conversation starts
// with, while the parked one stays where it is (ConversationCache::borrows).  Same failure contract as above.
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& session, const ModelGeometry& g,
                                                          const QsaState& draft, std::string& error);

// A layer split (docs/MULTIGPU.md): each GPU's session (its carve) on its device, in layer order; the draft layer is
// on the last one's.  The image is the one-GPU image's layout - the QSA layers' K/V in layer order, then the draft's -
// with each later GPU's running state in `live.stage_parts` (and every checkpoint's).  Device calls run on the device
// that holds the state; the caller does not synchronize.
struct ConversationStage {
    SessionState* session = nullptr;
    int device = 0;
};
bool conversation_split_bytes(const ConversationView& view, const std::vector<ConversationStage>& stages,
                              const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_split_save(SavedConversation& image, const ConversationView& view,
                             const std::vector<ConversationStage>& stages, const ModelGeometry& g,
                             const QsaState& draft, std::string& error);
bool conversation_split_validate(const SavedConversation& image, const std::vector<ConversationStage>& stages,
                                 const ModelGeometry& g, const QsaState& draft, std::string& error);
ConversationRestore conversation_split_restore(const SavedConversation& image,
                                               const std::vector<ConversationStage>& stages, const ModelGeometry& g,
                                               const QsaState& draft, std::string& error);

} // namespace strata::core
