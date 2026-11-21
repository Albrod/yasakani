// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// C API of the drafter add-on (drafter of Inkling-2 Mini, mtp.safetensors),
// in the same DLL as the engine (capability "drafter", ink2_draft_abi() = 1).
//
// The drafter lives inside the engine handle. Once loaded, every pass of the engine
// (ink2_step, ink2_feed, ink2_feed_many, ink2_step_many) also commits the drafter's
// context of its rows on the device (features of the visits feat_visits and the final
// hidden), so the drafter is always in step with ink2_position(); ink2_truncate and
// ink2_reset need nothing extra.
// ink2_spec_pass runs one draft call (recorded graph, its tokens go straight into the
// rows of the pass on the device) and the verification pass of depth + 1 rows.
#pragma once
#include "ink2_api.h"

#define INK2_DRAFT_ABI 1

typedef struct Ink2DraftConfig {
  int hidden, layers, heads, kv_heads, head_dim, inter;
  int window, block, rank, w1_rows;
  int local[8];        // 1 = sliding-window draft layer
  int feat_visits[3];  // engine visits whose OUTPUT is a feature (= the inputs of target visits 5, 23, 41)
  float eps, rope_theta, softcap;
} Ink2DraftConfig;

INK2_API int ink2_draft_abi(void);
// Drops a loaded drafter and starts a new one (tensors follow, then finalize).
INK2_API int ink2_draft_create(void* h, const Ink2DraftConfig* cfg);
// Tensors (names of ink2_drafter_native.py): BF16 weights, "fuse_alpha" F32 [4, 4],
// "fold16" BF16 and / or "fold32" F32 [vocab, rank] (U * hidden_out), "fold_b" F32 [vocab].
INK2_API int ink2_draft_set_tensor(void* h, const char* name, const void* data, long long nbytes);
// Checks, allocates, frees the recorded graphs of the engine (they are recorded again with
// the commit). The positions seen before have no drafter context: reset and feed again.
INK2_API int ink2_draft_finalize(void* h);
INK2_API int ink2_draft_unload(void* h);
INK2_API int ink2_draft_loaded(void* h);  // 1 = finalized drafter
// "fold": 16 = BF16 folded head (default when present), 32 = float32; "blocks": blocks of
// the head kernel; "profile": 1 = direct launches, device synchronised between the parts.
INK2_API int ink2_draft_set_option(void* h, const char* name, int value);
// One draft call of `depth` (1..7) tokens for `anchor` at position ink2_position() and the
// verification pass [anchor, d_1 .. d_depth] with the arg-max of every row (argmax_out
// [depth + 1], drafts_out [depth]). Returns the accepted count a (greedy rule); with
// truncate != 0 the rejected rows are dropped (position + 1 + a), else position + depth + 1.
// The logits of every row stay on the device (ink2_get_row INK2_GET_LOGITS).
INK2_API int ink2_spec_pass(void* h, int anchor, int depth, int* drafts_out, int* argmax_out, int truncate);
// A draft call alone for `anchor` at position `pos` (no state changes): the proposals.
INK2_API int ink2_draft_propose(void* h, int anchor, int pos, int depth, int* drafts_out);
// Debug: commits the drafter context of positions p0 .. p0 + n - 1 from given features
// feats [n, 4, hidden] (n <= max_rows); the engine is not touched.
INK2_API int ink2_draft_commit_features(void* h, const float* feats, int p0, int n);
// Time of a draft call of `depth` (reps calls, device synchronised): out[0] = ms per call
// (as used: recorded graph on cuda); out[1..3] = ms of its parts block / base logits /
// Markov head (direct launches, synchronised between the parts); out[4] = launches per call.
INK2_API int ink2_draft_time(void* h, int depth, int reps, double* out5);
INK2_API long long ink2_draft_memory(void* h);  // bytes allocated by the drafter
// Debug buffers of the last draft call: 0 = block hidden after the final norm [8, hidden],
// 1 = base logits [depth, vocab], 2 = f_rnn of position `index` [hidden].
INK2_API int ink2_draft_get(void* h, int what, int index, float* out, long long nfloats);
