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

// Version 2: the drafter matrices may be given packed (nvfp4 / mxfp8 / q3, capability bits
// INK2_CAP_DRAFTER_PACKED and INK2_CAP_Q3 of ink2_capabilities), and the folded head too ("foldp").
#define INK2_DRAFT_ABI 2

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
// Version 2: every matrix (L<l>.wq wk wv wo wkc wvc wg wu wd, w1, rnn_w) and the folded head "foldp" [vocab, rank]
// may be packed: "<name>" = the code bytes, "<name>_scale" = the block scales, "<name>_factor" = float32 [1]
// (nvfp4: amax / denominator; q3: the factor), "<name>_format" = int32 [1] (1 nvfp4, 2 mxfp8, 3 q3; without it:
// nvfp4 when a factor is given, else mxfp8). Layouts: GemvJob in ink2_kargs.h.
INK2_API int ink2_draft_set_tensor(void* h, const char* name, const void* data, long long nbytes);
// Checks, allocates, frees the recorded graphs of the engine (they are recorded again with
// the commit). The positions seen before have no drafter context: reset and feed again.
INK2_API int ink2_draft_finalize(void* h);
INK2_API int ink2_draft_unload(void* h);
INK2_API int ink2_draft_loaded(void* h);  // 1 = finalized drafter
// "fold": 4 = packed folded head (version 2; default when present), 16 = BF16 folded head (default when present
// and no packed one), 32 = float32; "mrows" (version 2): rows per block of the packed 8-row products of the
// draft call, 0 (the engine's rule: 2 for passes of up to 8 rows) / 2 / 4 / 8 / 16 (same results); "blocks": blocks of
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
INK2_API int ink2_draft_packed(void* h);         // version 2: drafter matrices held packed (0 = all BF16)
// Debug buffers of the last draft call: 0 = block hidden after the final norm [8, hidden],
// 1 = base logits [depth, vocab], 2 = f_rnn of position `index` [hidden].
INK2_API int ink2_draft_get(void* h, int what, int index, float* out, long long nfloats);

// confidence cut: confidence cut. mode 0 = off, 1 = top probability of the draft (after the soft cap), 2 = its top1 - top2
// logit, 3 = the running product of the top probabilities; a draft j whose value is below thr7[j - 1] and every later
// draft are not verified. Only ink2_spec_pass2 uses the rule (ink2_spec_pass / ink2_draft_propose never cut).
// ink2_draft_get(what = 3): the confidence of the drafts of the last draft call, [8][4] = top1, top2, log-sum-exp, product.
INK2_API int ink2_draft_set_cut(void* h, int mode, const float* thr7);
INK2_API int ink2_spec_pass2(void* h, int anchor, int depth, int* drafts_out, int* argmax_out, int truncate, int* info_out);
