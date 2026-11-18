// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// EXPERIMENTAL: the multi-stream pass (several requests in one pass of one engine).  The library exports these entries, but they are
// not part of the interface in include/yskn_abi.h: names, arguments and behavior may change in any release.  Callers look them up at
// run time and fall back to one request at a time when they are missing.
//
// One engine (the carrier) holds n_light stream slots: the recurrent states, their histories and the KV rows of one request each.
// A request is read by the carrier itself (its prompt pass, e.g. yskn_spec_generate with one token), then yskn_ms_load copies that
// state into a slot.  Each pass carries at most 16 rows; the rows of a slot are next to each other: its last token first, then its
// drafts.  Every entry returns 0, or -1 with yskn_last_error().  The single-request entries of the carrier (yskn_reset,
// yskn_spec_generate, yskn_verify, ...) first finish the pending work of the slots (yskn_ms_flush).
//
// The carrier's live sequence: with prompt snapshots (spec_generate snap = 1) a read whose prompt extends the tokens the carrier holds
// continues them (snapshot mode 1); otherwise it restores a saved state (mode 2) or reads from the start (mode 0).  The ids of a read
// may depend on how its tokens were read: where the read starts, where its passes are cut, and how the restored or continued tokens
// were read before.  A read from a saved state on the absolute prefill grid runs the passes a read from the start runs; a state at a
// chat start is not on that grid, and mode 1 continues tokens computed in other passes.  So two runs of a request give the same ids
// when they read it, and the requests before it in its conversation, the same way (same modes and starts: statistics 41 and 42 of
// yskn_spec_generate); whether a different start changes the ids is not settled for every GPU and setting.  So that a read never
// continues a stream that has moved, or one that only happens to match the head of the prompt:
//   - yskn_ms_bind2, yskn_ms_load, yskn_ms_draft and yskn_ms_step2 drop the live sequence (no mode 1 after them), and so does every
//     single-request entry that runs a pass or changes the state (yskn_reset, yskn_prefill, yskn_verify, yskn_commit, also a refused
//     one, ...); yskn_spec_generate with snap = 1 sets it at its end;
//   - yskn_ms_commit, yskn_ms_flush (also the one the single-request entries run first), yskn_ms_set_sampling, yskn_ms_seeds,
//     yskn_ms_opt and the queries keep it;
//   - yskn_ms_unload sets it to the stream it hands back; yskn_live_drop drops it (the saved states stay: mode 2 still works).
// A read on the carrier right after yskn_ms_unload (no pass in between) therefore continues the stream in mode 1, as the same request
// on its own would continue after its earlier stage.  This needs a slot loaded right after the carrier's read of the request
// (yskn_spec_generate with snap = 1, then yskn_ms_load with nothing in between): yskn_ms_unload checks the held ids against the
// source's live sequence, and without one it leaves the carrier without a live sequence (the next read is from the start).
//
// Back to one request at a time (a stream leaves the pass, or a stage that runs on the carrier): yskn_ms_unload(slot, ids) with ids
// = the tokens the slot holds, then yskn_spec_generate on the carrier with the prompt ids + [frontier] (the token the stream emitted
// last, not in its KV yet), either
//   - the usual form (as the package continues a request in a second call: prompt = the ids so far): the carrier reads the frontier
//     (one row) and goes on; or
//   - the frontier form, cfg[46] = 1 (n_cfg >= 47): nothing is read, the first pass checks the frontier with its drafts, as the next
//     pass of a stream that was never moved would; out_ids[0] is the token after the frontier and statistic 30 + 1 = the ids written,
//     as in the usual form (it may be 0, e.g. when the call stops before its first pass).  It needs snap = 1 and the carrier's live
//     sequence = the prompt without its last token (what yskn_ms_unload leaves), and the last prompt token must be the frontier
//     itself: the carrier keeps the token its stream emitted last with the live sequence (yskn_spec_generate with snap = 1 sets it
//     at its end, yskn_ms_unload hands over the slot's), and another token is refused (nothing changes; the live sequence stays).
// Either way the ids are those of the same request that never left the carrier (the result of a row does not depend on the other
// rows of its pass, which the multi-stream pass relies on as well); yskn_ms_load(slot, carrier) moves it back.
//
// The frontier of a slot (the token its stream emitted last, not in its KV yet) is kept the same way: yskn_ms_load takes the
// source's when the source has a live sequence (else it is not known until the slot's first pass), yskn_ms_step2 sets it to
// out_argmax of the stream's first row, yskn_ms_commit to out_argmax after the last accepted draft (a pass left uncommitted: no
// draft accepted).  While it is known, yskn_ms_draft and yskn_ms_step2 refuse a pass whose first row of the slot is another token
// (nothing changes; an open pass can still be committed).  A token the stream did not emit (a forced token, a stop the caller
// adds) goes through the carrier: yskn_ms_unload, a read of the prompt ids + that token, yskn_ms_load.
#pragma once
#include "yskn_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
/* n must be 0 (engines as slots are a test-only form); n_light = number of slots (1..64).  Refused (nothing changes) for settings the
   pass is not built for, with the reason in yskn_last_error (the codes of yskn_ms_info out[1]): KV not in KV5 (8-bit KV,
   YSKN_TB_CAP=0, or 16-bit KV), image input, the built-in draft head, attention or GEMV settings other than the Python package's
   (YSKN_ATX_PKV=3, YSKN_ATX_PK3_PLUT=1; YSKN_ATX_PK3_MMAX, YSKN_ATX_PK3_QK8 and, for a 4-bit KV profile, YSKN_ATX_PK3_NST unset).
   A failure after the checks (out of memory) leaves the carrier without a binding, frees the slots this call made and clears the
   CUDA error; the next yskn_ms_bind2 makes what is missing.  The slots keep their states and sampling across bindings; a new slot
   starts with the carrier's sampling.  Drops the live sequence. */
YSKN_API int yskn_ms_bind2(yskn_engine* e, yskn_engine** eng, int n, int n_light);
/* Copy the state of src (same model file and settings, no image input, nothing left to commit) into slot, and its sampling (the values
   of yskn_set_opt 10..13, 16, 29 on src): this replaces all of the slot's sampling, also what yskn_ms_set_sampling / yskn_ms_seeds
   set before; set other values after the load.  The drafter memory comes with it; a src without the drafter has none to give, and the
   slot's is cleared (the ids are the same: a pass gives the same token at every position whatever the drafts; only how many drafts
   the first passes confirm changes).  Drafts that yskn_ms_draft made for a pass with rows of this slot are dropped: a following
   yskn_ms_step2 with toks == NULL and those rows fails ("no tokens"); call yskn_ms_draft again (or give toks).  The frontier comes
   with the live sequence of src (see above; not known when src has none).  Drops the live sequence of the carrier (and of src).
   A slot that yskn_ms_bind2 made holds no stream: passes refuse its rows until a yskn_ms_load fills it. */
YSKN_API int yskn_ms_load(yskn_engine* e, int slot, yskn_engine* src);
/* The inverse of yskn_ms_load.  ids = the n tokens the slot holds (n = yskn_ms_slot_pos).  Every token after the load is checked
   against the rows the slot took in its passes, and the ones before against the source's live sequence at the load.  The carrier
   takes the slot's recurrent states, histories, KV rows below n, position, drafter memory (cleared when the slot has none) and
   sampling (as yskn_set_opt would set it: a caller that keeps a copy of the carrier's sampling takes the slot's); its live sequence
   becomes ids, with the slot's frontier (the frontier form checks it).  Its saved states stay only when they were taken on the KV
   rows the slot copied from the carrier (before the slot's
   yskn_ms_load from the carrier, at a position inside the tokens the carrier held then); the others are dropped (a later read
   restores another one or reads from the start: slower, and read another way; see the live sequence above).  When the source had no
   live sequence at the load (its tokens there cannot be checked), the carrier keeps no live sequence and no saved state.
   The slot's accepted rows go into its state first; drafts with rows of the slot are dropped; a verify pass of the carrier that is not
   committed is dropped.  Refused (nothing changes) while a multi-stream pass is not committed or the carrier has image input on.  The
   slot is then marked as moved out: yskn_ms_draft / yskn_ms_step2 refuse its rows until yskn_ms_load fills it again. */
YSKN_API int yskn_ms_unload(yskn_engine* e, int slot, const int32_t* ids, int n);
/* Drop the carrier's live sequence (see above). */
YSKN_API int yskn_live_drop(yskn_engine* e);
/* key 1: 1 = the drafter makes the drafts of the slots (needs the drafter loaded before yskn_ms_bind2), 0 = off. */
YSKN_API int yskn_ms_opt(yskn_engine* e, int key, int value);
/* Drafts for the NEXT pass: toks = the last token of each slot at its first row (its frontier: checked when known, see above);
   dev[r] != 0 at a first row = let the drafter fill the other rows of that slot (else the caller's drafts in toks are kept); dev ==
   NULL = the drafter fills them all.  slots and toks are needed.  Refused before anything changes (an open pass stays open): a slot
   no yskn_ms_load filled, a moved-out slot, a first row that is not the slot's frontier.  Drops the live sequence. */
YSKN_API int yskn_ms_draft(yskn_engine* e, int M, const int32_t* slots, const int32_t* toks, const int32_t* dev, int use_graph);
/* One pass of M <= 16 rows.  toks == NULL: the tokens of yskn_ms_draft with the same slots.  out_argmax[r] = the token after row r
   (sampled when the slot of row r samples, else the argmax), out_toks[r] = the token of row r.  Row r of a slot sits at the slot's
   position (yskn_ms_slot_pos before the pass) + its place among the slot's rows.  The first row of a slot is its frontier (given
   toks: checked when known; see above).  Refused before anything changes (an open pass stays open): a slot no yskn_ms_load filled,
   a moved-out slot, rows of a slot apart, a context overflow, slots == NULL.  Drops the live sequence. */
YSKN_API int yskn_ms_step2(yskn_engine* e, int M, const int32_t* slots, const int32_t* toks, int use_graph, int32_t* out_argmax, int32_t* out_toks,
                           int32_t* top_ids, float* top_vals);
/* Right after every yskn_ms_step2 (n may be 0): n_accept[i] = how many drafts of slots[i] the pass confirmed.  Every item is checked
   first: a refused call changes nothing and the pass can be committed again; a pass left uncommitted counts as no draft accepted.
   The slot's frontier becomes out_argmax after its last accepted draft (the next first row). */
YSKN_API int yskn_ms_commit(yskn_engine* e, int n, const int32_t* slots, const int32_t* n_accept);
/* Finish the pending work of the slots (accepted rows into their states, the drafter memory). */
YSKN_API int yskn_ms_flush(yskn_engine* e);
/* The sampling of one slot from its next yskn_ms_draft / yskn_ms_step2 on, with the values and checks of yskn_set_opt:
   temperature x 1000 (key 10; <= 0 = greedy: the argmax), top_p x 1000 (11; 1..1000), top_k (12; 1..20), min_p x 1000 (29; 0..1000),
   sampler (16; 2 or 3), seed (13).  A slot then gives the ids the carrier gives alone with the same values.  Greedy and sampling slots
   share a pass.  Nothing changes when a value is refused.  yskn_ms_load replaces the slot's sampling (also of a slot moved out by
   yskn_ms_unload): call this after the load. */
YSKN_API int yskn_ms_set_sampling(yskn_engine* e, int slot, int temperature, int top_p, int top_k, int min_p, int sampler, uint32_t seed);
/* The seed of every slot (n = the number of slots); the rest of each slot's sampling stays (see yskn_ms_set_sampling; call it after
   yskn_ms_load, which replaces the seed too). */
YSKN_API int yskn_ms_seeds(yskn_engine* e, const uint32_t* seeds, int n);
YSKN_API int yskn_ms_slot_pos(yskn_engine* e, int slot, int* pos);
/* The values yskn_state_hash would give for an engine in the slot's state. */
YSKN_API int yskn_ms_slot_hash(yskn_engine* e, int slot, uint64_t* out, int max_out, int* n_out);
/* What a caller needs to plan the slots; works before yskn_ms_bind2.  Writes min(max_out, 15) values and their number to *n_out:
     [0] layout of this list (1)
     [1] 0 = yskn_ms_bind2 takes this engine's settings; else why not: 1 16-bit KV, 2 8-bit KV (KV5 off: YSKN_TB_CAP=0), 3 image
         input on, 4 built-in draft head, 5 attention settings, 6 exact-sum GEMV off, 7 debug skips, 8 kept conv outputs off
         (set_opt 5), 9 no shared weights, 10 rows of a pass, 11 attention profile of this process (environment)
     [2] KV format: 5 = KV5, 8 = 8-bit, 16 = 16-bit          [3] bits of a packed KV5 code (4 or 5; 0 when not KV5)
     [4] KV bytes of one position of one sequence (all attention layers, K and V with their scales; the carrier's own context
         and every slot take max_ctx positions of it)
     [5] bytes of one slot besides its KV: recurrent states and histories, and the drafter memory when the drafter is loaded
     [6] max_ctx (positions of the carrier and of every slot)   [7] rows of a pass at most (16)   [8] slots at most (64)
     [9] drafter loaded   [10] drafter output head (.shead) loaded   [11] slots made so far   [12] slots of the current binding
     [13] pinned host bytes of one saved prompt state   [14] saved prompt states at most (YSKN_SNAP_SLOTS)
   A plan: (n_slots + 1) * max_ctx * [4] + n_slots * [5] + what the model and the passes take must fit in the free GPU memory. */
YSKN_API int yskn_ms_info(yskn_engine* e, int64_t* out, int max_out, int* n_out);
#ifdef __cplusplus
}
#endif
