// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Argument block of the drafter's Split-K attention (ink2_drafter_sk.inc).
// Included by ink2_drafter_kargs.h.
#pragma once

// Split-K form of dattn: the 8 block rows attend (non-causally) to the context keys of the layer (all, or the last
// `window` before st[0].pos) and to the 8 block keys. The context part is the same for every row, so
//   dattn_sk : block = (kv head, split); it loads each context key / value of its split ONCE and uses it for all
//              8 rows x G query heads of the group (G = nh / nkv <= DSK_MAXG), online softmax per query, and writes
//              (vector [128], max, mass) per query to `part`;
//   dattn_skr: block = (row, query head), 128 threads; merges the active splits and the 8 block keys, writes out
//              in the layout of dattn ([8, nh * 128]).
// The splits of a call are chosen on the device from st[0].pos (chunk = max(ceil(nc / splits), chunk_min)), so a
// recorded graph stays valid while the context grows. Scores are scaled by `scale` as in dattn.
#define DSK_MAXG 3
#define DSK_STRIDE 132  // floats per partial: vector [128], max, mass, 2 pad
struct DAttnSkArgs {
  const float* q;   // [8, nh * 128]
  const float* kb;  // [8, nkv * 128]
  const float* vb;
  const float* ck;  // [cap, nkv * 128]
  const float* cv;
  float* part;      // [nkv, splits, 8 * G, DSK_STRIDE]
  float* out;       // [8, nh * 128]
  const StepState* st;
  int nh, nkv, cap, window, splits, chunk_min;
  float scale;
};
