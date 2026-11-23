// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Argument blocks of the drafter kernels (ink2_drafter_kernels.inc): the
// drafter of Inkling-2 Mini (mtp.safetensors). Included by ink2_kargs.h.
// Arithmetic: our reference implementation of the drafter (not part of this repository); head size 128, block of 8 rows.
#pragma once

// Commit, part 1: c_l = RMSNorm(sum_j alpha[l, j] F_j) * w_l for the three draft
// layers (into cmix [3][max_rows][H]) and f_rnn (l = 3, into the ring frnn
// [fring][H] at the slot of the row's position). F_j = the outputs of the three
// feature visits and the final hidden of the main model. Block = row * 4 + l.
struct DMixArgs {
  const float* f[4];  // [rows, H]
  const u16* w[4];
  float alpha[4][4];
  float* cmix;
  float* frnn;
  const StepState* st;
  int H, max_rows, fring;
  float eps;
};
// Commit, part 2: context keys k_norm_ctx + RoPE and values into the caches of
// the three layers (slot = position % cap). kv: [6][max_rows][C] (k0 v0 k1 v1 k2 v2).
// Block = (row * 3 + layer) * nkv + head, 128 threads.
struct DCtxArgs {
  const float* kv;
  const u16* knorm[3];
  float* ck[3];
  float* cv[3];
  int cap[3];
  const float* rcos;  // [positions][64]
  const float* rsin;
  const StepState* st;
  int nkv, max_rows;
  float eps;
};
// Block input rows 1..7 = mask_embed (row 0: the engine's norm kernel). One block per row.
struct DEmbArgs {
  const u16* mask;
  float* h;  // [8, H]
  int H;
};
// q_norm / k_norm per head and RoPE at st[0].pos + row, in place.
// Block = row * (nh + nkv) + head (q heads first), 128 threads.
struct DQkArgs {
  float* q;  // [8, nh * 128]
  float* k;  // [8, nkv * 128]
  const u16* qn;
  const u16* kn;
  const float* rcos;
  const float* rsin;
  const StepState* st;
  int nh, nkv;
  float eps;
};
// Non-causal attention of the 8 block rows over the context keys (all, or the
// last `window` before st[0].pos) and the 8 block keys. Block = row * nh + head.
struct DAttnArgs {
  const float* q;   // [8, nh * 128]
  const float* kb;  // [8, nkv * 128]
  const float* vb;
  const float* ck;  // [cap, nkv * 128]
  const float* cv;
  float* scratch;  // [8 * nh, smax]
  float* out;      // [8, nh * 128]
  const StepState* st;
  int nh, nkv, cap, window, smax;
  float scale;
};
// h += y, out = RMSNorm(h) * w. One block per row.
struct DAddNormArgs {
  float* h;
  const float* y;
  const u16* w;
  float* out;
  int H;
  float eps;
};
// out = silu(g) * u, n values.
struct DSiluArgs {
  const float* g;
  const float* u;
  float* out;
  int n;
};
// Markov head, step `step` (0 .. depth-1): picks the token of step - 1 from the
// partial maxima of dfold (step > 0; drafts[step - 1] and st[step].token), then
// z = [s, markov_w1[prev], hn[step + 1], f_rnn]. step == depth: the pick only.
// One block.
struct DZinArgs {
  const float* pval;  // [G]
  const int* pidx;
  int* drafts;        // [8]
  StepState* st;
  const float* s;     // [rank]
  const u16* w1;      // [w1_rows, rank]: bfloat16, or the codes of a packed table (w1fmt != INK2_DFMT_BF16)
  const u8* w1sc;     // packed: its block scales; w1fac its row-sum factor (as GemvJob.fac)
  float w1fac;
  int w1fmt;
  const float* hn;    // [8, H]
  const float* frnn;  // [fring, H]
  float* z;           // [2 rank + 2 H]
  int G, step, depth, rank, H, fring, w1_rows;
  // Confidence of the drafts (NULL p2 = off): per block second-best value and sum of
  // exp(value - cap) from dfold; conf [8][4] = top1, top2, log-sum-exp, running product of the
  // top probability of drafts 1 .. j; ncut = drafts kept (depth = no cut); cutp [16] = rule
  // (cutp[0]: 0 off, 1 top probability, 2 top1 - top2, 3 running product; cutp[j] = threshold of draft j).
  const float* p2;
  const float* psum;
  float* conf;
  int* ncut;
  const float* cutp;
  float cap;
};
// Markov head, the rest of a step: gates from zz = rnn_proj z (+ bias): s (block 0
// writes it), latent = tanh(out); then logit_v = softcap(base_v + (fold_b_v +
// fold_v . latent)) for every vocabulary row v and the largest per block (first
// index on ties). One warp per row, rank 256.
struct DFoldArgs {
  const float* zz;    // [3 rank]
  const u16* rnn_b;   // [3 rank]
  float* s;           // [rank]
  const float* base;  // [V] row of this step
  const u16* f16;     // [V, rank] or NULL
  const float* f32;   // [V, rank] (used when f16 and fp are NULL)
  const u8* fp;       // [V, rank] packed (INK2_DFMT_*: fpfmt, scales fpsc, factor fpfac as GemvJob.fac) or NULL
  const u8* fpsc;
  float fpfac;
  int fpfmt;
  const float* fb;    // [V]
  float* pval;        // [G]
  int* pidx;
  int V, rank, step, G;  // G = blocks of the launch
  float cap;
  float* p2;        // [G] second-best value per block (NULL = off)
  float* psum;      // [G] sum of exp(value - cap) per block
  const int* ncut;  // drafts kept so far (< depth: cut, the step is skipped)
  int depth;
};

#include "ink2_drafter_sk_kargs.h"

#define INK2_DRAFT_KERNEL_LIST(X) \
  X(dmix, DMixArgs) \
  X(dctx, DCtxArgs) \
  X(demb, DEmbArgs) \
  X(dqk, DQkArgs) \
  X(dattn, DAttnArgs) \
  X(daddnorm, DAddNormArgs) \
  X(dsilu, DSiluArgs) \
  X(dzin, DZinArgs) \
  X(dfold, DFoldArgs) \
  X(dattn_sk, DAttnSkArgs) X(dattn_skr, DAttnSkArgs)
