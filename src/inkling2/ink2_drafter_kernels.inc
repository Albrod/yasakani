// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Kernels of the drafter add-on (drafter of Inkling-2 Mini, mtp.safetensors),
// in the dialect of ink2_kernels.inc (included at its end, so compiled by both
// backends). Arithmetic: our reference implementation of the drafter (not part of this repository), float32, BF16
// weights widened. Head size 128, block of 8 rows, Markov rank 256.

// Commit, part 1: feature mix + RMS norm (ink2_draft.mix_norm / compute_f_rnn).
KERNEL(dmix, DMixArgs) {
  KSHARED(float, sm, 33);
  int r = KBID >> 2, l = KBID & 3, H = a.H;
  size_t o = (size_t)r * (size_t)H;
  float a0 = a.alpha[l][0], a1 = a.alpha[l][1], a2 = a.alpha[l][2], a3 = a.alpha[l][3];
  const float *f0 = a.f[0] + o, *f1 = a.f[1] + o, *f2 = a.f[2] + o, *f3 = a.f[3] + o;
  float ss = 0.0f;
  for (int i = KTID; i < H; i += KNT) {
    float v = a0 * f0[i] + a1 * f1[i] + a2 * f2[i] + a3 * f3[i];
    ss += v * v;
  }
  float tot = k_block_sum(KCTXA ss, sm);
  float sc = 1.0f / sqrtf(tot / (float)H + a.eps);
  float* out = l < 3 ? a.cmix + ((size_t)l * (size_t)a.max_rows + (size_t)r) * (size_t)H
                     : a.frnn + (size_t)(a.st[r].pos % a.fring) * (size_t)H;
  const u16* w = a.w[l];
  for (int i = KTID; i < H; i += KNT) {
    float v = a0 * f0[i] + a1 * f1[i] + a2 * f2[i] + a3 * f3[i];
    out[i] = (v * sc) * k_bf(w[i]);
  }
}

// RoPE (rotate-half) of one head held in y (128 values), for thread t at position pos.
KINL float k_drope(const float* y, int t, const float* rcos, const float* rsin, int pos) {
  int j = t & 63;
  float c = rcos[(size_t)pos * 64 + j], s = rsin[(size_t)pos * 64 + j];
  return t < 64 ? y[t] * c - y[t + 64] * s : y[t] * c + y[t - 64] * s;
}

// Commit, part 2: context keys and values (ink2_draft.context_kv).
KERNEL(dctx, DCtxArgs) {
  KSHARED(float, sm, 33);
  KSHARED(float, y, 128);
  int b = KBID, h = b % a.nkv;
  b /= a.nkv;
  int l = b % 3, r = b / 3, t = KTID;
  int C = a.nkv * 128;
  int pos = a.st[r].pos;
  const float* kr = a.kv + ((size_t)(2 * l) * (size_t)a.max_rows + (size_t)r) * (size_t)C + (size_t)h * 128;
  const float* vr = a.kv + ((size_t)(2 * l + 1) * (size_t)a.max_rows + (size_t)r) * (size_t)C + (size_t)h * 128;
  float kx = kr[t];
  float tot = k_block_sum(KCTXA kx * kx, sm);
  float sc = 1.0f / sqrtf(tot / 128.0f + a.eps);
  y[t] = (kx * sc) * k_bf(a.knorm[l][t]);
  KSYNC();
  float o = k_drope(y, t, a.rcos, a.rsin, pos);
  size_t slot = (size_t)(pos % a.cap[l]);
  a.ck[l][slot * (size_t)C + (size_t)h * 128 + t] = o;
  a.cv[l][slot * (size_t)C + (size_t)h * 128 + t] = vr[t];
}

// Block input rows 1 .. 7 = mask_embed (ink2_draft.forward_embed; row 0, the normed embedding of
// the anchor, comes from the engine's norm kernel). One block per row.
KERNEL(demb, DEmbArgs) {
  int H = a.H;
  float* out = a.h + (size_t)(KBID + 1) * (size_t)H;
  for (int i = KTID; i < H; i += KNT) out[i] = k_bf(a.mask[i]);
}

// q_norm / k_norm and RoPE of the block rows (ink2_draft.layer).
KERNEL(dqk, DQkArgs) {
  KSHARED(float, sm, 33);
  KSHARED(float, y, 128);
  int nhh = a.nh + a.nkv;
  int r = KBID / nhh, hh = KBID - r * nhh, t = KTID;
  float* p;
  const u16* w;
  if (hh < a.nh) {
    p = a.q + ((size_t)r * (size_t)a.nh + (size_t)hh) * 128;
    w = a.qn;
  } else {
    p = a.k + ((size_t)r * (size_t)a.nkv + (size_t)(hh - a.nh)) * 128;
    w = a.kn;
  }
  float x = p[t];
  float tot = k_block_sum(KCTXA x * x, sm);
  float sc = 1.0f / sqrtf(tot / 128.0f + a.eps);
  y[t] = (x * sc) * k_bf(w[t]);
  KSYNC();
  p[t] = k_drope(y, t, a.rcos, a.rsin, a.st[0].pos + r);
}

// Attention of the block (non-causal, context keys + the 8 block keys).
// One warp per key for the scores and the weighted sum, 8 warps.
KERNEL(dattn, DAttnArgs) {
  KSHARED(float, sm, 33);
  KSHARED(float, red, 8 * 128);
  int r = KBID / a.nh, hq = KBID - r * a.nh, hk = hq / (a.nh / a.nkv);
  int C = a.nkv * 128, lane = KTID & 31, w = KTID >> 5, nw = KNT >> 5;
  int n_ctx = a.st[0].pos;
  int lo = (a.window > 0 && n_ctx > a.window) ? n_ctx - a.window : 0;
  int nc = n_ctx - lo, T = nc + 8;
  KF4 q4 = KLOADF4(a.q + ((size_t)r * (size_t)a.nh + (size_t)hq) * 128 + lane * 4);
  float* sc = a.scratch + (size_t)KBID * (size_t)a.smax;
  float mx = -3.0e38f;
  for (int t = w; t < T; t += nw) {
    const float* kp = t < nc ? a.ck + (size_t)((lo + t) % a.cap) * (size_t)C : a.kb + (size_t)(t - nc) * (size_t)C;
    KF4 k4 = KLOADF4(kp + (size_t)hk * 128 + lane * 4);
    float d = q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
    d = k_warp_sum(KCTXA d) * a.scale;
    if (lane == 0) sc[t] = d;
    mx = fmaxf(mx, d);
  }
  float m = k_block_max(KCTXA mx, sm);
  float ssum = 0.0f;
  for (int t = KTID; t < T; t += KNT) {
    float e = expf(sc[t] - m);
    sc[t] = e;
    ssum += e;
  }
  float tot = k_block_sum(KCTXA ssum, sm);
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  for (int t = w; t < T; t += nw) {
    const float* vp = t < nc ? a.cv + (size_t)((lo + t) % a.cap) * (size_t)C : a.vb + (size_t)(t - nc) * (size_t)C;
    KF4 v4 = KLOADF4(vp + (size_t)hk * 128 + lane * 4);
    float p = sc[t];
    c0 = KFMA(p, v4.x, c0);
    c1 = KFMA(p, v4.y, c1);
    c2 = KFMA(p, v4.z, c2);
    c3 = KFMA(p, v4.w, c3);
  }
  red[w * 128 + lane * 4 + 0] = c0;
  red[w * 128 + lane * 4 + 1] = c1;
  red[w * 128 + lane * 4 + 2] = c2;
  red[w * 128 + lane * 4 + 3] = c3;
  KSYNC();
  if (KTID < 128) {
    float s = 0.0f;
    for (int i = 0; i < nw; ++i) s += red[i * 128 + KTID];
    a.out[((size_t)r * (size_t)a.nh + (size_t)hq) * 128 + KTID] = s / tot;
  }
}

// Residual add and the next RMS norm.
KERNEL(daddnorm, DAddNormArgs) {
  KSHARED(float, sm, 33);
  int H = a.H;
  float* h = a.h + (size_t)KBID * (size_t)H;
  const float* y = a.y + (size_t)KBID * (size_t)H;
  float* out = a.out + (size_t)KBID * (size_t)H;
  float ss = 0.0f;
  for (int i = KTID; i < H; i += KNT) {
    float v = h[i] + y[i];
    h[i] = v;
    ss += v * v;
  }
  float tot = k_block_sum(KCTXA ss, sm);
  float sc = 1.0f / sqrtf(tot / (float)H + a.eps);
  for (int i = KTID; i < H; i += KNT) out[i] = (h[i] * sc) * k_bf(a.w[i]);
}

KERNEL(dsilu, DSiluArgs) {
  int i = KBID * KNT + KTID;
  if (i < a.n) {
    float g = a.g[i];
    a.out[i] = (g * k_sigmoid(g)) * a.u[i];
  }
}

// Largest (value, index) over the block, first index on ties; result in thread 0.
KINL void k_dargmax_block(KCTXP float& best, float& bi, float* sv, float* si) {
  for (int d = 16; d >= 1; d >>= 1) {
    float ov = KSHFL_DOWN(best, d), oi = KSHFL_DOWN(bi, d);
    if (ov > best || (ov == best && oi < bi)) {
      best = ov;
      bi = oi;
    }
  }
  if ((KTID & 31) == 0) {
    sv[KTID >> 5] = best;
    si[KTID >> 5] = bi;
  }
  KSYNC();
  if (KTID == 0) {
    int nw = KNT >> 5;
    best = sv[0];
    bi = si[0];
    for (int w = 1; w < nw; ++w) {
      if (sv[w] > best || (sv[w] == best && si[w] < bi)) {
        best = sv[w];
        bi = si[w];
      }
    }
  }
}

// Markov head: pick of the previous step, then the input z of this step (ink2_draft.rnn_step).
KERNEL(dzin, DZinArgs) {
  KSHARED(float, sv, 32);
  KSHARED(float, si, 32);
  KSHARED(int, tok, 1);
  int prev;
  if (a.step > 0) {
    float best = -3.0e38f, bi = 1.0e30f;  // indices < 2^24 are exact as floats
    for (int g = KTID; g < a.G; g += KNT) {
      float v = a.pval[g], ix = (float)a.pidx[g];
      if (v > best || (v == best && ix < bi)) {
        best = v;
        bi = ix;
      }
    }
    k_dargmax_block(KCTXA best, bi, sv, si);
    if (KTID == 0) {
      int t = (int)bi;
      if (t < 0 || t >= a.w1_rows) t = 0;
      a.drafts[a.step - 1] = t;
      a.st[a.step].token = t;
      tok[0] = t;
    }
    KSYNC();
    prev = tok[0];
  } else {
    prev = a.st[0].token;
  }
  if (a.step >= a.depth) return;
  int R = a.rank, H = a.H;
  const u16* e = a.w1 + (size_t)prev * (size_t)R;
  const float* hrow = a.hn + (size_t)(a.step + 1) * (size_t)H;
  int pl = a.st[0].pos - 1;
  if (pl < 0) pl = 0;
  const float* fr = a.frnn + (size_t)(pl % a.fring) * (size_t)H;
  int n = 2 * R + 2 * H;
  for (int i = KTID; i < n; i += KNT) {
    float v;
    if (i < R) v = a.step > 0 ? a.s[i] : 0.0f;
    else if (i < 2 * R) v = k_bf(e[i - R]);
    else if (i < 2 * R + H) v = hrow[i - 2 * R];
    else v = fr[i - 2 * R - H];
    a.z[i] = v;
  }
}

// Markov head: gates, folded head, soft cap and the largest logit per block
// (ink2_draft.rnn_step + step_bias + propose).
KERNEL(dfold, DFoldArgs) {
  KSHARED(float, lat, 256);
  KSHARED(float, sv, 32);
  KSHARED(float, si, 32);
  int R = a.rank;
  for (int j = KTID; j < R; j += KNT) {
    lat[j] = tanhf(a.zz[2 * R + j] + k_bf(a.rnn_b[2 * R + j]));
    if (KBID == 0) {
      float g = k_sigmoid(a.zz[j] + k_bf(a.rnn_b[j]));
      float c = tanhf(a.zz[R + j] + k_bf(a.rnn_b[R + j]));
      float so = a.step > 0 ? a.s[j] : 0.0f;
      a.s[j] = c + g * (so - c);
    }
  }
  KSYNC();
  int lane = KTID & 31, wid = KTID >> 5, nwb = KNT >> 5;
  float l8[8];
  for (int e = 0; e < 8; ++e) l8[e] = lat[lane * 8 + e];
  float best = -3.0e38f, bi = 1.0e30f;
  int stride = a.G * nwb;
  for (int v = KBID * nwb + wid; v < a.V; v += stride) {
    float d;
    if (a.f16) {
      const u16* p = a.f16 + (size_t)v * (size_t)R + lane * 8;
      u64 c0 = KLOAD64(p), c1 = KLOAD64(p + 4);
      float f[8];
      K_BF8(c0, c1, f)
      d = KFMA(f[0], l8[0], 0.0f);
      for (int e = 1; e < 8; ++e) d = KFMA(f[e], l8[e], d);
    } else {
      const float* p = a.f32 + (size_t)v * (size_t)R + lane * 8;
      KF4 x0 = KLOADF4(p), x1 = KLOADF4(p + 4);
      d = KFMA(x0.x, l8[0], 0.0f);
      d = KFMA(x0.y, l8[1], d);
      d = KFMA(x0.z, l8[2], d);
      d = KFMA(x0.w, l8[3], d);
      d = KFMA(x1.x, l8[4], d);
      d = KFMA(x1.y, l8[5], d);
      d = KFMA(x1.z, l8[6], d);
      d = KFMA(x1.w, l8[7], d);
    }
    d = k_warp_sum(KCTXA d);
    float val = k_softcap(a.base[v] + (a.fb[v] + d), a.cap);
    if (val > best) {
      best = val;
      bi = (float)v;
    }
  }
  k_dargmax_block(KCTXA best, bi, sv, si);
  if (KTID == 0) {
    a.pval[KBID] = best;
    a.pidx[KBID] = bi < 1.0e29f ? (int)bi : 0;
  }
}
