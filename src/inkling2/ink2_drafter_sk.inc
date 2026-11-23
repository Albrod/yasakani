// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Split-K attention of the drafter (argument block and layout: ink2_drafter_sk_kargs.h).
// Included at the end of ink2_drafter_kernels.inc. The draft call uses it instead of dattn when the drafter option
// "attn_sk" is set (1 = the full-attention layer, 2 = every layer). Only the drafter's own numbers change (the
// drafts are proposals; every emitted token is the main model's choice).

KINL void dsk_split(int nc, int splits, int chunk_min, int* chunk, int* active) {
  int c = (nc + splits - 1) / splits;
  if (c < chunk_min) c = chunk_min;
  c = (c + 31) & ~31;  // whole tiles of 32 keys
  *chunk = c;
  int n = nc > 0 ? (nc + c - 1) / c : 0;
  *active = n > splits ? splits : n;
}

// Block = hk * splits + sp, 256 threads (8 warps). Warp w owns the queries w, w + 8, w + 16 of the group (query
// index qi = row * G + g, head hq = hk * G + g); per tile of 32 keys, lane = key for the scores, lane = 4 columns
// for the values.
KERNELB(dattn_sk, DAttnSkArgs, 256, 2) {
  KSHARED(float, qs, 8 * DSK_MAXG * 128);
  KSHARED(float, ks, 32 * 129);
  KSHARED(float, vs, 32 * 128);
  const int lane = KTID & 31, w = KTID >> 5;
  const int G = a.nh / a.nkv, QB = 8 * G, C = a.nkv * 128;
  const int sp = KBID % a.splits, hk = KBID / a.splits;
  const int n_ctx = a.st[0].pos;
  const int lo = (a.window > 0 && n_ctx > a.window) ? n_ctx - a.window : 0;
  const int nc = n_ctx - lo;
  int chunk, active;
  dsk_split(nc, a.splits, a.chunk_min, &chunk, &active);
  if (sp >= active) return;
  const int start = lo + sp * chunk;
  int end = start + chunk;
  if (end > n_ctx) end = n_ctx;
  for (int i = KTID; i < QB * 128; i += KNT) {
    int qi = i >> 7, d = i & 127, r = qi / G, g = qi - r * G;
    qs[i] = a.q[((size_t)r * (size_t)a.nh + (size_t)(hk * G + g)) * 128 + d] * a.scale;
  }
  float m[DSK_MAXG], l[DSK_MAXG];
  KF4 o[DSK_MAXG];
  KUNROLL
  for (int j = 0; j < DSK_MAXG; ++j) {
    m[j] = -3.0e38f;
    l[j] = 0.0f;
    o[j] = KF4ZERO;
  }
  for (int t0 = start; t0 < end; t0 += 32) {
    const int n = end - t0 < 32 ? end - t0 : 32;
    KSYNC();  // the previous tile is no longer read (and qs is written before the first)
    for (int i = KTID; i < 32 * 32; i += KNT) {
      int key = i >> 5, c4 = (i & 31) * 4;
      if (key < n) {
        size_t off = (size_t)((t0 + key) % a.cap) * (size_t)C + (size_t)hk * 128 + (size_t)c4;
        KF4 k4 = KLOADF4(a.ck + off);
        KF4 v4 = KLOADF4(a.cv + off);
        float* kp = ks + key * 129 + c4;
        kp[0] = k4.x;
        kp[1] = k4.y;
        kp[2] = k4.z;
        kp[3] = k4.w;
        float* vp = vs + key * 128 + c4;
        vp[0] = v4.x;
        vp[1] = v4.y;
        vp[2] = v4.z;
        vp[3] = v4.w;
      }
    }
    KSYNC();
    KUNROLL
    for (int j = 0; j < DSK_MAXG; ++j) {
      const int qi = w + 8 * j;
      if (qi >= QB) break;  // uniform over the warp
      const float* qp = qs + qi * 128;
      float s = -3.0e38f;
      if (lane < n) {
        const float* kp = ks + lane * 129;
        float acc = 0.0f;
        for (int d = 0; d < 128; ++d) acc = KFMA(qp[d], kp[d], acc);
        s = acc;
      }
      const float tmax = k_warp_max(KCTXA s);
      const float mn = fmaxf(m[j], tmax);
      const float corr = expf(m[j] - mn);
      const float p = lane < n ? expf(s - mn) : 0.0f;
      const float ps = k_warp_sum(KCTXA p);
      l[j] = l[j] * corr + ps;
      KF4 acc = o[j];
      acc.x *= corr;
      acc.y *= corr;
      acc.z *= corr;
      acc.w *= corr;
      for (int k = 0; k < n; ++k) {
        const float pk = KSHFL(p, k);
        KF4 v4 = KLOADF4(vs + k * 128 + lane * 4);
        acc.x = KFMA(pk, v4.x, acc.x);
        acc.y = KFMA(pk, v4.y, acc.y);
        acc.z = KFMA(pk, v4.z, acc.z);
        acc.w = KFMA(pk, v4.w, acc.w);
      }
      o[j] = acc;
      m[j] = mn;
    }
  }
  KUNROLL
  for (int j = 0; j < DSK_MAXG; ++j) {
    const int qi = w + 8 * j;
    if (qi >= QB) break;
    float* pp = a.part + (((size_t)hk * (size_t)a.splits + (size_t)sp) * (size_t)QB + (size_t)qi) * DSK_STRIDE;
    pp[lane * 4 + 0] = o[j].x;
    pp[lane * 4 + 1] = o[j].y;
    pp[lane * 4 + 2] = o[j].z;
    pp[lane * 4 + 3] = o[j].w;
    if (lane == 0) {
      pp[128] = m[j];
      pp[129] = l[j];
    }
  }
}

// Block = row * nh + hq, 128 threads (one column each): the active splits of dattn_sk and the 8 block keys.
KERNEL(dattn_skr, DAttnSkArgs) {
  KSHARED(float, bs, 8);
  KSHARED(float, sm, 33);
  const int G = a.nh / a.nkv, QB = 8 * G, C = a.nkv * 128;
  const int r = KBID / a.nh, hq = KBID - r * a.nh, hk = hq / G, g = hq - hk * G, qi = r * G + g;
  const int d = KTID;
  const int n_ctx = a.st[0].pos;
  const int lo = (a.window > 0 && n_ctx > a.window) ? n_ctx - a.window : 0;
  int chunk, active;
  dsk_split(n_ctx - lo, a.splits, a.chunk_min, &chunk, &active);
  const float qd = a.q[((size_t)r * (size_t)a.nh + (size_t)hq) * 128 + d] * a.scale;
  for (int j = 0; j < 8; ++j) {  // block key j: q . k_j (scaled q, as the partials)
    float s = k_block_sum(KCTXA qd * a.kb[(size_t)j * (size_t)C + (size_t)hk * 128 + d], sm);
    if (d == 0) bs[j] = s;
  }
  KSYNC();
  float M = -3.0e38f;
  for (int j = 0; j < 8; ++j) M = fmaxf(M, bs[j]);
  const float* pb = a.part + ((size_t)hk * (size_t)a.splits * (size_t)QB + (size_t)qi) * DSK_STRIDE;
  const size_t pstep = (size_t)QB * DSK_STRIDE;
  for (int s = 0; s < active; ++s) M = fmaxf(M, pb[(size_t)s * pstep + 128]);
  float L = 0.0f, O = 0.0f;
  for (int s = 0; s < active; ++s) {
    const float* pp = pb + (size_t)s * pstep;
    const float e = expf(pp[128] - M);
    L = KFMA(pp[129], e, L);
    O = KFMA(pp[d], e, O);
  }
  for (int j = 0; j < 8; ++j) {
    const float e = expf(bs[j] - M);
    L += e;
    O = KFMA(e, a.vb[(size_t)j * (size_t)C + (size_t)hk * 128 + d], O);
  }
  a.out[((size_t)r * (size_t)a.nh + (size_t)hq) * 128 + d] = O / L;
}
