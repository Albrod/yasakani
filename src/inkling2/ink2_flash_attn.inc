// SPDX-License-Identifier: Apache-2.0
// Tiled Flash-GQA attention for passes of 8 rows or more (prompt reading, wide verification).
// An option (flash_attn), off by default: the sums run in another order than kernel attn.
//
// attn_rel: the relative-position logit of every (row, head, distance):
//   rel_logits[(row * nh + h) * rel_stride + dist] = sum_l r[row, h, l] * rel_proj[l, dist]   (dist < rel_extent)
// attn_flash (FP32 form, CUDA cores and the emulator): one block of 256 threads shares one key/value head across
//   8 rows x 3 query heads = 24 queries; the keys stream in tiles of 64 through shared memory (K in two halves of
//   64 dimensions, V in two halves of 32 keys), each tile loaded once for the 24 queries. Online softmax; with one
//   split the sink seeds the accumulator (m = sink, l = 1, o = 0), else the split writes the partial of kv_merge.
//   Grid: row tiles x splits x nkv (block index = (tile * splits + split) * nkv + kv head), as kv_attention.
// The tensor-core form (ink2_flash_tc.cuh) is chosen by the CUDA backend when flash_tc is set.

KERNEL(attn_rel, AttnArgs) {
  const size_t idx = KBID;  // row * nh + head
  float r[32];
  for (int l = 0; l < a.drel; ++l) r[l] = a.r[idx * a.drel + l];
  float* out = a.rel_logits + idx * a.rel_stride;
  for (int d = KTID; d < a.rel_extent; d += KNT) {
    float acc = 0.0f;
    for (int l = 0; l < a.drel; ++l) acc = KFMA(r[l], k_bf(a.rel_proj[(size_t)l * a.rel_extent + d]), acc);
    out[d] = acc;
  }
}

// The key range of a tile of rows r0 .. r0+rows-1 and one split of it (keys begin .. end-1).
KINL void flash_range(const AttnArgs& a, int r0, int rows, int sp, int* pmin_out, int* begin, int* end) {
  int pmin = a.st[r0].pos, pmax = pmin;
  for (int i = 1; i < rows; ++i) {
    int p = a.st[r0 + i].pos;
    pmin = p < pmin ? p : pmin;
    pmax = p > pmax ? p : pmax;
  }
  int first = a.window && pmin >= a.window ? pmin + 1 - a.window : 0;
  int total = pmax + 1 - first;
  int active = (total + 255) / 256;
  if (active > a.splits) active = a.splits;
  if (active < 1) active = 1;
  int chunk = (total + active - 1) / active;
  chunk = (chunk + 63) & ~63;
  int b = first + sp * chunk, e = b + chunk;
  if (e > pmax + 1) e = pmax + 1;
  *pmin_out = pmin;
  *begin = b;
  *end = e;
}

// the MXFP8 cache: the BF16 copies of the sinks and of the hot window, chosen for the last
// (largest) position of the tile; the FP32 cache reads no copy (-1).
KINL int flash_pnear(const AttnArgs& a, int r0, int rows) {
  if (a.kv.mode != INK2_KV_MXFP8) return -1;
  int p = a.st[r0].pos;
  for (int i = 1; i < rows; ++i) p = a.st[r0 + i].pos > p ? a.st[r0 + i].pos : p;
  return p;
}
KINL int flash_ns(const AttnArgs& a, int t, int pn) { return pn < 0 ? -1 : kv_near(a.kv, t, pn); }

KERNELB(attn_flash, AttnArgs, 256, 2) {
  const int NQ = 24, QS = 132, KS = 68, PS = 65;
  KSHARED(float, qs, 24 * 132);
  KSHARED(float, kvs, 64 * 68);  // K half: [64 keys][64 dims (+4)]; V half: [32 keys][128 dims]
  KSHARED(float, ps, 24 * 65);
  KSHARED(int, qpos, 24);
  const int tid = KTID, lane = tid & 31, warp = tid >> 5;
  const int G = a.nh / a.nkv;  // 3 (the engine checks)
  int b = KBID;
  const int kh = b % a.nkv;
  b /= a.nkv;
  const int sp = b % a.splits;
  b /= a.splits;
  const int BR = NQ / G;
  const int r0 = b * BR;
  int rows = a.rows - r0;
  if (rows > BR) rows = BR;
  const int nq = rows * G;
  int pmin, begin, end;
  flash_range(a, r0, rows, sp, &pmin, &begin, &end);
  const int pn = flash_pnear(a, r0, rows);
  const int C = a.nkv * 128;
  for (int i = tid; i < NQ * 32; i += 256) {
    int q = i >> 5, d4 = (i & 31) * 4;
    KF4 v = KF4ZERO;
    if (q < nq) v = KLOADF4(a.q + ((size_t)(r0 + q / G) * a.nh + kh * G + q % G) * 128 + d4);
    float* dst = qs + q * QS + d4;
    dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
  }
  if (tid < NQ) qpos[tid] = tid < nq ? a.st[r0 + tid / G].pos : -1;
  // softmax / value state: warp w owns the queries w, w + 8, w + 16; lane l the elements 4l .. 4l+3
  float m[3], l[3];
  KF4 o[3];
  for (int u = 0; u < 3; ++u) {
    int q = warp + 8 * u;
    bool seed = a.splits == 1 && q < nq;
    m[u] = seed ? a.sink[(size_t)(r0 + q / G) * a.nh + kh * G + q % G] : -3.402823466e38f;
    l[u] = seed ? 1.0f : 0.0f;
    o[u] = KF4ZERO;
  }
  const float neg_inf = KBITS2F(0xff800000u);
  const int j = tid & 63, qg = tid >> 6;  // scores: key j of the tile, queries 6 qg .. 6 qg + 5
  KSYNC();
  for (int t0 = begin; t0 < end; t0 += 64) {
    int n = end - t0;
    if (n > 64) n = 64;
    float dot[6];
    for (int u = 0; u < 6; ++u) dot[u] = 0.0f;
    for (int half = 0; half < 2; ++half) {
      for (int i = tid; i < 64 * 16; i += 256) {
        int kk = i >> 4, d4 = (i & 15) * 4;
        KF4 v = KF4ZERO;
        if (kk < n) {
          int t = t0 + kk, slot = a.ring ? t % a.ring : t;
          v = kv_read4(a.kv, a.kc, 0, slot, flash_ns(a, t, pn), C, kh * 128 + half * 64 + d4);
        }
        float* dst = kvs + kk * KS + d4;
        dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
      }
      KSYNC();
      const float* kr = kvs + j * KS;
      for (int d = 0; d < 64; d += 4) {
        KF4 k4 = KLOADF4(kr + d);
        for (int u = 0; u < 6; ++u) {
          KF4 q4 = KLOADF4(qs + (qg * 6 + u) * QS + half * 64 + d);
          float x = KFMA(q4.x, k4.x, dot[u]);
          x = KFMA(q4.y, k4.y, x);
          x = KFMA(q4.z, k4.z, x);
          dot[u] = KFMA(q4.w, k4.w, x);
        }
      }
      KSYNC();
    }
    for (int u = 0; u < 6; ++u) {
      int q = qg * 6 + u, t = t0 + j;
      float s = neg_inf;
      if (q < nq && j < n) {
        int p = qpos[q];
        if (t <= p && (!a.window || p - t < a.window)) {
          int dist = p - t;
          s = dot[u] * a.scale;
          if (dist < a.rel_extent)
            s = s + a.rel_logits[((size_t)(r0 + q / G) * a.nh + kh * G + q % G) * a.rel_stride + dist];
        }
      }
      ps[q * PS + j] = s;
    }
    KSYNC();
    for (int u = 0; u < 3; ++u) {
      float* pr = ps + (warp + 8 * u) * PS;
      float s0 = pr[lane], s1 = pr[lane + 32];
      float mn = fmaxf(m[u], k_warp_max(KCTXA fmaxf(s0, s1)));
      float al = expf(m[u] - mn);
      float p0 = expf(s0 - mn), p1 = expf(s1 - mn);
      pr[lane] = p0;
      pr[lane + 32] = p1;
      l[u] = l[u] * al + k_warp_sum(KCTXA p0 + p1);
      m[u] = mn;
      o[u].x *= al; o[u].y *= al; o[u].z *= al; o[u].w *= al;
    }
    for (int half = 0; half < 2; ++half) {
      for (int i = tid; i < 32 * 32; i += 256) {
        int kk = i >> 5, d4 = (i & 31) * 4, key = half * 32 + kk;
        KF4 v = KF4ZERO;
        if (key < n) {
          int t = t0 + key, slot = a.ring ? t % a.ring : t;
          v = kv_read4(a.kv, a.vc, 1, slot, flash_ns(a, t, pn), C, kh * 128 + d4);
        }
        float* dst = kvs + kk * 128 + d4;
        dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
      }
      KSYNC();
      int cnt = n - half * 32;
      if (cnt > 32) cnt = 32;
      for (int kk = 0; kk < cnt; ++kk) {
        KF4 v = KLOADF4(kvs + kk * 128 + lane * 4);
        for (int u = 0; u < 3; ++u) {
          float w = ps[(warp + 8 * u) * PS + half * 32 + kk];
          o[u].x = KFMA(w, v.x, o[u].x); o[u].y = KFMA(w, v.y, o[u].y);
          o[u].z = KFMA(w, v.z, o[u].z); o[u].w = KFMA(w, v.w, o[u].w);
        }
      }
      KSYNC();
    }
  }
  for (int u = 0; u < 3; ++u) {
    int q = warp + 8 * u;
    if (q >= nq) continue;
    size_t idx = (size_t)(r0 + q / G) * a.nh + kh * G + q % G;
    if (a.splits == 1) {
      KF4 g = KLOADF4(a.gate + idx * 128 + lane * 4);
      float* out = a.out + idx * 128 + lane * 4;
      out[0] = (o[u].x / l[u]) * g.x;
      out[1] = (o[u].y / l[u]) * g.y;
      out[2] = (o[u].z / l[u]) * g.z;
      out[3] = (o[u].w / l[u]) * g.w;
    } else {
      float* p = a.partial + (idx * a.splits + sp) * 130;
      kv_put4(p + lane * 4, o[u]);
      if (!lane) { p[128] = m[u]; p[129] = l[u]; }
    }
  }
}
