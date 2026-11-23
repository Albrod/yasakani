// SPDX-License-Identifier: Apache-2.0
// GQA-shared online-softmax Split-K attention for passes of fewer
// than 8 rows (decode, small verification). Option splitk_attn (default 0).
//
// Block = (row, kv head, split). The block loads each key/value of its split
// once and uses it for all G = nh / nkv query heads of the group (G <= 4).
// The history of a row is cut into `active` splits of at least sk_chunk keys,
// computed on the device from the row's position, so a recorded graph stays
// valid while the conversation grows. Each warp keeps an online softmax per
// head over a stripe of SK_U keys per step (SK_U loads in flight per lane).
// With one active split the block finishes the row itself (sink + gate);
// otherwise it writes (vector, max, mass) and attn_splitk_reduce merges the
// splits with the accumulator seeded by the sink: m = sink, l = 1, o = 0.
// The keys are read through kv_read4, so the FP32 and the MXFP8 caches
// both work. rel_t, when set, is rel_proj transposed to
// [rel_extent, drel] so that the 16 lanes read one 32-byte row.
#define SK_MAXG 4
#define SK_U 4
#define SK_STRIDE 132  // floats per partial: vector, max, mass, 2 pad (16-byte rows)

KINL int sk_active(const AttnArgs& a, int T, int* chunk) {
  int c = (T + a.splits - 1) / a.splits;
  if (c < a.sk_chunk) c = a.sk_chunk;
  *chunk = c;
  int n = (T + c - 1) / c;
  return n > a.splits ? a.splits : n;
}

KINL float sk_rel(const AttnArgs& a, int dist, int lane) {
  if (dist >= a.rel_extent || lane >= a.drel) return 0.0f;
  return a.rel_t ? k_bf(a.rel_t[(size_t)dist * a.drel + lane]) : k_bf(a.rel_proj[(size_t)lane * a.rel_extent + dist]);
}

KERNELB(attn_gqa_splitk, AttnArgs, 256, 2) {
  KSHARED(float, vec, 8 * SK_MAXG * 128);
  KSHARED(float, wmax, 8 * SK_MAXG);
  KSHARED(float, wmass, 8 * SK_MAXG);
  const int lane = KTID & 31, warp = KTID >> 5;
  const int G = a.nh / a.nkv;
  int b = KBID;
  const int sp = b % a.splits;
  b /= a.splits;
  const int kh = b % a.nkv, row = b / a.nkv;
  const int pos = a.st[row].pos;
  int T = pos + 1;
  if (a.window && T > a.window) T = a.window;
  int chunk;
  const int active = sk_active(a, T, &chunk);
  if (sp >= active) return;
  const int start = pos + 1 - T + sp * chunk;
  int end = start + chunk;
  if (end > pos + 1) end = pos + 1;
  const size_t base = (size_t)row * a.nh + (size_t)kh * G;
  const int C = a.nkv * 128, col = kh * 128 + lane * 4;
  KF4 q[SK_MAXG], o[SK_MAXG];
  float r[SK_MAXG], mx[SK_MAXG], mass[SK_MAXG];
  KUNROLL
  for (int g = 0; g < SK_MAXG; ++g) {
    if (g < G) {
      q[g] = KLOADF4(a.q + (base + g) * 128 + lane * 4);
      r[g] = lane < a.drel ? a.r[(base + g) * a.drel + lane] : 0.0f;
    } else {
      q[g] = KF4ZERO;
      r[g] = 0.0f;
    }
    o[g] = KF4ZERO;
    mx[g] = -3.402823466e38f;
    mass[g] = 0.0f;
  }
  for (int t0 = start + warp * SK_U; t0 < end; t0 += 8 * SK_U) {
    KF4 k[SK_U], v[SK_U];
    float rel[SK_U];
    KUNROLL
    for (int u = 0; u < SK_U; ++u) {
      const int t = t0 + u;
      if (t < end) {
        const int slot = a.ring ? t % a.ring : t, ns = kv_near(a.kv, t, pos);
        k[u] = kv_read4(a.kv, a.kc, 0, slot, ns, C, col);
        v[u] = kv_read4(a.kv, a.vc, 1, slot, ns, C, col);
        rel[u] = sk_rel(a, pos - t, lane);
      } else {
        k[u] = KF4ZERO;
        v[u] = KF4ZERO;
        rel[u] = 0.0f;
      }
    }
    KUNROLL
    for (int g = 0; g < SK_MAXG; ++g) {
      if (g >= G) continue;
      float x[SK_U], m = mx[g];
      KUNROLL
      for (int u = 0; u < SK_U; ++u) {
        float d = KFMA(q[g].x, k[u].x, 0.0f);
        d = KFMA(q[g].y, k[u].y, d); d = KFMA(q[g].z, k[u].z, d); d = KFMA(q[g].w, k[u].w, d);
        x[u] = k_warp_sum(KCTXA KFMA(r[g], rel[u], d * a.scale));
        if (t0 + u < end) m = fmaxf(m, x[u]);
      }
      const float f = expf(mx[g] - m);
      float l = mass[g] * f;
      KF4 acc = o[g];
      acc.x *= f; acc.y *= f; acc.z *= f; acc.w *= f;
      KUNROLL
      for (int u = 0; u < SK_U; ++u) {
        if (t0 + u >= end) continue;
        const float w = expf(x[u] - m);
        l += w;
        acc.x = KFMA(w, v[u].x, acc.x); acc.y = KFMA(w, v[u].y, acc.y);
        acc.z = KFMA(w, v[u].z, acc.z); acc.w = KFMA(w, v[u].w, acc.w);
      }
      o[g] = acc;
      mx[g] = m;
      mass[g] = l;
    }
  }
  KUNROLL
  for (int g = 0; g < SK_MAXG; ++g) {
    if (g >= G) continue;
    kv_put4(vec + (warp * SK_MAXG + g) * 128 + lane * 4, o[g]);
    if (!lane) {
      wmax[warp * SK_MAXG + g] = mx[g];
      wmass[warp * SK_MAXG + g] = mass[g];
    }
  }
  KSYNC();
  for (int i = KTID; i < G * 128; i += KNT) {
    const int g = i >> 7, d = i & 127;
    const size_t idx = base + g;
    float m = active == 1 ? a.sink[idx] : -3.402823466e38f;
    for (int w = 0; w < 8; ++w) m = fmaxf(m, wmax[w * SK_MAXG + g]);
    float l = active == 1 ? expf(a.sink[idx] - m) : 0.0f, val = 0.0f;
    for (int w = 0; w < 8; ++w) {
      const float e = expf(wmax[w * SK_MAXG + g] - m);
      l = KFMA(e, wmass[w * SK_MAXG + g], l);
      val = KFMA(e, vec[(w * SK_MAXG + g) * 128 + d], val);
    }
    if (active == 1) {
      a.out[idx * 128 + d] = val / l * (a.gate ? a.gate[idx * 128 + d] : 1.0f);
    } else {
      float* p = a.partial + (idx * a.splits + sp) * SK_STRIDE;
      p[d] = val;
      if (!d) { p[128] = m; p[129] = l; }
    }
  }
}

// One warp per (row, head); lane owns 4 components. Seeded with the sink:
// m = sink, l = exp(sink - sink) = 1, o = 0, then the standard online merge.
KERNEL(attn_splitk_reduce, AttnArgs) {
  const int lane = KTID & 31, d0 = lane * 4;
  const size_t idx = KBID;
  const int row = KBID / a.nh;
  const int pos = a.st[row].pos;
  int T = pos + 1;
  if (a.window && T > a.window) T = a.window;
  int chunk;
  const int active = sk_active(a, T, &chunk);
  if (active <= 1) return;  // the split block wrote the row
  float m = a.sink[idx], l = 1.0f;
  KF4 o = KF4ZERO;
  const float* p = a.partial + idx * a.splits * SK_STRIDE;
  for (int s = 0; s < active; ++s) {
    const float* ps = p + (size_t)s * SK_STRIDE;
    const float m2 = ps[128], l2 = ps[129];
    const KF4 o2 = KLOADF4(ps + d0);
    const float nm = fmaxf(m, m2), e1 = expf(m - nm), e2 = expf(m2 - nm);
    l = KFMA(l, e1, l2 * e2);
    o.x = KFMA(o.x, e1, o2.x * e2); o.y = KFMA(o.y, e1, o2.y * e2);
    o.z = KFMA(o.z, e1, o2.z * e2); o.w = KFMA(o.w, e1, o2.w * e2);
    m = nm;
  }
  const float inv = 1.0f / l;
  KF4 g = a.gate ? KLOADF4(a.gate + idx * 128 + d0) : KF4{1.0f, 1.0f, 1.0f, 1.0f};
  o.x = o.x * inv * g.x; o.y = o.y * inv * g.y; o.z = o.z * inv * g.z; o.w = o.w * inv * g.w;
  kv_put4(a.out + idx * 128 + d0, o);
}
