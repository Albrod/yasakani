// SPDX-License-Identifier: Apache-2.0
// A block shares one KV head across a tile of causal queries. A warp owns a
// query; its lanes own four adjacent components each. Barriers are per tile,
// not per history token. Split boundaries depend on device-resident positions
// so an already captured graph remains valid as the conversation grows.
KINL void kv_put4(float* p, KF4 v) {
  p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w;
}

KERNEL(kv_attention, AttnArgs) {
  const int tile = 16, warps = 8, owners = 3, max_queries = warps * owners;
  KSHARED(KF4, keys, tile * 32);
  KSHARED(KF4, vals, tile * 32);
  const int lane = KTID & 31, warp = KTID >> 5;
  int group = a.nh / a.nkv, bundles = (a.row_tile * group + max_queries - 1) / max_queries;
  int b = KBID, kh = b % a.nkv;
  b /= a.nkv;
  int sp = b % a.splits;
  b /= a.splits;
  int qbase = max_queries * (b % bundles);
  int r0 = (b / bundles) * a.row_tile;
  int rows = a.rows - r0;
  if (rows > a.row_tile) rows = a.row_tile;
  const int pos0 = a.st[r0].pos, pos1 = a.st[r0 + rows - 1].pos;
  int first = a.window && pos0 >= a.window ? pos0 + 1 - a.window : 0;
  int total = pos1 + 1 - first;
  int active = (total + 127) / 128;
  if (active > a.splits) active = a.splits;
  int chunk = (total + active - 1) / active;
  int begin = first + sp * chunk, end = begin + chunk;
  if (end > pos1 + 1) end = pos1 + 1;
  const int C = a.nkv * 128, col = kh * 128 + lane * 4;
  int pos[owners], head[owners], maxrow = rows * group;
  size_t index[owners];
  KF4 q[owners], acc[owners];
  float rel[owners], maximum[owners], mass[owners];
  for (int j = 0; j < owners; ++j) {
    int qi = qbase + warp + j * warps;
    if (qi >= maxrow) break;
    int row = r0 + qi / group;
    head[j] = kh * group + qi % group;
    index[j] = (size_t)row * a.nh + head[j];
    pos[j] = a.st[row].pos;
    q[j] = KLOADF4(a.q + index[j] * 128 + lane * 4);
    rel[j] = lane < a.drel ? a.r[index[j] * a.drel + lane] : 0;
    maximum[j] = -3.402823466e38f;
    mass[j] = 0;
    acc[j] = KF4ZERO;
  }
  // relative bias table: transposed (stride drel) or original (stride 1 from lane * rel_extent)
  const u16* rp = a.rel_proj_t ? a.rel_proj_t + (lane < a.drel ? lane : 0) : a.rel_proj + (size_t)(lane < a.drel ? lane : 0) * a.rel_extent;
  const int rs = a.rel_proj_t ? a.drel : 1;
  for (int t0 = begin; t0 < end; t0 += tile) {
    int n = end - t0;
    if (n > tile) n = tile;
    for (int off = warp; off < n; off += warps) {
      int t = t0 + off, slot = a.ring ? t % a.ring : t;
      int near_slot = kv_near(a.kv, t, pos1);
      keys[off * 32 + lane] = kv_read4(a.kv, a.kc, 0, slot, near_slot, C, col);
      vals[off * 32 + lane] = kv_read4(a.kv, a.vc, 1, slot, near_slot, C, col);
    }
    KSYNC();
    for (int j = 0; j < owners && qbase + warp + j * warps < maxrow; ++j) {
      int lo = a.window ? pos[j] + 1 - a.window - t0 : 0;
      if (lo < 0) lo = 0;
      int hi = pos[j] + 1 - t0;
      if (hi > n) hi = n;
      if (hi <= lo) continue;
      float logits[tile], m = maximum[j];
      for (int i = lo; i < hi; ++i) {
        int t = t0 + i, dist = pos[j] - t;
        KF4 k = keys[i * 32 + lane];
        // In the narrow band straddling the hot/cold boundary, only some
        // rows use the high-precision copy. All other keys are decoded once.
        if (a.kv.hot_slots && t >= a.kv.sinks && t <= pos1 - a.kv.hot && dist < a.kv.hot) {
          int near_slot = kv_near(a.kv, t, pos[j]);
          k = kv_read4(a.kv, a.kc, 0, t, near_slot, C, col);
        }
        KF4 x = q[j];
        float dot = KFMA(x.x, k.x, 0.0f);
        dot = KFMA(x.y, k.y, dot); dot = KFMA(x.z, k.z, dot); dot = KFMA(x.w, k.w, dot);
        dot *= a.scale;
        if (dist < a.rel_extent && lane < a.drel)
          dot = KFMA(rel[j], k_bf(rp[(size_t)dist * rs]), dot);
        logits[i] = k_warp_sum(KCTXA dot);
        m = fmaxf(m, logits[i]);
      }
      float old = expf(maximum[j] - m);
      KF4 o = acc[j];
      o.x *= old; o.y *= old; o.z *= old; o.w *= old;
      float l = mass[j] * old;
      for (int i = lo; i < hi; ++i) {
        int t = t0 + i;
        KF4 v = vals[i * 32 + lane];
        if (a.kv.hot_slots && t >= a.kv.sinks && t <= pos1 - a.kv.hot && pos[j] - t < a.kv.hot) {
          int near_slot = kv_near(a.kv, t, pos[j]);
          v = kv_read4(a.kv, a.vc, 1, t, near_slot, C, col);
        }
        float w = expf(logits[i] - m);
        l += w;
        o.x = KFMA(w, v.x, o.x); o.y = KFMA(w, v.y, o.y);
        o.z = KFMA(w, v.z, o.z); o.w = KFMA(w, v.w, o.w);
      }
      mass[j] = l; maximum[j] = m; acc[j] = o;
    }
    KSYNC();
  }
  for (int j = 0; j < owners && qbase + warp + j * warps < maxrow; ++j) {
    size_t idx = index[j];
    KF4 o = acc[j];
    if (a.splits == 1) {
      float sink = a.sink[idx], m = fmaxf(maximum[j], sink), f = expf(maximum[j] - m);
      float z = mass[j] * f + expf(sink - m);
      o.x = o.x * f / z; o.y = o.y * f / z;
      o.z = o.z * f / z; o.w = o.w * f / z;
      if (a.gate) {
        KF4 g = KLOADF4(a.gate + idx * 128 + lane * 4);
        o.x *= g.x; o.y *= g.y; o.z *= g.z; o.w *= g.w;
      }
      kv_put4(a.out + idx * 128 + lane * 4, o);
    } else {
      float* p = a.partial + (idx * a.splits + sp) * 130;
      kv_put4(p + lane * 4, o);
      if (!lane) { p[128] = maximum[j]; p[129] = mass[j]; }
    }
  }
}

// Decode / small verification batch on this model (GQA=3). Each warp walks
// its own stripe of keys, sharing a loaded key/value across all three heads.
// No block-wide synchronization in the history loop.
KERNELB(kv_stripes, AttnArgs, 256, 1) {
  KSHARED(float, vectors, 8 * 128);
  KSHARED(float, maxima, 8);
  KSHARED(float, masses, 8);
  int lane = KTID & 31, warp = KTID >> 5;
  int kh = KBID % a.nkv, s = (KBID / a.nkv) % a.splits;
  int row = KBID / (a.nkv * a.splits), pos = a.st[row].pos;
  int T = pos + 1;
  if (a.window && T > a.window) T = a.window;
  int active = (T + 127) / 128;
  if (active > a.splits) active = a.splits;
  int span = (T + active - 1) / active;
  int start = pos + 1 - T + s * span, end = start + span;
  if (end > pos + 1) end = pos + 1;
  const size_t base = (size_t)row * a.nh + kh * 3;
  const int C = a.nkv * 128, col = kh * 128 + lane * 4;
  KF4 q[3], o[3];
  float r[3], mx[3], mass[3];
  KUNROLL
  for (int h = 0; h < 3; ++h) {
    q[h] = KLOADF4(a.q + (base + h) * 128 + lane * 4);
    r[h] = lane < a.drel ? a.r[(base + h) * a.drel + lane] : 0;
    o[h] = KF4ZERO;
    mx[h] = -3.402823466e38f;
    mass[h] = 0;
  }
  const u16* rp = a.rel_proj_t ? a.rel_proj_t + (lane < a.drel ? lane : 0) : a.rel_proj + (size_t)(lane < a.drel ? lane : 0) * a.rel_extent;
  const int rs = a.rel_proj_t ? a.drel : 1;
  for (int t = start + warp; t < end; t += 8) {
    int slot = a.ring ? t % a.ring : t, dist = pos - t;
    int ns = kv_near(a.kv, t, pos);
    KF4 k = kv_read4(a.kv, a.kc, 0, slot, ns, C, col);
    KF4 v = kv_read4(a.kv, a.vc, 1, slot, ns, C, col);
    float rel = (dist < a.rel_extent && lane < a.drel) ? k_bf(rp[(size_t)dist * rs]) : 0;
    KUNROLL
    for (int h = 0; h < 3; ++h) {
      float x = KFMA(q[h].x, k.x, 0.0f);
      x = KFMA(q[h].y, k.y, x); x = KFMA(q[h].z, k.z, x); x = KFMA(q[h].w, k.w, x);
      x = k_warp_sum(KCTXA KFMA(r[h], rel, x * a.scale));
      if (x > mx[h]) {
        float old = expf(mx[h] - x);
        mx[h] = x; mass[h] = KFMA(mass[h], old, 1.0f);
        o[h].x = KFMA(o[h].x, old, v.x); o[h].y = KFMA(o[h].y, old, v.y);
        o[h].z = KFMA(o[h].z, old, v.z); o[h].w = KFMA(o[h].w, old, v.w);
      } else {
        float w = expf(x - mx[h]);
        mass[h] += w;
        o[h].x = KFMA(w, v.x, o[h].x); o[h].y = KFMA(w, v.y, o[h].y);
        o[h].z = KFMA(w, v.z, o[h].z); o[h].w = KFMA(w, v.w, o[h].w);
      }
    }
  }
  KUNROLL
  for (int h = 0; h < 3; ++h) {
    kv_put4(vectors + warp * 128 + lane * 4, o[h]);
    if (!lane) { maxima[warp] = mx[h]; masses[warp] = mass[h]; }
    KSYNC();
    if (KTID < 128) {
      float sink = a.sink[base + h];
      float m = a.splits == 1 ? sink : -3.402823466e38f;
      for (int i = 0; i < 8; ++i) m = fmaxf(m, maxima[i]);
      float l = a.splits == 1 ? expf(sink - m) : 0, val = 0;
      for (int i = 0; i < 8; ++i) {
        float w = expf(maxima[i] - m);
        l = KFMA(w, masses[i], l);
        val = KFMA(w, vectors[i * 128 + KTID], val);
      }
      if (a.splits == 1) {
        size_t idx = (base + h) * 128 + KTID;
        a.out[idx] = val / l * (a.gate ? a.gate[idx] : 1.0f);
      } else {
        float* p = a.partial + ((base + h) * a.splits + s) * 130;
        p[KTID] = val;
        if (!KTID) { p[128] = m; p[129] = l; }
      }
    }
    KSYNC();
  }
}

KERNEL(kv_merge, AttnArgs) {
  size_t idx = KBID;
  int d = KTID;
  const float* p = a.partial + idx * a.splits * 130;
  float sink = a.sink[idx], m = sink;
  for (int s = 0; s < a.splits; ++s) m = fmaxf(m, p[s * 130 + 128]);
  float den = expf(sink - m), o = 0;
  for (int s = 0; s < a.splits; ++s) {
    float w = expf(p[s * 130 + 128] - m);
    den = KFMA(w, p[s * 130 + 129], den);
    o = KFMA(w, p[s * 130 + d], o);
  }
  a.out[idx * 128 + d] = (o / den) * (a.gate ? a.gate[idx * 128 + d] : 1.0f);
}

KERNEL(kv_capture, KvCaptureArgs) {
  int col = (KBID * KNT + KTID) * 4;
  if (col >= a.channels) return;
  int t = a.st[0].pos, slot = a.ring ? t % a.ring : t, near_slot = kv_near(a.kv, t, t);
  kv_put4(a.k + col, kv_read4(a.kv, nullptr, 0, slot, near_slot, a.channels, col));
  kv_put4(a.v + col, kv_read4(a.kv, nullptr, 1, slot, near_slot, a.channels, col));
}
