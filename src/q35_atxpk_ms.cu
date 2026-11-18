// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Multi-stream pass (experimental): the one-row-per-CTA form of the attention kernel of q35_atxpk3.cuh for rows of several streams.
// A row reads the K/V rows of its own stream (tables in Q35PkMs) up to its own position.  These kernels are compiled in a translation
// unit of their own: the compiler then makes the single-request kernels of q35_atxpk.cu exactly as without them (helpers that more
// kernels of one unit share can be compiled differently).  q35_atxpk.cu is included for its kernel helpers and argument structs
// (PkRow, pk3_smem, the pk3_*_env settings); YSKN_ATXPK_MS_TU leaves out its launchers, q35_atxpk4.cuh and pk3_set_all, so
// that no single-request kernel is made in this unit.
#define YSKN_ATXPK_MS_TU 1
#include "q35_atxpk.cu"
namespace yskn {

// the CTA of row t0 holds rows t0 .. t0 + TPC - 1 as far as they are rows of the same stream that follow each other, and only when
// t0 is row 0, TPC, 2 TPC, .. of its stream in this pass; the CTAs of the other rows have nothing to do.  base = the position of the
// first row of the stream in this pass.
template <int TPC> __device__ __forceinline__ PkRow pk_row_ms(const Q35AtxPkArgs& U, const Q35PkMs& X, int base, int t0, int tile, int gi, int h, int kvh, int split) {
  const Q35AttnArgs& A = U.a;
  constexpr int G = Q35_NH / Q35_KVH;
  PkRow r;
  const int i = 16 * tile + gi + 8 * h; r.tk = t0 + i / G; r.hd = kvh * G + i % G; r.v = i < G * TPC && r.tk < A.M;
  if (r.v) r.v = X.ms.tl[t0] % TPC == 0 && X.ms.tl[r.tk] == X.ms.tl[t0] + (r.tk - t0);
  const int dp = r.v ? X.ms.tl[r.tk] : -1;                                 // the row's index among the rows of its stream in this pass
  const int len = r.v ? base + dp + 1 : 0; r.own = base + dp;
  const int S = (A.splits_short > 0 && len < A.long_len) ? A.splits_short : A.splits;
  r.act = r.v && split < S;
  const int per = (len + S - 1) / S;
  r.lo = split * per; const int hi = min(len, r.lo + per);
  r.ph = max(r.lo, min(hi, len - 1));
  r.self_ = r.v && r.lo <= len - 1 && len - 1 < r.lo + per;
  return r;
}

template <int KB, bool VPK, int RT, int WPT, int NST, bool SKEW, int XW, bool QK8, bool PLUT = false>
__global__ void __launch_bounds__((RT * WPT + XW) * 32, 1) q35_atxpk3_ms_kernel(Q35AtxPkArgs U, Q35PkMs X) {
  constexpr int TPC = (16 * RT) / 6, JW = 4 / WPT, NB = 32 / WPT, TL = PK3_TL, PB = 32 * KB, NWM = RT * WPT, NW = NWM + XW;
  static_assert(KB == 4 || KB == 5, "CDQ k = 4 or 5");
  static_assert(NST >= 2 && NST <= 3, "stage ring of 2 or 3 tiles (4 does not fit in shared memory with an INT8 V stage)");
  static_assert(XW == 0 || (RT == 1 && XW == 4 && SKEW), "extra (converter-only) warps: 4 of them, with the 1 x 4 shape, SKEW");
  static_assert(!SKEW || NW == 12 || (RT == 1 && XW == 4), "SKEW: 12-warp CTAs or 4 + 4 warps (group B = warps 4..7)");
  static_assert(!PLUT || (RT == 1 && !QK8 && (KB == 4 || NST == 2)), "PLUT: the 1 x 4 shape, BF16 scores; k = 5 with a 2-tile stage ring (shared memory)");
  static_assert(pk3_smem<KB, VPK, RT, WPT, NST, SKEW, PLUT>() + pk3_static<RT>() <= 101376, "shared memory: more than the 99 KB (101,376 B) a CTA may use on sm_120");
  const Q35AttnArgs& A = U.a;
  constexpr int D = Q35_HD;
  extern __shared__ __align__(16) uint16_t usm[];
  __shared__ int s_lo, s_hi;
  __shared__ float2 x_mx[RT][4][32], x_sm[RT][4][32];
  __shared__ uint2 x_pf[RT][4][32];
  __shared__ float vsc2[2][PK_KT];
  const int t0 = (int)blockIdx.x, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = warp / WPT, part = warp % WPT, gi = lane >> 2, tig = lane & 3;
  const bool early = A.early != 0;
  const bool grpB = SKEW && (warp >> 2) == 1;
  constexpr int NCONV = SKEW ? 128 : NW * 32;
  const int cid = SKEW ? threadIdx.x - 128 : threadIdx.x;

  constexpr int KP = PB, VP = VPK ? PB : D, KREG = PK_KT * KP, VREG = PK_KT * VP, SREG = KREG + VREG;
  uint16_t* const vt_b = usm;
  uint8_t* const stg = reinterpret_cast<uint8_t*>(usm + 2 * PK_KT * TL);
  float* const ssc = reinterpret_cast<float*>(stg + (size_t)NST * SREG);
  uint16_t* const tabk = reinterpret_cast<uint16_t*>(ssc + NST * 2 * PK_KT);
  uint16_t* const tabv = tabk + 256;
  uint8_t* const tab8k = reinterpret_cast<uint8_t*>(tabv + (VPK ? 256 : 0));
  unsigned* const plutk = reinterpret_cast<unsigned*>(tab8k + 272);
  unsigned* const plutv = plutk + (KB == 4 ? 4096 : 8192);
  const unsigned char* const rk = U.rec + (size_t)((U.layer * Q35_KVH + kvh) * 2) * PK_REC;
  for (int i = threadIdx.x; i < 32; i += blockDim.x) {
    reinterpret_cast<uint4*>(tabk)[i] = reinterpret_cast<const uint4*>(rk)[i];
    if (VPK) reinterpret_cast<uint4*>(tabv)[i] = reinterpret_cast<const uint4*>(rk + PK_REC)[i];
    if (QK8 && i < 17) reinterpret_cast<uint4*>(tab8k)[i] = reinterpret_cast<const uint4*>(rk + 1536)[i];
  }
  if constexpr (PLUT) {
    const uint16_t* const gk = reinterpret_cast<const uint16_t*>(rk);
    const uint16_t* const gv = reinterpret_cast<const uint16_t*>(rk + PK_REC);
    if constexpr (KB == 4) {
      for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        const unsigned p = i >> 8, a = (i >> 4) & 15, b = i & 15;
        plutk[i] = (unsigned)gk[(p << 4) | a] | ((unsigned)gk[(a << 4) | b] << 16);
        if (VPK) plutv[i] = (unsigned)gv[(p << 4) | a] | ((unsigned)gv[(a << 4) | b] << 16);
      }
    } else {
      for (int i = threadIdx.x; i < 8192; i += blockDim.x) plutk[i] = (unsigned)gk[(i >> 5) & 255] | ((unsigned)gk[i & 255] << 16);
    }
  }
  if (!early) { pdl_wait(); pdl_trigger(); }
  const int base = X.ms.pos[t0] - X.ms.tl[t0];            // keys below it were stored before this pass
  const PkRow r0 = pk_row_ms<TPC>(U, X, base, t0, tile, gi, 0, kvh, split), r1 = pk_row_ms<TPC>(U, X, base, t0, tile, gi, 1, kvh, split);
  const int lo0 = r0.lo, lo1 = r1.lo, ph0 = r0.ph, ph1 = r1.ph;
  const bool vrow0 = r0.v, vrow1 = r1.v;
  if (!__syncthreads_or(r0.act || r1.act)) { if (early) { pdl_wait(); pdl_trigger(); } return; }
  if (threadIdx.x == 0) { s_lo = 0x7fffffff; s_hi = 0; }
  __syncthreads();
  if (vrow0 && ph0 > lo0) { atomicMin(&s_lo, lo0); atomicMax(&s_hi, ph0); }
  if (vrow1 && ph1 > lo1) { atomicMin(&s_lo, lo1); atomicMax(&s_hi, ph1); }
  __syncthreads();
  const int klo = s_lo, khi = s_hi;
  const int tbeg = khi > klo ? (klo / PK_KT) * PK_KT : 0;
  const int ntiles = khi > klo ? (khi - tbeg + PK_KT - 1) / PK_KT : 0;
  const int msl = X.ms.slot[t0];                                 // the stream of this CTA's row
  const uint8_t* const kpg = X.kp[msl] + (size_t)kvh * A.max_ctx * PB;
  const uint8_t* const vpg = VPK ? X.vp[msl] + (size_t)kvh * A.max_ctx * PB : nullptr;
  const int8_t* const vg8 = VPK ? nullptr : X.v8[msl] + (size_t)kvh * A.max_ctx * D;
  const float* const ksg = X.kps[msl] + (size_t)kvh * A.max_ctx;
  const float* const vsg = (VPK ? X.vps[msl] : X.vs[msl]) + (size_t)kvh * A.max_ctx;
  constexpr int GK = 2 * KB, GV = VPK ? 2 * KB : 16;
  auto load_tile = [&](int t, int pt) {
    const int k0 = tbeg + t * PK_KT; uint8_t* const sb = stg + (size_t)(t % NST) * SREG; float* const sc = ssc + (t % NST) * 2 * PK_KT;
    auto mine = [&](int kidx) { return pt == 0 || (pt == 1 ? kidx < base : kidx >= base); };
    for (int c = threadIdx.x; c < PK_KT * (GK + GV); c += blockDim.x) {
      const int kv = c >= PK_KT * GK, cc = kv ? c - PK_KT * GK : c, ng = kv ? GV : GK;
      const int key = cc / ng, g = cc - key * ng;
      uint8_t* const dst = sb + (kv ? KREG : 0) + (size_t)key * (kv ? VP : KP) + g * 16;
      const int kidx = k0 + key;
      if (kidx >= klo && kidx < khi) {
        if (mine(kidx)) {
          const void* src;
          if (kv) { if constexpr (VPK) src = vpg + (size_t)kidx * PB + g * 16; else src = vg8 + (size_t)kidx * D + g * 16; }
          else    src = kpg + (size_t)kidx * PB + g * 16;
          pk_cp16(dst, src);
        }
      } else if (pt != 2) *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = threadIdx.x; c < 2 * PK_KT; c += blockDim.x) {
      const int kv = c / PK_KT, key = c % PK_KT, kidx = k0 + key;
      float* const dst = sc + kv * PK_KT + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) pk_cp4(dst, (kv ? vsg : ksg) + kidx); } else if (pt != 2) *dst = 0.f;
    }
  };

  auto convert_v = [&](int t) {
    const int b = t & 1; uint16_t* const vbuf = vt_b + (size_t)b * PK_KT * TL;
    const uint8_t* const sv = stg + (size_t)(t % NST) * SREG + KREG; const float* const sc = ssc + (t % NST) * 2 * PK_KT + PK_KT;
    for (int cc = cid; cc < PK_KT * 16; cc += NCONV) {
      int key, ch; unsigned w[8];
      if constexpr (VPK) {
        if constexpr (KB == 4) {
          key = cc >> 4; ch = cc & 15;
          if constexpr (PLUT) pk3_conv16_k4_lut(reinterpret_cast<const unsigned*>(sv + key * PB), ch, plutv, w);
          else pk_conv16_k4(reinterpret_cast<const unsigned*>(sv + key * PB), ch, tabv, w);
        } else {
          const int hf = cc >> 8, l = cc & 31;
          key = 4 * ((cc >> 5) & 7) + ((l >> 1) & 3);
          const int chp = 2 * ((l >> 3) & 3) + (l & 1);
          ch = 2 * chp + hf;
          const unsigned* const rw = reinterpret_cast<const unsigned*>(sv + key * PB);
          if (hf == 0) pk_conv16_k5<0>(rw, chp, tabv, w); else pk_conv16_k5<1>(rw, chp, tabv, w);
        }
      } else {
        key = cc >> 4; ch = cc & 15;
        const uint4 u = *reinterpret_cast<const uint4*>(sv + key * D + ch * 16);
        pk_s8x4_bf16(u.x, w[0], w[1]); pk_s8x4_bf16(u.y, w[2], w[3]); pk_s8x4_bf16(u.z, w[4], w[5]); pk_s8x4_bf16(u.w, w[6], w[7]);
      }
      uint16_t* const row = vbuf + key * TL; const int sw = key & 7;
      *reinterpret_cast<uint4*>(row + 8 * ((2 * ch) ^ sw)) = make_uint4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<uint4*>(row + 8 * ((2 * ch + 1) ^ sw)) = make_uint4(w[4], w[5], w[6], w[7]);
    }
    for (int c = cid; c < PK_KT; c += NCONV) vsc2[b][c] = sc[c];
  };
  if (0 < ntiles) load_tile(0, early ? 1 : 0);
  pk_commit();
  if (early) { pdl_wait(); pdl_trigger(); if (0 < ntiles) load_tile(0, 2); pk_commit(); }
  #pragma unroll
  for (int p = 1; p < NST; ++p) { if (p < ntiles) load_tile(p, 0); pk_commit(); }
  unsigned qa[QK8 ? 8 : 16][4];
  float sq0 = 1.f, sq1 = 1.f;
  {
    const unsigned* q0 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow0 ? r0.tk : 0) * Q35_NH + r0.hd) * D);
    const unsigned* q1 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow1 ? r1.tk : 0) * Q35_NH + r1.hd) * D);
    if constexpr (!QK8) {
      #pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        qa[ks][0] = vrow0 ? q0[ks * 8 + tig] : 0u; qa[ks][1] = vrow1 ? q1[ks * 8 + tig] : 0u;
        qa[ks][2] = vrow0 ? q0[ks * 8 + 4 + tig] : 0u; qa[ks][3] = vrow1 ? q1[ks * 8 + 4 + tig] : 0u;
      }
    } else {
      float qf[2][8][8]; float mx0 = 0.f, mx1 = 0.f;
      #pragma unroll
      for (int ks8 = 0; ks8 < 8; ++ks8) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const unsigned* q = h ? q1 : q0; const bool v = h ? vrow1 : vrow0;
          const unsigned wa = v ? q[16 * ks8 + 2 * tig] : 0u, wb = v ? q[16 * ks8 + 2 * tig + 1] : 0u, wc = v ? q[16 * ks8 + 8 + 2 * tig] : 0u, wd = v ? q[16 * ks8 + 8 + 2 * tig + 1] : 0u;
          qf[h][ks8][0] = pk_bf_lo(wa); qf[h][ks8][1] = pk_bf_hi(wa); qf[h][ks8][2] = pk_bf_lo(wb); qf[h][ks8][3] = pk_bf_hi(wb);
          qf[h][ks8][4] = pk_bf_lo(wc); qf[h][ks8][5] = pk_bf_hi(wc); qf[h][ks8][6] = pk_bf_lo(wd); qf[h][ks8][7] = pk_bf_hi(wd);
          #pragma unroll
          for (int i = 0; i < 8; ++i) { if (h) mx1 = fmaxf(mx1, fabsf(qf[1][ks8][i])); else mx0 = fmaxf(mx0, fabsf(qf[0][ks8][i])); }
        }
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      sq0 = mx0 > 0.f ? mx0 * (1.f / 127.f) : 1.f; sq1 = mx1 > 0.f ? mx1 * (1.f / 127.f) : 1.f;
      const float inv0 = 1.f / sq0, inv1 = 1.f / sq1;
      #pragma unroll
      for (int ks8 = 0; ks8 < 8; ++ks8) {
        qa[ks8][0] = pk3_q8pack(qf[0][ks8][0], qf[0][ks8][1], qf[0][ks8][2], qf[0][ks8][3], inv0); qa[ks8][1] = pk3_q8pack(qf[1][ks8][0], qf[1][ks8][1], qf[1][ks8][2], qf[1][ks8][3], inv1);
        qa[ks8][2] = pk3_q8pack(qf[0][ks8][4], qf[0][ks8][5], qf[0][ks8][6], qf[0][ks8][7], inv0); qa[ks8][3] = pk3_q8pack(qf[1][ks8][4], qf[1][ks8][5], qf[1][ks8][6], qf[1][ks8][7], inv1);
      }
    }
  }
  const float st8 = QK8 ? *reinterpret_cast<const float*>(rk + 1792) : 1.f;
  pk_waitn<NST - 1>();
  __syncthreads();
  if (!SKEW && 0 < ntiles) convert_v(0);
  const bool busy = __any_sync(0xffffffffu, vrow0 || vrow1);
  float o[NB][4];
  #pragma unroll
  for (int n = 0; n < NB; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  float s[JW][4], c0 = 1.f, c1 = 1.f;
  for (int t = 0; t <= ntiles; ++t) {
    __syncthreads();
    if (SKEW && grpB && t < ntiles) convert_v(t);
    if (busy && t >= 1) {
      float r0s = 0.f, r1s = 0.f;
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) { const float2 q = x_sm[tile][jj][lane]; r0s += q.x; r1s += q.y; }
      r0s += __shfl_xor_sync(0xffffffffu, r0s, 1); r0s += __shfl_xor_sync(0xffffffffu, r0s, 2);
      r1s += __shfl_xor_sync(0xffffffffu, r1s, 1); r1s += __shfl_xor_sync(0xffffffffu, r1s, 2);
      l0 = l0 * c0 + r0s; l1 = l1 * c1 + r1s;
      if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {
        #pragma unroll
        for (int n = 0; n < NB; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
      }
      const uint16_t* const vs = vt_b + (size_t)((t - 1) & 1) * PK_KT * TL;
      #pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const uint2 pA = x_pf[tile][2 * kk][lane], pB = x_pf[tile][2 * kk + 1][lane];
        const unsigned pa[4] = {pA.x, pA.y, pB.x, pB.y};
        #pragma unroll
        for (int q = 0; q < NB / 2; ++q) {
          const int np = (NB / 2) * part + q;
          unsigned bb[4];
          const int r = 16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7), c = 2 * np + (lane >> 4);
          pk_ldm4t(bb, vs + r * TL + 8 * (c ^ (r & 7)));
          pk_mma(o[2 * q], pa, bb[0], bb[1]);
          pk_mma(o[2 * q + 1], pa, bb[2], bb[3]);
        }
      }
    }
    if (t == ntiles) break;
    if (busy) {
      const uint8_t* const sk = stg + (size_t)(t % NST) * SREG; const float* const ksc = ssc + (t % NST) * 2 * PK_KT;
      const int kt0 = tbeg + t * PK_KT;
      const unsigned* krow[JW];
      #pragma unroll
      for (int w = 0; w < JW; ++w) { const int j = JW * part + w; krow[w] = reinterpret_cast<const unsigned*>(sk + (size_t)(8 * j + gi) * KP); }
      if constexpr (!QK8) {
        unsigned prev[JW];
        #pragma unroll
        for (int w = 0; w < JW; ++w) { prev[w] = 0u; s[w][0] = s[w][1] = s[w][2] = s[w][3] = 0.f; }
        #pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
          #pragma unroll
          for (int w = 0; w < JW; ++w) {
            unsigned b0, b1;
            if constexpr (PLUT) pk3_kfrag_lut<KB>(krow[w], ks, tig, plutk, prev[w], b0, b1);
            else pk3_kfrag<KB>(krow[w], ks, tig, tabk, prev[w], b0, b1);
            pk_mma(s[w], qa[ks], b0, b1);
          }
        }
      } else {
        int si[JW][4];
        #pragma unroll
        for (int w = 0; w < JW; ++w) si[w][0] = si[w][1] = si[w][2] = si[w][3] = 0;
        #pragma unroll
        for (int ks8 = 0; ks8 < 8; ++ks8) {
          #pragma unroll
          for (int w = 0; w < JW; ++w) {
            unsigned b0, b1;
            pk3_kfrag8<KB>(krow[w], ks8, tig, tab8k, b0, b1);
            pk3_mma_s8(si[w], qa[ks8], b0, b1);
          }
        }
        #pragma unroll
        for (int w = 0; w < JW; ++w) { s[w][0] = (float)si[w][0] * sq0; s[w][1] = (float)si[w][1] * sq0; s[w][2] = (float)si[w][2] * sq1; s[w][3] = (float)si[w][3] * sq1; }
      }
      #pragma unroll
      for (int w = 0; w < JW; ++w) {
        const int j = JW * part + w;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int kk = 8 * j + 2 * tig + (c & 1), key = kt0 + kk, h = c >> 1;
          const float v = s[w][c] * ((QK8 ? st8 : 1.f) * (ksc[kk] * 0.0625f));
          s[w][c] = (key >= (h ? lo1 : lo0) && key < (h ? ph1 : ph0)) ? v : -INFINITY;
        }
        x_mx[tile][j][lane] = make_float2(fmaxf(s[w][0], s[w][1]), fmaxf(s[w][2], s[w][3]));
      }
    }
    pk_waitn<NST - 2>();
    __syncthreads();
    if (t + NST < ntiles) load_tile(t + NST, 0);
    pk_commit();
    if (busy) {
      const float* const vsc = vsc2[t & 1];
      float x0 = m0, x1 = m1;
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) { const float2 q = x_mx[tile][jj][lane]; x0 = fmaxf(x0, q.x); x1 = fmaxf(x1, q.y); }
      x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1)); x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
      x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1)); x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
      c0 = exp2f((m0 - x0) * PK_LOG2E); c1 = exp2f((m1 - x1) * PK_LOG2E);
      m0 = x0; m1 = x1;
      const float y0 = x0 * PK_LOG2E, y1 = x1 * PK_LOG2E;
      #pragma unroll
      for (int w = 0; w < JW; ++w) {
        const int j = JW * part + w;
        s[w][0] = exp2f(s[w][0] * PK_LOG2E - y0); s[w][1] = exp2f(s[w][1] * PK_LOG2E - y0);
        s[w][2] = exp2f(s[w][2] * PK_LOG2E - y1); s[w][3] = exp2f(s[w][3] * PK_LOG2E - y1);
        x_sm[tile][j][lane] = make_float2(s[w][0] + s[w][1], s[w][2] + s[w][3]);
        const float va = vsc[8 * j + 2 * tig], vb = vsc[8 * j + 2 * tig + 1];
        x_pf[tile][j][lane] = make_uint2(pk_pack(s[w][0] * va, s[w][1] * vb), pk_pack(s[w][2] * va, s[w][3] * vb));
      }
    }
    if (!SKEW && t + 1 < ntiles) convert_v(t + 1);
  }

  const int base2 = X.ms.pos[t0] - X.ms.tl[t0];
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const PkRow r = pk_row_ms<TPC>(U, X, base2, t0, tile, gi, h, kvh, split);
    float& m = h ? m1 : m0; float& l = h ? l1 : l0;
    if (busy) {
      const bool sf = r.self_;
      float pt = 0.f; int pti = 0;
      if (sf) {
        const size_t row = (size_t)r.own;
        const unsigned* const kw = reinterpret_cast<const unsigned*>(kpg + row * PB);
        if constexpr (!QK8) {
          #pragma unroll
          for (int kq = 0; kq < 16; ++kq) {
            const unsigned qlo = qa[kq][h], qhi = qa[kq][2 + h];
            const float k0 = pk_tabf(tabk, pk3_state<KB>(kw, 16 * kq + 2 * tig)), k1 = pk_tabf(tabk, pk3_state<KB>(kw, 16 * kq + 2 * tig + 1));
            const float k2 = pk_tabf(tabk, pk3_state<KB>(kw, 16 * kq + 8 + 2 * tig)), k3 = pk_tabf(tabk, pk3_state<KB>(kw, 16 * kq + 9 + 2 * tig));
            pt = __fadd_rn(pt, __fmul_rn(pk_bf_lo(qlo), k0)); pt = __fadd_rn(pt, __fmul_rn(pk_bf_hi(qlo), k1));
            pt = __fadd_rn(pt, __fmul_rn(pk_bf_lo(qhi), k2)); pt = __fadd_rn(pt, __fmul_rn(pk_bf_hi(qhi), k3));
          }
        } else {
          #pragma unroll
          for (int ks8 = 0; ks8 < 8; ++ks8) {
            unsigned b0, b1; pk3_kfrag8<KB>(kw, ks8, tig, tab8k, b0, b1);
            const unsigned a0 = qa[ks8][h], a1 = qa[ks8][2 + h];
            pti = __dp4a((int)a0, (int)b0, pti); pti = __dp4a((int)a1, (int)b1, pti);
          }
        }
      }
      if constexpr (!QK8) { pt = __fadd_rn(pt, __shfl_xor_sync(0xffffffffu, pt, 1)); pt = __fadd_rn(pt, __shfl_xor_sync(0xffffffffu, pt, 2)); }
      else { pti += __shfl_xor_sync(0xffffffffu, pti, 1); pti += __shfl_xor_sync(0xffffffffu, pti, 2); pt = (float)pti * (h ? sq1 : sq0); }
      if (sf) {
        const float sc = pt * ((QK8 ? st8 : 1.f) * (ksg[r.own] * 0.0625f));
        const float mn = fmaxf(m, sc), c = exp2f((m - mn) * PK_LOG2E), p = exp2f((sc - mn) * PK_LOG2E);
        m = mn; l = l * c + p;
        const float pv = p * vsg[r.own];
        #pragma unroll
        for (int n = 0; n < NB; ++n) {
          const int e = 8 * (NB * part + n) + 2 * tig;
          float v0, v1;
          if constexpr (VPK) {
            const unsigned* const vw = reinterpret_cast<const unsigned*>(vpg + (size_t)r.own * PB);
            v0 = pk_tabf(tabv, pk3_state<KB>(vw, e)); v1 = pk_tabf(tabv, pk3_state<KB>(vw, e + 1));
          } else {
            const int8_t* vr = vg8 + (size_t)r.own * D + e; v0 = (float)vr[0]; v1 = (float)vr[1];
          }
          o[n][2 * h] = o[n][2 * h] * c + pv * v0; o[n][2 * h + 1] = o[n][2 * h + 1] * c + pv * v1;
        }
      }
    }
    if (!r.v) continue;
    if constexpr (VPK) {
      const float* const muv = reinterpret_cast<const float*>(rk + PK_REC + 512);
      #pragma unroll
      for (int n = 0; n < NB; ++n) {
        const float2 mu = *reinterpret_cast<const float2*>(muv + 8 * (NB * part + n) + 2 * tig);
        o[n][2 * h] += l * mu.x; o[n][2 * h + 1] += l * mu.y;
      }
    }
    const size_t pidx = ((size_t)r.tk * A.splits + split) * Q35_NH + r.hd;
    if (tig == 0 && part == 0) { A.pm[pidx] = m; A.pl[pidx] = l; }
    float* const pa = A.pacc + pidx * D + 2 * tig;
    #pragma unroll
    for (int n = 0; n < NB; ++n) *reinterpret_cast<float2*>(pa + 8 * (NB * part + n)) = make_float2(o[n][2 * h], o[n][2 * h + 1]);
  }
}

// Launches: the kernel shape pk3_launch takes for a pass of at most 2 rows with the settings of the Python package (YSKN_ATX_PK3_PLUT=1,
// YSKN_ATX_PK3_QK8 and YSKN_ATX_PK3_NST unset: no INT8 scores, a 3-tile stage ring for k = 4; k = 5 takes 2 tiles), one CTA per row.
// Only these four kernels (k = 4 / 5, V packed or INT8) are built: the other settings of pk3_launch are refused here and in ms_bind
// (q35_atx_pk_ms_unsupported), so that the library does not carry 34 kernels no multi-stream pass of the package can reach.
template <int KB, bool VPK, int NST> static void pk3ms_set_lut(cudaError_t& e) {
  const cudaError_t r = cudaFuncSetAttribute((const void*)q35_atxpk3_ms_kernel<KB, VPK, 1, 4, NST, true, 4, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pk3_smem<KB, VPK, 1, 4, NST, true, true>());
  if (r != cudaSuccess && e == cudaSuccess) e = r;
}
template <int KB, bool VPK, int NST> static void pk3ms_go_lut(const Q35AtxPkArgs& U, const Q35PkMs& X, cudaStream_t st) {
  launch_k(q35_atxpk3_ms_kernel<KB, VPK, 1, 4, NST, true, 4, false, true>, dim3(U.a.M, U.a.splits, Q35_KVH), dim3(256), pk3_smem<KB, VPK, 1, 4, NST, true, true>(), st, U, X);
}
static constexpr int pk3ms_nst(int kb) { return kb == 5 ? 2 : 3; }      // the stage ring of the built kernels
template <int KB, bool VPK> static void pk3ms_launch(const Q35AtxPkArgs& U, const Q35PkMs& X, cudaStream_t st) {
  pk3ms_go_lut<KB, VPK, pk3ms_nst(KB)>(U, X, st);
}
static void pk3ms_set_all(cudaError_t& e) {
  pk3ms_set_lut<4, true, pk3ms_nst(4)>(e); pk3ms_set_lut<4, false, pk3ms_nst(4)>(e); pk3ms_set_lut<5, true, pk3ms_nst(5)>(e); pk3ms_set_lut<5, false, pk3ms_nst(5)>(e);
}
// "" when this process's attention settings take a kernel built above (what pk3_launch would choose for <= 2 rows), else why not
std::string q35_atx_pk_ms_unsupported(int kb) {
  if (kb != 4 && kb != 5) return "the KV profile packs " + std::to_string(kb) + "-bit codes (the multi-stream pass takes 4 or 5)";
  if (pk3_qk8_env()) return "YSKN_ATX_PK3_QK8 is set (the multi-stream pass is built for the settings of the Python package: unset it)";
  if (!pk3_plut_env()) return "YSKN_ATX_PK3_PLUT is not 1 (the multi-stream pass is built for the settings of the Python package: set it to 1)";
  if (kb == 4 && pk3_nst_env() != pk3ms_nst(4)) return "YSKN_ATX_PK3_NST is not 3 (the multi-stream pass is built for the settings of the Python package: unset it)";
  return std::string();
}
// multi-stream pass (experimental): one CTA per row, each row with the KV containers of its own stream (above)
void launch_q35_atx_pk_ms(const Q35AttnArgs& A, const Q35PkArgs& P, const Q35PkMs& X, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    cudaError_t e = cudaSuccess;
    pk3ms_set_all(e);
    if (e != cudaSuccess) throw std::runtime_error(std::string("launch_q35_atx_pk_ms: cudaFuncSetAttribute failed: ") + cudaGetErrorString(e));
    attr = true; }
  if (A.kv_fp16) throw std::runtime_error("launch_q35_atx_pk_ms: FP16 KV");
  if (A.M < 1 || A.M > 16) throw std::runtime_error("launch_q35_atx_pk_ms: 1..16 rows");
  if (!P.rec || !P.kp || !P.kps || !X.ms.pos || !X.ms.slot || !X.ms.tl || !X.kp || !X.kps || (P.vp ? (!P.vps || !X.vp || !X.vps) : (!A.v8 || !X.v8 || !X.vs)))
    throw std::runtime_error("launch_q35_atx_pk_ms: a multi-stream pass needs packed K and the stream tables");
  { const std::string why = q35_atx_pk_ms_unsupported(P.kb); if (!why.empty()) throw std::runtime_error("launch_q35_atx_pk_ms: " + why); }
  const Q35AtxPkArgs U{A, nullptr, P.kp, P.vp, P.kps, P.vps, P.rec, P.layer};
  if (P.kb == 4) { if (P.vp) pk3ms_launch<4, true>(U, X, st); else pk3ms_launch<4, false>(U, X, st); }
  else           { if (P.vp) pk3ms_launch<5, true>(U, X, st); else pk3ms_launch<5, false>(U, X, st); }
}
}
