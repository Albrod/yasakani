// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Self-test: every kernel on the same random inputs with two backends (the
// CPU emulation and the backend under test); the largest difference per
// output is reported, so that a failure names the kernel.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "ink2_kargs.h"

void ink2_make_tables(float* e4m3, float* exp2t);

namespace {

std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

struct TOut {
  std::string name;
  void* p;
  size_t n;
  bool is_int;
};

struct TB {
  Backend* b;
  unsigned long long seed = 0x9E3779B97F4A7C15ull;
  std::vector<TOut> outs;
  std::vector<void*> allocs;
  explicit TB(Backend* be) : b(be) {}
  ~TB() {
    for (void* p : allocs) b->release(p);
  }
  unsigned long long next() {
    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed ^= seed >> 27;
    return seed * 0x2545F4914F6CDD1Dull;
  }
  float uni(float lo, float hi) { return lo + (hi - lo) * (float)((next() >> 40) * (1.0 / 16777216.0)); }
  void* up(const void* host, size_t bytes) {
    void* p = b->alloc(bytes);
    allocs.push_back(p);
    b->upload(p, host, bytes);
    return p;
  }
  float* f(size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto& x : v) x = uni(lo, hi);
    return (float*)up(v.data(), n * 4);
  }
  u16* bf(size_t n, float lo, float hi) {
    std::vector<u16> v(n);
    for (auto& x : v) {
      float fl = uni(lo, hi);
      unsigned bits;
      memcpy(&bits, &fl, 4);
      x = (u16)(bits >> 16);
    }
    return (u16*)up(v.data(), n * 2);
  }
  u8* bytes(size_t n, int lo, int hi, bool no_nan) {
    std::vector<u8> v(n);
    for (auto& x : v) {
      x = (u8)(lo + (int)(next() >> 33) % (hi - lo + 1));
      if (no_nan && (x & 0x7F) == 0x7F) x ^= 1;
    }
    return (u8*)up(v.data(), n);
  }
  int* ints(const std::vector<int>& v) { return (int*)up(v.data(), v.size() * 4); }
  StepState* st(int token, int pos, float tau) {
    StepState s{token, pos, tau, 0};
    return (StepState*)up(&s, sizeof s);
  }
  float* zf(size_t n) {
    void* p = b->alloc(n * 4);
    allocs.push_back(p);
    return (float*)p;
  }
  void out(const char* name, void* p, size_t n, bool is_int = false) { outs.push_back({name, p, n, is_int}); }
};

void t_gemvc(TB& t) {
  GemvArgs a{};
  const int rows[3] = {37, 128, 5}, cols = 2304;
  const int epi[3] = {INK2_EPI_NONE, INK2_EPI_SIGMOID, INK2_EPI_SOFTCAP};
  int chunks = 0;
  for (int j = 0; j < 3; ++j) {
    a.job[j].w = t.bf((size_t)rows[j] * cols, -0.05f, 0.05f);
    if (j == 1) {
      a.job[j].x = t.f(1024, -1, 1);
      a.job[j].x2 = t.f(cols - 1024, -1, 1);
      a.job[j].split = 1024;
    } else {
      a.job[j].x = t.f(cols, -1, 1);
      a.job[j].split = cols;
    }
    a.job[j].y = t.zf(rows[j]);
    a.job[j].rows = rows[j];
    a.job[j].cols = cols;
    a.job[j].epi = epi[j];
    a.job[j].cap = 30.0f;
    a.total_rows += rows[j];
    chunks += (rows[j] + 15) / 16;
    t.out(j == 0 ? "plain" : j == 1 ? "sigmoid, split input" : "softcap", a.job[j].y, rows[j]);
  }
  a.njobs = 3;
  t.b->run(KID_gemvc, &a, chunks, cols / 8);
}

void t_exp13c(TB& t, int fmtv, float* e4m3, float* exp2t) {
  const int E = 8, k = 8, H = 2304, I = 1152;
  ExpArgs a{};
  a.e4m3 = e4m3;
  a.exp2t = exp2t;
  a.fmt = fmtv;
  a.k = k;
  a.gate_cap = 4.0f;
  a.up_cap = 25.0f;
  a.idx = t.ints({5, 0, 7, 2, 3, 6, 1, 4});
  a.rows = 2 * I;
  a.cols = H;
  size_t n13 = (size_t)E * a.rows * a.cols;
  if (fmtv == INK2_FMT_NVFP4) {
    a.w = t.bytes(n13 / 2, 0, 255, false);
    a.sc = t.bytes(n13 / 16, 0x28, 0x40, false);
    a.factor = t.f(E, 0.01f, 0.03f);
  } else {
    a.w = t.bytes(n13, 0, 255, true);
    a.sc = t.bytes(n13 / 32, 110, 118, false);
  }
  a.x = t.f(H, -1, 1);
  a.y = t.zf((size_t)k * I);
  t.out(fmtv == INK2_FMT_NVFP4 ? "nvfp4 activation" : "mxfp8 activation", a.y, (size_t)k * I);
  t.b->run(KID_exp13c, &a, k * ((I + 7) / 8), (H / 16 + 31) / 32 * 32);
}

void t_gemv(TB& t) {
  GemvArgs a{};
  const int rows[4] = {37, 5, 300, 64}, cols[4] = {256, 2304, 512, 4608};
  const int epi[4] = {INK2_EPI_NONE, INK2_EPI_SIGMOID, INK2_EPI_SOFTCAP, INK2_EPI_TANH};
  for (int j = 0; j < 4; ++j) {
    a.job[j].w = t.bf((size_t)rows[j] * cols[j], -0.05f, 0.05f);
    if (j == 3) {  // input in two pieces
      a.job[j].x = t.f(cols[j] / 2, -1, 1);
      a.job[j].x2 = t.f(cols[j] / 2, -1, 1);
      a.job[j].split = cols[j] / 2;
    } else {
      a.job[j].x = t.f(cols[j], -1, 1);
      a.job[j].split = cols[j];
    }
    a.job[j].y = t.zf(rows[j]);
    a.job[j].rows = rows[j];
    a.job[j].cols = cols[j];
    a.job[j].epi = epi[j];
    a.job[j].cap = 30.0f;
    a.total_rows += rows[j];
    t.out(j == 0 ? "plain" : j == 1 ? "sigmoid" : j == 2 ? "softcap" : "tanh, split input", a.job[j].y, rows[j]);
  }
  a.njobs = 4;
  t.b->run(KID_gemv, &a, (a.total_rows + 7) / 8, 256);
}

void t_norm(TB& t) {
  NormArgs a{};
  const int n = 2304;
  a.job[0].x = t.bf((size_t)4 * n, -3, 3);
  a.job[0].x_bf16 = 1;
  a.job[0].w = t.bf(n, 0.5f, 1.5f);
  a.job[0].out = t.zf(n);
  a.job[1].x = t.f(n, -3, 3);
  a.job[1].out = t.zf(n);
  a.st = t.st(2, 0, 1.0f);
  a.embed_stride = n;
  a.n = n;
  a.eps = 1e-6f;
  t.out("table row with weight", a.job[0].out, n);
  t.out("float32 without weight", a.job[1].out, n);
  t.b->run(KID_norm, &a, 2, 256);
}

void t_topk(TB& t) {
  RouterArgs a{};
  a.E = 128;
  a.k = 8;
  a.route_scale = 8.0f;
  a.logits = t.f(a.E, -4, 4);
  a.bias = t.f(a.E, -0.2f, 0.2f);
  a.idx = (int*)t.zf(a.k);
  a.wt = t.zf(a.k);
  t.out("indices", a.idx, a.k, true);
  t.out("weights", a.wt, a.k);
  t.b->run(KID_topk, &a, 1, 128);
}

void t_gemvw(TB& t) {
  GemvArgs a{};
  const int rows[2] = {64, 128}, cols[2] = {4608, 2304};
  for (int j = 0; j < 2; ++j) {
    a.job[j].w = t.bf((size_t)rows[j] * cols[j], -0.05f, 0.05f);
    if (j == 0) {
      a.job[j].x = t.f(cols[j] / 2, -1, 1);
      a.job[j].x2 = t.f(cols[j] / 2, -1, 1);
      a.job[j].split = cols[j] / 2;
    } else {
      a.job[j].x = t.f(cols[j], -1, 1);
      a.job[j].split = cols[j];
    }
    a.job[j].y = t.zf(rows[j]);
    a.job[j].rows = rows[j];
    a.job[j].cols = cols[j];
    a.total_rows += rows[j];
    t.out(j == 0 ? "split input" : "plain", a.job[j].y, rows[j]);
  }
  a.njobs = 2;
  t.b->run(KID_gemvw, &a, a.total_rows, 256);
}

void t_bar(TB& t) {
  BarArgs a{};
  const int H = 2304, K = 4;
  a.st = t.st(0, 5, 1.0f);
  a.H = H;
  a.K = K;
  a.hscale = (float)pow((double)H, -0.5);
  a.eps = 1e-6f;
  a.cdiv = (float)sqrt((double)(K + 1));
  for (int jb = 0; jb < 2; ++jb) {
    BarJob& j = a.job[jb];
    j.ns = jb ? 3 : 5;
    j.nq = jb ? 1 : 4;
    for (int i = 0; i < j.ns; ++i) j.st[i] = t.f(H, -4, 4);
    j.q = t.f((size_t)j.nq * H, -2, 2);
    if (jb) {  // the last state comes from the convolution
      j.cx = t.f(H, -3, 3);
      j.cw = t.bf((size_t)H * K, -1, 1);
      j.chist = t.f((size_t)(K - 1) * H, -3, 3);
      j.cres = t.f(H, -3, 3);
      j.cout = t.zf(H);
      j.st[j.ns - 1] = j.cout;
      t.out("convolution + residual", j.cout, H);
      t.out("history", j.chist, (size_t)(K - 1) * H);
    }
    for (int i = 0; i < j.nq; ++i) {
      j.out[i] = t.zf(H);
      j.normw[i] = (i == 0 || i == 2) ? t.bf(H, 0.5f, 1.5f) : nullptr;
      j.outn[i] = i == 3 ? t.zf(H) : nullptr;
      t.out(i == 1 ? "plain" : i == 3 ? "plain (with copy)" : "with norm weight", j.out[i], H);
      if (j.outn[i]) t.out("normalised copy", j.outn[i], H);
    }
  }
  t.b->run(KID_bar, &a, 2, (H / 4 + 31) / 32 * 32);
}

void t_gmix(TB& t) {
  GmixArgs a{};
  const int H = 2304;
  a.H = H;
  a.eps = 1e-6f;
  for (int j = 0; j < 3; ++j) {
    a.job[j].hs = t.f(H, -3, 3);
    a.job[j].msg = t.f(H, -3, 3);
    a.job[j].f = t.f(H, -1, 1);
    a.job[j].g = t.f(H, -1, 1);
    a.job[j].normw = j == 2 ? nullptr : t.bf(H, 0.5f, 1.5f);
    a.job[j].out = t.zf(H);
    t.out(j == 2 ? "without norm" : "with norm weight", a.job[j].out, H);
  }
  t.b->run(KID_gmix, &a, 3, (H / 4 + 31) / 32 * 32);
}

void t_prep(TB& t, int sliding) {
  PrepArgs a{};
  const int nh = 24, nkv = 8, hd = 128, drel = 16, K = 4, C = nkv * hd, cap = 16;
  a.q = t.f((size_t)nh * hd, -3, 3);
  a.q_scale = t.bf((size_t)nh * hd, 0.5f, 2.0f);
  a.r = t.f((size_t)nh * drel, -3, 3);
  a.r_scale = t.bf((size_t)nh * drel, 0.5f, 2.0f);
  a.kraw = t.f(C, -3, 3);
  a.vraw = t.f(C, -3, 3);
  a.khist = t.f((size_t)(K - 1) * C, -3, 3);
  a.vhist = t.f((size_t)(K - 1) * C, -3, 3);
  a.kc = t.zf((size_t)cap * C);
  a.vc = t.zf((size_t)cap * C);
  a.k_w = t.bf((size_t)C * K, -1, 1);
  a.v_w = t.bf((size_t)C * K, -1, 1);
  a.k_dyn = t.f((size_t)nkv * K, -1, 1);
  a.v_dyn = t.f((size_t)nkv * K, -1, 1);
  a.st = t.st(0, sliding ? 21 : 7, 1.07f);  // sliding: slot 21 % 16 = 5
  a.nh = nh;
  a.nkv = nkv;
  a.hd = hd;
  a.drel = drel;
  a.K = K;
  a.sliding = sliding;
  a.ring = sliding ? cap : 0;
  a.qk_thr = 22.627417f;
  a.rel_thr = 8.0f;
  a.eps = 1e-6f;
  a.conv_div = (float)sqrt((double)(sliding ? K + 1 : 2 * K + 1));
  t.out("q", a.q, (size_t)nh * hd);
  t.out("rel", a.r, (size_t)nh * drel);
  t.out("key cache", a.kc, (size_t)cap * C);
  t.out("value cache", a.vc, (size_t)cap * C);
  t.out("k history", a.khist, (size_t)(K - 1) * C);
  t.out("v history", a.vhist, (size_t)(K - 1) * C);
  t.b->run(KID_prep, &a, (2 * nh + 2 * nkv + 7) / 8, 256);
}

void t_attn(TB& t, int sliding) {
  AttnArgs a{};
  const int nh = 24, nkv = 8, hd = 128, drel = 16, C = nkv * hd;
  const int cap = sliding ? 128 : 1400;
  const int pos = sliding ? 300 : 1300;
  a.nh = nh;
  a.nkv = nkv;
  a.hd = hd;
  a.drel = drel;
  a.rel_extent = sliding ? 128 : 256;
  a.ring = sliding ? 128 : 0;
  a.st = t.st(0, pos, 1.0f);
  a.scratch_stride = cap;
  a.scale = (float)pow((double)hd, -0.5);
  a.q = t.f((size_t)nh * hd, -2, 2);
  a.r = t.f((size_t)nh * drel, -1, 1);
  a.sink = t.f(nh, -2, 2);
  a.gate = t.f((size_t)nh * hd, 0, 1);
  a.kc = t.f((size_t)cap * C, -2, 2);
  a.vc = t.f((size_t)cap * C, -1, 1);
  a.rel_proj = t.bf((size_t)drel * a.rel_extent, -1, 1);
  a.scratch = t.zf((size_t)nh * cap);
  a.out = t.zf((size_t)nh * hd);
  t.b->run(KID_attn, &a, nh, 1024);
  t.out("output", a.out, (size_t)nh * hd);
  t.out("softmax numerators", a.scratch, (size_t)nh * cap);
}

void t_exp(TB& t, int fmtv, float* e4m3, float* exp2t) {
  const int E = 8, k = 8, H = 2304, I = 1152, K = 4;
  ExpArgs a{};
  a.e4m3 = e4m3;
  a.exp2t = exp2t;
  a.fmt = fmtv;
  a.k = k;
  a.gate_cap = 4.0f;
  a.up_cap = 25.0f;
  a.idx = t.ints({5, 0, 7, 2, 3, 6, 1, 4});
  a.wt = t.f(k, 0.5f, 1.5f);
  // w13 [E, 2I, H]
  a.rows = 2 * I;
  a.cols = H;
  size_t n13 = (size_t)E * a.rows * a.cols;
  if (fmtv == INK2_FMT_NVFP4) {
    a.w = t.bytes(n13 / 2, 0, 255, false);
    a.sc = t.bytes(n13 / 16, 0x28, 0x40, false);
    a.factor = t.f(E, 0.01f, 0.03f);
  } else {
    a.w = t.bytes(n13, 0, 255, true);
    a.sc = t.bytes(n13 / 32, 110, 118, false);
  }
  a.x = t.f(H, -1, 1);
  float* actv = t.zf((size_t)k * I);
  a.y = actv;
  t.out("w13 activation", actv, (size_t)k * I);
  t.b->run(KID_exp13, &a, (k * I + 7) / 8, 256);
  // w2 [E, H, I], followed by the short convolution + residual
  a.rows = H;
  a.cols = I;
  size_t n2 = (size_t)E * a.rows * a.cols;
  if (fmtv == INK2_FMT_NVFP4) {
    a.w = t.bytes(n2 / 2, 0, 255, false);
    a.sc = t.bytes(n2 / 16, 0x28, 0x40, false);
    a.factor = t.f(E, 0.01f, 0.03f);
  } else {
    a.w = t.bytes(n2, 0, 255, true);
    a.sc = t.bytes(n2 / 32, 110, 118, false);
  }
  a.x = t.f((size_t)k * I, -1, 1);
  a.y = t.zf(H);
  a.cw = t.bf((size_t)H * K, -1, 1);
  a.chist = t.f((size_t)(K - 1) * H, -3, 3);
  a.cres = t.f(H, -3, 3);
  a.cout = t.zf(H);
  a.st = t.st(0, 4, 1.0f);
  a.K = K;
  a.cdiv = (float)sqrt((double)(K + 1));
  t.out("w2 output", a.y, H);
  t.out("w2 + convolution + residual", a.cout, H);
  t.out("history", a.chist, (size_t)(K - 1) * H);
  t.b->run(KID_exp2, &a, H, 16 * k);
}

// The three multi-row products on 11 right-hand sides (a full group of 8 and one of 3).
void t_gemv_multi(TB& t) {
  const int R = 11;
  {  // gemvcm: column owner, three jobs of 2304 columns, one with the input in two pieces
    GemvArgs a{};
    const int rows[3] = {37, 128, 5}, cols = 2304;
    const int epi[3] = {INK2_EPI_NONE, INK2_EPI_SIGMOID, INK2_EPI_SOFTCAP};
    int chunks = 0;
    for (int j = 0; j < 3; ++j) {
      a.job[j].w = t.bf((size_t)rows[j] * cols, -0.05f, 0.05f);
      if (j == 1) {
        a.job[j].x = t.f((size_t)R * 1024, -1, 1);
        a.job[j].x2 = t.f((size_t)R * (cols - 1024), -1, 1);
        a.job[j].split = 1024;
      } else {
        a.job[j].x = t.f((size_t)R * cols, -1, 1);
        a.job[j].split = cols;
      }
      a.job[j].y = t.zf((size_t)R * rows[j]);
      a.job[j].rows = rows[j];
      a.job[j].cols = cols;
      a.job[j].epi = epi[j];
      a.job[j].cap = 30.0f;
      a.total_rows += rows[j];
      chunks += (rows[j] + 15) / 16;
      t.out(j == 0 ? "gemvcm plain" : j == 1 ? "gemvcm sigmoid, split" : "gemvcm softcap", a.job[j].y, (size_t)R * rows[j]);
    }
    a.njobs = 3;
    a.nrhs = R;
    a.blocks = chunks;
    t.b->run(KID_gemvcm, &a, chunks * 2, cols / 8);
  }
  {  // gemvm: one warp per row, mixed widths
    GemvArgs a{};
    const int rows[3] = {37, 300, 64}, cols[3] = {256, 512, 64};
    for (int j = 0; j < 3; ++j) {
      a.job[j].w = t.bf((size_t)rows[j] * cols[j], -0.05f, 0.05f);
      a.job[j].x = t.f((size_t)R * cols[j], -1, 1);
      a.job[j].split = cols[j];
      a.job[j].y = t.zf((size_t)R * rows[j]);
      a.job[j].rows = rows[j];
      a.job[j].cols = cols[j];
      a.job[j].epi = j == 2 ? INK2_EPI_TANH : INK2_EPI_NONE;
      a.total_rows += rows[j];
      t.out(j == 2 ? "gemvm tanh" : "gemvm plain", a.job[j].y, (size_t)R * rows[j]);
    }
    a.njobs = 3;
    a.nrhs = R;
    a.blocks = (a.total_rows + 7) / 8;
    t.b->run(KID_gemvm, &a, a.blocks * 2, 256);
  }
  {  // gemvwm: one block per row, input in two pieces
    GemvArgs a{};
    const int rows = 64, cols = 4608;
    a.job[0].w = t.bf((size_t)rows * cols, -0.05f, 0.05f);
    a.job[0].x = t.f((size_t)R * cols / 2, -1, 1);
    a.job[0].x2 = t.f((size_t)R * cols / 2, -1, 1);
    a.job[0].split = cols / 2;
    a.job[0].y = t.zf((size_t)R * rows);
    a.job[0].rows = rows;
    a.job[0].cols = cols;
    a.total_rows = rows;
    a.njobs = 1;
    a.nrhs = R;
    a.blocks = rows;
    t.out("gemvwm split", a.job[0].y, (size_t)R * rows);
    t.b->run(KID_gemvwm, &a, rows * 2, 256);
  }
}

// The experts of a 20-row pass: grouping, w13, w2, sum + convolution.
void t_exp_multi(TB& t, int fmtv, float* e4m3, float* exp2t) {
  const int E = 16, k = 8, H = 2304, I = 1152, K = 4, R = 20, hr = 24;  // about 10 pairs per expert
  std::vector<int> idx;
  for (int r = 0; r < R; ++r) {  // k distinct experts per row, increasing
    std::vector<int> pick;
    while ((int)pick.size() < k) {
      int e = (int)(t.next() >> 40) % E;
      bool dup = false;
      for (int x : pick) dup = dup || x == e;
      if (!dup) pick.push_back(e);
    }
    for (int i = 0; i < k; ++i)
      for (int j = i + 1; j < k; ++j)
        if (pick[j] < pick[i]) std::swap(pick[i], pick[j]);
    idx.insert(idx.end(), pick.begin(), pick.end());
  }
  GroupArgs g{};
  g.idx = t.ints(idx);
  g.gcount = (int*)t.zf(1);
  g.gexp = (int*)t.zf(E);
  g.goff = (int*)t.zf(E + 1);
  g.gpairs = (int*)t.zf((size_t)R * k);
  g.R = R;
  g.k = k;
  g.E = E;
  t.out("group: experts", g.gcount, 1, true);
  t.out("group: ids", g.gexp, E, true);
  t.out("group: offsets", g.goff, E + 1, true);
  t.out("group: pairs", g.gpairs, (size_t)R * k, true);
  t.b->run(KID_egroup, &g, 1, 32);
  ExpMArgs a{};
  a.e4m3 = e4m3;
  a.exp2t = exp2t;
  a.gcount = g.gcount;
  a.gexp = g.gexp;
  a.goff = g.goff;
  a.gpairs = g.gpairs;
  a.fmt = fmtv;
  a.k = k;
  a.gate_cap = 4.0f;
  a.up_cap = 25.0f;
  a.rows = 2 * I;
  a.cols = H;
  size_t n13 = (size_t)E * a.rows * a.cols;
  if (fmtv == INK2_FMT_NVFP4) {
    a.w = t.bytes(n13 / 2, 0, 255, false);
    a.sc = t.bytes(n13 / 16, 0x28, 0x40, false);
    a.factor = t.f(E, 0.01f, 0.03f);
  } else {
    a.w = t.bytes(n13, 0, 255, true);
    a.sc = t.bytes(n13 / 32, 110, 118, false);
  }
  a.x = t.f((size_t)R * H, -1, 1);
  float* actv = t.zf((size_t)R * k * I);
  a.y = actv;
  a.nchunk = (I + 15) / 16;
  a.grid = 200;  // fewer blocks than items: every block takes several
  t.out("w13 activations", actv, (size_t)R * k * I);
  t.b->run(KID_exp13cm, &a, a.grid, 256);
  a.rows = H;
  a.cols = I;
  size_t n2 = (size_t)E * a.rows * a.cols;
  if (fmtv == INK2_FMT_NVFP4) {
    a.w = t.bytes(n2 / 2, 0, 255, false);
    a.sc = t.bytes(n2 / 16, 0x28, 0x40, false);
    a.factor = t.f(E, 0.01f, 0.03f);
  } else {
    a.w = t.bytes(n2, 0, 255, true);
    a.sc = t.bytes(n2 / 32, 110, 118, false);
  }
  a.x = actv;
  float* dots = t.zf((size_t)R * k * H);
  a.y = dots;
  a.nchunk = (H + 15) / 16;
  a.grid = 300;  // fewer blocks than items: every block takes several
  t.out("w2 products", dots, (size_t)R * k * H);
  t.b->run(KID_exp2m, &a, a.grid, 256);
  ECombArgs c{};
  c.dots = dots;
  c.wt = t.f((size_t)R * k, 0.5f, 1.5f);
  c.y = t.zf((size_t)R * H);
  c.cw = t.bf((size_t)H * K, -1, 1);
  c.chist = t.f((size_t)hr * H, -3, 3);
  c.cres = t.f((size_t)R * H, -3, 3);
  c.cout = t.zf((size_t)R * H);
  std::vector<StepState> sts(R);
  for (int r = 0; r < R; ++r) sts[r] = StepState{0, 9 + r, 1.0f, 0};
  c.st = (StepState*)t.up(sts.data(), sizeof(StepState) * R);
  c.R = R;
  c.k = k;
  c.H = H;
  c.K = K;
  c.hr = hr;
  c.cdiv = (float)sqrt((double)(K + 1));
  t.out("sum of the products", c.y, (size_t)R * H);
  t.out("+ convolution + residual", c.cout, (size_t)R * H);
  t.out("history", c.chist, (size_t)hr * H);
  t.b->run(KID_ecomb, &c, (R * H + 255) / 256, 256);
}

void t_argmax(TB& t) {
  ArgmaxArgs a{};
  const int rows = 3, n = 200058;
  a.logits = t.f((size_t)rows * n, -30, 30);
  a.idx = (int*)t.zf(rows);
  a.val = t.zf(rows);
  a.n = n;
  t.out("index", a.idx, rows, true);
  t.out("value", a.val, rows);
  t.b->run(KID_argmax, &a, rows, 1024);
}

}  // namespace

int ink2_run_selftest(Backend* ref, Backend* dut, std::string& report) {
  const char* names[] = {"gemv",         "norm",      "bar",    "gmix",          "prep sliding", "prep full",
                         "attn sliding", "attn full", "experts nvfp4", "experts mxfp8", "gemvw", "topk", "gemvc", "exp13c",
                         "gemv multi-row", "experts multi nvfp4", "experts multi mxfp8", "argmax"};
  const int ntests = (int)(sizeof names / sizeof names[0]);
  int failures = 0;
  float t1[256], t2[256];
  ink2_make_tables(t1, t2);
  report += fmt("self-test: %s against %s, the same random inputs (tolerance 1e-5 * max(1, max |reference|))\n",
                dut->name(), ref->name());
  for (int kind = 0; kind < ntests; ++kind) {
    TB ta(ref), tb(dut);
    TB* both[2] = {&ta, &tb};
    for (TB* t : both) {
      float* e4 = (float*)t->up(t1, sizeof t1);
      float* ex = (float*)t->up(t2, sizeof t2);
      switch (kind) {
        case 0: t_gemv(*t); break;
        case 1: t_norm(*t); break;
        case 2: t_bar(*t); break;
        case 3: t_gmix(*t); break;
        case 4: t_prep(*t, 1); break;
        case 5: t_prep(*t, 0); break;
        case 6: t_attn(*t, 1); break;
        case 7: t_attn(*t, 0); break;
        case 8: t_exp(*t, INK2_FMT_NVFP4, e4, ex); break;
        case 9: t_exp(*t, INK2_FMT_MXFP8, e4, ex); break;
        case 10: t_gemvw(*t); break;
        case 11: t_topk(*t); break;
        case 12: t_gemvc(*t); break;
        case 13:
          t_exp13c(*t, INK2_FMT_NVFP4, e4, ex);
          t_exp13c(*t, INK2_FMT_MXFP8, e4, ex);
          break;
        case 14: t_gemv_multi(*t); break;
        case 15: t_exp_multi(*t, INK2_FMT_NVFP4, e4, ex); break;
        case 16: t_exp_multi(*t, INK2_FMT_MXFP8, e4, ex); break;
        case 17: t_argmax(*t); break;
      }
      t->b->sync();
    }
    bool bad = false;
    std::string lines;
    for (size_t o = 0; o < ta.outs.size(); ++o) {
      size_t n = ta.outs[o].n;
      std::vector<float> va(n), vb(n);
      ref->download(va.data(), ta.outs[o].p, n * 4);
      dut->download(vb.data(), tb.outs[o].p, n * 4);
      double maxd = 0, maxr = 0;
      size_t ndiff = 0, nbad = 0;
      if (ta.outs[o].is_int) {
        for (size_t i = 0; i < n; ++i)
          if (memcmp(&va[i], &vb[i], 4)) ++ndiff, ++nbad, maxd = 1;
      } else {
        for (size_t i = 0; i < n; ++i) {
          double d = fabs((double)va[i] - (double)vb[i]);
          if (!(d == d)) d = 1e30, ++nbad;  // NaN
          if (d > maxd) maxd = d;
          if (fabs(va[i]) > maxr) maxr = fabs(va[i]);
          if (memcmp(&va[i], &vb[i], 4)) ++ndiff;
        }
      }
      double tol = 1e-5 * (maxr > 1 ? maxr : 1);
      bool obad = maxd > tol || nbad > 0;
      bad = bad || obad;
      lines += fmt("    %-24s n %-8zu max |diff| %.3g (max |ref| %.3g), %zu values not bit-equal%s\n",
                   ta.outs[o].name.c_str(), n, maxd, maxr, ndiff, obad ? "  <-- OVER" : "");
    }
    const char* e1 = ref->error();
    const char* e2 = dut->error();
    if (e1 || e2) bad = true;
    report += fmt("  %-14s %s\n", names[kind], bad ? "FAIL" : "ok");
    report += lines;
    if (e1) report += fmt("    %s error: %s\n", ref->name(), e1);
    if (e2) report += fmt("    %s error: %s\n", dut->name(), e2);
    if (bad) ++failures;
    if (e1 || e2) break;
  }
  report += fmt("self-test result: %d of %d kernels over the tolerance\n", failures, ntests);
  return failures;
}
