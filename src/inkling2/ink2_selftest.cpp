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
// small = 1: the kernels of passes of few rows (exp13s / exp2s), R = 6 rows over 24 experts (1 .. 6 pairs per expert)
void t_exp_multi(TB& t, int fmtv, float* e4m3, float* exp2t, int small = 0) {
  const int E = small ? 24 : 16, k = 8, H = 2304, I = 1152, K = 4, R = small ? 6 : 20, hr = 24;  // about 10 pairs per expert
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
  a.nchunk = small ? (I + 7) / 8 : (I + 15) / 16;
  a.grid = 200;  // fewer blocks than items: every block takes several
  t.out("w13 activations", actv, (size_t)R * k * I);
  if (small) t.b->run(fmtv == INK2_FMT_NVFP4 ? KID_exp13s : KID_exp13sx, &a, a.grid, (H / 16 + 31) / 32 * 32);
  else t.b->run(KID_exp13cm, &a, a.grid, 256);
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
  if (small) t.b->run(fmtv == INK2_FMT_NVFP4 ? KID_exp2s : KID_exp2sx, &a, a.grid, 256);
  else t.b->run(KID_exp2m, &a, a.grid, 256);
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

// A packed (or bfloat16) dense job with random content: rows x cols, input in one or two pieces.
void pk_job(TB& t, GemvJob& J, int fmtv, int rows, int cols, int split, int epi, float fac) {
  size_t n = (size_t)rows * cols;
  if (fmtv == INK2_DFMT_NVFP4) {
    J.w = (const u16*)t.bytes(n / 2, 0, 255, false);
    J.sc = t.bytes(n / 16, 0x28, 0x40, false);
  } else if (fmtv == INK2_DFMT_MXFP8) {
    J.w = (const u16*)t.bytes(n, 0, 255, true);
    J.sc = t.bytes(n / 32, 110, 118, false);
  } else if (fmtv == INK2_DFMT_Q3) {
    J.w = (const u16*)t.bytes(n * 3 / 8, 0, 255, false);
    J.sc = t.bytes(n / 16, 0x28, 0x40, false);
  } else {
    J.w = t.bf(n, -0.05f, 0.05f);
  }
  J.fmt = fmtv;
  J.fac = fac;
  if (split < cols) {
    J.x = t.f(split, -1, 1);
    J.x2 = t.f(cols - split, -1, 1);
  } else {
    J.x = t.f(cols, -1, 1);
  }
  J.split = split;
  J.y = t.zf(rows);
  J.rows = rows;
  J.cols = cols;
  J.epi = epi;
  J.cap = 30.0f;
}

void t_pgemvc(TB& t, int kid = KID_pgemvc, int nr = 16) {
  GemvArgs a{};
  const int cols = 2304;
  // as the projections of a visit: mixed formats, epilogues, a factor, one split input, a short last chunk
  pk_job(t, a.job[0], INK2_DFMT_NVFP4, 3072, cols, cols, INK2_EPI_NONE, 0.004f);
  pk_job(t, a.job[1], INK2_DFMT_MXFP8, 384, cols, cols, INK2_EPI_NONE, 0.0f);
  pk_job(t, a.job[2], INK2_DFMT_NVFP4, 37, cols, 1024, INK2_EPI_SIGMOID, 0.003f);
  pk_job(t, a.job[3], INK2_DFMT_MXFP8, 24, cols, cols, INK2_EPI_NONE, 0.0f);
  pk_job(t, a.job[4], INK2_DFMT_BF16, 21, cols, cols, INK2_EPI_SOFTCAP, 0.0f);
  pk_job(t, a.job[5], INK2_DFMT_NVFP4, 1000, cols, cols, INK2_EPI_SOFTCAP, 0.002f);
  const char* names[6] = {"nvfp4", "mxfp8", "nvfp4 sigmoid split", "mxfp8 24 rows", "bf16 softcap", "nvfp4 softcap"};
  int chunks = 0;
  a.njobs = 6;
  for (int j = 0; j < a.njobs; ++j) {
    a.total_rows += a.job[j].rows;
    chunks += (a.job[j].rows + nr - 1) / nr;
    t.out(names[j], a.job[j].y, a.job[j].rows);
  }
  t.b->run(kid, &a, chunks, cols / 8);
}

void t_pgemv(TB& t) {
  GemvArgs a{};
  // controllers (forget / gain: 64 columns) and a long row form
  pk_job(t, a.job[0], INK2_DFMT_MXFP8, 2304, 64, 64, INK2_EPI_TANH, 0.0f);
  pk_job(t, a.job[1], INK2_DFMT_NVFP4, 300, 2304, 2304, INK2_EPI_NONE, 0.004f);
  pk_job(t, a.job[2], INK2_DFMT_MXFP8, 37, 512, 256, INK2_EPI_SIGMOID, 0.0f);
  t.out("mxfp8 64 columns tanh", a.job[0].y, a.job[0].rows);
  t.out("nvfp4", a.job[1].y, a.job[1].rows);
  t.out("mxfp8 split sigmoid", a.job[2].y, a.job[2].rows);
  a.njobs = 3;
  for (int j = 0; j < a.njobs; ++j) a.total_rows += a.job[j].rows;
  t.b->run(KID_pgemv, &a, (a.total_rows + 7) / 8, 256);
}

void t_pgemvw(TB& t) {
  GemvArgs a{};
  pk_job(t, a.job[0], INK2_DFMT_MXFP8, 64, 4608, 2304, INK2_EPI_NONE, 0.0f);  // controller down
  pk_job(t, a.job[1], INK2_DFMT_NVFP4, 128, 2304, 2304, INK2_EPI_NONE, 0.004f);
  t.out("mxfp8 split", a.job[0].y, a.job[0].rows);
  t.out("nvfp4", a.job[1].y, a.job[1].rows);
  a.njobs = 2;
  for (int j = 0; j < a.njobs; ++j) a.total_rows += a.job[j].rows;
  t.b->run(KID_pgemvw, &a, a.total_rows, 256);
}

// bfloat16 kernels with a row factor (the decoded nvfp4 matrices)
void t_gemv_fac(TB& t) {
  GemvArgs a{};
  pk_job(t, a.job[0], INK2_DFMT_BF16, 40, 2304, 2304, INK2_EPI_NONE, 0.004f);
  pk_job(t, a.job[1], INK2_DFMT_BF16, 17, 2304, 1024, INK2_EPI_SIGMOID, 0.003f);
  a.njobs = 2;
  a.total_rows = 57;
  t.out("gemvc factor", a.job[0].y, 40);
  t.out("gemvc factor sigmoid split", a.job[1].y, 17);
  t.b->run(KID_gemvc, &a, 3 + 2, 2304 / 8);
  GemvArgs b{};
  pk_job(t, b.job[0], INK2_DFMT_BF16, 40, 512, 512, INK2_EPI_NONE, 0.004f);
  b.njobs = 1;
  b.total_rows = 40;
  t.out("gemv factor", b.job[0].y, 40);
  t.b->run(KID_gemv, &b, 5, 256);
  GemvArgs c{};
  pk_job(t, c.job[0], INK2_DFMT_BF16, 9, 4608, 2304, INK2_EPI_NONE, 0.004f);
  c.njobs = 1;
  c.total_rows = 9;
  t.out("gemvw factor split", c.job[0].y, 9);
  t.b->run(KID_gemvw, &c, 9, 256);
}

void t_norm_packed(TB& t) {
  NormArgs a{};
  const int n = 2304, trows = 4;
  a.job[0].x = t.bytes((size_t)trows * n, 0, 255, true);
  a.esc = t.bytes((size_t)trows * n / 32, 120, 130, false);
  a.efmt = INK2_DFMT_MXFP8;
  a.job[0].w = t.bf(n, 0.5f, 1.5f);
  a.job[0].out = t.zf(n);
  a.st = t.st(3, 0, 1.0f);
  a.embed_stride = n;
  a.n = n;
  a.eps = 1e-6f;
  t.out("packed table row with weight", a.job[0].out, n);
  t.b->run(KID_norm, &a, 1, 256);
}

// pgemvn: narrow rows (64 and 128 columns), also with 3 right-hand sides
void t_pgemvn(TB& t) {
  GemvArgs a{};
  pk_job(t, a.job[0], INK2_DFMT_MXFP8, 2304, 64, 64, INK2_EPI_TANH, 0.0f);
  pk_job(t, a.job[1], INK2_DFMT_NVFP4, 1001, 64, 64, INK2_EPI_NONE, 0.004f);
  a.njobs = 2;
  a.total_rows = 3305;
  a.blocks = (3305 + 31) / 32;
  t.out("mxfp8 64 columns tanh", a.job[0].y, 2304);
  t.out("nvfp4 64 columns", a.job[1].y, 1001);
  t.b->run(KID_pgemvn, &a, a.blocks, 256);
  GemvArgs b{};
  const int R = 3;
  pk_job(t, b.job[0], INK2_DFMT_MXFP8, 333, 128, 128, INK2_EPI_NONE, 0.0f);
  b.job[0].x = t.f((size_t)R * 128, -1, 1);
  b.job[0].y = t.zf((size_t)R * 333);
  b.njobs = 1;
  b.total_rows = 333;
  b.nrhs = R;
  b.blocks = (333 + 15) / 16;
  t.out("mxfp8 128 columns 3 rows", b.job[0].y, (size_t)R * 333);
  t.b->run(KID_pgemvn, &b, b.blocks * R, 256);
}

// pgemvc / pgemv / pgemvw with 3 right-hand sides (blocks * nrhs blocks)
void t_packed_multi(TB& t) {
  const int R = 3, cols = 2304;
  GemvArgs a{};
  pk_job(t, a.job[0], INK2_DFMT_NVFP4, 40, cols, cols, INK2_EPI_NONE, 0.004f);
  pk_job(t, a.job[1], INK2_DFMT_MXFP8, 24, cols, 1024, INK2_EPI_SIGMOID, 0.0f);
  a.job[0].x = t.f((size_t)R * cols, -1, 1);
  a.job[0].y = t.zf((size_t)R * 40);
  a.job[1].x = t.f((size_t)R * 1024, -1, 1);
  a.job[1].x2 = t.f((size_t)R * (cols - 1024), -1, 1);
  a.job[1].y = t.zf((size_t)R * 24);
  a.njobs = 2;
  a.total_rows = 64;
  a.nrhs = R;
  a.blocks = 3 + 2;
  t.out("pgemvc 3 rows nvfp4", a.job[0].y, (size_t)R * 40);
  t.out("pgemvc 3 rows mxfp8 split", a.job[1].y, (size_t)R * 24);
  t.b->run(KID_pgemvc, &a, a.blocks * R, cols / 8);
  GemvArgs b{};
  pk_job(t, b.job[0], INK2_DFMT_MXFP8, 2304, 64, 64, INK2_EPI_TANH, 0.0f);
  b.job[0].x = t.f((size_t)R * 64, -1, 1);
  b.job[0].y = t.zf((size_t)R * 2304);
  b.njobs = 1;
  b.total_rows = 2304;
  b.nrhs = R;
  b.blocks = 2304 / 8;
  t.out("pgemv 3 rows", b.job[0].y, (size_t)R * 2304);
  t.b->run(KID_pgemv, &b, b.blocks * R, 256);
  GemvArgs c{};
  pk_job(t, c.job[0], INK2_DFMT_MXFP8, 64, 4608, 2304, INK2_EPI_NONE, 0.0f);
  c.job[0].x = t.f((size_t)R * 2304, -1, 1);
  c.job[0].x2 = t.f((size_t)R * 2304, -1, 1);
  c.job[0].y = t.zf((size_t)R * 64);
  c.njobs = 1;
  c.total_rows = 64;
  c.nrhs = R;
  c.blocks = 64;
  t.out("pgemvw 3 rows split", c.job[0].y, (size_t)R * 64);
  t.b->run(KID_pgemvw, &c, 64 * R, 256);
}

// pgemvcm (packed column owner, 8 right-hand sides per block): R = 11 rows (a full and a partial group)
void t_pgemvcm(TB& t, int kid, int nr, int nh = 8, int R = 11) {
  const int cols = 2304;
  GemvArgs a{};
  pk_job(t, a.job[0], INK2_DFMT_NVFP4, 300, cols, cols, INK2_EPI_NONE, 0.004f);
  pk_job(t, a.job[1], INK2_DFMT_MXFP8, 37, cols, 1024, INK2_EPI_SIGMOID, 0.0f);
  pk_job(t, a.job[2], INK2_DFMT_BF16, 21, cols, cols, INK2_EPI_SOFTCAP, 0.0f);
  pk_job(t, a.job[3], INK2_DFMT_NVFP4, 100, cols, cols, INK2_EPI_SOFTCAP, 0.002f);
  const char* names[4] = {"nvfp4 11 rows", "mxfp8 sigmoid split 11 rows", "bf16 softcap 11 rows", "nvfp4 softcap 11 rows"};
  a.njobs = 4;
  int chunks = 0;
  for (int j = 0; j < a.njobs; ++j) {
    GemvJob& J = a.job[j];
    if (J.split < cols) {
      J.x = t.f((size_t)R * J.split, -1, 1);
      J.x2 = t.f((size_t)R * (cols - J.split), -1, 1);
    } else {
      J.x = t.f((size_t)R * cols, -1, 1);
    }
    J.y = t.zf((size_t)R * J.rows);
    a.total_rows += J.rows;
    chunks += (J.rows + nr - 1) / nr;
    t.out(names[j], J.y, (size_t)R * J.rows);
  }
  a.nrhs = R;
  a.blocks = chunks;
  t.b->run(kid, &a, chunks * ((R + nh - 1) / nh), cols / 8);
}

// q3 (the converted drafter files): column owner with mixed formats, 8-row passes (pgemvcm), long rows
// with 3 right-hand sides (pgemv), narrow rows (pgemvn)
void t_q3(TB& t) {
  {
    GemvArgs a{};
    const int cols = 2304;
    pk_job(t, a.job[0], INK2_DFMT_Q3, 3072, cols, cols, INK2_EPI_NONE, 0.004f);
    pk_job(t, a.job[1], INK2_DFMT_NVFP4, 40, cols, cols, INK2_EPI_NONE, 0.003f);
    pk_job(t, a.job[2], INK2_DFMT_Q3, 37, cols, 1024, INK2_EPI_SIGMOID, 0.002f);
    a.njobs = 3;
    int chunks = 0;
    const char* names[3] = {"q3 pgemvc", "nvfp4 next to q3", "q3 pgemvc sigmoid split"};
    for (int j = 0; j < a.njobs; ++j) {
      a.total_rows += a.job[j].rows;
      chunks += (a.job[j].rows + 15) / 16;
      t.out(names[j], a.job[j].y, a.job[j].rows);
    }
    t.b->run(KID_pgemvc, &a, chunks, cols / 8);
  }
  {
    const int R = 11, cols = 2304;
    GemvArgs a{};
    pk_job(t, a.job[0], INK2_DFMT_Q3, 300, cols, cols, INK2_EPI_NONE, 0.004f);
    pk_job(t, a.job[1], INK2_DFMT_Q3, 37, cols, 1024, INK2_EPI_SOFTCAP, 0.002f);
    a.njobs = 2;
    int chunks = 0;
    const char* names[2] = {"q3 pgemvcm8 11 rows", "q3 pgemvcm8 softcap split"};
    for (int j = 0; j < a.njobs; ++j) {
      GemvJob& J = a.job[j];
      if (J.split < cols) {
        J.x = t.f((size_t)R * J.split, -1, 1);
        J.x2 = t.f((size_t)R * (cols - J.split), -1, 1);
      } else {
        J.x = t.f((size_t)R * cols, -1, 1);
      }
      J.y = t.zf((size_t)R * J.rows);
      a.total_rows += J.rows;
      chunks += (J.rows + 7) / 8;
      t.out(names[j], J.y, (size_t)R * J.rows);
    }
    a.nrhs = R;
    a.blocks = chunks;
    t.b->run(KID_pgemvcm8, &a, chunks * ((R + 7) / 8), cols / 8);
  }
  {
    const int R = 3;
    GemvArgs b{};
    pk_job(t, b.job[0], INK2_DFMT_Q3, 64, 9216, 9216, INK2_EPI_NONE, 0.003f);
    b.job[0].x = t.f((size_t)R * 9216, -1, 1);
    b.job[0].y = t.zf((size_t)R * 64);
    b.njobs = 1;
    b.total_rows = 64;
    b.nrhs = R;
    b.blocks = 64 / 8;
    t.out("q3 pgemv 9216 columns 3 rows", b.job[0].y, (size_t)R * 64);
    t.b->run(KID_pgemv, &b, b.blocks * R, 256);
  }
  {
    GemvArgs a{};
    pk_job(t, a.job[0], INK2_DFMT_Q3, 1001, 256, 256, INK2_EPI_TANH, 0.004f);
    a.njobs = 1;
    a.total_rows = 1001;
    a.blocks = (1001 + 7) / 8;
    t.out("q3 pgemvn 256 columns", a.job[0].y, 1001);
    t.b->run(KID_pgemvn, &a, a.blocks, 256);
  }
}

// The drafter's Markov head with a packed table (dzin) and a packed folded head (dfold): fmtv = INK2_DFMT_*
void t_dhead(TB& t, int fmtv) {
  const int R = 256, H = 256, V = 5003, G = 64, rows = 300;
  size_t n = (size_t)rows * R, nf = (size_t)V * R;
  auto codes = [&](size_t m) -> const u8* {
    if (fmtv == INK2_DFMT_NVFP4) return t.bytes(m / 2, 0, 255, false);
    if (fmtv == INK2_DFMT_MXFP8) return t.bytes(m, 0, 255, true);
    return t.bytes(m * 3 / 8, 0, 255, false);
  };
  auto scales = [&](size_t m) -> const u8* {
    if (fmtv == INK2_DFMT_MXFP8) return t.bytes(m / 32, 110, 118, false);
    return t.bytes(m / 16, 0x28, 0x40, false);
  };
  float fac = fmtv == INK2_DFMT_MXFP8 ? 0.0f : 0.004f;
  DZinArgs z{};
  std::vector<StepState> s(8);
  for (int i = 0; i < 8; ++i) s[i] = StepState{17 + i, 40 + i, 1.0f, 0};
  z.st = (StepState*)t.up(s.data(), s.size() * sizeof(StepState));
  z.s = t.f(R, -1, 1);
  z.w1 = (const u16*)codes(n);
  z.w1sc = scales(n);
  z.w1fac = fac;
  z.w1fmt = fmtv;
  z.hn = t.f((size_t)8 * H, -1, 1);
  z.frnn = t.f((size_t)16 * H, -1, 1);
  z.z = t.zf((size_t)(2 * R + 2 * H));
  z.drafts = (int*)t.zf(8);
  z.pval = t.f(G, -5, 5);
  std::vector<int> pi(G);
  for (int g = 0; g < G; ++g) pi[g] = (g * 37) % rows;
  z.pidx = t.ints(pi);
  z.G = G;
  z.step = 1;
  z.depth = 3;
  z.rank = R;
  z.H = H;
  z.fring = 16;
  z.w1_rows = rows;
  t.out("dzin packed table z", z.z, (size_t)(2 * R + 2 * H));
  t.out("dzin pick", z.drafts, 8, true);
  t.b->run(KID_dzin, &z, 1, 256);
  DFoldArgs f{};
  f.zz = t.f(3 * R, -1, 1);
  f.rnn_b = t.bf(3 * R, -0.1f, 0.1f);
  f.s = t.f(R, -1, 1);
  f.base = t.f(V, -10, 10);
  f.fp = codes(nf);
  f.fpsc = scales(nf);
  f.fpfac = fac;
  f.fpfmt = fmtv;
  f.fb = t.f(V, -1, 1);
  f.pval = t.zf(G);
  f.pidx = (int*)t.zf(G);
  f.V = V;
  f.rank = R;
  f.step = 1;
  f.G = G;
  f.cap = 30.0f;
  t.out("dfold packed head max", f.pval, G);
  t.out("dfold packed head index", f.pidx, G, true);
  t.out("dfold state s", f.s, R);
  t.b->run(KID_dfold, &f, G, 256);
}

}  // namespace

int ink2_run_selftest(Backend* ref, Backend* dut, std::string& report) {
  const char* names[] = {"gemv",         "norm",      "bar",    "gmix",          "prep sliding", "prep full",
                         "attn sliding", "attn full", "experts nvfp4", "experts mxfp8", "gemvw", "topk", "gemvc", "exp13c",
                         "gemv multi-row", "experts multi nvfp4", "experts multi mxfp8", "argmax", "pgemvc", "pgemv", "pgemvw",
                         "bf16 with factor", "norm packed table", "packed multi-row", "pgemvc8", "pgemvc4", "pgemvc2", "pgemvn",
                         "pgemvcm", "pgemvcm8", "pgemvcm4", "pgemvcm2",
                         "experts small nvfp4", "experts small mxfp8", "pgcm h2 (16 8 4 2 rows, R 2 and 5)",
                         "pgcm h4 (16 8 4 2 rows, R 3 and 4)", "pgemvcm8 R 6", "q3 products", "drafter head nvfp4",
                         "drafter head mxfp8", "drafter head q3"};
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
        case 18: t_pgemvc(*t); break;
        case 19: t_pgemv(*t); break;
        case 20: t_pgemvw(*t); break;
        case 21: t_gemv_fac(*t); break;
        case 22: t_norm_packed(*t); break;
        case 23: t_packed_multi(*t); break;
        case 24: t_pgemvc(*t, KID_pgemvc8, 8); break;
        case 25: t_pgemvc(*t, KID_pgemvc4, 4); break;
        case 26: t_pgemvc(*t, KID_pgemvc2, 2); break;
        case 27: t_pgemvn(*t); break;
        case 28: t_pgemvcm(*t, KID_pgemvcm, 16); break;
        case 29: t_pgemvcm(*t, KID_pgemvcm8, 8); break;
        case 30: t_pgemvcm(*t, KID_pgemvcm4, 4); break;
        case 31: t_pgemvcm(*t, KID_pgemvcm2, 2); break;
        case 32: t_exp_multi(*t, INK2_FMT_NVFP4, e4, ex, 1); break;
        case 33: t_exp_multi(*t, INK2_FMT_MXFP8, e4, ex, 1); break;
        case 34:
          t_pgemvcm(*t, KID_pgcm16h2, 16, 2, 2);
          t_pgemvcm(*t, KID_pgcm8h2, 8, 2, 2);
          t_pgemvcm(*t, KID_pgcm4h2, 4, 2, 5);
          t_pgemvcm(*t, KID_pgcm2h2, 2, 2, 2);
          break;
        case 35:
          t_pgemvcm(*t, KID_pgcm16h4, 16, 4, 4);
          t_pgemvcm(*t, KID_pgcm8h4, 8, 4, 3);
          t_pgemvcm(*t, KID_pgcm4h4, 4, 4, 4);
          t_pgemvcm(*t, KID_pgcm2h4, 2, 4, 3);
          break;
        case 36: t_pgemvcm(*t, KID_pgemvcm8, 8, 8, 6); break;
        case 37: t_q3(*t); break;
        case 38: t_dhead(*t, INK2_DFMT_NVFP4); break;
        case 39: t_dhead(*t, INK2_DFMT_MXFP8); break;
        case 40: t_dhead(*t, INK2_DFMT_Q3); break;
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
