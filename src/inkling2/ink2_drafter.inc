// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Drafter add-on of the engine (drafter of Inkling-2 Mini, mtp.safetensors).
// Included at the end of ink2_engine.cpp (it uses the engine's Engine, Ten, run(),
// k_gemv(), job()). Arithmetic: our reference implementation of the drafter (not part of this repository); kernels:
// ink2_drafter_kernels.inc. C API: ink2_drafter_api.h.
//
// State on the device, kept in step with the engine by drf_commit, which runs at the
// end of every pass of the engine (inside its recorded graph): per committed position
// the context key / value of the three draft layers (sliding layers: rings of
// window + margin + 8 slots, the full layer: max_positions slots) and f_rnn (a ring of
// margin + 16 slots). Rejected rows of a pass are committed too and later overwritten;
// a draft call reads only the positions below ink2_position(), so truncate and reset
// need nothing extra.
#include "ink2_drafter_api.h"

namespace {

// A product job over a BF16 drafter matrix (written field by field, so that this file also fits an
// engine whose GemvJob has more fields, e.g. packed dense matrices: they stay zero = BF16).
GemvJob drf_job(const u16* w, const float* x, float* y, int rows, int cols) {
  GemvJob j{};
  j.w = w;
  j.x = x;
  j.split = cols;
  j.y = y;
  j.rows = rows;
  j.cols = cols;
  j.epi = INK2_EPI_NONE;
  return j;
}
// The embedding table of the engine as the source of the norm kernel: BF16 pointer (this engine)
// or a dense-matrix record with a packed format (an engine with packed dense matrices).
inline void drf_embed_src(NormArgs& n, const u16* table) {
  n.job[0].x = table;
  n.job[0].x_bf16 = 1;
}
template <class M>
inline void drf_embed_src(NormArgs& n, const M& m) {
  n.job[0].x = m.w;
  n.job[0].x_bf16 = 1;
  n.esc = m.sc;
  n.efmt = m.fmt;
}

struct Drafter {
  Ink2DraftConfig c{};
  std::map<std::string, Ten> ten;
  std::vector<void*> owned;
  size_t bytes = 0;
  bool finalized = false;
  struct DL {
    const u16 *dflare = nullptr, *in_ln = nullptr, *post_ln = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr,
              *wo = nullptr, *qn = nullptr, *kn = nullptr, *wkc = nullptr, *wvc = nullptr, *knc = nullptr, *wg = nullptr,
              *wu = nullptr, *wd = nullptr;
    int local = 0, cap = 0;
    float *ck = nullptr, *cv = nullptr;
  } L[3];
  const u16 *norm = nullptr, *mask = nullptr, *w1 = nullptr, *rnn_w = nullptr, *rnn_b = nullptr, *head_norm = nullptr;
  const u16* f16 = nullptr;
  const float *f32 = nullptr, *fb = nullptr;
  float alpha[4][4] = {};
  float *cmix = nullptr, *kvraw = nullptr, *frnn = nullptr, *rcos = nullptr, *rsin = nullptr, *featbuf = nullptr;
  float *h = nullptr, *x = nullptr, *hn = nullptr, *qb = nullptr, *kb = nullptr, *vb = nullptr, *ab = nullptr, *ob = nullptr;
  float *gb = nullptr, *ub = nullptr, *mb = nullptr, *base = nullptr, *z = nullptr, *zz = nullptr, *s = nullptr;
  float *pval = nullptr, *scratch = nullptr;
  int *pidx = nullptr, *drafts = nullptr;
  int fring = 0, smax = 0, G = 1024;
  bool use16 = true, dbg_feat = false, prof = false;
  int pending = 0;                // depth of a draft call to run before the next pass
  std::map<int, void*> graphs;    // draft calls: key = 2 * depth + use16
  double part_t[3] = {};          // profile: block, base logits, Markov head (seconds)
  long long launches = 0;
};

}  // namespace

void Engine::drf_destroy() {
  if (!drf) return;
  for (auto& kv : drf->graphs)
    if (kv.second) be->graph_free(kv.second);
  for (void* p : drf->owned) be->release(p);
  for (auto& kv : drf->ten)
    if (kv.second.p) be->release(kv.second.p);
  delete drf;
  drf = nullptr;
}

namespace {

template <class T>
T* drf_alloc(Engine* e, size_t count) {
  void* p = e->be->alloc(count * sizeof(T));
  if (!p) fail(e->be->error() ? e->be->error() : "allocation failed");
  e->drf->owned.push_back(p);
  e->drf->bytes += (count * sizeof(T) + 255) & ~(size_t)255;
  return (T*)p;
}

const void* drf_need(Drafter& d, const std::string& name, long long nbytes) {
  auto it = d.ten.find(name);
  if (it == d.ten.end()) fail("drafter: missing tensor " + name);
  if (it->second.nbytes != nbytes)
    fail(fmt("drafter: tensor %s has %lld bytes, expected %lld", name.c_str(), it->second.nbytes, nbytes));
  return it->second.p;
}

void drf_finalize(Engine* e) {
  Drafter& d = *e->drf;
  const Ink2DraftConfig& c = d.c;
  const long long H = c.hidden, A = (long long)c.heads * 128, C = (long long)c.kv_heads * 128, I = c.inter, R = c.rank;
  const long long V = e->c.vocab;
  if (!e->finalized) fail("drafter: finalize the engine first");
  if (d.finalized) fail("drafter: finalize called twice");
  if (H != e->c.hidden) fail("drafter: hidden size differs from the engine's");
  if (c.layers != 3 || c.head_dim != 128 || c.block != 8 || c.rank != 256)
    fail("drafter: needs 3 layers, head size 128, block 8, Markov rank 256");
  if (c.heads % c.kv_heads || c.heads > 64 || H % 256 || I % 8) fail("drafter: unsupported head counts or sizes");
  if (c.w1_rows < V) fail("drafter: markov_w1 has fewer rows than the vocabulary");
  for (int j = 0; j < 3; ++j)
    if (c.feat_visits[j] < 0 || c.feat_visits[j] >= (int)e->visits.size()) fail("drafter: feature visit out of range");
  auto bf = [&](const std::string& n, long long count) { return (const u16*)drf_need(d, n, count * 2); };
  for (int l = 0; l < 3; ++l) {
    Drafter::DL& Lr = d.L[l];
    std::string p = fmt("L%d.", l);
    Lr.dflare = bf(p + "dflare", H);
    Lr.in_ln = bf(p + "in_ln", H);
    Lr.post_ln = bf(p + "post_ln", H);
    Lr.wq = bf(p + "wq", A * H);
    Lr.wk = bf(p + "wk", C * H);
    Lr.wv = bf(p + "wv", C * H);
    Lr.wo = bf(p + "wo", H * A);
    Lr.qn = bf(p + "qn", 128);
    Lr.kn = bf(p + "kn", 128);
    Lr.wkc = bf(p + "wkc", C * H);
    Lr.wvc = bf(p + "wvc", C * H);
    Lr.knc = bf(p + "knc", 128);
    Lr.wg = bf(p + "wg", I * H);
    Lr.wu = bf(p + "wu", I * H);
    Lr.wd = bf(p + "wd", H * I);
    Lr.local = c.local[l] != 0;
  }
  d.norm = bf("norm", H);
  d.mask = bf("mask", H);
  d.w1 = bf("w1", (long long)c.w1_rows * R);
  d.rnn_w = bf("rnn_w", 3 * R * (2 * R + 2 * H));
  d.rnn_b = bf("rnn_b", 3 * R);
  d.head_norm = bf("head_norm", H);
  d.f16 = d.ten.count("fold16") ? bf("fold16", V * R) : nullptr;
  d.f32 = d.ten.count("fold32") ? (const float*)drf_need(d, "fold32", V * R * 4) : nullptr;
  if (!d.f16 && !d.f32) fail("drafter: needs fold16 or fold32");
  d.use16 = d.f16 != nullptr;
  d.fb = (const float*)drf_need(d, "fold_b", V * 4);
  drf_need(d, "fuse_alpha", 64);
  memcpy(d.alpha, d.ten["fuse_alpha"].host.data(), 64);

  const size_t MR = (size_t)e->max_rows, MP = (size_t)e->c.max_positions;
  d.fring = e->margin + 16;
  for (int l = 0; l < 3; ++l) {
    long long ring = (long long)c.window + e->margin + 8;
    d.L[l].cap = d.L[l].local ? (int)(ring < (long long)MP ? ring : (long long)MP) : (int)MP;
    d.L[l].ck = drf_alloc<float>(e, (size_t)d.L[l].cap * C);
    d.L[l].cv = drf_alloc<float>(e, (size_t)d.L[l].cap * C);
  }
  d.cmix = drf_alloc<float>(e, 3 * MR * H);
  d.kvraw = drf_alloc<float>(e, 6 * MR * C);
  d.frnn = drf_alloc<float>(e, (size_t)d.fring * H);
  d.featbuf = drf_alloc<float>(e, 4 * MR * H);
  d.h = drf_alloc<float>(e, 8 * H);
  d.x = drf_alloc<float>(e, 8 * H);
  d.hn = drf_alloc<float>(e, 8 * H);
  d.qb = drf_alloc<float>(e, 8 * A);
  d.kb = drf_alloc<float>(e, 8 * C);
  d.vb = drf_alloc<float>(e, 8 * C);
  d.ab = drf_alloc<float>(e, 8 * A);
  d.ob = drf_alloc<float>(e, 8 * H);
  d.gb = drf_alloc<float>(e, 8 * I);
  d.ub = drf_alloc<float>(e, 8 * I);
  d.mb = drf_alloc<float>(e, 8 * I);
  d.base = drf_alloc<float>(e, 7 * (size_t)V);
  d.z = drf_alloc<float>(e, (size_t)(2 * R + 2 * H));
  d.zz = drf_alloc<float>(e, (size_t)(3 * R));
  d.s = drf_alloc<float>(e, (size_t)R);
  d.pval = drf_alloc<float>(e, (size_t)d.G);
  d.pidx = drf_alloc<int>(e, (size_t)d.G);
  d.drafts = drf_alloc<int>(e, 8);
  d.smax = (int)MP + 8;
  d.scratch = drf_alloc<float>(e, (size_t)8 * c.heads * d.smax);
  // RoPE tables (rotate-half, theta of the drafter), double precision as ink2_draft.rope
  const size_t NP = MP + 8;
  d.rcos = drf_alloc<float>(e, NP * 64);
  d.rsin = drf_alloc<float>(e, NP * 64);
  {
    std::vector<float> tc(NP * 64), ts(NP * 64);
    for (int j = 0; j < 64; ++j) {
      double inv = 1.0 / pow((double)c.rope_theta, (double)(2 * j) / 128.0);
      for (size_t p = 0; p < NP; ++p) {
        double ang = (double)p * inv;
        tc[p * 64 + j] = (float)cos(ang);
        ts[p * 64 + j] = (float)sin(ang);
      }
    }
    e->be->upload(d.rcos, tc.data(), tc.size() * 4);
    e->be->upload(d.rsin, ts.data(), ts.size() * 4);
  }
  for (auto& kv : d.ten) std::vector<unsigned char>().swap(kv.second.host);
  e->be->sync();
  e->check_backend();
  e->free_graphs();  // the passes are recorded again, now with the commit
  d.finalized = true;
}

}  // namespace

// Commit of the R rows of the pass that just ran (their features are on the device).
void Engine::drf_commit(int R) {
  Drafter& d = *drf;
  if (!d.finalized) return;
  const int H = c.hidden, C = d.c.kv_heads * 128;
  const size_t MR = (size_t)max_rows;
  DMixArgs m{};
  for (int j = 0; j < 3; ++j) m.f[j] = d.dbg_feat ? d.featbuf + (size_t)j * MR * H : visits[d.c.feat_visits[j]].hid;
  m.f[3] = d.dbg_feat ? d.featbuf + 3 * MR * H : fh;
  for (int l = 0; l < 3; ++l) m.w[l] = d.L[l].dflare;
  m.w[3] = d.head_norm;
  memcpy(m.alpha, d.alpha, sizeof m.alpha);
  m.cmix = d.cmix;
  m.frnn = d.frnn;
  m.st = st;
  m.H = H;
  m.max_rows = max_rows;
  m.fring = d.fring;
  m.eps = d.c.eps;
  run(KID_dmix, &m, 4 * R, 256, INK2_PROF_MIX_NORM);
  GemvArgs g{};
  for (int l = 0; l < 3; ++l) {
    g.job[2 * l] = drf_job(d.L[l].wkc, d.cmix + (size_t)l * MR * H, d.kvraw + (size_t)(2 * l) * MR * C, C, H);
    g.job[2 * l + 1] = drf_job(d.L[l].wvc, d.cmix + (size_t)l * MR * H, d.kvraw + (size_t)(2 * l + 1) * MR * C, C, H);
  }
  g.njobs = 6;
  k_gemv(g, INK2_PROF_MATRIX, R);
  DCtxArgs x{};
  x.kv = d.kvraw;
  for (int l = 0; l < 3; ++l) {
    x.knorm[l] = d.L[l].knc;
    x.ck[l] = d.L[l].ck;
    x.cv[l] = d.L[l].cv;
    x.cap[l] = d.L[l].cap;
  }
  x.rcos = d.rcos;
  x.rsin = d.rsin;
  x.st = st;
  x.nkv = d.c.kv_heads;
  x.max_rows = max_rows;
  x.eps = d.c.eps;
  run(KID_dctx, &x, R * 3 * d.c.kv_heads, 128, INK2_PROF_ATTENTION);
}

namespace {

// The kernels of one draft call for the anchor st[0] (ink2_draft.Ink2Drafter.propose).
void drf_kernels(Engine* e, int depth) {
  Drafter& d = *e->drf;
  const Ink2DraftConfig& c = d.c;
  const int H = c.hidden, A = c.heads * 128, C = c.kv_heads * 128, I = c.inter, R = c.rank, V = e->c.vocab;
  auto mark = [&](int part) {
    if (!d.prof) return;
    static double t_last = 0;
    e->be->sync();
    double t = now_s();
    if (part >= 0) d.part_t[part] += t - t_last;
    t_last = t;
  };
  long long n0 = e->n_launch;
  mark(-1);
  {  // row 0: RMSNorm(embed[anchor]) * embed_norm, the engine's own embedding kernel
    NormArgs n0{};
    drf_embed_src(n0, e->embed);
    n0.job[0].w = e->embed_norm;
    n0.job[0].out = d.h;
    n0.st = e->st;
    n0.embed_stride = H;
    n0.n = H;
    n0.eps = c.eps;
    n0.nj = 1;
    n0.rows = 1;
    e->run(KID_norm, &n0, 1, 256, INK2_PROF_MIX_NORM);
  }
  DEmbArgs em{};
  em.mask = d.mask;
  em.h = d.h;
  em.H = H;
  e->run(KID_demb, &em, 7, 256, INK2_PROF_MIX_NORM);
  NormArgs n{};
  n.job[0].x = d.h;
  n.job[0].w = d.L[0].in_ln;
  n.job[0].out = d.x;
  n.st = e->st;
  n.n = H;
  n.eps = c.eps;
  n.nj = 1;
  n.rows = 8;
  e->run(KID_norm, &n, 8, 256, INK2_PROF_MIX_NORM);
  for (int l = 0; l < 3; ++l) {
    const Drafter::DL& Lr = d.L[l];
    GemvArgs a{};
    a.job[0] = drf_job(Lr.wq, d.x, d.qb, A, H);
    a.job[1] = drf_job(Lr.wk, d.x, d.kb, C, H);
    a.job[2] = drf_job(Lr.wv, d.x, d.vb, C, H);
    a.njobs = 3;
    e->k_gemv(a, INK2_PROF_MATRIX, 8);
    DQkArgs q{};
    q.q = d.qb;
    q.k = d.kb;
    q.qn = Lr.qn;
    q.kn = Lr.kn;
    q.rcos = d.rcos;
    q.rsin = d.rsin;
    q.st = e->st;
    q.nh = c.heads;
    q.nkv = c.kv_heads;
    q.eps = c.eps;
    e->run(KID_dqk, &q, 8 * (c.heads + c.kv_heads), 128, INK2_PROF_ATTENTION);
    DAttnArgs t{};
    t.q = d.qb;
    t.kb = d.kb;
    t.vb = d.vb;
    t.ck = Lr.ck;
    t.cv = Lr.cv;
    t.scratch = d.scratch;
    t.out = d.ab;
    t.st = e->st;
    t.nh = c.heads;
    t.nkv = c.kv_heads;
    t.cap = Lr.cap;
    t.window = Lr.local ? c.window : 0;
    t.smax = d.smax;
    t.scale = (float)(1.0 / sqrt(128.0));
    e->run(KID_dattn, &t, 8 * c.heads, 256, INK2_PROF_ATTENTION);
    GemvArgs o{};
    o.job[0] = drf_job(Lr.wo, d.ab, d.ob, H, A);
    o.njobs = 1;
    e->k_gemv(o, INK2_PROF_MATRIX, 8);
    DAddNormArgs an{};
    an.h = d.h;
    an.y = d.ob;
    an.w = Lr.post_ln;
    an.out = d.x;
    an.H = H;
    an.eps = c.eps;
    e->run(KID_daddnorm, &an, 8, 256, INK2_PROF_MIX_NORM);
    GemvArgs gu{};
    gu.job[0] = drf_job(Lr.wg, d.x, d.gb, I, H);
    gu.job[1] = drf_job(Lr.wu, d.x, d.ub, I, H);
    gu.njobs = 2;
    e->k_gemv(gu, INK2_PROF_MATRIX, 8);
    DSiluArgs sl{};
    sl.g = d.gb;
    sl.u = d.ub;
    sl.out = d.mb;
    sl.n = 8 * I;
    e->run(KID_dsilu, &sl, (8 * I + 255) / 256, 256, INK2_PROF_MIX_NORM);
    GemvArgs dn{};
    dn.job[0] = drf_job(Lr.wd, d.mb, d.ob, H, I);
    dn.njobs = 1;
    e->k_gemv(dn, INK2_PROF_MATRIX, 8);
    DAddNormArgs a2{};
    a2.h = d.h;
    a2.y = d.ob;
    a2.w = l < 2 ? d.L[l + 1].in_ln : d.norm;
    a2.out = l < 2 ? d.x : d.hn;
    a2.H = H;
    a2.eps = c.eps;
    e->run(KID_daddnorm, &a2, 8, 256, INK2_PROF_MIX_NORM);
  }
  mark(0);
  {  // base logits U h of rows 1 .. depth (ink2_draft.compute_base_logits)
    GemvArgs b{};
    b.job[0] = Engine::job(e->unembed, d.hn + H, d.base, V, H);
    b.njobs = 1;
    e->k_gemv(b, INK2_PROF_OUTPUT, depth);
  }
  mark(1);
  for (int i = 0; i <= depth; ++i) {
    DZinArgs zi{};
    zi.pval = d.pval;
    zi.pidx = d.pidx;
    zi.drafts = d.drafts;
    zi.st = e->st;
    zi.s = d.s;
    zi.w1 = d.w1;
    zi.hn = d.hn;
    zi.frnn = d.frnn;
    zi.z = d.z;
    zi.G = d.G;
    zi.step = i;
    zi.depth = depth;
    zi.rank = R;
    zi.H = H;
    zi.fring = d.fring;
    zi.w1_rows = c.w1_rows;
    e->run(KID_dzin, &zi, 1, 256, INK2_PROF_GATES);
    if (i == depth) break;
    GemvArgs rp{};
    rp.job[0] = drf_job(d.rnn_w, d.z, d.zz, 3 * R, 2 * R + 2 * H);
    rp.njobs = 1;
    e->k_gemv(rp, INK2_PROF_GATES, 1);
    DFoldArgs f{};
    f.zz = d.zz;
    f.rnn_b = d.rnn_b;
    f.s = d.s;
    f.base = d.base + (size_t)i * V;
    f.f16 = d.use16 ? d.f16 : nullptr;
    f.f32 = d.f32;
    f.fb = d.fb;
    f.pval = d.pval;
    f.pidx = d.pidx;
    f.V = V;
    f.rank = R;
    f.step = i;
    f.G = d.G;
    f.cap = c.softcap;
    e->run(KID_dfold, &f, d.G, 256, INK2_PROF_OUTPUT);
  }
  mark(2);
  d.launches = e->n_launch - n0;
}

void drf_draft(Engine* e, int depth) {
  Drafter& d = *e->drf;
  bool direct = !(e->use_graph && e->be->graph_supported()) || e->prof || e->bench_reps > 0 || d.prof;
  if (direct) {
    drf_kernels(e, depth);
    return;
  }
  void*& g = d.graphs[2 * depth + (d.use16 ? 1 : 0)];
  if (!g) {
    e->be->graph_begin();
    drf_kernels(e, depth);
    g = e->be->graph_end();
    e->check_backend();
  }
  e->be->graph_launch(g);
}

Engine* drf_engine(void* h) {
  Engine* e = (Engine*)h;
  if (!e->drf || !e->drf->finalized) fail("no drafter loaded");
  return e;
}

}  // namespace

// Before the kernels of a pass (its rows are in st): the pending draft call writes its
// tokens into st[1 .. depth].
void Engine::drf_before_pass(int R) {
  Drafter& d = *drf;
  if (!d.pending) return;
  int depth = d.pending;
  d.pending = 0;
  if (R != depth + 1) fail("drafter: pass size does not match the pending draft");
  drf_draft(this, depth);
}

INK2_API int ink2_draft_abi(void) { return INK2_DRAFT_ABI; }

INK2_API int ink2_draft_create(void* h, const Ink2DraftConfig* cfg) {
  API_BEGIN(h)
  if (!e->finalized) fail("drafter: finalize the engine first");
  e->drf_destroy();
  e->free_graphs();
  e->drf = new Drafter();
  e->drf->c = *cfg;
  return 0;
  API_END
}

INK2_API int ink2_draft_set_tensor(void* h, const char* name, const void* data, long long nbytes) {
  API_BEGIN(h)
  if (!e->drf) fail("drafter: create first");
  Drafter& d = *e->drf;
  if (d.finalized) fail("drafter: set_tensor after finalize");
  if (nbytes <= 0) fail(std::string("drafter: empty tensor ") + name);
  Ten& t = d.ten[name];
  if (t.p) fail(std::string("drafter: tensor set twice: ") + name);
  t.p = e->be->alloc((size_t)nbytes);
  if (!t.p) fail(e->be->error() ? e->be->error() : "allocation failed");
  d.bytes += ((size_t)nbytes + 255) & ~(size_t)255;
  t.nbytes = nbytes;
  e->be->upload(t.p, data, (size_t)nbytes);
  if (nbytes <= 65536) t.host.assign((const unsigned char*)data, (const unsigned char*)data + nbytes);
  e->check_backend();
  return 0;
  API_END
}

INK2_API int ink2_draft_finalize(void* h) {
  API_BEGIN(h)
  if (!e->drf) fail("drafter: create first");
  try {
    drf_finalize(e);
  } catch (...) {
    e->drf_destroy();
    e->free_graphs();
    throw;
  }
  return 0;
  API_END
}

INK2_API int ink2_draft_unload(void* h) {
  API_BEGIN(h)
  e->drf_destroy();
  e->free_graphs();
  return 0;
  API_END
}

INK2_API int ink2_draft_loaded(void* h) {
  Engine* e = (Engine*)h;
  return e && e->drf && e->drf->finalized ? 1 : 0;
}

INK2_API int ink2_draft_set_option(void* h, const char* name, int value) {
  API_BEGIN(h)
  Engine* x = drf_engine(h);
  Drafter& d = *x->drf;
  std::string n = name;
  if (n == "fold") {
    if (value == 16 && !d.f16) fail("drafter: no BF16 folded head loaded");
    if (value == 32 && !d.f32) fail("drafter: no float32 folded head loaded");
    if (value != 16 && value != 32) fail("drafter: fold must be 16 or 32");
    d.use16 = value == 16;
  } else if (n == "profile") {
    d.prof = value != 0;
  } else fail("drafter: unknown option " + n);
  return 0;
  API_END
}

INK2_API int ink2_spec_pass(void* h, int anchor, int depth, int* drafts_out, int* argmax_out, int truncate) {
  API_BEGIN(h)
  drf_engine(h);
  if (depth < 1 || depth > 7) fail("drafter: depth must be 1 .. 7");
  if (e->pos < 1) fail("drafter: no context yet (feed the prompt first)");
  int R = depth + 1;
  if (R > e->logit_rows || R > e->max_rows) fail("drafter: pass of depth + 1 rows is too large for this engine");
  std::vector<int> ids((size_t)R, anchor);  // rows 1 .. depth: written by the draft call on the device
  e->drf->pending = depth;
  try {
    e->pass(ids.data(), R, true, nullptr, argmax_out, nullptr);
  } catch (...) {
    e->drf->pending = 0;
    throw;
  }
  e->be->download(drafts_out, e->drf->drafts, (size_t)depth * 4);
  e->check_backend();
  int acc = 0;
  while (acc < depth && drafts_out[acc] == argmax_out[acc]) ++acc;
  if (truncate) e->truncate(e->pos - (depth - acc));
  return acc;
  API_END
}

INK2_API int ink2_draft_propose(void* h, int anchor, int pos, int depth, int* drafts_out) {
  API_BEGIN(h)
  drf_engine(h);
  if (depth < 1 || depth > 7) fail("drafter: depth must be 1 .. 7");
  if (pos < 1 || pos >= e->c.max_positions) fail("drafter: position out of range");
  if (anchor < 0 || anchor >= e->c.embed_rows) fail("token id out of range");
  StepState s = e->state_at(anchor, pos);
  e->be->upload(e->st, &s, sizeof s);
  drf_draft(e, depth);
  e->be->download(drafts_out, e->drf->drafts, (size_t)depth * 4);
  e->check_backend();
  return 0;
  API_END
}

INK2_API int ink2_draft_commit_features(void* h, const float* feats, int p0, int n) {
  API_BEGIN(h)
  drf_engine(h);
  if (n < 1 || n > e->max_rows) fail("drafter: 1 .. max_rows rows");
  if (p0 < 0 || p0 + n > e->c.max_positions) fail("drafter: positions out of range");
  Drafter& d = *e->drf;
  const size_t H = (size_t)e->c.hidden, MR = (size_t)e->max_rows;
  std::vector<float> rowbuf(H);
  for (int r = 0; r < n; ++r) {
    for (int j = 0; j < 4; ++j)
      e->be->upload(d.featbuf + ((size_t)j * MR + r) * H, feats + ((size_t)r * 4 + j) * H, H * 4);
    e->st_host[r] = e->state_at(0, p0 + r);
  }
  e->be->upload(e->st, e->st_host.data(), (size_t)n * sizeof(StepState));
  d.dbg_feat = true;
  try {
    e->drf_commit(n);
  } catch (...) {
    d.dbg_feat = false;
    throw;
  }
  d.dbg_feat = false;
  e->be->sync();
  e->check_backend();
  return 0;
  API_END
}

INK2_API int ink2_draft_time(void* h, int depth, int reps, double* out5) {
  API_BEGIN(h)
  drf_engine(h);
  Drafter& d = *e->drf;
  if (depth < 1 || depth > 7 || reps < 1) fail("drafter: depth 1 .. 7, reps >= 1");
  int pos = e->pos > 0 ? e->pos : 1;
  StepState s = e->state_at(0, pos);
  e->be->upload(e->st, &s, sizeof s);
  drf_draft(e, depth);  // records the graph
  e->be->sync();
  double t0 = now_s();
  for (int r = 0; r < reps; ++r) drf_draft(e, depth);
  e->be->sync();
  out5[0] = 1e3 * (now_s() - t0) / reps;
  bool was = d.prof;
  d.prof = true;
  for (int i = 0; i < 3; ++i) d.part_t[i] = 0;
  for (int r = 0; r < reps; ++r) drf_draft(e, depth);
  d.prof = was;
  for (int i = 0; i < 3; ++i) out5[1 + i] = 1e3 * d.part_t[i] / reps;
  out5[4] = (double)d.launches;
  e->check_backend();
  return 0;
  API_END
}

INK2_API long long ink2_draft_memory(void* h) {
  Engine* e = (Engine*)h;
  return e && e->drf ? (long long)e->drf->bytes : 0;
}

INK2_API int ink2_draft_get(void* h, int what, int index, float* out, long long nfloats) {
  API_BEGIN(h)
  drf_engine(h);
  Drafter& d = *e->drf;
  const long long H = e->c.hidden, V = e->c.vocab;
  const float* src = nullptr;
  long long n = 0;
  if (what == 0) src = d.hn, n = 8 * H;
  else if (what == 1) {
    if (index < 1 || index > 7) fail("drafter: index = depth 1 .. 7");
    src = d.base, n = index * V;
  } else if (what == 2) {
    if (index < 0) fail("drafter: position out of range");
    src = d.frnn + (size_t)(index % d.fring) * H, n = H;
  } else fail("drafter: unknown buffer");
  if (nfloats != n) fail(fmt("drafter: buffer has %lld floats, the data %lld", nfloats, n));
  e->be->download(out, src, (size_t)n * 4);
  e->check_backend();
  return 0;
  API_END
}
