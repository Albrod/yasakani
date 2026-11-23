// SPDX-License-Identifier: Apache-2.0
// Fast kernels, second set (ink2_fast2.h): members and methods of struct Engine, included inside its body after
// ink2_fast_engine.inc. Active only on the fast path (fast != 0); the general path is untouched.
//   option "f2_pf"      : 0 off, 1 prefetch.global.L2 lines, 2 cp.async.bulk.prefetch.L2, 3 (default) code words in
//                         registers where the kernel has that form (pgemvc, fgemvm, router, fx2c), else lines - weight bytes of a
//                         kernel asked for before its griddepcontrol.wait (needs programmatic dependent launch)
//   option "f2_extra"   : bits 1 dense / 2 expert w2 / 4 expert w13: the first wave also prefetches the later waves
//   option "f2_budget"  : MB of such further prefetch per launch (default 24)
//   option "f2_trig"    : 1 = the small kernels before a weight kernel trigger their dependents before their own wait
//   option "f2_gemvm"   : 1 = fgemvm for packed passes of R > 1 rows (inputs in registers, many matrix rows per block)
//   option "f2_router"  : 1 = router product and top-k in one launch
//   option "f2_nb"      : matrix rows per block of fgemvm (0 = by size, else 8, 16, 24, 32)
//   further options (defaults = what measured best on the RTX 5090, quality folder, x_fast2 jobs j5 .. j14):
//   f2_router_rows 1 (fused router only for one row: slower for passes), f2_min_rows 4 (fgemvm from 4 rows),
//   f2_reg 31 (register forms: 1 dense, 2 fgemvm, 4 router, 8 fx2c; 16 = attn key / value prefetch), f2_fuse 0 and
//   f2_ccomb 0 (fxcomb inside the next bar / inside fx2c: both measured slower), f2_npf 0 (prefetch blocks for the next
//   visit's projections: no gain), f2_spec 0 (L2 prefetch of the previous step's experts: no gain), f2_pipe 0 (resident
//   grid for the expert kernels: slower), f2_prepb 1, f2_wv 4, f2_extra 0 (later-wave prefetch: slower).
// Every one of them keeps the numbers: the wrapped kernels run the unchanged bodies, fgemvm has the sums of pgemvc,
// the fused router runs the bodies of gemvw / gemvwm and topk.
int f2_pf = 3, f2_extra = 0, f2_budget = 24, f2_trig = 1, f2_gemvm = 1, f2_router = 1, f2_nb = 0;
int f2_min_blocks = 300;  // fgemvm: the largest nb that still gives this many blocks
int f2_min_rows = 4;      // option "f2_min_rows": fgemvm from this many rows (fewer: pgemvc once per row; 2, 3 measured slower)
int f2_max_rows = 16;     // option "f2_max_rows": fgemvm for passes of 2 .. this many rows (prompt passes: general)
int f2_wv = 4;            // option "f2_wv": rows per wave of fgemvm (4 or 8)
int f2_fuse = 0;          // option "f2_fuse": the fxcomb step of a visit inside the next bar (FBARC; measured slower, off)
FxArgs f2_pend{};         // a deferred fxcomb (its launch waits for the next kernel: the bar takes it over, else it runs)
int f2_pend_grid = 0;
bool f2_has_pend = false;
unsigned* f2_done = nullptr;
bool f2_active = false;

// attention at long context (INK2_F2_ATTN, ink2_fast2.h). Option "f2_attn": 1 = steps and passes of
// up to f2_attn_rows rows from position f2_attn_from on run kf2_attn_score + kf2_attn_value instead of the general attn
// kernel (the same bits; a step and a pass record one graph for each side of the threshold); 0 = the general kernel.
// "f2_attn_sa" score blocks per (row, kv head), "f2_attn_sw" stripes per value block, "f2_attn_pf" bits 1 / 2 / 4
// (the score blocks ask L2 for their key / value rows, the value blocks for their value rows, before the wait),
// "f2_attn_ng" groups of 8 keys a warp of the score blocks loads at once.
// defaults: measured on the RTX 5090 (L2 flushed): sa 32 / sw 8 best at 1,024 keys (the sliding window), within
// 2-4 % of the best at 4K - 32K keys; the prefetch forms (pf) measured no faster; below about 512 keys the general kernel wins.
int f2_attn = 1, f2_attn_from = 512, f2_attn_rows = 8, f2_attn_sa = 32, f2_attn_sw = 8, f2_attn_pf = 0, f2_attn_ng = 2;
bool f2_attn_lv = false;  // the step / pass being run (or recorded) uses it
float *f2_apmax = nullptr, *f2_apart = nullptr, *f2_aes = nullptr;
unsigned* f2_acnt = nullptr;
int f2_attn_nh = 0;
static const int F2_ATTN_ROWS = 16;  // rows the buffers hold
bool f2_attn_use(int p, int R) const {
  return f2_active && f2_attn && f2_apmax && R <= f2_attn_rows && R <= F2_ATTN_ROWS && p >= f2_attn_from;
}
bool f2_attn_run(const AttnArgs& a, int R) {
  if (!f2_attn_lv || !f2_active || !f2_apmax || R > F2_ATTN_ROWS) return false;
  if (a.hd != 128 || a.nkv < 1 || a.nh % a.nkv || a.nh / a.nkv > 4 || a.drel > 32 || a.nh > f2_attn_nh) return false;
  F2AttnArgs f{};
  f.a = a;
  f.pmax = f2_apmax;
  f.part = f2_apart;
  f.es = f2_aes;
  f.cnt = f2_acnt;
  f.sa = f2_attn_sa;
  f.sw = f2_attn_sw;
  f.pf = f2_attn_pf;  // (L2 prefetch only: safe with or without programmatic dependent launch)
  f.ng = f2_attn_ng;
  run_f2k(INK2_F2_ATTN, &f, 0, 0, INK2_PROF_ATTENTION);
  return true;
}

void f2_resolve() {
  f2_active = false;
  if (!finalized || !fast) {
    be->set_f2(nullptr);
    return;
  }
  if (f2_attn && !f2_apmax) {  // (never during a graph capture: options and finalize only)
    int nh = 0;
    for (const Layer& L : layers) nh = L.nh > nh ? L.nh : nh;
    f2_attn_nh = nh;
    const size_t rows = F2_ATTN_ROWS;
    f2_apmax = alloc<float>(rows * nh * 32);
    f2_apart = alloc<float>(rows * nh * 32 * 128);
    f2_aes = alloc<float>(rows * nh * 1024);
    f2_acnt = alloc<unsigned>(rows * nh);
  }
  F2Opts o{};
  o.pf = pdl_on ? f2_pf : 0;
  o.pf_extra = f2_extra;
  o.pf_budget = (long long)f2_budget << 20;
  o.trig_small = pdl_on ? f2_trig : 0;
  o.gemvm = f2_gemvm;
  o.prep_minb = f2_prepb;
  o.reg = f2_reg;
  o.pipe = f2_pipe;
  o.pipe_waves = f2_pwaves;
  o.bfm = f2_bfm;
  be->set_f2(&o);
  if (!f2_done) f2_done = alloc<unsigned>(64);
  if (!f2_ccnt) f2_ccnt = alloc<unsigned>(4096);
  f2_active = true;
}
// experts_fast: keep the fxcomb launch for the bar that follows (false: launch it now)
bool f2_defer_comb(const FxArgs& f, int grid) {
  if (!f2_active || !f2_fuse || !f.cw || !fsmall || prof || cap_visit >= 0 || bench_reps > 0) return false;
  f2_pend = f;
  f2_pend_grid = grid;
  f2_has_pend = true;
  return true;
}
// experts_fast: fx2c (FX2CL4) with the fxcomb step done by the last block of every chunk (true: launched)
int f2_ccomb = 0;           // option "f2_ccomb" (measured slower: off)
int f2_pipe = 0;            // option "f2_pipe": bits 1 fx13, 2 fx2c in the PIPE form
int f2_pwaves = 1;          // option "f2_pwaves": PIPE grid = resident blocks times this
int f2_reg = 31;            // option "f2_reg": register forms with f2_pf 3 (bits 1 dense, 2 fgemvm, 4 router, 8 fx2c, 16 attn keys)
int f2_prepb = 1;           // option "f2_prepb": fprep for 1 or 2 blocks per SM
unsigned* f2_ccnt = nullptr;
bool f2_col_comb(const FxArgs& f, int grid, int block) {
  if (!f2_active || !f2_ccomb || !f.cw || f.nchunk > 4096 || prof || cap_visit >= 0 || bench_reps > 0) return false;
  if (!f2_ccnt) return false;  // allocated by f2_resolve (never during a graph capture)
  F2ColComb cc{};
  cc.a = f;
  cc.c = f;
  cc.c.x = ydots;
  cc.c.y = ey;
  cc.cnt = f2_ccnt;
  run_fx(INK2_FX2CL4C, &cc, grid, block, 0, INK2_PROF_EXPERTS);
  return true;
}
// fxcomb + f2_npf blocks that ask L2 for the projection weights of the next visit (true: launched)
int f2_npf = 0;    // option "f2_npf": prefetch blocks (0 = off; 16 .. 128 measured no faster)
int f2_pfel = 1;   // option "f2_pfel": the prefetch with evict_last
static size_t f2_mbytes(const DMat& m, int rows, int cols) {
  if (!m.w) return 0;
  if (m.fmt == INK2_DFMT_NVFP4) return (size_t)rows * (size_t)(cols / 2);
  if (m.fmt == INK2_DFMT_MXFP8) return (size_t)rows * (size_t)cols;
  return (size_t)rows * (size_t)cols * 2;
}
static size_t f2_sbytes(const DMat& m, int rows, int cols) {
  if (!m.sc) return 0;
  return m.fmt == INK2_DFMT_NVFP4 ? (size_t)rows * (size_t)(cols / 16) : m.fmt == INK2_DFMT_MXFP8 ? (size_t)rows * (size_t)(cols / 32) : 0;
}
bool f2_comb_pf(const Visit& V, const FxArgs& f, int grid) {
  if (!f2_active || !f2_npf || !pdl_on || prof || cap_visit >= 0 || bench_reps > 0) return false;
  const size_t vi = (size_t)(&V - visits.data());
  if (vi + 1 >= visits.size()) return false;
  const Layer& N = layers[visits[vi + 1].layer];
  const int H = c.hidden, A = N.nh * N.hd, C = N.nkv * N.hd;
  F2CombPf cp{};
  cp.a = f;
  int n = 0;
  auto add = [&](const DMat& m, int rows) {
    size_t b = f2_mbytes(m, rows, H), s = f2_sbytes(m, rows, H);
    if (b && n < 12) cp.s.p[n] = m.w, cp.s.n[n++] = b;
    if (s && n < 12) cp.s.p[n] = m.sc, cp.s.n[n++] = s;
  };
  add(N.wq, A);
  add(N.wr, N.nh * c.d_rel);
  add(N.w_gate, A);
  add(N.w_sink, N.nh);
  add(N.wk, C);
  add(N.wv, C);
  cp.s.cnt = n;
  cp.s.nwork = grid;
  cp.s.evict_last = f2_pfel;
  run_fx(INK2_FXCOMBP, &cp, grid + f2_npf, 256, 0, INK2_PROF_EXPERTS);
  return true;
}
// every other launch first runs a deferred fxcomb
void f2_flush() {
  if (!f2_has_pend) return;
  f2_has_pend = false;
  run_fx(INK2_FXCOMB, &f2_pend, f2_pend_grid, 256, 0, INK2_PROF_EXPERTS);
}
// k_bar: the deferred fxcomb inside this bar when job 0 continues from its output (true: launched)
bool f2_bar_comb(const BarArgs& a, int R) {
  if (!f2_has_pend) return false;
  const BarJob& J = a.job[0];
  if (!fast_bar(a) || J.cw || J.ns < 1 || J.st[J.ns - 1] != f2_pend.cout || f2_pend.R != R) {
    f2_flush();
    return false;
  }
  F2BarComb bc{};
  bc.b = a;
  bc.c = f2_pend;
  f2_has_pend = false;
  run_fx(INK2_FBARC, &bc, a.nj * R, quad_block(), 0, INK2_PROF_MIX_NORM);
  return true;
}
void run_f2k(int fid, const void* a, int grid, int block, int cat) {
  f2_flush();
  if (bench_reps > 0) {
    be->run_f2(fid, a, grid, block);
    return;
  }
  be->run_f2(fid, a, grid, block);
  ++n_launch;
  if (prof) {
    be->sync();
    double t = now_s();
    prof_t[cat] += t - prof_mark;
    prof_mark = t;
  }
}
// fgemvm2 (INK2_F2_GEMVM2) for packed passes of f2_gm2_min .. f2_gm2_max rows: two waves of code
// words in flight, 2 / 4 / 8 rows of the pass per block (by R), the groups of a chunk next to each other in the grid.
//   option "f2_gm2"     : 1 (default) = fgemvm2, 0 = the first set (fgemvm for f2_min_rows .. f2_max_rows rows, else general)
//   option "f2_gm2_min" / "f2_gm2_max" : its pass sizes (default 2 .. 128: verification, copy and prompt passes)
//   option "f2_gm2_wv"  : rows per wave (4 or 8), "f2_gm2_mb": the size rule of its rows per block (default f2_min_blocks)
//   option "f2_bfm"     : 0 = bfloat16 dense jobs of passes run the general kernels without the wrapper of the second set
//   option "f2_gm2_nh"  : the most rows of the pass per block (4 or 8; longer passes: groups of that many rows)
//   option "f2_gm2_occ" : bits 1 = nh 2, 2 = nh 4, 4 = nh 8: blocks of up to 288 threads compiled for 1 block per SM
//   option "f2_gm2_nh8" : passes from this many rows take 8 rows per block (shorter ones: 4; default 5)
int f2_gm2 = 1, f2_gm2_min = 2, f2_gm2_max = 128, f2_gm2_wv = 4, f2_gm2_mb = 300, f2_bfm = 1, f2_gm2_nh = 8, f2_gm2_occ = 0;
int f2_gm2_nh8 = 5;
//   option "f2_gm2_pg"  : 0 = one block per (chunk, group); n > 0 (default 1) = persistent grid (resident blocks times n) for
//                         passes of f2_gm2_pgmin (default 3) rows or more, chunks of f2_gm2_pnb rows (default 16); measured
//                         (a4, converted): R = 5 .. 8 -4.7 .. -6.7 % against the first set with it, -1.8 .. -3.4 % without
int f2_gm2_pg = 1, f2_gm2_pnb = 16, f2_gm2_pgmin = 3;
int f2_gm2_mid = 1;  // option "f2_gm2_mid": 1 = fgemvm2 also for passes of 5 .. f2_max_rows rows (comment-only edits after the final build)
// Packed dense jobs of a pass of R > 1 rows through fgemvm; false: the general kernels do it.
bool f2_dense_multi(GemvArgs& a, int cat, int R) {
  if (!f2_active || !f2_gemvm) return false;
  // f2_gm2_mid 0 = passes of 5 .. f2_max_rows rows keep the earlier fgemvm (default 1: the persistent
  // fgemvm2 is faster there, a4: R = 5 .. 16 -4.7 .. -9.8 % against -4.3 .. -9.1 % with the earlier fgemvm)
  const bool mid = R > 4 && R >= f2_min_rows && R <= f2_max_rows;
  const bool gm2 = f2_gm2 && R >= f2_gm2_min && R <= f2_gm2_max && (f2_gm2_mid || !mid);
  if (!gm2 && (R < f2_min_rows || R > f2_max_rows)) return false;
  const int cols = a.job[0].cols;
  bool fp4 = true;
  for (int j = 0; j < a.njobs; ++j) {
    const GemvJob& J = a.job[j];
    if (J.cols != cols || (J.fmt != INK2_DFMT_NVFP4 && J.fmt != INK2_DFMT_MXFP8) || J.split != cols) return false;
    fp4 = fp4 && J.fmt == INK2_DFMT_NVFP4;
  }
  const int mb = be->f2_gemvm_block();
  if (cols % 256 || cols / 8 > 384 || mb < cols / 8) return false;
  auto nblocks = [&](int n) {
    int b = 0;
    for (int j = 0; j < a.njobs; ++j) b += (a.job[j].rows + n - 1) / n;
    return b;
  };
  int nb = f2_nb;
  const bool pers = gm2 && f2_gm2_pg > 0 && R >= f2_gm2_pgmin;  // persistent grid, chunks of f2_gm2_pnb rows
  if (nb == 0 && pers) nb = f2_gm2_pnb;
  if (nb == 0) {
    nb = 32;
    const int minb = gm2 ? f2_gm2_mb : f2_min_blocks;
    while (nb > 8 && nblocks(nb) < minb) nb -= 8;
  }
  if (gm2) {
    F2GemvM2 m{};
    m.g = a;
    m.g.nrhs = R;
    m.nb = nb;
    m.wv = f2_gm2_wv;
    m.nh = R <= 2 ? 2 : (R <= 4 || f2_gm2_nh == 4 || R < f2_gm2_nh8) ? 4 : 8;
    m.fp4 = fp4 ? 1 : 0;
    m.occ1 = (f2_gm2_occ & (m.nh == 2 ? 1 : m.nh == 4 ? 2 : 4)) ? 1 : 0;
    m.pgrid = pers ? f2_gm2_pg : 0;
    m.blocks = nblocks(nb);
    m.g.blocks = m.blocks;
    run_f2k(INK2_F2_GEMVM2, &m, m.blocks * ((R + m.nh - 1) / m.nh), cols / 8, cat);
    return true;
  }
  F2GemvM m{};
  m.g = a;
  m.g.nrhs = R;
  m.nb = nb;
  m.wv = f2_wv;
  m.blocks = nblocks(nb);
  m.g.blocks = m.blocks;
  run_f2k(INK2_F2_GEMVM, &m, m.blocks * ((R + 7) / 8), cols / 8, cat);
  return true;
}
// Router product (bfloat16 rows) and top-k of R rows in one launch; false: the general two launches.
int f2_router_rows = 1;   // option "f2_router_rows": fused router for passes up to this many rows (R > 1 measured slower)
bool f2_grouped = false;  // the router launch of this visit also grouped the pairs (experts_fast skips egroup)
int f2_spec = 0;  // option "f2_spec": the fused router (one row) asks L2 for the previous step's experts (gate/up)
const Layer* f2_layer = nullptr;
bool f2_router_topk(const GemvArgs& g0, const RouterArgs& r, int R, bool group) {
  f2_grouped = false;
  if (!f2_active || !f2_router || R > f2_router_rows || c.experts > 256) return false;
  if (any_packed(g0) || g0.njobs != 1) return false;
  F2RouterArgs f{};
  f.g = g0;
  f.g.total_rows = g0.job[0].rows;
  f.r = r;
  f.done = f2_done;
  int grid = f.g.total_rows;
  if (R > 1) {
    f.g.nrhs = R;
    f.g.blocks = f.g.total_rows;
    grid = f.g.total_rows * ((R + 7) / 8);
    if (group && c.experts <= 256 && R * c.top_k <= 2048) {
      f.gr.idx = r.idx;
      f.gr.gcount = gcount;
      f.gr.gexp = gexp;
      f.gr.goff = goff;
      f.gr.gpairs = gpairs;
      f.gr.R = R;
      f.gr.k = c.top_k;
      f.gr.E = c.experts;
      f2_grouped = true;
    }
  }
  if (R == 1 && f2_spec && group && f2_layer && f2_layer->fmt13 == INK2_FMT_NVFP4) {
    f.pw = f2_layer->w13;
    f.ps = f2_layer->w13s;
    f.prows = 2 * c.expert_inner;
    f.pcols = c.hidden;
  }
  run_f2k(INK2_F2_ROUTER, &f, grid, 256, INK2_PROF_CONV_ROUTER);
  return true;
}
