// SPDX-License-Identifier: Apache-2.0
// Fast kernels (ink2_fast.h): members and methods of struct Engine, included inside its body.
// The general kernels stay the reference (they match the CPU emulation bit for bit); the fast kernels are chosen at
// load time when the device has code for them (RTX 30 / 40 / 50: sm_86, sm_89, sm_120 in the library) and nothing
// forces the general path:
//   option "fast" (ink2_set_option): -1 or 0 = general kernels (default -1), 1 = fast kernels where available.
// The fast kernels keep the greedy ids of the checks but not every bit of the logits (other sum orders).
// What the fast path changes today: the expert products of every layer with NVFP4 experts (fx13, fx2), for one row
// and for passes of any R. Everything else runs the general kernels, which already give a pass the bits of the
// single steps.
int fast_opt = -1;
int fast = 0;       // after finalize: 0 = general kernels, else the compute capability the fast kernels run on
int fx13_p = 4;     // option "fx13_p": row pairs per item of fx13 (8 or 4; 4 measured faster, same numbers)
int fx2_e = 4;      // option "fx2_e": experts per round of loads of fx2 (8 or 4)
int fx_lut = 1;     // option "fx_lut": 1 = E2M1 codes decoded through a table in shared memory (same values)
int fx2_mode = 1;   // option "fx2_mode": 1 = column-owner down products + fxcomb, 0 = one warp per output row (fx2)
int fx2c_p = 4;     // option "fx2c_p": rows per item of the column-owner down products / 2 (8 or 4; 4 measured faster)
int fx2_hb = 8;     // option "fx2_hb": output rows (warps) per block of fx2 (1 .. 8)
int pdl_opt = 1;    // option "pdl": 1 = programmatic dependent launch of every kernel on the fast path (sm_90 and later)
int pdl_on = 0;     // after fast_resolve: the backend launches with it
int fsmall = 1;     // option "fast_small": 1 = also the fast bar and prep (fbar, fprep), 0 = experts only (2 = fbar only, 3 = fprep only)
// the expert kernels of passes (R > 1) decode every code word once for fx_g pairs
// (passes of up to fx_gp_rows - 1 rows) or fx_gp pairs (longer passes: prompts); 1 = the first set (fxcol for every R)
int fx_g = 2, fx_gp = 2, fx_gp_rows = 32;
int fx_peel = 1;  // option "fx_peel": 1 = the inputs of the first pairs loaded with the code words (G 2)

// Decides the path from the option and the device. Called at the end of finalize and when the
// option changes (the recorded graphs are dropped by ink2_set_option).
void fast_resolve() {
  fast = 0;
  pdl_on = be->set_pdl(false) ? 1 : 0;
  if (!finalized || fast_opt <= 0) return;
  const int cc = be->fast_level();
  if (cc < 86) return;
  const int H = c.hidden, I = c.expert_inner, k = c.top_k;
  // shapes the fast kernels handle: fx13 one thread per 16 columns (<= 256), fx2 two or three units per lane
  if (H % 16 || H / 16 > 256 || I % 16 || I / 16 <= 64 || I / 16 > 96 || I % 8 || k > 32) return;
  bool any = false;
  for (const Layer& L : layers) any = any || fast_layer_ok(L);
  if (!any) return;
  // shared memory of fx2: (R * k + R) * hb floats for the largest pass
  int hb = fx2_hb;
  while (hb > 1 && (long long)(max_rows * k + max_rows) * hb * 4 > be_fast_smem()) hb >>= 1;
  if ((long long)(max_rows * k + max_rows) * hb * 4 > be_fast_smem()) return;
  fx2_hb_used = hb;
  fast = cc;
  pdl_on = be->set_pdl(pdl_opt != 0) ? 1 : 0;
}
int fx2_hb_used = 8;
int be_fast_smem() { return be->fast_smem(); }
bool fast_layer_ok(const Layer& L) const {
  return L.fmt13 == INK2_FMT_NVFP4 && L.fmt2 == INK2_FMT_NVFP4 && L.w13 && L.w13s && L.w2 && L.w2s && L.f13 && L.f2;
}
bool fast_experts(const Layer& L) const { return fast && fast_layer_ok(L); }

bool fast_bar(const BarArgs& a) const {
  return fast && (fsmall == 1 || fsmall == 2) && a.H % 4 == 0 && a.H / 4 <= 576 /* second set: fbar block bound */ && a.job[0].ns <= 6 && a.job[0].nq <= 4 &&
         (a.nj < 2 || (a.job[1].ns <= 6 && a.job[1].nq <= 4));
}
// fprep writes the FP32 cache only (the MXFP8 visits keep the general prep)
bool fast_prep(const PrepArgs& a) const { return fast && (fsmall == 1 || fsmall == 3) && a.kv.mode == INK2_KV_FP32 && a.hd <= 128 && a.K <= 8 && a.drel <= 128; }
void run_fx(int fid, const void* a, int grid, int block, int smem, int cat) {
  f2_flush();  // second set
  if (bench_reps > 0) {  // the per-kernel bench times the general kernels only; the fast ones run once
    be->run_fast(fid, a, grid, block, smem);
    return;
  }
  be->run_fast(fid, a, grid, block, smem);
  ++n_launch;
  if (prof) {
    be->sync();
    double t = now_s();
    prof_t[cat] += t - prof_mark;
    prof_mark = t;
  }
}

// The experts of a visit on the fast path, R rows (R = 1: the router choices directly, no grouping kernel).
void experts_fast(Visit& V, const Layer& L, int R) {
  const int H = c.hidden, I = c.expert_inner, k = c.top_k;
  int slots = k;
  if (R > 1) {
    GroupArgs g{};
    g.idx = V.ridx;
    g.gcount = gcount;
    g.gexp = gexp;
    g.goff = goff;
    g.gpairs = gpairs;
    g.R = R;
    g.k = k;
    g.E = c.experts;
    int nb = 32;
    while (nb < c.experts) nb <<= 1;
    if (!f2_grouped) run(KID_egroup, &g, 1, nb, INK2_PROF_EXPERTS);  // second set: else done by the router launch
    f2_grouped = false;
    // blocks for about 1.3 times the distinct experts expected (as experts_multi); more items loop inside a block
    slots = R * k < c.experts ? R * k : c.experts;
    int expect = (int)ceil(1.3 * (k + 0.38 * k * (R - 1)));
    if (expect < slots) slots = expect;
  }
  FxArgs f{};
  f.w = L.w13;
  f.sc = L.w13s;
  f.factor = L.f13;
  f.idx = R == 1 ? V.ridx : nullptr;
  f.gcount = gcount;
  f.gexp = gexp;
  f.goff = goff;
  f.gpairs = gpairs;
  f.x = xmn;
  f.y = act;
  f.R = R;
  f.k = k;
  f.rows = 2 * I;
  f.cols = H;
  f.gate_cap = c.gate_cap;
  f.up_cap = c.up_cap;
  const int p = fx13_p == 4 ? 4 : 8;
  f.nchunk = I / p;  // 2 I rows, 2 p per item
  int k13 = p == 4 ? (fx_lut ? INK2_FX13L4 : INK2_FX13_P4) : (fx_lut ? INK2_FX13L : INK2_FX13);
  const int g = R < 2 ? 1 : R < fx_gp_rows ? fx_g : fx_gp;  // pairs per decode (R = 1: fxcol always)
  if (g > 1 && k13 == INK2_FX13L4) k13 = g == 4 ? INK2_FX13G4 : fx_peel ? INK2_FX13G2 : INK2_FX13G2N;
  run_fx(k13, &f, slots * f.nchunk, (H / 16 + 31) / 32 * 32, 0, INK2_PROF_EXPERTS);
  f.w = L.w2;
  f.sc = L.w2s;
  f.factor = L.f2;
  f.x = act;
  f.y = ey;
  f.wt = V.rwt;
  f.cw = L.mlp_sconv;
  f.chist = V.mhist;
  f.cres = resid;
  f.cout = V.hid;
  f.st = st;
  f.rows = H;
  f.cols = I;
  f.K = c.conv_k;
  f.hr = hr;
  f.cdiv = (float)sqrt((double)(c.conv_k + 1));
  f.hb = fx2_hb_used;
  if (fx2_mode == 1 && H % 16 == 0) {  // down product of every pair [R * k, H] (ydots), then the slots, conv, residual
    f.y = ydots;
    const int p2 = fx2c_p == 4 ? 4 : 8;
    f.nchunk = H / (2 * p2);
    int k2 = p2 == 4 ? INK2_FX2CL4 : (fx_lut ? INK2_FX2CL : INK2_FX2C);
    if (k2 == INK2_FX2CL4 && f2_col_comb(f, slots * f.nchunk, (I / 16 + 31) / 32 * 32)) return;  // second set
    if (g > 1 && k2 == INK2_FX2CL4) k2 = g == 4 ? INK2_FX2CG4 : fx_peel ? INK2_FX2CG2 : INK2_FX2CG2N;
    run_fx(k2, &f, slots * f.nchunk, (I / 16 + 31) / 32 * 32, 0, INK2_PROF_EXPERTS);
    f.x = ydots;
    f.y = ey;
    if (f2_defer_comb(f, (R * H + 255) / 256)) return;  // second set: the next bar does it
    if (f2_comb_pf(V, f, (R * H + 255) / 256)) return;  // second set: + prefetch of the next visit's projections
    run_fx(INK2_FXCOMB, &f, (R * H + 255) / 256, 256, 0, INK2_PROF_EXPERTS);
    return;
  }
  run_fx(fx2_e == 8 ? INK2_FX2 : INK2_FX2_E4, &f, (H + f.hb - 1) / f.hb, 32 * f.hb, (R * k + R) * f.hb * 4, INK2_PROF_EXPERTS);
}
