// SPDX-License-Identifier: Apache-2.0
// Fast kernels, second set: kernel boundaries and passes of few rows on RTX 50 / 40 / 30.
// Shared by the engine (ink2_fast2_engine.inc) and the CUDA backend (ink2_fast2_gen.inc, compiled inside ink2_cuda.cu
// with the flags of the general kernels, so that it can wrap their bodies without changing a single rounding).
//
// 1. Weights before the wait (programmatic dependent launch, sm_90 and later): a weight-reading kernel is launched
//    while the kernel before it still runs; before griddepcontrol.wait it asks the L2 cache for its own weight bytes
//    (they never change), after the wait it reads the activations and runs the unchanged body. Small kernels that come
//    right before a weight kernel trigger their dependents before their own wait, so the weight kernel is already
//    resident one kernel earlier. The arithmetic of every kernel stays exactly as it was.
// 2. fgemvm: packed dense matrices for a pass of R <= 8 rows per group, each matrix row read once for all rows of the
//    group, many matrix rows per block (the inputs of the group stay in registers), the sums of pgemvc exactly.
// 3. Fewer launches: the router product and the top-k choice in one kernel (the last block chooses).
#pragma once
#include "ink2_kargs.h"
#include "ink2_fast.h"

// Launch details of a wrapped kernel (passed by value next to the kernel's own argument block).
struct F2Launch {
  int pf;        // 0 = no prefetch, 1 = prefetch.global.L2 per 128-byte line, 2 = cp.async.bulk.prefetch.L2 (sm_90+)
  int extra;     // > 0: block b < extra also prefetches the regions of blocks b + extra, b + 2 extra, ... (rounds)
  int rounds;    // how many such further regions (0 = only its own)
  int trig;      // 1 = trigger the dependents before the wait (else after it)
};

// Router product + top-k in one launch (R rows): the general gemvw body for every (row, expert) block, the last block
// to finish runs the general topk arithmetic for every row.
struct F2RouterArgs {
  GemvArgs g;      // the router product (bfloat16 rows, one block per (row of the pass, expert))
  RouterArgs r;    // the top-k of every row
  unsigned* done;  // device counter, 0 between launches (the last block sets it back)
  GroupArgs gr;    // R > 1 and gr.idx != NULL: the last block also groups the pairs by expert (egroup)
  // pw != NULL (one row): before the wait the blocks ask L2 for the gate/up rows of the experts the previous step chose
  // in this visit (r.idx still holds them; the last block overwrites it only after every block passed this point)
  const u8* pw;    // NVFP4 codes [E, prows, pcols / 2]
  const u8* ps;    // E4M3 scales [E, prows, pcols / 16]
  int prows, pcols;
};

// fgemvm: GemvArgs (njobs packed jobs, NVFP4 or MXFP8, all the same cols) and the shape of the launch.
struct F2GemvM {
  GemvArgs g;
  int nb;      // matrix rows per block (8, 16, 24 or 32)
  int blocks;  // blocks per group of 8 rows of the pass
  int wv;      // matrix rows per wave of loads (4 or 8)
};
// INK2_F2_GEMVM2 (fgemvm2): the same fields and
struct F2GemvM2 {
  GemvArgs g;
  int nb, blocks, wv;
  int nh;   // rows of the pass per block (2, 4 or 8)
  int fp4;  // 1 = every job NVFP4 (32-bit code words)
  int occ1; // 1 = blocks of up to 288 threads compiled for 1 block per SM (no register cap of 2 blocks)
  int pgrid; // > 0: persistent grid of the resident blocks times this (items walked with the stride of the grid)
};

// Attention at long context (INK2_F2_ATTN): the general attn kernel (one block of 1024 threads per
// (row, head)) spread over more blocks with the same bits. Two launches (ink2_fast2_gen.inc): kf2_attn_score (the logits,
// grid rows * nkv * sa, 128 threads) and kf2_attn_value (the value stripes, grid rows * nkv * 32 / sw, sw warps; the last
// block of a (row, kv head) adds the exp sums and the stripes in the order of the general kernel). hd = 128, nh / nkv <= 4.
struct F2AttnArgs {
  AttnArgs a;
  float* pmax;    // [rows * nh * sa] the largest logit of every score block
  float* part;    // [rows * nh * 32 * hd] the sums of the 32 stripes
  float* es;      // [rows * nh * 1024] the exp sums of the 1024 threads of the general kernel
  unsigned* cnt;  // [rows * nkv] zeroed counters (the last value block of a (row, kv head) sets its counter back to 0)
  int sa;         // score blocks per (row, kv head), 1 .. 32
  int sw;         // stripes (warps) per value block: 1, 2, 4 or 8
  int pf;         // before the wait: bit 1 the score blocks ask L2 for their key rows, bit 2 for their value rows,
                  // bit 4 the value blocks for their value rows
  int ng;         // score blocks: groups of 8 keys a warp loads at once (1 or 2)
};

enum {
  INK2_F2_ROUTER = 0,  // F2RouterArgs, grid = 128 * R (rows of the router), block 256
  INK2_F2_GEMVM = 1,   // F2GemvM
  INK2_F2_ATTN = 2,    // F2AttnArgs (grid and block from the arguments: two launches)
  INK2_F2_GEMVM2 = 3,  // F2GemvM2, fgemvm2 (two waves of code words in flight, nh rows per block)
  INK2_F2_COUNT
};

// fbar that first does the fxcomb step of the visit before (INK2_FBARC): the bar of job 0 takes cout of its row as
// its last state. Saves the fxcomb launch of every visit whose next kernel is such a bar.
struct F2BarComb {
  BarArgs b;
  FxArgs c;
};

// fx2c with the fxcomb step inside (INK2_FX2CL4C): a = the fx2c arguments, c = the fxcomb arguments, cnt = one zeroed
// counter per chunk of matrix rows (the last block of a chunk does the step and sets it back to 0).
struct F2ColComb {
  FxArgs a;
  FxArgs c;
  unsigned* cnt;
};

// fxcomb with extra blocks that only ask L2 for byte ranges (the projection weights of the next visit) and exit at
// once (INK2_FXCOMBP; the first nwork blocks are the fxcomb of a). DRAM is idle while fxcomb and the next bar run.
struct F2Spans {
  const void* p[12];
  unsigned long long n[12];
  int cnt;
  int nwork;
  int evict_last;
};
struct F2CombPf {
  FxArgs a;
  F2Spans s;
};

// Options of the second set (engine -> backend).
struct F2Opts {
  int pf;          // prefetch mode of the weight kernels (0 off, 1 lines, 2 bulk)
  int pf_extra;    // bits: 1 dense, 2 expert w2 (fx2c), 4 expert w13 (fx13, after its wait): the first wave of blocks
                   // also prefetches the regions of the later waves (up to pf_budget bytes)
  long long pf_budget;
  int trig_small;  // 1: attn, norm, gmix (and the fast bar / prep through FxArgs) trigger before their wait
  int gemvm;       // 1: fgemvm for packed passes of R > 1 rows
  int prep_minb;   // 2: fprep compiled for 2 blocks per SM (fewer registers), else 1
  int pipe;        // bits 1 fx13, 2 fx2c: the PIPE form of fxcol (resident grid, next item loaded early)
  int pipe_waves;  // its grid: resident blocks times this
  int reg;         // pf 3: where the code words go to registers (bits 1 dense, 2 fgemvm, 4 router, 8 fx2c)
  int bfm;         // 0 = bfloat16 dense jobs of passes (R > 1) run the general kernels without the wrapper
};
