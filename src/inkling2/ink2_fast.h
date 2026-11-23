// SPDX-License-Identifier: Apache-2.0
// Fast path of the CUDA backend (RTX 30 = sm_86, RTX 40 = sm_89, RTX 50 = sm_120): kernels that are allowed to round
// differently from the general kernels (fused multiply-add, other summation order), so they have no CPU emulation.
// Rule of the fast path: a pass of R rows computes every row with exactly the numbers of a single step (R = 1) of the
// same path, so greedy decoding with the drafter stays identical to plain greedy decoding of the same path.
//
// Expert kernels (NVFP4 experts only; a layer with other expert formats keeps the general kernels):
//   fx13 : gate/up products and the activation, any R. Item = (expert, 8 row pairs), one block of H / 16 threads
//          (thread t owns the 16-column unit t, x in registers), the 16 code words of the item loaded before any sum.
//          Every (row, slot) pair that chose the expert is handled by the same block, so each chosen expert is read
//          once per pass. Row sum of a pair: units in thread order -> xor butterfly over the warp -> warps in order.
//   fx2  : down products, the sum over the k slots, the short convolution and the residual, any R. One warp per
//          output row h (8 rows per block); the warp walks over all chosen experts of the pass (8 at a time, all
//          loads first). Row sum of a pair: 3 units per lane in order -> xor butterfly. The combine of the k slots,
//          the convolution and the residual follow in the same kernel (no egroup / ecomb launch for one row).
// R = 1 uses the router choices directly (idx); R > 1 the groups of the general egroup kernel. The per-pair numbers
// do not depend on which other pairs share the expert or the block, so R = 1 and R > 1 give the same bits.
#pragma once
#include "ink2_kargs.h"

// INK2_FX13: 8 row pairs per item (nchunk = rows / 16); INK2_FX13_P4: 4 row pairs (nchunk = rows / 8).
// INK2_FX2: loads of 8 experts at a time; INK2_FX2_E4: 4 at a time. The variants give the same numbers.
// INK2_FBAR (BarArgs): bar with two block reductions (squares and query dots in one, output squares in the other).
// INK2_FPREP (PrepArgs): prep with all loads of a lane first (head size <= 128).
// Column-owner forms (fxcol): INK2_FX13L / INK2_FX13L4 = INK2_FX13 / _P4 with the codes decoded through a table in
// shared memory; INK2_FX2C (8 row pairs = 16 rows per item), INK2_FX2CL (with the table), INK2_FX2CL4 (8 rows) = the
// down products per pair [R * k, H], then INK2_FXCOMB sums the slots and applies the convolution and the residual.
enum {
  INK2_FX13 = 0, INK2_FX2 = 1, INK2_FX13_P4 = 2, INK2_FX2_E4 = 3, INK2_FBAR = 4, INK2_FPREP = 5,
  INK2_FX13L = 6, INK2_FX13L4 = 7, INK2_FX2C = 8, INK2_FX2CL = 9, INK2_FX2CL4 = 10, INK2_FXCOMB = 11,
  INK2_FBARC = 12,  // second set: F2BarComb (ink2_fast2.h), fbar that first does the fxcomb step of job 0
  INK2_FX2CL4C = 13,
  INK2_FXCOMBP = 14,  // second set: F2CombPf, fxcomb + blocks that prefetch the projection weights of the next visit
  // (INK2_FX2CL4C, second set: F2ColComb (ink2_fast2.h), FX2CL4 that also does the fxcomb step (last block of a chunk))
  // INK2_FX13L4 / INK2_FX2CL4 for passes (R > 1) with the code words of every matrix row
  // decoded once for 2 or 4 pairs of the expert (fxcolg, the same numbers per pair)
  INK2_FX13G2 = 15, INK2_FX13G4 = 16, INK2_FX2CG2 = 17, INK2_FX2CG4 = 18,
  INK2_FX13G2N = 19, INK2_FX2CG2N = 20,  // the same without the early inputs of the first pairs (option "fx_peel" 0)
  INK2_FX_COUNT
};

struct FxArgs {
  const u8* w;          // [E, rows, cols / 2] NVFP4 codes
  const u8* sc;         // [E, rows, cols / 16] E4M3 block scales
  const float* factor;  // [E] amax / denominator / 2 (codes decoded as twice the E2M1 value)
  const int* idx;       // one row: [k] expert of every slot (pair = slot); NULL: the groups below
  const int* gcount;    // grouped: distinct experts, their ids, first pair, pairs (general egroup)
  const int* gexp;
  const int* goff;
  const int* gpairs;
  const float* x;    // fx13: [R, cols] inputs; fx2: [R * k, cols] activations
  float* y;          // fx13: [R * k, rows / 2] activations; fx2: [R, rows] sums of the k products
  const float* wt;   // fx2: [R, k] router weights
  const u16* cw;     // fx2: [rows, K] convolution taps (bfloat16) or NULL
  float* chist;      // fx2: [hr, rows] history ring
  const float* cres; // fx2: [R, rows] residual
  float* cout;       // fx2: [R, rows] output
  const StepState* st;
  int R, k, rows, cols, K, hr;
  int nchunk;  // fx13: items per expert (row pairs / 8, or / 4 for INK2_FX13_P4)
  int hb;      // fx2: output rows per block (warps), 1 .. 8
  float gate_cap, up_cap, cdiv;
};

// Host side (ink2_fast.cu). ink2_fast_probe: compute capability (major * 10 + minor) of device 0 when the fast kernels
// have code for it, else 0. ink2_fast_run: 0 on success, else a CUDA error code.
int ink2_fast_probe(void);
int ink2_fast_run(int fid, const void* args, int grid, int block, int smem, void* stream, bool pdl = false);  // args: FxArgs, BarArgs or PrepArgs
int ink2_fast_smem_max(void);
struct F2Opts;
void ink2_fast_set_f2(const F2Opts* o);  // second set (ink2_fast2.h)
int ink2_fast_ptx_version(void);  // virtual architecture the fast kernels that run were compiled for (86, 89, 120, ...)  // largest dynamic shared memory of fx2 (bytes)
