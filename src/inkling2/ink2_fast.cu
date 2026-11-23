// SPDX-License-Identifier: Apache-2.0
// Fast path kernels of the CUDA backend (see ink2_fast.h). Compiled with the default -fmad=true; every sum that
// matters is written with explicit fmaf / __fadd_rn / __fmul_rn, so the compiler cannot contract it differently in
// different places: the one-row form and the grouped form run the same instructions on every pair.
#include <cuda_runtime.h>
#include <math.h>

#include "ink2_fast.h"
#include "ink2_fast2.h"  // second set

#define FX_FULL 0xffffffffu
#define FX13_G 8  // pairs of the pass per round of the shared sums

namespace {

__device__ __forceinline__ float fx_fp4(unsigned c) {  // E2M1 code in bits 0-3 -> twice its value (exact)
  return __fmul_rn(__uint_as_float(((c & 7u) << 22) | ((c & 8u) << 28)), __uint_as_float(0x7F000000u));
}
__device__ __forceinline__ float fx_e4m3(unsigned b) {  // E4M3 byte -> its value (exact)
  return __fmul_rn(__uint_as_float(((b & 0x7Fu) << 20) | ((b & 0x80u) << 24)), __uint_as_float(0x7B800000u));
}
__device__ __forceinline__ float fx_bf(unsigned b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ float fx_softcap(float x, float cap) { return tanhf(x / cap) * cap; }
__device__ __forceinline__ float fx_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ int fx_slot(int p, int hr) { return ((p % hr) + hr) % hr; }

// 16 elements of one unit (element i in bits 4i .. 4i+3 of c) times x: even and odd elements in two chains, added.
__device__ __forceinline__ float fx_dot16(unsigned long long c, float4 a, float4 b, float4 d, float4 e) {
  unsigned lo = (unsigned)c, hi = (unsigned)(c >> 32);
  float s0 = __fmul_rn(fx_fp4(lo), a.x);
  float s1 = __fmul_rn(fx_fp4(lo >> 4), a.y);
  s0 = fmaf(fx_fp4(lo >> 8), a.z, s0);
  s1 = fmaf(fx_fp4(lo >> 12), a.w, s1);
  s0 = fmaf(fx_fp4(lo >> 16), b.x, s0);
  s1 = fmaf(fx_fp4(lo >> 20), b.y, s1);
  s0 = fmaf(fx_fp4(lo >> 24), b.z, s0);
  s1 = fmaf(fx_fp4(lo >> 28), b.w, s1);
  s0 = fmaf(fx_fp4(hi), d.x, s0);
  s1 = fmaf(fx_fp4(hi >> 4), d.y, s1);
  s0 = fmaf(fx_fp4(hi >> 8), d.z, s0);
  s1 = fmaf(fx_fp4(hi >> 12), d.w, s1);
  s0 = fmaf(fx_fp4(hi >> 16), e.x, s0);
  s1 = fmaf(fx_fp4(hi >> 20), e.y, s1);
  s0 = fmaf(fx_fp4(hi >> 24), e.z, s0);
  s1 = fmaf(fx_fp4(hi >> 28), e.w, s1);
  return __fadd_rn(s0, s1);
}

// Sum over the warp by the xor butterfly (offsets 16, 8, 4, 2, 1), returned to every lane.
__device__ __forceinline__ float fx_xsum(float v) {
  v = __fadd_rn(v, __shfl_xor_sync(FX_FULL, v, 16));
  v = __fadd_rn(v, __shfl_xor_sync(FX_FULL, v, 8));
  v = __fadd_rn(v, __shfl_xor_sync(FX_FULL, v, 4));
  v = __fadd_rn(v, __shfl_xor_sync(FX_FULL, v, 2));
  v = __fadd_rn(v, __shfl_xor_sync(FX_FULL, v, 1));
  return v;
}
// The same butterfly for 16 values at once (16 shuffles instead of 80): every value gets exactly the sums of
// fx_xsum (a + b == b + a bit for bit). Lane l returns the sum of value (l >> 1) & 15.
// selp in PTX: keeps both values in registers (a C select of array elements can turn the array into local memory)
__device__ __forceinline__ float fx_sel(int p, float a, float b) {
  float r;
  asm("{ .reg .pred q; setp.ne.s32 q, %3, 0; selp.f32 %0, %1, %2, q; }" : "=f"(r) : "f"(a), "f"(b), "r"(p));
  return r;
}
template <int N>
__device__ __forceinline__ void fx_half(float* v, int lane, int off) {
  const int hi = lane & off;
#pragma unroll
  for (int j = 0; j < N / 2; ++j) {
    float send = fx_sel(hi, v[j], v[j + N / 2]);
    float keep = fx_sel(hi, v[j + N / 2], v[j]);
    v[j] = __fadd_rn(keep, __shfl_xor_sync(FX_FULL, send, off));
  }
}
// N values (N = 2, 4, 8 or 16): halving levels at offsets 16, 8, ... until one value is left, then plain xor adds.
// Lane l returns the sum of value (l >> (5 - log2 N)) & (N - 1).
template <int N>
__device__ __forceinline__ float fx_tsum(float* v, int lane) {
  if (N >= 16) fx_half<16>(v, lane, 16);
  if (N >= 8) fx_half<(N >= 16 ? 8 : N)>(v + 0, lane, N >= 16 ? 8 : 16);
  if (N >= 4) fx_half<(N >= 8 ? 4 : N)>(v, lane, N >= 16 ? 4 : N >= 8 ? 8 : 16);
  fx_half<2>(v, lane, N >= 16 ? 2 : N >= 8 ? 4 : N >= 4 ? 8 : 16);
  float r = v[0];
  for (int off = (N >= 16 ? 1 : N >= 8 ? 2 : N >= 4 ? 4 : 8); off >= 1; off >>= 1) r = __fadd_rn(r, __shfl_xor_sync(FX_FULL, r, off));
  return r;
}
template <int N>
__device__ __forceinline__ int fx_tsum_index(int lane) {
  return (lane >> (N >= 16 ? 1 : N >= 8 ? 2 : N >= 4 ? 3 : 4)) & (N - 1);
}

// programmatic dependent launch (sm_90 and later): wait for the kernel before, then let the next one be scheduled
__device__ __forceinline__ void fx_pdl() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}
// second set (ink2_fast2.h): the trigger before or after the wait, prefetch into L2 before it
__device__ __forceinline__ void fx2_trig() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}
__device__ __forceinline__ void fx2_wait(const F2Launch& l) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  if (!l.trig) fx2_trig();
}
__device__ __forceinline__ void fx2_span(const void* p, size_t n) {  // [p, p + n) into L2, by the threads of the block
  const size_t a0 = (size_t)p & ~(size_t)127, a1 = (size_t)p + n;
  for (size_t o = a0 + (size_t)threadIdx.x * 128; o < a1; o += (size_t)blockDim.x * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"((const void*)o));
}
__device__ __forceinline__ float4 fx_ld4(const float* p) { return __ldg((const float4*)p); }
// The code word as seen inside a loop over pairs: an empty volatile asm, so the decoding is not hoisted out of the loop
// (16 rows x 16 decoded values would not fit in the registers).
__device__ __forceinline__ unsigned long long fx_opaque(unsigned long long c) {
  asm volatile("" : "+l"(c));
  return c;
}

// One (row r, channel h) of fxcomb: the k products in slot order times the router weights, then the short convolution
// and the residual; writes y, the ring and cout, returns cout (second set: also used by fbar, which takes this step over).
template <int CG = 0>  // CG 1: the products read from L2 (written by other blocks of the same launch)
__device__ __forceinline__ float fx_comb1(const FxArgs& a, int r, int h) {
  const int H = a.rows;
  const int K1 = a.K - 1;
  float ov[8];
#pragma unroll
  for (int m = 0; m < 8; ++m) {
    ov[m] = 0.0f;
    const int rr = r - m;
    if (m > K1 || rr < 0 || (m > 0 && !a.cw)) continue;
    float o = 0.0f;
#pragma unroll 8
    for (int j = 0; j < a.k; ++j) {
      const float d = CG ? __ldcg(a.x + ((size_t)rr * a.k + j) * H + h) : a.x[((size_t)rr * a.k + j) * H + h];
      o = fmaf(d, a.wt[(size_t)rr * a.k + j], o);
    }
    ov[m] = o;
  }
  const float x = ov[0];
  a.y[(size_t)r * H + h] = x;
  if (a.cw) {
    const int pos = a.st[r].pos;
    float o = 0.0f;
#pragma unroll
    for (int tt = 0; tt < 7; ++tt) {
      if (tt >= K1) break;
      const int lag = K1 - tt;
      float hv = 0.0f;
#pragma unroll
      for (int m = 1; m < 8; ++m)
        if (m == lag) hv = r >= lag ? ov[m] : a.chist[(size_t)fx_slot(pos - lag, a.hr) * H + h];
      o = fmaf(hv, fx_bf(a.cw[(size_t)h * a.K + tt]), o);
    }
    o = fmaf(x, fx_bf(a.cw[(size_t)h * a.K + K1]), o);
    o = __fdiv_rn(__fadd_rn(o, x), a.cdiv);
    a.chist[(size_t)(pos % a.hr) * H + h] = x;
    const float c = __fadd_rn(a.cres[(size_t)r * H + h], o);
    a.cout[(size_t)r * H + h] = c;
    return c;
  }
  return x;
}
// fxcol: column-owner expert product, any R. Item = (chosen expert, NR = 2 P consecutive rows); block of cols / 16
// threads (whole warps), thread t owns the 16-column unit t (one 8-byte code word and one scale byte per row), all
// loads of the item in flight before the first sum. Every (row, slot) pair that chose the expert is handled by the
// block: x unit of the pair (16 floats) -> NR row partials -> transposed butterfly over the warp -> warps added in
// order. MODE 0 (w13): rows 2i / 2i+1 are gate / up -> activation y[p, row pair]; MODE 1 (w2): y[p, row] = the
// product of the pair (dots, summed over the slots by fxcomb). LUT 1: E2M1 codes decoded through a 16-entry table in
// shared memory (same values as the bit form, so the same bits).
template <int LUT>
__device__ __forceinline__ float fx_dec(unsigned w, int sh, const float* lut) {
  if (LUT) return *(const float*)((const char*)lut + ((sh >= 2 ? (w >> (sh - 2)) : (w << (2 - sh))) & 0x3Cu));
  return fx_fp4(w >> sh);
}
template <int LUT>
__device__ __forceinline__ float fx_dot16t(unsigned long long c, float4 a, float4 b, float4 d, float4 e, const float* lut) {
  unsigned lo = (unsigned)c, hi = (unsigned)(c >> 32);
  float s0 = __fmul_rn(fx_dec<LUT>(lo, 0, lut), a.x);
  float s1 = __fmul_rn(fx_dec<LUT>(lo, 4, lut), a.y);
  s0 = fmaf(fx_dec<LUT>(lo, 8, lut), a.z, s0);
  s1 = fmaf(fx_dec<LUT>(lo, 12, lut), a.w, s1);
  s0 = fmaf(fx_dec<LUT>(lo, 16, lut), b.x, s0);
  s1 = fmaf(fx_dec<LUT>(lo, 20, lut), b.y, s1);
  s0 = fmaf(fx_dec<LUT>(lo, 24, lut), b.z, s0);
  s1 = fmaf(fx_dec<LUT>(lo, 28, lut), b.w, s1);
  s0 = fmaf(fx_dec<LUT>(hi, 0, lut), d.x, s0);
  s1 = fmaf(fx_dec<LUT>(hi, 4, lut), d.y, s1);
  s0 = fmaf(fx_dec<LUT>(hi, 8, lut), d.z, s0);
  s1 = fmaf(fx_dec<LUT>(hi, 12, lut), d.w, s1);
  s0 = fmaf(fx_dec<LUT>(hi, 16, lut), e.x, s0);
  s1 = fmaf(fx_dec<LUT>(hi, 20, lut), e.y, s1);
  s0 = fmaf(fx_dec<LUT>(hi, 24, lut), e.z, s0);
  s1 = fmaf(fx_dec<LUT>(hi, 28, lut), e.w, s1);
  return __fadd_rn(s0, s1);
}

// The rows of item it of fxcol into L2 (expert id read from L2: written two launches back, see ink2_fast2_gen.inc).
template <int NR>
__device__ __forceinline__ void fx2_pf_item(const FxArgs& a, int it) {
  const int nexp = a.idx ? a.k : __ldcg(a.gcount);
  if (it >= nexp * a.nchunk) return;
  const int ue = it / a.nchunk;
  const int r0 = (it - ue * a.nchunk) * NR;
  int nr = a.rows - r0;
  if (nr > NR) nr = NR;
  const int id = a.idx ? __ldcg(a.idx + ue) : __ldcg(a.gexp + ue);
  const size_t row0 = (size_t)id * a.rows + (size_t)r0;
  const size_t wbytes = (size_t)(a.cols >> 1), sbytes = (size_t)(a.cols >> 4);
  fx2_span(a.w + row0 * wbytes, (size_t)nr * wbytes);
  fx2_span(a.sc + row0 * sbytes, (size_t)nr * sbytes);
}
// The code words and scales of item it of fxcol for this thread (expert id from L2, see fx2_pf_item); false: no item.
template <int NR>
__device__ __forceinline__ bool fx2_load_item(const FxArgs& a, int it, unsigned long long* c, unsigned* sb) {
  const int nexp = a.idx ? a.k : __ldcg(a.gcount);
  if (it >= nexp * a.nchunk) return false;
  const int ue = it / a.nchunk;
  const int r0 = (it - ue * a.nchunk) * NR;
  int nr = a.rows - r0;
  if (nr > NR) nr = NR;
  const int id = a.idx ? __ldcg(a.idx + ue) : __ldcg(a.gexp + ue);
  const size_t wbytes = (size_t)(a.cols >> 1), sbytes = (size_t)(a.cols >> 4);
  const size_t row0 = (size_t)id * a.rows + (size_t)r0;
  const int u = threadIdx.x;
  const bool act = u < (a.cols >> 4);
  const u8* wp = a.w + row0 * wbytes + (size_t)u * 8;
  const u8* sp = a.sc + row0 * sbytes + (size_t)u;
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    c[r] = 0ull;
    sb[r] = 0u;
    if (act && r < nr) {
      c[r] = __ldg((const unsigned long long*)(wp + (size_t)r * wbytes));
      sb[r] = __ldg(sp + (size_t)r * sbytes);
    }
  }
  return true;
}
// COMB (w2, second set): the block that writes the last product of a chunk of matrix rows (counter cb.cnt per chunk) also
// does the fxcomb step of those rows for every row of the pass (fx_comb1, the same bits); no fxcomb launch.
// PIPE (second set): a grid of resident blocks, each loads the code words of its next item before the sums of this one.
template <int P, int MODE, int LUT, int COMB = 0, int PIPE = 0>
__global__ void __launch_bounds__(256, PIPE ? 1 : 2) fxcol(FxArgs a, F2Launch l, F2ColComb cb) {
  constexpr int NR = 2 * P;
  if (l.trig) fx2_trig();
  unsigned long long c[NR];
  unsigned sb[NR];
  bool have = false;  // pf 3, w2: the code words of the first item already in registers (loaded before the wait)
  if (MODE == 1 && l.pf == 3) have = fx2_load_item<NR>(a, (int)blockIdx.x, c, sb);
  if (MODE == 1 && l.pf && (l.pf != 3 || l.extra > 0)) {  // w2: the experts are known (router two launches back): own rows and later waves first
    if (l.pf != 3) fx2_pf_item<NR>(a, (int)blockIdx.x);
    if (l.extra > 0 && (int)blockIdx.x < l.extra)
      for (int k = 1; k <= l.rounds; ++k) fx2_pf_item<NR>(a, (int)blockIdx.x + k * l.extra);
  }
  fx2_wait(l);
  if (MODE == 0 && l.pf && l.extra > 0 && (int)blockIdx.x < l.extra)  // w13: the later waves, after the router is known
    for (int k = 1; k <= l.rounds; ++k) fx2_pf_item<NR>(a, (int)blockIdx.x + k * l.extra);
  __shared__ float sm[FX13_G][NR][9];  // [pair][row][warp], at most 8 warps
  __shared__ float lut[16];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int nunit = a.cols >> 4;
  const int nchunkrows = MODE == 0 ? (a.rows >> 1) : a.rows;  // activations (row pairs) or rows per expert
  const int u = threadIdx.x;
  const bool act = u < nunit;
  const size_t wbytes = (size_t)(a.cols >> 1), sbytes = (size_t)(a.cols >> 4);
  if (LUT && threadIdx.x < 16) lut[threadIdx.x] = fx_fp4(threadIdx.x);
  if (LUT) __syncthreads();
  const int nexp = a.idx ? a.k : a.gcount[0];
  const int items = nexp * a.nchunk;
#pragma unroll 1
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int ue = it / a.nchunk;
    const int r0 = (it - ue * a.nchunk) * NR;  // first matrix row of the item
    int nr = a.rows - r0;
    if (nr > NR) nr = NR;
    int id, qb, qe;
    if (a.idx) {
      id = a.idx[ue];
      qb = ue;
      qe = ue + 1;
    } else {
      id = a.gexp[ue];
      qb = a.goff[ue];
      qe = a.goff[ue + 1];
    }
    const size_t row0 = (size_t)id * a.rows + (size_t)r0;
    const u8* wp = a.w + row0 * wbytes + (size_t)u * 8;
    const u8* sp = a.sc + row0 * sbytes + (size_t)u;
    const bool pre = (have && it == (int)blockIdx.x) || (PIPE && it != (int)blockIdx.x);
#pragma unroll
    for (int r = 0; r < NR; ++r) {  // every load of the item in flight before the first sum
      if (pre) continue;
      c[r] = 0ull;
      sb[r] = 0u;
      if (act && r < nr) {
        c[r] = __ldg((const unsigned long long*)(wp + (size_t)r * wbytes));
        sb[r] = __ldg(sp + (size_t)r * sbytes);
      }
    }
    float s[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) s[r] = fx_e4m3(sb[r]);
    const float fac = a.factor[id];
    unsigned long long cn[PIPE ? NR : 1];
    unsigned sbn[PIPE ? NR : 1];
    if (PIPE) fx2_load_item<NR>(a, it + (int)gridDim.x, cn, sbn);  // the next item of this block, in flight now
#pragma unroll 1
    for (int q0 = qb; q0 < qe; q0 += FX13_G) {
      int nq = qe - q0;
      if (nq > FX13_G) nq = FX13_G;
#pragma unroll 1
      for (int qq = 0; qq < nq; ++qq) {
        const int p = a.idx ? q0 + qq : a.gpairs[q0 + qq];
        float4 x0 = make_float4(0.f, 0.f, 0.f, 0.f), x1 = x0, x2 = x0, x3 = x0;
        if (act) {
          const size_t xrow = MODE == 0 ? (size_t)(p / a.k) : (size_t)p;  // w13: input row of the pass; w2: the pair
          const float* xx = a.x + xrow * (size_t)a.cols + ((size_t)u << 4);
          x0 = fx_ld4(xx);
          x1 = fx_ld4(xx + 4);
          x2 = fx_ld4(xx + 8);
          x3 = fx_ld4(xx + 12);
        }
        float v[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) v[r] = __fmul_rn(fx_dot16t<LUT>(fx_opaque(c[r]), x0, x1, x2, x3, lut), s[r]);
        float t = fx_tsum<NR>(v, lane);
        if ((lane & (32 / NR - 1)) == 0) sm[qq][fx_tsum_index<NR>(lane)][warp] = t;
      }
      __syncthreads();
      if (MODE == 0) {
        const int np = nr >> 1;
        for (int t = threadIdx.x; t < nq * np; t += blockDim.x) {  // (nq * np can exceed the threads of the block)
          const int qq = t / np, i = t - qq * np;
          float g = sm[qq][2 * i][0], up = sm[qq][2 * i + 1][0];
          for (int w = 1; w < nw; ++w) {
            g = __fadd_rn(g, sm[qq][2 * i][w]);
            up = __fadd_rn(up, sm[qq][2 * i + 1][w]);
          }
          g = __fmul_rn(g, fac);
          up = __fmul_rn(up, fac);
          const int p = a.idx ? q0 + qq : a.gpairs[q0 + qq];
          a.y[(size_t)p * nchunkrows + (r0 >> 1) + i] =
              __fmul_rn(__fmul_rn(fx_softcap(g, a.gate_cap), fx_sigmoid(g)), fx_softcap(up, a.up_cap));
        }
      } else {
        for (int t = threadIdx.x; t < nq * nr; t += blockDim.x) {  // (nq * nr can exceed the threads of the block)
          const int qq = t / nr, i = t - qq * nr;
          float d = sm[qq][i][0];
          for (int w = 1; w < nw; ++w) d = __fadd_rn(d, sm[qq][i][w]);
          const int p = a.idx ? q0 + qq : a.gpairs[q0 + qq];
          a.y[(size_t)p * nchunkrows + r0 + i] = __fmul_rn(d, fac);
        }
      }
      __syncthreads();
    }
    if (PIPE) {
#pragma unroll
      for (int r = 0; r < NR; ++r) c[r] = cn[r], sb[r] = sbn[r];
    }
    if (COMB) {  // all products of this item are written: count the chunk
      __shared__ int last;
      __threadfence();
      __syncthreads();
      const int chunk = it - ue * a.nchunk;
      if (threadIdx.x == 0) last = atomicAdd(cb.cnt + chunk, 1u) == (unsigned)(nexp - 1);
      __syncthreads();
      if (last) {
        __threadfence();
        for (int t = threadIdx.x; t < a.R * nr; t += blockDim.x) {
          const int rr = t / nr, i = t - rr * nr;
          fx_comb1<1>(cb.c, rr, r0 + i);
        }
        if (threadIdx.x == 0) cb.cnt[chunk] = 0u;
      }
    }
  }
}

// fxcolg = fxcol<P, MODE, 1> (codes through the table) for passes (R > 1) with the code
// words of a matrix row decoded once for G pairs of the expert: the inputs of G pairs in registers (all their loads in
// flight together), for every row of the item the 16 values of the unit decoded once, then per pair the products in the
// order of fx_dot16t (two chains from the first product, added), times the block scale. Per (pair, row) these are the
// numbers of fxcol exactly, and the transposed butterfly, the warp order and the epilogue are the same code, so a pass
// keeps the bits of single steps (R = 1 always runs fxcol itself).
__device__ __forceinline__ void fx_dec16(unsigned long long c, const float* lut, float* d) {
  const unsigned lo = (unsigned)c, hi = (unsigned)(c >> 32);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    d[i] = fx_dec<1>(lo, 4 * i, lut);
    d[8 + i] = fx_dec<1>(hi, 4 * i, lut);
  }
}
__device__ __forceinline__ float fx_dot16d(const float* d, float4 a, float4 b, float4 e, float4 f) {
  float s0 = __fmul_rn(d[0], a.x);
  float s1 = __fmul_rn(d[1], a.y);
  s0 = fmaf(d[2], a.z, s0);
  s1 = fmaf(d[3], a.w, s1);
  s0 = fmaf(d[4], b.x, s0);
  s1 = fmaf(d[5], b.y, s1);
  s0 = fmaf(d[6], b.z, s0);
  s1 = fmaf(d[7], b.w, s1);
  s0 = fmaf(d[8], e.x, s0);
  s1 = fmaf(d[9], e.y, s1);
  s0 = fmaf(d[10], e.z, s0);
  s1 = fmaf(d[11], e.w, s1);
  s0 = fmaf(d[12], f.x, s0);
  s1 = fmaf(d[13], f.y, s1);
  s0 = fmaf(d[14], f.z, s0);
  s1 = fmaf(d[15], f.w, s1);
  return __fadd_rn(s0, s1);
}
template <int P, int MODE, int G, int PEEL = 1>
__global__ void __launch_bounds__(256, G >= 4 ? 1 : 2) fxcolg(FxArgs a, F2Launch l) {
  constexpr int NR = 2 * P;
  if (l.trig) fx2_trig();
  unsigned long long c[NR];
  unsigned sb[NR];
  bool have = false;  // pf 3, w2: the code words of the first item already in registers (loaded before the wait)
  if (MODE == 1 && l.pf == 3) have = fx2_load_item<NR>(a, (int)blockIdx.x, c, sb);
  if (MODE == 1 && l.pf && (l.pf != 3 || l.extra > 0)) {
    if (l.pf != 3) fx2_pf_item<NR>(a, (int)blockIdx.x);
    if (l.extra > 0 && (int)blockIdx.x < l.extra)
      for (int k = 1; k <= l.rounds; ++k) fx2_pf_item<NR>(a, (int)blockIdx.x + k * l.extra);
  }
  fx2_wait(l);
  if (MODE == 0 && l.pf && l.extra > 0 && (int)blockIdx.x < l.extra)
    for (int k = 1; k <= l.rounds; ++k) fx2_pf_item<NR>(a, (int)blockIdx.x + k * l.extra);
  __shared__ float sm[FX13_G][NR][9];  // [pair][row][warp], at most 8 warps
  __shared__ float lut[16];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int nunit = a.cols >> 4;
  const int nchunkrows = MODE == 0 ? (a.rows >> 1) : a.rows;
  const int u = threadIdx.x;
  const bool act = u < nunit;
  const size_t wbytes = (size_t)(a.cols >> 1), sbytes = (size_t)(a.cols >> 4);
  if (threadIdx.x < 16) lut[threadIdx.x] = fx_fp4(threadIdx.x);
  __syncthreads();
  const int nexp = a.idx ? a.k : a.gcount[0];
  const int items = nexp * a.nchunk;
#pragma unroll 1
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int ue = it / a.nchunk;
    const int r0 = (it - ue * a.nchunk) * NR;
    int nr = a.rows - r0;
    if (nr > NR) nr = NR;
    int id, qb, qe;
    if (a.idx) {
      id = a.idx[ue];
      qb = ue;
      qe = ue + 1;
    } else {
      id = a.gexp[ue];
      qb = a.goff[ue];
      qe = a.goff[ue + 1];
    }
    const size_t row0 = (size_t)id * a.rows + (size_t)r0;
    const u8* wp = a.w + row0 * wbytes + (size_t)u * 8;
    const u8* sp = a.sc + row0 * sbytes + (size_t)u;
    const bool pre = have && it == (int)blockIdx.x;
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      if (pre) continue;
      c[r] = 0ull;
      sb[r] = 0u;
      if (act && r < nr) {
        c[r] = __ldg((const unsigned long long*)(wp + (size_t)r * wbytes));
        sb[r] = __ldg(sp + (size_t)r * sbytes);
      }
    }
    // the inputs of the first G pairs in flight with the code words, before the scales are decoded (PEEL)
    float4 x[G][4];
#define FXG_LDX(Q0, NG)                                                                  _Pragma("unroll") for (int g = 0; g < G; ++g) {                                          x[g][0] = make_float4(0.f, 0.f, 0.f, 0.f);                                             x[g][1] = x[g][0];                                                                     x[g][2] = x[g][0];                                                                     x[g][3] = x[g][0];                                                                     if (act && g < (NG)) {                                                                   const int p = a.idx ? (Q0) + g : a.gpairs[(Q0) + g];                                   const size_t xrow = MODE == 0 ? (size_t)(p / a.k) : (size_t)p;                         const float* xx = a.x + xrow * (size_t)a.cols + ((size_t)u << 4);                      x[g][0] = fx_ld4(xx);                                                                  x[g][1] = fx_ld4(xx + 4);                                                              x[g][2] = fx_ld4(xx + 8);                                                              x[g][3] = fx_ld4(xx + 12);                                                           }                                                                                    }
    if (PEEL) {
      const int n0 = qe - qb < G ? qe - qb : G;
      FXG_LDX(qb, n0)
    }
    float s[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) s[r] = fx_e4m3(sb[r]);
    const float fac = a.factor[id];
#pragma unroll 1
    for (int q0 = qb; q0 < qe; q0 += FX13_G) {
      int nq = qe - q0;
      if (nq > FX13_G) nq = FX13_G;
#pragma unroll 1
      for (int g0 = 0; g0 < nq; g0 += G) {
        const int ng = nq - g0 < G ? nq - g0 : G;
        if (!PEEL || q0 != qb || g0 != 0) {
          FXG_LDX(q0 + g0, ng)
        }
        float v[G][NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
          float d[16];
          fx_dec16(fx_opaque(c[r]), lut, d);
#pragma unroll
          for (int g = 0; g < G; ++g)
            if (g < ng) v[g][r] = __fmul_rn(fx_dot16d(d, x[g][0], x[g][1], x[g][2], x[g][3]), s[r]);  // (ng: the same in the block)
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
          if (g < ng) {
            float t = fx_tsum<NR>(v[g], lane);
            if ((lane & (32 / NR - 1)) == 0) sm[g0 + g][fx_tsum_index<NR>(lane)][warp] = t;
          }
        }
      }
      __syncthreads();
      if (MODE == 0) {
        const int np = nr >> 1;
        for (int t = threadIdx.x; t < nq * np; t += blockDim.x) {
          const int qq = t / np, i = t - qq * np;
          float g = sm[qq][2 * i][0], up = sm[qq][2 * i + 1][0];
          for (int w = 1; w < nw; ++w) {
            g = __fadd_rn(g, sm[qq][2 * i][w]);
            up = __fadd_rn(up, sm[qq][2 * i + 1][w]);
          }
          g = __fmul_rn(g, fac);
          up = __fmul_rn(up, fac);
          const int p = a.idx ? q0 + qq : a.gpairs[q0 + qq];
          a.y[(size_t)p * nchunkrows + (r0 >> 1) + i] =
              __fmul_rn(__fmul_rn(fx_softcap(g, a.gate_cap), fx_sigmoid(g)), fx_softcap(up, a.up_cap));
        }
      } else {
        for (int t = threadIdx.x; t < nq * nr; t += blockDim.x) {
          const int qq = t / nr, i = t - qq * nr;
          float d = sm[qq][i][0];
          for (int w = 1; w < nw; ++w) d = __fadd_rn(d, sm[qq][i][w]);
          const int p = a.idx ? q0 + qq : a.gpairs[q0 + qq];
          a.y[(size_t)p * nchunkrows + r0 + i] = __fmul_rn(d, fac);
        }
      }
      __syncthreads();
    }
  }
#undef FXG_LDX
}

// fxcomb: the end of the down products for every (row, channel): the k products of the row in slot order times the
// router weights, then the short convolution of that sum + residual (the sums of the K - 1 rows before are recomputed
// when they belong to the pass, else read from the ring), as ecomb. One thread per (row, channel), any R.
// the lines of the spans, spread over the threads of blocks nwork .. grid - 1 (no wait: weights only)
__device__ __forceinline__ void fx2_spans(const F2Spans& s) {
  const long long nt = (long long)(gridDim.x - s.nwork) * blockDim.x;
  long long g = (long long)(blockIdx.x - s.nwork) * blockDim.x + threadIdx.x;
  for (int i = 0; i < s.cnt; ++i) {
    const size_t a0 = (size_t)s.p[i] & ~(size_t)127, a1 = (size_t)s.p[i] + s.n[i];
    const long long nl = (long long)((a1 - a0 + 127) >> 7);
    for (long long q = g; q < nl; q += nt) {
      const void* ad = (const void*)(a0 + ((size_t)q << 7));
      if (s.evict_last) asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(ad));
      else asm volatile("prefetch.global.L2 [%0];" ::"l"(ad));
    }
    g = (g + nl) % nt;  // spread the next span over the threads evenly
  }
}
template <int PF>
__global__ void __launch_bounds__(256) fxcomb(FxArgs a, F2Launch l, F2Spans ps) {  // x = dots [R * k, rows], y = sums [R, rows]
  if (PF && (int)blockIdx.x >= ps.nwork) {
    fx2_spans(ps);
    return;
  }
  if (l.trig) fx2_trig();
  fx2_wait(l);
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int H = a.rows;
  if (t >= a.R * H) return;
  const int r = t / H, h = t - r * H;
  fx_comb1(a, r, h);
}

template <int FX2_E>
__global__ void __launch_bounds__(256, 2) fx2(FxArgs a, F2Launch l) {
  if (l.trig) fx2_trig();
  fx2_wait(l);
  extern __shared__ float dsm[];  // dots [R * k][hb], then the row sums [R][hb]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int hb = a.hb;
  const int h = blockIdx.x * hb + warp;
  if (h >= a.rows) return;  // whole warp
  const int nunit = a.cols >> 4;
  const size_t wbytes = (size_t)(a.cols >> 1), sbytes = (size_t)(a.cols >> 4);
  const bool ok2 = lane + 64 < nunit;
  float* dots = dsm;
  float* sums = dsm + (size_t)a.R * a.k * hb;
  const int nexp = a.idx ? a.k : a.gcount[0];
#pragma unroll 1
  for (int e0 = 0; e0 < nexp; e0 += FX2_E) {
    int ne = nexp - e0;
    if (ne > FX2_E) ne = FX2_E;
    unsigned long long c[FX2_E][3];
    unsigned sb[FX2_E][3];
#pragma unroll
    for (int e = 0; e < FX2_E; ++e) {  // the loads of up to 8 experts before the first sum
      c[e][0] = c[e][1] = c[e][2] = 0ull;
      sb[e][0] = sb[e][1] = sb[e][2] = 0u;
      if (e < ne) {
        const int id = a.idx ? a.idx[e0 + e] : a.gexp[e0 + e];
        const size_t row = (size_t)id * a.rows + (size_t)h;
        const u8* wp = a.w + row * wbytes;
        const u8* sp = a.sc + row * sbytes;
        c[e][0] = __ldg((const unsigned long long*)(wp + (size_t)lane * 8));
        c[e][1] = __ldg((const unsigned long long*)(wp + (size_t)(lane + 32) * 8));
        sb[e][0] = __ldg(sp + lane);
        sb[e][1] = __ldg(sp + lane + 32);
        if (ok2) {
          c[e][2] = __ldg((const unsigned long long*)(wp + (size_t)(lane + 64) * 8));
          sb[e][2] = __ldg(sp + lane + 64);
        }
      }
    }
#pragma unroll
    for (int e = 0; e < FX2_E; ++e) {
      if (e < ne) {
        const int id = a.idx ? a.idx[e0 + e] : a.gexp[e0 + e];
        const float fac = a.factor[id];
        const float s0 = fx_e4m3(sb[e][0]), s1 = fx_e4m3(sb[e][1]), s2 = fx_e4m3(sb[e][2]);
        int qb, qe;
        if (a.idx) {
          qb = e0 + e;
          qe = qb + 1;
        } else {
          qb = a.goff[e0 + e];
          qe = a.goff[e0 + e + 1];
        }
#pragma unroll 1
        for (int q = qb; q < qe; ++q) {
          const int p = a.idx ? q : a.gpairs[q];
          const float* xx = a.x + (size_t)p * (size_t)a.cols;
          const float* x0 = xx + ((size_t)lane << 4);
          float acc = __fmul_rn(fx_dot16(fx_opaque(c[e][0]), fx_ld4(x0), fx_ld4(x0 + 4), fx_ld4(x0 + 8), fx_ld4(x0 + 12)), s0);
          const float* x1 = xx + ((size_t)(lane + 32) << 4);
          acc = fmaf(fx_dot16(fx_opaque(c[e][1]), fx_ld4(x1), fx_ld4(x1 + 4), fx_ld4(x1 + 8), fx_ld4(x1 + 12)), s1, acc);
          if (ok2) {
            const float* x2 = xx + ((size_t)(lane + 64) << 4);
            acc = fmaf(fx_dot16(fx_opaque(c[e][2]), fx_ld4(x2), fx_ld4(x2 + 4), fx_ld4(x2 + 8), fx_ld4(x2 + 12)), s2, acc);
          }
          const float t = fx_xsum(acc);
          if (lane == 0) dots[(size_t)p * hb + warp] = __fmul_rn(t, fac);
        }
      }
    }
  }
  __syncwarp();
  for (int r = lane; r < a.R; r += 32) {  // the k products of a row in slot order
    float o = 0.0f;
    for (int j = 0; j < a.k; ++j) o = fmaf(dots[((size_t)r * a.k + j) * hb + warp], a.wt[(size_t)r * a.k + j], o);
    sums[(size_t)r * hb + warp] = o;
    a.y[(size_t)r * a.rows + h] = o;
  }
  __syncwarp();
  if (a.cw) {
    const int K1 = a.K - 1;
    for (int r = lane; r < a.R; r += 32) {
      const float x = sums[(size_t)r * hb + warp];
      const int pos = a.st[r].pos;
      float o = 0.0f;
      for (int t = 0; t < K1; ++t) {
        const int lag = K1 - t;
        const float hv = r >= lag ? sums[(size_t)(r - lag) * hb + warp] : a.chist[(size_t)fx_slot(pos - lag, a.hr) * a.rows + h];
        o = fmaf(hv, fx_bf(a.cw[(size_t)h * a.K + t]), o);
      }
      o = fmaf(x, fx_bf(a.cw[(size_t)h * a.K + K1]), o);
      o = __fdiv_rn(__fadd_rn(o, x), a.cdiv);
      a.chist[(size_t)(pos % a.hr) * a.rows + h] = x;
      a.cout[(size_t)r * a.rows + h] = __fadd_rn(a.cres[(size_t)r * a.rows + h], o);
    }
  }
}

// ---------------------------------------------------------------------------
// fbar: the general bar (block attention residual, kernels.inc) with two block reductions instead of 1 + 2 nq:
// the squares of the states and the dot products of every query with the raw states go into one reduction
// (q . (f s) = f (q . s)), the squares of all outputs into the second. One block per (row, job), H / 4 threads.
// ---------------------------------------------------------------------------
#define FB_S 6
#define FB_Q 4
// Sum of 32 values over the block, to every thread: transposed butterfly in every warp (lane l then holds value l),
// the warps added in order by warp 0. red: [32 warps][33], tot: [32].
__device__ __forceinline__ void fx_block_sum32(float* v, float* red, float* tot) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  fx_half<32>(v, lane, 16);
  fx_half<16>(v, lane, 8);
  fx_half<8>(v, lane, 4);
  fx_half<4>(v, lane, 2);
  fx_half<2>(v, lane, 1);
  red[warp * 33 + lane] = v[0];
  __syncthreads();
  if (warp == 0) {
    float t = red[lane];
    for (int w = 1; w < nw; ++w) t = __fadd_rn(t, red[w * 33 + lane]);
    tot[lane] = t;
  }
  __syncthreads();
}
__device__ __forceinline__ float fx_conv_k4(float h0, float h1, float h2, float x, unsigned long long taps, float div) {
  float o = __fmul_rn(h0, fx_bf((unsigned)(taps & 0xffffu)));
  o = fmaf(h1, fx_bf((unsigned)((taps >> 16) & 0xffffu)), o);
  o = fmaf(h2, fx_bf((unsigned)((taps >> 32) & 0xffffu)), o);
  o = fmaf(x, fx_bf((unsigned)(taps >> 48)), o);
  return __fdiv_rn(__fadd_rn(o, x), div);
}
// general K: the inputs of earlier rows of the pass from xrow, else from the ring (as k_conv1)
__device__ __forceinline__ float fx_conv1(float* hist, int hr, const u16* w, const float* xrow, int row, float x, int c,
                                          int C, int K, int pos, float div) {
  int K1 = K - 1;
  float o = 0.0f;
  for (int t = 0; t < K1; ++t) {
    int lag = K1 - t;
    float hv = row >= lag ? xrow[(size_t)c - (size_t)lag * (size_t)C] : hist[(size_t)fx_slot(pos - lag, hr) * C + c];
    o = fmaf(hv, fx_bf(w[(size_t)c * K + t]), o);
  }
  o = fmaf(x, fx_bf(w[(size_t)c * K + K1]), o);
  o = __fdiv_rn(__fadd_rn(o, x), div);
  hist[(size_t)(pos % hr) * C + c] = x;
  return o;
}

// second set: cb.x != NULL: job 0 first does the work of fxcomb for its row (its last state = cb.cout of this row)
template <int COMB>
__global__ void __launch_bounds__(576) fbar(BarArgs a, F2Launch l, FxArgs cb) {  // second set: bound 576 (was 768), H <= 2304
  if (l.trig) fx2_trig();
  __shared__ float red[32 * 33];
  __shared__ float tot[32];
  const int H = a.H;
  int jb = blockIdx.x, row = 0;
  if (a.nj > 0) {
    jb = blockIdx.x % a.nj;
    row = blockIdx.x / a.nj;
  }
  const BarJob& J = a.job[jb];
  const size_t ro = (size_t)row * (size_t)H;
  const int ns = J.ns, nq = J.nq;
  const int j0 = threadIdx.x * 4;
  const bool act = j0 < H;
  const bool conv = J.cw != 0;
  // second set: the parameters (queries, convolution taps, norm weights) before the wait, the states after it
  float4 q[FB_Q];
#pragma unroll
  for (int qi = 0; qi < FB_Q; ++qi) {
    q[qi] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (act && qi < nq) q[qi] = *(const float4*)(J.q + (size_t)qi * H + j0);
  }
  unsigned long long taps[4] = {0ull, 0ull, 0ull, 0ull};
  if (conv && act && a.K == 4) {
    const unsigned long long* wp = (const unsigned long long*)(J.cw + (size_t)j0 * 4);
    taps[0] = wp[0];
    taps[1] = wp[1];
    taps[2] = wp[2];
    taps[3] = wp[3];
  }
  unsigned long long nwv[FB_Q] = {0ull, 0ull, 0ull, 0ull};
#pragma unroll
  for (int qi = 0; qi < FB_Q; ++qi)
    if (act && qi < nq && J.normw[qi]) nwv[qi] = *(const unsigned long long*)(J.normw[qi] + j0);
  fx2_wait(l);
  const bool comb = COMB && jb == 0;
  float4 s[FB_S];
#pragma unroll
  for (int i = 0; i < FB_S; ++i) {
    s[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (act && i < ns && !((conv || comb) && i == ns - 1)) s[i] = *(const float4*)(J.st[i] + ro + j0);
  }
  if (COMB && comb && act) {
    float4 c;
    c.x = fx_comb1(cb, row, j0);
    c.y = fx_comb1(cb, row, j0 + 1);
    c.z = fx_comb1(cb, row, j0 + 2);
    c.w = fx_comb1(cb, row, j0 + 3);
#pragma unroll
    for (int i = 0; i < FB_S; ++i)
      if (i == ns - 1) s[i] = c;
  }
  if (conv && act) {
    const int pos = a.st[row].pos;
    const int K = a.K;
    const int hr = a.hr > 0 ? a.hr : K - 1;
    float* hist = J.chist;
    const float* cxr = J.cx + ro;
    const float4 x = *(const float4*)(cxr + j0);
    float4 o;
    if (K == 4) {
      const float4 h0 = row >= 3 ? *(const float4*)(cxr - 3 * (size_t)H + j0) : *(const float4*)(hist + (size_t)fx_slot(pos - 3, hr) * H + j0);
      const float4 h1 = row >= 2 ? *(const float4*)(cxr - 2 * (size_t)H + j0) : *(const float4*)(hist + (size_t)fx_slot(pos - 2, hr) * H + j0);
      const float4 h2 = row >= 1 ? *(const float4*)(cxr - (size_t)H + j0) : *(const float4*)(hist + (size_t)fx_slot(pos - 1, hr) * H + j0);
      o.x = fx_conv_k4(h0.x, h1.x, h2.x, x.x, taps[0], a.cdiv);
      o.y = fx_conv_k4(h0.y, h1.y, h2.y, x.y, taps[1], a.cdiv);
      o.z = fx_conv_k4(h0.z, h1.z, h2.z, x.z, taps[2], a.cdiv);
      o.w = fx_conv_k4(h0.w, h1.w, h2.w, x.w, taps[3], a.cdiv);
      *(float4*)(hist + (size_t)(pos % hr) * H + j0) = x;
    } else {
      o.x = fx_conv1(hist, hr, J.cw, cxr, row, x.x, j0, H, K, pos, a.cdiv);
      o.y = fx_conv1(hist, hr, J.cw, cxr, row, x.y, j0 + 1, H, K, pos, a.cdiv);
      o.z = fx_conv1(hist, hr, J.cw, cxr, row, x.z, j0 + 2, H, K, pos, a.cdiv);
      o.w = fx_conv1(hist, hr, J.cw, cxr, row, x.w, j0 + 3, H, K, pos, a.cdiv);
    }
    if (J.cres) {
      const float4 r = *(const float4*)(J.cres + ro + j0);
      o.x = __fadd_rn(r.x, o.x);
      o.y = __fadd_rn(r.y, o.y);
      o.z = __fadd_rn(r.z, o.z);
      o.w = __fadd_rn(r.w, o.w);
    }
    *(float4*)(J.cout + ro + j0) = o;
#pragma unroll
    for (int i = 0; i < FB_S; ++i)
      if (i == ns - 1) s[i] = o;
  }
  // reduction 1: v[i] = |s_i|^2 (i < 6), v[6 + 6 qi + i] = q_qi . s_i
  float v[32];
#pragma unroll
  for (int i = 0; i < FB_S; ++i) {
    float t = __fmul_rn(s[i].x, s[i].x);
    t = fmaf(s[i].y, s[i].y, t);
    t = fmaf(s[i].z, s[i].z, t);
    v[i] = fmaf(s[i].w, s[i].w, t);
  }
#pragma unroll
  for (int qi = 0; qi < FB_Q; ++qi)
#pragma unroll
    for (int i = 0; i < FB_S; ++i) {
      float t = __fmul_rn(q[qi].x, s[i].x);
      t = fmaf(q[qi].y, s[i].y, t);
      t = fmaf(q[qi].z, s[i].z, t);
      v[FB_S + qi * FB_S + i] = fmaf(q[qi].w, s[i].w, t);
    }
  v[30] = v[31] = 0.0f;
  fx_block_sum32(v, red, tot);
  float f[FB_S];
#pragma unroll
  for (int i = 0; i < FB_S; ++i) f[i] = i < ns ? 1.0f / sqrtf(tot[i] / (float)H + 1.0e-6f) : 0.0f;
  float4 o[FB_Q];
  float w2[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) w2[i] = 0.0f;
#pragma unroll
  for (int qi = 0; qi < FB_Q; ++qi) {
    o[qi] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (qi < nq) {
      float sc[FB_S];
      float m = -3.0e38f;
#pragma unroll
      for (int i = 0; i < FB_S; ++i) {
        sc[i] = 0.0f;
        if (i < ns) {
          sc[i] = __fmul_rn(__fmul_rn(tot[FB_S + qi * FB_S + i], f[i]), a.hscale);
          m = fmaxf(m, sc[i]);
        }
      }
      float sum = 0.0f;
#pragma unroll
      for (int i = 0; i < FB_S; ++i)
        if (i < ns) {
          sc[i] = expf(sc[i] - m);
          sum = __fadd_rn(sum, sc[i]);
        }
#pragma unroll
      for (int i = 0; i < FB_S; ++i)
        if (i < ns) {
          const float cf = __fmul_rn(__fdiv_rn(sc[i], sum), f[i]);
          o[qi].x = fmaf(cf, s[i].x, o[qi].x);
          o[qi].y = fmaf(cf, s[i].y, o[qi].y);
          o[qi].z = fmaf(cf, s[i].z, o[qi].z);
          o[qi].w = fmaf(cf, s[i].w, o[qi].w);
        }
      float t = __fmul_rn(o[qi].x, o[qi].x);
      t = fmaf(o[qi].y, o[qi].y, t);
      t = fmaf(o[qi].z, o[qi].z, t);
      w2[qi] = fmaf(o[qi].w, o[qi].w, t);
    }
  }
  bool need = false;
  for (int qi = 0; qi < nq; ++qi) need = need || J.normw[qi] || J.outn[qi];
  if (need) fx_block_sum32(w2, red, tot);  // (the first reduction ended with a barrier after its last read of red)
#pragma unroll
  for (int qi = 0; qi < FB_Q; ++qi) {
    if (qi >= nq) break;
    float4 r = o[qi];
    const u16* w = J.normw[qi];
    float* on = J.outn[qi] ? J.outn[qi] + ro : 0;
    if ((w || on) && act) {
      const float scv = 1.0f / sqrtf(tot[qi] / (float)H + a.eps);
      const float4 n = make_float4(__fmul_rn(r.x, scv), __fmul_rn(r.y, scv), __fmul_rn(r.z, scv), __fmul_rn(r.w, scv));
      if (on) *(float4*)(on + j0) = n;
      if (w) {  // weighted norm in place; with outn only the output itself stays as it is (as bar)
        const unsigned long long wv = nwv[qi];
        r.x = __fmul_rn(fx_bf((unsigned)(wv & 0xffffu)), n.x);
        r.y = __fmul_rn(fx_bf((unsigned)((wv >> 16) & 0xffffu)), n.y);
        r.z = __fmul_rn(fx_bf((unsigned)((wv >> 32) & 0xffffu)), n.z);
        r.w = __fmul_rn(fx_bf((unsigned)(wv >> 48)), n.w);
      }
    }
    if (act) *(float4*)(J.out[qi] + ro + j0) = r;
  }
}

// ---------------------------------------------------------------------------
// fprep: the general prep (kernels.inc) with every load of a lane issued before the sums (head size <= 128: 4
// elements per lane), the outputs scaled from registers. One warp per q / r head or k / v head, as prep.
// ---------------------------------------------------------------------------
template <int MINB>
__global__ void __launch_bounds__(256, MINB) fprep(PrepArgs a, F2Launch l) {
  if (l.trig) fx2_trig();
  const int lane = threadIdx.x & 31;
  int row = 0, b = blockIdx.x;
  if (a.bpr > 0) {
    row = blockIdx.x / a.bpr;
    b = blockIdx.x - row * a.bpr;
  }
  const int gw = b * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int nh = a.nh, nkv = a.nkv;
  if (gw >= 2 * nh + 2 * nkv) {
    fx2_wait(l);
    return;
  }
  if (gw < 2 * nh) {
    const int isr = gw >= nh;
    const int h = isr ? gw - nh : gw;
    const int n = isr ? a.drel : a.hd;
    float* v = (isr ? a.r + (size_t)row * nh * a.drel : a.q + (size_t)row * nh * a.hd) + (size_t)h * n;
    const u16* scl = (isr ? a.r_scale : a.q_scale) + (size_t)h * n;
    const float thr = isr ? a.rel_thr : a.qk_thr;
    float xv[4], sv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) sv[i] = lane + 32 * i < n ? fx_bf(scl[lane + 32 * i]) : 0.0f;  // second set: before the wait
    fx2_wait(l);
    const float tau = a.st[row].tau;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int j = lane + 32 * i;
      xv[i] = j < n ? v[j] : 0.0f;
    }
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      xv[i] = __fmul_rn(xv[i], sv[i]);
      ss = fmaf(xv[i], xv[i], ss);
    }
    const float tot = fx_xsum(ss);
    float f = thr / (sqrtf(tot) + 1.0e-6f);
    if (f > 1.0f) f = 1.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int j = lane + 32 * i;
      if (j < n) {
        float x = __fmul_rn(xv[i], f);
        if (!a.sliding) x = __fmul_rn(x, tau);
        v[j] = x;
      }
    }
    return;
  }
  const int isv = gw >= 2 * nh + nkv;
  const int hk = gw - 2 * nh - (isv ? nkv : 0);
  const int hd = a.hd, K = a.K, K1 = a.K - 1;
  const int C = nkv * hd;
  const int hr = a.hr > 0 ? a.hr : K1;
  float wraw[4][8];  // second set: the convolution weights before the wait
  {
    const u16* w = isv ? a.v_w : a.k_w;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int j = lane + 32 * i;
      const int c = hk * hd + (j < hd ? j : 0);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        wraw[i][t] = 0.0f;
        if (j < hd && t < K) wraw[i][t] = a.sliding ? fx_bf(w[(size_t)c * K + t]) : fx_bf(w[(size_t)t * C + c]);
      }
    }
  }
  fx2_wait(l);
  const int pos = a.st[row].pos;
  const int cslot = a.ring ? pos % a.ring : pos;
  const float* raw = (isv ? a.vraw : a.kraw) + (size_t)row * C;
  float* hist = isv ? a.vhist : a.khist;
  float* dst = (isv ? a.vc : a.kc) + (size_t)cslot * C;
  const float* dyn = (isv ? a.v_dyn : a.k_dyn) + (a.sliding ? 0 : (size_t)row * nkv * K);
  float xr[4], ov[4], wl[4];
  float hvs[4][8], wts[4][8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {  // every load of the lane first
    const int j = lane + 32 * i;
    const int c = hk * hd + (j < hd ? j : 0);
    xr[i] = j < hd ? raw[c] : 0.0f;
    wl[i] = 0.0f;
#pragma unroll
    for (int t = 0; t < 8; ++t)
      if (t == K1 && j < hd && a.sliding) wl[i] = wraw[i][t];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      hvs[i][t] = wts[i][t] = 0.0f;
      if (j < hd && t < K) {
        if (a.sliding) {
          if (t < K1) {
            const int lag = K1 - t;
            hvs[i][t] = row >= lag ? raw[(size_t)c - (size_t)lag * C] : hist[(size_t)fx_slot(pos - lag, hr) * C + c];
          }
          if (t < K1) wts[i][t] = wraw[i][t];
        } else {
          if (t > 0) hvs[i][t] = row >= t ? raw[(size_t)c - (size_t)t * C] : hist[(size_t)fx_slot(pos - t, hr) * C + c];
          wts[i][t] = __fadd_rn(wraw[i][t], dyn[hk * K + t]);
        }
      }
    }
  }
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    ov[i] = 0.0f;
    const int j = lane + 32 * i;
    if (j < hd) {
      const int c = hk * hd + j;
      const float x = xr[i];
      float o;
      if (a.sliding) {
        o = 0.0f;
#pragma unroll
        for (int t = 0; t < 8; ++t)
          if (t < K1) o = fmaf(hvs[i][t], wts[i][t], o);
        o = fmaf(x, wl[i], o);
      } else {
        o = __fmul_rn(x, wts[i][0]);
#pragma unroll
        for (int t = 1; t < 8; ++t)
          if (t < K) o = fmaf(hvs[i][t], wts[i][t], o);
      }
      o = __fdiv_rn(__fadd_rn(o, x), a.conv_div);
      hist[(size_t)(pos % hr) * C + c] = x;
      ov[i] = o;
      ss = fmaf(o, o, ss);
    }
  }
  const float tot = fx_xsum(ss);
  float f;
  if (isv) {
    f = 1.0f / sqrtf(tot / (float)hd + a.eps);
  } else {
    f = a.qk_thr / (sqrtf(tot) + 1.0e-6f);
    if (f > 1.0f) f = 1.0f;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int j = lane + 32 * i;
    if (j < hd) dst[hk * hd + j] = __fmul_rn(ov[i], f);
  }
}

int g_smem_max = 48 * 1024;

}  // namespace

int ink2_fast_probe(void) {
  cudaFuncAttributes fa{};
  if (cudaFuncGetAttributes(&fa, fxcol<8, 0, 0, 0>) != cudaSuccess || cudaFuncGetAttributes(&fa, fx2<8>) != cudaSuccess) {
    cudaGetLastError();
    return 0;  // no code of the fast kernels for this device
  }
  int dev = 0, maj = 0, mnr = 0, optin = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  int cc = maj * 10 + mnr;
  if (cc < 86) return 0;
  int dyn = optin - (int)fa.sharedSizeBytes;
  if (dyn > 48 * 1024 && cudaFuncSetAttribute(fx2<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn) == cudaSuccess &&
      cudaFuncSetAttribute(fx2<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn) == cudaSuccess)
    g_smem_max = dyn;
  cudaGetLastError();
  return cc;
}

int ink2_fast_smem_max(void) { return g_smem_max; }

int ink2_fast_ptx_version(void) {
  cudaFuncAttributes fa{};
  if (cudaFuncGetAttributes(&fa, fxcol<8, 0, 0, 0>) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return fa.ptxVersion;
}

static void fx_launch_pf(void (*k)(FxArgs, F2Launch, F2Spans), const FxArgs& a, const F2Spans& c, int grid, int block,
                         cudaStream_t s, bool pdl, const F2Launch& l) {
  if (!pdl) {
    k<<<grid, block, 0, s>>>(a, l, c);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3((unsigned)block);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k, a, l, c);
}
static void fx_launch_col(void (*k)(FxArgs, F2Launch, F2ColComb), const FxArgs& a, const F2ColComb& c, int grid, int block,
                          cudaStream_t s, bool pdl, const F2Launch& l) {
  if (!pdl) {
    k<<<grid, block, 0, s>>>(a, l, c);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3((unsigned)block);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k, a, l, c);
}
template <class A>
static void fx_launch(void (*k)(A, F2Launch), const A& a, int grid, int block, int smem, cudaStream_t s, bool pdl,
                      const F2Launch& l) {
  if (!pdl) {
    k<<<grid, block, smem, s>>>(a, l);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3((unsigned)block);
  cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k, a, l);
}

static void fx_launch_bar(void (*k)(BarArgs, F2Launch, FxArgs), const BarArgs& a, const FxArgs& c, int grid, int block,
                          cudaStream_t s, bool pdl, const F2Launch& l) {
  if (!pdl) {
    k<<<grid, block, 0, s>>>(a, l, c);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3((unsigned)block);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k, a, l, c);
}

// second set: the options of ink2_fast2.h (set by the backend) and the launch details of every fast kernel
static F2Opts g_f2 = {};
void ink2_fast_set_f2(const F2Opts* o) { g_f2 = o ? *o : F2Opts{}; }
template <int NR, int MODE, int LUT>
static F2Launch fx2_launch_col(const FxArgs& a, int grid, int block, bool pdl) {
  F2Launch l{};
  l.pf = pdl && g_f2.pf ? g_f2.pf : 0;
  if (l.pf == 3 && !(g_f2.reg & 8)) l.pf = 1;  // MODE 1: own rows before the wait (3: in registers); extra waves: pf_extra bit 2 (w2) / bit 4 (w13)
  if (!l.pf || !(g_f2.pf_extra & (MODE == 1 ? 2 : 4))) return l;
  static int res = -1;
  if (res < 0) {
    int per = 0, dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fxcol<NR / 2, MODE, LUT, 0>, block, 0) != cudaSuccess) per = 0;
    cudaGetLastError();
    res = per * sms;
  }
  if (res <= 0 || grid <= res) return l;
  const size_t per_item = (size_t)NR * (size_t)((a.cols >> 1) + (a.cols >> 4));
  long long rounds = (grid + res - 1) / res - 1;
  const long long fit = g_f2.pf_budget / (long long)(per_item * (size_t)res) - 1;
  if (rounds > fit) rounds = fit;
  if (rounds > 0) l.extra = res, l.rounds = (int)rounds;
  return l;
}

// grid of the PIPE form: at most the blocks that are resident at once (times g_f2.pipe_waves)
static int fx2_pipe_grid(void (*k)(FxArgs, F2Launch, F2ColComb), int grid, int block) {
  static const void* kk[4] = {};
  static int rr[4] = {};
  int res = 0;
  for (int i = 0; i < 4; ++i)
    if (kk[i] == (const void*)k) res = rr[i];
  if (!res) {
    int per = 0, dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, block, 0) != cudaSuccess) per = 1;
    cudaGetLastError();
    res = per * sms;
    for (int i = 0; i < 4; ++i)
      if (!kk[i]) {
        kk[i] = (const void*)k;
        rr[i] = res;
        break;
      }
  }
  const int g = res * (g_f2.pipe_waves > 0 ? g_f2.pipe_waves : 1);
  return grid < g ? grid : g;
}
int ink2_fast_run(int fid, const void* args, int grid, int block, int smem, void* stream, bool pdl) {
  cudaStream_t s = (cudaStream_t)stream;
  const FxArgs& a = *(const FxArgs*)args;
  F2Launch l0{};  // the general order: wait, then trigger
  F2Launch lb{};  // fbar: it comes right before a weight kernel
  lb.trig = pdl && g_f2.trig_small;
  F2Launch lp{};  // fprep
  switch (fid) {
    case INK2_FX13: fx_launch_col(fxcol<8, 0, 0>, a, F2ColComb{}, grid, block, s, pdl, fx2_launch_col<16, 0, 0>(a, grid, block, pdl)); break;
    case INK2_FX13_P4: fx_launch_col(fxcol<4, 0, 0>, a, F2ColComb{}, grid, block, s, pdl, fx2_launch_col<8, 0, 0>(a, grid, block, pdl)); break;
    case INK2_FX13L: fx_launch_col(fxcol<8, 0, 1>, a, F2ColComb{}, grid, block, s, pdl, fx2_launch_col<16, 0, 1>(a, grid, block, pdl)); break;
    case INK2_FX13L4:
      if (g_f2.pipe & 1) fx_launch_col(fxcol<4, 0, 1, 0, 1>, a, F2ColComb{}, fx2_pipe_grid(fxcol<4, 0, 1, 0, 1>, grid, block), block, s, pdl, fx2_launch_col<8, 0, 1>(a, grid, block, pdl));
      else fx_launch_col(fxcol<4, 0, 1>, a, F2ColComb{}, grid, block, s, pdl, fx2_launch_col<8, 0, 1>(a, grid, block, pdl));
      break;
    case INK2_FX2C: fx_launch_col(fxcol<8, 1, 0>, a, F2ColComb{}, grid, block, s, pdl, fx2_launch_col<16, 1, 0>(a, grid, block, pdl)); break;
    case INK2_FX2CL: fx_launch_col(fxcol<8, 1, 1>, a, F2ColComb{}, grid, block, s, pdl, fx2_launch_col<16, 1, 1>(a, grid, block, pdl)); break;
    case INK2_FX2CL4:
      if (g_f2.pipe & 2) fx_launch_col(fxcol<4, 1, 1, 0, 1>, a, F2ColComb{}, fx2_pipe_grid(fxcol<4, 1, 1, 0, 1>, grid, block), block, s, pdl, fx2_launch_col<8, 1, 1>(a, grid, block, pdl));
      else fx_launch_col(fxcol<4, 1, 1>, a, F2ColComb{}, grid, block, s, pdl, fx2_launch_col<8, 1, 1>(a, grid, block, pdl));
      break;
    case INK2_FX2CL4C: {  // second set: F2ColComb (fx2c args in .a of the struct)
      const F2ColComb& cc = *(const F2ColComb*)args;
      fx_launch_col(fxcol<4, 1, 1, 1>, cc.a, cc, grid, block, s, pdl, fx2_launch_col<8, 1, 1>(cc.a, grid, block, pdl));
      break;
    }
    // passes, codes decoded once for 2 / 4 pairs (launch details of FX13L4 / FX2CL4)
    case INK2_FX13G2: fx_launch(fxcolg<4, 0, 2>, a, grid, block, 0, s, pdl, fx2_launch_col<8, 0, 1>(a, grid, block, pdl)); break;
    case INK2_FX13G2N: fx_launch(fxcolg<4, 0, 2, 0>, a, grid, block, 0, s, pdl, fx2_launch_col<8, 0, 1>(a, grid, block, pdl)); break;
    case INK2_FX2CG2N: fx_launch(fxcolg<4, 1, 2, 0>, a, grid, block, 0, s, pdl, fx2_launch_col<8, 1, 1>(a, grid, block, pdl)); break;
    case INK2_FX13G4: fx_launch(fxcolg<4, 0, 4>, a, grid, block, 0, s, pdl, fx2_launch_col<8, 0, 1>(a, grid, block, pdl)); break;
    case INK2_FX2CG2: fx_launch(fxcolg<4, 1, 2>, a, grid, block, 0, s, pdl, fx2_launch_col<8, 1, 1>(a, grid, block, pdl)); break;
    case INK2_FX2CG4: fx_launch(fxcolg<4, 1, 4>, a, grid, block, 0, s, pdl, fx2_launch_col<8, 1, 1>(a, grid, block, pdl)); break;
    case INK2_FXCOMB: fx_launch_pf(fxcomb<0>, a, F2Spans{}, grid, block, s, pdl, l0); break;
    case INK2_FXCOMBP: {
      const F2CombPf& cp = *(const F2CombPf*)args;
      fx_launch_pf(fxcomb<1>, cp.a, cp.s, grid, block, s, pdl, l0);
      break;
    }
    case INK2_FX2: fx_launch(fx2<8>, a, grid, block, smem, s, pdl, l0); break;
    case INK2_FX2_E4: fx_launch(fx2<4>, a, grid, block, smem, s, pdl, l0); break;
    case INK2_FBAR: fx_launch_bar(fbar<0>, *(const BarArgs*)args, FxArgs{}, grid, block, s, pdl, lb); break;
    case INK2_FBARC: fx_launch_bar(fbar<1>, ((const F2BarComb*)args)->b, ((const F2BarComb*)args)->c, grid, block, s, pdl, lb); break;
    case INK2_FPREP: fx_launch(g_f2.prep_minb == 2 ? fprep<2> : fprep<1>, *(const PrepArgs*)args, grid, block, 0, s, pdl, lp); break;
    default: return (int)cudaErrorInvalidValue;
  }
  return (int)cudaGetLastError();
}
