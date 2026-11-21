// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Argument blocks of the kernels. Plain data, passed by value to a CUDA
// kernel and by reference to the host emulation of the same kernel.
#pragma once
#include <stddef.h>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned long long u64;

enum { INK2_EPI_NONE = 0, INK2_EPI_SIGMOID = 1, INK2_EPI_SOFTCAP = 2, INK2_EPI_TANH = 3 };
enum { INK2_FMT_NVFP4 = 0, INK2_FMT_MXFP8 = 1 };

// What changes from token to token. The kernels read it from the backend
// memory, so that the argument blocks of a step are the same for every token
// (a step can then be replayed as one captured graph). A pass over R
// consecutive positions has one StepState per row (array of R).
//
// Rows. Every kernel that takes `rows` (or `nrhs`) handles R consecutive
// positions of one pass; per-row buffers are [R, n] with the row stride n of
// the buffer named in the comment. rows == 0 means one row (old argument
// blocks of the self-test). The arithmetic of a row is the same as with one
// row, so that a pass over R rows gives bit for bit the results of R single
// steps on the same backend.
// Short convolution histories are rings of hr >= K - 1 slots, position p in
// slot p % hr; a row whose earlier inputs lie inside the pass reads them from
// the input buffer of the pass instead (hr >= R + K - 1 keeps the slots read
// and written by one kernel apart).
struct StepState {
  int token, pos;
  float tau;  // log scaling factor of this position
  int pad;
};

// y = W x, W bfloat16 [rows, cols] (cols % 8 == 0), x and y float32.
// The input may come in two pieces: x for the columns below split, x2 for the rest.
struct GemvJob {
  const u16* w;
  const float* x;
  const float* x2;
  float* y;
  int rows, cols, split, epi;
  float cap;
};
// Multi-row forms (gemvm, gemvwm, gemvcm): nrhs right-hand sides, x row
// stride split, x2 row stride cols - split, y row stride rows; groups of up to
// 8 right-hand sides, `blocks` blocks per group.
struct GemvArgs {
  GemvJob job[8];
  int njobs, total_rows;
  int nrhs, blocks;
};

// out = w * (x * rsqrt(mean(x^2) + eps)); w == NULL: no weight. One block per job.
struct NormJob {
  const void* x;  // float32, or bfloat16 bits when x_bf16
  const u16* w;
  float* out;
  int x_bf16;
};
struct NormArgs {
  NormJob job[4];
  const StepState* st;
  int embed_stride;  // > 0: job 0 reads row st[row].token of the bfloat16 table x
  int n;
  float eps;
  int rows, nj;  // block = row * nj + job; x and out [rows, n]
};

// Top-k routing from the router logits. One block of >= E threads.
struct RouterArgs {
  const float* logits;  // [rows, E]
  const float* bias;
  int* idx;   // [rows, k], increasing
  float* wt;  // [rows, k]
  int E, k;
  float route_scale;
  int rows;  // one block per row
};

// Block attention residual: nq queries over ns <= 6 states. One block of H / 4
// threads per job.
// Optional first step: the last state st[ns-1] (= cout) is produced by a short
// convolution of cx (+ residual cres). Per output: RMS norm with weight in
// place (normw), or an extra RMS-normalised copy (outn).
struct BarJob {
  const float* st[8];
  const float* q;  // [nq, H]
  float* out[4];
  float* outn[4];
  const u16* normw[4];
  const float* cx;
  const u16* cw;  // [H, K] or NULL
  float* chist;   // [(K-1), H]
  const float* cres;
  float* cout;
  int ns, nq;
};
// Rows: block = row * nj + job; every per-row pointer of a job ([rows, H]:
// states, outputs, cx, cres, cout) advances by H per row, q is shared.
struct BarArgs {
  BarJob job[2];
  const StepState* st;
  int H, K;
  float hscale;  // H ** -0.5
  float eps;     // of the optional norms
  float cdiv;
  int rows, nj, hr;
};

// Last step of a message gate: (1 - f) * hs + g * msg, then the optional RMS
// norm with weight. One block per job.
struct GmixJob {
  const float* hs;
  const float* msg;
  const float* f;  // tanh(forget projection)
  const float* g;  // tanh(gain projection)
  const u16* normw;
  float* out;
};
struct GmixArgs {
  GmixJob job[4];
  int H;
  float eps;
  int rows, nj;  // block = row * nj + job; all pointers [rows, H]
};

// Query / relative state scaling and clipping, key / value convolution,
// clipping / normalisation and the write into the cache slot.
struct PrepArgs {
  float* q;  // [nh, hd] in place
  const u16* q_scale;
  float* r;  // [nh, drel] in place
  const u16* r_scale;
  const float* kraw;  // [nkv * hd]
  const float* vraw;
  float* khist;  // [(K-1), nkv * hd]
  float* vhist;
  float* kc;  // cache [cap, nkv * hd]; the slot of the position is written
  float* vc;
  const u16* k_w;  // sliding: [C, K]; full: static [K, C]
  const u16* v_w;
  const float* k_dyn;  // full: [nkv, K]
  const float* v_dyn;
  const StepState* st;
  int nh, nkv, hd, drel, K, sliding, ring;
  float qk_thr, rel_thr, eps, conv_div;
  // rows: q [rows, nh*hd], r [rows, nh*drel], kraw/vraw [rows, nkv*hd],
  // k_dyn/v_dyn [rows, nkv*K]; bpr blocks per row; hr history ring slots
  int rows, bpr, hr;
};

// Attention over the cached keys. One block per head.
struct AttnArgs {
  const float* q;
  const float* r;
  const float* sink;  // [nh]
  const float* gate;  // [nh * hd], sigmoid already applied
  const float* kc;    // [cap, nkv * hd]
  const float* vc;
  const u16* rel_proj;  // [drel, rel_extent]
  float* scratch;       // [nh, scratch_stride]
  float* out;           // [nh * hd]
  const StepState* st;
  int nh, nkv, hd, drel, rel_extent, ring, scratch_stride;
  float scale;  // hd ** -0.5
  // window: keys visible to a query (0 = all); ring: slots of the cache (0 =
  // no wrap). rows: block = row * nh + head; q, gate, out [rows, nh*hd],
  // r [rows, nh*drel], sink [rows, nh], scratch [rows * nh, scratch_stride]
  int window, rows;
};

// Experts, multiplied straight from the packed bytes.
struct ExpArgs {
  const u8* w;          // [E, rows, cols/2] (nvfp4) or [E, rows, cols] (mxfp8)
  const u8* sc;         // [E, rows, cols/16] or [E, rows, cols/32]
  const float* factor;  // nvfp4: [E] amax / denominator / 2
  const int* idx;       // [k]
  const float* wt;      // [k] (w2 only)
  const float* x;       // w13: [cols]; w2: [k, cols]
  float* y;             // w13: act [k, rows/2]; w2: [rows]
  const float* e4m3;    // [256]
  const float* exp2t;   // [256]
  // w2 only, optional: short convolution of the output + residual -> cout
  const u16* cw;  // [rows, K] or NULL
  float* chist;   // [(K-1), rows]
  const float* cres;
  float* cout;
  const StepState* st;
  int fmt, rows, cols, k, K;
  float gate_cap, up_cap, cdiv;
  int hr;
};

// Experts of a multi-row pass, grouped by expert: egroup lists the distinct
// experts of the pass and for each the (row, slot) pairs p = row * k + slot
// that chose it (in increasing p). Every needed expert is then read once.
struct GroupArgs {
  const int* idx;  // [R, k]
  int* gcount;     // [1] distinct experts
  int* gexp;       // [<= E] their ids, increasing
  int* goff;       // [<= E + 1] first pair of each in gpairs
  int* gpairs;     // [R * k]
  int R, k, E;
};
struct ExpMArgs {
  const u8* w;
  const u8* sc;
  const float* factor;
  const float* e4m3;
  const float* exp2t;
  const int* gcount;
  const int* gexp;
  const int* goff;
  const int* gpairs;
  const float* x;  // w13: [R, cols]; w2: activations [R * k, cols]
  float* y;        // w13: activations [R * k, rows / 2]; w2: products [R * k, rows]
  int fmt, rows, cols, k;
  int nchunk;  // blocks of matrix rows per expert
  int grid;    // blocks of the launch (the items distinct * nchunk are spread over them)
  float gate_cap, up_cap;
};
// Sum of the k expert products of every row (in slot order, as exp2), then
// the short convolution + residual. One thread per (row, channel).
struct ECombArgs {
  const float* dots;  // [R * k, H]
  const float* wt;    // [R, k]
  float* y;           // [R, H]
  const u16* cw;      // [H, K] or NULL
  float* chist;       // [hr, H]
  const float* cres;  // [R, H]
  float* cout;        // [R, H]
  const StepState* st;
  int R, k, H, K, hr;
  float cdiv;
};
// Largest logit of every row (first index on ties). One block per row.
struct ArgmaxArgs {
  const float* logits;  // [rows, n]
  int* idx;             // [rows]
  float* val;           // [rows]
  int n;
};

#define INK2_KERNEL_LIST(X) \
  X(gemv, GemvArgs) \
  X(gemvw, GemvArgs) \
  X(gemvc, GemvArgs) \
  X(norm, NormArgs) \
  X(bar, BarArgs) \
  X(gmix, GmixArgs) \
  X(prep, PrepArgs) \
  X(attn, AttnArgs) \
  X(topk, RouterArgs) \
  X(exp13, ExpArgs) \
  X(exp13c, ExpArgs) \
  X(exp2, ExpArgs) \
  X(gemvm, GemvArgs) \
  X(gemvwm, GemvArgs) \
  X(gemvcm, GemvArgs) \
  X(egroup, GroupArgs) \
  X(exp13cm, ExpMArgs) \
  X(exp2m, ExpMArgs) \
  X(ecomb, ECombArgs) \
  X(argmax, ArgmaxArgs)

enum {
#define X(name, A) KID_##name,
  INK2_KERNEL_LIST(X)
#undef X
  KID_COUNT
};

// The interface between the shared engine and a backend.
struct Backend {
  virtual ~Backend() {}
  virtual const char* name() = 0;
  virtual void* alloc(size_t bytes) = 0;  // zero filled, 256-byte aligned
  virtual void release(void* p) = 0;
  virtual void upload(void* dst, const void* host_src, size_t bytes) = 0;
  virtual void download(void* host_dst, const void* src, size_t bytes) = 0;
  virtual void zero(void* dst, size_t bytes) = 0;
  virtual void sync() = 0;
  virtual void run(int kid, const void* args, int grid, int block) = 0;
  virtual size_t bytes_allocated() = 0;
  virtual bool device_memory(long long* free_bytes, long long* total_bytes) = 0;
  virtual const char* error() = 0;  // NULL when no error happened
  // Optional: record the kernels of a step once and replay them with one call.
  virtual bool graph_supported() { return false; }
  virtual void graph_begin() {}
  virtual void* graph_end() { return nullptr; }
  virtual void graph_launch(void*) {}
  virtual void graph_free(void*) {}
};

Backend* ink2_make_host_backend();
Backend* ink2_make_cuda_backend(char* err, int err_cap);  // NULL on failure
