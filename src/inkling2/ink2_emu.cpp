// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Host backend: runs the kernels of ink2_kernels.inc on the CPU through a
// small emulation of CUDA thread blocks. Every emulated thread is a Windows
// fiber; shuffles and barriers are the points where a fiber yields to the
// scheduler of its block. Blocks are independent and run under OpenMP.
// The emulator also checks the rules a real GPU does not report: a warp whose
// lanes do not all take part in a shuffle, and barriers that can never be met.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <immintrin.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "ink2_kargs.h"

#pragma fp_contract(off)

namespace {

enum { ST_RUN = 0, ST_SHFL = 1, ST_SYNC = 2, ST_DONE = 3 };

struct EmuPool;

struct EmuT {
  int tid = 0, bid = 0, nthr = 0;
  int state = ST_DONE;
  int src = 0;  // lane whose value this lane asks for
  unsigned val = 0, res = 0;
  size_t sh_off = 0;
  EmuPool* pool = nullptr;
  void* fiber = nullptr;

  inline unsigned yield_shfl(unsigned bits, int src_lane);
  inline void sync();
  inline void* shared(size_t bytes);
  float shfl_down(float v, int d) {
    int lane = tid & 31;
    unsigned b;
    memcpy(&b, &v, 4);
    b = yield_shfl(b, lane + d < 32 ? lane + d : lane);
    memcpy(&v, &b, 4);
    return v;
  }
  float shfl(float v, int src_lane) {
    unsigned b;
    memcpy(&b, &v, 4);
    b = yield_shfl(b, src_lane & 31);
    memcpy(&v, &b, 4);
    return v;
  }
};

typedef void (*Thunk)(EmuT&, const void*);

struct EmuPool {
  void* main_fiber = nullptr;
  std::vector<EmuT*> thr;
  std::vector<unsigned char> shmem;
  Thunk thunk = nullptr;
  const void* args = nullptr;
};

std::mutex g_err_mutex;
std::string g_err;
std::atomic<bool> g_has_err(false);

void set_error(const std::string& s) {
  std::lock_guard<std::mutex> lock(g_err_mutex);
  if (!g_has_err.load()) {
    g_err = s;
    g_has_err.store(true);
  }
}

inline unsigned EmuT::yield_shfl(unsigned bits, int src_lane) {
  val = bits;
  src = src_lane;
  state = ST_SHFL;
  SwitchToFiber(pool->main_fiber);
  return res;
}
inline void EmuT::sync() {
  state = ST_SYNC;
  SwitchToFiber(pool->main_fiber);
}
inline void* EmuT::shared(size_t bytes) {
  size_t off = (sh_off + 15) & ~(size_t)15;
  sh_off = off + bytes;
  if (sh_off > pool->shmem.size()) {
    set_error("emulator: shared memory of a block exceeds 48 KB");
    return pool->shmem.data();
  }
  return pool->shmem.data() + off;
}

VOID CALLBACK fiber_proc(PVOID p) {
  EmuT* t = (EmuT*)p;
  for (;;) {
    t->pool->thunk(*t, t->pool->args);
    t->state = ST_DONE;
    SwitchToFiber(t->pool->main_fiber);
  }
}

EmuPool* get_pool() {
  static thread_local EmuPool* pool = nullptr;
  if (!pool) {
    pool = new EmuPool();
    pool->main_fiber = IsThreadAFiber() ? GetCurrentFiber() : ConvertThreadToFiberEx(nullptr, FIBER_FLAG_FLOAT_SWITCH);
    pool->shmem.resize(48 * 1024);
  }
  return pool;
}

void run_block(EmuPool* pool, Thunk thunk, const void* args, int bid, int nthr, const char* kname) {
  while ((int)pool->thr.size() < nthr) {
    EmuT* t = new EmuT();
    t->pool = pool;
    t->fiber = CreateFiberEx(0, 64 * 1024, FIBER_FLAG_FLOAT_SWITCH, fiber_proc, t);
    if (!t->fiber) {
      set_error("emulator: CreateFiberEx failed");
      delete t;
      return;
    }
    pool->thr.push_back(t);
  }
  pool->thunk = thunk;
  pool->args = args;
  memset(pool->shmem.data(), 0xFF, pool->shmem.size());  // uninitialised shared memory reads as NaN
  for (int i = 0; i < nthr; ++i) {
    EmuT* t = pool->thr[i];
    t->tid = i;
    t->bid = bid;
    t->nthr = nthr;
    t->state = ST_RUN;
    t->sh_off = 0;
  }
  int live = nthr;
  bool broken = false;
  while (live > 0) {
    bool progress = false;
    for (int i = 0; i < nthr; ++i) {
      EmuT* t = pool->thr[i];
      if (broken && t->state != ST_DONE) t->state = ST_RUN, t->res = 0;
      if (t->state != ST_RUN) continue;
      SwitchToFiber(t->fiber);
      progress = true;
      if (t->state == ST_DONE) --live;
    }
    if (broken) continue;  // let the fibers run to their end with dummy values
    // shuffles, warp by warp
    int nsync = 0;
    for (int w = 0; w * 32 < nthr; ++w) {
      int n_shfl = 0, n_other = 0, n_done = 0;
      int hi = (w + 1) * 32 < nthr ? (w + 1) * 32 : nthr;
      for (int i = w * 32; i < hi; ++i) {
        int s = pool->thr[i]->state;
        if (s == ST_SHFL) ++n_shfl;
        else if (s == ST_DONE) ++n_done;
        else ++n_other;
        if (s == ST_SYNC) ++nsync;
      }
      if (n_shfl == 0) continue;
      if (n_done > 0 || hi - w * 32 != 32) {
        char msg[200];
        snprintf(msg, sizeof msg, "emulator: kernel %s block %d warp %d: shuffle with %d of 32 lanes (%d finished)", kname,
                 bid, w, n_shfl, n_done);
        set_error(msg);
        broken = true;
        break;
      }
      if (n_other > 0) continue;  // the other lanes wait at a barrier: not yet resolvable
      unsigned vals[32];
      for (int i = 0; i < 32; ++i) vals[i] = pool->thr[w * 32 + i]->val;
      for (int i = 0; i < 32; ++i) {
        EmuT* t = pool->thr[w * 32 + i];
        t->res = vals[t->src];
        t->state = ST_RUN;
      }
      progress = true;
    }
    if (broken) {
      for (int i = 0; i < nthr; ++i)
        if (pool->thr[i]->state != ST_DONE) pool->thr[i]->state = ST_RUN, pool->thr[i]->res = 0;
      continue;
    }
    if (nsync > 0 && nsync == live) {
      for (int i = 0; i < nthr; ++i)
        if (pool->thr[i]->state == ST_SYNC) pool->thr[i]->state = ST_RUN;
      progress = true;
    }
    if (!progress && live > 0) {
      char msg[200];
      snprintf(msg, sizeof msg, "emulator: kernel %s block %d: deadlock (%d threads alive, %d at a barrier)", kname, bid,
               live, nsync);
      set_error(msg);
      broken = true;
      for (int i = 0; i < nthr; ++i)
        if (pool->thr[i]->state != ST_DONE) pool->thr[i]->state = ST_RUN, pool->thr[i]->res = 0;
    }
  }
}

inline unsigned emu_load32(const void* p) {
  unsigned v;
  memcpy(&v, p, 4);
  return v;
}
inline u64 emu_load64(const void* p) {
  u64 v;
  memcpy(&v, p, 8);
  return v;
}
inline float emu_fma(float a, float b, float c) {
  return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}
struct EmuF4 {
  float x, y, z, w;
};
inline EmuF4 emu_loadf4(const float* p) {
  EmuF4 v;
  memcpy(&v, p, 16);
  return v;
}
inline float emu_bits2f(unsigned u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}

#define KERNEL(name, A) static void kk_##name(EmuT& _t, const A& a)
#define KERNELB(name, A, nthr, nblk) static void kk_##name(EmuT& _t, const A& a)
#define KINL static inline
#define KCTXP EmuT& _t,
#define KCTXA _t,
#define KTID (_t.tid)
#define KBID (_t.bid)
#define KNT (_t.nthr)
#define KSYNC() _t.sync()
#define KSHFL_DOWN(v, d) _t.shfl_down((v), (d))
#define KSHFL(v, src) _t.shfl((v), (src))
#define KSHARED(T, name, n) T* name = (T*)_t.shared(sizeof(T) * (n))
#define KLOAD64(p) emu_load64(p)
#define KLOAD32(p) emu_load32(p)
#define KF4 EmuF4
#define KLOADF4(p) emu_loadf4(p)
#define KF4ZERO (EmuF4{0.0f, 0.0f, 0.0f, 0.0f})
#define KFMA(a, b, c) emu_fma((a), (b), (c))
#define KBITS2F(u) emu_bits2f(u)
#define KUNROLL
#define KNOUNROLL

#include "ink2_kernels.inc"

template <class A, void (*K)(EmuT&, const A&)>
void thunk_of(EmuT& t, const void* p) {
  K(t, *(const A*)p);
}

void emu_run(Thunk thunk, const void* args, int grid, int block, const char* kname) {
  if (block <= 0 || (block & 31) != 0 || block > 1024) {
    set_error(std::string("emulator: bad block size for kernel ") + kname);
    return;
  }
#pragma omp parallel for schedule(dynamic, 1) if (grid > 1)
  for (int b = 0; b < grid; ++b) {
    run_block(get_pool(), thunk, args, b, block, kname);
  }
}

struct HostBackend : Backend {
  std::atomic<size_t> used{0};
  std::string err_copy;
  const char* name() override { return "host"; }
  void* alloc(size_t bytes) override {
    size_t n = (bytes + 255) & ~(size_t)255;
    if (n == 0) n = 256;
    void* p = _aligned_malloc(n + 256, 256);  // 256 bytes of slack behind every buffer
    if (!p) {
      set_error("host backend: out of memory");
      return nullptr;
    }
    memset(p, 0, n + 256);
    used += n;
    return p;
  }
  void release(void* p) override { _aligned_free(p); }
  void upload(void* dst, const void* src, size_t bytes) override { memcpy(dst, src, bytes); }
  void download(void* dst, const void* src, size_t bytes) override { memcpy(dst, src, bytes); }
  void zero(void* dst, size_t bytes) override { memset(dst, 0, bytes); }
  void sync() override {}
  void run(int kid, const void* args, int grid, int block) override {
    switch (kid) {
#define X(name, A)                                                     \
  case KID_##name:                                                     \
    emu_run(&thunk_of<A, &kk_##name>, args, grid, block, #name); \
    break;
      INK2_KERNEL_LIST(X)
#undef X
      default:
        set_error("host backend: unknown kernel id");
    }
  }
  size_t bytes_allocated() override { return used.load(); }
  bool device_memory(long long*, long long*) override { return false; }
  const char* error() override {
    if (!g_has_err.load()) return nullptr;
    std::lock_guard<std::mutex> lock(g_err_mutex);
    err_copy = g_err;
    return err_copy.c_str();
  }
};

}  // namespace

Backend* ink2_make_host_backend() { return new HostBackend(); }
