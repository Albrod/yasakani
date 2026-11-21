// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// CUDA backend: the kernels of ink2_kernels.inc as CUDA kernels.
// Build with -fmad=false: plain a * b + c stays unfused, KFMA is the fused form.
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <string>

#include "ink2_kargs.h"

#define KERNEL(name, A) __global__ void kk_##name(A a)
#define KINL __device__ __forceinline__
#define KCTXP
#define KCTXA
#define KTID ((int)threadIdx.x)
#define KBID ((int)blockIdx.x)
#define KNT ((int)blockDim.x)
#define KSYNC() __syncthreads()
#define KSHFL_DOWN(v, d) __shfl_down_sync(0xffffffffu, (v), (d))
#define KSHFL(v, src) __shfl_sync(0xffffffffu, (v), (src))
#define KSHARED(T, name, n) __shared__ T name[n]
#define KLOAD64(p) (*(const unsigned long long*)(p))
#define KLOAD32(p) (*(const unsigned*)(p))
#define KF4 float4
#define KLOADF4(p) (*(const float4*)(p))
#define KF4ZERO make_float4(0.0f, 0.0f, 0.0f, 0.0f)
#define KFMA(a, b, c) fmaf((a), (b), (c))
#define KBITS2F(u) __uint_as_float(u)
#define KUNROLL _Pragma("unroll")
#define KNOUNROLL _Pragma("unroll 1")

#include "ink2_kernels.inc"

namespace {

struct CudaBackend : Backend {
  size_t used = 0;
  bool failed = false;
  std::string err;
  cudaStream_t stream = nullptr;  // every kernel runs here; copies use the default stream, which waits for it

  void fail(const char* what, cudaError_t e) {
    if (failed) return;
    failed = true;
    char msg[300];
    snprintf(msg, sizeof msg, "cuda backend: %s: %s", what, cudaGetErrorString(e));
    err = msg;
  }
  const char* name() override { return "cuda"; }
  void* alloc(size_t bytes) override {
    size_t n = (bytes + 255) & ~(size_t)255;
    if (n == 0) n = 256;
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, n);
    if (e != cudaSuccess) {
      fail("cudaMalloc", e);
      return nullptr;
    }
    e = cudaMemset(p, 0, n);
    if (e != cudaSuccess) fail("cudaMemset", e);
    used += n;
    return p;
  }
  void release(void* p) override { cudaFree(p); }
  void upload(void* dst, const void* src, size_t bytes) override {
    cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) fail("cudaMemcpy to device", e);
  }
  void download(void* dst, const void* src, size_t bytes) override {
    cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) fail("cudaMemcpy to host", e);
  }
  void zero(void* dst, size_t bytes) override {
    cudaError_t e = cudaMemset(dst, 0, bytes);
    if (e != cudaSuccess) fail("cudaMemset", e);
  }
  void sync() override {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) fail("cudaDeviceSynchronize", e);
  }
  void run(int kid, const void* args, int grid, int block) override {
    if (failed) return;
    switch (kid) {
#define X(name, A)                                        \
  case KID_##name:                                        \
    kk_##name<<<grid, block, 0, stream>>>(*(const A*)args); \
    break;
      INK2_KERNEL_LIST(X)
#undef X
      default:
        break;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fail("kernel launch", e);
  }
  int max_block(int kid) override {
    static int cache[KID_COUNT] = {};
    if (kid < 0 || kid >= KID_COUNT) return 0;
    if (cache[kid] == 0) {
      cudaFuncAttributes fa{};
      cudaError_t e = cudaErrorInvalidValue;
      switch (kid) {
#define X(name, A)                                   \
  case KID_##name:                                   \
    e = cudaFuncGetAttributes(&fa, kk_##name);       \
    break;
        INK2_KERNEL_LIST(X)
#undef X
        default:
          break;
      }
      cache[kid] = e == cudaSuccess ? fa.maxThreadsPerBlock : 256;
    }
    return cache[kid];
  }
  size_t bytes_allocated() override { return used; }
  bool device_memory(long long* free_bytes, long long* total_bytes) override {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return false;
    *free_bytes = (long long)f;
    *total_bytes = (long long)t;
    return true;
  }
  const char* error() override { return failed ? err.c_str() : nullptr; }

  bool graph_supported() override { return true; }
  void graph_begin() override {
    cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) fail("cudaStreamBeginCapture", e);
  }
  void* graph_end() override {
    cudaGraph_t graph = nullptr;
    cudaError_t e = cudaStreamEndCapture(stream, &graph);
    if (e != cudaSuccess) {
      fail("cudaStreamEndCapture", e);
      return nullptr;
    }
    cudaGraphExec_t exec = nullptr;
    e = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess) {
      fail("cudaGraphInstantiate", e);
      return nullptr;
    }
    return exec;
  }
  void graph_launch(void* g) override {
    cudaError_t e = cudaGraphLaunch((cudaGraphExec_t)g, stream);
    if (e != cudaSuccess) fail("cudaGraphLaunch", e);
  }
  void graph_free(void* g) override {
    if (g) cudaGraphExecDestroy((cudaGraphExec_t)g);
  }
};

}  // namespace

Backend* ink2_make_cuda_backend(char* errbuf, int err_cap) {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess || count < 1) {
    snprintf(errbuf, err_cap, "cuda backend: no device (%s)", cudaGetErrorString(e));
    return nullptr;
  }
  e = cudaSetDevice(0);
  if (e == cudaSuccess) e = cudaFree(0);  // creates the context
  if (e != cudaSuccess) {
    snprintf(errbuf, err_cap, "cuda backend: cannot start the device (%s)", cudaGetErrorString(e));
    return nullptr;
  }
  CudaBackend* b = new CudaBackend();
  e = cudaStreamCreate(&b->stream);
  if (e != cudaSuccess) {
    snprintf(errbuf, err_cap, "cuda backend: cudaStreamCreate (%s)", cudaGetErrorString(e));
    delete b;
    return nullptr;
  }
  return b;
}
