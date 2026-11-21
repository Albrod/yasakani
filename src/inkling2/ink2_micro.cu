// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Micro measurements of the device (not part of the engine): what a kernel
// launch costs, and what a barrier across all blocks of one kernel costs.
// They decide how the kernels of a token are packaged.
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <stdarg.h>
#include <stdio.h>

#include <chrono>
#include <string>

namespace cg = cooperative_groups;

namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void addf(std::string& s, const char* f, ...) {
  char buf[400];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  s += buf;
}

__global__ void mb_empty(int* sink) {
  if (threadIdx.x == 0 && blockIdx.x == 0) sink[0] += 1;
}

// n official grid barriers
__global__ void mb_gridsync(int n, int* sink) {
  cg::grid_group g = cg::this_grid();
  for (int i = 0; i < n; ++i) {
    if (threadIdx.x == 0 && blockIdx.x == 0) sink[0] += 1;
    g.sync();
  }
}

// n hand-made barriers: every block reports its arrival, block 0 opens the
// next generation when all have arrived. st[0] = arrivals, st[1] = generation.
__global__ void mb_spin(int n, int nblocks, volatile int* st, int* sink) {
  for (int i = 0; i < n; ++i) {
    if (threadIdx.x == 0 && blockIdx.x == 0) sink[0] += 1;
    __syncthreads();
    if (threadIdx.x == 0) {
      if (blockIdx.x == 0) {
        for (long long spin = 0; spin < 2000000000LL && st[0] < (nblocks - 1) * (i + 1); ++spin) {
        }
        __threadfence();
        st[1] = i + 1;
      } else {
        atomicAdd((int*)st, 1);
        for (long long spin = 0; spin < 2000000000LL && st[1] < i + 1; ++spin) {
        }
      }
    }
    __syncthreads();
  }
}

// n hand-offs "one block works alone, then all blocks go on": the other
// blocks only wait for the flag of block 0 (no arrival count).
__global__ void mb_flag(int n, volatile int* st, int* sink) {
  for (int i = 0; i < n; ++i) {
    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        sink[0] += 1;
        __threadfence();
        st[1] = i + 1;
      }
    } else if (threadIdx.x == 0) {
      for (long long spin = 0; spin < 2000000000LL && st[1] < i + 1; ++spin) {
      }
    }
    __syncthreads();
  }
}

}  // namespace

int ink2_cuda_microbench(char* report, int cap) {
  std::string r;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    snprintf(report, cap, "microbench: no device\n");
    return -1;
  }
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int* sink = nullptr;
  int* st = nullptr;
  cudaMalloc(&sink, 256);
  cudaMalloc(&st, 256);
  cudaMemset(sink, 0, 256);
  addf(r, "device: %s, %d multiprocessors, compute capability %d.%d\n", prop.name, prop.multiProcessorCount, prop.major,
       prop.minor);

  // 1. plain launches, back to back
  const int shapes[3][2] = {{1, 32}, {24, 1024}, {1152, 256}};
  for (auto& sh : shapes) {
    const int n = 2000;
    for (int warm = 0; warm < 50; ++warm) mb_empty<<<sh[0], sh[1], 0, stream>>>(sink);
    cudaStreamSynchronize(stream);
    double t0 = now_s();
    for (int i = 0; i < n; ++i) mb_empty<<<sh[0], sh[1], 0, stream>>>(sink);
    double t1 = now_s();
    cudaStreamSynchronize(stream);
    double t2 = now_s();
    addf(r, "empty kernel, %4d blocks x %4d threads, %d launches: %.2f us each until all are done (%.2f us to queue one)\n",
         sh[0], sh[1], n, 1e6 * (t2 - t0) / n, 1e6 * (t1 - t0) / n);
  }
  // 2. the same as one recorded graph
  {
    const int n = 1000;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    for (int i = 0; i < n && e == cudaSuccess; ++i) mb_empty<<<1, 32, 0, stream>>>(sink);
    if (e == cudaSuccess) e = cudaStreamEndCapture(stream, &graph);
    if (e == cudaSuccess) e = cudaGraphInstantiate(&exec, graph, 0);
    if (e == cudaSuccess) {
      cudaGraphLaunch(exec, stream);
      cudaStreamSynchronize(stream);
      double t0 = now_s();
      for (int rep = 0; rep < 5; ++rep) cudaGraphLaunch(exec, stream);
      cudaStreamSynchronize(stream);
      addf(r, "recorded graph of %d empty kernels (1 block x 32 threads): %.2f us per kernel\n", n, 1e6 * (now_s() - t0) / (5.0 * n));
      cudaGraphExecDestroy(exec);
      cudaGraphDestroy(graph);
    } else {
      addf(r, "recorded graph: failed (%s)\n", cudaGetErrorString(e));
    }
  }
  // 3. barriers inside one kernel
  const int block_sizes[2] = {256, 1024};
  for (int bs : block_sizes) {
    int per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mb_gridsync, bs, 0);
    int full = per_sm * prop.multiProcessorCount;
    addf(r, "block size %d: at most %d blocks per multiprocessor, %d blocks at once\n", bs, per_sm, full);
    const int grids[3] = {prop.multiProcessorCount, full / 2, full};
    for (int grid : grids) {
      if (grid < 2) continue;
      const int n = 2000;
      for (int variant = 0; variant < 3; ++variant) {
        cudaMemset(st, 0, 256);
        cudaStreamSynchronize(stream);
        cudaError_t e = cudaSuccess;
        double t0 = now_s();
        if (variant == 0) {
          int nn = n;
          void* args[2] = {&nn, &sink};
          e = cudaLaunchCooperativeKernel((void*)mb_gridsync, dim3(grid), dim3(bs), args, 0, stream);
        } else if (variant == 1) {
          mb_spin<<<grid, bs, 0, stream>>>(n, grid, st, sink);
        } else {
          mb_flag<<<grid, bs, 0, stream>>>(n, st, sink);
        }
        cudaError_t e2 = cudaStreamSynchronize(stream);
        double t1 = now_s();
        const char* name = variant == 0 ? "official grid barrier" : variant == 1 ? "hand-made barrier (arrival count)" : "hand-off from block 0 (flag only)";
        if (e != cudaSuccess || e2 != cudaSuccess)
          addf(r, "  %5d blocks, %s: failed (%s)\n", grid, name, cudaGetErrorString(e != cudaSuccess ? e : e2));
        else
          addf(r, "  %5d blocks, %-34s %.2f us each (%d in one kernel)\n", grid, name, 1e6 * (t1 - t0) / n, n);
      }
    }
  }
  cudaFree(sink);
  cudaFree(st);
  cudaStreamDestroy(stream);
  snprintf(report, cap, "%s", r.c_str());
  return 0;
}
