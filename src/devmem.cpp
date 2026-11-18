// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
namespace yskn {
namespace {
struct Rec { void* owner; size_t bytes; bool host; };
std::mutex g_mu;
std::unordered_map<void*, Rec>& reg() { static std::unordered_map<void*, Rec> r; return r; }
thread_local void* t_owner = nullptr;


int fill_byte() { static const int f = [] { const char* e = std::getenv("YSKN_DEVMEM_FILL"); return e && e[0] ? std::atoi(e) : 0; }(); return f; }
}
void* devmem_owner() { return t_owner; }
void devmem_set_owner(void* owner) { t_owner = owner; }

cudaError_t copy_weight_to_device(void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return cudaSuccess;
#if defined(__linux__) && defined(__aarch64__)
  int device = 0;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return e;
  cudaDeviceProp props{};
  e = cudaGetDeviceProperties(&props, device);
  if (e != cudaSuccess) return e;
  if (props.integrated && props.major == 12 && props.minor == 1) {
    // Direct CUDA copies from file-backed mappings can stall on GB10.
    constexpr size_t chunk = 4 * 1024 * 1024;
    thread_local std::vector<unsigned char> staging(chunk);
    for (size_t off = 0; off < bytes;) {
      const size_t n = std::min(bytes - off, chunk);
      std::memcpy(staging.data(), static_cast<const unsigned char*>(src) + off, n);
      e = cudaMemcpy(static_cast<unsigned char*>(dst) + off, staging.data(), n, cudaMemcpyHostToDevice);
      if (e != cudaSuccess) return e;
      off += n;
    }
    return cudaSuccess;
  }
#endif
  return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

cudaError_t tracked_malloc(void** p, size_t n) {
  const cudaError_t e = ::cudaMalloc(p, n);
  if (e == cudaSuccess && *p && t_owner) {
    { std::lock_guard<std::mutex> l(g_mu); reg()[*p] = Rec{t_owner, n, false}; }
    ::cudaMemset(*p, fill_byte(), n); ::cudaDeviceSynchronize();      
  }
  return e;
}
cudaError_t tracked_malloc_host(void** p, size_t n) {
  const cudaError_t e = ::cudaMallocHost(p, n);
  if (e == cudaSuccess && *p && t_owner) { { std::lock_guard<std::mutex> l(g_mu); reg()[*p] = Rec{t_owner, n, true}; } std::memset(*p, fill_byte(), n); }
  return e;
}
cudaError_t tracked_free(void* p) {
  if (p) { std::lock_guard<std::mutex> l(g_mu); reg().erase(p); }
  return ::cudaFree(p);
}
cudaError_t tracked_free_host(void* p) {
  if (p) { std::lock_guard<std::mutex> l(g_mu); reg().erase(p); }
  return ::cudaFreeHost(p);
}
size_t devmem_release(void* owner) {
  std::vector<std::pair<void*, Rec>> mine;
  { std::lock_guard<std::mutex> l(g_mu);
    for (auto it = reg().begin(); it != reg().end();) { if (it->second.owner == owner) { mine.emplace_back(*it); it = reg().erase(it); } else ++it; } }
  size_t bytes = 0;
  for (auto& [p, r] : mine) { if (r.host) ::cudaFreeHost(p); else { ::cudaFree(p); bytes += r.bytes; } }
  return bytes;
}
}
