// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include "kernels.hpp"
#include <map>
namespace yskn {
struct Q35G6Workspace {
  std::map<const void*, const uint4*> cached;
  uint4* weights = nullptr; size_t weight_bytes = 0;
  long long* partial = nullptr; int* flags = nullptr;
  size_t partial_bytes = 0, flags_bytes = 0; int resident[4] = {};
  int tile = 0, tile_rows = 0, tile_cols = 0;
  int tile_resident = 0, tile_registers = 0;
  size_t tile_shared = 0;
  bool wide_ready = false;
};
void q35_g6_decode(const TensorDev& t, uint4* dst, cudaStream_t st);
void q35_g6_prepare(Q35G6Workspace& W, int sms);
bool launch_q35_g6(const GemvLaunch& P, Q35G6Workspace& W, int pm, cudaStream_t st);
}
