// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ink2_kargs.h"

// A row of the gate/up matrix alternates gate and up channels. The down
// projection writes back to the original router slot, preserving combine order.
template<bool GateUp>
__device__ __forceinline__ void ink2_store_pair(const ExpMArgs& a, int pair, int col, float v0, float v1) {
  if (GateUp) {
    float gate = tanhf(v0 / a.gate_cap) * a.gate_cap;
    float up = tanhf(v1 / a.up_cap) * a.up_cap;
    a.y[static_cast<size_t>(pair) * (a.rows / 2) + col / 2] = gate * (1.0f / (1.0f + expf(-v0))) * up;
  } else {
    float* dst = a.y + static_cast<size_t>(pair) * a.rows + col;
    dst[0] = v0;
    dst[1] = v1;
  }
}
