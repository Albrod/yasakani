// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <cstdlib>
namespace yskn {
// YSKN_Q35_PF_V2=0 reads prompts and runs passes of 64 rows as 0.1.7 did: the old attention layout and combine for 64-row passes,
// no shared-row KV compression, the old GEMM schedule, the output layer on every prompt row in 16-row launches, no extra prompt
// snapshots, and at most 1,024 rows per pass and 4,096 rows of layer work area.  Default 1 (the faster prompt reading).
inline bool q35_prefill_v2() {
  static const bool on = [] { const char* e = std::getenv("YSKN_Q35_PF_V2"); return !e || !e[0] || std::atoi(e) != 0; }();
  return on;
}
}
