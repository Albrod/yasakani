// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
// Copy-from-input lookup (no CUDA): every 4-token window of the sequence is chained by its hash (newest first), so all earlier
// places with the same 4 tokens can be visited.  find() picks the place to copy from:
//   1. the nearest place at or after `after` (where the last copy stopped), so that sequential copying resumes in order, unless
//      its backward match is shorter than both 8 and the longest one;
//   2. otherwise the place with the longest backward match (up to max_back tokens), the newest on ties.
#include <algorithm>
#include <cstdint>
#include <vector>
namespace yskn {
struct CopyIndex {
  static constexpr int N = 4;
  std::vector<int32_t> head, prev;
  uint32_t mask = 0;
  int upto = 0;                                  // windows ending at positions < upto are indexed
  static uint32_t h4(const int32_t* s) {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (int k = 0; k < N; ++k) { h ^= (uint64_t)(uint32_t)s[k]; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 29; }
    return (uint32_t)(h >> 32);
  }
  // Size the tables for sequences up to n ids and forget everything.
  void reset(int n) {
    size_t want = 1024; while (want < (size_t)n * 2) want <<= 1;
    if (head.size() != want) { head.assign(want, -1); mask = (uint32_t)(want - 1); }
    else std::fill(head.begin(), head.end(), -1);
    if ((int)prev.size() < n) prev.resize((size_t)n);
    upto = 0;
  }
  void index_upto(const int32_t* S, int end) {
    if (end > (int)prev.size()) {                                   // longer than planned: grow and index again from the start
      reset(std::max(end, (int)prev.size() * 2));
    }
    for (; upto < end; ++upto) {
      if (upto < N - 1) { prev[(size_t)upto] = -1; continue; }
      const uint32_t h = h4(S + upto - N + 1) & mask;
      prev[(size_t)upto] = head[h]; head[h] = upto;
    }
  }
  // Earlier places p < F whose 4 tokens S[p-3..p] equal S[F-3..F].  Returns p (or -1) and the backward match length in *out_len.
  int find(const int32_t* S, int F, int after, int max_cand, int max_back, int* out_len) const {
    *out_len = 0;
    if (F < N - 1 || head.empty()) return -1;
    const uint32_t h = h4(S + F - N + 1) & mask;
    int best = -1, best_len = 0, cand = -1, cand_len = 0, c = 0, steps = 0;
    for (int p = head[h]; p >= 0 && c < max_cand && steps < 4 * max_cand; p = prev[(size_t)p], ++steps) {
      if (p >= F) continue;
      int l = 0; while (l < max_back && p - l >= 0 && S[p - l] == S[F - l]) ++l;
      if (l < N) continue;                                           // another 4-token window in the same bucket
      ++c;
      if (l > best_len) { best_len = l; best = p; }                  // newest first: strict > keeps the newest on ties
      if (after >= 0 && p >= after) { cand = p; cand_len = l; }      // walking down: the last one seen is the nearest
    }
    if (cand >= 0 && cand_len >= std::min(best_len, 8)) { *out_len = cand_len; return cand; }   // unless clearly worse
    *out_len = best_len; return best;
  }
};
}
