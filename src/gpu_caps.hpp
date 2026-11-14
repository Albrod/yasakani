// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
namespace yskn {
// Makes `device` the current CUDA device and checks that this build can run on it. Throws std::runtime_error with a
// readable message for GPUs older than compute capability 8.6 and for GPUs this build has no code for.
void select_gpu(int device);
// True when every GPU opened in this process runs code built for compute capability 9.0 or newer. Only then may the
// engine use programmatic dependent launch and the bulk L2 prefetch (cp.async.bulk.prefetch).
bool gpu_sm90_features();
}
