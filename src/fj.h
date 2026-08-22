#pragma once
#include <cstddef>

class IncumbentSink;
struct ExecutionContext;
struct HeuristicBudget;
struct ProblemView;

namespace fj {

// Stall threshold, in effort units per constraint-matrix nonzero
// (issue #111).  FJ's counter is in step units, so this is not
// comparable with the other three heuristics' constants.
//
// Scope: **per worker**, matching `mip_heuristic_fj_effort`, which sizes
// one worker's allowance rather than a whole dispatch (`per_worker` in
// mode_dispatch's `kChain`).  The runner-level gate is therefore
// `num_workers` times this.
//
// 256 = `nnz << 8`, the value FJ has always used and the model the other
// three were moved onto: it is exactly a quarter of FJ's default
// per-worker budget (`nnz << 10`), so both the worker-level and the
// runner-level gate are unchanged at the shipped default.
//
// PROVISIONAL, pending the per-heuristic budget calibration (#106).
// #106 sweeps each heuristic's effort option and will show where each
// one actually stops producing solutions; these values are placeholders
// chosen to reproduce the pre-#111 gate at the shipped default effort,
// not the result of a measurement.
inline constexpr size_t kStallPerNnzFj = 256;

// Runs N continuous `parallel::for_each` FjWorkers with per-worker
// self-termination, each seeded differently; a worker that finishes is
// rebuilt in place with a fresh seed.  Set `threads=1` for a single
// worker whose behaviour is reproducible under a fixed `random_seed`.
//
// Implements the uniform runner contract; see heuristic_context.h.
size_t run(const ProblemView& problem, const HeuristicBudget& budget, ExecutionContext& exec,
           IncumbentSink& sink);
}  // namespace fj
