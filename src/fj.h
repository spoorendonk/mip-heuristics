#pragma once
#include <cstddef>

class HighsMipSolver;
class SolutionPool;

namespace fj {

// Runs N continuous `parallel::for_each` FjWorkers with per-worker
// self-termination, each seeded differently; a worker that finishes is
// rebuilt in place with a fresh seed.  Set `threads=1` for a single
// worker whose behaviour is reproducible under a fixed `random_seed`.
//
// `pool` is owned by the caller (mode_dispatch::run_sequential).  Workers
// insert solutions into it with kSolutionSourceFJ, and the caller is
// responsible for flushing the pool to HiGHS once the sequential chain
// of heuristics is complete.
//
// Returns the total effort consumed.  The caller is responsible for
// booking the effort into `mipdata->heuristic_effort_used` — same
// contract as `local_mip::run_parallel`, `fpr::run_parallel`, and
// `scylla::run_parallel` (issue #79).  This makes mode_dispatch.cpp the
// single point of effort accounting for the four presolve heuristics.
size_t run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort);
}  // namespace fj
