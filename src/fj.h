#pragma once
#include <cstddef>

class HighsMipSolver;
class SolutionPool;

namespace fj {

// Return shape for `run_parallel`: total effort consumed plus the proven-
// infeasible flag.  FJ is the only one of the four sequential heuristics
// that can in principle short-circuit the chain by proving infeasibility,
// so it carries a flag alongside the effort count rather than just
// returning `size_t` (issue #79 follow-up).
struct ParallelResult {
    size_t effort = 0;
    bool infeasible = false;
};

// Parallel mode. When `opportunistic=false`, runs with epoch-gated
// synchronization: N FjWorkers run in parallel, each with a different
// seed, and finished workers are restarted with a new seed.  When
// `opportunistic=true`, runs continuous `parallel::for_each` workers
// with per-worker self-termination.
//
// `pool` is owned by the caller (mode_dispatch::run_sequential).  Workers
// insert solutions into it with kSolutionSourceFJ, and the caller is
// responsible for flushing the pool to HiGHS once the sequential chain
// of heuristics is complete.
//
// Returns the total effort consumed and an infeasible flag.  The caller
// is responsible for booking the effort into
// `mipdata->heuristic_effort_used` — same contract as
// `local_mip::run_parallel` (issue #79).  This makes mode_dispatch.cpp
// the single point of FJ effort accounting.
ParallelResult run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort,
                            bool opportunistic = false);
}  // namespace fj
