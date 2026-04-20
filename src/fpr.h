#pragma once

#include <cstddef>

class HighsMipSolver;
class SolutionPool;

namespace fpr {

// Parallel mode. When `opportunistic=false`, runs with epoch-gated
// synchronization: N FprWorkers run in parallel, synchronizing at
// epoch boundaries.  Each epoch increments the attempt index (new
// random init, shuffled variable order).  After K stale epochs a
// worker randomizes its config from the full strategy x mode space.
// When `opportunistic=true`, runs continuous `parallel::for_each`
// workers with per-worker self-termination.
//
// `pool` is owned by the caller (mode_dispatch::run_sequential).  Workers
// insert solutions with kSolutionSourceFPR; the caller flushes the pool
// to HiGHS once the whole sequential chain has run.
void run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort,
                  bool opportunistic = false);

}  // namespace fpr
