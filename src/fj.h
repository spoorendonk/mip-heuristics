#pragma once
#include <cstddef>

class HighsMipSolver;
class SolutionPool;

namespace fj {
// Parallel mode. When `opportunistic=false`, runs with epoch-gated
// synchronization: N FjWorkers run in parallel, each with a different
// seed, and finished workers are restarted with a new seed.  When
// `opportunistic=true`, runs continuous `parallel::for_each` workers
// with per-worker self-termination.  Returns true if proven infeasible.
//
// `pool` is owned by the caller (mode_dispatch::run_sequential).  Workers
// insert solutions into it with kSolutionSourceFJ, and the caller is
// responsible for flushing the pool to HiGHS once the sequential chain
// of heuristics is complete.
bool run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort,
                  bool opportunistic = false);
}  // namespace fj
