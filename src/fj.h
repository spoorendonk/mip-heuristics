#pragma once
#include <cstddef>
class HighsMipSolver;
namespace fj {
// Returns true if FJ detected proven infeasibility (empty integer domain).
bool run(HighsMipSolver &mipsolver, size_t max_effort);

// Parallel mode. When `opportunistic=false`, runs with epoch-gated
// synchronization: N FjWorkers run in parallel, each with a different
// seed, and finished workers are restarted with a new seed.  When
// `opportunistic=true`, runs continuous `parallel::for_each` workers
// with per-worker self-termination — added in #61; currently accepted
// but ignored.  Returns true if proven infeasible.
bool run_parallel(HighsMipSolver &mipsolver, size_t max_effort, bool opportunistic = false);
}  // namespace fj
