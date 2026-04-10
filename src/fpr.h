#pragma once

#include <cstddef>

class HighsMipSolver;

namespace fpr {

void run(HighsMipSolver &mipsolver, size_t max_effort);

// Parallel mode. When `opportunistic=false`, runs with epoch-gated
// synchronization: N FprWorkers run in parallel, synchronizing at
// epoch boundaries.  Each epoch increments the attempt index (new
// random init, shuffled variable order).  After K stale epochs a
// worker randomizes its config from the full strategy x mode space.
// When `opportunistic=true`, runs continuous `parallel::for_each`
// workers with per-worker self-termination — added in #61; currently
// accepted but ignored.
void run_parallel(HighsMipSolver &mipsolver, size_t max_effort, bool opportunistic = false);

}  // namespace fpr
