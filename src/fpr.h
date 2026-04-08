#pragma once

#include <cstddef>

class HighsMipSolver;

namespace fpr {

void run(HighsMipSolver &mipsolver, size_t max_effort);

// Parallel mode with epoch-gated synchronization: N FprWorkers run in
// parallel, synchronizing at epoch boundaries.  Each epoch increments
// the attempt index (new random init, shuffled variable order).  After
// K stale epochs a worker randomizes its config from the full
// strategy x mode space.
void run_parallel(HighsMipSolver &mipsolver, size_t max_effort);

}  // namespace fpr
