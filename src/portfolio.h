#pragma once

#include <cstddef>

class HighsMipSolver;

namespace portfolio {

// Run presolve-based portfolio (pre-root): FPR, LocalMIP, FJ, Scylla arms.
// `opportunistic=false` uses the deterministic epoch-gated path with
// `PortfolioWorker` + `run_epoch_loop`; `opportunistic=true` uses the
// continuous `parallel::for_each` + `while(!stop)` path.
void run_presolve(HighsMipSolver &mipsolver, size_t max_effort, bool opportunistic);

}  // namespace portfolio
