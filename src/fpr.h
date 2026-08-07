#pragma once

#include <cstddef>

class HighsMipSolver;
class IncumbentSink;

namespace fpr {

// Runs N continuous `parallel::for_each` FprWorkers with per-worker
// self-termination.  Each attempt advances the worker's attempt index
// (new random init, shuffled variable order) and rotates its config
// through the paper-curated strategy x mode list.  Set `threads=1` for
// a single worker whose behaviour is reproducible under a fixed
// `random_seed`.
//
// `sink` is owned by the caller (mode_dispatch::run_sequential), which
// also sets the source tag workers' solutions are attributed with.
//
// Returns the total effort consumed.  The caller is responsible for
// booking it into `mipdata->heuristic_effort_used` — same contract as
// `local_mip::run_parallel` (issue #79).  This makes mode_dispatch.cpp
// the single point of FPR effort accounting.
size_t run_parallel(HighsMipSolver &mipsolver, IncumbentSink &sink, size_t max_effort);

#ifndef NDEBUG
// Test-only lifecycle counters for the issue #77 pause/resume path.
// Defined in fpr.cpp.  Tests assert these are non-zero after a
// solve at small `mip_heuristic_presolve_effort` to verify the kBudgetGate /
// multi-attempt-fill paths actually fired (objective equality alone
// is a proxy that misses lifecycle-path regressions where the
// rotation diverges but converges back to the same final objective).
size_t budget_gate_hits();
size_t multi_attempt_iters();
void reset_test_counters();
#endif

}  // namespace fpr
