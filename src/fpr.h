#pragma once

#include <cstddef>

class HighsMipSolver;
class SolutionPool;

namespace fpr {

// Runs N continuous `parallel::for_each` FprWorkers with per-worker
// self-termination.  Each attempt advances the worker's attempt index
// (new random init, shuffled variable order) and rotates its config
// through the paper-curated strategy x mode list.  Set `threads=1` for
// a single worker whose behaviour is reproducible under a fixed
// `random_seed`.
//
// `pool` is owned by the caller (mode_dispatch::run_sequential).  Workers
// insert solutions with kSolutionSourceFPR; the caller flushes the pool
// to HiGHS once the whole sequential chain has run.
//
// Returns the total effort consumed.  The caller is responsible for
// booking it into `mipdata->heuristic_effort_used` — same contract as
// `local_mip::run_parallel` (issue #79).  This makes mode_dispatch.cpp
// the single point of FPR effort accounting.
size_t run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort);

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
