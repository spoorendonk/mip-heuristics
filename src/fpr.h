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
//
// Returns the total effort consumed.  The caller is responsible for
// booking it into `mipdata->heuristic_effort_used` — same contract as
// `local_mip::run_parallel` (issue #79).  This makes mode_dispatch.cpp
// the single point of FPR effort accounting.
size_t run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort,
                    bool opportunistic = false);

#ifndef NDEBUG
// Test-only lifecycle counters for the issue #77 pause/resume path.
// Defined in fpr.cpp.  Tests assert these are non-zero after a
// solve at small `mip_heuristic_effort` to verify the kBudgetGate /
// multi-attempt-fill paths actually fired (objective equality alone
// is a proxy that misses lifecycle-path regressions where the
// rotation diverges but converges back to the same final objective).
size_t budget_gate_hits();
size_t multi_attempt_iters();
void reset_test_counters();
#endif

}  // namespace fpr
