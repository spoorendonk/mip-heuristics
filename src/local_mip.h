#pragma once

#include <cstddef>
#include <cstdint>

struct CscMatrix;
struct HeuristicResult;
class HighsMipSolver;
class SolutionPool;

namespace local_mip {

// Test-only introspection: counts how many times each branch of
// `resolve_worker_start` fired during the current process.  The three
// branches correspond to the three #75 cold-start fallback rungs:
//   - `pool`   : pool.copy_best succeeded (warm-start from #74 pool).
//   - `incumbent` : pool was empty, mipdata->incumbent picked up.
//   - `construction` : both empty, paper's construction phase ran.
// Call `reset_warm_start_counters()` before a HiGHS run, then read
// `warm_start_counters()` after to assert which path actually fired.
// Used by R1-8 / R2-7 / R3-3 round-3 review tests to distinguish #74
// (pool warm-start) from #75 (cold-start construction); without this
// the integration tests can't tell those paths apart since both
// produce non-zero effort.
struct WarmStartCounters {
    int pool;
    int incumbent;
    int construction;
};

void reset_warm_start_counters();
WarmStartCounters warm_start_counters();
// Parallel mode. When `opportunistic=false`, runs with epoch-gated
// synchronization: N workers run local search in parallel, synchronizing
// at epoch boundaries.  Worker 0 starts from the unperturbed incumbent;
// workers 1..N-1 start from perturbed incumbents.  Stalled workers are
// restarted from the pool's best solution with fresh perturbation.
// When `opportunistic=true`, runs continuous `parallel::for_each`
// workers with per-worker self-termination.
//
// `pool` is owned by the caller (mode_dispatch::run_sequential).  Workers
// insert solutions with kSolutionSourceLocalMIP and may pull restarts
// from the pool; the caller flushes the pool once all sequential
// heuristics have run.
void run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort,
                  bool opportunistic = false);

// Single-worker variant for portfolio mode. Returns result without submitting.
// If initial_solution is non-null, uses it as starting point.
// max_effort: effort budget (coefficient accesses).
HeuristicResult worker(HighsMipSolver &mipsolver, const CscMatrix &csc, uint32_t seed,
                       const double *initial_solution, size_t max_effort);
}  // namespace local_mip
