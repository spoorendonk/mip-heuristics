#pragma once

#include <cstddef>
#include <cstdint>

struct CscMatrix;
struct HeuristicResult;
class HighsMipSolver;

namespace local_mip {
void run(HighsMipSolver &mipsolver, size_t max_effort);

// Parallel mode with epoch-gated synchronization: N workers run local
// search in parallel, synchronizing at epoch boundaries.  Worker 0 starts
// from the unperturbed incumbent; workers 1..N-1 start from perturbed
// incumbents.  Stalled workers are restarted from the pool's best
// solution with fresh perturbation.
void run_parallel(HighsMipSolver &mipsolver, size_t max_effort);

// Single-worker variant for portfolio mode. Returns result without submitting.
// If initial_solution is non-null, uses it as starting point.
// max_effort: effort budget (coefficient accesses).
HeuristicResult worker(HighsMipSolver &mipsolver, const CscMatrix &csc, uint32_t seed,
                       const double *initial_solution, size_t max_effort);
}  // namespace local_mip
