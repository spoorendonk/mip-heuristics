#pragma once

#include <cstddef>

class HighsMipSolver;

namespace scylla {

// Scylla feasibility pump: PDLP approximate LP solve + fix-and-propagate
// rounding with objective perturbation (Mexi et al. 2023, Algorithm 1.1).
void run(HighsMipSolver &mipsolver, size_t max_effort);

// Parallel mode with epoch-gated synchronization: N workers run pump
// chains in parallel, synchronizing at epoch boundaries.  At each sync
// point the main thread checks termination, staleness, and shares
// improvements across workers.  Deterministic: behavior is identical
// across runs with the same input.
void run_parallel(HighsMipSolver &mipsolver, size_t max_effort);

} // namespace scylla
