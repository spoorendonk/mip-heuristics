#pragma once

#include <cstddef>

class HighsMipSolver;

namespace scylla {

// Scylla feasibility pump: PDLP approximate LP solve + fix-and-propagate
// rounding with objective perturbation (Mexi et al. 2023, Algorithm 1.1).
void run(HighsMipSolver &mipsolver, size_t max_effort);

// Single-PDLP parallel FPR rounding mode: one PDLP instance produces
// LP solutions, then M FPR configs round in parallel.  Avoids GPU
// contention and preserves warm-start chain.  Deterministic.
void run_parallel(HighsMipSolver &mipsolver, size_t max_effort);

}  // namespace scylla
