#pragma once

#include <cstddef>

class HighsMipSolver;

namespace scylla {

// Scylla feasibility pump: PDLP approximate LP solve + fix-and-propagate
// rounding with objective perturbation (Mexi et al. 2023, Algorithm 1.1).
// Uses a single PDLP instance with M-way parallel FPR rounding, where M
// is capped by thread count, memory, and a hard limit of 4.
void run_parallel(HighsMipSolver &mipsolver, size_t max_effort);

}  // namespace scylla
