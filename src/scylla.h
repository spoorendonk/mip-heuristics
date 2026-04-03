#pragma once

#include <cstddef>

class HighsMipSolver;

namespace scylla {

// Scylla feasibility pump: PDLP approximate LP solve + fix-and-propagate
// rounding with objective perturbation (Mexi et al. 2023, Algorithm 1.1).
void run(HighsMipSolver &mipsolver, size_t max_effort);

} // namespace scylla
