#pragma once

#include <cstddef>

class HighsMipSolver;

namespace portfolio {

// Run presolve-based portfolio (pre-root): FPR, LocalMIP, FJ, Scylla arms.
void run_presolve(HighsMipSolver &mipsolver, size_t max_effort);

}  // namespace portfolio
