#pragma once

#include <cstddef>

class HighsMipSolver;

namespace heuristics {

// Top-level presolve heuristic dispatch. Reads mip_heuristic_* options
// and routes to sequential, portfolio deterministic, or portfolio
// opportunistic mode.  Returns true if the model was proven infeasible.
bool run_presolve(HighsMipSolver &mipsolver, size_t budget);

}  // namespace heuristics
